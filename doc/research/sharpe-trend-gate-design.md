# Sharpe-Trend Gate — Design Note

**Origin**: User direction 2026-05-04. "你应该看的是 monthly sharpe 的趋势，
sharpe 在涨的考虑买，sharepe 在跌的考虑卖."

**Status**: design captured; not yet implemented. Defer until current B2
hold-out finishes and Sharpe<1 decision is made.

---

## Problem the gate solves

Current TournamentJob emits a single scalar `oos_sharpe` over a fixed
8-month window (today − 2y → today). Two failure modes:

1. **Static evaluation misses regime shifts.** A model that had Sharpe
   1.5 in 2024-Q3 and 0.0 in 2024-Q4 still passes a 0.5 floor with
   blended ~0.75 — but it's clearly going stale.
2. **Static evaluation under-weights recent wins.** A model with
   Sharpe 0.0 → 0.4 → 0.8 → 1.2 over four quarters reads 0.6 average
   — gets the same admission decision as a flat-Sharpe model with
   no momentum.

User's insight: rising Sharpe = signal still has edge → trust it for
new entries. Falling Sharpe = signal decaying → reduce/exit.

## Proposed mechanism

### Layer 1 — Tournament training-time

Instead of `run_tournament` returning a scalar `sharpe`, also emit:
```
oos_sharpe_series : pd.Series   # rolling 63-bar Sharpe over the OOS window
```

Persist the series in the per-ticker artifact JSON.

### Layer 2 — Inference-time gate

New `SharpeTrendGateTask` consumes the per-ticker series and computes:
```
recent_sharpe = series.iloc[-63:].mean()                # last ~3mo level
sharpe_slope  = (last30_mean - last90to60_mean) / 60d   # slope over last 60d
```

Decision rules (config-flag-gated, OFF by default):
- BUY admission: pass if `recent_sharpe >= floor` AND `sharpe_slope >= 0`
- SELL acceleration: when `sharpe_slope < -slope_floor`, lower the
  per-ticker `consecutive_sell_signals` requirement by 1 (faster exit
  on degrading-edge holds)

### Layer 3 — Position-sizing modulation (optional)

`conviction_multiplier` already exists in `kernel/sizing.py`. Add
`sharpe_trend_multiplier` that scales position size:
- slope ≥ +0.2/mo: ×1.20 (boost on improving model)
- slope ∈ [-0.2, +0.2]/mo: ×1.00
- slope ≤ -0.2/mo: ×0.50 (size-down on decaying)

## Statistical caveat — slope detectability

| Window | Per-bar Sharpe noise | Detectable slope (3σ) |
|---|---|---|
| 21 bars (1mo) | ±0.22 | ±0.66/mo |
| 63 bars (3mo) | ±0.13 | ±0.39/mo |
| 126 bars (6mo) | ±0.09 | ±0.27/mo |
| 252 bars (1yr) | ±0.063 | ±0.19/mo |

**Use rolling 63-bar Sharpe smoothed**, not 21-bar monthly. The slope
estimate from rolling-63 over a 60-day window is much cleaner.

## Config sketch

```json
{
  "panel_ltr": {
    "tournament": {
      "sharpe_trend": {
        "enabled": false,
        "rolling_window_bars": 63,
        "slope_window_bars": 60,
        "slope_floor": 0.20,
        "slope_above_zero_for_buy": true,
        "slope_below_minus_floor_accelerates_sell": true,
        "size_multiplier_curve": {
          "rising":  1.20,
          "flat":    1.00,
          "falling": 0.50
        }
      }
    }
  }
}
```

## Implementation cost estimate

| Step | Hours |
|---|---|
| `run_tournament` emit `oos_sharpe_series` | 1 |
| Persist series in artifact JSON | 0.5 |
| New `SharpeTrendGateTask` + tests | 1.5 |
| Wire into `BuyGatesJob` + `TickerSellJob` | 1 |
| §5.2 sanity (A/A + shuffled-label + placebo on slope rule) | 1 |
| Retrain + B2 validate | 1.5 |
| **Total** | **~6.5 hours** |

## When to ship

- **Don't ship now** — current B2 still running. §5.11: don't add
  features mid-experiment.
- **Trigger condition for shipping**: B2 produces Sharpe ≥ 1 (mandate)
  AND we want to push the model further. This is improvement, not
  bug fix.
- **Alternate trigger**: B2 produces Sharpe < 1 with high turnover
  + mediocre per-trade alpha — that's exactly the symptom of "model
  edge fading on some held tickers" → sharpe-trend gate is a candidate
  fix.

## §5.2 sanity protocol for this feature

Mandatory before ship:
1. **A/A test**: same data, two seeds — does the slope-gate produce
   the same admission set within reasonable noise?
2. **Shuffled-label test**: shuffle y in tournament training, retrain,
   verify slope is ≈ 0 across all tickers (no spurious trend on noise).
3. **Placebo (time-shift)**: shift labels by 1 year — slope should
   match neither real nor zero; randomly-distributed.
4. **Per-ticker sample size**: how many tickers have ≥ 60 bars of OOS
   Sharpe history? If <50% of universe, the gate is too sparse to
   matter; defer.
