# RenQuant 102 vs 103 — Performance Comparison Report

**Generated:** 2026-04-11  
**Data source:** Per-symbol OOS Sharpe ratios from notebook walk-forward simulation (`policy-metadata.json`)  
**Training window:** 3-year rolling, 70/30 train/test split  
**Backtest window (notebook sim):** 2024-01-01 – 2026-03-26

---

## Important Caveat

> **No LEAN backtest results exist for either strategy.** This report compares per-symbol OOS Sharpe from notebook-level simulation only — not portfolio-level results from the LEAN engine. The claim "103 is better than 102" cannot be substantiated at the portfolio level without running `lean backtest` for both strategies against the same time window. The per-symbol Sharpe numbers are a proxy, not a definitive verdict.

---

## Configuration Comparison

| Parameter | renquant-102 | renquant-103 |
|-----------|-------------|-------------|
| Watchlist size | 21 | 24 |
| `min_hold_days` | 1 | 20 |
| `max_hold_days` | 500 | 500 (BULL), 10 (CHOPPY) |
| `lookahead` (labels) | 10 | 5 |
| `threshold` (label) | 0.04 (4%) | 0.03 (3%) |
| Feature columns | 7 standard | 7 standard + 4 regime context |
| Regime detection | None | 3-layer (Hurst + CUSUM + GMM) |
| Position sizing | Flat 30% max | Regime-conditional (30% / 20% / 15% / 0%) |
| Consecutive-sell filter | No | Yes (3 signals before exit) |
| Defensive tickers | No | Yes (GLD, TLT, XLV, XLU) |
| Trailing stop | No | Yes (BULL_CALM only, 5% trigger/trail) |
| Stop-loss | 8% flat | Regime-dependent (5–8%) |
| Drawdown halt | 15% | Regime-dependent (5–15%) |
| Initial cash | $100,000 | $100,000 |
| Max concurrent positions | 5 | 5 |
| Volume filter | P85 percentile | P85 percentile |
| Tax rates | 50% ST / 32% LT | 50% ST / 32% LT |

**102-only symbols:** ARKK, COIN, SHOP  
**103-only symbols:** AAPL, GLD, META, TLT, XLU, XLV (adds defensive counter-cyclicals)  
**Common symbols (18):** AMD, AMZN, BA, CAT, CRM, GOOG, JPM, LLY, MSFT, NFLX, NVDA, PLTR, TSLA, UBER, UNH, XLE, XLF, XOM

---

## Per-Symbol OOS Sharpe Summary

### Aggregate Statistics

| Metric | renquant-102 (n=21) | renquant-103 (n=24) |
|--------|--------------------|--------------------|
| Mean OOS Sharpe | **0.918** | **1.035** |
| Median OOS Sharpe | 0.922 | 0.914 |
| Std Dev | **0.117** | **0.334** |
| Min | 0.622 (CRM) | 0.596 (XLV) |
| Max | 1.082 (COIN*) | 1.959 (PLTR) |
| Above Sharpe 0.8 | 20/21 **(95%)** | 21/24 **(88%)** |
| Above Sharpe 1.0 | 5/21 **(24%)** | 9/24 **(38%)** |

\* COIN is not in the 103 watchlist.

**Reading the aggregate:** 103 has a higher mean (+0.12) driven by a handful of high performers (PLTR 1.96, NVDA 1.58, XLF 1.47, GOOG 1.51). The median is nearly identical (0.922 vs 0.914), and 103's standard deviation is 3× larger — meaning 103 is more uneven across symbols. 102 is tighter and more consistent.

### Model Type Distribution (Watchlist Symbols)

| Model Type | renquant-102 | renquant-103 |
|------------|-------------|-------------|
| classification | 15 (71%) | 8 (33%) |
| qlearning | 3 (14%) | 10 (42%) |
| manual | 3 (14%) | 6 (25%) |

103 leans heavily on Q-Learning (42% vs 14%). This is partly a consequence of the shorter `lookahead=5` making classification thresholding harder.

---

## Symbol-by-Symbol Comparison (Common 18 Symbols)

| Symbol | 102 Sharpe (type) | 103 Sharpe (type) | Delta |
|--------|-------------------|-------------------|-------|
| AMD    | 0.831 (classif.) | 0.921 (classif.) | +0.090 |
| AMZN   | 0.847 (classif.) | 0.852 (manual)   | +0.005 |
| BA     | 0.913 (classif.) | 0.635 (classif.) | **−0.279** |
| CAT    | 1.060 (manual)   | 1.226 (qlearn.)  | +0.166 |
| CRM    | 0.622 (classif.) | 0.802 (classif.) | +0.181 |
| GOOG   | 0.965 (classif.) | 1.508 (classif.) | **+0.544** |
| JPM    | 0.828 (classif.) | 0.851 (qlearn.)  | +0.023 |
| LLY    | 0.832 (classif.) | 1.095 (qlearn.)  | **+0.263** |
| MSFT   | 0.982 (qlearn.)  | 0.907 (qlearn.)  | −0.075 |
| NFLX   | 0.817 (classif.) | 0.964 (qlearn.)  | +0.147 |
| NVDA   | 1.020 (classif.) | 1.583 (manual)   | **+0.563** |
| PLTR   | 0.997 (manual)   | 1.959 (qlearn.)  | **+0.962** |
| TSLA   | 1.067 (classif.) | 0.883 (manual)   | −0.184 |
| UBER   | 0.821 (qlearn.)  | 0.810 (classif.) | −0.010 |
| UNH    | 0.922 (classif.) | 0.683 (manual)   | **−0.239** |
| XLE    | 1.076 (classif.) | 0.832 (classif.) | **−0.244** |
| XLF    | 0.808 (classif.) | 1.473 (manual)   | **+0.665** |
| XOM    | 0.994 (manual)   | 1.131 (qlearn.)  | +0.137 |

**103 better on 13/18 common symbols; 102 better on 5/18 (BA, MSFT, TSLA, UBER, UNH, XLE).**  
The 5 symbols where 103 regressed notably: BA (−0.28), UNH (−0.24), XLE (−0.24), TSLA (−0.18), MSFT (−0.07).

---

## What the Data Shows

### Where 103 has an edge
- **Mean OOS Sharpe is 12.7% higher** (1.035 vs 0.918), driven by concentrated outperformers (PLTR, XLF, GOOG, NVDA).
- **More Sharpe > 1.0 symbols** (38% vs 24%) — 103 produces more "high-quality" models per symbol.
- **Regime context features** (`spy_realized_vol`, `hurst_proxy`, etc.) appear to help classification learn market-conditional patterns.
- **Defensive tickers** (GLD, TLT, XLU, XLV) have OOS Sharpes of 0.89, 0.97, 0.81, 0.60 respectively — mostly acceptable, providing portfolio diversification in bear regimes.

### Where 103 is weaker or unproven
- **Higher variance** (σ=0.33 vs 0.12) — the distribution is more fat-tailed, not uniformly better. A few big wins inflate the mean.
- **Fewer symbols above 0.8 floor** (88% vs 95%) — 3 symbols (BA, UNH, XLV) are below the 0.8 floor in 103 but only 0 in 102.
- **BA, UNH, XLE all regressed** — the shorter lookahead (5 vs 10 days) may be mismatched for slower-moving industrial/healthcare/energy names.
- **No portfolio-level backtest exists.** The regime logic (position sizing by regime, BEAR=no buys, trailing stops, consecutive-sell filter) only runs in LEAN or the live runner. OOS Sharpe from the notebook sim does not capture these portfolio-level effects at all. 103's advantage could shrink or reverse when regime constraints reduce trade frequency.
- **min_hold_days=20** in 103 is a meaningful constraint not in 102 — this will reduce turnover and tax drag, but may also prevent timely exits.

---

## Conclusion

The per-symbol OOS Sharpe data provides **weak evidence that 103 has better individual model quality** (higher mean, more Sharpe > 1.0 symbols), but this is not a fair apples-to-apples comparison:

1. The watchlists differ (103 has 6 new symbols, removed 3).
2. The label configuration differs (lookahead 5 vs 10, threshold 3% vs 4%).
3. The most important differences in 103 (regime gating, trailing stops, consecutive-sell filter, min_hold_days=20) are **not captured in the per-symbol OOS Sharpe at all**.

**To actually know which strategy is better, you need to run `lean backtest` for both over the same 2024-01-01 – 2026-03-26 window and compare portfolio-level Sharpe, drawdown, and after-tax return.** The current data is insufficient to make a definitive claim.

---

## Next Steps to Get a Real Answer

```bash
# Run LEAN backtests for both
cd backtesting/renquant_102 && lean backtest .
cd backtesting/renquant_103 && lean backtest .

# Then analyze results
python scripts/analyze_backtest.py --strategy renquant_102
python scripts/analyze_backtest.py --strategy renquant_103
```

Or use the combined script with notifications:
```bash
python scripts/backtest_and_analyze.py --strategy renquant_102
python scripts/backtest_and_analyze.py --strategy renquant_103
```
