# 2026-05-18 — MCD rebuy incident postmortem

## Incident

**13:53 PT**: Live runner placed `BUY MCD 3 shares @ $276.39` (= $829 invest).

**Suspicious**: MCD had been SOLD by the strategy on **2026-05-17 16:09 PT** at ~$272 (realized **+$14.02 gain** per CLAUDE.md status). Re-buying 21 hours later at **+$5/share higher** = **−$15 round-trip cost** that ate the entire prior gain.

**Detection**: User caught the pattern visually within minutes of the trade landing in the daily summary log. Order was still in `accepted` state (market closed); MCD order **canceled 14:00 PT**, DUK order kept.

## Root cause analysis

Three causal layers:

### Layer 1 (proximate): Calibrator saturation

```
2026-05-18 13:53:17 WARNING: CALIBRATOR-SATURATED:
  rank_score IQR=0.000 (<0.05) OR fraction>=0.95=1% (>=50%).
  Ranking is degenerate today; top-K selection is tie-broken by
  panel_score / ticker order, not by calibrated probability.
```

The fresh calibrator (refit today at 13:28 with the new sentiment-enriched model) collapsed 50%+ of today's 69 candidates to the same calibrated probability. The post-calibration `rank_score` had IQR=0, so the top-K selection fell back to tie-breaking by `panel_score` / ticker order rather than by real probability.

Possible mechanisms:
- Today's regime = BULL_CALM → sentiment gate OFF → sentiment cols zeroed → tighter μ̂ distribution → all μ̂ values land in a flat region of the isotonic curve.
- The isotonic calibrator was refit on FULL panel (incl. sentiment) but inference today fires with sentiment zeroed → calibrator inputs are out-of-distribution → flat-region landing.

This is **the primary technical defect** — it leaks today's "no model signal" condition into "model picks arbitrarily" rather than "model holds cash + skips trades".

### Layer 2 (architectural): No anti-churn memory

The existing wash-sale guard (`is_wash_sale_blocked_with_cost`) is **CORRECT per IRC §1091**: it blocks only LOSS sales (the legal cost-deferral cost). Gain sales pass through because §1091 explicitly excludes them.

But there was **no behavioral guard** against immediately re-buying a name JUST SOLD at a gain. The model's selection pipeline has no concept of "I just sold this; today's pick should require materially different conviction." So when the calibrator saturation tie-broke today's ranking, MCD got back in via panel_score / ticker order with zero opposition from the lot-history layer.

This is the **deeper architectural gap** that allowed Layer 1's failure to translate into a real trade.

### Layer 3 (strategic): QP tie-break optimizes for the wrong objective

When `rank_score` is degenerate, the QP optimizer still has to allocate $4,780 cash. It maximizes `μ·w − γ·σ²·w²` subject to caps. With μ values ≈ constant, the optimizer essentially:
- Pick candidates with **low σ** (Kelly denominator) → defensive names favored
- **Available cash** drives upper bound on each position
- **No current holding** = more room to add

MCD: low vol (defensive consumer staple), no current holding, available cash → ✓✓✓ for QP. The "I just sold this yesterday" is invisible at this layer.

## The trade in dollar terms

```
2026-05-17 ~16:00 PT:  SELL MCD ×3 @ ~$272.00  →  +$14.02 realized gain
2026-05-18 13:53 PT:   BUY  MCD ×3 @ $276.39   →  $829.17 cost
                                                  (= $13.17/share higher
                                                  than sold price)

Round-trip economic cost:
  Slippage (1 spread-cross × 2):     ~$1-3
  Repurchase premium ($5/share × 3):  $15
  Net: realized gain $14 - rebuy cost $15 = −$1 net
  
Excluding taxes; including taxes (~30% on the $14):
  Actually +$14 → +$9.80 after tax
  Net round-trip: $9.80 - $15 = −$5.20

Conclusion: round trip burned $5 of capital while the strategy
generated zero new conviction signal.
```

## Fix shipped (commits today)

### 1. Hard guard: `min_reentry_days` in QP wash-sale mask
- `kernel/portfolio_qp/tasks.py::ComputeWashSaleMaskTask`
- Compounds with §1091 wash-sale: ticker is blocked from Δw>0 if EITHER fires
- Default `min_reentry_days = 5` (≈ 1 trading week — enough for sentiment + technical signals to materially shift)
- Operator override via `strategy_config.json::min_reentry_days`

### 2. Regression tests
- `tests/test_min_reentry_block.py` — 11 tests pinning:
  - Config stamping (golden + live + provenance string)
  - Behavioral: gain-sale yesterday blocked by anti-churn
  - Boundary: gain-sale 10d ago not blocked (outside window)
  - Compounding: loss-sale 10d ago still blocked by §1091
  - Backwards compat: `min_reentry_days=0` disables anti-churn
  - Source-code shape (key strings present)

### 3. Live order canceled
- MCD order id `7b3d5445` canceled via Alpaca API at 14:00 PT (still `accepted`, market closed)
- DUK order kept (new wl200 ticker, defensible exploration; not affected by Layer 2/3 since never held before)

## Open issues (Layer 1 calibrator still NOT fully fixed)

The anti-churn guard fixes **the SYMPTOM** (bad ticker selected). The root **calibrator saturation** is still active and will degrade tomorrow's ranking too.

Next-session priority:
- **Diagnose μ̂ distribution today** vs calibrator's effective input range
- **Hypothesis to test**: when sentiment is gated OFF, the panel-LTR scoring on the 169-effective-feature subset produces tighter μ̂ → flat calibrator region.
  - Fix option A: train a SECOND calibrator on no-sentiment data, swap at inference time per regime
  - Fix option B: pre-clip μ̂ to calibrator's empirically-busy input range with a small noise injection
  - Fix option C: ditch isotonic for non-saturating monotone function (Platt scaling? quantile-spline?)
- **Add CALIBRATOR-SATURATED → preflight HARD FAIL** so cron skips inference rather than tie-breaking blindly.

## What the user said (verbatim)

> "mcd again？why?"
> "all the tradings make perfect sense?"
> "当然A" (cancel MCD only, keep DUK)
> "现在开修！你为什么又要偷懒？！" (Fix NOW! Why are you slacking again?!)
> "mcd的行为完全是不可接受的！" (MCD's behavior is completely unacceptable!)

## Lesson logged to memory

[[anti_churn_principle]]: any sell + immediate-rebuy pattern (regardless of P/L sign) is a strategy defect. The model has NO memory of recent decisions at the selection layer; that's exactly where guards belong. Add `min_reentry_days = 5` as the default for any signal-driven strategy.
