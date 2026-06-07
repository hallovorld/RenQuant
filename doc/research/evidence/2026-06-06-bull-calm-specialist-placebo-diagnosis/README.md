# 2026-06-06 — Why the BULL_CALM specialist's IC collapses under placebo

**Question (task "A"):** the Track-C BULL_CALM specialist (#233) shows raw
BULL_CALM IC **+0.0241** — best of any scorer — but it drops to ≈**+0.006**
net of a **+0.0178** time-shift placebo. Why, and can it be made
placebo-robust?

**Answer:** the specialist's signal is a **near-static cross-sectional factor
tilt**, not timing alpha. ~⅔ of its gain comes from features whose 60-day
ranking barely moves, so its predictions correlate with returns at *any* time
displacement. There is **no placebo-robust BULL_CALM timing alpha** in the
current 172-feature set. Two diagnostics prove it.

## Diagnostic 1 — 65% of gain is persistent features

XGB gain importances of `artifacts/walkforward_bull_calm_specialist/bull_calm/2025-11-24/panel-ltr.json`
(kind `panel_ltr_xgboost`, 172 features, `fwd_60d_excess`, `regime_filter=BULL_CALM`),
bucketed by feature lookback:

| Bucket | Gain share |
|---|---:|
| long-window ≥30d technical (MIN60, STD60, MAX60, RSV60, QTLD60, CORD60 …) | **53.1%** |
| quarterly fundamentals (book_to_price, earnings_yield, roe, asset_growth …) | 12.1% |
| mid-window 11–29d | 16.5% |
| **short-window ≤10d (fast)** | **13.5%** |
| other | 4.9% |

**Persistent (long-window + fundamental) = 65.2%.** A 60-day-window feature on
day *t* and day *t+60* shares almost its entire window, so its cross-sectional
ranking is nearly invariant over the placebo horizon. The model is ranking
stocks on slow-moving exposures.

## Diagnostic 2 — placebo IC never decays (static-ranking signature)

Specialist overall shift profile (model-placebo IC = predictions vs
time-shifted forward returns):

| Shift (days) | Aligned real IC | **Model-placebo IC** |
|---:|---:|---:|
| 5 | +0.0449 | **+0.0494** |
| 20 | +0.0492 | **+0.0479** |
| 60 (true) | +0.0636 | **+0.0266** |
| 120 | +0.0713 | **+0.0524** |
| 252 | −0.0079 | **+0.0409** |

The placebo stays ≈+0.04 at **every** displacement, including 252 days. For
comparison, PatchTST's placebo decays to **−0.001** by shift 60. A model whose
predictions correlate with returns at unrelated horizons has an essentially
**static ranking** — it is expressing a persistent factor tilt, and the
val-window IC is that tilt's realized return, not timing skill.

## What this means

- **No placebo-robust BULL_CALM timing alpha exists in the current features.**
  Every scorer in BULL_CALM is either a static tilt (specialist — placebo-dirty)
  or ≈zero (PatchTST/XGB — placebo-clean). Removing the persistent component
  from the specialist leaves ~+0.006, the same ~zero band as the others.
- The feature set (alpha158 windows + fundamentals + PEAD/SUE) is **exhausted
  for BULL_CALM timing**: its informative content there is slow factor exposure
  already largely priced.

## Fork for next work

1. **Harvest the tilt as a low-turnover BULL_CALM factor overlay.** The static
   ranking, *if* its forward value is confirmed out-of-window (the val-window
   IC could be one-window factor luck — placebo can't tell), is cheap to trade
   (low turnover, since static). Needs a true OOS-on-a-second-window check
   before trusting it — the placebo flags it as "not timing," not "tradeable."
2. **Pursue genuine timing alpha with NEW fast-moving data** the current set
   lacks: news-sentiment velocity, options flow, short-interest deltas,
   intraday microstructure — signals that change faster than the 60d horizon.
   This is the only path to placebo-robust BULL_CALM *timing* IC.

**Not worth doing:** more model-architecture swaps or hyperparameter sweeps on
the existing features — the diagnosis says the signal isn't there to find.

Reproduction: gain bucketing parses `booster_raw_json` from the specialist
artifact; placebo profile is `artifacts/diagnostics/sanity_placebo_specialist_bull_calm_20260606/`.

Agent-Origin: Claude
