# 2026-06-06 — Why the BULL_CALM specialist's IC collapses under placebo

**Question (task "A"):** the Track-C BULL_CALM specialist (#233) shows raw
BULL_CALM IC **+0.0241** — best of any scorer — but it drops to ≈**+0.006**
net of a **+0.0178** time-shift placebo. Why, and can it be made
placebo-robust?

**Answer:** the pooled +0.024 is ~⅔ persistent-feature structure, BUT a
two-window split (Diagnostic 3, added after the harvest check) shows the real
story is **non-stationarity, not a stable tilt**: the specialist had **zero**
real BULL_CALM IC in 2024 (pure persistence) and flipped to strong,
placebo-*clean* alpha in 2025 (+0.088 aligned, +0.009 placebo). A signal that
is zero one year and strong the next is **not a harvestable edge**, and the
recent strength is unproven. Verdict: **harvest = NO** — pursue new fast data
(see companion scoping doc).

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

The pooled placebo stays ≈+0.04 at most displacements (vs PatchTST decaying to
−0.001). This pooled persistence is **concentrated in 2024** — see Diagnostic 3.

## Diagnostic 3 — two-window split: the edge is NON-STATIONARY (harvest check)

The specialist's 400 BULL_CALM OOS dates split into two disjoint halves,
re-scored through the same manifest contract:

| Window | real mean IC | aligned-60 real IC | 60d placebo IC | NET (real−placebo) |
|---|---:|---:|---:|---:|
| **H1** 2024-02 → 2025-01 (201d) | **+0.0005** | +0.0004 | +0.0245 | **−0.024** |
| **H2** 2025-01 → 2026-02 (199d) | **+0.0480** | +0.0884 | +0.0088 | **+0.080** |
| FULL (400d) | +0.0241 | +0.0382 | +0.0178 | +0.020 |

**The entire BULL_CALM edge lives in H2.** In 2024 the specialist had *zero*
real IC (+0.0005) and what registered was pure persistence (placebo +0.024 with
real ≈0 → NET −0.024). In 2025 it flipped to genuinely strong, **placebo-clean**
alpha (real +0.088, placebo +0.009 → NET +0.080). The pooled +0.018 placebo was
an H1 artifact, not a property of the H2 signal.

A signal that is zero one year and strong the next is **not a harvestable
factor edge** — non-stationary, and you cannot know in advance which regime
you are in. The placebo-clean H2 strength is a real *lead*, but on a single
~200-date window it is **unproven** (could be 2025-specific) and needs a fresh
forward window (post-2026-02 live data) to trust.

## What this means

- **Harvest the tilt: NO.** The "edge" is non-stationary (zero in 2024, strong
  in 2025), not a stable low-turnover factor. Untradeable as-is.
- The current feature set (alpha158 windows + fundamentals + PEAD/SUE) does not
  yield a *reliable* BULL_CALM signal — when it works (H2) it is recent and
  unconfirmed; when it doesn't (H1) it is pure persistence.
- Architecture/hyperparameter sweeps on these features won't fix
  non-stationarity. The lever is **new, faster-moving information**.

## Fork → resolved

1. ~~Harvest the tilt as a low-turnover overlay~~ — **rejected** by Diagnostic 3
   (non-stationary, unproven).
2. **Pursue genuine timing alpha with NEW fast-moving data** the current set
   lacks: news-sentiment velocity, options flow, short-interest deltas,
   intraday microstructure — signals that change faster than the 60d horizon.
   **This is the path.** Scoped in the companion doc
   `2026-06-06-bull-calm-fast-data-scoping/`.

Optional follow-up: re-check the H2 signal on post-2026-02 live data once enough
accrues — if the placebo-clean +0.088 persists forward, revisit harvest.

Reproduction: gain bucketing parses `booster_raw_json` from the specialist
artifact; placebo profile is `artifacts/diagnostics/sanity_placebo_specialist_bull_calm_20260606/`;
two-window split via `/tmp/specialist_two_window.py` (reuses the manifest
scoring path; H1/H2 = median-date split of the 400 BULL_CALM OOS dates).

Agent-Origin: Claude
