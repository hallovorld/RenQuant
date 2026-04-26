# Calibrator Saturation — Pool IC Collapse Across Backends (2026-04-26)

**Discovered**: 2026-04-26 11:14 PT during e2e R7 score_distribution review.

## Symptom

Score-distribution snapshot for 2026-04-26 R7 (live runner, XGBoost calibrator
active at the time):

| ticker | rank_score | panel_score (raw) |
|---|---|---|
| APP    | 0.34474 | +0.0067 |
| FTNT   | 0.34474 | +0.0009 |
| NFLX   | 0.34474 | +0.0036 |
| BA     | 0.34474 | +0.0054 |
| PLTR   | 0.34474 | +0.0126 |
| TSM    | 0.34474 | +0.0081 |
| CAT    | 0.30332 | -0.0009 |
| JNJ    | 0.27698 | -0.0029 |
| GLD    | 0.26460 | -0.0043 |
| GOOG   | 0.26460 | -0.0036 |
| AMD    | 0.23894 | -0.0129 |
| IBM    | 0.23894 | -0.0096 |
| PANW   | 0.23894 | -0.0100 |
| AMZN   | 0.23894 | -0.0092 |

Six of the top seven candidates collapse to **rank_score = 0.34474**. Daily
percentile row: `p75 = p85 = p90 = p95 = 0.34474` — saturation across the
entire upper quartile.

## Root cause

Production calibrators today have **almost no resolution**:

| Calibrator (artifact) | x_len | unique y | y range | pool_ic | scorer_oos_ic |
|---|---:|---:|---|---:|---:|
| `panel-rank-calibration.xgboost.bak.json` (active during e2e R6/R7) | 10 | **6** | 0.239 → 0.345 | 0.0011 | 0.0482 |
| `panel-rank-calibration.lightgbm.bak.json` (post-LGBM, transient at 11:00 PT) | 2 | **1** | 0.274 → 0.274 | 0.0097 | 0.0291 |
| `panel-rank-calibration.lgbm.bak.json` (April 23 — healthy reference) | 64 | **33** | 0.0 → 1.0 | 0.0291 | 0.0269 |

The XGBoost calibrator has only **6 distinct output values** — top tier
saturates at 0.34474, mid at 0.30, bottom at 0.24. Anything that lands in
the same isotonic bin gets the identical y.

The LightGBM calibrator collapsed to **constant** y = base_rate (0.274).
Pool data couldn't distinguish positive from negative class on the panel
scores at all.

## What the gap signals

**`pool_ic` is 50× lower than `scorer_oos_mean_ic`** for both XGBoost and
LightGBM:

```
xgb:   scorer_oos_ic = 0.0482   pool_ic = 0.0011   ratio = 44×
lgbm:  scorer_oos_ic = 0.0291   pool_ic = 0.0097   ratio =  3×
```

`scorer_oos_mean_ic` is computed by `panel.ltr` on the held-out fold
during CPCV. `pool_ic` is computed by `scripts/recalibrate_scores.py` on
the calibrator's training pool — same scorer, but a different time
window / aggregation. The gap means the panel scores **lose ~98% of
their predictive power between the CPCV eval and the calibrator fit
pool**. Possibilities:

1. **Pool window mismatch** — calibrator pool may include very-recent
   bars where the scorer hasn't generalised, while CPCV uses purgeable
   embargo splits.
2. **Z-score parity issue** — calibrator may receive different
   neutralisation/factor z-scores than the scorer's training input,
   shrinking the score's ability to rank.
3. **Forward-return labelling drift** — calibrator labels (binary
   indicator on `forward_return > threshold`) may use a different
   threshold or lookahead than CPCV's spearman target.
4. **Class imbalance** — base rate 0.274 means ~27% of pool labels are
   positive. With only ~225k pool rows pooled across 101 tickers, the
   per-bucket signal-to-noise after binning into isotonic intervals
   may be too low for the calibrator to discriminate.

## Why this didn't break R6/R7 buys

The decision pipeline uses **`panel_score` (raw) and `μ` (NGBoost)** for
`net_alpha` ranking — not `rank_score`. `panel_buy_floor` is `null` in
the current strategy_config, so the calibrator-collapse doesn't gate the
buy. Calibrated `rank_score` only affects:

- The `score_distribution` analytics table (saturated percentiles,
  diagnostic-only)
- Any downstream that READS `rank_score` for ranking — currently only
  the rotation `swap_margin` veto, which compares against held's
  `rank_score`. With both held and cand collapsing to the top tier (0.345),
  rotation can't distinguish.

So R6/R7 buys were driven by μ ≈ +0.008 (NGBoost) and panel_score
(NET=+0.0176 > APP=+0.0067), with rank_score acting as a noisy tie.

## Action queue

This is the actual blocker for ranking quality, ahead of further
transformer work.

1. **Audit calibrator fit pool vs scorer eval pool** — `recalibrate_scores.py`
   should consume the same `panel-ltr.json` test indices the scorer
   reported `oos_mean_ic` on. Today the pool is reconstructed
   independently and ~50× weaker.
2. **Document the pool/eval split contract** in `panel-rank-calibration.json`
   metadata (currently only `n_rows`, `n_tickers` are stamped).
3. **Add a CI/test guard** that rejects any calibrator with `unique_y < 5`
   or `pool_ic < 0.5 × scorer_oos_ic`. Today it would have caught the
   constant LGBM calibrator at write time.
4. **Re-evaluate base_rate threshold** — `0.03` forward return at 10
   lookahead days produces 27% positive rate on this universe. If the
   threshold is too tight relative to typical 10-day moves in BULL_CALM,
   isotonic loses signal. Try `threshold = 0.5 × er_std × √lookahead_days`
   or quantile-based.

## Cross-reference

- Implementation: `backtesting/renquant_104/kernel/panel_pipeline/scoring.py::ApplyGlobalCalibrationTask`
- Fitter: `scripts/recalibrate_scores.py` + `kernel.calibration.fit_panel_calibrator`
- Schema: `panel-rank-calibration.json` (probability + expected_return maps)
- Related: `doc/post_tier1_followups_2026-04-25.md` (Tier 1 retrain notes)
