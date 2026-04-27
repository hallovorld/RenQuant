# Full self-audit on model training + buy/sell/rotate code (2026-04-27)

User concern: prior audits found 46+ bugs across narrow scope (LGBM, T2-4,
macro v2). User no longer trusts IC numbers from the 8-variant tournament
because the underlying training code may have systemic bugs not yet caught.

**Goal**: deep audit of EVERY file in the training + decision pipeline.
Catalog issues by severity. Fix all HIGH/MED. Then re-run the experiment
suite with detailed records.

## Audit scope

### Tier A — Validity-critical (could change IC numbers)

| # | File | LoC | Status |
|---|------|-----|--------|
| A1 | `training_panel/pipeline.py` | ? | pending |
| A2 | `training_panel/pp_panel_training.py` | ? | pending |
| A3 | `training_panel/labels.py` (forward-return labels) | ? | pending |
| A4 | `training_panel/factors.py` (feature engineering) | ? | pending |
| A5 | `training_panel/panel_frame.py` (panel assembly) | ? | pending |
| A6 | `training_panel/purged_cv.py` (cross-validation) | ? | pending |
| A7 | `training_panel/ltr_model.py` (XGBoost backend) | ? | pending |
| A8 | `training_panel/lgbm_ltr.py` (LightGBM backend) | ? | partial — 12-bug audit done; re-check post-fix |
| A9 | `kernel/pipeline/pp_training_full.py` (top-level orchestrator) | ? | pending |
| A10 | `kernel/panel_pipeline/feature_matrix.py` (inference) | ? | pending |
| A11 | `kernel/panel_pipeline/panel_scorer.py` (inference) | ? | pending |

### Tier B — Important (data quality)

| # | File | Status |
|---|------|--------|
| B1 | `training_panel/neutralization.py` | pending |
| B2 | `training_panel/imputation.py` | pending |
| B3 | `training_panel/global_calibrator.py` | pending |
| B4 | `training_panel/hourly_features.py` | pending |
| B5 | `training_panel/minute_features.py` | pending |
| B6 | `training_panel/ngboost_head.py` | pending |
| B7 | `kernel/macro.py` | pending |
| B8 | `kernel/macro_per_ticker.py` | done (commit dafdbbc) |

### Tier C — Buy/sell/rotate

| # | File | Status |
|---|------|--------|
| C1 | `kernel/pipeline/job_sell.py` + `task_sell.py` | pending |
| C2 | `kernel/pipeline/job_candidates.py` + `task_candidates.py` | pending |
| C3 | `kernel/pipeline/job_rotation.py` + `task_rotation.py` | pending |
| C4 | `kernel/pipeline/job_selection.py` + `task_selection.py` | pending |
| C5 | `kernel/rotation.py` (legacy) | pending |
| C6 | `kernel/rotation_convex.py` (T2-4) | done (commit 5389f22) |
| C7 | `kernel/pipeline/job_gates.py` + `task_gates.py` | pending |

## Findings

### 🔴 HIGH-1 — `purged_cv.py` purge uses calendar days but labels use bars

**File:line**: `training_panel/purged_cv.py:80, 272-274` (PurgedKFold + CombinatorialPurgedCV)

**Code**:
```python
purge_start = pd.Timestamp(test_start) - pd.Timedelta(days=int(self.lookahead_days))
```

**But labels are constructed as BAR-shift**:
```python
# pp_panel_training.py:1263
fwd_returns[t] = c.shift(-lookahead) / c - 1.0   # shift by `lookahead` BARS
```

**Effect**: With production `lookahead_days=10`:
- Label horizon = 10 BARS = ~14 calendar days
- CV purge = 10 CALENDAR days = ~7 trading days
- Training rows in `[test_start − 14cal, test_start − 10cal]` (≈ 3 trading days × 99 tickers) carry
  labels that REACH INTO the test window
- Per CPCV split: ~300 leaked training rows
- Across 15 splits: ~4500 leak instances per CV run

**IC impact**: every reported IC in the 8-variant tournament is inflated.
PROD's "+0.0482" likely overstates true OOS IC by 10-25%.

**Both PurgedKFold (line 80) and CombinatorialPurgedCV (line 272) have
this bug.** Same fix needed for both.

**Fix**: count purge in BARS using positional offsets on the unique-dates
array, not calendar-day Timedeltas. Same for embargo_days (line 82, 275).

**Invalidates**: every IC in the 8-variant tournament. Re-run after fix
required to get true OOS numbers.



## Re-experiment plan

After all HIGH/MED bugs are fixed, re-run:

1. **Baseline confirm**: PROD XGB no-macro retrain — verify IC ≈ 0.0482
   (anchor for all subsequent comparisons)
2. **Macro v2** (XGB + per-ticker β): re-confirm IC stays ~0.037 or moves
3. **LGBM no-macro / LGBM + macro v2**: re-confirm
4. **Asset embeddings**: train, then XGB + embeddings
5. **Regime ensemble (4 models)**: train 4, route at inference

Each experiment recorded in `doc/experiments/full-tournament-2026-04-XX.md`
with the exact commit, config, IC mean ± std, n_splits, calibrator status,
and runtime.

## Trust-restoration principle

A bug found in a Tier A file invalidates ALL prior IC numbers from runs
that touched that file. Document scope of invalidation explicitly so the
operator can decide what to re-run.
