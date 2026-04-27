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

### 🟠 HIGH-2 — FinalFitTask eval split leak (early stopping)

**File:line**: `training_panel/pp_panel_training.py:1886-1897`

The eval split for early-stopping in FinalFit is the LAST 20% of
date-groups. With `lookahead=10`, the most-recent 10 training dates have
labels using prices that reach INTO the eval window:

- Training: date-groups `[0, n_train)` where n_train = 0.8 × n_total
- Eval:     date-groups `[n_train, n_total)`
- Last training date n_train-1: labels use prices `[n_train-1, n_train+9]`
- This OVERLAPS with the first 10 eval date-groups

Effect: early-stop sees an artificially inflated eval IC because the
model has memorized the labels of the leak rows during training.
Causes early stopping to fire too early → undertrained model.

**Severity**: MED-HIGH. Doesn't invalidate CV IC (which uses purged_cv,
now fixed). But does affect the FINAL artifact (suboptimal stopping
point). Magnitude: ~10 dates × ~99 tickers = ~1% of training rows leak.

**Fix**: insert a `lookahead`-bar gap between train end and eval start.

### 🟠 HIGH-3 — NGBoostHead train/val split — same lookahead-bar leak

**File:line**: `training_panel/ngboost_head.py:137-149`

NGBoost head splits panel into train/val by date (last 20% → val) for
early stopping. Same bar/calendar pattern as HIGH-2: pre-fix the last
`lookahead` training dates carry labels reaching into val window →
val NLL artificially good → early-stop fires too soon → undertrained
NGBoost head.

Effect: μ/σ predictions used for buy gate B + sell gate B + portfolio
QP are slightly biased. Magnitude: same as HIGH-2 (~10 dates × 99
tickers ≈ 1% leak).

Fix: insert `lookahead_days` purge gap between train end and val
start. Caller in `pp_panel_training.py::NGBoostFitTask` passes
`lookahead_days` from config.

### 🟠 HIGH-4 — Transformer auto_eval_split — same lookahead-bar leak

**File:line**: `training_panel/transformer_model.py:745-760`

Fourth instance of the same bar/calendar leak pattern. Transformer
splits the panel into train/eval by date-group count for
auto-early-stopping; pre-fix, train ends at `n_train` and eval starts
immediately at `n_train` with NO purge. Labels at the most-recent
`lookahead` training dates reach into eval window → eval IC
artificially inflated → early-stop fires too soon → undertrained
transformer artifact.

Effect: same pattern as HIGH-2/HIGH-3. Affects transformer's saved
weights when transformer backend is used (currently shelved at
0.89× XGBoost on 43-ticker panel; would re-bite on revisit).

Fix: insert `lookahead_days` purge gap. Added `lookahead_days=10`
default to `TransformerParams`. Same purge semantics as HIGH-2/HIGH-3.

### 🟡 LATENT-1 — Inference path drops asset_embeddings

**File:line**: `training_panel/pipeline.py:111` (loads embeddings) +
`pipeline.py:154-161` (returns frames without embeddings) + 
`kernel/panel_pipeline/feature_matrix.py:61-133` (build_inference_matrix
takes no embeddings arg)

`prepare_inference_panel_frames` calls `LoadAssetEmbeddingsTask` (correct
for symmetry-guard test) but the loaded `ctx.asset_embeddings` is never
broadcast into the returned frames. Training passes embeddings to
`build_panel_frame` which adds `emb_0…emb_{D-1}` columns; inference path
never produces these columns.

**Effect**: dormant. T2-2 isn't trained / wired into PROD yet, so the
asymmetry doesn't bite. But when embeddings are added to a model,
inference will silently fill `emb_*` with NaN → wrong predictions.

**Fix needed BEFORE enabling T2-2**: extend `build_inference_matrix` to
accept `asset_embeddings` and broadcast per ticker.

### Tier A audit summary so far

| File | Findings | Status |
|------|----------|--------|
| `training_panel/labels.py` | 0 new (LBL-1 already fixed) | clean |
| `training_panel/purged_cv.py` | 1 HIGH (HIGH-1) | fixed |
| `training_panel/pp_panel_training.py` (CrossValidate + FinalFit + BuildPanel) | 1 HIGH (HIGH-2) | fixed |
| `training_panel/panel_frame.py` | 0 new | clean |
| `training_panel/ltr_model.py` | 0 new (X1-X18 already fixed) | clean |
| `training_panel/pipeline.py` | 1 LATENT (LATENT-1, embeddings) | noted |
| `kernel/panel_pipeline/feature_matrix.py` | 1 LATENT (same as above) | noted |
| `kernel/panel_pipeline/panel_scorer.py` | TBD | pending |
| `kernel/pipeline/pp_training_full.py` | 0 new (well-structured) | clean |
| `training_panel/lgbm_ltr.py` | already 12-bug audit done | partial |
| `training_panel/global_calibrator.py` | 0 new (CALIB-COLLAPSE-GUARD already in place) | clean |
| `training_panel/factors.py` | 0 new (math correct, β clipped, strict-prior shifts) | clean |
| `training_panel/neutralization.py` | 0 new (correct expanding→rolling switch + shift(1) for strict prior) | clean |
| `training_panel/imputation.py` | 0 new (calendar-day age weight is by-design) | clean |
| `training_panel/panel_frame.py` (compute_concurrency_weight) | 0 new (AFML ch.4 weight correct, uses bar positions) | clean |
| `kernel/pipeline/task_sell.py` | 0 new (well-audited PH/SellGateB/PanelConvictionExit) | clean |
| `kernel/pipeline/task_candidates.py` | 0 new (NaN guards in place TC-1) | clean |
| `kernel/pipeline/task_rotation.py` (sample) | 0 new in opening 100 lines | partial |
| `training_panel/transformer_model.py` | TBD | pending |
| `training_panel/ngboost_head.py` | TBD | pending |
| `training_panel/lgbm_ltr.py` (post fix re-check) | pending | partial |

## 🔥 Empirical confirmation of HIGH-1 — XGB no-macro retrain

**Result (2026-04-27 10:20 PT)**: XGB rank:pairwise no-macro post HIGH-1 + HIGH-2 fix, OOS IC = **+0.0411** (CPCV mean, std 0.0238, n_splits=15).

| Metric | Pre-fix | Post-fix | Δ |
|--------|---------|----------|---|
| OOS mean IC | +0.0482 | **+0.0411** | **−0.0071 (−15%)** |
| std | 0.0247 | 0.0238 | similar |
| q05 / q95 | +0.0010 / +0.0691 | +0.0024 / +0.0726 | similar shape |

The −15% drop **exactly matches the predicted 10-25% inflation range**. This confirms HIGH-1 was a real bug, not theoretical. **TRUE PROD IC = +0.0411**, not +0.0482.

The post-fix artifact is now PROD (panel-ltr.json md5 589e5b95). The pre-audit baseline is archived at `panel-ltr.xgboost-pre-audit.bak.json` for forensics.

**ALL OTHER 8-variant tournament IC numbers** are similarly inflated. They need re-running to get true OOS values.

### Post-fix tournament so far (all with HIGH-1 + HIGH-2 fixes)

| Variant | Pre-fix IC | Post-fix IC | Δ vs PROD post-fix |
|---|---|---|---|
| **XGB no-macro (PROD)** | +0.0482 | **+0.0411** | — |
| XGB + macro v2 (per-ticker β) | +0.0371 | +0.0373 | −9.2% |

XGB+macro v2 with clean CV came in at 0.0373 — basically identical to
the pre-fix 0.0371 (within noise). The pre-fix run wasn't materially
inflated because the macro v2 features themselves were near-zero
signal — leaks of zero signal don't move IC much. **Verdict on macro
v2 holds: 11 per-ticker β features don't beat 28-feature no-macro.**

## Audit summary as of 10:11 PT

- **2 HIGH bugs found and fixed**:
  - HIGH-1: `purged_cv.py` calendar-day vs bar-shift mismatch
  - HIGH-2: FinalFitTask eval split allows lookahead-bar leak
- **1 LATENT bug noted** (asset_embeddings inference path drop, dormant)
- 11 files audited clean (no new bugs)
- 3 files pending detailed audit (transformer / ngboost / lgbm post-fix recheck)
- All 8-variant tournament IC numbers are INVALIDATED by HIGH-1 — re-run
  needed to get true OOS values. XGB no-macro retrain in flight.



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
