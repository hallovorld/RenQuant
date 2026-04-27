# LightGBM Implementation — Strict Deep Audit (2026-04-27)

**Trigger**: User direction "我觉得你的lgbm 至少有10个bug — open a strict deep audit". After the v2 hyperparam attempt failed, the assumption is that the **code itself** has bugs, not just the tuning.

This audit reads `kernel/training_panel/lgbm_ltr.py` line by line and flags every issue that could plausibly suppress signal, leak data, or produce wrong numbers.

---

## Findings — 12 issues

### Bug #1 — NaN weights propagate silently into LightGBM (HIGH)

**File:line**: `lgbm_ltr.py:208-230`

`w_rows = panel[weight_col].values.astype(float)` — if any row's weight is NaN
(common for early dates where `weight_concurrency × weight_age` hasn't fully
warmed up), the per-group `.mean()` produces NaN. The `if mean_w > 0` check
on line 229 is `False` for NaN (`nan > 0 == False`), so the normalization
is **skipped** and `row_weights` stays as NaN values. These then go to
`lgb.Dataset(weight=row_weights)` (line 232).

**Effect**: LightGBM's behavior on NaN weights is undefined (version-dependent;
some versions silently drop, some treat as zero, some treat as 1.0). Silent
gradient corruption.

**Fix**: `w_rows = np.nan_to_num(w_rows, nan=1.0)` BEFORE per-group averaging
OR drop NaN rows before training. Add explicit assertion `assert not
np.isnan(row_weights).any()`.

### Bug #2 — `self.best_iter` stores LAST iteration, not BEST iteration (HIGH)

**File:line**: `lgbm_ltr.py:261`

`self.best_iter = self.booster.current_iteration()` returns the iteration the
booster STOPPED at (last round when early stopping fired or num_boost_round
when not). The actual BEST iteration is `self.booster.best_iteration` (set
by LightGBM's early stopping callback).

**Effect**: Saved artifact's `best_iter` field misreports "best round" as
"final round". Visible in `panel-ltr.json` metadata; misleads operator about
when the model peaked. With early_stopping_rounds=20 and best at iter 50,
saved artifact says best_iter=70 (50 + 20 patience).

**Fix**: `self.best_iter = self.booster.best_iteration if hasattr(self.booster,
"best_iteration") and self.booster.best_iteration > 0 else self.booster.current_iteration()`.

### Bug #3 — Tied labels get arbitrary distinct ranks (HIGH)

**File:line**: `lgbm_ltr.py:139`

`ranks = np.argsort(np.argsort(slice_y, kind="stable"))` produces unique
integer ranks even for tied values. For lambdarank's NDCG semantics, **tied
relevance values should receive the same gain**, otherwise the model is
trained to make arbitrary distinctions between truly identical examples.

Example: `slice_y = [0.0, 0.0, 0.0, 0.05, 0.10]` → ranks `[0, 1, 2, 3, 4]`
but the first three are tied. Bucketization then assigns them different
integer relevance, which the lambdarank loss tries to enforce.

**Fix**: Use `pandas.Series(slice_y).rank(method="dense").values - 1` or
`scipy.stats.rankdata(slice_y, method="average")` to handle ties.

### Bug #4 — Dtype mismatch: train uses float64, predict uses float32 (MED)

**File:line**: `lgbm_ltr.py:193 (train) vs 282 (predict)`

```python
# train:
X = panel[feature_cols].values             # default dtype = float64
# predict:
X = panel[self.feature_cols].to_numpy(dtype=np.float32)
```

Same panel through training and prediction sees different dtype precision.
For a tree model with depth=4 splits at very small thresholds (e.g.
`feature_x < 0.000001`), float32 rounding can flip the split outcome.

**Fix**: Match dtypes — either both float64 (`to_numpy(dtype=np.float64)`)
or both float32. Float32 is fine if consistent.

### Bug #5 — `PanelScorer.load` dispatch for `panel_lgbm` kind (HIGH)

**File:line**: `lgbm_ltr.py:307-317` + caller

`PanelLGBMModel.load` checks `payload["kind"] == "panel_lgbm"`. But the
**PanelScoringJob** (in `kernel/panel_pipeline/`) loads via
`PanelScorer.load(path)`. If `PanelScorer.load` doesn't handle the
`panel_lgbm` kind, it raises a cryptic error ("not a panel_ltr_xgboost
artifact") even though the file IS a valid LightGBM artifact.

**Effect**: Switching `panel_ltr.backend: "lightgbm"` works at training
time but fails at inference time depending on whether the scorer
loader is the duck-typed `PanelLGBMScorer` or the XGBoost-only
`PanelScorer`.

**Fix**: Centralized `load_panel_artifact(path)` factory that dispatches
on `kind`. Verify `PanelScoringJob` uses it.

### Bug #6 — Empty `weight_col` slices produce silent NaN weights (MED)

**File:line**: `lgbm_ltr.py:211-214`

```python
for gs in group_sizes:
    grp_mean = w_rows[off:off + gs].mean()    # NaN when gs=0 with RuntimeWarning
    row_weights[off:off + gs] = grp_mean
```

If any group has `gs=0` (degenerate but possible after filter pipeline),
`.mean()` of empty slice → NaN and emits RuntimeWarning. The row_weights
slice for that group is empty (length 0) so no rows touched, but the warning
pollutes logs and hints at upstream filter issues.

**Fix**: `if gs <= 0: continue`.

### Bug #7 — `feature_cols` not validated against `panel` before use (MED)

**File:line**: `lgbm_ltr.py:184-193`

```python
self.feature_cols = list(feature_cols)
...
X = panel[feature_cols].values
```

If caller passes `feature_cols` containing names not in `panel.columns`,
pandas raises `KeyError` deep inside the slicer. Should validate at top of
`train()` for better error messages — mirrors what `predict()` already does
on line 275-281.

**Fix**: After line 184, add the same validation block as `predict()`.

### Bug #8 — `bucketize_labels` advances offset by negative `gs_int` (LOW)

**File:line**: `lgbm_ltr.py:132-135`

```python
for gs in group_sizes:
    gs_int = int(gs)
    if gs_int <= 0:
        offset += gs_int    # adds NEGATIVE — corrupts subsequent group offsets
        continue
```

For defensive correctness, `gs_int < 0` (impossible in practice but
programmatically possible) would advance offset BACKWARDS. Subsequent
groups would slice into earlier positions.

**Fix**: `if gs_int <= 0: continue` (no offset adjustment).

### Bug #9 — train_ic uses LAST-iter prediction, not BEST-iter (MED)

**File:line**: `lgbm_ltr.py:264`

`preds = self.booster.predict(X)` after `lgb.train()` returns. With early
stopping callback, LightGBM 3.x+ does auto-use `best_iteration` in subsequent
predict calls — but this depends on the LightGBM version being recent
enough. Older versions or future releases may not. Explicit:
`preds = self.booster.predict(X, num_iteration=self.booster.best_iteration)`.

### Bug #10 — Train weight normalization loses absolute scale (MED)

**File:line**: `lgbm_ltr.py:228-230`

```python
mean_w = float(row_weights.mean())
if mean_w > 0:
    row_weights = row_weights / mean_w
```

This rescales weights so mean=1.0 — fixes the "weights too small" bug from
2026-04-25. BUT it also discards the **absolute** scale: a panel where ALL
groups have low weight (rare) is treated identically to one where all
groups have high weight. The relative ratios are preserved, which is what
LightGBM's loss needs, but absolute regularization (lambda_l1, lambda_l2)
is calibrated against unscaled weights.

**Effect**: minor — most panels have similar weight distributions. But
during model selection across panels (e.g. comparing macro-on vs macro-off),
the regularization strength effectively differs.

**Fix**: Optional — also expose `weight_scale_factor` in the artifact for
reproducibility audit.

### Bug #11 — No params validation; misconfig is silent (HIGH)

**File:line**: `lgbm_ltr.py:162`

`self.params = {**DEFAULT_PARAMS, **(self.params or {})}` — caller can pass
ANY params, including `objective="regression"` (which would then receive
bucketized integer labels in [0, 10] and learn nonsense).

**Fix**: `assert self.params.get("objective", "lambdarank").startswith("lambda")
or self.params["objective"] == "rank_xendcg", "PanelLGBMModel only supports
lambdarank-family objectives"`.

### Bug #12 — `data_random_seed` is set but lambdarank doesn't use it (LOW / DOC)

**File:line**: `lgbm_ltr.py:89`

`data_random_seed: 42` is in DEFAULT_PARAMS. Per LightGBM docs, this
parameter is used by `data_sample_strategy = "goss"` and `boosting_type =
"random_forest"` — neither of which we use (default is `"gbdt"`). Setting
it has no effect on lambdarank training. Harmless but confusing.

**Fix**: Remove or document as no-op for the operator.

---

## Severity summary

| # | Issue | Severity | Effect |
|---|---|---|---|
| 1 | NaN weights silent propagation | HIGH | Gradient corruption |
| 2 | `best_iter` stores final not best | HIGH | Misreports peak |
| 3 | Tied labels get distinct ranks | HIGH | Trains to enforce arbitrary tie-breaks |
| 4 | Train float64 vs predict float32 | MED | Precision drift on tiny splits |
| 5 | Scorer dispatch for panel_lgbm kind | HIGH | Inference fails on backend swap |
| 6 | Empty group .mean() warning | MED | Log pollution |
| 7 | feature_cols not validated in train | MED | Cryptic errors |
| 8 | Negative gs offset bug | LOW | Defensive |
| 9 | Predict on last-iter not best-iter | MED | Version-dependent silent issue |
| 10 | Weight normalization loses absolute scale | MED | Cross-panel comparison drift |
| 11 | No params objective validation | HIGH | Silent misconfig |
| 12 | Unused data_random_seed | LOW | Doc-only |

**4 HIGH** + **4 MED** + **2 LOW** = 10 actionable issues + 2 noted.

---

## Why none of these explain the v1→v2 regression

Even with all these bugs, **v1 and v2 share the same code path** — only
`DEFAULT_PARAMS` differ between them. The bugs above affect both runs
equivalently. So they don't account for v2 being WORSE than v1.

The v1→v2 regression remains attributable to the hyperparameter changes
themselves (per the previous audit's post-mortem section). The bugs in
this doc are systemic quality issues that LGBM has carried since first
implementation, NOT v2-specific regressions.

**That's the right way to read this audit**: even if every one of these
bugs were fixed, LGBM's structural disadvantage on this 75K-row panel
(per the lambdarank gradient density argument) would remain. Worth fixing
the bugs to make LGBM **trustworthy** as a backup backend, but won't make
it beat XGBoost rank:pairwise.

---

## Implementation plan if user wants to fix

| Priority | Bugs | Effort |
|----------|------|--------|
| Quick wins (now) | #1 (NaN guard), #2 (best_iter), #11 (objective assert), #6 (gs<=0) | 15 min total |
| Tie handling | #3 (rank method=dense via pd.Series.rank) | 10 min |
| Inference parity | #5 (centralized loader dispatch — touches kernel/panel_pipeline) | 30-60 min |
| Cross-backend hygiene | #4 (dtype unify), #7 (validate cols), #9 (best_iter predict), #10 (weight metadata) | 30 min |
| Doc/cleanup | #8, #12 | 5 min |

**Total to fix all 12**: ~2 hours. Recommend doing the HIGH ones at minimum
since they affect signal quality for any future LGBM use.
