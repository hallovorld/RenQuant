# Panel-LTR training data format — rigorous spec + invariants

**Audience**: anyone touching `training_panel/`, `kernel/panel_pipeline/`,
`training_panel/purged_cv.py`, or any code that splits/labels/CV's the panel.

**Why this doc exists**: 4 HIGH-severity bugs (2026-04-27) all came from the
same anti-pattern — using `pd.Timedelta(days=L)` (calendar days) where the
labels are constructed via `c.shift(-L)` (BAR shift). This doc fixes the
contract once so it doesn't drift again.

This is a **living doc**. Update it when you change anything about the
panel schema, label construction, CV semantics, or split logic.

## 1. Panel layout (long-form DataFrame)

### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `date` | `pd.Timestamp` (no tz) | Bar's trading date |
| `ticker` | `str` | Ticker symbol |
| `sector` | `str` | GICS sector (UNKNOWN if missing) |
| `label` | `float` | Gaussianized residual forward-return |
| `residual_return_raw` | `float` | RAW residual (returns-scale, for NGBoost head) |
| `weight` | `float` | Sample weight = `weight_concurrency × weight_age` |
| `weight_concurrency` | `float` | AFML ch.4 average uniqueness weight |
| `weight_age` | `float` | Young-listing damping weight |
| `weight_recency` | `float` (optional) | Round-5 exponential recency weight |
| `<feature>` | `float` | Z-scored / neutralized feature columns |
| `<factor>_z` | `float` | Cross-sectional z-scored factor (size_z, mom_12_1_z, …) |
| `<col>_is_missing` | `int8 ∈ {0,1}` | Missingness indicator for nan_prone_cols |

### Sort + group invariants

1. **Sort**: panel MUST be sorted by `(date, ticker)` ascending (mergesort
   for stability) before consumption.
2. **Group sizes**: `group_sizes[i]` = count of rows whose date is the i-th
   unique date. Computed via `panel.groupby("date", sort=True).size()`.
3. **Constraint**: `sum(group_sizes) == len(panel)` always.
4. **Feature columns** = `panel.columns − {date, ticker, sector, label,
   residual_return_raw, weight*, <user drop_cols>}`.

### What's NOT in the panel

- Raw OHLCV (held in `ctx.ohlcv`, not the panel)
- Per-ticker feature frames (held in `ctx.feature_frames`,
  `ctx.factor_frames`)
- Macro broadcast frame (held in `ctx.macro_factor_frame`, only merged
  into panel rows by `build_panel_frame` with `ffill` per ticker)
- Macro per-ticker β (held in `ctx.macro_betas`, merged into
  `raw_factor_frames` BEFORE z-scoring)
- Asset embeddings (held in `ctx.asset_embeddings`, broadcast per
  ticker in `build_panel_frame` if provided)

## 2. Label construction — STRICT semantics

### Forward return (the label's underlying signal)

```python
fwd_returns[t] = c.shift(-lookahead) / c - 1.0    # at row t, this is c[t+L]/c[t] − 1
```

**lookahead is in BARS (trading days)**, not calendar days. With `lookahead=10`
and a business-day index, this spans 10 trading days ≈ 14 calendar days.

### Residualization (β-neutral)

```python
residual[t] = fwd[t]
              − β_spy[t] · spy_fwd[t]
              − β_sec[t] · (sec_fwd[t] − β_sec_on_spy[t]·spy_fwd[t])
                       # FWL orthogonalization (LBL-1 fix)
```

`β_*` use `_rolling_beta_purged` with `purge=lookahead_days` (BARS). The
`shift(purge)` operates positionally on the bar index — strict-prior.

`β` is clipped to `[clip_low=-3, clip_high=+5]` (D-1 fix).

### Cross-sectional gaussianize

Per date, rank residuals across tickers, map uniform to N(0,1) via inverse
normal CDF. Single-ticker dates get 0.0 (median).

### Output

`ctx.labels[ticker]` = pd.Series indexed by date → gaussianized residual
forward return, used by LTR model.

`ctx.raw_residuals[ticker]` = pd.Series → raw (returns-scale) residual,
attached to panel as `residual_return_raw` column for NGBoost head.

## 3. Train / test / eval / val splits — THE RULE

**Any time you split the panel into two date-disjoint segments, you MUST
purge `lookahead_days` BAR positions between them.**

This is the rule that HIGH-1, HIGH-2, HIGH-3, HIGH-4 all violated:

| Bug | File | What was wrong |
|-----|------|----------------|
| HIGH-1 | `purged_cv.py` | Used `pd.Timedelta(days=L)` — CALENDAR days, not bars |
| HIGH-2 | `pp_panel_training.py::FinalFitTask` | Train ended at n_train, eval started immediately at n_train — NO gap |
| HIGH-3 | `ngboost_head.py` | Same train/val pattern, NO gap |
| HIGH-4 | `transformer_model.py::auto_eval_split` | Same train/eval pattern, NO gap |

### Correct pattern

```python
# Given:
unique_dates = sorted(set(panel["date"]))   # bar positions
n_total = len(unique_dates)

# Compute split positions:
test_lo, test_hi = compute_split(...)            # test window (bar positions)
purge_lo  = max(0, test_lo - lookahead_days)
embargo_hi = min(n_total, test_hi + embargo_days)

# Identify dates in each segment:
test_dates    = unique_dates[test_lo:test_hi]
purge_dates   = unique_dates[purge_lo:test_lo]   # before test
embargo_dates = unique_dates[test_hi:embargo_hi] # after test

# Train: everything NOT in test ∪ purge ∪ embargo:
train_mask = (
    ~np.isin(dates, test_dates)
    & ~np.isin(dates, purge_dates)
    & ~np.isin(dates, embargo_dates)
)
```

### Why the purge MUST be in BAR positions

The label at row `t` is `c[t+L]/c[t] − 1`. It uses prices through bar
`t+L`. So a training row at bar `t` "sees" data through bar `t+L`.

To prevent training labels from leaking into a held-out segment that
starts at bar `s`, the most-recent eligible training row must satisfy
`t + L < s`, i.e. `t < s − L`. So we drop dates in `[s − L, s)` —
that's **L bars**, regardless of how many calendar days that spans.

`pd.Timedelta(days=L)` is wrong because trading days are not contiguous
calendar days. With L=10, calendar-day purge = ~7 trading days; bar
purge = 10 trading days. The 3-day gap leaks.

## 4. Cross-validation contract

Use `purged_cv.py::CombinatorialPurgedCV` (CPCV, López de Prado AFML
ch.12):

- `n_splits=6, n_test_groups=2` → 15 distinct train/test splits
- Each split applies the BAR-positional purge above
- IC is computed per-date Spearman across each test fold, aggregated to
  mean ± std + quantiles

Reportable IC fields:
- `mean_ic`: grand-average across splits
- `std_ic`: std across split-level mean ICs
- `quantiles`: `{q05, q25, q50, q75, q95}`
- `per_fold_ic_series`: per-date IC Series for each split (forensics)

## 5. Strict-prior discipline checklist

When computing ANY feature at bar `t`:
- ☐ Does it use only data with index ≤ `t`? (No `c.shift(-k)` for any k>0)
- ☐ For rolling stats: are we using `shift(1)` or
  `closed='left'`/`min_periods` to exclude bar `t` itself? (Required for
  features used as predictors of the label at `t`)
- ☐ For β estimates: are we using `purge=lookahead_days` shift on
  inputs?
- ☐ For label construction: is `shift(-lookahead)` on prices the only
  forward-looking operation, and is the result used ONLY as the label
  (never as a feature)?

## 6. Train-inference symmetry

Every Load*Task in `pp_panel_training.py::PanelDataJob.tasks` MUST be
mirrored in `training_panel/pipeline.py::prepare_inference_panel_frames`.
The symmetry-guard test
`tests/test_train_inference_symmetry.py` enforces this.

When adding a new data load (Tier 1 macro expansion, Tier 2 FRED
integration, asset embeddings, etc.), update BOTH:
1. `PanelDataJob.tasks` (training side)
2. `prepare_inference_panel_frames` (inference side)
3. `kernel/panel_pipeline/feature_matrix.py::build_inference_matrix`
   if the data needs to be broadcast/joined per ticker

## 7. Asset embeddings format (T2-2, dormant until trained)

When `asset_embeddings` is enabled (`panel_ltr.asset_embeddings.enabled=true`):

```
asset_embeddings: dict[ticker, np.ndarray]  shape (D,) float32
```

Columns added to panel: `emb_0`, `emb_1`, …, `emb_{D-1}`. Each row in
the panel for ticker `t` gets `asset_embeddings[t]` broadcast.

**LATENT-1 (2026-04-27)**: inference path currently DROPS embeddings.
Must fix `prepare_inference_panel_frames` + `feature_matrix.py::build_inference_matrix`
BEFORE enabling T2-2.

## 8. Macro v2 per-ticker β format

When `panel_ltr.macro.version="v2"` and macro is enabled:

```
ctx.macro_betas: dict[ticker, pd.DataFrame]
                 columns = [f"beta_{macro_col}_{rolling_window}d", ...]
                 indexed by date
```

Merged into `raw_factor_frames[ticker]` BEFORE `FactorZScoreTask`. So the
β columns get z-scored cross-sectionally per date alongside other
factor columns.

**Strict-prior**: β is computed with `shift(1)` so β at bar `t` uses
data through `t-1`.

## 9. Audit lineage

Bugs found and fixed in the 2026-04-27 deep audit (kept here as a
reference of "what to look for next time"):

- HIGH-1 `ba198a4`: purged_cv calendar-vs-bar
- HIGH-2 `f77de27`: FinalFit eval split, missing gap
- HIGH-3 `753d4bb`: NGBoost val split, missing gap
- HIGH-4 `df301c9`: Transformer auto_eval_split, missing gap
- LATENT-1 (open): inference path drops asset_embeddings
- M3 `dafdbbc`: macro_per_ticker constants centralized
- M4 `dafdbbc`: skipped tickers logged
- LGBM #12 `cb570ee`: removed unused data_random_seed
- ntfy silent-intraday `9afa87b`: no-op cycles silent

Empirical IC drop confirming HIGH-1: XGB no-macro pre-fix +0.0482 →
post-fix +0.0411 (−15%, exactly within the predicted 10-25% range).

## 10. When you next add a feature/data source

**Required steps (codified from the past 4 HIGH bugs):**

1. Decide: broadcast (same value per date) vs per-ticker (varies by ticker on the same date)? Pure broadcast adds 0 within-date variance — DO NOT use for cross-sectional rank loss.
2. Add training-side: `Load*Task` in `pp_panel_training.py::PanelDataJob.tasks`
3. Mirror inference-side: `prepare_inference_panel_frames` + `feature_matrix.py::build_inference_matrix` if needed
4. Add to symmetry-guard test
5. If you split panel anywhere: use BAR-positional purge of `lookahead_days`, not calendar-day Timedelta
6. Pin contract with at least ONE test in `tests/test_panel_*.py`
7. Run full suite: `python -m pytest tests/ -q`
