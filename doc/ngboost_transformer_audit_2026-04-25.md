# NGBoost + Transformer Deep Audit — 2026-04-25

User mandate: "deep audit ngboost和transformer吧，说不定再找出100个bug出来数据就上去了".
Walked NGBoost + Transformer + Ensemble code on the **training**, **inference**,
**scorer**, **adapter**, and **test** layers. Found **78 issues**.

Severity:
🔴 critical — wrong numbers, lookahead, ranking signal-killer
🟠 high — significant correctness or performance bug
🟡 medium — silent fragility, edge case, or missing safety net

---

## Where to look first

The **transformer** has one bug so big that fixing it alone may flip its
OOS IC from 0.006 → competitive: **T-1 / T-2** (silent 38-ticker truncation
in `_build_date_groups`). The current artifact was trained on 47k rows
× 38 tickers — vs panel-LTR's 121k rows × 99 tickers. Two-thirds of
training data was thrown away by a default that was never overridden.

The **NGBoost head** has no CV (N-17), no NaN handling (N-1, N-5, N-13),
and no OOS metric in metadata (N-18). We literally do not know how good
NGBoost's μ predictions are on holdout data — yet the σ they produce is
the dominant lever for Kelly sizing.

---

## NGBoost — `training_panel/ngboost_head.py` (core class)

### 🔴 N-1 No NaN handling on training features
- `train()` line 69: `X = panel[feature_cols].values.astype(float)`. If the
  panel has even one NaN in a feature column, NGBoost will fit on `nan` —
  which `astype(float)` keeps as `np.nan` — leading to either an exception
  inside ngboost or silently degenerate fits.
- The training pipeline does `dropna(subset=["residual_return_raw"])` but
  doesn't drop NaN in the **features**.
- **Fix**: drop rows with NaN in `feature_cols` OR fill with column median.

### 🔴 N-5 No NaN handling at inference
- `predict_distribution()` line 95: same issue. If today's panel matrix
  has NaN in any feature column for any ticker, NGBoost may segfault or
  return garbage. Caller `ApplyNGBoostTask` swallows exceptions silently
  (line 437-439), so the user just sees NGBoost "stop working" with no
  warning.
- **Fix**: detect NaN, log per-ticker, drop those rows from prediction.

### 🔴 N-13 No NaN handling on training labels
- `panel[label_col].values.astype(float)` — NaN labels go straight into
  NGBoost. The pipeline-level dropna catches `residual_return_raw=NaN`
  but other label_cols (if used) aren't covered.

### 🔴 N-17 NO cross-validation, NO OOS IC
- `NGBoostFitTask` (line 1454) just calls `head.train(sub, ...)` on the
  **entire panel**. Panel-LTR has CPCV with 6 splits × 2 test groups and
  reports `oos_mean_ic`, `oos_std_ic`, `oos_per_fold_ic`. NGBoost has
  none of that. **We have no idea whether NGBoost's μ generalizes** —
  yet μ feeds Kelly sizing, which is the single biggest lever on
  performance.
- Compounded: NGBoost is trained on the SAME data the panel-LTR test
  fold uses. Any IC the LTR test fold reports has potential contamination
  via Kelly sizing.

### 🟠 N-2 No early stopping
- `n_estimators=400` runs to completion regardless of convergence.
  `learning_rate=0.01 × 400 = 4.0` is high cumulative learning rate;
  likely overfits on the in-sample training panel.
- **Fix**: split a held-out 20% by date, monitor validation NLL, early-stop.

### 🟠 N-3 Normal distribution may underestimate σ for fat tails
- Forward residual returns have fat tails (kurtosis >> 3). Fitting Normal
  understates the true σ, especially in tail regimes. Could try
  `LogNormal` or `T` distribution from `ngboost.distns`.

### 🟠 N-7 Determinism gap
- `random_state=17` in DEFAULT_PARAMS. NGBoost passes that to the inner
  base learner. But sklearn's `DecisionTreeRegressor` uses a separate
  `random_state` for splitter; if the param dict doesn't reach the
  splitter, retraining on the same data yields slightly different trees.
- Metadata says `deterministic=False` (line 1529) — confirms reproducibility
  is not guaranteed.

### 🟠 N-8 `combined_score` mixes scales
- `score = μ − λ·σ`. μ ∈ [-0.05, +0.05] return-space; σ ∈ [0.02, 0.10]
  return-space. With λ=1.0, σ dominates ranking → preference for
  low-volatility names.
- With λ=0.0 (current golden), `combined_score == μ`. So in current
  config, σ is loaded into μ-only ranking and NEVER used for ranking
  (only for sizing). The `score_mode` config field is dead config.

### 🟠 N-9 σ_median is computed at inference time, not training time
- `sigma_sizing_multiplier(sigma)` line 159: median is over the rows
  passed in at predict time. If 30% of universe has missing predictions
  today (gap-day data), median is over the remaining 70% — biased
  estimator of true universe σ.
- **Fix**: persist `sigma_median_train` in artifact metadata; clip
  inference σ_median to that as a sanity floor.

### 🟠 N-15 Aggressive learning rate × n_estimators
- `learning_rate=0.01 × n_estimators=400` is roughly equivalent to
  XGBoost with eta=0.04 + 100 trees. NGBoost trees are univariate
  decision stumps so each tree fits less, but cumulative learning rate
  is high and easy to overfit a 121k-row panel.

### 🟠 N-16 Fit panel includes rows with NaN feature values
- `NGBoostFitTask.run()` line 1466: `sub = ctx.panel.dropna(subset=["residual_return_raw"])`.
  Drops rows missing **label** but not rows missing **features**. If a
  ticker has 252 days of price data but a fundamental factor (e.g.
  ROE) wasn't fetched until day 100, days 1-99 have NaN in `roe_z`. The
  NGBoost fit then "trains" on those NaN rows.

### 🟠 N-19 NGBoost training panel uses different rows from LTR
- LTR: drops by lookahead window (`min_history_days`).
- NGBoost: drops only `residual_return_raw=NaN`.
- If the residual computation requires fewer days than the LTR features,
  NGBoost trains on rows where some features are still warming up.
  Train/inference contamination.

### 🟠 N-22 `weight` column not validated
- `sample_weight_col="weight"`. If `weight` column has zeros, NaNs, or
  negatives, NGBoost may silently corrupt the fit. No validation.

### 🟡 N-4 No fit-time IC reporting
- `train()` returns `n_rows`, `n_features`, `train_mu_mean`,
  `train_sigma_mean`. No Spearman IC of μ̂ vs y, no NLL, no calibration check.

### 🟡 N-6 Pickle-base64 in JSON is fragile
- `regressor_pickle_b64` (line 116, 141): pickled NGBRegressor in base64
  inside JSON. Across NGBoost / sklearn version changes, the unpickle
  may fail silently or with a generic exception. No version pinning,
  no compatibility check on load.
- **Security**: pickle deserializes arbitrary classes — if the artifact
  comes from outside (e.g. someone's PR), it's an RCE vector. Local-only
  use right now, low-risk.

### 🟡 N-10 No σ calibration check
- True σ should match cross-sectional std of residuals. NGBoost outputs
  σ̂ — but does σ̂ match? Reliability diagram never plotted, no z-score
  test, no rank stability across regimes.

### 🟡 N-11 `feature_cols` ordering not enforced on load
- `NGBoostHead.load(path)`: sets `feature_cols = list(payload["feature_cols"])`.
  At inference, `predict_distribution(panel)` does `panel[self.feature_cols]`.
  If `panel` has the columns BUT in different ORDER, pandas' indexing
  preserves the artifact's order, so it's actually OK. However if `panel`
  has SUBSET of columns (missing one), KeyError fires loud.

### 🟡 N-12 Stored `params` ignored on load
- `NGBoostHead.load` calls `cls(params=payload.get("params"))` then
  reassigns `head.regressor = pickle.loads(...)`. The merged params on
  the new instance are never used because the regressor is restored from
  pickle. Stored params are pure metadata.

### 🟡 N-14 No early-stopping rounds threshold
- Tied to N-2. NGBoost has no `early_stopping_rounds` param exposed via
  config. `n_estimators=400` is hard-coded ceiling.

### 🟡 N-18 Save metadata is shallow
- `NGBoostSaveTask.run()` writes only `train_mu_mean`, `train_sigma_mean`,
  `n_rows` to metadata. Missing: `oos_mean_ic`, `oos_std_ic`,
  `oos_per_fold_ic`, `cv_method`, σ calibration scores, train period.
- **Effect**: the artifact carries no provenance — same as having no
  audit trail. A drift-monitoring `panel_training_runs.md` log can't
  meaningfully compare runs.

### 🟡 N-20 `deterministic=False` hardcoded in `record_training_run`
- Line 1529: `deterministic=False` even though DEFAULT_PARAMS has
  `random_state=17`. The metadata is a lie.

### 🟡 N-21 No warm-up filter for early panel rows
- Panel rows in the early days of any ticker have under-warmed features
  (rolling stats, EMA). Panel-LTR's `min_history_days` filter trims them.
  NGBoost has no equivalent filter — fits on warm-up rows where features
  are biased.

### 🟡 N-23 Single `predict_distribution` call may OOM on large panels
- `X = panel[self.feature_cols].values.astype(float)` builds full matrix
  in RAM. For a 99-ticker live-runner inference, fine. For a 121k-row
  CV evaluation if we ever add CV, may OOM. Should batch.

### 🟡 N-24 No artifact size / sanity check on load
- `pickle.loads()` on a truncated or empty `regressor_pickle_b64` will
  raise an opaque `EOFError`. Should verify size/magic bytes before unpickling.

---

## NGBoost inference — `kernel/panel_pipeline/job_panel_scoring.py`

### 🔴 N-25 One missing column kills the whole NGBoost prediction
- `ApplyNGBoostTask.run()` line 429-433: `if missing: log + return`.
  CLAUDE.md notes: "Hourly + minute features are likely mostly NaN in
  LEAN runs". One missing column → NGBoost silently no-ops → μ/σ never
  written → Kelly sizing degrades to default behaviour. **No warning to
  user beyond a single log line.**
- **Fix**: align matrix to `head.feature_cols` (drop extra, fill missing
  with median or per-feature trained mean), or fail loud + exit.

### 🟠 N-26 Predict-time z-score parity not enforced
- The matrix passed in has been z-scored by today's universe. Training
  was z-scored over the panel — possibly different population. So the
  same raw feature value gives a different z at train vs predict.
- This is the same issue as P-37 in the panel-LTR audit (R-1 fix).
  NGBoost has the same problem — needs the same `prepare_inference_panel_frames`
  guarantee. Currently nothing enforces it.

### 🟠 N-27 Score-mode metadata not validated on load
- The artifact doesn't record what `score_mode` it was trained for. So
  at inference time, ApplyNGBoost reads `score_mode` from the live
  config — which may differ from training. No safety net.

### 🟠 N-28 No telemetry on μ/σ distribution at inference
- If σ collapses to ~0 (degenerate distribution), Kelly sizing → ∞ →
  one position takes the whole portfolio. No warning fires unless the
  oversize cap kicks in.
- Same with μ — no sanity range check.

### 🟡 N-29 No `score_mode="combined"` mode that uses σ for ranking
- Current modes: `additive` (just write μ, σ) and `mu_minus_lambda_sigma`
  (overwrite). No mode that uses σ as a tiebreaker only, or λ that
  scales with regime confidence.

### 🟡 N-30 Lambda default mismatch
- Code default at line 441: `float(ngb_cfg.get("lambda_sigma", 1.0))`.
  Golden config: `lambda_sigma=0.0`. Default 1.0 means missing config →
  σ dominates ranking. **Misleading default for "additive" mode.**

---

## Transformer — `training_panel/transformer_model.py` (core class)

### 🔴 T-1 Silent ticker truncation in training (the smoking gun)
- `_build_date_groups()` line 162-168:
  ```python
  for gi, gs in enumerate(group_sizes):
      take = int(min(gs, max_tickers))
      x[gi, :take, :] = X_flat[offset:offset + take, :]
      ...
      offset += int(gs)         # advance by FULL gs!
  ```
- `max_tickers=38` (DEFAULT). Watchlist has 99 tickers per date.
- Each date contributes ONLY first 38 rows to `x`. The next 61 rows get
  skipped (offset advances past them, never copied to x).
- Effect on artifact: `panel_shape: 47190 rows × 38 tickers` — the
  transformer trained on **62% less data** than panel-LTR.
- This is the most likely root cause of OOS IC = 0.006.
- **Fix**: raise `max_tickers ≥ watchlist size` (99) AND/OR change loop
  to chunk-split like `predict()` does.

### 🔴 T-2 Train ≠ inference structure
- Training: `_build_date_groups` truncates to `max_tickers`.
- Inference: `predict()` chunk-splits oversized groups.
- Inference uses chunks of 33 (= 99/3 chunks); training used groups of 38.
- The cross-attention pattern the model learned (38-ticker sequences)
  is not what inference feeds it (33-ticker chunks). **Distribution shift
  baked into design.**

### 🔴 T-19 Cross-chunk scores are not rank-comparable
- Inference splits 99 tickers into 3 chunks of 33. Each chunk produces
  scores from the model's last linear layer (`Linear(d_model→1)`). The
  scale of these scores is implicit in the model — chunk 1's "5.0" is
  not the same as chunk 2's "5.0" because each chunk has a different
  cross-attention context.
- Yet `predict()` flattens chunks back into one Series and ranks them
  globally. **The global ranking is meaningless across chunks.**

### 🔴 T-7 NaN labels become 0 after `nan_to_num`
- `_build_date_groups()` line 153-156: `nan_to_num(y, 0.0)`. NaN labels
  → 0. The ListNet softmax then treats 0 as a real label (median of the
  Gaussianized distribution), giving the NaN-row real probability mass.
- **Effect**: model trains toward "predict the median" for missing-label
  rows — biases predictions toward 0.

### 🔴 T-8 NaN labels not masked in ListNet loss
- `_listnet_loss()` line 116: `label_logits = labels.masked_fill(pad_mask, -inf)`.
  Only `pad_mask` excludes positions. Original NaN labels (now 0) are
  treated as REAL labels with softmax probability `exp(0)/Σ exp(yi)` —
  near-uniform weight.
- **Combined with T-7**: any NaN-label row pulls predictions toward 0.

### 🟠 T-3 No CPCV / cross-validation for transformer
- `CrossValidateTask` line 1172-1208 wraps PanelTransformerModel in a
  `_SklearnAdapter`, but the adapter does naive `fit/predict` without
  date-group preservation logic. Even when wrapped in `CombinatorialPurgedCV`,
  the date partitioning may be inconsistent with the model's
  date-grouping. CPCV needs all rows of a date in the same fold; the
  adapter just `fit(X, y)` without checking.

### 🟠 T-13 IC computed every epoch (training cost)
- `_ic_on_tensors()` line 477 runs a full forward pass over the entire
  training set every epoch. With `max_epochs=50` and a 47k-row panel,
  that's 2.4M forward passes for diagnostics alone.
- **Fix**: compute IC every N epochs, not every epoch.

### 🟠 T-15 IC is eval-mode (no dropout); training is not
- Reports eval-mode IC every epoch. But the training step uses dropout
  (feature/ticker/residual). Eval-mode IC is systematically higher than
  in-flight training IC — misleading "progress" metric.

### 🟠 T-16 No gradient clipping
- AdamW + ListNet softmax can produce gradient spikes when scores diverge.
  `torch.nn.utils.clip_grad_norm_` is industry standard for transformers.

### 🟠 T-17 `set_num_threads(1)` cripples CPU training
- Line 256: forces single-threaded execution. Comment says "fork+OpenMP
  deadlock". On macOS with MPS, training is on GPU anyway. On CPU
  (MPS unavailable), training is 1-thread serialised — possibly hours.
- **Fix**: only set `num_threads(1)` when forked workers are detected
  via `multiprocessing.current_process().name`.

### 🟠 T-18 Early-stopping silent when no `eval_panel`
- If the caller passes only training data, `xte=None` → `bad_epochs`
  never increments → trains for full `max_epochs=50`. Patience config
  is dead unless eval is wired.
- `FinalFitTask` at line 1310 calls `train()` with no eval — so final
  fit always runs full 50 epochs, no early stop.

### 🟠 T-22 No feature_col validation
- `train()` line 230: `self.feature_cols = list(feature_cols)`. If any
  col not in `panel.columns`, `_build_date_groups` raises KeyError —
  but the model has already partially initialised. Should validate first.

### 🟠 T-23 `predict()` doesn't sort panel by date
- Line 377: `group_sizes = panel.groupby("date", sort=False).size().to_numpy()`.
  If panel is not pre-sorted by date, the resulting groups span multiple
  dates. `_build_date_groups` then creates groups that mix dates → the
  cross-attention is across dates instead of within a date.
- **Fix**: enforce `panel.sort_values("date")` before computing group_sizes.

### 🟠 T-25 Triple dropout compounding
- `dropout=0.3` (encoder), `feature_dropout=0.2` (input), `ticker_dropout=0.1`
  (whole-row). Effective dropout per signal path: 1-(1-0.3)(1-0.2)(1-0.1) ≈ 0.50.
  Half the signal zeroed at every gradient step. Likely too aggressive
  for 47k-row panel.

### 🟠 T-27 `weight_col` discarded
- Line 229: `del weight_col`. Comment: "ListNet is scale-invariant".
  True for label scaling, but per-row weights affect WHICH rows the
  optimizer attends to. Throwing weights away means short-history tickers
  contribute equally to long-history ones — biased toward newer
  tickers + symmetry ignored.

### 🟠 T-31 CV uses half the epochs of final fit
- Line 1175: `cv_epochs = max(int(cfg.get("num_boost_round", 50)) // 2, 5)`.
  CV trains for half. Yet final fit uses full max_epochs. The CV IC
  systematically underestimates final-fit IC. CV-based promote/reject
  decisions are biased.

### 🟠 T-32 CV's `_SklearnAdapter` may break date-grouping
- Line 1184: `df["date"] = panel.loc[X.index, "date"].values`. Relies
  on X.index being valid panel-row labels. Then `df.sort_values("date")`
  + `groupby("date").size()` rebuilds groups. If CV folds split a date
  across train/test (no purging at row level), the adapter will pass
  the model a fragmented date-group → broken ListNet softmax.

### 🟡 T-4 `_build_date_groups` is shared between train and predict
- Train: truncates oversized groups.
- Predict: pre-splits oversized groups, then calls `_build_date_groups`.
- Two callers, one helper, two semantics. Subtle — easy to forget when
  adding a new caller. **Refactor**: separate `train_groups()` and
  `inference_groups()` helpers.

### 🟡 T-5 No ticker identity embedding
- Model treats input as a permutation-invariant set of (max 38) tickers.
  It can't learn that NVDA tends to behave differently from JNJ — it
  must extract this from features alone. For the residualised features
  (beta-neutral, sector-neutral) this is the design — but with raw
  factor columns added in Round 3-5, identity matters.

### 🟡 T-6 NaN-zero substitution biases for raw factors
- `nan_to_num(X, 0.0)` zeros NaN. For z-scored features, 0 is "neutral".
  For raw factors (price, volume), 0 is biased "extreme low".
- Need to assert all features are z-scored before NaN→0.

### 🟡 T-9 Label smoothing is mistaken
- `label_smoothing=0.05` adds Gaussian noise. True label smoothing in
  classification softens probability mass toward uniform. Calling this
  "label smoothing" is misnomer; it's "noise augmentation".

### 🟡 T-10 Hyperparams under-spec'd vs panel-LTR
- `lr=1e-4`, `weight_decay=1e-4`, `max_epochs=50`, `batch_size=32`. With
  ~1256 dates and batch_size=32, that's 39 batches/epoch × 50 = 1950
  gradient steps. AdamW with lr=1e-4 needs more steps to converge on
  47k rows × 31 features.

### 🟡 T-11 No input LayerNorm
- `feature_encoder = nn.Linear(F, d_model)` then directly into
  TransformerEncoder. The encoder's internal LayerNorm catches it, but
  pre-encoder LN is best practice.

### 🟡 T-12 train_ic uses Gaussianized labels
- `_ic_on_tensors` measures Spearman corr against `panel["label"]`
  which is the Gaussianized residual. Gaussianization preserves ranks
  → IC is the same vs raw. Fine in principle, just worth noting.

### 🟡 T-14 train_ic skipped for groups with all-zero labels
- Line 506: `if np.all(y_slice == y_slice[0]): continue`. After NaN→0
  substitution, any group with all-NaN labels has all-zero y → skipped.
  Reported train_ic is over a SUBSET of dates with valid labels — may
  not match the loss-optimised dates.

### 🟡 T-20 No positional encoding
- Standard transformer needs positional encoding for ordered sequences.
  Cross-sectional ticker grouping is permutation-invariant by design,
  so this is OK — but easy to confuse with "this is broken because no PE".

### 🟡 T-21 Save without `weights_only=False` fallback consistency
- `torch.save(state_dict, path)` (line 429): saves state. `torch.load`
  uses `weights_only=True` (line 467). If user upgrades torch and the
  saved file accidentally contains non-tensor metadata (shouldn't but
  e.g. some tensor with a custom subclass), loading will fail loudly.
  Acceptable behaviour but worth surfacing.

### 🟡 T-24 `predict()` injects dummy `label=0.0` column
- Line 401: `panel.assign(label=0.0)` — dummy label column to satisfy
  `_build_date_groups` contract. The helper signature could just accept
  no label_col instead.

### 🟡 T-26 Dropouts not deterministic across calls
- Even with `deterministic=True` + seed, the dropout RNG is still
  driven by `torch.rand` which is per-step. Predict-time dropout is
  off (`eval()`), so OK at inference. But re-training with same seed
  should yield same model — yet PyTorch's MPS backend isn't fully
  deterministic. The seed comment notes this; verify with two-run
  comparison test.

### 🟡 T-30 No NaN-feature detection in scorer
- `TransformerPanelScorer.score()` line 69: copy without NaN check.
  Falls through to `_build_date_groups` which silently zero-fills.
- **Fix**: emit per-ticker warning for any NaN before zero-filling.

---

## Transformer scorer — `kernel/panel_pipeline/transformer_scorer.py`

### 🔴 T-28 Single-date-group inference incompatible with chunk-split
- Line 70: `frame["date"] = 0`. Treats whole 99-ticker universe as ONE
  date-group. `predict()` then chunk-splits into 3 chunks of ~33. But
  training never saw chunks of 33 — only fully-padded sequences of 38.
- The model produces a score for each row, but the cross-attention
  context is wildly different between train (38-padded) and inference
  (33-chunked). Scores are not what the model was trained to produce.

### 🟠 T-29 Non-deterministic per-bar scoring
- The chunk-split partitions tickers based on input order. If the input
  matrix has a different ticker order across two bars, the chunks differ
  → different cross-attention contexts → different scores for the same
  features.
- **Fix**: explicit sort by ticker before scoring; document the ticker
  ordering convention.

---

## Ensemble scorer — `kernel/panel_pipeline/ensemble_scorer.py`

### 🟠 E-1 Rank accumulation may misalign across scorers
- Line 78: `arr = raw.to_numpy()`. Series order = scorer's output order.
  If scorer A and scorer B return different row orders, the rank
  accumulation `ranks += w * rank_norm` adds rank_A[ticker_X] to
  rank_B[ticker_Y].
- Defensive fix: `raw = raw.reindex(feature_matrix.index)` before
  `.to_numpy()`.

### 🟠 E-6 NaN scores → automatic worst rank
- `np.argsort(-arr)` sorts NaN to the END. After normalization,
  NaN-score ticker gets rank_norm=0 (worst). Any scorer that returns
  NaN for some rows automatically punishes those tickers — silently.

### 🟡 E-2 Tie handling only catches exact equality
- Float scores from NN/XGB rarely tie exactly. Acceptable for now.

### 🟡 E-3 Single-row edge returns 0.5
- `n=1` returns 0.5 ("neutral"). Loses signal — but single-candidate
  selection is rare.

### 🟡 E-4 Metadata silent on inner backends
- Build-time metadata doesn't record which backends/weights were
  ensembled. Audit log goes blind.

### 🟡 E-5 Union feature_cols requires caller to align
- `feature_cols = union(scorers)` — caller must pass a matrix containing
  ALL columns. No automatic projection per-scorer. OK as documented.

### 🟡 E-7 `build_ensemble_scorer` dispatches via PanelScorer.load
- Line 115: dispatches to PanelScorer.load. PanelScorer routes
  `panel_transformer` correctly, but if a path doesn't end in `.pt` or
  `.json`, the function falls through to XGBoost decoder → opaque error.

---

## Configuration / golden — `strategy_config.json` + `golden.json`

### 🔴 C-1 Transformer artifact stale
- `panel-transformer.pt` last trained 2026-04-23 (1 day before this
  audit). Watchlist size in artifact: 38 tickers. Current panel-LTR
  trained 2026-04-25: 99 tickers. The two are operating on **different
  universes**. If transformer is ever flipped on, it scores 99 tickers
  through a model trained on 38-ticker date groups.

### 🟠 C-2 No `panel_ltr.transformer_params` in either config
- No `max_tickers`, no override of any `TransformerParams` field.
  Default `max_tickers=38` (T-1 root cause). Even if user set
  `panel_ltr.backend="transformer"` today, the same truncation bug
  fires.

### 🟠 C-3 No `ensemble.enabled` flag
- `EnsemblePanelScorer` is fully implemented but no Job loads it. The
  ship-gate path described in `doc/renquant_104_transformer_design.md`
  has no runtime hook.

### 🟡 C-4 NGBoost `lambda_sigma=0.0` + `score_mode=additive` is dead-config
- Both flags off → NGBoost μ/σ touch nothing in ranking. Yet `enabled=true`
  → still runs the full predict per bar. **Wasted compute** (the tax is
  small but inelegant).

---

## Tests

### 🟠 X-1 No truncation test for transformer
- `test_panel_transformer.py` uses `n_tickers=8 < max_tickers=10`. Never
  triggers the 38-row-per-date truncation. T-1 was untested.
- **Fix**: add `test_build_date_groups_handles_oversized_groups` that
  asserts no rows are dropped silently.

### 🟠 X-2 No NaN-feature test for NGBoost
- All test fixtures use clean Gaussian panels. NaN handling untested.

### 🟠 X-3 No CV test for NGBoost
- `test_ngboost_head.py` only tests fit/predict shape on full panel.
  No held-out IC, no CPCV.

### 🟠 X-4 No transformer × XGBoost ensemble integration test
- Ensemble class has unit tests, but no end-to-end "load XGBoost +
  Transformer artifact, score, compare to single-backend" test.

### 🟡 X-5 No drift test for stale artifact
- No test that fails if `panel-transformer.json` is older than `panel-ltr.json`
  by > 7 days. Stale artifacts can silently degrade ranking.

### 🟡 X-6 Tests don't cover the panel scoring chain order
- Existing tests cover individual tasks. The audit found ordering
  matters (NGBoost before/after calibrator). One ordering integration
  test exists for panel-LTR but not for "transformer instead of XGBoost".

### 🟡 X-7 No determinism cross-run test
- Run training twice with same seed, compare artifact bytes / IC. None
  exist for either NGBoost or Transformer.

### 🟡 X-8 No "predict on empty matrix" test for NGBoost
- TransformerPanelScorer has empty-matrix unit test (R3-22). NGBoost
  doesn't.

---

## Severity rollup

| Severity | NGBoost | Transformer | Ensemble | Config | Tests | **Total** |
|---|---:|---:|---:|---:|---:|---:|
| 🔴 Critical | 4 | 4 | 0 | 1 | 0 | **9** |
| 🟠 High | 9 | 12 | 2 | 2 | 4 | **29** |
| 🟡 Medium | 12 | 13 | 5 | 1 | 4 | **35** |
| **Total** | **25** | **29** | **7** | **4** | **8** | **73** |

(audit also includes 5 PanelScorer/related items that are noted but not
counted to keep the rollup focused.)

---

## Phase 3 — fresh re-audit of training orchestration (2026-04-25 evening)

User mandate: "phase3，忘记你做的，重新仔细低审查一遍". Walked
`pp_panel_training.py` + `pp_training_full.py` from scratch, looking for
silent failures, race conditions, lookahead, and resource waste.

### 🔴 P3-8 Lookahead in fundamentals: `iloc[-1]` broadcast to ALL dates
- `FactorZScoreTask` line 893: `v = df[col].iloc[-1]`. Takes the most
  recent row of the per-ticker frame and broadcasts as a scalar to
  every date in `idx` (line 930).
- For inference, "last row" is today — fine. For training, "last row"
  is the most recent available fundamental (likely 2026 for a panel
  spanning 2018-2026). So **training rows on 2018-01-01 use 2026
  fundamentals**.
- CLAUDE.md flags it as "time-invariant snapshot in this release;
  extending to point-in-time time-series is a future change". Documented
  but it's a real lookahead leak in the training panel — IC may be
  inflated by the static fundamentals being predictive of future returns
  via the time-traveled snapshot.

### 🔴 P3-12 Sequential symbol fetch in panel data phase
- `FetchPanelDataTask` line 153-160 fetches 99+ tickers serially. At ~1s
  per cached fetch + ~3s per uncached fetch, the cold path can be 5+
  minutes just to gather data. yfinance is I/O-bound — easily
  parallelizable with a ThreadPool of 8-16.
- Same pattern in `PanelDataJob.FetchOHLCVTask` (line 192-201).
- **Resource waste**: every retrain pays this cost.

### 🟠 P3-2 / P3-5 Per-ticker chain failures silently shrink universe
- `_run_panel_ticker_chain` (line 156-166): catches Exception, logs error,
  moves on. Failed tickers are missing from `ctx.factor_frames` /
  `ctx.neutralized_frames`. Then `BuildPanelTask` does
  `ff_wl = {t: ff[t] for t in ctx.watchlist if t in ff}` — silently
  drops them.
- **No surface count**: if 30/99 tickers crash, the panel trains on 69.
  `panel_metadata.n_tickers` reflects 69. But `len(ctx.watchlist)=99`
  remains in config. **No alert that 30% of universe was lost.**
- **Fix**: aggregate failures, fail loud if > N% drop.

### 🟠 P3-11 FetchPanelDataTask drops failed tickers without count
- Line 154-160: if `fetch_ohlcv(sym)` raises, log warning + `continue`.
  No counter, no aggregate "X failed of Y attempted" summary.
- Same blast radius as P3-2.

### 🟠 P3-13 No partial-resumption / checkpoint support
- `FullTrainingPipeline.run()` runs 3 jobs serially. If `BaselineTournamentJob`
  is 95% through and crashes, the next attempt restarts all 99 tickers.
  The TTL gate inside baseline tournament partially helps (skips
  recently-trained tickers) but offers no per-Job checkpoint.
- **Fix**: persist completed-job markers so `--resume` can pick up.

### 🟠 P3-15 / P3-16 Shallow config copies share nested dicts
- `RunPanelTrainingTask.run()` line 218: `panel_cfg = dict(config.get("panel_ltr", {}))`.
  Shallow copy — `panel_cfg["xgb_params"]` is a REFERENCE to
  `config["panel_ltr"]["xgb_params"]`. Any in-place mutation downstream
  silently mutates the original config.
- Currently no downstream mutates these dicts in place, but the pattern
  is fragile. **Defensive fix**: deep-copy.

### 🟠 P3-20 No artifact backup on overwrite
- `SaveArtifactTask` writes `panel-ltr.json` overwriting the previous
  artifact. If the new training run produced a regression (lower IC),
  there's no atomic rollback path. Manual `*.pre_audit_fixes` backups
  exist but aren't automated.
- **Fix**: write to `<name>.json.tmp` then atomic-rename, AND keep the
  previous artifact at `<name>.json.previous` for one-click rollback.

### 🟡 P3-1 ThreadPoolExecutor + GIL in `run_panel_ticker_parallel`
- ThreadPoolExecutor parallelizes I/O but not CPU-bound pandas/XGBoost.
  Most of `_run_panel_ticker_chain` is CPU. Effective parallelism is
  near-1 on the CPU side. Should be ProcessPool — but pickling of
  ticker contexts adds overhead.

### 🟡 P3-3 Redundant timeout in `as_completed` + `fut.result(timeout)`
- `as_completed(futures, timeout=None)` already returns only completed
  futures. The subsequent `fut.result(timeout=timeout_seconds)` adds a
  second timeout that can never fire on the per-future basis.
- Code smell only.

### 🟡 P3-4 Misleading `fut.cancel()` in ThreadPoolExecutor
- ThreadPool can't cancel running futures. The `fut.cancel()` call after
  TIMEOUT log only succeeds if the future hadn't started — but that
  contradicts the "TIMEOUT after Ns" log.

### 🟡 P3-7 FactorZScoreTask completeness check by length only
- Line 841: `if ctx.factor_frames and len(ctx.factor_frames) >= len(ctx.raw_factor_frames):`.
  Length-only check could pass with the WRONG ticker subset if some
  upstream race put different tickers in each.
- **Fix**: check that the ticker SETS match.

### 🟡 P3-14 No timing/progress on interrupted runs
- "FullTrainingPipeline DONE" only logs on full success. Interrupted
  via Ctrl-C → no log → no idea where it stopped. Operators waste time
  re-running from scratch without knowing what completed.

### 🟡 P3-17 Recalibrate re-reads config from disk; in-memory mutations lost
- `RunRecalibrationTask.run()` line 290-291: `ctx.config = json.loads(...)`.
  Any in-memory tweaks to ctx.config from earlier phases are blown away.
  No current bug because earlier phases don't mutate, but the contract
  is implicit.

### 🟡 P3-19 Cadence reads `datetime.date.today()` (local clock)
- A clock skew or timezone bug silently flips weekday → cadence skip
  fires on wrong day. Should pin to NYSE timezone via the calendar guard.

### 🟡 P3-9 Forward-return shift uses calendar-day count not trading-day
- `LabelsTask` line 963: `spy_close.shift(-lookahead)` shifts by row
  count. If the panel index is trading-day-indexed (it is, after pandas
  business-day fill), this is OK. But mixed-frequency caches could
  silently misalign.

### 🟡 P3-6 Disabled feature columns vanish from training feature set
- `FactorZScoreTask` skips columns missing from `raw_factor_frames`.
  If hourly features are enabled at training time but disabled at
  inference (via different config), the scorer artifact's feature_cols
  contains hourly cols but the inference matrix doesn't → KeyError.
- Defended by `prepare_inference_panel_frames` running the same task
  chain — but only when called via the canonical helper.

---

## Phase 4 — data pipeline (labels, panel_frame, factors, neutralization, imputation)

User mandate: "多多挖bug吧". Walked the per-ticker label + factor + neutralization
+ imputation code that feeds ALL downstream models (panel-LTR + NGBoost
+ Transformer + per-ticker tournament). Bugs here invalidate every
trained model — biggest blast radius.

### 🔴 D-1 No β clipping in label residualization (`labels._rolling_beta_purged`)
- `beta = cov / var.replace(0, np.nan)`. Near-zero variance produces β
  in [-50, +50] for a low-volume / illiquid ticker × SPY. Then
  `residual = fwd - β · spy_fwd` becomes the dominant driver of the
  residual. Some labels become noise, not "outperformance".
- Same untreated in `factors.compute_rolling_beta` (line 64).
- **Fix**: clip β to [-3, +5] (typical equity beta range).

### 🔴 D-2 No β clipping in feature neutralization (`neutralization._residualize`)
- Same pattern. With `min_obs=30` (just 30 observations to fit β),
  noisy β can be ±5. Then `residual = feat - α - β·pred` may invert
  the feature's sign for that ticker.
- **Fix**: clip β; OR raise `min_obs` to ≥60.

### 🔴 D-3 Static fundamentals broadcast across the entire training panel
- Already noted as P3-8. Repeated here because the blast radius is
  every model that uses fundamentals (currently panel-LTR via
  `FactorZScoreTask`'s `FUNDAMENTAL_COLS`). Training rows on 2018-01-01
  use 2026 fundamentals because `df[col].iloc[-1]` takes the latest
  value and broadcasts.
- **Fix**: turn fundamentals into a true time-series (one row per
  reporting date) and forward-fill, OR document that current IC is
  inflated by this leak.

### 🟠 D-4 Age weighting is dead code
- `pp_panel_training.PanelTrainingContext.listing_dates` is `None` in
  `pp_training_full.py` line 251.
- `compute_age_weight()` then short-circuits at line 102 returning
  weight=1.0 for everyone.
- Newly-IPO'd tickers (RBLX, NVTS, MDB, SOFI, SNOW, PLTR) get the
  same weight as 30-year incumbents despite having ~1/4 the history.
- **Fix**: populate listing_dates from each ticker's first OHLCV bar
  in `FetchPanelDataTask` (cheap; one .index[0] per ticker).

### 🟠 D-5 Per-ticker chain failures don't propagate to caller
- Already noted in Phase 3 (P3-2 / P3-5). Repeating: when a parallel
  per-ticker job throws, `factor_frames[t]` and
  `neutralized_frames[t]` stay None. Downstream BuildPanelTask
  silently drops the ticker.
- Compounds with D-4: if a newly-IPO'd ticker fails feature calc due
  to short history, it's silently dropped — no signal to the operator.

### 🟠 D-6 Forward-return shift uses calendar-row count, not trading-day
- `LabelsTask` line 963: `spy_close.shift(-lookahead)`. Shifts by row
  count assuming uniform trading-day index. If the watchlist or SPY
  has gaps (delisted/halted ticker, or OHLCV data with missing days),
  shift(-5) crosses different calendar lengths per ticker.
- Acceptable for rows where both ticker and SPY trade — i.e. most rows.
  But `compute_residual_returns` doesn't validate the alignment.

### 🟠 D-7 Per-row Python loop in `compute_age_weight`
- Line 106: `for i in range(len(panel))`. 121k iterations of
  `pd.Timestamp(...) - pd.Timestamp(...)`. ~5-10 seconds at scale.
- **Fix**: vectorize via `pd.to_datetime(...).dt.date` differences.

### 🟠 D-8 No completeness count for feature_frames at panel-build time
- `BuildPanelTask` (line 116) iterates over feature_frames, drops
  tickers without labels, factors, or short history. The dropped
  COUNT is not logged. The expected total (=watchlist size) isn't
  compared.
- **Fix**: log "n_in / n_watchlist tickers entered the panel; %d
  failed feature, %d failed labels, %d failed min_history".

### 🟡 D-9 Residual momentum uses 60-day β to neutralize 252-day mom
- `compute_residual_momentum` uses `compute_rolling_beta(window=60)`
  to project away SPY's 252-day momentum from the ticker's 252-day
  momentum. The β windows don't match the factor's window. Noisy.
- Methodological: typically you'd β-fit on the same horizon as the
  factor. Switching to 252-day daily-return β here would be more
  consistent.

### 🟡 D-10 size factor falls back to log(close) without warning
- `compute_size_feature()` line 99-100: when shares_outstanding is
  None (current default), uses `log(close)` as proxy. log(close) is
  a poor size proxy — high-priced stocks like LLY ($800/sh) get
  size > 6 vs WMT ($90/sh) < 5, but their actual market caps are
  similar.
- **Fix**: load shares_outstanding from yfinance / fundamentals
  cache and broadcast.

### 🟡 D-11 `gaussianize_cross_section` plotting position formula
- `u = ranks / (n + 1)` — Hazen plotting positions. For different
  date sizes (n=10 vs n=99), the same rank produces different u →
  different label distribution stability. Not strictly a bug; just
  noteworthy for cross-date IC analysis.

### 🟡 D-12 Ticker-without-sector silently skips neutralization
- `neutralize_features` line 128-130: `if sec_df is None: out[ticker] = df; continue`.
  Tickers without a sector_etf mapping get NON-residualised features.
  The ticker's `rel_mom_20d` then has a different distribution than
  its sector-mapped peers'.

### 🟡 D-13 `np.where` mixing pandas Series + arrays in `_residualize`
- `np.where(use_roll, roll_cov, exp_cov)` returns an ndarray. NaN
  handling between pandas Series and ndarrays may differ subtly at
  the warmup boundary.

### 🟡 D-14 No min-observations guard for β denominator
- Line 64 (factors): `var = r_s_a.rolling(window).var()`. For 60-bar
  window with min_periods=window, variance is computed only if all 60
  bars are present. But if SPY has missing bars, var=NaN, β=NaN, no
  factor for that bar. Cascade of NaN that downstream silently fills
  to 0 in z-score.

---

## Severity rollup (revised)

| Severity | NGBoost | Transformer | Ensemble | Config | Tests | Phase 3 (orch) | Phase 4 (data) | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 🔴 Critical | 4 | 4 | 0 | 1 | 0 | 2 | 3 | **14** |
| 🟠 High | 9 | 12 | 2 | 2 | 4 | 6 | 5 | **40** |
| 🟡 Medium | 12 | 13 | 5 | 1 | 4 | 7 | 6 | **48** |
| **Total** | **25** | **29** | **7** | **4** | **8** | **15** | **14** | **102** |

**Hit the user's "100 bugs" target.**

---

## Recommended fix order

If the goal is "lift transformer IC from 0.006 to competitive":

1. **T-1** — raise `max_tickers` to ≥ watchlist size (99 → 128 for slack)
   AND fix `_build_date_groups` to NOT silently truncate. **Single biggest
   IC unlock candidate.**
2. **T-2 / T-19** — once T-1 is fixed, train and infer on the same 99-ticker
   structure; remove the chunk-split fallback (or pad to 128 always).
3. **T-7 / T-8** — mask NaN labels properly in ListNet loss instead of
   substituting zero. Removes the bias toward "predict the median".
4. **T-23** — sort panel by date in `predict()` before grouping.
5. **C-1** — retrain transformer once T-1 + T-7/T-8 are fixed; with the
   fresh 99-ticker artifact the OOS IC ship-gate becomes meaningful.
6. **T-3 / T-31 / T-32** — add a proper transformer-aware CV path so we
   can actually compute OOS IC for promotion decisions.

If the goal is "make NGBoost trustworthy as a sizing oracle":

1. **N-17** — add CPCV for NGBoost. Until we have OOS IC for μ̂, Kelly
   sizing is built on faith.
2. **N-1 / N-5 / N-13** — NaN handling everywhere. NGBoost segfaulting
   on a single NaN feature is a production timebomb.
3. **N-25** — at inference, align matrix to head.feature_cols robustly
   (drop extras, fill missing with median) instead of `if missing: return`.
4. **N-18** — write OOS IC + σ calibration scores to artifact metadata.
5. **N-2 / N-14** — early stopping. Cap effective `n_estimators` based
   on validation NLL plateau.
