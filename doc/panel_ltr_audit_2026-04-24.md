# Panel-LTR Deep Audit — Notebook + LEAN + Live (2026-04-24)

User mandate: "panel ltr 是我的心腹大患"。Walked the panel-LTR pipeline
end-to-end on three surfaces — notebook (cell 6 train + cell 15 sim),
LEAN backtest (`main.py` + `LeanAdapter`), and live (`live/runner.py` +
`RunnerAdapter`). Found **37 issues**.

Severity:
🔴 critical — wrong numbers, lookahead, or breaks LEAN/live
🟠 high — significant correctness or performance bug
🟡 medium — silent fragility, edge case, or reproducibility leak

---

## Architecture-level (worst — these break everything)

### 🔴 P-37 Two `data/` cache dirs with inconsistent contents
- `/<repo>/data/fundamentals/` → 102 tickers, AAPL md5 `23ea4608…`
- `/<repo>/backtesting/renquant_104/data/fundamentals/` → 43 tickers, AAPL md5 `4e2eee2c…`
- Same for `earnings_surprise/` (98 vs 34) and likely `insider_trades/`, `intraday/`.
- The cache-resolver uses **relative path** `data/fundamentals` →
  resolved against current working directory.
  - Notebook CWD: depends on where Jupyter started (Notebooks/ vs strategy/)
  - LEAN Docker CWD: strategy_dir
  - Live runner CWD: repo root (where `python -m live.runner` is invoked)
- **Result: notebook / LEAN / live each see different data.** Predictions
  are non-reproducible across surfaces.
- **Fix**: resolve via `_strategy_dir` (already used elsewhere) for cache
  paths, or hardcode `repo_root / data` as the single canonical store.
  Delete the duplicate. Add a CI check for path drift.

### 🔴 P-36 LEAN Docker can't access `data/intraday/`, `data/fundamentals/` if mount is wrong
- Tasks `LoadHourlyBarsTask`, `LoadMinuteBarsTask`, `LoadFundamentalsTask`,
  etc. all read parquet caches from a relative path. Docker only sees
  what's mounted in `lean.json`'s container config.
- If the mount is `backtesting/renquant_104/` only, the strategy-dir
  variants of these caches (which have FEWER tickers — see #P-37) are
  what LEAN actually loads. Hourly + minute features are likely mostly
  NaN in LEAN runs.
- **Effect**: LEAN backtest's panel scoring uses ~5-10 NaN columns
  while training uses populated columns → distribution shift.

### 🟠 P-1 SimAdapter ≠ LEAN ≠ live (architectural asymmetry)
- LEAN: `LeanAdapter.make_context` calls `prepare_inference_panel_frames`
  **per bar** (rebuilds full feature pipeline every bar)
- Live: `RunnerAdapter.make_context` same — **per bar**, but daily so OK
- Sim: `SimAdapter` does **NOT** call `prepare_inference_panel_frames` —
  caller (cell 15 / `run_backtest`) must supply pre-built frames
- Notebook (pre-fix): wrong manual chain → no panel scoring
- Notebook (post-fix): now calls `prepare_inference_panel_frames` once
- **Asymmetry means each path can silently diverge**

### 🟠 P-3 LEAN rebuilds the entire panel pipeline per bar
- Per `OnData`, `LeanAdapter.make_context` invokes
  `prepare_inference_panel_frames(watchlist, ohlcv, …)`. Internal:
  `SectorMomentumTask`, 5× `Load*Task`, 99× per-ticker
  `Feature/Neutralize/Factor` Jobs in parallel, then 2 cross-sectional
  z-score tasks. Same data computed every bar.
- 700-bar LEAN backtest = 700 reprocessings of identical full-history
  features. Estimated cost: hours per backtest.
- **Fix**: cache panel frames at strategy init (mirror SimAdapter's
  one-build-many-slice pattern), invalidate only when underlying OHLCV
  buffer rolls forward.

### 🟠 P-4 LEAN uses 520-bar lookback for daily but full cache for hourly/minute
- Daily OHLCV: `algo.History(sym, 520, Resolution.Daily)` — 520-bar slice
- Hourly: `HourlyBarStore.load(sym)` — full parquet cache
- Within `prepare_inference_panel_frames`, daily features (mom_12_1,
  realized_vol, drawdown_peak) computed on 520-bar window. Hourly features
  computed on full cache.
- Inconsistent training/inference: training uses full daily history;
  LEAN inference uses 520-bar window. EMA200, 252-day rolling stats are
  edge-affected.

---

## Notebook cell 15 (the cell I just rewrote)

### 🟠 P-Notebook-1 Pre-fix: manually chained tasks, missing 7 of them
- Original cell 15 only ran `SectorMomentumTask` + `LoadFundamentalsTask`
  + per-ticker `NeutralizeJob` + `FactorJob`.
- Missing: `LoadEarningsSurprise`, `LoadInsiderTrades`, `LoadHourlyBars`,
  `LoadMinuteBars`, `NeutralizedFeatureZScoreTask`, `FactorZScoreTask`.
- Result: panel-LTR's expected 41 columns had ~30 NaN / wrong-scale →
  panel score garbage → identical to baseline downstream.
- **Status**: FIXED — cell 15 now calls `prepare_inference_panel_frames`.

---

## prepare_inference_panel_frames internals

### 🟠 P-9 No per-ticker exception isolation in parallel chain
- `prepare_inference_panel_frames` line 110-114:
  ```python
  with ThreadPoolExecutor(max_workers=n_workers, ...) as ex:
      futs = [ex.submit(_chain, tc) for tc in ticker_ctxs]
      for f in as_completed(futs):
          f.result()   # ← raises immediately on any worker error
  ```
- Compare to training's `run_panel_ticker_parallel` which does
  `try: fut.result() except Exception: log.error(...)`.
- One ticker's exception → entire panel inference for the bar fails →
  LEAN's outer `try/except` silently disables panel scoring this bar.

### 🟡 P-32 Re-runs `build_training_features` redundantly
- `prepare_inference_panel_frames` calls `TickerPanelFeatureJob` which
  calls `training/features.build_training_features` for each ticker.
  In the notebook, cell 6 already produced these — wasteful duplicate.
  In LEAN/live, no cache, so unavoidable. Strategy_dir-level cache
  recommended.

### 🟡 P-31 Cell 15 doesn't pass `listing_dates`
- `PanelTrainingContext.listing_dates=None` → `compute_age_weight`
  defaults to 1.0 for all tickers.
- Training side may pass listing_dates if config has one.
- Inference side ignores anyway (only used in label/weight computation,
  not in frame outputs). Functionally no impact, but inconsistent.

---

## TickerPanelFactorJob

### 🟠 P-15 Wrap-everything `try/except Exception` swallows all errors
- Lines 565-673: entire factor computation wrapped in single try.
- Failure → `tc.raw_factor_frame` stays None → ticker silently dropped
  from `ctx.raw_factor_frames` → its z-scores never computed → at
  inference time `build_inference_matrix` skips its factor row →
  feature matrix has NaN factor cols for this ticker → XGBoost
  uses default-direction predictions. **Silent degradation.**

### 🟡 P-13 Hourly/minute index alignment fragile
- Lines 636-643:
  ```python
  h_feats.index = pd.DatetimeIndex(h_feats.index).normalize()
  daily_idx = pd.DatetimeIndex(idx).normalize()
  for col in HOURLY_FEATURE_COLS:
      series = h_feats[col] if col in h_feats.columns else pd.Series(dtype=float)
      aligned = series.reindex(daily_idx)
      aligned.index = idx
      cols[col] = aligned
  ```
- `aligned.index = idx` only correct if `len(aligned) == len(idx)`. If a
  daily date has no matching normalized hourly date, reindex drops NaN —
  but reindex returns same length. Works in current pandas; fragile.
- Tz mismatch between `h_feats.index` (e.g. UTC) and `idx` (naive)
  causes `.reindex(daily_idx)` to silently match nothing.

### 🟡 P-17 Latent fundamentals lookahead via `df[col].iloc[-1]`
- `FactorZScoreTask` line 805: `v = df[col].iloc[-1]   # broadcast scalar`
- Today fundamentals are time-invariant scalars so iloc[-1] = iloc[0].
- **If fundamentals ever become time-series (a planned change), iloc[-1]
  uses the LATEST (future) value to z-score every historical bar.**
  Lookahead bomb waiting to happen.

---

## NeutralizedFeatureZScoreTask + FactorZScoreTask

### 🔴 P-16 FactorZScoreTask early-returns on truthy `ctx.factor_frames`
- Line 753-754:
  ```python
  if ctx.factor_frames:
      return
  ```
- If caller pre-populates `ctx.factor_frames` with a partial dict (or a
  cache from a previous bar), this task **silently skips** and uses
  stale data. Should compare against expected ticker count or reset on
  every call.

### 🟡 P-18 In-place mutation of `ctx.neutralized_frames`
- `NeutralizedFeatureZScoreTask` line 737: `frame[col] = z[t].reindex(frame.index)`.
  Mutates the caller's input frames. Subsequent code that holds a
  reference to "neutralized_frames" gets z-scored ones.

### 🟡 P-19 Single-ticker dates produce `std=0` → z-score = 0
- `cross_sectional_zscore` line 228: `(long["std"] > 0)` else 0.0 fallback.
- For a watchlist date with only 1 ticker (cold start, IPO), all values
  collapse to z=0. Below thresholds → no candidate.

---

## BuildFeatureMatrixTask + build_inference_matrix

### 🟡 P-7 Tz mismatch in `_pick_today_row`
- `idx = pd.to_datetime(df.index); mask = idx <= today`
- If `df.index` is tz-aware and `today` is naive, raises TypeError.
  Sim path's `today` is `pd.Timestamp(date_obj)` (naive); panel feature
  frames built from `fetch_ohlcv` (often tz-aware on yfinance). **Fragile.**

### 🟡 P-30 Column ordering relies on dict-insertion + reindex
- `build_inference_matrix` line 96: `pd.DataFrame.from_dict(rows, orient='index')`.
  Column order is dict-insertion order of the FIRST row. Then
  `out[feature_cols]` reorders to artifact's order.
- If `feature_cols` has a column that's not in any input row, np.nan
  fills. Silent — caller never warned that the model expects a column
  the inference matrix doesn't carry.

### 🟡 P-21 Empty matrix shape
- `if not rows: return pd.DataFrame(columns=feature_cols)`. Empty df
  with feature_cols schema. Downstream `if X.empty` guards correctly.

---

## ApplyScoresTask

### 🔴 P-21 `return False` short-circuits the rest of the chain
- Line 145: `if scorer is None or X is None or X.empty: return False`.
- After this, `VetoWeakBuysTask`, `LoadNGBoostTask`, `ApplyNGBoostTask`,
  `LoadGlobalCalibrationTask`, `ApplyGlobalCalibrationTask`, and
  `ApplyKellySizingTask` **don't run**.
- Specifically `ApplyKellySizingTask` skipping means `kelly_target_pct`
  on candidates and holdings stays stale (from the previous bar).
  Downstream `SizeAndEmitTask`, `TopUpHeldTask`, `RotationJob` use
  stale Kelly numbers.
- **Fix**: return `None` (continue), let downstream tasks no-op
  individually (they all have None guards already).

### 🟠 P-22 NaN panel_score not vetoed
- `VetoWeakBuysTask` line 191: `if ps is not None and ps < floor`.
  NaN < float is False — NaN survives the veto. Downstream
  `SelectionJob` may pick a NaN-rank candidate.

### 🟡 P-23 Veto only applies to candidates, not holdings
- A held ticker whose panel_score drops below `buy_floor` is NOT
  exited. The exit path `PanelConvictionExitTask` uses a different
  threshold (`panel_sell_floor`). Asymmetric — buy and sell decisions
  use different cutoffs without explicit reasoning.

---

## ApplyNGBoostTask

### 🟡 P-15 Missing-cols silently skips the entire NGBoost predict
- Line 396-400: any missing col in `X.columns` → log warning + return.
  After all the work to build NGBoost head, one missing column kills
  μ/σ population. Suggested: fill missing cols with NaN (NGBoost
  handles NaN) or fail loud.

### 🟡 P-37 NGBoost `score_mode = mu_minus_lambda_sigma` + isotonic calibrator
- ApplyNGBoost overwrites `panel_score` and `rank_score` with `μ−λσ`.
- ApplyGlobalCalibration then maps `panel_score` → probability via
  isotonic head. **But the isotonic was fit on raw LTR panel_score**,
  whose distribution is approximately N(0, 1). `μ−λσ` distribution is
  ~N(0.001, 0.02) — entirely inside the fitted training range's center.
- Calibrator output: nearly all candidates → P(out) ≈ 0.5 (the central
  isotonic value). Tier thresholds become uniform → no ranking signal.
- Code comment acknowledges: "not strictly metric-calibrated; acceptable
  for ranking" — but it actively COMPRESSES the σ-aware ranker's range.
- **Fix**: when score_mode is mu_minus_lambda_sigma, skip calibrator
  (use raw μ−λσ as rank), OR fit a separate calibrator on μ−λσ.

---

## ApplyKellySizingTask

### 🟡 P-26 `min_edge=0.0` default is aggressive
- `kelly_target_pct(mu, sigma, …, min_edge=0.0)`. Need μ > 0 strictly.
  μ=0 (no edge) → kelly = 0 → SizeAndEmit skips. Combined with NGBoost
  μ being very small (~ ±0.02), small noise can flip a candidate from
  buyable to filtered.

### 🟡 P-25 Confidence applied once in Kelly, but unconditional in legacy path
- `ApplyKellySizingTask` line 478: `max_pct = max_position_pct * confidence`.
- `SizeAndEmitTask` line 112 (when kelly_on=False): `base_max_pct =
  max_position_pct * confidence` — applied AGAIN.
- When kelly_on=True, kelly_target_pct already has confidence baked in,
  but `SizeAndEmitTask` line 109 ALSO uses `base_max_pct = max_position_pct
  * confidence`. Used in the fallback non-Kelly path. Audit needed for
  potential double-application during the kelly_on edge cases.

---

## Training side bugs

### 🟡 P-Train-1 LabelsTask `spy_returns` parameter naming
- `compute_residual_returns(fwd_returns, spy_fwd, sec_fwd_by_ticker, ...)`.
  Inside `labels.compute_residual_returns(spy_returns: pd.Series, ...)`.
  Parameter named `spy_returns` actually receives forward returns.
  Misleading — caller could pass daily returns and break beta math.

### 🟡 P-Train-2 BuildPanelTask group_sizes computed before merge
- `build_panel_frame` line 153-154 sorts by (date, ticker), then later
  pp_panel_training BuildPanelTask line 956 merges raw_residuals.
  Pandas merge typically preserves left order — fragile invariant.

### 🟡 P-Train-3 Panel CV adapter captures outer `panel` by reference
- `_SklearnAdapter.fit/predict` reads `panel.loc[X.index, 'date']` from
  outer scope. If panel mutates between adapter creation and use, dates
  stale. Today's flow safe; brittle.

### 🟡 P-Train-4 LightGBM/XGBoost adapter `predict()` doesn't attach `date`
- Transformer adapter explicitly sets `df['date']`. lightgbm + xgboost
  adapters omit. Asymmetric — if any backend ever needs date for
  grouping, breaks.

### 🟡 P-Train-5 SaveArtifactTask can KeyError on missing `cv_result`
- Line 1289: `meta = { ... "oos_mean_ic": ctx.cv_result['mean_ic'], ... }`.
  No null guard on `ctx.cv_result`.

---

## Live runner / RunnerAdapter

### 🟠 P-Live-1 `prepare_inference_panel_frames` per bar overhead
- Same as #P-3 but daily — once per `daily_104.sh`. Less critical
  (1 call per day vs LEAN's per-bar) but still wastes ~20-30 seconds
  every cron run.

---

## Severity rollup

| Severity | Count |
|---|---:|
| 🔴 Critical | 5 |
| 🟠 High | 9 |
| 🟡 Medium | 23 |
| **Total** | **37** |

---

## Recommended fix order (by impact / 触手可及度)

1. **P-37** (data/ duplicate dirs) — single source of truth via `_strategy_dir`. 1 hour, prevents most reproducibility issues.
2. **P-21** (ApplyScoresTask `return False` short-circuit) — change to `return None`. 1-line fix. **Immediately unblocks Kelly/calibration when X is empty.**
3. **P-16** (FactorZScoreTask early-return on truthy) — guard on completeness, not truthiness. 1-line fix.
4. **P-22** (NaN panel_score not vetoed) — add `pd.isna(ps)` veto. 1-line fix.
5. **P-9** (no error isolation in parallel inference) — add try/except in `_chain`. 5 lines.
6. **P-23 / P-37** (μ−λσ calibrator wrong) — skip calibrator in `mu_minus_lambda_sigma` mode. 2-line fix.
7. **P-3 / P-Live-1** (rebuild per bar) — cache panel frames at adapter init. ~50 lines.
8. **P-15** (try/except blanket in TickerPanelFactorJob) — finer-grained. ~30 lines.
9. **P-7 / P-21** (tz mismatch) — explicit tz_localize(None) somewhere central. 5 lines.
10. **P-26** (min_edge=0 default) — config knob with sensible default. Minor.
