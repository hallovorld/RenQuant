# renquant_104 — Panel-LTR Cross-Sectional Ranking

**Status**: Active daily strategy. Current golden: **+44.20% APY (after-tax)**, 82% win, 26d max streak over the 27-month OOS sim.
**Author**: Ren Hao
**Last updated**: 2026-04-23
**Based on**: renquant_103 (adaptive regime multi-stock)

**Active feature set**: panel-LTR on 47k rows × **31 features** (16 neutralized per-ticker indicators + 4 factor z-scores + 5 fundamentals + 6 hourly-bar aggregates). Panel training driven by `scripts/train_104.py`; hourly features sourced from `data/intraday/{SYM}/1h.parquet` via `scripts/fetch_hourly_bars.py` (Alpaca IEX). `panel_ltr.hourly.enabled: true` is golden as of 2026-04-23. Shelved experiments (transformer backend, regime-conditional calibration, LightGBM backend) retain their infra behind off-by-default flags — see `doc/improvement_roadmap.md` History.

---

## 1. What's different from renquant_103

renquant_104 inherits the entire renquant_103 decision graph — regime detection, sell
priority, buy gates, sector/wash-sale guards, rotation — and adds a
**cross-sectional panel-LTR ranker** on top of it. Every other node in the
logic graph is unchanged.

| Concern | renquant_103 | renquant_104 |
|---|---|---|
| Per-ticker model | Champion from tournament (Classification / QLearning / XGBoost / Manual) | Same |
| Candidate rank | Per-ticker `rank_score` (calibrated Platt/isotonic) | **Cross-sectional panel-LTR `rank_score`** replaces per-ticker when `panel_scoring.enabled=true` |
| Feature scope | Per-ticker indicators only | Per-ticker indicators + panel-level neutralized factors (sector momentum, size z-score, beta-residuals) |
| Training driver | `Notebooks/renquant_103.ipynb` | **`scripts/train_104.py`** (no notebook — `FullTrainingPipeline` Job/Task chain) |
| History lookback (inference) | 60 daily bars | **520 bars** when panel scoring is enabled (neutralization + factor windows need ≥504) |

Everything else — exits, selection ledger, tiered thresholds, rotation — is
identical. The logic graph in `doc/logic_graph_103.md` continues to apply
after inserting a single node between CandidateScan and Ranking:

```
… → CandidateJob → PanelScoringJob → RankingJob → RotationJob → SelectionJob
                       ↑
                       only runs when ranking.panel_scoring.enabled=true
                       otherwise skipped via should_skip()
```

---

## 2. Panel-LTR design

The panel-LTR model is a single XGBoost learning-to-rank model fitted on the
cross-section of all watchlist tickers per day. Labels are forward
excess-return ranks neutralized by:

- Sector (via sector ETF returns)
- Size (log market-cap proxy via price × volume moving average)
- Beta-residuals vs SPY

The artifact written by training (`artifacts/panel-ltr.json`) contains:

- `booster_raw_json` — serialized XGBoost model
- `feature_cols` — exact column order used at inference
- `oos_mean_ic` — mean information coefficient across CV folds
- `trained_date`

At inference time `PanelScoringJob` performs four atomic tasks:

1. **LoadScorerTask** — deserialize the booster, resolve `artifact_path` against the
   strategy dir if relative. Short-circuits the chain if disabled or missing.
2. **BuildFeatureMatrixTask** — stack today's row from each candidate's
   neutralized feature frame + factor frame into a single matrix keyed by ticker.
3. **ApplyScoresTask** — predict and write `panel_score` onto both candidates **and**
   current holdings (so rotation compares apples-to-apples). Also overwrites each
   candidate's `rank_score` with its `panel_score`.
4. **VetoWeakBuysTask** — drops candidates whose `panel_score` is below
   `ranking.panel_scoring.buy_floor` (if configured). Only affects buys —
   holdings keep their `panel_score` for rotation.

When the flag is off, `PanelScoringJob.should_skip()` returns True and the
per-ticker `rank_score` set by `CandidateJob` is used as-is (identical to 103).

### Panel-driven policy knobs

Three additional knobs under `ranking.panel_scoring` let the panel score shape
downstream decisions without touching pipeline code:

| Knob | Where it plugs in | Effect |
|---|---|---|
| `buy_floor` | `VetoWeakBuysTask` | Drops candidates with `panel_score < buy_floor` before ranking |
| `sizing.{enabled, floor, ceiling, min_mult}` | `SizeAndEmitTask`, `EmitRotationsTask` via `conviction_multiplier()` | Scales `max_position_pct` by a multiplier in `[min_mult, 1.0]` based on `panel_score`'s location in `[floor, ceiling]` |
| `rotation_advantage` | `find_rotation_pairs` / `RotationJob` | Requires the candidate's `panel_score` to beat the held position's by at least this fraction before a rotation pair is emitted |

All three short-circuit cleanly when unset or when the panel flag is off.

---

## 3. FullTrainingPipeline (`pp_training_full.py`)

`scripts/train_104.py` is a thin CLI wrapper. All orchestration lives in
`kernel/pipeline/pp_training_full.py`:

```
FullTrainingPipeline
  ├─ BaselineTournamentJob     wraps TrainingPipeline (per-ticker champion)
  │    └─ RunBaselineTask
  │
  ├─ PanelTrainingJob          wraps PanelTrainingPipeline
  │    ├─ FetchPanelDataTask         OHLCV for watchlist ∪ SPY ∪ sector ETFs
  │    ├─ BuildPanelFeatureFramesTask  per-ticker labelled feature frames
  │    └─ RunPanelTrainingTask       panel-LTR model → artifacts/panel-ltr.json
  │
  └─ RecalibrationJob          wraps scripts.recalibrate_scores.recalibrate
       └─ RunRecalibrationTask refresh blend weights + per-symbol calibrations
```

Every phase is skippable via CLI flag (`--skip-baseline`, `--skip-panel`,
`--skip-recalibrate`) — each Job's `should_skip(ctx)` reads the corresponding
bool on `FullTrainingContext`. Tasks short-circuit the enclosing Job's chain
by returning False, matching the convention in `pp_inference.py`.

---

## 4. Runtime wiring

Three runtime entry points must set the same panel flag. All of them read
`ranking.panel_scoring.enabled` from `strategy_config.json`:

| Entry point | Responsibility |
|---|---|
| `main.py` (LEAN) | Uses `LeanAdapter` which pulls **520 bars** from LEAN History when the flag is on, then calls `prepare_inference_panel_frames` before `InferencePipeline.run()` |
| `live/runner.py` | Uses `RunnerAdapter` — identical prep, but fetches OHLCV from parquet cache via `common.fetch_ohlcv` |
| `sim/runner.py` | Pipeline-only since April 2026 — `SimAdapter` + `InferencePipeline` mirror LEAN and live. Legacy `_run_backtest_legacy` + `swap_in_panel_scores` + `apply_ngboost_head` deleted. `panel_feature_frames` + `panel_factor_frames` pre-built by caller, sliced per-bar. |

The lazy import pattern in `pp_inference.py` (`from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob` inside `run()`) is load-bearing: `kernel.panel_pipeline.__init__` pulls in `job_panel_scoring`, which imports from `kernel.pipeline.context`, which triggers `kernel.pipeline.__init__` → `pp_inference`. Without the deferral we have a cycle. See `tests/test_panel_alignment.py::TestPipelineOrdering::test_panel_job_imported_lazily_inside_run` for the guard.

---

## 5. Test coverage

All renquant_103 alignment + policy tests ported to renquant_104 paths. Plus
panel-specific coverage:

| Test file | What it covers |
|---|---|
| `tests/test_panel_scoring_job.py` | Load / BuildMatrix / ApplyScores / VetoWeakBuys / Job wiring |
| `tests/test_panel_training_pipeline.py` | PanelTrainingPipeline end-to-end with Job/Task ABCs |
| `tests/test_panel_pipeline_e2e.py` | `prepare_inference_panel_frames` path |
| `tests/test_panel_inference.py` | inference-time feature / factor flows |
| `tests/test_panel_alignment.py` | **34 tests** — flag parity across LeanAdapter / RunnerAdapter / PanelScoringJob, pipeline ordering invariant, panel veto / conviction sizing / rotation advantage, plus NGBoost: `TestNGBoostFlagParity`, `TestApplyNGBoostScoring` (μ−λσ / additive / λ-scaling / missing features), `TestSigmaSizing` (median / bounds / end-to-end `SizeAndEmitTask`) |
| `tests/test_ngboost_head.py` | **12 tests** — fit / predict / save-load / μ-σ recovery / σ heteroskedasticity / combined score / σ-sizing bounds |
| `tests/test_training_cadence.py` | **8 tests** — daily preserves existing behaviour, weekly short-circuits off-cadence days, `--force` bypasses |
| `tests/test_fundamentals_cache.py` | **9 tests** — `FundamentalsStore` parquet cache + injected-provider fetch + watchlist iteration with per-ticker error isolation |
| `tests/test_panel_factors.py` | extended with **5 tests** for fundamental z-columns (no-op when `fundamentals=None`, four-z-column emission, cross-sectional normalisation, sector-median fill, empty-dict guard) |
| `tests/test_panel_*` (frame, labels, neutralization, imputation, purged_cv, ltr_model, feature_matrix) | tests for the underlying building blocks |

Total test count after Stage 2 + Stage 3.1: **857 collected — 855 passing, 2 skipped**.

## 5a. Stage 2 — NGBoost μ,σ head (default off)

`training_panel/ngboost_head.py::NGBoostHead` fits a separate NGBoost Normal(μ, σ) regressor on **raw** residual forward returns (not Gaussianized), producing location + scale per ticker. Enabled via `panel_ltr.ngboost.enabled` (training) and `ranking.panel_scoring.ngboost.enabled` (inference). `PanelTrainingPipeline` adds `PanelNGBoostJob` as Phase 5 (`NGBoostFitTask` → `NGBoostSaveTask`); `PanelScoringJob` grows to 6 tasks (`LoadScorer → BuildFeatureMatrix → ApplyScores → VetoWeakBuys → LoadNGBoost → ApplyNGBoost`).

- `score_mode` (default `mu_minus_lambda_sigma`) — overrides `rank_score` and `panel_score` with `μ − λσ` so selection + rotation use the uncertainty-aware score. Set to `additive` to keep LTR rank_score and only populate `μ/σ` fields for sizing.
- `lambda_sigma` (default 1.0) — penalty multiplier on σ.
- `ranking.panel_scoring.sigma_sizing.{enabled,floor,ceiling}` gates a new `sigma_multiplier()` that scales `max_position_pct` by `clip(σ_median / σ_i, floor, ceiling)` in both `SizeAndEmitTask` and `EmitRotationsTask`.

Artifact: single JSON at `artifacts/ngboost-head.json` with a base64-encoded pickle blob (NGBoost has no pure-JSON serializer); still self-contained and loadable without a side-car `.pkl`.

## 5b. Stage 3.1 — Fundamental factor features (enabled)

`kernel/fundamentals.py::FundamentalsStore` caches OpenBB snapshots at `data/fundamentals/{SYMBOL}.parquet`. Columns: `earnings_yield`, `roe`, `gross_profitability`, `book_to_price`, `short_pct_float` (yfinance `.info.shortPercentOfFloat`). `scripts/fetch_fundamentals.py` is the watchlist driver. `LoadFundamentalsTask` (Phase 1 of `PanelTrainingPipeline`) loads the cache into `PanelTrainingContext.fundamentals`; `TickerPanelFactorJob` broadcasts each ticker's scalar factors to the daily index; `FactorZScoreTask` cross-sectionally z-scores them (with sector-median fill for missing values). Fully wired — enable via `panel_ltr.fundamentals.enabled: true`.

## 5c. Stage 3.2 — Orthogonal time-series factors (Round 3-5, all wired)

`training_panel/factors.py` emits 5 new OHLCV-derived orthogonal factors in addition to the 4 base factors (`size_z`, `mom_12_1_z`, `beta_60d_z`, `resid_mom_z`):

- `amihud_illiq_z` — Amihud (2002) illiquidity: rolling mean `|return| / $volume` × 1e6. Illiquidity premium.
- `volume_shift_z` — `log(20d_volume / 60d_volume)` — trading-interest shifts.
- `price_to_high_z` — close / 252d rolling max. 52-week-high anchor (George & Hwang 2004 behavioral).
- `realized_vol_z` — 20d annualized return σ. Low-vol anomaly (monotone `-1`).
- `drawdown_peak_z` — `close/peak - 1`. Reluctance-to-realize behavioral (dropped as redundant with price_to_high).

Plus time-varying fundamentals (opt-in via config):

- `earnings_surprise_cum_z` — via `kernel/earnings_surprise.py` + `LoadEarningsSurpriseTask` + `compute_earnings_surprise_cum`. Trailing-4Q cumulative EPS surprise %, daily-ffilled between announcements (yfinance `.earnings_dates`).
- `insider_net_buy_90d_z` — via `kernel/insider_trades.py` + `LoadInsiderTradesTask` + `compute_insider_net_buy_cum`. SEC Form 4 executive-only (isOfficer=true) open-market P/S transactions, trailing-90d net USD buy (monotone `+1`).

Individual feature IC (Spearman vs Gaussianized residual label, CPCV):

```
beta_60d_z              -0.063    (low-beta anomaly — strongest)
realized_vol_z          -0.044
roe_z                   +0.037
amihud_illiq_z          +0.026
gross_profitability_z   +0.025
price_to_high_z         +0.025
book_to_price_z         -0.022    (value factor inverted in recent regimes)
earnings_surprise_cum_z -0.033    (no constraint applied — data vs theory conflict)
insider_net_buy_90d_z   TBD       (fetch populating)
```

Panel IC improvement arc (Round 0 → Round 5):

```
0.038  baseline (April 2026 start-of-session)
0.052  + hyperparam regularisation (Round 2)
0.061  + Round 1 bug fixes (calibration + z-score wiring)
0.062  + CPCV 15-split (more robust estimate)
0.066  + monotone constraints on 6 economically-signed factors
0.065  + 5 orthogonal price/volume factors (Round 3)
0.064  + short interest (Round 4)
0.06x  + earnings surprise + insider trades (Round 5, pending final run)
```

Ship gate for production: panel OOS IC ≥ 0.05 across CPCV folds. Currently comfortably above.

## 5d. No-trade monitoring

`kernel/pipeline/task_monitor.py::MonitorIdleStreakTask` runs at the end of `InferencePipeline` and tracks:

- `no_trade_streak` — consecutive days with zero orders AND zero exits
- `no_candidate_streak` — consecutive days with zero candidates surviving CandidateJob

State persists across bars via `SimAdapter._monitor_state` and `RunnerAdapter.live_state.json` (`monitor_state` field). Emits WARNING logs when either streak exceeds `monitoring.max_no_trade_days` (default 15) / `max_no_candidate_days` (default 15) — the scheduled-run ntfy surfaces these.

`SimResult.longest_no_trade_streak` is a post-hoc stat (computed from the trade log + equity curve). Opt-in invariant test `RENQUANT_FULL_SIM=1 pytest tests/test_no_trade_invariant.py` asserts `longest_no_trade_streak < 20d`. Enforces the user contract: **"it's OK not to trade, but NOT systematically."**

Companion structural fix: `FilterUniverseFloorTask` always admits `defensive_tickers` regardless of floor type/threshold — prevents the monitor's trigger condition (low-confidence regime restricts universe to defensives → all defensives filtered by Sharpe floor → systemic no-trade).

## 5e. Future: transformer panel backend

See `doc/renquant_104_transformer_design.md`. Cross-sectional attention across the date-group as an alternative `panel_ltr.backend`. MPS-targeted. Ship gate: ≥1.3× XGBoost OOS IC.

## 5c. Stage 1 cleanups (all behind flags, defaults preserve existing behaviour)

- `training.cadence` (default `"daily"`) — set to `"weekly"` with `training.weekly_weekday: 6` (Sunday) to short-circuit `FullTrainingPipeline.run()` on non-cadence days. `scripts/train_104.py --force` and `scripts/retrain_panel.sh` bypass the gate.
- `ranking.tournament.exclude_models` (default `[]`) — e.g. `["qlearning"]` drops that approach from `run_tournament_all`. `QLearningModel` is kept intact so rollback is just a config flip.
- rs_score retired from ranking math. `BlendScoresTask` hardcodes `(w_rank, w_rs) = (1.0, 0.0)`; a legacy `ranking.blend_weights` with non-zero rs weight triggers a one-time warning. `recalibrate_scores.py` no longer writes that key (offline diagnostic helper `_compute_blend_weights` is retained for tests). `rs_score` is still carried on `CandidateResult` for log readability.

---

## 6. Scheduled runs

Daily automation mirrors renquant_103 schedule but drives the 104 scripts:

| Run | Time (PT) | Script | What it does |
|---|---|---|---|
| Market open | 6:32 AM | `live_only_104.sh --sell-only` | Exit stop-loss / gap-down positions using today's opening price |
| Pre-close | 12:44 PM | `live_only_104.sh --sell-only` | Exit intraday stop breaches before close |
| After close | 1:55 PM | `daily_104.sh` | `FullTrainingPipeline` → `export_lean_watchlist` → `live.runner --broker alpaca --once` |

LaunchAgents installed at `~/Library/LaunchAgents/com.renquant.{open,preclose,daily}104.plist`. Log paths: `logs/daily_104/`, `logs/live_104/`. Lock files under `/tmp/renquant_104_*` prevent concurrent runs. NYSE holiday guard and already-ran-today guard are identical to 103.
