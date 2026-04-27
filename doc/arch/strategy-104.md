# renquant_104 — Panel-LTR Cross-Sectional Ranking

**Status**: Active daily strategy.
**Author**: Ren Hao
**Last updated**: 2026-04-26 (round-7 + model-selection systematization)
**Based on**: renquant_103 (adaptive regime multi-stock)

**Current production artifact** (`artifacts/panel-ltr.json`, 2026-04-26):
- Backend: XGBoost rank:pairwise (`kind: panel_ltr_xgboost`)
- 28 feature_cols (per-ticker indicators + factor z-scores + fundamentals)
- Panel: 74,547 rows × 99 tickers × 753 dates
- OOS mean IC: **0.0482** (q25 0.028 / q50 0.055 / q95 0.082, std 0.025)
- best_iter 9, params `eta=0.02 max_depth=3 min_child_weight=60 ss=cs=0.5 λ=5 α=2 seed=42`

**Round-7 additions** (2026-04-26):
- **Macro factor frame** (VXX/HYG/UUP/DBC/GLD/TLT/XLV/XLU/KRE/MTUM/USMV × {level_z, chg_5d_z, chg_20d_z}, `kernel/macro.py`). Default OFF in prod; macro-enabled XGBoost variant (61 features, OOS IC 0.0393) preserved at `panel-ltr.macro-enabled.bak.json` but NOT promoted (-18% IC vs prod). LGBM-with-macro experiment (2026-04-26 evening) confirmed macro reduces IC further to 0.0224. See [`../components/macro-factor-frame-design.md`](../components/macro-factor-frame-design.md).
- **Model-selection 4-tier SOP** ([`../components/model-selection.md`](../components/model-selection.md)): 11 acceptance gates (G1-G11), backend tournament (`scripts/select_best_model.py`), shadow/challenger infrastructure (`kernel/challenger.py`).
- **Atomic-swap promote** with staging→`.previous.json` rollback target.
- **Operator UX**: `scripts/model_dashboard.py`, `scripts/finalize_challenger.py`, `scripts/check_challenger_window.sh`.

**Watchlist** (99 tickers): 60 tech split into 4 sub-buckets: `giant_tech` (8), `ai_chip` (18), `datacenter_hw` (10), `software` (24). 39 non-tech across finance/healthcare/industrial/consumer/energy/commodity/utility. Mutual-fund-overlap-weighted curation (VPMAX + FCNTX + AGTHX top holdings prioritized).

**Resolved on 2026-04-27 (S1 investigation)** — the apparent 41→28 "regression" was a **deliberate cleanup**:
- Commit `e9d71e6` (2026-04-25 18:46 PT, "Tier 1 batch: drop 13 noise/sparse features") explicitly dropped 13 columns flagged as low-IC or sparse by `FeatureDiagnosticTask`: `earnings_yield_z` (IC=−0.0009), `rsi`, `macd_hist`, `amihud_illiq_z`, `obv_slope`, `vol_ratio_z`, `m_vol_ratio_z`, `m_morning_drift_z`, `m_afternoon_drift_z`, `m_closing_30min_drift_z`, `m_first_hour_vol_pct_z`, `overnight_gap_z`, `insider_net_buy_90d_z` (80/99 tickers >50% NaN). Same commit added `book_to_price_z: -1` monotone constraint (data-driven sign reversal: actual IC=−0.0474 → low B/P / growth wins in current regime).
- Result is a feature-quality win, not a regression: prod OOS IC moved from PRE-MINUTE era's 0.0391 (31 features) to current 0.0482 (28 features) — **+23% IC despite fewer columns**.
- The doc's prior "41-feature, +30.90% APY" line was from a never-shipped earlier configuration. The current 28-feature artifact is the cleaned model.

**Open questions remaining**:
- Sim APY/Sharpe metrics for the current 28-feature artifact have not been re-measured. Run `scripts/sim_smoke.py`-style verification before quoting any APY number against current prod.

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
identical. The logic graph in `doc/arch/decision-graph-103.md` continues to apply
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

See `doc/components/transformer-104.md`. Cross-sectional attention across the date-group as an alternative `panel_ltr.backend`. MPS-targeted. Ship gate: ≥1.3× XGBoost OOS IC.

## 5c. Stage 1 cleanups (all behind flags, defaults preserve existing behaviour)

- `training.cadence` (default `"daily"`) — set to `"weekly"` with `training.weekly_weekday: 6` (Sunday) to short-circuit `FullTrainingPipeline.run()` on non-cadence days. `scripts/train_104.py --force` and `scripts/retrain_panel.sh` bypass the gate.
- `training.model_ttl_days` (default `0` = disabled) — per-ticker TTL gate inside `_run_ticker_chain`. When > 0, a ticker whose `models/{TICKER}/{TICKER}-policy-metadata.json` has a `trained_date` within TTL is skipped for the current run (cached model is kept as-is, `tc.ttl_skipped=True`). `--force` on `train_104.py` propagates to `config["_force_retrain"]` and bypasses TTL just like cadence. daily_104.sh counts "TTL skip" lines in the log and appends e.g. `(12 ticker-TTL skips)` to the ntfy notification body. Calibration + panel + recalibration phases still run — only the per-ticker tournament is skipped. Covered by `tests/test_model_ttl.py` (10 tests: disabled, fresh, stale, boundary, no artifact, corrupt json, missing trained_date, None strategy_dir, chain short-circuits, chain runs under --force).

### 2026-04-24 PT session — flag-gated rotation + panel-exit improvements (all default-off)

- **Rotation V1 gates** — `rotation.min_raw_advantage_pct` (pre-tax/cost edge floor) + `rotation.persistence_bars` (same pair must appear N prior bars). Default 0.0/0 both → pre-V1 behaviour. Tests: `tests/test_rotation_v1_gates.py` (10).
- **Rotation V2 scoring** — `rotation.scoring_mode = "mu_minus_lambda_sigma"` swaps the isotonic-calibrated ER for NGBoost-direct μ−λσ driver. λ defaults to 1.0 (override via `rotation.lambda_` or `ranking.panel_scoring.ngboost.lambda_`). Falls back to ER on missing μ/σ. Tests: `tests/test_rotation_v2_scoring.py` (4).
- **Rotation V3 gates** — `rotation.enabled_regimes` (allow-list of regime names) + `rotation.held_max_unrealized_pct` (cap on held unrealized, protects hot runners). Default None/None → pre-V3. Tests: `tests/test_rotation_v3_gates.py` (7).
- **Panel conviction exit V2** — `risk.panel_exit.trigger_mode` = "and" (default, backwards-compat) | "or". OR mode fires when EITHER panel_score<floor OR μ<=ceiling. Intended for exits where panel and μ disagree; V1's strict AND almost never fired. Tests: `tests/test_panel_exit_v2.py` (9).
- **snapshot default** — `sim.runner.run_backtest(snapshot=True)` is the default, isolating notebook sims from concurrent retrains via `kernel/artifact_snapshot.py`.
- **10-minute bar infra** — `MinuteBarStore` + `compute_minute_features` (10 `m_`-prefixed factor columns) + `scripts/fetch_minute_bars.py`. Flag `panel_ltr.minute.enabled` (default false). Prereq for transformer retry (shelved at panel < 200k rows, 10-min would ~6× the data to ~280k). Tests: `tests/test_minute_features.py` (15).
- **Notebook robustness tool** — `sim/analysis.strip_top_n_trades(result, n=3)` removes top-N realized-return trades and reports the expected-case APY (answers "am I riding 3 lucky mega-winners?"). Tests: `tests/test_sim_analysis.py` (8).
- `ranking.tournament.exclude_models` (default `[]`) — e.g. `["qlearning"]` drops that approach from `run_tournament_all`. `QLearningModel` is kept intact so rollback is just a config flip.
- rs_score retired from ranking math. `BlendScoresTask` hardcodes `(w_rank, w_rs) = (1.0, 0.0)`; a legacy `ranking.blend_weights` with non-zero rs weight triggers a one-time warning. `recalibrate_scores.py` no longer writes that key (offline diagnostic helper `_compute_blend_weights` is retained for tests). `rs_score` is still carried on `CandidateResult` for log readability.

---

## 6. Scheduled runs

Daily + weekly automation mirrors renquant_103 schedule with additional 104-specific jobs:

| Run | Time (PT) | Script | What it does |
|---|---|---|---|
| Market open | 6:32 AM Mon-Fri | `live_only_104.sh --sell-only` | Exit stop-loss / gap-down positions |
| Intraday 5-min | 7:00-12:30 every 30min | `intraday_sell_104.sh` | Alpaca IEX overlay — intraday SDL / trailing-stop |
| Pre-close | 12:44 PM Mon-Fri | `live_only_104.sh --sell-only` | Exit intraday stop breaches |
| Conditional retrain | 13:10 PM Mon-Fri (new 2026-04-24) | `conditional_retrain_104.sh` | Fire `train_104.py --force` if SPY \|daily Δ\| > 2% or VIX \|daily Δ\| > 5% |
| Daily pass | 1:55 PM Mon-Fri | `daily_104.sh` | `FullTrainingPipeline` (cadence-gated) → `export_lean_watchlist` → `backfill_forward_returns` → `compute_portfolio_metrics` → `live.runner --broker alpaca --once` |
| Weekly APY check | 12:00 PM Sun | `weekly_apy_check.py` | 30-day rolling APY + DD streak; surfaces latest Sharpe |
| Watchlist screen | 12:05 PM Sun (new 2026-04-24) | `screen_watchlist.py` | 6-month per-ticker Sharpe; flag DROP + ADD candidates |
| Sunday retrain | 10:00 AM Sun | `retrain_panel.sh` | Forced weekly panel + ngboost retrain |

LaunchAgents at `~/Library/LaunchAgents/com.renquant.{open,intraday,preclose,conditional-retrain,daily,weekly-apy,screen-watchlist,retrain-panel}104.plist`. Log paths: `logs/daily_104/`, `logs/live_104/`, `logs/intraday_104/`, `logs/conditional_retrain_104/`, `logs/watchlist_screen/`. Lock files under `/tmp/renquant_104_*` prevent concurrent runs. NYSE holiday guard and already-ran-today guard in every script.

---

## 7. Decision surface symmetry (2026-04-24)

All 5 decision surfaces now consult model + policy together (user spec):

| Surface | Per-ticker tournament | Panel score | NGBoost μ,σ | Kelly target | Notes |
|---|---|---|---|---|---|
| **Buy new** | ✓ | ✓ | ✓ (μ→size, σ→size) | ✓ (size cap) | `ScoreBuyTask` + gate chain + `SizeAndEmitTask` |
| **Sell — price rules** | ✓ (model-streak only) | — | — | — | `trailing_stop`, `stop_loss`, `max_hold`, `sdl` are price-only by design |
| **Sell — panel conviction** (new) | — | ✓ | ✓ | — | `PanelConvictionExitTask` — tiebreaker when price rules don't fire. Flag `risk.panel_exit.enabled` (default off pending A/B) |
| **Top-up held** | — | ✓ | ✓ (via kelly_target) | ✓ | `TopUpHeldTask` — when `kelly_target - current > top_up_threshold` |
| **Trim held** | — | ✓ (mu guard) | ✓ (mu guard) | ✓ | `TrimHeldTask` — when `current - kelly_target > trim_threshold`. Opt-in via `trim_enabled=false` default (A/B regressed). Guards: skip when `kelly_target < 0.05` OR `mu <= 0` per §2b audit |
| **Rotate (swap)** | — | ✓ | ✓ (via kelly_target) | ✓ | `RotationJob` — three filter layers: `panel_rotation_advantage`, `kelly_rotation_advantage`, thesis-A. Route B alternative: `rotation.mode="thesis_primary"` uses thesis-degradation as primary gate (not filter) |

---

## 8. Kelly sizing stack (2026-04-24)

Continuous-returns Kelly: `f* = μ / σ²`, capped at `max_concentration` + regime `max_position_pct × confidence`. `ApplyKellySizingTask` writes `kelly_target_pct` on every candidate AND holding every bar; four tasks consume it.

Config block (golden v4.1):
```json
"ranking.kelly_sizing": {
  "enabled":           true,
  "fractional":        0.50,       // half-Kelly (estimation error absorption)
  "max_concentration": 0.35,
  "min_edge":          0.0,
  "top_up_threshold":  0.05,
  "base_rate":         0.273,
  "trim_enabled":      false,      // A/B showed regression — opt-in only
  "trim_threshold":    0.10,
  "trim_target_floor": 0.05,       // §2b audit guard
  "rotation_advantage":      0.0,  // BC gate (dormant until model improves)
  "rotation_target_floor":   0.05, // §2b audit guard
  "disable_extra_multipliers": false,  // pure-Kelly mode flag
  "per_session_buy_cap":     null  // multi-entry accumulation (null = off)
}
```

Decision math unified — one source of truth for `kelly_target_pct`, consumed by:
1. `SizeAndEmitTask` — caps new-buy size
2. `TopUpHeldTask` — triggers top-up
3. `TrimHeldTask` — triggers trim (opt-in)
4. `RotationJob.BuildPairsTask` — Kelly-delta gate filter (dormant)

---

## 9. Thesis-degradation rotation (2026-04-24)

User insight: today's Kelly target is noisy. Compare instead against held's FIXED entry-time baseline. `HoldingState` gains 3 entry-stamp fields:
- `entry_rank_score` — tournament+panel calibrated score at buy
- `entry_panel_score` — panel score at buy
- `entry_kelly_target_pct` — Kelly target at buy

Stamped by all 3 adapters (sim, LEAN, live — live persists in `live_state.json::entry_signals`). Cleared on full exit.

Two modes:
- **Filter mode (default)** — `ranking.thesis_rotation.enabled: true` — runs alongside ER-based rotation, filters pairs. Default OFF pending A/B.
- **Primary mode (new, Route B)** — `rotation.mode: "thesis_primary"` — bypasses ER discovery entirely, uses thesis-degradation as primary swap criterion. Config in `rotation.thesis.{degradation_pct, uplift_pct}`.

---

## 10. Decision-trace DB (Plan AA, 2026-04-24)

All pipeline decisions written to SQLite for audit + tuning. Split into two roles:
- `data/runs.db` — live + LEAN (authoritative, permanent)
- `data/sim_runs.db` — sim (ephemeral; TRUNCATEd at start of each `run_backtest`)

8 tables: `pipeline_runs`, `candidate_scores`, `trades`, `rotations`, `training_runs`, `ticker_forward_returns`, `live_state_snapshots`, `portfolio_daily_metrics`.

Full schema reference: `doc/components/databases.md`. Every row carries `commit_sha` for reproducibility.

Analysis: `scripts/analyze_decision_factors.py` and `scripts/compute_portfolio_metrics.py` produce empirical IC, tier-realization, regime-conditional, Sharpe/VaR reports.
