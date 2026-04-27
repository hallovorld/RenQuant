# Architecture

## Design Principle: Glass-Box Pipeline

RenQuant is built around **strict layer decoupling**. Each layer has one job, communicates via well-defined interfaces (JSON files), and can be developed or replaced independently. Every decision in the pipeline is inspectable — no end-to-end black boxes.

---

## Four-Layer Pipeline

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Research (Notebooks/)                     │
│  - Fetch OHLCV (yfinance/IBKR, cached as Parquet)  │
│  - Compute indicators via registry                  │
│  - Relativize indicators vs SPY benchmark           │
│  - Train model (Manual/RF/QL/FQI/Optimization)      │
│  - Export: JSON model artifacts + policy-metadata    │
└───────────────────────┬─────────────────────────────┘
                        │ JSON artifacts
┌───────────────────────▼─────────────────────────────┐
│  Layer 2: Model Artifacts (backtesting/<strategy>/) │
│  - JSON models (XGBoost, Q-table, rules, RF trees)  │
│  - policy-metadata.json → state cols, indicator      │
│    params, gate rules, model type                    │
└────────────┬──────────────────────┬─────────────────┘
             │ LEAN backtest        │ live runner
┌────────────▼──────────┐  ┌───────▼─────────────────┐
│  Layer 3: Backtesting  │  │  Layer 3b: Live Trading  │
│  (LEAN / Docker)       │  │  (python -m live.runner)  │
│  - Loads JSON models   │  │  - Paper / Alpaca / IBKR │
│  - Daily inference     │  │  - Loads same artifacts   │
│  - Event-driven sim    │  │  - Scheduled or --once    │
└────────────┬──────────┘  └───────┬─────────────────┘
             │                     │
┌────────────▼─────────────────────▼─────────────────┐
│  Layer 4: Analysis (scripts/analyze_backtest.py)   │
│  - Load LEAN results or live logs                  │
│  - Dashboard + normalized performance chart        │
│  - Performance statistics + decision telemetry     │
└─────────────────────────────────────────────────────┘
```

---

## Shared Library: `common/` (renquant_101 and renquant_102 only)

`common/` is the shared library for the older strategies. **It is not used by renquant_103 or renquant_104** — each of those ships its own self-contained `kernel/` package (see the kernel section below). `common/` will be removed once 101 and 102 are retired.

| Module | Contents |
|--------|----------|
| `common/config.py` | `load_strategy_config`, `split_date_parts`, `build_model_path` |
| `common/data/` | `fetch_ohlcv` (Parquet cache + yfinance/IBKR sources), `DataSource` ABC, `LocalStore` |
| `common/indicators/` | `compute_indicators`, `add_indicators`, `list_indicators`, `@register` decorator; 12 registered indicators across 4 categories; regime detection (non-registered): `compute_hurst`, `rolling_hurst`, `compute_cusum`, `rolling_cusum`, `build_gmm_features`, `RegimeGMM` |
| `common/models/` | `BaseModel` ABC, 6 implementations: `ManualModel`, `ClassificationModel`, `QLearningModel`, `FQIModel`, `OptimizationModel`, `XGBoostModel`, `create_model` factory, and `scoring.py` for raw-score extraction and cross-model calibration (`ScoreCalibration`: isotonic for n≥300, Platt scaling for 120≤n<300, constant base-rate for n<120) |
| `common/models/learners/` | `RTLearner`, `BagLearner`, `TabularQLearner` |
| `common/strategy.py` | `StrategyConfig` dataclass, `Strategy` class (composes data + indicators + model) |
| `common/portfolio.py` | `compute_portvals`, `portfolio_stats` — local portfolio simulator |
| `common/tax.py` | `compute_trade_tax`, `compute_after_tax_pnl`, `load_tax_config`, `add_tax_columns`, `tax_rate_for_holding` — after-tax return analysis |
| `common/plotting.py` | `backtest_dashboard`, `plot_normalized_performance`, parse/plot utilities |

---

## Layer 1: Research

**Location**: `backtesting/<strategy>/` (for 101/102, notebooks live in `Notebooks/`; for 103 and 104, the notebook lives alongside code in `backtesting/renquant_10X/`)
**Environment**: `renquant` conda env

The notebook is where training happens. The typical workflow for 101/102:

1. **Data ingestion** — `common.fetch_ohlcv` fetches daily OHLCV for both the target stock and SPY benchmark (cached locally as Parquet)
2. **Indicator computation** — `common.compute_indicators` applied to both stock and SPY
3. **Relative feature construction** — indicators are relativized against SPY:
   - **Ratio** (`stock / SPY`) for always-positive indicators: RSI, ADX
   - **Difference** (`stock - SPY`) for zero-crossing indicators: MACD hist, CCI, BBP, Williams %R, OBV slope
   - Additional trend-following features: `trend` (price/50EMA), `trend_long` (price/200EMA), `rel_mom_20d`, `rel_mom_60d`
4. **Model training** — depends on model type (see below)
5. **Comparison** — all models simulated on the 30% OOS test set with constraints (wash sale 30d, min hold 20d, max hold 500d), compared with stock and SPY buy-and-hold benchmarks. Both OOS Sharpe and in-sample Sharpe are recorded.
6. **Export** — best model by OOS after-tax Sharpe auto-exported to `backtesting/<strategy>/` (Sharpe floor: 0.8 OOS for renquant_102). Orphan model directories for symbols not in the current watchlist are purged on each run.

---

## Layer 2: Model Artifacts

**Location**: `backtesting/<strategy>/`

All artifacts are JSON (not pickle) — required for LEAN compatibility and human-readability.

| File | Contents |
|------|----------|
| `*-policy-metadata.json` | Model type, state columns, indicator parameters, gate rules, hyperparams, `sharpe` (OOS), `sharpe_is` (in-sample, diagnostic), `trained_date`, `best_approach`, and optional `score_calibration` |
| `*-q-hold/buy/sell.json` | XGBoost models (FQI) |
| `*-rf-trees.json` | Random Forest tree structure (Classification) |
| `*-qtable.json` + `*-bin-edges.json` | Q-table + discretization (Q-Learning) |
| `*-manual-rules.json` | Threshold rules (Manual) |
| `*-xgb-buy.json` + `*-xgb-sell.json` | XGBoost buy/sell classifiers (XGBoost model) |

The policy metadata acts as a contract between research and execution — both must use identical indicator parameters.

For multi-stock strategies (renquant_102, renquant_103), models are organized per-symbol under `models/{SYMBOL}/`:
```
backtesting/renquant_102/models/
  TSLA/TSLA-policy-metadata.json, TSLA-rf-trees.json
  AMZN/AMZN-policy-metadata.json, AMZN-manual-rules.json
  ...
```
Each symbol's model may be a different type (the notebook picks the best approach per symbol).

renquant_103 adds three additional strategy-level artifacts (under `artifacts/` subdir):
```
backtesting/renquant_103/artifacts/
  spy-gmm-regime.json          # GMM parameters for 3-state regime classifier
  watchlist-correlation.json   # 120-day pairwise return correlations for correlation guard
  earnings-calendar.json       # Upcoming earnings dates per ticker (refreshed weekly)
```

---

## Layer 3: Backtesting (LEAN)

**Location**: `backtesting/<strategy>/main.py`
**Runtime**: QuantConnect LEAN engine (Docker)

`main.py` implements `QCAlgorithm`. The two strategies differ in their approach:

**renquant_101** (single-stock): Loads pre-exported model artifacts (Manual, Classification, or Q-Learning). `Initialize()` reads policy metadata and model files. `OnData()` computes indicators inline and scores actions via the exported policy.

> **Note**: renquant_101's `main.py` feeds raw indicators to the model (not relative to SPY). This is a known gap for that strategy.

**renquant_102** (multi-stock): Loads pre-trained models per symbol from `models/{SYMBOL}/`. `Initialize()` reads `strategy_config.json`, loads all per-symbol model artifacts (checking staleness), and sets up watchlist equities + SPY benchmark. `OnData()`:
  1. Processes sells first for held positions (apply per-stock model + constraints + max hold)
  2. Scans volume z-scores for non-held watchlist stocks (DETECT stage)
  3. For each spike candidate, computes 60-day indicators + relative features for stock and SPY (CONFIRM stage)
  4. Applies that stock's pre-trained model (classification, manual, or qlearning) to get buy/sell/hold signal
  5. Applies position sizing and trading constraints, executes orders (EXECUTE stage)

**renquant_103** (adaptive regime multi-stock): Extends 102's architecture with a 3-layer regime detector. `Initialize()` additionally loads `spy-gmm-regime.json`, `watchlist-correlation.json`, and `earnings-calendar.json`. `OnData()`:
  1. Accumulates SPY daily returns; runs Hurst (Layer 1), CUSUM (Layer 2), GMM (Layer 3) to classify regime and confidence
  2. Sets regime-adaptive parameters (stop-loss, position size, max hold, drawdown halt) — position sizing scales continuously with GMM confidence
  3. Processes sells (regime-adaptive stop-loss and trailing stop in BULL_CALM: 20% gain trigger, 18% trail below HWM; single-day loss gate 10% in BULL_CALM; max hold 500d/10d CHOPPY; model-sell requires min_hold=20d + 3 consecutive signals)
  4. BUY GATES (all short-circuit): open slots check → transition uncertainty window (3 bars post-CUSUM) → BEAR branch (1 defensive slot for GLD/TLT/XLV/XLU only) → SPY velocity crash filter (>3% drop in 3 days) → SPY EMA50 trend gate
           5. SCAN: model signal is the entry trigger (no separate volume-scan gate). Each candidate: earnings filter → model buy signal → calibrated `rank_score` threshold. The native model output is kept as `raw_score`, then mapped to a comparable probability-like `rank_score` for cross-model ranking. Relative strength (20d vs sector ETF) is blended with `rank_score` using config-driven `ranking.blend_weights`.
     6. EXECUTE: greedy selection by rank; each slot checks tiered threshold → wash-sale → sector guard (max 3/sector) → correlation guard (max 0.70). Position sized as `min(cash − portfolio × cash_reserve_pct × regime_confidence, portfolio × max_position_pct × regime_confidence)`.

**Scoring note**: mixed model families are not directly comparable in raw form. Notebook training now fits per-symbol `score_calibration`, exports it in `policy-metadata.json`, and uses calibrated `rank_score` for simulation ranking. LEAN consumes that exported calibration when present. The live runner also uses `kernel/scoring.py` for renquant_103, with a runtime fallback if metadata is missing. Raw scores are still logged as diagnostics.

**Important**: `main.py` is self-contained. It does **not** import `common/` because LEAN Docker cannot access it.

**renquant_103 kernel**: Strategy logic is extracted into `backtesting/renquant_103/kernel/` — a self-contained package with zero `common/` imports (only stdlib + numpy + pandas). LEAN imports it locally (`from kernel.x import ...`), and `live/runner.py` adds the strategy dir to `sys.path` at runtime. This is the canonical source of truth for all 103 inference logic:

| Module | What it provides |
|--------|-----------------|
| `kernel/config.py` | Regime constants, `artifact_path()` helper |
| `kernel/regime.py` | `RegimeState`, `detect_regime()`, Hurst + CUSUM + GMM |
| `kernel/indicators.py` | `compute_all()`, `build_feature_frame()`, `build_spy_context()` |
| `kernel/models.py` | `load_artifact()`, `score_artifact()`, `calibrate_score()` |
| `kernel/exits.py` | `HoldingState`, `ExitSignal`, `compute_exits()` (5-exit priority + tax hold gate) |
| `kernel/selection.py` | `CandidateResult`, `SelectionContext`, `run_selection_loop()`, guards, `compute_relative_strength()` |
| `kernel/sizing.py` | `compute_position_size()` with oversize fallback |
| `kernel/market_gates.py` | `check_spy_velocity_crash()`, `check_spy_ema_trend()` |
| `kernel/portfolio.py` | `update_drawdown_circuit_breaker()`, `compute_trade_tax()` |
| `kernel/rotation.py` | `find_rotation_pairs()`, `tax_drag()`, `effective_swap_margin()` (cross-sectional swap selector) |

**renquant_103 training**: Training-time logic lives in `backtesting/renquant_103/training/` — requires sklearn and xgboost. The notebook calls these modules directly; LEAN only uses `kernel/`.

| Module | What it provides |
|--------|-----------------|
| `training/features.py` | `build_training_features()`, `build_all_training_features()` — per-ticker labelled feature frames with relative indicators and rolling regime context |
| `training/tournament.py` | `run_tournament()`, `run_tournament_all()`, `oos_sharpe()` — 4-approach tournament (Classification, QLearning, Manual, XGBoost); fixed 2024-01-01 OOS split |
| `training/export.py` | `export_models()`, `retrain_live_models()` — save tournament winners to JSON; expanding-window retrain for live trading |
| `training/models.py` | `ClassificationModel`, `QLearningModel`, `ManualModel`, `XGBoostModel`, `create_model()` |
| `training/scoring.py` | `fit_probability_calibration()`, `ScoreCalibration` — fits isotonic/Platt/constant calibration |
| `training/regime.py` | `build_gmm_features()`, `RegimeGMM` — GMM training for the spy-gmm-regime.json artifact |
| `training/portfolio.py` | `portfolio_stats()`, `compute_portvals()` |
| `training/learners/` | `RTLearner`, `BagLearner`, `TabularQLearner` |

**renquant_103 pipeline**: The live execution and LEAN backtesting are both orchestrated by a structured parallel pipeline that replaces the monolithic `run_once_multi()` function. Two pipeline types share the same 3-phase pattern.

---

## Pipeline Architecture

### Two Pipelines

**`InferencePipeline`** — used by LEAN (`main.py` via `LeanAdapter`) and the live runner (`live/runner.py` via `RunnerAdapter`). Runs in 3 phases:

```
Phase 1: Global sequential
  RegimeJob → DrawdownJob → BuyGatesJob

Phase 2a: Parallel (ThreadPoolExecutor, per held ticker)
  TickerSellJob [AAPL] ──┐
  TickerSellJob [GOOG] ──┤─→ collect ctx.exits
  TickerSellJob [TSLA] ──┘

Phase 2b: Parallel (ThreadPoolExecutor, per candidate ticker)
  TickerCandidateJob [AMD] ──┐
  TickerCandidateJob [CAT] ──┤─→ collect ctx.candidates
  ...                         ┘

Phase 3: Global sequential
  RankingJob → RotationJob → SelectionJob
```

In renquant_104 (active strategy), a `PanelScoringJob` is slotted **between Phase 2b and Phase 3** — it re-scores both candidates and holdings with the cross-sectional panel-LTR model so rotation and selection compare everything on the same scale. It short-circuits via `should_skip()` when `ranking.panel_scoring.enabled=false`, in which case the pipeline behaves identically to 103. See `doc/arch/strategy-104.md` for the full spec.

**`SellOnlyPipeline`** — intraday sell-only variant (used on market-open and pre-close runs). Runs Phase 1 (RegimeJob → DrawdownJob) then parallel `TickerSellJob` — no buy phase.

**`TrainingPipeline`** (renquant_103) — notebook-driven training. Same 3-phase pattern:

```
Phase 1: Global sequential
  DataFetchJob → RegimeFitJob

Phase 2: Per-ticker parallel (ThreadPoolExecutor)
  TickerFeatureJob → TickerTournamentJob → TickerExportJob → TickerCalibrationJob
  (all four run in sequence within each ticker's worker thread)
  orchestrated by FeatureJob (the global Job that dispatches run_ticker_parallel())

Phase 3: Global sequential
  CorrelationJob
```

**`FullTrainingPipeline`** (renquant_104) — driven by `scripts/train_104.py` (no notebook required). Three Jobs in sequence:

```
BaselineTournamentJob   (wraps TrainingPipeline — per-ticker champions)
    ↓
PanelTrainingJob        (wraps PanelTrainingPipeline — single panel-LTR model)
    ↓
RecalibrationJob        (refresh blend weights + per-symbol calibrations)
```

Each phase is skippable via CLI flag (`--skip-baseline`, `--skip-panel`, `--skip-recalibrate`); each Job's `should_skip(ctx)` reads the flag off `FullTrainingContext`.

### Context Dataclasses

| Dataclass | Used by | Purpose |
|-----------|---------|---------|
| `InferenceContext` | `InferencePipeline`, `SellOnlyPipeline` | ~50 fields: config, models, ohlcv, regime, holdings, exits, candidates, orders |
| `TickerInferenceContext` | `TickerSellJob`, `TickerCandidateJob` | Per-ticker slice: ticker, ohlcv, model, holding, exit_signal, candidate |
| `TrainingContext` | `TrainingPipeline` | Global: config, ohlcv, gmm artifact, regime series, per-ticker results |
| `TickerTrainingContext` | `TickerFeatureJob`, `TickerTournamentJob`, etc. | Per-ticker: feature frame, tournament results, calibration metadata |

### Adapter Pattern

Both LEAN and the live runner need to translate their own state representations into `InferenceContext` and commit results back. Adapters handle this translation:

| Adapter | Location | Translates | DB role |
|---------|----------|-----------|---------|
| `LeanAdapter` | `adapters/lean.py` | LEAN `Portfolio`, `Securities`, `History()` → `InferenceContext`; commits via `Liquidate()` / `SetHoldings()` | (none — LEAN backtester currently does not write to DB; see `doc/components/databases.md` future work) |
| `RunnerAdapter` | `adapters/runner.py` | Broker account state + parquet OHLCV + `live_state.json` → `InferenceContext`; commits via `broker.place_order()` + state save + DB append | `data/runs.db` (authoritative, permanent) |
| `SimAdapter` | `adapters/sim.py` | Pre-loaded OHLCV + simulated portfolio state → `InferenceContext` for backsimulation (drives notebook + `sim.runner.run_backtest`) | `data/sim_runs.db` (ephemeral; TRUNCATEd at start of every `run_backtest`) |

**Isolation rules:**
- `kernel/` — no `common/` imports; stdlib + numpy + pandas only (Docker-safe)
- `adapters/` — can import `kernel/` and broker libs; not used inside LEAN Docker
- `main.py` — imports `kernel/` and `adapters/lean.py` only

### File Map

**Inference + training pipeline** (`kernel/pipeline/`, flat layout — `pp_*` orchestrators, `job_*` jobs, `task_*` atomic tasks at the same level):

| File | Contents |
|------|----------|
| `context.py` | `InferenceContext` (~50 fields), `TickerInferenceContext` |
| `pipeline.py` | `Task`, `Job`, `TickerJob` ABCs + `run_parallel()` |
| `pp_inference.py` | `InferencePipeline`, `SellOnlyPipeline` (+ ticker-context builders) |
| `pp_training.py` | `TrainingContext`, `TickerTrainingContext`, `TrainingTask`, `TrainingJob`, `TrainingTickerJob`, `TrainingPipeline` + all training jobs/tasks |
| `job_regime.py` | `RegimeJob` |
| `job_drawdown.py` | `DrawdownJob` |
| `job_gates.py` | `BuyGatesJob` |
| `job_sell.py` | `TickerSellJob` (per-ticker) |
| `job_candidates.py` | `TickerCandidateJob` (per-ticker) |
| `job_ranking.py` | `RankingJob` |
| `job_rotation.py` | `RotationJob` (held vs candidates on calibrated rank_score) |
| `job_selection.py` | `SelectionJob` |
| `task_regime.py` | `HurstTask`, `CUSUMTask`, `GMMTask`, `BEAROverrideTask`, `RegimeFinalizeTask` |
| `task_drawdown.py` | `HWMUpdateTask`, `DrawdownCircuitTask` |
| `task_gates.py` | `DrawdownGateTask`, `TransitionWindowTask`, `BEARBranchTask`, `VelocityCrashTask`, `EMA50GateTask` |
| `task_sell.py` | `PrepareHoldingTask`, `ScoreModelTask`, `EvaluateExitsTask` |
| `task_candidates.py` | `EarningsFilterTask`, `WashSaleFilterTask`, `BuildFeaturesTask`, `ScoreBuyTask`, `ScoreThresholdTask`, `RelativeStrengthTask`, `AssembleCandidateTask` |
| `task_ranking.py` | `BlendScoresTask`, `SortCandidatesTask` |
| `task_rotation.py` | `BuildPairsTask`, `ValidatePairsTask`, `EmitRotationsTask` |
| `task_selection.py` | `PrepareSelectionTask`, `RunSelectionTask`, `SizeAndEmitTask` |

renquant_104 additions under `kernel/panel_pipeline/`:

| File | Contents |
|------|----------|
| `panel_scorer.py` | `PanelScorer` — loads `artifacts/panel-ltr.json`, exposes `predict(feature_matrix)` |
| `feature_matrix.py` | `build_inference_feature_matrix` — stacks today's neutralized feature + factor rows into a single DataFrame keyed by ticker |
| `job_panel_scoring.py` | `PanelScoringJob` — 6 Tasks: `LoadScorerTask` → `BuildFeatureMatrixTask` → `ApplyScoresTask` → `VetoWeakBuysTask` → `LoadGlobalCalibrationTask` → `ApplyGlobalCalibrationTask` (+ optional `LoadNGBoostTask` → `ApplyNGBoostTask`) |

Universe + monitoring under `kernel/pipeline/`:

| File | Contents |
|------|----------|
| `job_universe.py` | `LoadUniverseJob` — 3 Tasks: `LoadArtifactsTask` → `FilterStalenessTask` → `FilterUniverseFloorTask`. Configurable floor (none/sharpe/ic) via `ranking.universe_floor`. Defensive tickers always exempt. One source of truth across LEAN / live / sim adapters. |
| `task_monitor.py` | `MonitorIdleStreakTask` — runs at end of `InferencePipeline`, tracks consecutive no-trade / no-candidate streaks; WARN when `monitoring.max_no_trade_days` is exceeded. |

`training/pipeline.py` is a thin re-export shim (notebook imports unchanged).

**Training pipeline classes** (defined in `kernel/pipeline/pp_training.py`):

| Class | Phase | Contents |
|-------|-------|----------|
| `DataFetchJob` | 1 (global) | Fetch OHLCV for all tickers |
| `RegimeFitJob` | 1 (global) | Fit GMM, build `final_regime` series |
| `FeatureJob` | 2 orchestrator | Dispatches `run_ticker_parallel()` for all per-ticker jobs |
| `TickerFeatureJob` | 2 (per-ticker) | Build labelled feature frame |
| `TickerTournamentJob` | 2 (per-ticker) | Train all model types, select best |
| `TickerExportJob` | 2 (per-ticker) | Write `models/` JSON artifacts |
| `TickerCalibrationJob` | 2 (per-ticker) | Write `score_calibration` to policy-metadata |
| `TournamentJob`, `ExportJob`, `CalibrationJob` | 2 no-ops | Skip-if-populated stubs for notebook backward compatibility |
| `CorrelationJob` | 3 (global) | Compute 120-day correlations, save artifact |

**Adapters** (`adapters/`):

| File | Contents |
|------|----------|
| `adapters/lean.py` | `LeanAdapter` — LEAN ↔ `InferenceContext` bridge |
| `adapters/runner.py` | `RunnerAdapter` — live runner ↔ `InferenceContext` bridge |

**LEAN entry point**: `main.py` is now ~200 lines. `OnData()` = 5 lines: `make_context → pipeline.run → commit → plot → debug`.

Three strategy-level artifacts live in `artifacts/` (not strategy root):
```
backtesting/renquant_103/artifacts/
  spy-gmm-regime.json          # GMM parameters for regime classifier
  watchlist-correlation.json   # 120-day pairwise correlations for correlation guard
  earnings-calendar.json       # Upcoming earnings dates per ticker
```

---

## Layer 3b: Live Trading

**Location**: `live/`
**Entry point**: `python -m live.runner --strategy <name> --broker paper|alpaca-paper|alpaca|ibkr --once`

The live runner loads the same model artifacts as LEAN but executes via broker API:

- `PaperBroker` — simulates fills locally for testing
- `AlpacaBroker` — connects to Alpaca Markets API for paper or live trading (requires `alpaca-py` and `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` env vars)
- `IBKRBroker` — connects to Interactive Brokers TWS/Gateway (stub, pending IBKR setup)
- Logs every signal and order to `live/logs/<strategy>/<date>.json`

**renquant_103 dispatch**: `live/runner.py` uses `RunnerAdapter` to build an `InferenceContext` from broker account state and parquet OHLCV, then runs `InferencePipeline` (full run) or `SellOnlyPipeline` (intraday sell-only). The legacy 900-line path remains for renquant_102.

For renquant_103, live trade records keep both `raw_model_score` and `rank_model_score`. Filtering, slot thresholds, and candidate ranking use `rank_model_score`; operator diagnostics and post-mortems can still inspect the original `raw_model_score`.

---

## Relative Indicator Framework

All features are computed relative to SPY to answer "is the stock outperforming the market?" rather than "is the stock going up?"

| Transform | When to use | Features |
|-----------|-------------|----------|
| **Ratio** (`stock / SPY`) | Always-positive indicators | `rsi`, `adx` |
| **Difference** (`stock - SPY`) | Zero-crossing indicators | `macd_hist`, `cci`, `bbp`, `williams_r`, `obv_slope` |
| **Absolute trend** | Price vs moving average | `trend` (price/50EMA), `trend_long` (price/200EMA) |
| **Relative momentum** | Relative price changes | `rel_mom_20d`, `rel_mom_60d` |

This means models learn relative patterns (outperformance/underperformance vs market), not absolute patterns that break in bull/bear regime changes.

---

## Trading Constraints

All models are subject to execution constraints during both notebook simulation and LEAN backtesting:

| Constraint | Value | Purpose |
|------------|-------|---------|
| Wash sale avoidance | 30 calendar days | Cannot buy within 30 days of selling (IRS wash sale rule) |
| Minimum hold | 20 days (all multi-stock strategies) | Prevents noise-driven model-signal exits during consolidation; stop-loss still triggers immediately |
| Maximum hold | 500 calendar days | Forces position review; allows long-term capital gains rate |

## Tax-Aware Returns

After-tax returns are computed at each sell event using configurable capital gains rates from the `tax` block in `strategy_config.json`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `short_term_rate` | 0.50 (50%) | Tax rate on gains held < `long_term_threshold_days` |
| `long_term_rate` | 0.32 (32%) | Tax rate on gains held ≥ `long_term_threshold_days` |
| `long_term_threshold_days` | 365 | Days to qualify for long-term rate |

Losses pass through untaxed (loss harvesting is not modeled). In notebooks, tax is deducted from cash at each sell, producing after-tax equity curves. LEAN strategies report tax as metadata via `SetRuntimeStatistic()` (LEAN equity stays gross). The analysis notebook uses `common.add_tax_columns()` to enrich LEAN trade data with per-trade tax breakdowns.

> **Note**: With `max_hold_days: 500`, trades held over 365 days qualify for the 32% long-term rate instead of 50% short-term.

## Position Sizing

Position sizing is configured in `strategy_config.json` under the `position_sizing` block and enforced during both notebook simulation and LEAN backtesting:

| Parameter | renquant_101/102 | renquant_103 | Purpose |
|-----------|-----------------|--------------|---------|
| `max_position_pct` | 0.30 (30%) | 0.15% (BULL_CALM/CHOPPY), 0.20% (BULL_VOLATILE), 0% (BEAR offensive) | Max single-stock exposure; scales with GMM confidence |
| `cash_reserve_pct` | 0.00 (0%) | 0% (BULL_CALM), 20% (BULL_VOLATILE), 30% (CHOPPY), 100% (BEAR) | Regime-adaptive cash cushion; scales with regime confidence and is deducted before sizing each buy |
| `max_concurrent_positions` | 5 | 8 | Max simultaneous holdings |

**Rules:**
1. **Cash-only buys** — only use available cash for new positions; never sell existing holdings to fund a new buy
2. **Max position cap** — `target_pct = min(max_position_pct, (available_cash - cash_reserve) / portfolio_value)`
3. **Whole shares only** — notebook simulation buys whole shares; LEAN uses `SetHoldings` which handles this internally

These rules are used by single-stock (renquant_101) and multi-stock (renquant_102, renquant_103) strategies. In multi-stock mode, each position is independently capped and the cash reserve is maintained across all positions.

---

## State Space

renquant_102 uses 7 shared relative indicator features for ML models (Classification, Q-Learning) per symbol. renquant_103 adds 4 regime-context features (see below):

| Feature | Transform | Description |
|---------|-----------|-------------|
| `rsi` | ratio | Relative Strength Index (stock/SPY) |
| `macd_hist` | diff | MACD histogram (stock − SPY) |
| `cci` | diff | Commodity Channel Index (stock − SPY) |
| `bbp` | diff | Bollinger Band Percentage (stock − SPY) |
| `adx` | ratio | Average Directional Index (stock/SPY) |
| `williams_r` | diff | Williams %R (stock − SPY) |
| `obv_slope` | diff | OBV rate of change (stock − SPY) |

The Manual model uses trend-following features instead (see Strategy Details below).
Q-Learning uses a subset of 5 indicator features (`rsi`, `macd_hist`, `cci`, `bbp`, `adx`) to keep state space tractable (5^5 × 3 = 9,375 states).

renquant_103 adds 4 regime-context features (computed from SPY, appended to the stock frame):

| Feature | Computation | Purpose |
|---------|------------|---------|
| `spy_realized_vol` | SPY 20d return std × √252 | Volatility regime signal |
| `spy_adx` | ADX(14) on SPY | Trend strength of the market |
| `spy_trend` | SPY close / SPY EMA50 | Market trend direction |
| `hurst_proxy` | Autocorr(lag=1) of SPY 20d returns | Fast momentum-persistence proxy |

All 12 registered indicators can be combined freely.

---

## Strategy Details

### renquant_102 — Multi-Stock Pre-Trained Scanner

A 3-stage pipeline strategy: **DETECT** → **CONFIRM** → **EXECUTE**.

**Notebook** (`renquant_102.ipynb`): Trains 3 approaches per symbol on a rolling 3-year window (`training_years=3`), picks the best by OOS after-tax Sharpe ratio, exports one model per symbol to `models/{SYMBOL}/` (minimum Sharpe floor: 0.8). After export, a portfolio-level simulation replicates the LEAN multi-stock logic in Python — scanning bullish volume z-scores, confirming with models, managing concurrent positions — and renders a 4-panel dashboard (equity vs SPY, drawdown, positions held, cash allocation). This enables parameter tuning (z-score threshold, lookback, position sizing) before running LEAN. The 3 approaches are:
1. Dual Momentum — trend-following ManualModel rules
2. Classification — BagLearner(RTLearner) random forest on relative features
3. Q-Learning — tabular RL with discretized trend features

Each symbol's best model may be a different type. The daily automation retrains all models via `daily_103.sh` (renquant_103 is the active live strategy). Models include a `trained_date` field; LEAN skips models older than `model_staleness_days` (default 60).

**Stage 1: DETECT** — compute adaptive per-stock volume score for each watchlist symbol. Default mode: **percentile** — triggers when today's volume is in the top 15% of the 20-day lookback window (P85). Legacy **zscore** mode (threshold 1.5σ) also supported via `volume_filter.mode` config. Bullish filter: only enter on up-close days.

**Stage 2: CONFIRM** — on bullish spike days, apply that stock's pre-trained model to 60-day relative feature history (indicators relativized vs SPY).

**Stage 3: EXECUTE** — if model says "buy" and all constraints pass, enter position. Max 5 concurrent positions.

**Risk management** (all configured in `strategy_config.json` under `risk`):
- **Stop-loss**: exit any position that falls 8% below entry price before waiting for model signal
- **Drawdown circuit breaker**: halt all new buys when portfolio is down ≥15% from its high-water mark
- **Regime filter**: configurable (currently disabled — relative features already encode market context vs SPY)
- **Sector guard**: max 3 concurrent positions per sector (prevents all 5 slots being correlated tech names)

**Config** uses `watchlist` (array of symbols) instead of `stock_symbol`:
```json
{
  "watchlist": ["TSLA", "AMZN", "GOOG", "MSFT", "AMD", "NFLX", "..."],
  "train_split": 0.70,
  "model_staleness_days": 60,
  "volume_zscore_lookback": 20,
  "volume_filter": {"mode": "percentile", "percentile_threshold": 85},
  "max_concurrent_positions": 5,
  "risk": {
    "stop_loss_pct": 0.08,
    "portfolio_drawdown_halt_pct": 0.15,
    "regime_filter": {"enabled": false, "symbol": "SPY", "sma_period": 200}
  },
  "sector_map": {"TSLA": "tech", "JPM": "finance", "...": "..."},
  "max_positions_per_sector": 3
}
```

**LEAN `main.py` flow** (`PreTrainedMultiStockStrategy`):
1. `Initialize()` — load pre-trained models from `models/{SYMBOL}/`, check staleness (log warning if <50% loaded), `AddEquity` for watchlist + SPY
2. `OnData()` — update portfolio high-water mark; process sells first (stop-loss check → max hold check → model signal + constraints); if drawdown circuit breaker triggered, skip all buys; check SPY regime filter; scan volume for non-held stocks (DETECT); rank candidates; apply sector guard; apply pre-trained model (CONFIRM); execute buys (EXECUTE)

**Live runner**: auto-detects multi-stock strategies by checking for `"watchlist"` in config. Uses `run_once_multi()` which checks sell signals (cumulative stop-loss, single-day loss gate, model signal), scans volume for new candidates, ranks them by model conviction score (`_get_model_score`), applies tiered thresholds (slot 1 easiest, each successive slot requires higher model score), sector guard, and executes buys.

### renquant_104 — Panel-LTR Cross-Sectional Ranking (active)

Successor to 103, now the active daily strategy. Inherits the entire 103 decision graph — regime detection, sell priority, buy gates, sector/wash-sale guards, rotation — and replaces per-ticker `rank_score` with a cross-sectional panel-LTR score. See full design: [`doc/arch/strategy-104.md`](renquant_104_design.md).

Key differences from 103:
- **Cross-sectional panel-LTR ranker**: single XGBoost ranker trained on the whole watchlist panel each day. `PanelScoringJob` (4 Tasks) is slotted between `CandidateJob` and `RankingJob` in `InferencePipeline`, and scores both candidates and current holdings so rotation/sizing compare apples-to-apples.
- **Hybrid scoring stack**: per-ticker tournament still picks champion Buy/Hold/Sell models, filtered by OOS Sharpe floor (`sharpe_floor=1.0`). Panel model is filtered by OOS mean IC at training time.
- **Panel-gated policies** (all optional, driven by `ranking.panel_scoring` block in `strategy_config.json`):
  - `buy_floor`: drops candidates whose panel score is below the floor (`VetoWeakBuysTask`)
  - `sizing.{enabled,floor,ceiling,min_mult}`: scales `max_position_pct` by a conviction multiplier derived from the panel score (applied in `SizeAndEmitTask` and `EmitRotationsTask`)
  - `rotation_advantage`: rotation pair is only emitted when the candidate's panel score beats the held position's by at least this margin
- **Training driver**: `scripts/train_104.py` → `FullTrainingPipeline` (Baseline → Panel → Recalibrate). No notebook dependency for training.
- **Inference lookback**: 520 bars (vs 60 for 103) — panel feature neutralization and factor windows need ≥504 days. Both `LeanAdapter` and `RunnerAdapter` pull the longer history and inject `config["_strategy_dir"]` so `LoadScorerTask` resolves the panel artifact relative to the strategy dir.

### renquant_103 — Adaptive Regime Multi-Stock (reference / rollback)

Still supported and usable for rollback. See full design: [`doc/arch/strategy-103.md`](renquant_103_design.md).

Key differences from 102:
- **3-layer regime detection** always running on SPY: Hurst (slow baseline) + CUSUM (fast transition trigger) + GMM (continuous confidence)
- **Regime-conditional entry**: momentum, capitulation, divergence, or blocked depending on regime
- **Stock selection**: earnings filter → volume scan → relative-strength ranking vs sector ETF → continuous model score → combined rank → correlation-aware selection
- **Regime-adaptive parameters**: all risk parameters (stop-loss, position size, cash reserve) adapt per regime and scale with GMM confidence; `max_hold_days` is 500 for most regimes and 40 in CHOPPY
- **Relative-outperformance labels**: Classification model trained with `close = stock_close / spy_close × 100`, so the 5-day forward return label measures stock outperformance vs SPY (not raw return), preventing bull-market always-buy bias
- **Sharpe floor**: 1.0 (raised from 0.8) — only high-conviction models pass
- **min_hold_days: 5** — model-sell blocked before 5 calendar days (stop-loss and single-day gate trigger immediately regardless); combined with the 3-consecutive-sell requirement, avg hold stays long enough to get LT tax treatment on winners
- **Consecutive-sell filter**: requires 3 consecutive daily sell signals before exiting a position; eliminates one-day noise flips that would otherwise trigger short-term tax events
- **Defensive tickers**: GLD, TLT, XLV, XLU — triggered as counter-cyclical buys in BULL_VOLATILE (SPY weak + defensive showing relative strength) and in BEAR regime (up to 1 defensive slot per portfolio while all offensive buys are blocked).
- **Trailing stop**: active in BULL_CALM after 20% gain from entry, trails at 18% below the position's rolling high-water mark — wide enough for high-beta tech corrections, locks in gains on large winners without capping upside.
- **BEAR regime**: all offensive buys blocked; up to 1 defensive position (GLD/TLT/XLV/XLU) allowed at 15% of portfolio. Existing positions held until stop-loss, max hold, or 3-consecutive-sell exit.
- **SPY EMA50 trend gate**: blocks all new offensive buys when SPY is below its 50-day EMA — prevents entering individual stocks during macro downtrends where technical signals are overwhelmed by market-wide selling.
- **Fixed training cutoff + expanding-window live models**: notebook trains per-symbol models on 2016–2023 for clean OOS validation (2024+). After export, models are retrained on the last 4 years up to today and re-exported for live trading — keeping live signals current without contaminating the backtest.
- **Live model score ranking**: simulation, LEAN, and live runner all rank candidates by today's actual model confidence (continuous score from `predict_score_bulk()`), not by static OOS Sharpe — ensures the highest-conviction signal on each day gets executed first.
- **Tiered thresholds**: each successive buy slot in a single day requires a progressively higher model score — slot 1: 0.10, slot 2: 0.30, slot 3: 0.50. Prevents overcommitting on low-conviction multi-candidate days. Configured in `tiered_thresholds` array in `strategy_config.json`; identical logic in LEAN, notebook simulation, and live runner.
- **Single-day loss gate**: in BULL_CALM, a position is exited if today's close drops ≥10% from yesterday's close (`max_single_day_loss_pct: 0.10`). Protects against gap-down days where a stock falls 15–20%+ in a single session before the 15% cumulative stop would fire on a daily bar. Disabled in other regimes (5% cumulative stop already tight).
- **Simulation output**: includes trade log (buys/sells, avg hold, avg pnl per trade, total tax, win rate, exit reason breakdown)

### renquant_101 — Single-Stock Classification

Trains a single model on relative indicators (stock vs SPY) for one symbol. The notebook trains 3 model types (Manual/Dual Momentum, Classification/RF, Q-Learning), compares them with stock and SPY buy-and-hold benchmarks, and exports the best by Sharpe ratio. Config uses `stock_symbol` (single string).

### Manual — Dual Momentum + Trend Following

Based on Gary Antonacci's Dual Momentum principles. Uses **trend-following features** where threshold checks have clear meaning, unlike oscillator thresholds that whipsaw:

| Rule | Feature | Buy condition | Sell condition | Rationale |
|------|---------|---------------|----------------|-----------|
| Absolute trend (50d) | `trend` | > 1.0 | < 0.97 | Price above 50-day EMA = uptrend |
| Absolute trend (200d) | `trend_long` | > 1.0 | < 0.97 | Price above 200-day EMA = structural uptrend |
| Relative momentum (20d) | `rel_mom_20d` | > 0.0 | < -0.03 | Stock outperforming SPY over 20 days |
| Relative momentum (60d) | `rel_mom_60d` | > 0.0 | < -0.05 | Stock outperforming SPY over 60 days |
| MACD confirmation | `macd_hist` diff | > 0 | < 0 | Stock momentum leads SPY |
| Volume confirmation | `obv_slope` diff | > 0 | < 0 | Accumulation exceeds market |

**Entry**: 4 of 6 rules agree bullish (trend + relative momentum + confirmations).
**Exit**: 3 of 6 rules agree bearish (trend break + momentum loss).

### Classification — Bagged Random Forest

Uses relative indicator features (7 for renquant_102; 11 for renquant_103, adding 4 SPY regime-context columns). Labels each day by N-day forward return vs a threshold — default: `lookahead=10, threshold=±4%`. renquant_103 overrides to `lookahead=5, threshold=±3%` and uses a **relative close price** (`stock_close / spy_close × 100`) so the label becomes the stock's relative outperformance vs SPY, not its raw return. This prevents the bull-market bias where every stock looks like a buy. BagLearner(RTLearner) ensemble with 15 bags and `leaf_size=25`. Buy/sell thresholds at ±0.1 on the raw tree output. The RF learns nonlinear relationships between relative features automatically — it effectively discovers crossover patterns and conditional logic from the data.

### Q-Learning — Tabular RL with Relative Reward

Uses 5 indicator features (`rsi`, `macd_hist`, `cci`, `bbp`, `adx`) with 5 bins = 9,375 states (5^5 × 3 holding buckets). Key design choice: **reward is relative price returns** (stock/SPY ratio changes), not raw stock returns. In a bull market, raw returns are always positive and the Q-learner learns "buy and never sell." Relative returns can go negative, giving the agent a reason to exit when the stock underperforms the market. Training uses a deterministic per-ticker seed (`abs(hash(ticker)) % 2^32`) for reproducible results across daily retraining runs.

---

## Data Flow Summary

```
yfinance / IBKR
       ↓
  Parquet cache (data/ohlcv/)
       ↓
  OHLCV DataFrames (stock + SPY)
       ↓
  Indicator registry (compute_indicators) × 2
       ↓
  Relative features (ratio or diff vs SPY)
       ↓
  Model training (Manual / RF / QL / FQI / Optimization)
       ↓
  JSON artifacts → backtesting/<strategy>/
       ↓
  ┌─────────────────────────────────┐
  │ LEAN backtest (Docker)          │
  │ Live trader (Alpaca / paper)    │
  └─────────────────────────────────┘
       ↓
  Analysis dashboard + normalized performance chart
  (stock vs SPY vs model equity)
```
