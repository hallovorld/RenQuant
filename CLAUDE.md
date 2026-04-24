# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RenQuant is a personal quantitative trading workstation for Apple Silicon. It implements a "Glass-box" pipeline: data ingestion → ML signal generation → backtesting (LEAN) → live trading (Alpaca/IBKR). All components are statistically interpretable and strictly decoupled.

## Environment Setup

Single conda environment (Miniconda, Apple Silicon arm64):

```bash
conda create -n renquant python=3.10
conda activate renquant
pip install pandas numpy matplotlib seaborn yfinance scikit-learn xgboost jupyterlab pyarrow
pip install "openbb[all]" openbb-cli backtesting scipy
pip install lean alpaca-py
lean login
```

Docker must be allocated 16GB+ memory for LEAN engine.

## Workflow: Four Modes

**Research mode** (fast iteration, no Docker): Run notebooks to train and export models.

**Validation mode** (final backtest): Run LEAN on exported models.

```bash
# Single symbol export
python scripts/export_lean_data.py --symbol NVDA

# Batch export: all watchlist symbols for a strategy (recommended before first backtest)
python scripts/export_lean_watchlist.py --strategy renquant_102
python scripts/export_lean_watchlist.py --strategy renquant_102 --force  # re-export all
python scripts/export_lean_watchlist.py --strategy renquant_102 --symbols CRM SHOP  # specific symbols

cd backtesting/renquant_101
lean backtest .
```

**IMPORTANT**: LEAN uses its own data files at `backtesting/data/equity/usa/daily/`, NOT the yfinance parquet cache at `data/ohlcv/`. After training models in a notebook, always run `export_lean_watchlist.py` before backtesting to ensure LEAN has data for all watchlist symbols. Missing data causes silent failures (History() returns empty, no trades execute).

To backtest and render charts in one step (with notifications):

```bash
python scripts/backtest_and_analyze.py --strategy renquant_101
python scripts/backtest_and_analyze.py --strategy renquant_101 --ntfy other  # custom ntfy topic
python scripts/backtest_and_analyze.py --strategy renquant_101 --silent      # no notifications
```

Notifications (on by default, `--silent` to disable): macOS banner via `terminal-notifier` (`brew install terminal-notifier`) and iPhone push via [ntfy.sh](https://ntfy.sh) (default topic: `renquant`, override with `--ntfy <topic>`).

**Analysis mode**: Run `python scripts/analyze_backtest.py --strategy renquant_101` to visualize LEAN results, including decision telemetry for score/threshold inspection.

**Live mode**: Run the live trader with paper, Alpaca, or IBKR broker.

```bash
python -m live.runner --strategy renquant_101 --broker paper --once
python -m live.runner --strategy renquant_102 --broker alpaca-paper --once  # multi-stock
python -m live.runner --strategy renquant_102 --broker alpaca --once  # real money
```

**Scheduled mode**: Five automation runs via macOS launchd (all NYSE-holiday-aware). renquant_104 is the **active strategy**; 103 scripts remain for reference and rollback.

| Run | Time (PT) | Time (ET) | Script | What it does |
|-----|-----------|-----------|--------|--------------|
| Market open | 6:32 AM | 9:32 AM | `live_only_104.sh --sell-only` | Exit stop-loss / gap-down positions early using today's opening price |
| Intraday | 7:00 – 12:30 every 30min | 10:00 – 15:30 | `intraday_sell_104.sh --sell-only --intraday` | Alpaca IEX 5-min overlay → trigger stop-loss / trailing-stop / SDL mid-session (never places buys) |
| Pre-close | 12:44 PM | 3:44 PM | `live_only_104.sh --sell-only` | Exit intraday stop breaches before close using near-final daily price |
| After close | 1:55 PM | 4:55 PM | `daily_104.sh` | Full run: `FullTrainingPipeline` (cadence-gated) → export LEAN data → buy + sell signals |
| Sunday retrain | Sun 10:00 AM | Sun 13:00 ET | `retrain_panel.sh` | Force-retrain (cadence bypass) when daily_104 would otherwise skip the weekend |

```bash
# Manual runs — 104 (active)
bash scripts/daily_104.sh              # full retrain + trade
bash scripts/live_only_104.sh          # intraday sell check (no retrain)
python scripts/train_104.py --skip-baseline --skip-recalibrate  # partial retrain
python -m live.runner --strategy renquant_104 --broker alpaca --once --sell-only

# LaunchAgents (104 active; 103 unloaded)
# ~/Library/LaunchAgents/com.renquant.open104.plist
# ~/Library/LaunchAgents/com.renquant.intraday104.plist      (NEW: 20 slots 07:00-12:30 Mon-Fri)
# ~/Library/LaunchAgents/com.renquant.preclose104.plist
# ~/Library/LaunchAgents/com.renquant.daily104.plist
# ~/Library/LaunchAgents/com.renquant.retrain-panel104.plist (NEW: Sun 10:00)
# Logs: logs/live_104/{date}-open.log, {date}-preclose.log
#       logs/intraday_104/{date}.log
#       logs/daily_104/{date}.log
#       logs/retrain_panel/{date}.log
# NYSE calendar guard: all scripts skip US market holidays automatically

# Manual runs — 103 (legacy, kept for rollback)
bash scripts/daily_103.sh
bash scripts/live_only_103.sh
python -m live.runner --strategy renquant_103 --broker alpaca --once --sell-only
```

Alpaca credentials are stored in `.env` (gitignored): `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. Notifications (macOS + iPhone/ntfy) are sent on success or failure. Notification body includes current holdings with unrealized P&L percentages (e.g. `BUY AAPL x5 | Held: NVDA+12% META-2%`).

## Shared Library: `common/`

Import as `import common`. **Do not import `common/` from inside `backtesting/` — LEAN Docker cannot access it.**

| Module | Key exports |
|--------|-------------|
| `common/config.py` | `load_strategy_config`, `build_model_path` |
| `common/data/` | `fetch_ohlcv` (with Parquet cache), `DataSource` ABC, `LocalStore` |
| `common/indicators/` | `compute_indicators`, `add_indicators` (compat), `list_indicators`, `@register`; regime detection: `compute_hurst`, `rolling_hurst`, `compute_cusum`, `rolling_cusum`, `build_gmm_features`, `RegimeGMM` |
| `common/models/` | `BaseModel`, `ManualModel`, `ClassificationModel`, `QLearningModel`, `FQIModel`, `OptimizationModel`, `XGBoostModel`, `create_model`; all models implement `predict_score_bulk(df)` returning native continuous float scores, and `common/models/scoring.py` calibrates them for live cross-model ranking |
| `common/models/learners/` | `RTLearner`, `BagLearner`, `TabularQLearner` |
| `common/strategy.py` | `StrategyConfig`, `Strategy` |
| `common/portfolio.py` | `compute_portvals`, `portfolio_stats` |
| `common/tax.py` | `compute_trade_tax`, `compute_after_tax_pnl`, `load_tax_config`, `add_tax_columns`, `tax_rate_for_holding` |
| `common/plotting.py` | `backtest_dashboard`, `plot_normalized_performance`, parse/plot helpers |

## Architecture

### Data Layer (`common/data/`)

- `DataSource` ABC with `YFinanceSource` (working) and `IBKRSource` (stub)
- `LocalStore` caches OHLCV as Parquet at `data/ohlcv/{SYMBOL}/1d.parquet`
- `fetch_ohlcv()` checks cache first, fetches missing dates, saves locally

**Two separate data stores** — research and LEAN use different paths:

| Store | Path | Format | Used By |
|-------|------|--------|---------|
| Parquet cache | `data/ohlcv/{SYMBOL}/1d.parquet` | Parquet | Notebooks, live runner |
| LEAN data | `backtesting/data/equity/usa/daily/{symbol}.zip` | LEAN CSV zip | LEAN backtester (Docker) |

`export_lean_data.py` (single) or `export_lean_watchlist.py` (batch) bridge the two by converting parquet → LEAN format. Both also generate required map files and factor files.

### Indicator Registry (`common/indicators/`)

Uniform API: `(df: DataFrame, **params) -> DataFrame`. All indicators registered via `@register` decorator.

- Momentum: `rsi`, `macd`, `ema`, `momentum`, `williams_r`
- Volatility: `cci`, `bbp`, `stochastic`, `ppo`, `atr`
- Trend: `adx`
- Volume: `obv`
- Regime detection (not registered, used directly): `compute_hurst` / `rolling_hurst` (Hurst exponent), `compute_cusum` / `rolling_cusum` (changepoint detection), `build_gmm_features` / `RegimeGMM` (GMM classifier, serialises to JSON)
- `compute_indicators(df, {"rsi": {"period": 14}, "macd": {}})` applies any subset

### Model Types (`common/models/`)

All implement `BaseModel` ABC: `train()`, `predict()`, `predict_bulk()`, `predict_score_bulk()`, `save()`, `load()`. JSON artifacts only.

| Type | Training | Prediction | `predict_score_bulk()` |
|------|----------|------------|------------------------|
| `manual` | No-op (generic score_rules at construction) | Multi-indicator threshold voting | Raw vote count (positive = buy pressure, negative = sell pressure) |
| `classification` | Forward-return labels → BagLearner(RTLearner) | Forest query | Raw BagLearner continuous output |
| `qlearning` | Discretize states, Q-table over epochs | Q-table argmax | Q(buy)−Q(sell) per row |
| `fqi` | Transition tuples → FQI with XGBoost | Score per action, argmax | Not implemented (FQI not used in 103) |
| `optimization` | Nelder-Mead over indicator params + inner model | Delegate to best inner | Delegates to inner model |
| `xgboost` | Forward-return labels → two XGBClassifier (buy-vs-rest, sell-vs-rest) with L1/L2 regularisation | P(buy)−P(sell) net score | P(buy)−P(sell) continuous score |

### Pipeline

**1. Research** (Notebooks) — `renquant` conda env
- `common.fetch_ohlcv` → `common.compute_indicators` → model.train() → model.save()
- Strategy-essential logic (gate rules, transitions) stays in notebooks

**2. Model Artifacts** (`backtesting/{strategy}/*.json` or `models/{SYMBOL}/*.json`)
- JSON (not pickle) for LEAN compatibility
- `*-policy-metadata.json` is the contract between research and execution; it may also carry optional `score_calibration` metadata for live ranking
- Single-stock (101): artifacts at strategy root; multi-stock (102): `models/{SYMBOL}/` subdirectories

**2b. LEAN Data Export** (required before backtesting)
- `python scripts/export_lean_watchlist.py --strategy renquant_102`
- Converts parquet cache → LEAN daily zips + map files + factor files
- Skips symbols already exported (use `--force` to re-export)
- Must re-run after adding new symbols to a watchlist
- Both export scripts check `data/ohlcv/{SYMBOL}/` first, then fall back to `Notebooks/data/ohlcv/{SYMBOL}/` (notebook kernel's working directory). If both are missing, run `python -c "import common; common.fetch_ohlcv('SYMBOL')"` from the repo root.

**3. Backtesting** (`backtesting/`) — QuantConnect LEAN engine (Docker)
- `main.py`: self-contained `QCAlgorithm` (no `common/` dependency)
- Loads JSON models, recomputes indicators inline
- Enforces trading constraints (wash sale, min/max hold, stop-loss), position sizing, portfolio drawdown circuit breaker, SPY regime filter, and sector concentration guard — all from `strategy_config.json`
- Reports after-tax metrics via `SetRuntimeStatistic()` when `tax` config is present
- Single-stock strategies (renquant_101): one symbol per backtest
- Multi-stock strategies (renquant_102): volume z-score scanner, loads pre-trained models per symbol from `models/{SYMBOL}/`, max N concurrent positions
- Adaptive regime strategies (renquant_103): 3-layer regime detection (Hurst + CUSUM + GMM), regime-conditional entry direction, relative-strength ranking, correlation-aware selection, earnings filter, defensive tickers

**4. Live Trading** (`live/`)
- `python -m live.runner --strategy X --broker paper|alpaca-paper|alpaca|ibkr --once`
- `PaperBroker` for testing, `AlpacaBroker` for real/paper trading, `IBKRBroker` (stub)
- Auto-detects single-stock vs multi-stock strategies (presence of `watchlist` in config)
- Logs to `live/logs/{strategy}/{date}.json`
- **renquant_103 pipeline**: `live/runner.py` uses `RunnerAdapter` to build an `InferenceContext` from broker account state and parquet OHLCV, then runs `InferencePipeline` (full run) or `SellOnlyPipeline` (intraday sell-only). The legacy 900-line path is retained for renquant_102.

**5. Analysis** (`scripts/analyze_backtest.py`)
- `common.backtest_dashboard` — price, decision telemetry, equity, drawdown, and stats
- `common.plot_normalized_performance` — normalized equity with entry markers
- Multi-stock trade detail table (CSV + console) with per-symbol colored markers

### Pipeline Package (renquant_103)

Two 3-phase parallel pipelines shared by the live runner, LEAN, and the notebook.

All pipeline components live in `kernel/pipeline/` as a flat layout:
`pp_*` = pipeline orchestrators, `job_*` = sequential task chains, `task_*` = atomic steps.
This lets pipelines share jobs and jobs share tasks without subdirectory plumbing.

**InferencePipeline** (`kernel/pipeline/`):

| Module | Contents |
|--------|----------|
| `pipeline/context.py` | `InferenceContext` (~50 fields), `TickerInferenceContext` (per-ticker slice) |
| `pipeline/pipeline.py` | `Task`, `Job`, `TickerJob` ABCs + `run_parallel()` |
| `pipeline/pp_inference.py` | `InferencePipeline`, `SellOnlyPipeline` |
| `pipeline/job_regime.py` | `RegimeJob` (Phase 1) |
| `pipeline/job_drawdown.py` | `DrawdownJob` (Phase 1) |
| `pipeline/job_gates.py` | `BuyGatesJob` (Phase 1) |
| `pipeline/job_sell.py` | `TickerSellJob` (Phase 2a, per-ticker parallel) |
| `pipeline/job_candidates.py` | `TickerCandidateJob` (Phase 2b, per-ticker parallel) |
| `pipeline/job_ranking.py` | `RankingJob` (Phase 3) |
| `pipeline/job_selection.py` | `SelectionJob` (Phase 3) |
| `pipeline/task_*.py` | Atomic tasks per concern (regime, drawdown, gates, sell, candidates, ranking, selection) |

**TrainingPipeline** (`kernel/pipeline/pp_training.py` — `training/pipeline.py` is a re-export shim for notebook imports):

| Class | Phase | Contents |
|-------|-------|----------|
| `DataFetchJob` | 1 (global) | Fetch OHLCV for all tickers |
| `RegimeFitJob` | 1 (global) | Fit GMM, build regime series |
| `FeatureJob` | 2 orchestrator | Dispatches `run_ticker_parallel()` for all per-ticker jobs |
| `TickerFeatureJob` | 2 (per-ticker) | Build labelled feature frame |
| `TickerTournamentJob` | 2 (per-ticker) | Train all model types, select best |
| `TickerExportJob` | 2 (per-ticker) | Write `models/` JSON artifacts |
| `TickerCalibrationJob` | 2 (per-ticker) | Write `score_calibration` to policy-metadata |
| `CorrelationJob` | 3 (global) | Compute 120-day correlations, save artifact |

### Strategies

**Single-stock** (renquant_101): One symbol, one model. Config uses `stock_symbol`.

**Multi-stock pre-trained scanner** (renquant_102): 3-stage pipeline (DETECT → CONFIRM → EXECUTE). See also `Notebooks/renquant_102.ipynb`. Notebook trains 3 approaches per symbol (Dual Momentum, Classification/RF, Q-Learning) on a 70/30 walk-forward train/test split of a rolling 3yr window (`training_years=3`), exports best model per symbol to `models/{SYMBOL}/` by OOS after-tax Sharpe (floor: 0.8). Orphan model directories (symbols removed from watchlist) are automatically purged on each export. Both OOS Sharpe (`sharpe`) and in-sample Sharpe (`sharpe_is`) are stored in `policy-metadata.json` for diagnostics. The notebook also includes a portfolio-level simulation that mirrors the LEAN multi-stock logic, enabling parameter tuning in Python before running LEAN. LEAN loads pre-trained models, scans watchlist for bullish volume spikes (percentile mode P85 by default, up-close days only), applies that stock's model for confirmation, and enforces: per-position 8% stop-loss, 15% portfolio drawdown circuit breaker (halts new buys), configurable regime filter (currently disabled — relative features already encode market context vs SPY), and sector concentration guard (max 3 positions per sector). Models include `trained_date`; LEAN skips models older than `model_staleness_days` (default 60). Config uses `watchlist` array (21 stocks + ETFs), `train_split`, `model_staleness_days`, `volume_filter`, `max_concurrent_positions`, `risk` (stop-loss, drawdown halt, regime filter), `sector_map`, `max_positions_per_sector`. Notebook-only config fields (not read by LEAN): `sample_start`/`sample_end` (defines full data window for indicator warmup), `training_years` (rolling training window size), `indicator_spec` (indicator hyperparameters), `model_params` (model hyperparameters).

**Adaptive regime multi-stock** (renquant_103): Extends 102 with a 3-layer regime detector. Layer 1: rolling Hurst exponent (63-day window, configurable thresholds: `hurst_trending_threshold=0.65`, `hurst_reversion_threshold=0.52`; characterises momentum vs mean-reversion). Layer 2: CUSUM changepoint detection (fast transition trigger, uncertainty window after each break). Layer 3: GMM clustering on 4 SPY features (10d return, 20d realised vol, ADX, return autocorr) — outputs continuous P(regime) to scale position size. Four regimes: BULL_CALM (momentum entry, 15% max position), BULL_VOLATILE (capitulation entry on high-vol down-close, 20%), CHOPPY (divergence entry — stock outperforming SPY, 15%), BEAR (offensive buys blocked; 1 defensive slot for GLD/TLT/XLV/XLU at 15% of portfolio). Stock selection pipeline: earnings filter (±3d) → SPY EMA50 trend gate (blocks all new buys when SPY < 50-day EMA) → SPY velocity crash filter (blocks buys if SPY down >3% in last 3 days) → transition uncertainty window (no buys for 3 bars after CUSUM changepoint) → BEAR branch (defensives only) → model buy signal (entry trigger, no separate volume-scan gate) → calibrated `rank_score` filter (regime-aware: 0.10 BULL_CALM, 0.15 others) → relative-strength score vs sector ETF → combined rank (data-driven blend weights from `ranking.blend_weights` in strategy_config.json, config-driven at runtime and refreshed daily by `scripts/recalibrate_scores.py`) → selection loop: tiered threshold escalation (slot 1: 0.10, slot 2: 0.30, slot 3: 0.50, all on calibrated `rank_score`) → wash-sale guard → sector guard (max 3 per sector) → correlation-aware greedy selection (threshold 0.70). New artifacts (in `artifacts/` subdir): `spy-gmm-regime.json`, `watchlist-correlation.json`, `earnings-calendar.json`. Notebook saves artifacts to `STRATEGY_DIR/artifacts/`; DataJob checks `artifacts/` first with fallback to strategy root. Watchlist (24): adds GLD, TLT, XLV, XLU as defensive counter-cyclical tickers. Notebook: `Notebooks/renquant_103.ipynb`. Earnings calendar refreshed via `python scripts/fetch_earnings_calendar.py --strategy renquant_103`. Key implementation details: (1) Classification model receives `close = stock_close / spy_close × 100` so forward-return labels become relative outperformance vs SPY (prevents bull-market always-buy bias); (2) BEAR hard override: `spy_20d_ann_vol > 0.35` or `spy_20d_cumret < -0.08` forces BEAR regime regardless of GMM output (GMM reacts too slowly to macro shocks like tariff events); thresholds are config-driven via `bear_vol_threshold` and `bear_return_threshold`; (3) `predict_bulk()` returns string Series — map with `{"buy": 1, "hold": 0, "sell": -1}` for numeric use; (3) `predict_score_bulk()` returns native continuous float scores: Classification returns raw BagLearner output, QLearning returns Q(buy)−Q(sell), XGBoost returns P(buy)−P(sell), Manual returns raw vote count; these are not cross-model comparable, so notebook export, LEAN, and the live runner calibrate them through `common.models.scoring` into a common `rank_score` (P(outperform SPY by threshold% in lookahead days)) using metadata `score_calibration` when present or a runtime fallback; calibration method is selected by sample size: isotonic for n≥300, Platt scaling (sigmoid) for 120≤n<300, constant base-rate for n<120; `score_calibration` is refreshed daily by `scripts/recalibrate_scores.py` after each model retrain; (4) blend weights are refreshed daily by `scripts/recalibrate_scores.py` using a logistic regression on `[norm(rank_score), norm(rs_score)]`, with positive coefficients normalised into `ranking.blend_weights`; (5) OOS Sharpe floor is 1.0 (raised from 0.8) — marginal models below 0.8 are excluded; (6) `max_hold_days` is 500 for BULL_CALM/BULL_VOLATILE/BEAR, 40 for CHOPPY (raised from 23 to accommodate min_hold_days=30 + 3 consecutive sell signals); (7) position sizing: 8 concurrent positions, with both `max_position_pct` and `cash_reserve_pct` scaled by regime confidence before sizing each entry as `min(cash - reserve, portfolio × scaled_max_pct)`; (8) `min_hold_days: 30` — model-sell blocked before 30 days (stop-loss and single-day loss gate trigger immediately); (9) 3 consecutive daily sell signals required before model exit; (10) all three model types (Classification, QLearning, XGBoost) use `abs(hash(ticker)) % 2^32` as a per-ticker seed for reproducibility; (11) tournament training: trains Classification + QLearning + Manual + XGBoost per ticker, exports best by OOS Sharpe; (12) BULL_CALM trailing stop: 20% gain trigger, 18% trail below high-water mark; stop uses peak gain (HWM-based) so it stays armed after pullbacks; (13) exit priority order (same in LEAN, notebook, live runner): trailing stop → cumulative stop-loss (15% BULL_CALM, 5% others) → single-day loss gate (10% BULL_CALM, disabled others) → max hold → model sell streak; (14) `max_single_day_loss_pct: 0.10` in BULL_CALM — exits when today's close drops ≥10% from yesterday's close; protects against gap-down days where cumulative stop fires too late on daily bars; (15) fixed OOS cutoff 2024-01-01 for stable simulation; live models retrained on last 4 years of data (not all history) for relevance; (16) notebook simulation, LEAN, and the live runner are aligned on calibrated `rank_score`, config-driven blend weights, tiered thresholds, wash-sale rechecks, and confidence-scaled sizing; (17) live runner uses per-model `feature_columns` from policy-metadata.json, not global config (regime-context columns only exist during notebook training); (18) tiered thresholds configured in `tiered_thresholds` array in `strategy_config.json` — identical slot logic across components using `tier_idx = min(slots_filled, len(tiers) - 1)`; (19) `tests/test_runner_ranking.py` covers calibrated rank scoring, mixed-model ordering, score logging, tier application, and blend estimation; (20) `tests/test_strategy_ledger_parity.py` replays a synthetic multi-day buy ledger to keep notebook-like and LEAN-like selection ledgers aligned; (21) 600 tests are currently collected under `tests/` (598 passed, 2 skipped: kernel unit tests including rotation, pipeline tests, training module tests, policy alignment tests including TestRotationAlignment); (25) wash-sale is re-checked in the selection loop (not just the scan phase) to catch same-day sell-then-buy attempts — matches notebook and LEAN; (26) live runner fetches sector ETFs (XLK, XLI, etc.) in addition to the watchlist so RS scores are computed for all sectors — notebook fetches WATCHLIST ∪ sector_etf_map.values() ∪ {SPY}; (27) CHOPPY regime confidence uses Hurst distance from reversion threshold (`(hurst_rev-H)/(hurst_rev-choppy_hurst_floor)`, hurst_rev=0.52, floor=0.20 from config) instead of GMM probability — GMM's two main clusters have ~48/48% weights making it structurally unable to produce meaningful CHOPPY confidence; BULL/BEAR regimes still use GMM posterior probability; (22) oversize fallback: if max_position_pct × confidence can't cover 1 share, retry at 25% cap — prevents high-priced stocks (e.g. LLY) from being silently skipped; (23) OHLCV freshness guard: live runner checks last date of each symbol's parquet cache before model inference, force-refreshes via yfinance if >5 trading days old; (24) duplicate-order guard: checks Alpaca open orders before any BUY; skips if a pending order exists for that symbol; (28) cross-sectional rotation: `RotationJob` (between `RankingJob` and `SelectionJob`) lets held positions compete with new candidates on the same calibrated `rank_score` every bar via `kernel.rotation.find_rotation_pairs()`. Tax-adjusted swap_margin uses `effective_swap_margin = base_margin + tax_drag(unrealized_pnl, hold_days, ST/LT rate)`; positions within `lt_protection_days` of the long-term threshold sitting on a gain are pinned (+inf) to preserve the upcoming LT discount. Each pair is re-validated against wash-sale, sector, and correlation guards on the virtual post-swap holdings set, then emits `ExitSignal(exit_type="rotation")` plus a sized buy. Skipped entirely in BEAR regime. Notebook simulation mirrors via the same `kernel/rotation.py` primitive; configured under `rotation` block in `strategy_config.json` (`enabled`, `swap_margin`, `min_rotation_hold_days`, `lt_protection_days`, `max_rotations_per_bar`).

Volume filter supports two modes via `volume_filter.mode` in `strategy_config.json`:
- `"percentile"` (default): Adaptive per-stock filter. Triggers when today's volume is in the top N% of the lookback window (default: P85 = top 15%). Self-calibrates for each stock's own volume distribution, so stable large-caps and volatile small-caps are held to the same relative standard.
- `"zscore"` (legacy): Fixed z-score threshold across all stocks. Uses `volume_zscore_threshold` (default: 1.5).

**Panel-LTR cross-sectional ranking** (renquant_104): Extends 103 by replacing each candidate's per-ticker `rank_score` with a cross-sectional panel-LTR score computed via a single XGBoost learning-to-rank model fitted on the whole watchlist panel. Enabled via `ranking.panel_scoring.enabled: true` (config block also includes `artifact_path`, `nan_prone_cols`). Panel training labels use beta-neutralized + sector-/size-neutralized forward excess returns; `PanelScoringJob` (four Tasks: `LoadScorerTask` → `BuildFeatureMatrixTask` → `ApplyScoresTask` → `VetoWeakBuysTask`) is slotted between `CandidateJob` and `RankingJob` in `InferencePipeline`, and short-circuits via `should_skip()` when the flag is off so 103 behavior is preserved. `ApplyScoresTask` writes `panel_score` onto both candidates (and overwrites `rank_score`) and holdings so rotation can compare apples-to-apples. Three additional knobs under `ranking.panel_scoring`: `buy_floor` drops candidates below a panel-score threshold (`VetoWeakBuysTask`), `sizing.{enabled,floor,ceiling,min_mult}` drives `conviction_multiplier()` which scales `max_position_pct` in `SizeAndEmitTask` + `EmitRotationsTask`, and `rotation_advantage` requires a candidate's panel score to beat a held position's by at least that fraction before a rotation pair is emitted. Training is driven by `scripts/train_104.py` (thin entrypoint) backed by `kernel/pipeline/pp_training_full.py::FullTrainingPipeline` (`BaselineTournamentJob` → `PanelTrainingJob` → `RecalibrationJob`) — no notebook dependency. Inference requires 520 bars of history (vs 60 for 103) to warm up neutralization + factor windows; both `LeanAdapter` (LEAN main.py path) and `RunnerAdapter` (live runner path) read the flag and call `prepare_inference_panel_frames()` before running the pipeline, and each adapter injects `config["_strategy_dir"]` so `LoadScorerTask` can resolve the artifact path from the strategy directory (not CWD). The 104 notebook lives at `backtesting/renquant_104/renquant_104.ipynb` (parallel to the 103 notebook layout). Full spec: `doc/renquant_104_design.md`.

**renquant_104 Stage 2 — NGBoost μ,σ head** (default off): `training_panel/ngboost_head.py` fits a separate NGBoost Normal(μ, σ) regressor on raw (pre-Gaussianized) residual forward returns. When `ranking.panel_scoring.ngboost.enabled` is true, `PanelScoringJob` appends `LoadNGBoostTask` + `ApplyNGBoostTask` (6-task chain total): μ,σ are written to both `CandidateResult` and `HoldingState`, and in the default `score_mode=mu_minus_lambda_sigma` the combined score `μ − λσ` overrides `rank_score` + `panel_score` (set `score_mode=additive` to leave rank_score untouched and only populate μ,σ for sizing). `ranking.panel_scoring.sigma_sizing.{enabled,floor,ceiling}` drives `sigma_multiplier()` which scales `max_position_pct` by `σ_median / σ_i` (clipped) in both `SizeAndEmitTask` and `EmitRotationsTask`. Training adds `PanelNGBoostJob` as Phase 5 of `PanelTrainingPipeline`. Artifact is a self-contained JSON with a base64-encoded pickle (ngboost has no pure-JSON serializer).

**renquant_104 Stage 3.1 — fundamental factors** (now wired + enabled): `kernel/fundamentals.py` caches OpenBB snapshots at `data/fundamentals/{SYMBOL}.parquet` (columns: `earnings_yield, roe, gross_profitability, book_to_price`). `scripts/fetch_fundamentals.py` is the watchlist driver. A new `LoadFundamentalsTask` runs inside `PanelDataJob` (Phase 1 of `PanelTrainingPipeline`) and populates `PanelTrainingContext.fundamentals`. `TickerPanelFactorJob` broadcasts the per-ticker scalar factors into `raw_factor_frame`; `FactorZScoreTask` cross-sectionally z-scores them with sector-median fill for missing values. Training artifact now has 24 feature columns (16 neutralized indicators + 4 technical factor z-columns + 4 fundamental z-columns). Enabled via `panel_ltr.fundamentals.enabled: true`. Time-invariant snapshot in this release; extending to point-in-time time-series is a future change.

**renquant_104 Stage 1 cleanups** (all behind flags, defaults preserve existing behaviour):
- `training.cadence` — `"daily"` (default, runs every weekday), `"weekly"` with `training.weekly_weekday: 6`, or `"custom"` with `training.allowed_weekdays: [1, 3, 6]` (Python weekdays, Mon=0…Sun=6) to short-circuit `FullTrainingPipeline.run()` on non-cadence days. **104 is configured with `cadence: "custom"` + `allowed_weekdays: [1, 3, 6]` → Tue/Thu/Sun training.** `daily_104.sh` fires Mon-Fri 1:55 PM PT and trades daily (cadence gate skips training on Mon/Wed/Fri); `retrain_panel.sh` fires Sunday 10 AM PT via `com.renquant.retrain-panel104.plist` and forces a retrain (no trading, market closed). `scripts/train_104.py --force` bypasses the gate for manual runs.
- `ranking.tournament.exclude_models` — e.g. `["qlearning"]` drops that approach from `run_tournament_all`. Default is empty (all four approaches run).
- rs_score is retired from ranking math: `BlendScoresTask` hardcodes `(w_rank, w_rs) = (1.0, 0.0)` and logs a warning if a legacy `ranking.blend_weights` with non-zero rs weight is found. `recalibrate_scores.py` no longer writes that key (but `_compute_blend_weights` remains for offline diagnostics). `rs_score` is still populated on `CandidateResult` for logs.
- **Universe admission** is consolidated into `kernel/pipeline/job_universe.py::LoadUniverseJob` and driven by `ranking.universe_floor.{type, threshold}`. `type` ∈ `{"none", "sharpe", "ic", ...}` (default `"none"` — admit all models that load). Sharpe reads `live_holdout_sharpe`/`sharpe` from each model's policy-metadata; IC reads `panel_oos_ic`. New floor types register themselves by adding an entry to `FLOOR_EVALUATORS`. LEAN (`main.py`), the live runner (`live/runner.py`), and the notebook sim (`adapters/sim.py`) all call `LoadUniverseJob` — the hand-written per-adapter filter loops have been deleted (test enforcement: `tests/test_universe_alignment.py::TestAdapterParity::test_no_hand_written_filter_loops_remain`).
- **Legacy sim loop deleted**: `sim/runner.run_backtest` now always drives `SimAdapter + InferencePipeline`. The legacy hand-written loop (`_run_backtest_legacy`) and its helpers (`swap_in_panel_scores`, `apply_ngboost_head`, `_GlobalCalibAdapter`) are gone. LEAN, live, and sim share one decision-logic source of truth (`InferencePipeline`); `run_backtest_via_pipeline` is retained as a back-compat alias for `run_backtest`.
- **Panel Round 1-5 improvements** (OOS IC 0.038 → 0.066, CPCV 15-split):
  - *Round 1*: fixed calibration bug (`ApplyGlobalCalibrationTask` used to be a no-op whenever NGBoost enabled). **Task-#2 refactor (2026-04-23):** calibration now ALWAYS runs, regardless of score_mode. `PanelScoringJob` order reordered so `LoadNGBoost` + `ApplyNGBoost` execute BEFORE `LoadGlobalCalibration` + `ApplyGlobalCalibration`. In `mu_minus_lambda_sigma` mode, NGBoost overwrites `panel_score` with μ−λσ first, then the calibrator maps it to a probability via the same isotonic head. This unlocks σ-aware ranking as a live-testable option (raw μ−λσ previously stayed below the 0.10 tier threshold → zero trades). Also fixed inference z-score parity (`prepare_inference_panel_frames` and `scripts/fit_panel_calibrator.py` both call `NeutralizedFeatureZScoreTask` + `LoadFundamentalsTask` so LEAN/live/sim see the same feature distribution the panel was trained on).
  - *Round 2*: stronger regularisation + CPCV + monotone constraints on economically-signed factors (`beta_60d_z`:-1, `mom_12_1_z`:+1, `resid_mom_z`:+1, `earnings_yield_z`:+1, `roe_z`:+1, `gross_profitability_z`:+1).
  - *Round 3*: 5 orthogonal factor functions in `training_panel/factors.py` — `compute_amihud_illiquidity`, `compute_volume_shift`, `compute_price_to_high`, `compute_realized_vol`, `compute_drawdown_from_peak`. Weak ones dropped via `panel_ltr.drop_cols`. Strong new factors get monotone constraints too.
  - *Round 4*: `short_pct_float` from yfinance (`.info.shortPercentOfFloat`) added to `FUNDAMENTAL_COLS` with monotone -1.
  - *Round 5*: `earnings_surprise_cum` (yfinance `.earnings_dates` → trailing-4Q cum surprise %) via `LoadEarningsSurpriseTask` in PanelDataJob + `compute_earnings_surprise_cum` in `kernel/earnings_surprise.py`; SEC Form 4 executive-only insider trades via `LoadInsiderTradesTask` + `kernel/insider_trades.py` → `insider_net_buy_90d` (trailing-90d net executive buy USD) with monotone +1.
  - Cache locations: `data/fundamentals/{SYM}.parquet`, `data/earnings_surprise/{SYM}.parquet`, `data/insider_trades/{SYM}.parquet`.
  - Fetch scripts: `scripts/fetch_fundamentals.py`, `scripts/fetch_earnings_surprise.py`, `scripts/fetch_insider_trades.py` (SEC EDGAR rate-limited to 8 req/sec).
- **No-trade monitoring** (`kernel/pipeline/task_monitor.py::MonitorIdleStreakTask`): pipeline Task at end of `InferencePipeline` that tracks consecutive days with zero orders/exits and zero candidates. Warns above `monitoring.max_no_trade_days` (default 15) + `max_no_candidate_days` (default 15); ntfy surfaces the WARN via log scraping. State persisted across bars by `SimAdapter._monitor_state` and `RunnerAdapter.live_state.json`. `SimResult.longest_no_trade_streak` available post-backtest; opt-in `RENQUANT_FULL_SIM=1` invariant test asserts < 20d. **Enforces the "it's OK not to trade, but NOT systematically" user contract.**
- **Defensive-ticker universe exemption**: `FilterUniverseFloorTask` always admits `defensive_tickers` regardless of floor type/threshold — guarantees `bear_only` / `ConfidenceVetoTask` regimes always have something to buy (previously a single weak defensive passed the Sharpe gate → systemic no-trade). Regression tested in `tests/test_universe_alignment.py::TestDefensivesExemptFromFloor`.
- **Transformer panel backend (design-only, not yet implemented)**: see `doc/renquant_104_transformer_design.md` — cross-sectional attention across the date-group as alternative `panel_ltr.backend`. MPS target. Ship gate: ≥1.3× XGBoost OOS IC.

### Adding a New Strategy

```bash
python scripts/new_strategy.py --name foo --symbol AAPL --type classification
```

1. Scaffolds `backtesting/foo/` with `strategy_config.json`
2. Open a notebook, use `Strategy` class or manual workflow to train
3. Export model artifacts to the strategy directory
4. Export LEAN data: `python scripts/export_lean_watchlist.py --strategy foo`
5. `cd backtesting/foo && lean backtest .`
6. `python -m live.runner --strategy foo --broker paper --once`

---

## Development Rules (mandatory, always follow)

### 1. Logic Graph is the Source of Truth
`doc/logic_graph_103.md` is the canonical decision flowchart for renquant_103.
- **Whenever the notebook simulation cell (657a4a6c) changes**, update the logic graph first, then verify LEAN matches every node.
- **Whenever LEAN main.py changes**, check it against the logic graph and update if LEAN is intentionally extended.
- The logic graph covers: regime detection → mark-to-market → drawdown breaker → sell loop (all 5 exits in priority order) → buy gates (transition window, BEAR branch, velocity crash, EMA50) → candidate scan (all filters) → ranking → selection loop (tiered thresholds, sector guard, correlation guard, position sizing).

### 1b. Every Logical Unit Is a Task, Job, or Pipeline
**All new decision logic belongs in `kernel/pipeline/` or `kernel/panel_pipeline/` as a Task, Job, or Pipeline.** No new hand-written loops that bypass the orchestration layer.
- **Task**: an atomic step that reads/writes `InferenceContext` (or a ticker slice). Return `False` to short-circuit the enclosing Job's chain.
- **Job**: a sequential Task chain with a `should_skip(ctx)` gate. Jobs may run serially or via `run_parallel()` for per-ticker work.
- **Pipeline**: orders Jobs into phases and owns the full run (`InferencePipeline`, `SellOnlyPipeline`, `FullTrainingPipeline`, `PanelTrainingPipeline`).
- LEAN (`main.py`), the live runner (`live/runner.py`), and the notebook sim (`sim/runner.py`) all go through `InferencePipeline` via `LeanAdapter` / `RunnerAdapter` / `SimAdapter`. The legacy hand-written sim loop has been deleted — there is **one source of truth** for decision logic across all three surfaces. Universe admission is similarly consolidated into `kernel/pipeline/job_universe.py::LoadUniverseJob` (Tasks: `LoadArtifactsTask` → `FilterStalenessTask` → `FilterUniverseFloorTask`) so adding a new admission rule only requires touching one file.
- When adding a new decision:
  1. Write it as a Task (or new Job + Task chain) under `kernel/pipeline/` or `kernel/panel_pipeline/`.
  2. Wire it into the correct phase of the owning Pipeline.
  3. (LEAN / live / sim already all route through `InferencePipeline` — no per-surface mirroring needed.)
  4. Add paired alignment tests in `tests/test_panel_alignment.py` or `tests/test_policy_alignment.py`.

### 2. Tests for Every Feature — and Every Bug
Every policy in notebook and LEAN must have a corresponding test. **Every bug gets fixed as soon as it's found, and the fix ships with a regression test that would fail before the fix.** A bug without a test is a bug you'll see again.

Bug-fix workflow:
  1. Reproduce the bug with a failing test (unit or sim-level).
  2. Commit the test alone if it clarifies the bug scope (optional).
  3. Land the fix. The test now passes.
  4. Keep both together in the same branch — never merge a fix without its regression test.

This applies to:
  - Production incidents (e.g. empty policy-metadata.json crashing daily_104.sh → `tests/test_universe_alignment.py::TestResilience`)
  - Silent failure modes (e.g. systemic no-trade periods → `tests/test_no_trade_monitor.py` + `tests/test_no_trade_invariant.py`)
  - Upstream-task ordering bugs (e.g. ApplyGlobalCalibrationTask skipped in additive mode → `tests/test_panel_bugfixes.py`)


- Tests live in `tests/` (run with `python -m pytest tests/ -v`).
- `tests/test_policy_alignment.py`: 18 policy classes (Trailing/Stop/SDL/MaxHold/MinHold/SellStreak/EMA50/Velocity/Transition/Earnings/Tiered/Correlation/Sector/WashSale/MinScore/CombinedRanking/Sizing/**Rotation**), each with at least 6 `test_nb_*` + 6 `test_lean_*` + 1 cross-check. A meta-test enforces equal counts per class.
- `tests/test_lean_policies.py`: regression tests for LEAN-specific behavior (172 tests).
- `tests/test_panel_alignment.py`: 34 tests — flag parity across LeanAdapter / RunnerAdapter / PanelScoringJob, pipeline ordering invariant, panel-veto, conviction sizing, panel-rotation gate, **plus NGBoost** (flag parity, μ−λσ scoring, σ-sizing multiplier).
- `tests/test_ngboost_head.py`: 12 tests — NGBoost Normal(μ, σ) fit/predict/save/load, σ tracks heteroskedasticity, μ−λσ combined score, σ sizing multiplier bounds.
- `tests/test_training_cadence.py`: 8 tests — `training.cadence` gate (daily preserves existing behavior, weekly short-circuits off-cadence days, `--force` bypass).
- `tests/test_fundamentals_cache.py`: 9 tests — `FundamentalsStore` parquet cache + `fetch_fundamentals` with injected provider.
- **When adding any new feature to notebook or LEAN**, add paired tests to `test_policy_alignment.py` before committing. Both sides must be covered with equal test counts.
- Total test count as of last update: **~1307 collected (2026-04-24)** — up from ~1072 on prior. 2026-04-24 session added: test_training_run_audit migration tests (+3), test_blocked_by_population (+9), test_forward_returns_backfill (+8), test_bull_vol_block (+9), test_partial_sell (+7), test_trim_held (+13), test_kelly_rotation_gate (+11), test_kelly_pure_mode (+4), test_cusum_cooldown_v2 (+14), test_live_state_snapshot (+7), test_multi_entry_accumulation (+7), test_thesis_rotation (+8), test_db_separation (+13), test_panel_scoring_job update (+1). Plan G: test_hourly_features (+17), test_panel_hourly_wiring (+10). Plan F: test_regime_calibrator (+10). Other: test_panel_transformer (+12), test_transformer_scorer (+5), test_transformer_pipeline_integration (+4), test_ensemble_scorer (+6), test_recalibrate_scores (+2), universe_floor + drawdown-reset regressions. 2 invariants opt-in via `RENQUANT_FULL_SIM=1`. Run `python -m pytest tests/ --collect-only -q | tail -3` for exact.
- `tests/test_no_trade_monitor.py` — 11 tests for MonitorIdleStreakTask + SimResult streak surface + adapter round-trip. Guards the "no systematic no-trade periods" contract (CLAUDE.md 2b).
- `tests/test_no_trade_invariant.py` — 2 opt-in full-sim tests asserting `longest_no_trade_streak < 20` on current config.
- `tests/test_earnings_surprise.py` — 9 tests for yfinance-backed surprise cache + trailing-4Q factor.
- `tests/test_insider_trades.py` — 11 tests for SEC Form 4 parser (executive-only filter, P/S code filter, sign rules) + cache round-trip + trailing-90d net-buy factor.
- `tests/test_panel_bugfixes.py` — 6 tests guarding three session bugs: ApplyGlobalCalibrationTask additive-mode path, prepare_inference_panel_frames z-score wiring, fit_panel_calibrator z-score wiring.
- `tests/test_panel_orthogonal_factors.py` — 9 tests for the Round-3 factor functions (Amihud, volume_shift, price_to_high, realized_vol, drawdown_peak).
- `tests/test_universe_alignment.py` — 18 tests (up from 16): adapter parity, floor dispatch, staleness, defensives exemption, resilience to malformed artifacts.
- `tests/test_daily_104_e2e.py` — 3 scheduled-run smoke tests.
- `tests/test_hourly_features.py` — 17 tests for Plan G `compute_hourly_features()` (synthetic-session OHLCV → morning/afternoon_drift, vwap_premium, vol_ratio, intraday_realized_vol, overnight_gap).
- `tests/test_panel_hourly_wiring.py` — 8 tests for Plan G step 2 (HourlyBarStore parquet cache, LoadHourlyBarsTask flag/load paths, TickerPanelFactorJob merges 6 hourly columns, FactorZScoreTask emits `{col}_z`).
- `tests/test_regime_calibrator.py` — 10 tests for Plan F (`fit_regime_conditional` split/skip/round-trip, `LoadGlobalCalibrationTask` populates pooled + regime dict, `ApplyGlobalCalibrationTask` dispatches to regime or pooled fallback).
- `tests/test_blocked_by_population.py` — 9 tests for Plan P — `run_selection_loop` out-param populates per-ticker rejection reason (wash_sale / sector / correlation / tier / defensive_non_bear), adapter plumbing + DB write-through verified.
- `tests/test_forward_returns_backfill.py` — 8 tests for Plan AA — `ticker_forward_returns` table schema + upsert semantics + `scripts/backfill_forward_returns.py` end-to-end.
- `tests/test_bull_vol_block.py` — 9 tests for BULL_VOL-reversal gate (flag default off, defensives-only mode, full-cash mode, job ordering).
- `tests/test_partial_sell.py` — 7 tests for `ExitSignal.quantity` partial-sell infra (enables AB-trim Kelly rebalance).
- `tests/test_trim_held.py` — 17 tests for `TrimHeldTask` Kelly rebalance trim (opt-in default off per A/B regression; audit guards for Kelly target floor + mu <= 0).
- `tests/test_kelly_rotation_gate.py` — 11 tests for BC Kelly-delta rotation gate + audit guards (kelly_target_floor, mu <= 0 dispatch).
- `tests/test_kelly_pure_mode.py` — 4 tests for `disable_extra_multipliers` flag (pure Kelly sizing without conv/sigma_mult stacking).
- `tests/test_cusum_cooldown_v2.py` — 14 tests for Design C confidence-scaled CUSUM cooldown (wall-time mode promoted to golden v4.1 on 2026-04-24).
- `tests/test_live_state_snapshot.py` — 7 tests for Plan S — `live_state_snapshots` table, record_live_state_snapshot, state_json roundtrip.
- `tests/test_multi_entry_accumulation.py` — 7 tests for `per_session_buy_cap` in SizeAndEmitTask + TopUpHeldTask.
- `tests/test_thesis_rotation.py` — 8 tests for Approach A — thesis-degradation rotation gate (compare today vs fixed entry-time baseline, not noisy kelly deltas).
- `tests/test_db_separation.py` — 13 tests for sim/live DB split (`get_connection(role=)`, `clear_sim_tables`, per-run TRUNCATE, derived tables preserved).

### 2a. Promotion Thresholds Are Not Floors for Theoretically-Sound Wins

**Default rule:** APY win ≥ +2 pts on 27-mo OOS = promote to golden.

**Exception (user spec 2026-04-24):** when variables are rigorously controlled, **any positive margin is meaningful**. Specifically:

- **Live/sim parity fixes** (e.g. CUSUM-v2 wall_time mode closing known bar-count-vs-calendar-day drift) ship even at < +2 pt — the fix correctness is the point; the APY lift is incidental confirmation.
- **Theory-aligned wins where the predicted magnitude matches** (e.g. CUSUM-v2 predicted "~2 pt drift closure" in the roadmap, result +1.97) are signal, not noise — theory was specific, result matched.
- **Mechanism-clean changes** (no hyperparameter drift, same panel, same everything) with positive margin are shipped.

**Not exceptions:** new strategies, hyperparameter sweeps, panel retrains — those need the full +2 pt to clear the promotion floor because they have more ways to accidentally look good.

### 2b. Unexpected A/B Results = Audit Before Accepting

**When a theory we believed would improve the model produces the OPPOSITE result, the first hypothesis must be: "my implementation has a bug or my assumptions were wrong" — not "the theory is wrong".** Accept the negative result only AFTER the implementation audit.

Typical bugs that masquerade as "theory failed":
- Ordering bug in the pipeline (Task A overwrites Task B's output)
- Wrong-sign delta (e.g. `current - target` vs `target - current`)
- Unit mismatch (fraction vs pct, shares vs dollars)
- Silent guard fires (new Task blocked by an older flag default)
- Stale data (config not applied to the sim's retrained panel)
- Side effects from an upstream flag's default being wrong

**Audit checklist on an unexpected A/B result, before shipping the finding:**
1. Print a per-bar log of the new Task's inputs on ≥3 sample bars — do they look sane?
2. Reason through every input/output independently: if everything were correct, what would we *expect*? Compare to what we got.
3. Re-read the commit's defaults — does the "GOLDEN" variant in the A/B actually preserve v4 behaviour, or did we accidentally turn something on?
4. Check whether any other task/config reads the same data — possible interaction.
5. Only after all four → document the audit and shelve the theory with evidence.

Example (2026-04-24): **AB-trim appeared to hurt APY by 12.7 pts**. First response was "trim hurts, default off". Correct response per this rule: audit TrimHeldTask inputs for (a) timing with TopUp, (b) Kelly target volatility causing trim churn, (c) wrong-sign delta, (d) share rounding. The audit may find the theory is fine and the implementation had a bug.

### 3. Git Commits — Sync Everything, Guard Secrets
After completing any task, commit and push all changed files so the remote is always up to date.

**Before every commit:**
- Check `git status` for untracked or modified files — all should be staged unless explicitly excluded.
- If a file contains sensitive data (credentials, API keys, personal info), add it to `.gitignore` first, then commit the `.gitignore` change. Never commit the sensitive file itself.
- Currently gitignored sensitive/large paths: `.env` (API keys), `live/logs/` (trade logs), `data/` (OHLCV cache), `backtesting/data/` (LEAN data), `backtesting/*/backtests/` (LEAN run output).
- Everything else — model artifacts, strategy configs, notebooks, scripts, tests, docs — should be committed and pushed.

**Rule of thumb**: if it's not in `.gitignore`, it should be in the remote. When in doubt, add to `.gitignore` rather than leaving files untracked silently.

### 4. Always Keep Docs Up to Date
After any non-trivial change, run `/update-docs` or manually sync:
- `doc/logic_graph_103.md` — decision flowchart (update before LEAN changes)
- `doc/architecture.md` — overall pipeline and data flow
- `doc/models.md` — model types, exit logic, stop-loss params
- `doc/renquant_103_design.md` — full 103 strategy spec including test counts
- `doc/renquant_104_design.md` — renquant_104 panel-LTR spec (diff vs 103, FullTrainingPipeline, panel flag wiring, test coverage)
- `CLAUDE.md` — this file; keep test counts and rule set current

### 4. Documentation Index

| Doc | What it covers |
|-----|----------------|
| `doc/golden_config_2026-04-23.md` | **Current golden = v4.1** (CUSUM wall_time, +39.82% sweep APY, ~+65% expected live). v1 → v2 → v3 → v4 → v4.1 history inline at bottom. Revert here if a future change drops portfolio APY. Frozen copy at `backtesting/renquant_104/strategy_config.golden.json`. |
| `doc/database.md` | **DB reference** — two-file architecture (data/runs.db live + data/sim_runs.db sim), 7 tables × full column schema, common queries, schema migration rules, retention plan. DB is a core asset — consult before adding columns. |
| `doc/improvement_roadmap.md` | **Living roadmap** — Active queue (pending work) + Completed archive. Work through top-down. |
| `doc/session_handoff_2026-04-23.md` | Session-end state for 2026-04-23 (G promoted, F+H shelved). Superseded on 2026-04-24 by roadmap update + 26 commits (M⁺/P/AA/Trim/BC/CUSUM-v2-PROMOTED-v4.1/S/Kelly-pure/Multi-entry/Thesis-A/DB split). |
| `doc/panel_training_runs.md` | Per-run training log (config diffs, IC, feature importance, verdict). Prepend new runs to top. |
| `doc/panel_ltr_primer.md` | Tutorial on Panel-LTR + NGBoost training methodology + abbreviation glossary. |
| `doc/environment.md` | Python deps (`requirements.lock.txt`), critical lib versions, non-Python tooling (Docker / LEAN / launchd), env vars. Reproducibility source of truth. |
| `doc/logic_graph_103.md` | **Complete decision flowchart** — every branch in notebook simulation and LEAN, regime param table, alignment table |
| `doc/architecture.md` | Pipeline overview, data stores, indicator registry, model types, strategy list |
| `doc/models.md` | Model ABC, all model types, exit logic, stop-loss, single-day gate, trailing stop |
| `doc/renquant_103_design.md` | Full renquant_103 design spec: regime layers, stock selection pipeline, artifact list, key implementation details |
| `doc/renquant_104_design.md` | renquant_104 panel-LTR design: cross-sectional ranking, FullTrainingPipeline, panel flag wiring across LEAN/live/sim |
| `doc/indicators.md` | All registered indicators and their parameters |
| `doc/usage.md` | CLI commands, live runner flags, scheduled runs |
| `doc/setup.md` | Environment setup, Docker, credentials |
| `doc/tech-stack.md` | Technology choices and rationale |
| `doc/renquant_102_vs_103_report.md` | Comparison report between strategies |
| `tests/test_policy_alignment.py` | 235 paired NB/LEAN alignment tests (18 policies, including TestRotationAlignment) |
| `tests/test_lean_policies.py` | LEAN regression tests (172 tests) |
| `tests/test_kernel_isolation.py` | CI enforcement: kernel/ must not import common/ |
| `tests/test_kernel_units.py` | 136 unit tests for all 10 kernel modules (includes market_gates, portfolio, compute_relative_strength, rotation) |
| `tests/test_pipeline.py` | 31 tests for PipelineContext, run_tasks, Job/Pipeline |
| `tests/test_universe_alignment.py` | 16 tests — LoadUniverseJob, universe_floor type dispatch (none / sharpe / ic), staleness, extensibility, adapter parity (no hand-written filter loops remain in main.py / adapters/sim.py / live/runner.py) |
| `tests/test_training_modules.py` | 16 tests for training/features.py, training/tournament.py, training/export.py |

---

## General Coding Guidelines

**Bias toward caution over speed. For trivial tasks, use judgment.**

### 1. Think Before Coding

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that YOUR changes made unused.

Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

Transform tasks into verifiable goals before implementing:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"

For multi-step tasks, state a brief plan with verify steps before starting.
