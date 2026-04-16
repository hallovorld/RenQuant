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

**Scheduled mode**: Three daily automation runs via macOS launchd (all NYSE-holiday-aware):

| Run | Time (PT) | Time (ET) | Script | What it does |
|-----|-----------|-----------|--------|--------------|
| Market open | 6:32 AM | 9:32 AM | `live_only_103.sh --sell-only` | Exit stop-loss / gap-down positions early using today's opening price |
| Pre-close | 12:44 PM | 3:44 PM | `live_only_103.sh --sell-only` | Exit intraday stop breaches before close using near-final daily price |
| After close | 1:55 PM | 4:55 PM | `daily_103.sh` | Full run: retrain models → export LEAN data → recalibrate scores → buy + sell signals |

```bash
# Manual runs
bash scripts/daily_103.sh              # full run (retrain + trade)
bash scripts/live_only_103.sh          # intraday sell check (no retrain)
python -m live.runner --strategy renquant_103 --broker alpaca --once --sell-only

# LaunchAgents (all loaded)
# ~/Library/LaunchAgents/com.renquant.open103.plist
# ~/Library/LaunchAgents/com.renquant.preclose103.plist
# ~/Library/LaunchAgents/com.renquant.daily103.plist
# Logs: logs/live_103/{date}-open.log, {date}-preclose.log
#       logs/daily_103/{date}.log
# NYSE calendar guard: all three scripts skip US market holidays automatically
```

Alpaca credentials are stored in `.env` (gitignored): `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. Notifications (macOS + iPhone/ntfy) are sent on success or failure.

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

**5. Analysis** (`scripts/analyze_backtest.py`)
- `common.backtest_dashboard` — price, decision telemetry, equity, drawdown, and stats
- `common.plot_normalized_performance` — normalized equity with entry markers
- Multi-stock trade detail table (CSV + console) with per-symbol colored markers

### Strategies

**Single-stock** (renquant_101): One symbol, one model. Config uses `stock_symbol`.

**Multi-stock pre-trained scanner** (renquant_102): 3-stage pipeline (DETECT → CONFIRM → EXECUTE). See also `Notebooks/renquant_102.ipynb`. Notebook trains 3 approaches per symbol (Dual Momentum, Classification/RF, Q-Learning) on a 70/30 walk-forward train/test split of a rolling 3yr window (`training_years=3`), exports best model per symbol to `models/{SYMBOL}/` by OOS after-tax Sharpe (floor: 0.8). Orphan model directories (symbols removed from watchlist) are automatically purged on each export. Both OOS Sharpe (`sharpe`) and in-sample Sharpe (`sharpe_is`) are stored in `policy-metadata.json` for diagnostics. The notebook also includes a portfolio-level simulation that mirrors the LEAN multi-stock logic, enabling parameter tuning in Python before running LEAN. LEAN loads pre-trained models, scans watchlist for bullish volume spikes (percentile mode P85 by default, up-close days only), applies that stock's model for confirmation, and enforces: per-position 8% stop-loss, 15% portfolio drawdown circuit breaker (halts new buys), configurable regime filter (currently disabled — relative features already encode market context vs SPY), and sector concentration guard (max 3 positions per sector). Models include `trained_date`; LEAN skips models older than `model_staleness_days` (default 60). Config uses `watchlist` array (21 stocks + ETFs), `train_split`, `model_staleness_days`, `volume_filter`, `max_concurrent_positions`, `risk` (stop-loss, drawdown halt, regime filter), `sector_map`, `max_positions_per_sector`. Notebook-only config fields (not read by LEAN): `sample_start`/`sample_end` (defines full data window for indicator warmup), `training_years` (rolling training window size), `indicator_spec` (indicator hyperparameters), `model_params` (model hyperparameters).

**Adaptive regime multi-stock** (renquant_103): Extends 102 with a 3-layer regime detector. Layer 1: rolling Hurst exponent (63-day, characterises momentum vs mean-reversion). Layer 2: CUSUM changepoint detection (fast transition trigger, uncertainty window after each break). Layer 3: GMM clustering on 4 SPY features (10d return, 20d realised vol, ADX, return autocorr) — outputs continuous P(regime) to scale position size. Four regimes: BULL_CALM (momentum entry, 15% max position), BULL_VOLATILE (capitulation entry on high-vol down-close, 20%), CHOPPY (divergence entry — stock outperforming SPY, 15%), BEAR (offensive buys blocked; 1 defensive slot for GLD/TLT/XLV/XLU at 15% of portfolio). Stock selection pipeline: earnings filter (±3d) → SPY EMA50 trend gate (blocks all new buys when SPY < 50-day EMA) → SPY velocity crash filter (blocks buys if SPY down >3% in last 3 days) → transition uncertainty window (no buys for 3 bars after CUSUM changepoint) → BEAR branch (defensives only) → model buy signal (entry trigger, no separate volume-scan gate) → calibrated `rank_score` filter (regime-aware: 0.10 BULL_CALM, 0.15 others) → relative-strength score vs sector ETF → combined rank (data-driven blend weights from `ranking.blend_weights` in strategy_config.json, default 50% rank + 50% RS, updated daily by `scripts/recalibrate_scores.py`) → selection loop: tiered threshold escalation (slot 1: 0.10, slot 2: 0.30, slot 3: 0.50, all on calibrated `rank_score`) → wash-sale guard → sector guard (max 3 per sector) → correlation-aware greedy selection (threshold 0.70). New artifacts: `spy-gmm-regime.json`, `watchlist-correlation.json`, `earnings-calendar.json`. Watchlist (24): adds GLD, TLT, XLV, XLU as defensive counter-cyclical tickers. Notebook: `Notebooks/renquant_103.ipynb`. Earnings calendar refreshed via `python scripts/fetch_earnings_calendar.py --strategy renquant_103`. Key implementation details: (1) Classification model receives `close = stock_close / spy_close × 100` so forward-return labels become relative outperformance vs SPY (prevents bull-market always-buy bias); (2) `predict_bulk()` returns string Series — map with `{"buy": 1, "hold": 0, "sell": -1}` for numeric use; (3) `predict_score_bulk()` returns native continuous float scores: Classification returns raw BagLearner output, QLearning returns Q(buy)−Q(sell), XGBoost returns P(buy)−P(sell), Manual returns raw vote count; these are not cross-model comparable, so the live runner calibrates them through `common.models.scoring` into a common `rank_score` (P(outperform SPY by threshold% in lookahead days)) using metadata `score_calibration` when present or a runtime fallback; calibration method is selected by sample size: isotonic for n≥300, Platt scaling (sigmoid) for 120≤n<300, constant base-rate for n<120; `score_calibration` is refreshed daily by `scripts/recalibrate_scores.py` after each model retrain; (4) OOS Sharpe floor is 0.8 — marginal models below 0.8 are excluded; (5) `max_hold_days` is 500 for BULL_CALM/BULL_VOLATILE/BEAR, 10 for CHOPPY; (6) position sizing: 8 concurrent positions, 15% max each — `min(cash, portfolio × 0.15)`; (7) `min_hold_days: 20` — model-sell blocked before 20 days (stop-loss and single-day loss gate trigger immediately); (8) 3 consecutive daily sell signals required before model exit; (9) all three model types (Classification, QLearning, XGBoost) use `abs(hash(ticker)) % 2^32` as a per-ticker seed for reproducibility; (10) tournament training: trains Classification + QLearning + Manual + XGBoost per ticker, exports best by OOS Sharpe; (11) BULL_CALM trailing stop: 20% gain trigger, 18% trail below high-water mark; stop uses peak gain (HWM-based) so it stays armed after pullbacks; (12) exit priority order (same in LEAN, notebook, live runner): trailing stop → cumulative stop-loss (15% BULL_CALM, 5% others) → single-day loss gate (10% BULL_CALM, disabled others) → max hold → model sell streak; (13) `max_single_day_loss_pct: 0.10` in BULL_CALM — exits when today's close drops ≥10% from yesterday's close; protects against gap-down days where cumulative stop fires too late on daily bars; (14) fixed OOS cutoff 2024-01-01 for stable simulation; live models retrained on last 4 years of data (not all history) for relevance; (15) notebook simulation and LEAN are aligned on the major gating and selection rules, while live now additionally calibrates mixed-model scores before ranking; (16) live runner uses per-model `feature_columns` from policy-metadata.json, not global config (regime-context columns only exist during notebook training); (17) tiered thresholds configured in `tiered_thresholds` array in `strategy_config.json` — identical slot logic across components using `tier_idx = min(slots_filled, len(tiers) - 1)`; (18) `tests/test_runner_ranking.py` covers calibrated rank scoring, mixed-model ordering, score logging, and tier application; (19) 403 tests are currently collected under `tests/`.

Volume filter supports two modes via `volume_filter.mode` in `strategy_config.json`:
- `"percentile"` (default): Adaptive per-stock filter. Triggers when today's volume is in the top N% of the lookback window (default: P85 = top 15%). Self-calibrates for each stock's own volume distribution, so stable large-caps and volatile small-caps are held to the same relative standard.
- `"zscore"` (legacy): Fixed z-score threshold across all stocks. Uses `volume_zscore_threshold` (default: 1.5).

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

### 2. Tests for Every Feature
Every policy in notebook and LEAN must have a corresponding test.
- Tests live in `tests/` (run with `python -m pytest tests/ -v`).
- `tests/test_policy_alignment.py`: 17 policy classes, each with exactly 6 `test_nb_*` + 6 `test_lean_*` + 1 cross-check. A meta-test enforces equal counts per class.
- `tests/test_lean_policies.py`: regression tests for LEAN-specific behavior (172 tests).
- **When adding any new feature to notebook or LEAN**, add paired tests to `test_policy_alignment.py` before committing. Both sides must be covered with equal test counts.
- Total test count as of last update: 403 collected tests. Run `python -m pytest tests/ -v` to verify.

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
- `CLAUDE.md` — this file; keep test counts and rule set current

### 4. Documentation Index

| Doc | What it covers |
|-----|----------------|
| `doc/logic_graph_103.md` | **Complete decision flowchart** — every branch in notebook simulation and LEAN, regime param table, alignment table |
| `doc/architecture.md` | Pipeline overview, data stores, indicator registry, model types, strategy list |
| `doc/models.md` | Model ABC, all model types, exit logic, stop-loss, single-day gate, trailing stop |
| `doc/renquant_103_design.md` | Full renquant_103 design spec: regime layers, stock selection pipeline, artifact list, key implementation details |
| `doc/indicators.md` | All registered indicators and their parameters |
| `doc/usage.md` | CLI commands, live runner flags, scheduled runs |
| `doc/setup.md` | Environment setup, Docker, credentials |
| `doc/tech-stack.md` | Technology choices and rationale |
| `doc/renquant_102_vs_103_report.md` | Comparison report between strategies |
| `tests/test_policy_alignment.py` | 222 paired NB/LEAN alignment tests (17 policies) |
| `tests/test_lean_policies.py` | LEAN regression tests (172 tests) |
