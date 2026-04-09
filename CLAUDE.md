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

**Scheduled mode**: Daily automation retrains models and trades via Alpaca.

```bash
bash scripts/daily_102.sh          # manual run
# Automated via macOS launchd: weekdays at 1:55 PM PST (4:55 PM EST, after market close)
# LaunchAgent: ~/Library/LaunchAgents/com.renquant.daily102.plist
# Logs: logs/daily_102/{date}.log
```

Alpaca credentials are stored in `.env` (gitignored): `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. Notifications (macOS + iPhone/ntfy) are sent on success or failure.

## Shared Library: `common/`

Import as `import common`. **Do not import `common/` from inside `backtesting/` — LEAN Docker cannot access it.**

| Module | Key exports |
|--------|-------------|
| `common/config.py` | `load_strategy_config`, `build_model_path` |
| `common/data/` | `fetch_ohlcv` (with Parquet cache), `DataSource` ABC, `LocalStore` |
| `common/indicators/` | `compute_indicators`, `add_indicators` (compat), `list_indicators`, `@register` |
| `common/models/` | `BaseModel`, `ManualModel`, `ClassificationModel`, `QLearningModel`, `FQIModel`, `OptimizationModel`, `create_model` |
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
- `compute_indicators(df, {"rsi": {"period": 14}, "macd": {}})` applies any subset

### Model Types (`common/models/`)

All implement `BaseModel` ABC: `train()`, `predict()`, `save()`, `load()`. JSON artifacts only.

| Type | Training | Prediction |
|------|----------|------------|
| `manual` | No-op (generic score_rules at construction) | Multi-indicator threshold voting |
| `classification` | Forward-return labels → BagLearner(RTLearner) | Forest query |
| `qlearning` | Discretize states, Q-table over epochs | Q-table argmax |
| `fqi` | Transition tuples → FQI with XGBoost | Score per action, argmax |
| `optimization` | Nelder-Mead over indicator params + inner model | Delegate to best inner |

### Pipeline

**1. Research** (Notebooks) — `renquant` conda env
- `common.fetch_ohlcv` → `common.compute_indicators` → model.train() → model.save()
- Strategy-essential logic (gate rules, transitions) stays in notebooks

**2. Model Artifacts** (`backtesting/{strategy}/*.json` or `models/{SYMBOL}/*.json`)
- JSON (not pickle) for LEAN compatibility
- `*-policy-metadata.json` is the contract between research and execution
- Single-stock (101): artifacts at strategy root; multi-stock (102): `models/{SYMBOL}/` subdirectories

**2b. LEAN Data Export** (required before backtesting)
- `python scripts/export_lean_watchlist.py --strategy renquant_102`
- Converts parquet cache → LEAN daily zips + map files + factor files
- Skips symbols already exported (use `--force` to re-export)
- Must re-run after adding new symbols to a watchlist

**3. Backtesting** (`backtesting/`) — QuantConnect LEAN engine (Docker)
- `main.py`: self-contained `QCAlgorithm` (no `common/` dependency)
- Loads JSON models, recomputes indicators inline
- Enforces trading constraints (wash sale, min/max hold, stop-loss), position sizing, portfolio drawdown circuit breaker, SPY regime filter, and sector concentration guard — all from `strategy_config.json`
- Reports after-tax metrics via `SetRuntimeStatistic()` when `tax` config is present
- Single-stock strategies (renquant_101): one symbol per backtest
- Multi-stock strategies (renquant_102): volume z-score scanner, loads pre-trained models per symbol from `models/{SYMBOL}/`, max N concurrent positions

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

**Multi-stock pre-trained scanner** (renquant_102): 3-stage pipeline (DETECT → CONFIRM → EXECUTE). Notebook trains 3 approaches per symbol (Dual Momentum, Classification/RF, Q-Learning) on a 70/30 walk-forward train/test split of a rolling 2yr window, exports best model per symbol to `models/{SYMBOL}/` by OOS after-tax Sharpe (floor: 0.5). Orphan model directories (symbols removed from watchlist) are automatically purged on each export. Both OOS Sharpe (`sharpe`) and in-sample Sharpe (`sharpe_is`) are stored in `policy-metadata.json` for diagnostics. The notebook also includes a portfolio-level simulation that mirrors the LEAN multi-stock logic, enabling parameter tuning in Python before running LEAN. LEAN loads pre-trained models, scans watchlist for bullish volume spikes (percentile mode P85 by default, up-close days only), applies that stock's model for confirmation, and enforces: per-position 8% stop-loss, 15% portfolio drawdown circuit breaker (halts new buys), SPY 200-SMA regime filter (suppresses buys in bear markets), and sector concentration guard (max 1 position per sector). Models include `trained_date`; LEAN skips models older than `model_staleness_days` (default 60). Config uses `watchlist` array (19 stocks + ETFs), `train_split`, `model_staleness_days`, `volume_filter`, `max_concurrent_positions`, `risk` (stop-loss, drawdown halt, regime filter), `sector_map`, `max_positions_per_sector`.

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
