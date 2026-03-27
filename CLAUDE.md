# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RenQuant is a personal quantitative trading workstation for Apple Silicon. It implements a "Glass-box" pipeline: data ingestion → ML signal generation → backtesting (LEAN) → live trading (IBKR). All components are statistically interpretable and strictly decoupled.

## Environment Setup

Single conda environment (Miniconda, Apple Silicon arm64):

```bash
conda create -n renquant python=3.10
conda activate renquant
pip install pandas numpy matplotlib seaborn yfinance scikit-learn xgboost jupyterlab pyarrow
pip install "openbb[all]" openbb-cli backtesting scipy
pip install lean
lean login
```

Docker must be allocated 16GB+ memory for LEAN engine.

## Workflow: Four Modes

**Research mode** (fast iteration, no Docker): Run notebooks to train and export models.

**Validation mode** (final backtest): Run LEAN on exported models.

```bash
python scripts/export_lean_data.py --symbol NVDA
cd backtesting/renquant_101
lean backtest .
```

To backtest and render charts in one step:

```bash
python scripts/backtest_and_analyze.py --strategy renquant_101
```

**Analysis mode**: Run `python scripts/analyze_backtest.py --strategy renquant_101` to visualize LEAN results, including decision telemetry for score/threshold inspection.

**Live mode**: Run the live trader with paper or IBKR broker.

```bash
python -m live.runner --strategy renquant_101 --broker paper --once
python -m live.runner --strategy renquant_102 --broker paper --once  # multi-stock
```

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
| `common/plotting.py` | `backtest_dashboard`, `plot_normalized_performance`, parse/plot helpers |

## Architecture

### Data Layer (`common/data/`)

- `DataSource` ABC with `YFinanceSource` (working) and `IBKRSource` (stub)
- `LocalStore` caches OHLCV as Parquet at `data/ohlcv/{SYMBOL}/1d.parquet`
- `fetch_ohlcv()` checks cache first, fetches missing dates, saves locally

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

**3. Backtesting** (`backtesting/`) — QuantConnect LEAN engine (Docker)
- `main.py`: self-contained `QCAlgorithm` (no `common/` dependency)
- Loads JSON models, recomputes indicators inline
- Enforces trading constraints (wash sale, min/max hold) and position sizing (max position %, cash reserve) from `strategy_config.json`
- Single-stock strategies (renquant_101): one symbol per backtest
- Multi-stock strategies (renquant_102): volume z-score scanner, loads pre-trained models per symbol from `models/{SYMBOL}/`, max N concurrent positions

**4. Live Trading** (`live/`)
- `python -m live.runner --strategy X --broker paper|ibkr --once`
- `PaperBroker` for testing, `IBKRBroker` for real execution
- Auto-detects single-stock vs multi-stock strategies (presence of `watchlist` in config)
- Logs to `live/logs/{strategy}/{date}.json`

**5. Analysis** (`scripts/analyze_backtest.py`)
- `common.backtest_dashboard` — price, decision telemetry, equity, drawdown, and stats
- `common.plot_normalized_performance` — normalized equity with entry markers
- Multi-stock trade detail table (CSV + console) with per-symbol colored markers

### Strategies

**Single-stock** (renquant_101): One symbol, one model. Config uses `stock_symbol`.

**Multi-stock pre-trained scanner** (renquant_102): 3-stage pipeline (DETECT → CONFIRM → EXECUTE). Notebook trains 4 approaches per symbol (Dual Momentum, Classification/RF, Q-Learning, Mean Reversion) on rolling 2yr window, exports best model per symbol to `models/{SYMBOL}/`. The notebook also includes a portfolio-level simulation that mirrors the LEAN multi-stock logic (volume z-score scan → model confirmation → multi-position management with configurable sizing and cash reserve), enabling parameter tuning in Python before running LEAN. LEAN loads pre-trained models, scans watchlist for volume z-score spikes, applies that stock's model for confirmation. Models include `trained_date`; LEAN skips models older than `model_staleness_days` (default 30). Config uses `watchlist` array (30 stocks + ETFs), `model_staleness_days`, `volume_zscore_lookback`, `volume_zscore_threshold`, `max_concurrent_positions`.

### Adding a New Strategy

```bash
python scripts/new_strategy.py --name foo --symbol AAPL --type classification
```

1. Scaffolds `backtesting/foo/` with `strategy_config.json`
2. Open a notebook, use `Strategy` class or manual workflow to train
3. Export model artifacts to the strategy directory
4. `cd backtesting/foo && lean backtest .`
5. `python -m live.runner --strategy foo --broker paper --once`
