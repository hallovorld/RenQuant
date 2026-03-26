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
cd backtesting/test_001_nvda
lean backtest .
```

**Analysis mode**: Open `Notebooks/backtest_analysis.ipynb` to visualize LEAN results.

**Live mode**: Run the live trader with paper or IBKR broker.

```bash
python -m live.runner --strategy test_001_nvda --broker paper --once
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

**2. Model Artifacts** (`backtesting/{strategy}/*.json`)
- JSON (not pickle) for LEAN compatibility
- `*-policy-metadata.json` is the contract between research and execution

**3. Backtesting** (`backtesting/`) — QuantConnect LEAN engine (Docker)
- `main.py`: self-contained `QCAlgorithm` (no `common/` dependency)
- Loads JSON models, recomputes indicators inline

**4. Live Trading** (`live/`)
- `python -m live.runner --strategy X --broker paper|ibkr --once`
- `PaperBroker` for testing, `IBKRBroker` for real execution
- Logs to `live/logs/{strategy}/{date}.json`

**5. Analysis** (`Notebooks/backtest_analysis.ipynb`)
- `common.backtest_dashboard` — 4-panel dashboard
- `common.plot_normalized_performance` — normalized equity with entry markers

### Adding a New Strategy

```bash
python scripts/new_strategy.py --name foo --symbol AAPL --type classification
```

1. Scaffolds `backtesting/foo/` with `strategy_config.json`
2. Open a notebook, use `Strategy` class or manual workflow to train
3. Export model artifacts to the strategy directory
4. `cd backtesting/foo && lean backtest .`
5. `python -m live.runner --strategy foo --broker paper --once`
