# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RenQuant is a personal quantitative trading workstation for Apple Silicon. It implements a "Glass-box" three-layer pipeline: data ingestion → ML signal generation → backtesting execution. All components are statistically interpretable and strictly decoupled.

## Environment Setup

Single conda environment (Miniforge, arm64-optimized):

```bash
conda create -n renquant python=3.10
conda activate renquant
pip install pandas numpy matplotlib seaborn yfinance scikit-learn xgboost jupyterlab
pip install "openbb[all]" openbb-cli backtesting
pip install lean
lean login
```

Docker must be allocated 16GB+ memory for LEAN engine.

## Workflow: Three Modes

**Research mode** (fast iteration, no Docker): Run `Notebooks/test_001_nvda.ipynb` to train and export models.

**Validation mode** (final backtest): Run LEAN on the exported models.

```bash
# Validation only — not for every iteration
cd backtesting/test_001_nvda
lean backtest .
```

**Analysis mode**: Open `Notebooks/backtest_analysis.ipynb` to visualize the latest LEAN result.

## Shared Library: `common/`

All reusable notebook/research logic lives in `common/`. Import as `import common as rq`.
**Do not import `common/` from inside `backtesting/` — LEAN Docker cannot access it.**

| Module | Key exports |
|--------|-------------|
| `common/config.py` | `load_strategy_config`, `build_model_path` |
| `common/data.py` | `fetch_ohlcv` |
| `common/indicators.py` | `add_indicators`, `compute_macd/rsi/cci` |
| `common/features.py` | `add_gate_signals`, `build_transitions`, `STATE_COLUMNS` |
| `common/training.py` | `fitted_q_iteration`, `score_valid_actions` |
| `common/plotting.py` | `backtest_dashboard`, `load_latest_backtest`, parse/plot helpers |

## Architecture

### Pipeline

**1. Research** (`Notebooks/test_001_nvda.ipynb`) — `renquant` conda env
- `rq.fetch_ohlcv` → `rq.add_indicators` → `rq.add_gate_signals` → `rq.build_transitions`
- `rq.fitted_q_iteration` (8 iterations, gamma=0.95, 5bps transaction cost)
- Exports 3 model JSON files (hold/buy/sell) + policy metadata JSON

**2. Model Artifacts** (`backtesting/test_001_nvda/*.json`)
- JSON (not pickle) for LEAN compatibility
- `*-policy-metadata.json` is the contract between research and backtesting layers

**3. Backtesting** (`backtesting/`) — QuantConnect LEAN engine (Docker)
- `main.py`: `QCAlgorithm` that loads JSON models, recomputes indicators inline each day, picks `argmax Q(s, a)`
- `config.py`: self-contained LEAN-local config loader (no `common/` dependency)

**4. Analysis** (`Notebooks/backtest_analysis.ipynb`)
- `rq.load_latest_backtest` → `rq.backtest_dashboard`
- 4-panel dashboard: price+signals, equity curve, drawdown, statistics table

### State Features

All strategies: `macd_line`, `macd_signal`, `macd_hist`, `rsi`, `cci`, `position_flag`

MACD(12,26,9), RSI(14), CCI(20) — stored in policy metadata, must match between notebook and `main.py`.

### Adding a New Strategy

1. Copy `Notebooks/test_001_nvda.ipynb` → new notebook
2. Copy `backtesting/test_001_nvda/` → `backtesting/<strategy_name>/`
3. Update `strategy_config.json` (symbol, dates, model name)
4. Run notebook → exports JSON models to the strategy directory
5. `lean backtest .` from the strategy directory
6. Open `backtest_analysis.ipynb`, point `STRATEGY_DIR` at the new strategy
