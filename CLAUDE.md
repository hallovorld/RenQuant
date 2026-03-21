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

## Workflow: Two Modes

**Research mode** (fast iteration, no Docker): Use `backtesting.py` directly inside notebooks to test strategy ideas quickly.

**Validation mode** (final backtest): Once a strategy is ready, export models to JSON and run LEAN for rigorous event-driven backtesting.

```bash
# Validation only — not for every iteration
cd backtesting/test_001_nvda
lean backtest .
```

## Architecture

### Three-Layer Pipeline

**1. Research Layer** (`Notebooks/`) — run in `renquant` conda env
- Fetches OHLCV data via OpenBB/yfinance
- Computes technical indicators (MACD, RSI, CCI)
- Trains XGBoost Q-value models via Fitted Q-Iteration (8 iterations, gamma=0.95, 5bps transaction cost)
- Exports 3 model JSON files (hold/buy/sell actions) + policy metadata JSON

**2. Model Layer** (`models/`, `backtesting/test_001_nvda/*.json`)
- Models are serialized as JSON (not pickle) for LEAN compatibility
- Each model corresponds to one Q-value: `Q(state, hold)`, `Q(state, buy)`, `Q(state, sell)`
- Policy metadata (`*-policy-metadata.json`) defines: state columns, indicator parameters, gate rules (position constraints), action definitions

**3. Backtesting Layer** (`backtesting/`) — QuantConnect LEAN engine (Docker)
- `main.py`: `QCAlgorithm` subclass that loads JSON models at `Initialize()`, then each trading day fetches 60-day history, computes features, runs XGBoost inference, and picks `argmax Q(s, a)`
- `strategy_config.json`: symbol, initial cash ($100k), backtest date range
- `config.json`: LEAN algorithm entry point configuration

### Data Flow

```
Notebook → trains XGBoost → exports *.json models
                                    ↓
                          LEAN loads models at init
                                    ↓
                    Each day: history → features → argmax Q → order
```

### State Features

All strategies use: `macd_line`, `macd_signal`, `macd_hist`, `rsi`, `cci`, `position_flag`

MACD(12,26,9), RSI(14), CCI(20) — parameters stored in policy metadata, must match between notebook and LEAN algorithm.

### Adding a New Strategy

1. Copy `Notebooks/test_001_nvda.ipynb` → new notebook
2. Update `strategy_config.json` with new symbol/dates
3. Train models, export JSON files to `backtesting/<strategy_name>/`
4. Copy and adapt `main.py` to load the new model files
5. Run `lean backtest .` from the strategy directory
