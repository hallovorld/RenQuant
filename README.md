# RenQuant

Personal quantitative trading workstation for Apple Silicon. Glass-box pipeline: data ingestion, ML signal generation, backtesting (LEAN), and live trading (IBKR).

## Architecture

```
  Data Layer          Model Layer              Execution Layer
 ┌──────────┐    ┌──────────────────┐    ┌────────────────────────┐
 │ yfinance  │───>│ Manual (rules)   │───>│ LEAN backtest (Docker) │
 │ IBKR      │    │ Classification   │    │ Live trader (IBKR)     │
 │ Parquet   │    │ Q-Learning       │    │ Paper broker           │
 │ cache     │    │ FQI (XGBoost)    │    └────────────────────────┘
 └──────────┘    │ Optimization     │              │
                  └──────────────────┘              v
                           │              ┌──────────────────┐
                           └──────────────│ Analysis charts  │
                                          └──────────────────┘
```

## Directory Structure

```
RenQuant/
├── common/                  # Shared library
│   ├── data/                # DataSource ABC, yfinance, IBKR stub, Parquet cache
│   ├── indicators/          # Registry-based: rsi, macd, cci, bbp, ema, stochastic, ppo, momentum
│   ├── models/              # BaseModel ABC + 5 implementations
│   │   └── learners/        # RTLearner, BagLearner, TabularQLearner
│   ├── strategy.py          # StrategyConfig + Strategy composition
│   ├── portfolio.py         # Local portfolio simulator
│   ├── plotting.py          # Backtest dashboard + normalized performance chart
│   └── config.py            # Config loading utilities
├── Notebooks/               # Research notebooks
├── backtesting/             # LEAN strategies (self-contained, no common/ imports)
├── live/                    # Live trading runner + broker abstraction
├── scripts/                 # Scaffolding tools
├── data/                    # Local Parquet cache (gitignored)
└── doc/                     # Detailed documentation
```

## Quick Start

```bash
# Setup
conda create -n renquant python=3.10
conda activate renquant
pip install pandas numpy matplotlib seaborn yfinance scikit-learn xgboost jupyterlab pyarrow
pip install "openbb[all]" openbb-cli backtesting scipy
pip install lean
lean login
```

### 1. Scaffold a new strategy
```bash
python scripts/new_strategy.py --name nvda_rf --symbol NVDA --type classification
```

### 2. Research — train in notebook
```bash
jupyter lab  # open notebook, run all cells to train and export models
```

### 3. Backtest with LEAN
```bash
cd backtesting/nvda_rf && lean backtest .
```

### 4. Analyze results
```bash
# Open Notebooks/backtest_analysis.ipynb
```

### 5. Live trading
```bash
# Paper trading (test)
python -m live.runner --strategy nvda_rf --broker paper --once

# Real trading (requires IBKR setup)
python -m live.runner --strategy nvda_rf --broker ibkr
```

## Model Types

| Type | Method | Use Case |
|------|--------|----------|
| **Manual** | Rule-based indicator voting | Baseline, interpretable |
| **Classification** | Random Forest on forward-return labels | Fast, deterministic |
| **Q-Learning** | Tabular Q with discretized states | Model-free RL |
| **FQI** | Fitted Q-Iteration with XGBoost | Function approximation RL |
| **Optimization** | Nelder-Mead parameter search + inner model | Auto-tuning |

## Indicator Library

All indicators share a uniform API: `(df, **params) -> DataFrame`

| Category | Indicators |
|----------|-----------|
| Momentum | RSI, MACD (line/signal/hist), EMA, Momentum |
| Volatility | CCI, BBP (Bollinger Band %), Stochastic (%K/%D), PPO |

## Documentation

- [Architecture](doc/architecture.md) — Pipeline design, data flow, state space
- [Usage](doc/usage.md) — Workflow for all 4 modes
- [Indicators](doc/indicators.md) — Indicator catalog with parameters
- [Models](doc/models.md) — Model type reference and decision guide
- [Setup](doc/setup.md) — Environment setup for Apple Silicon
