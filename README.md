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
│   ├── indicators/          # Registry-based: 12 indicators (momentum, volatility, trend, volume)
│   ├── models/              # BaseModel ABC + 5 implementations
│   │   └── learners/        # RTLearner, BagLearner, TabularQLearner
│   ├── strategy.py          # StrategyConfig + Strategy composition
│   ├── portfolio.py         # Local portfolio simulator
│   ├── plotting.py          # Backtest dashboard + normalized performance chart
│   └── config.py            # Config loading utilities
├── Notebooks/               # Research notebooks (renquant_101, renquant_102)
├── backtesting/             # LEAN strategies (self-contained, no common/ imports)
│   ├── renquant_101/        # Single-stock classification strategy
│   └── renquant_102/        # Multi-stock pre-trained scanner strategy
│       └── models/{SYMBOL}/ # Per-symbol model artifacts
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
python scripts/new_strategy.py --name my_nvda --symbol NVDA --type classification
```

If you want a known-good example that already exists in the repo, use `renquant_101` in the commands below instead.

### 2. Research — train in notebook
```bash
jupyter lab  # open notebook, run all cells to train and export models
```

### 3. Backtest with LEAN
```bash
cd backtesting/my_nvda && lean backtest .
```

If you want the performance charts to be generated immediately after the backtest completes, use the wrapper instead:

```bash
python scripts/backtest_and_analyze.py --strategy my_nvda
```

On macOS, add `--open` to open the generated PNG files automatically.

Backtest execution respects `wash_sale_days`, `min_hold_days`, `max_hold_days`, and `position_sizing` from `strategy_config.json`.

If LEAN is missing local daily equity data for a symbol, export it from the cached parquet store first:

```bash
python scripts/export_lean_data.py --symbol NVDA
```

### 4. Analyze results
```bash
python scripts/analyze_backtest.py --strategy my_nvda
```

### 5. Live trading
```bash
# Paper trading (test)
python -m live.runner --strategy my_nvda --broker paper --once

# Real trading (requires IBKR setup)
python -m live.runner --strategy my_nvda --broker ibkr
```

## Relative Indicator Framework

All indicators are computed relative to SPY to answer "is the stock outperforming the market?" rather than "is the stock going up?" This prevents bull-market bias where every stock looks like a buy.

- **Ratio** (`stock / SPY`) for always-positive indicators: RSI, ADX
- **Difference** (`stock - SPY`) for zero-crossing indicators: MACD hist, CCI, BBP, Williams %R, OBV slope
- **Trend features**: price/50EMA, price/200EMA, 20d/60d relative momentum

## Model Types

| Type | Method | Use Case |
|------|--------|----------|
| **Manual** | Dual Momentum + trend following | Baseline, interpretable, no ML |
| **Classification** | Random Forest on forward-return labels | Best default, handles relative features well |
| **Q-Learning** | Tabular Q with relative reward | Model-free RL, small state space |
| **FQI** | Fitted Q-Iteration with XGBoost | Function approximation RL |
| **Optimization** | Nelder-Mead parameter search + inner model | Auto-tuning |

## Trading Constraints

| Constraint | Value | Purpose |
|------------|-------|---------|
| Wash sale | 30 days | Cannot buy within 30 days of selling |
| Min hold | 20 days | Prevents short-term trading |
| Max hold | 150 days | Forces position review |

## Position Sizing

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Max position | 33% of portfolio | No single stock exceeds 1/3 of total value |
| Cash reserve | 10% of portfolio | Always maintain cash buffer |

Rules: cash-only buys (never sell to fund a new buy), whole shares only. Configured in `strategy_config.json` under `position_sizing`.

## Strategies

### renquant_101 — Single-Stock Classification

Trains a classification model (BagLearner/RTLearner) on relative indicators (stock vs SPY) for a single symbol. Notebook trains 3 model types (Manual, Classification, Q-Learning), exports the best by Sharpe.

### renquant_102 — Multi-Stock Pre-Trained Scanner

3-stage pipeline: **DETECT** (volume z-score spike) → **CONFIRM** (pre-trained model) → **EXECUTE** (trade). Scans a watchlist of 30 stocks/ETFs for volume z-score spikes (default threshold: 2.0σ, lookback: 15 days). The notebook trains 4 approaches per symbol (Dual Momentum, Classification/RF, Q-Learning, Mean Reversion) on a rolling 2yr window, exports the best by Sharpe to `models/{SYMBOL}/`. The notebook also includes a portfolio-level simulation that replicates the LEAN multi-stock logic in Python (volume z-score scan → model confirmation → position management), with a 4-panel dashboard (equity vs SPY, drawdown, positions held, cash allocation) for iterating on parameters before running LEAN. LEAN loads pre-trained models and applies them on spike days. Models older than 30 days are skipped (configurable via `model_staleness_days`). Max 3 concurrent positions.

```bash
# Train and compare approaches
jupyter lab  # run Notebooks/renquant_102.ipynb

# Backtest
cd backtesting/renquant_102 && lean backtest .

# Live (paper)
python -m live.runner --strategy renquant_102 --broker paper --once
```

## Indicator Library

All indicators share a uniform API: `(df, **params) -> DataFrame`

| Category | Indicators |
|----------|-----------|
| Momentum | RSI, MACD (line/signal/hist), EMA, Momentum, Williams %R |
| Volatility | CCI, BBP (Bollinger Band %), Stochastic (%K/%D), PPO, ATR |
| Trend | ADX (+DI/-DI) |
| Volume | OBV (with EMA signal + slope) |

## Documentation

- [Architecture](doc/architecture.md) — Pipeline design, data flow, state space
- [Usage](doc/usage.md) — Workflow for all 4 modes
- [Indicators](doc/indicators.md) — Indicator catalog with parameters
- [Models](doc/models.md) — Model type reference and decision guide
- [Setup](doc/setup.md) — Environment setup for Apple Silicon

The dashboard now includes decision telemetry from LEAN runtime output, so you can inspect model score, buy/sell thresholds, and the final constrained action alongside price, equity, and drawdown.
