# RenQuant

Personal quantitative trading workstation for Apple Silicon. Glass-box pipeline: data ingestion, ML signal generation, backtesting (LEAN), and live trading (Alpaca/IBKR).

## Architecture

```
  Data Layer          Model Layer              Execution Layer
 ┌──────────┐    ┌──────────────────┐    ┌────────────────────────┐
 │ yfinance  │───>│ Manual (rules)   │───>│ LEAN backtest (Docker) │
 │ IBKR      │    │ Classification   │    │ Live trader (Alpaca)   │
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
│   │                        #   + regime detection: Hurst, CUSUM, RegimeGMM (not registered)
│   ├── models/              # BaseModel ABC + 6 implementations
│   │   └── learners/        # RTLearner, BagLearner, TabularQLearner
│   ├── strategy.py          # StrategyConfig + Strategy composition
│   ├── portfolio.py         # Local portfolio simulator
│   ├── tax.py               # After-tax return computation (ST/LT capital gains)
│   ├── plotting.py          # Backtest dashboard + normalized performance chart
│   └── config.py            # Config loading utilities
├── Notebooks/               # Research notebooks (renquant_101, renquant_102, renquant_103)
├── backtesting/             # LEAN strategies (self-contained, no common/ imports)
│   ├── renquant_101/        # Single-stock classification strategy
│   ├── renquant_102/        # Multi-stock pre-trained scanner (21 symbols)
│   │   └── models/{SYMBOL}/ # Per-symbol model artifacts
│   └── renquant_103/        # Adaptive regime multi-stock (24 symbols, 3-layer regime detector)
│       └── models/{SYMBOL}/ # Per-symbol model artifacts
├── live/                    # Live trading runner + broker abstraction
├── scripts/                 # Scaffolding + data pipeline tools
├── data/                    # Local Parquet cache (gitignored)
└── doc/                     # Detailed documentation + strategy design specs
```

## Quick Start

```bash
# Setup
conda create -n renquant python=3.10
conda activate renquant
pip install pandas numpy matplotlib seaborn yfinance scikit-learn xgboost jupyterlab pyarrow
pip install "openbb[all]" openbb-cli backtesting scipy
pip install lean alpaca-py
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
python scripts/backtest_and_analyze.py --strategy my_nvda --silent  # no notifications
```

On macOS, add `--open` to open the generated PNG files automatically. Notifications (macOS banner + iPhone push via ntfy.sh) are sent by default after each run; use `--silent` to disable.

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

# Alpaca paper trading
python -m live.runner --strategy my_nvda --broker alpaca-paper --once

# Real trading (Alpaca)
python -m live.runner --strategy my_nvda --broker alpaca --once
```

### Daily automation (renquant_103)

`scripts/daily_103.sh` retrains models and trades via Alpaca, scheduled weekdays at 1:55 PM PST (4:55 PM EST, after market close) via macOS launchd. Automatically skips US market holidays via NYSE calendar check (`pandas-market-calendars`). Sends trade summary notifications (macOS + iPhone/ntfy). See [doc/usage.md](doc/usage.md) for setup details.

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
| Min hold | 1 day (102) / 20 days (103) | Prevents noise-driven exits in early holding period |
| Max hold | 500 days | Forces position review (allows long-term tax rate) |

## Position Sizing

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Max position | 30% of portfolio | No single stock exceeds 30% of total value |
| Cash reserve | 0% of portfolio | All capital available for positions |

Rules: cash-only buys (never sell to fund a new buy), whole shares only. Configured in `strategy_config.json` under `position_sizing`.

## Tax-Aware Returns

All strategies compute after-tax returns using configurable capital gains rates:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Short-term rate | 50% | Gains on positions held < 365 days |
| Long-term rate | 32% | Gains on positions held ≥ 365 days |

Losses pass through untaxed (loss harvesting not modeled). Tax is deducted at each sell in notebook simulations, producing after-tax equity curves and Sharpe ratios. LEAN strategies report tax as metadata via `SetRuntimeStatistic()`. Configured in `strategy_config.json` under `tax`.

## Strategies

### renquant_101 — Single-Stock Classification

Trains a classification model (BagLearner/RTLearner) on relative indicators (stock vs SPY) for a single symbol. Notebook trains 3 model types (Manual, Classification, Q-Learning), exports the best by Sharpe.

### renquant_102 — Multi-Stock Pre-Trained Scanner

3-stage pipeline: **DETECT** (bullish volume spike) → **CONFIRM** (pre-trained model) → **EXECUTE** (trade). Scans a watchlist of 24 stocks/ETFs for volume spikes on up-close days (default: P85 percentile of 20-day lookback, adaptive per-stock). The notebook trains 3 approaches per symbol (Dual Momentum, Classification/RF, Q-Learning) on a 70/30 walk-forward train/test split of a rolling 3yr window (`training_years=3`), exports the best by OOS after-tax Sharpe to `models/{SYMBOL}/` (floor: 0.8 OOS). Orphan model directories are purged on each run. The notebook also includes a portfolio-level simulation for parameter tuning before running LEAN. LEAN loads pre-trained models, applies risk management: per-position 8% stop-loss, 15% portfolio drawdown circuit breaker, configurable regime filter (currently disabled — relative features encode market context), and sector concentration guard (max 3 per sector). Models older than 60 days are skipped (`model_staleness_days`). Max 5 concurrent positions.

```bash
# Train and compare approaches
jupyter lab  # run Notebooks/renquant_102.ipynb

# Backtest
cd backtesting/renquant_102 && lean backtest .

# Live (paper)
python -m live.runner --strategy renquant_102 --broker paper --once
```

### renquant_103 — Adaptive Regime Multi-Stock

Successor to 102, designed for volatile/choppy markets. Adds a **3-layer regime detector** on top of 102's architecture:

- **Layer 1 (Hurst)**: Rolling 63-day Hurst exponent. H > 0.55 = momentum, H < 0.45 = mean-reversion, ambiguous in between.
- **Layer 2 (CUSUM)**: Changepoint detection — flags regime transitions within 2–5 days. Triggers a 3-bar uncertainty window (no new buys, tighter stops).
- **Layer 3 (GMM)**: Gaussian Mixture Model on 4 SPY features outputs continuous P(regime), used to scale position sizes smoothly.

Four regimes with distinct parameters: `BULL_CALM` (momentum entry, 15% max position, 15% stop, 20% trailing-stop trigger / 18% trail, `max_hold_days=500`), `BULL_VOLATILE` (capitulation entry on high-volume down-close, 20% max, 5% stop, `max_hold_days=500`), `CHOPPY` (divergence-from-SPY entry, 15% max, 5% stop, `max_hold_days=10`), `BEAR` (all offensive buys blocked; up to 1 defensive position in GLD/TLT/XLV/XLU at 15% of portfolio; existing positions held until stop-loss or sell signal). Entry gates: **SPY EMA50 trend gate** blocks all new buys when SPY is below its 50-day EMA; **SPY velocity crash filter** blocks if SPY fell >3% in last 3 days. Stock selection pipeline: earnings filter (±3 days) → regime-conditional scan → relative-strength ranking vs sector ETF → live model score ranking (50/50 blend, uses continuous `predict_score_bulk()` not static Sharpe) → min_model_score filter (0.10) → correlation-aware greedy selection (threshold 0.70) → sector guard (max 3 per sector). Watchlist (24 symbols): equity names minus ARKK/SHOP/COIN, plus GLD, TLT, XLV, XLU as counter-cyclical defensives. Classification model trained with relative-close prices (`stock / SPY × 100`) so labels measure outperformance vs SPY rather than raw returns — prevents the bull-market always-buy bias. OOS Sharpe floor: 0.8 (matching renquant_102). `min_hold_days: 20` and 3-consecutive-sell-signal requirement prevent noise exits and extend avg hold to 150-200 days, improving long-term tax treatment. Training: fixed cutoff 2024-01-01 for stable OOS simulation; live models retrained on last 4 years and re-exported daily. Q-Learning models use a deterministic per-ticker seed (`abs(hash(ticker)) % 2^32`) so daily retraining produces stable, reproducible model selections. **Active daily strategy** — replaces renquant_102 as the live trading engine, scheduled via `scripts/daily_103.sh`. Unit tests: 108 tests in `tests/` covering all policies (`python -m pytest tests/ -v`).

```bash
# Refresh earnings calendar (weekly)
python scripts/fetch_earnings_calendar.py --strategy renquant_103

# Train
jupyter lab  # run Notebooks/renquant_103.ipynb

# Export LEAN data (includes new symbols GLD, TLT, XLV, XLU)
python scripts/export_lean_watchlist.py --strategy renquant_103

# Backtest
cd backtesting/renquant_103 && lean backtest .

# Live (paper)
python -m live.runner --strategy renquant_103 --broker paper --once
```

Full design specification: [`doc/renquant_103_design.md`](doc/renquant_103_design.md).

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
