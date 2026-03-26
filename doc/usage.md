# Usage Guide

## Workflow Overview

RenQuant uses a four-mode workflow:

| Mode | Tool | When to use |
|------|------|-------------|
| **Research** | Jupyter notebooks | Train models and export JSON artifacts |
| **Validation** | QuantConnect LEAN (Docker) | Rigorous event-driven backtest |
| **Analysis** | `Notebooks/backtest_analysis.ipynb` | Visualise LEAN results |
| **Live Trading** | `python -m live.runner` | Paper or real trading via IBKR |

---

## Research Mode (Daily Work)

Start JupyterLab:
```bash
conda activate renquant
jupyter lab
```

Open a strategy notebook. The pipeline runs top-to-bottom:

1. **Setup** — configure paths via `import common`, define strategy gate signals
2. **Data + indicators** — `common.fetch_ohlcv` (cached as Parquet) → `common.compute_indicators`
3. **Model training** — depends on model type (see [doc/models.md](models.md))
4. **Export** — `model.save()` writes JSON artifacts to the strategy directory

---

## Validation Mode (LEAN Backtest)

Run only after the notebook research is complete and models are exported.

```bash
cd backtesting/test_001_nvda
lean backtest .
```

Results land in `backtesting/test_001_nvda/backtests/<timestamp>/`.

To adjust the backtest period or initial capital, edit `strategy_config.json`:
```json
{
  "model_name": "test-001-nvda",
  "stock_symbol": "NVDA",
  "model_type": "classification",
  "initial_cash": 100000,
  "backtest_start": "2022-01-01",
  "backtest_end": "2023-01-01"
}
```

---

## Backtest Analysis

After a LEAN run, open `Notebooks/backtest_analysis.ipynb` to visualize results.

The notebook auto-loads the most recent backtest and renders a 4-panel dashboard:

| Panel | Content |
|-------|---------|
| **Price + Signals** | Close price with model buy/sell signals; LEAN entry/exit points overlaid |
| **Equity Curve** | Portfolio value over time with profit/loss shading |
| **Drawdown** | Rolling drawdown (%) with max drawdown labelled |
| **Statistics** | Performance table: win rate, Sharpe, Sortino, max drawdown, alpha, beta, fees |

A normalized performance chart with long/short entry markers is also available via `common.plot_normalized_performance`.

---

## Live Trading Mode

Run a trained strategy with paper or real broker:

```bash
# Paper trading (test, no real money)
python -m live.runner --strategy test_001_nvda --broker paper --once

# Real trading (requires IBKR TWS/Gateway)
python -m live.runner --strategy test_001_nvda --broker ibkr

# Scheduled mode (runs every 24h by default)
python -m live.runner --strategy test_001_nvda --broker paper --interval 86400
```

Trade logs are saved to `live/logs/<strategy>/<date>.json`.

---

## Adding a New Strategy

The recommended approach uses the scaffold script:

```bash
python scripts/new_strategy.py --name nvda_rf --symbol NVDA --type classification
```

This creates `backtesting/nvda_rf/` with `strategy_config.json` and `config.json`.

Then:

1. **Train in a notebook** — use `common.compute_indicators`, `common.create_model`, or the `Strategy` class
2. **Export artifacts** — `model.save(strategy_dir, model_name)` writes JSON to the strategy directory
3. **Backtest** — `cd backtesting/nvda_rf && lean backtest .`
4. **Analyze** — open `backtest_analysis.ipynb`, set `STRATEGY_DIR` to the new strategy
5. **Live trade** — `python -m live.runner --strategy nvda_rf --broker paper --once`

Available model types: `manual`, `classification`, `qlearning`, `fqi`, `optimization` (see [doc/models.md](models.md)).

---

## File Reference

```
common/                            # Shared library — import as `import common`
├── config.py                      # load_strategy_config, build_model_path
├── data/                          # fetch_ohlcv (Parquet cache + yfinance/IBKR)
├── indicators/                    # compute_indicators, 12 registered indicators
├── models/                        # BaseModel ABC + 5 implementations + learners/
├── strategy.py                    # StrategyConfig + Strategy composition class
├── portfolio.py                   # compute_portvals, portfolio_stats
└── plotting.py                    # backtest_dashboard, plot_normalized_performance

Notebooks/
├── test_001_nvda.ipynb            # Strategy research: data → model → export
└── backtest_analysis.ipynb        # Backtest visualisation: signals, equity, stats

backtesting/<strategy>/
├── main.py                        # LEAN QCAlgorithm — loads models, runs daily inference
├── config.py                      # LEAN-local config loader (self-contained for Docker)
├── strategy_config.json           # Symbol, dates, cash, model type, indicators
├── config.json                    # LEAN entry point (rarely needs editing)
├── *-policy-metadata.json         # Contract: state columns, indicator params, model type
└── *.json                         # Model artifacts (varies by model type)

live/
├── runner.py                      # Entry point: python -m live.runner
├── broker.py                      # BaseBroker ABC
├── paper_broker.py                # PaperBroker (simulated fills)
└── ibkr_broker.py                 # IBKRBroker (stub, pending setup)

scripts/
└── new_strategy.py                # Scaffold a new strategy directory
```
