# Usage Guide

## Workflow Overview

RenQuant uses a four-mode workflow:

| Mode | Tool | When to use |
|------|------|-------------|
| **Research** | Jupyter notebooks | Train models and export JSON artifacts |
| **Validation** | QuantConnect LEAN (Docker) | Rigorous event-driven backtest |
| **Analysis** | `python scripts/analyze_backtest.py --strategy ...` | Visualise LEAN results |
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
cd backtesting/renquant_101
lean backtest .
```

`lean backtest .` itself only prints logs and summary stats in the terminal. If you want charts immediately after the run, use the wrapper script instead:

```bash
python scripts/backtest_and_analyze.py --strategy renquant_101
```

On macOS, append `--open` to open the generated chart PNGs automatically after analysis finishes.

Results land in `backtesting/renquant_101/backtests/<timestamp>/`.

The LEAN strategy enforces config-backed execution constraints and position sizing during backtests:

- `wash_sale_days`: blocks a new buy for `N` calendar days after a sell
- `min_hold_days`: blocks a sell until a position has been held for `N` calendar days
- `max_hold_days`: forces a sell if a position has been held for `N` calendar days
- `position_sizing.max_position_pct`: caps any single position at this fraction of portfolio value
- `position_sizing.cash_reserve_pct`: always maintains this fraction of portfolio as cash reserve

If LEAN reports missing local symbol files such as `/equity/usa/daily/nvda.zip`, export the cached parquet data into LEAN format first:

```bash
python scripts/export_lean_data.py --symbol NVDA
```

To adjust the backtest period or initial capital, edit `strategy_config.json`.

**Single-stock** (renquant_101):
```json
{
  "model_name": "renquant-101",
  "stock_symbol": "TSLA",
  "model_type": "classification",
  "initial_cash": 10000,
  "tax": {"short_term_rate": 0.50, "long_term_rate": 0.32, "long_term_threshold_days": 365},
  "backtest_start": "2024-01-01",
  "backtest_end": "2026-03-26"
}
```

**Multi-stock pre-trained scanner** (renquant_102):
```json
{
  "model_name": "renquant-102",
  "watchlist": ["TSLA", "AMZN", "GOOG", "MSFT", "AMD", "NFLX", "CRM", "PLTR", "COIN", "SHOP", "..."],
  "benchmark": "SPY",
  "initial_cash": 100000,
  "model_staleness_days": 30,
  "volume_zscore_lookback": 15,
  "volume_zscore_threshold": 2.0,
  "training_years": 2,
  "max_concurrent_positions": 3,
  "tax": {"short_term_rate": 0.50, "long_term_rate": 0.32, "long_term_threshold_days": 365},
  "backtest_start": "2024-01-01",
  "backtest_end": "2026-03-26"
}
```

---

## Backtest Analysis

After a LEAN run, run the analysis script to render charts and print summary statistics.

```bash
python scripts/analyze_backtest.py --strategy renquant_101
```

The script auto-loads the most recent backtest, writes charts to the run directory, and prints a metric summary.

When the algorithm emits runtime statistics in `OnEndOfAlgorithm`, the analysis output also includes execution metrics such as buy/sell/hold decision counts, executed orders, and blocked wash-sale or minimum-hold actions.

When the LEAN strategy emits decision telemetry, the dashboard also plots model score, buy/sell thresholds, and final constrained actions so you can see why a trade fired or was suppressed.

| Panel | Content |
|-------|---------|
| **Price + Trades** | Close price with LEAN entry/exit points overlaid |
| **Decision Telemetry** | Model score, thresholds, and final buy/hold/sell actions |
| **Equity Curve** | Portfolio value over time with profit/loss shading |
| **Drawdown** | Rolling drawdown (%) with max drawdown labelled |
| **Statistics** | Performance table: win rate, Sharpe, Sortino, max drawdown, alpha, beta, fees |

It also writes a normalized performance chart against buy-and-hold when LEAN produced an equity series.

When the `tax` block is present in `strategy_config.json`, LEAN strategies report after-tax metrics via `SetRuntimeStatistic()` (total tax, short-term/long-term trade counts). The analysis notebook (`backtest_analysis.ipynb`) uses `common.add_tax_columns()` to enrich closed trades with per-trade holding days, tax rate, tax amount, and after-tax P&L.

---

## Live Trading Mode

Run a trained strategy with paper or real broker:

```bash
# Paper trading (test, no real money)
python -m live.runner --strategy renquant_101 --broker paper --once

# Multi-stock strategy
python -m live.runner --strategy renquant_102 --broker paper --once

# Real trading (requires IBKR TWS/Gateway)
python -m live.runner --strategy renquant_101 --broker ibkr

# Scheduled mode (runs every 24h by default)
python -m live.runner --strategy renquant_101 --broker paper --interval 86400
```

The runner auto-detects single-stock vs multi-stock strategies by checking for `"watchlist"` in the config. Multi-stock strategies use `run_once_multi()` which computes volume z-scores, processes sell signals, and executes buy orders across the watchlist.

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
4. **Analyze** — run `python scripts/analyze_backtest.py --strategy nvda_rf`
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
├── tax.py                         # compute_trade_tax, load_tax_config, add_tax_columns
└── plotting.py                    # backtest_dashboard, telemetry plots, normalized performance

Notebooks/
├── renquant_101.ipynb            # Single-stock strategy: data → model → export
└── renquant_102.ipynb            # Multi-stock: train 3 approaches per symbol → export best → portfolio simulation

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
├── export_lean_data.py           # Convert cached parquet OHLCV into LEAN daily equity files
├── backtest_and_analyze.py       # Run LEAN backtest, then render/open charts
├── analyze_backtest.py            # Render backtest charts + summary metrics
└── new_strategy.py                # Scaffold a new strategy directory
```
