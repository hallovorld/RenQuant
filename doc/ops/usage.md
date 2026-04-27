# Usage Guide

## Workflow Overview

RenQuant uses a four-mode workflow:

| Mode | Tool | When to use |
|------|------|-------------|
| **Research** | Jupyter notebooks | Train models and export JSON artifacts |
| **Validation** | QuantConnect LEAN (Docker) | Rigorous event-driven backtest |
| **Analysis** | `python scripts/analyze_backtest.py --strategy ...` | Visualise LEAN results |
| **Live Trading** | `python -m live.runner` | Paper or real trading via Alpaca/IBKR |

---

## Research Mode (Daily Work)

Start JupyterLab:
```bash
conda activate renquant
jupyter lab
```

Open a strategy notebook. The pipeline runs top-to-bottom:

1. **Setup** — configure paths, load config, import kernel + training modules
2. **Data fetch** — `kernel.data.fetch_ohlcv` (cached as Parquet) for watchlist + SPY + sector ETFs
3. **Regime detection** — Hurst (Layer 1) → CUSUM (Layer 2) → GMM training (Layer 3) → save `spy-gmm-regime.json`
4. **Parallel training** — `TrainingPipeline().run(ctx)` dispatches `FeatureJob`, which fans out `run_ticker_parallel()`: each ticker's `TickerFeatureJob → TickerTournamentJob → TickerExportJob → TickerCalibrationJob` chain runs concurrently in its own worker thread; results are collected back into `TrainingContext`
5. **Summary** — `train_ctx.results` / `train_ctx.calibration_summary` show OOS Sharpe, best model, calibration method per ticker; the notebook cell displays a formatted summary table
6. **Correlation artifact** — `CorrelationJob` (Phase 3) computes 120-day pairwise correlations and saves `watchlist-correlation.json`
7. **Correlation artifact** — computes 120-day pairwise correlations; saves `watchlist-correlation.json`
8. **Portfolio simulation** — regime-aware simulation mirroring LEAN; all 5 exit types, market gates, tiered selection
9. **Charts + stats** — equity vs SPY, drawdown, regime timeline, per-symbol OOS curves, trade log

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
python scripts/backtest_and_analyze.py --strategy renquant_101 --ntfy other  # custom ntfy topic
python scripts/backtest_and_analyze.py --strategy renquant_101 --silent      # no notifications
```

On macOS, append `--open` to open the generated chart PNGs automatically after analysis finishes.

**Notifications** (on by default, `--silent` to disable): after each run, a macOS banner is sent via `terminal-notifier` (`brew install terminal-notifier`) and an iPhone push is sent via [ntfy.sh](https://ntfy.sh) (default topic: `renquant`, override with `--ntfy <topic>`). Install the ntfy app on your iPhone and subscribe to the same topic to receive push notifications.

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

Both export scripts (`export_lean_data.py` and `export_lean_watchlist.py`) check `data/ohlcv/{SYMBOL}/` first, then fall back to `Notebooks/data/ohlcv/{SYMBOL}/` — the path where the notebook kernel caches data when run from the `Notebooks/` directory. If both are missing, fetch with `python -c "import common; common.fetch_ohlcv('SYMBOL')"` from the repo root.

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
  "train_split": 0.70,
  "model_staleness_days": 60,
  "volume_zscore_lookback": 20,
  "volume_filter": {"mode": "percentile", "percentile_threshold": 85},
  "training_years": 3,
  "max_concurrent_positions": 5,
  "risk": {
    "stop_loss_pct": 0.08,
    "portfolio_drawdown_halt_pct": 0.15,
    "regime_filter": {"enabled": false, "symbol": "SPY", "sma_period": 200}
  },
  "sector_map": {"TSLA": "tech", "JPM": "finance", "UNH": "healthcare", "...": "..."},
  "max_positions_per_sector": 3,
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

# Alpaca paper trading
python -m live.runner --strategy renquant_102 --broker alpaca-paper --once

# Real trading (Alpaca)
python -m live.runner --strategy renquant_102 --broker alpaca --once

# Scheduled mode (runs every 24h by default)
python -m live.runner --strategy renquant_101 --broker paper --interval 86400
```

Broker options: `paper`, `alpaca-paper`, `alpaca`, `ibkr`. Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` environment variables for Alpaca.

The runner auto-detects single-stock vs multi-stock strategies by checking for `"watchlist"` in the config. Multi-stock strategies use `run_once_multi()` which computes volume z-scores, processes sell signals, and executes buy orders across the watchlist.

Trade logs are saved to `live/logs/<strategy>/<date>.json`.

**Daily log contents** (renquant_103 / renquant_104): REGIME PARAMS block (regime name, confidence, stop/hold/reserve values) printed after MARKET CONTEXT; price source tag `[Alpaca]` or `[OHLCV <date>]` on each sell decision header; EXIT 3 max_hold entries with realized P&L; per-buy position sizing math (invest, shares, price, conviction multiplier); warnings for all bare exception paths. On 104 runs, `PanelScoringJob` emits a line showing how many candidates + holdings were scored by the panel model; `VetoWeakBuysTask` logs dropped candidates when `buy_floor` is configured; rotation logs reject reasons including `panel_advantage`.

### Daily Automation

Three NYSE-holiday-aware launchd agents run each trading day. **renquant_104 is the active strategy**; 103 scripts remain for rollback.

| Run | Time (PT) | Time (ET) | Script | What it does |
|-----|-----------|-----------|--------|--------------|
| Market open | 6:32 AM | 9:32 AM | `live_only_104.sh --sell-only` | Exit stop-loss / gap-down positions early using today's opening price |
| Pre-close | 12:44 PM | 3:44 PM | `live_only_104.sh --sell-only` | Exit intraday stop breaches before close using near-final daily price |
| After close | 1:55 PM | 4:55 PM | `daily_104.sh` | Full run: `FullTrainingPipeline` (tournament → panel-LTR → recalibrate) → export LEAN data → buy + sell |

```bash
# Manual runs — 104 (active)
bash scripts/daily_104.sh              # full retrain + trade
bash scripts/live_only_104.sh          # intraday sell check (no retrain)
python scripts/train_104.py --skip-baseline --skip-recalibrate  # partial retrain
python -m live.runner --strategy renquant_104 --broker alpaca --once --sell-only

# LaunchAgents (104 active; 103 unloaded)
# ~/Library/LaunchAgents/com.renquant.open104.plist
# ~/Library/LaunchAgents/com.renquant.preclose104.plist
# ~/Library/LaunchAgents/com.renquant.daily104.plist
# ~/Library/LaunchAgents/com.renquant.conditional-retrain104.plist  (13:10 PT Mon-Fri, SPY/VIX anomaly → force retrain)
# ~/Library/LaunchAgents/com.renquant.screen-watchlist.plist        (Sun 12:05 PT, DROP/ADD candidate report)
# Logs: logs/live_104/{date}-open.log, {date}-preclose.log
#       logs/daily_104/{date}.log
#       logs/conditional_retrain_104/{stdout,stderr}.log
#       logs/watchlist_screen/{stdout,stderr}.log

# Manual runs — 103 (legacy, kept for rollback)
bash scripts/daily_103.sh
bash scripts/live_only_103.sh
python -m live.runner --strategy renquant_103 --broker alpaca --once --sell-only
```

**One-shot LaunchAgent install** (run once per new plist):

```bash
# Copy every plist under scripts/launchd/ and load it with launchctl
for p in /Users/renhao/git/github/RenQuant/scripts/launchd/*.plist; do
    cp "$p" ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/$(basename "$p")
done

# To uninstall one:
#   launchctl unload ~/Library/LaunchAgents/com.renquant.screen-watchlist.plist
#   rm           ~/Library/LaunchAgents/com.renquant.screen-watchlist.plist
```

Alpaca credentials are stored in `.env` (gitignored) as `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. Notifications (macOS banner + iPhone via ntfy.sh) are sent on success or failure:
- After retraining: model count (e.g., `Models retrained: 34 watchlist models ready`), or a warning if fewer than 10 models passed the OOS Sharpe floor
- After live trading: trade summary (e.g., `BUY TSLA x15; SELL AMZN x8; STOP COIN (12.3% loss)`) or error details
- Notification body appends current holdings with unrealized P&L (e.g. `BUY AAPL x5 | Held: NVDA+12% META-2%`)

Logs are written to `logs/daily_104/{date}.log`.

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

Available model types: `manual`, `classification`, `qlearning`, `fqi`, `optimization`, `xgboost` (see [doc/arch/models.md](models.md)).

---

## File Reference

```
common/                            # Shared library — import as `import common`
├── config.py                      # load_strategy_config, build_model_path
├── data/                          # fetch_ohlcv (Parquet cache + yfinance/IBKR)
├── indicators/                    # compute_indicators, 12 registered indicators
├── models/                        # BaseModel ABC + 6 implementations + learners/
├── strategy.py                    # StrategyConfig + Strategy composition class
├── portfolio.py                   # compute_portvals, portfolio_stats
├── tax.py                         # compute_trade_tax, load_tax_config, add_tax_columns
└── plotting.py                    # backtest_dashboard, telemetry plots, normalized performance

Notebooks/
├── renquant_101.ipynb            # Single-stock strategy: data → model → export
├── renquant_102.ipynb            # Multi-stock: train 3 approaches per symbol → export best → portfolio simulation
└── backtest_analysis.ipynb       # Post-LEAN analysis: enrich LEAN trades with tax breakdown (add_tax_columns)

backtesting/renquant_103/renquant_103.ipynb   # Adaptive regime multi-stock — reference/rollback
backtesting/renquant_104/renquant_104.ipynb   # Panel-LTR — parallel notebook (training driven by scripts/train_104.py)

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
├── alpaca_broker.py               # AlpacaBroker (Alpaca Markets API, paper + live)
└── ibkr_broker.py                 # IBKRBroker (stub, pending setup)

scripts/
├── export_lean_data.py           # Convert cached parquet OHLCV into LEAN daily equity files
├── export_lean_watchlist.py      # Batch export all watchlist symbols for a strategy
├── backtest_and_analyze.py       # Run LEAN backtest, render charts, send notifications
├── analyze_backtest.py            # Render backtest charts + summary metrics
├── new_strategy.py                # Scaffold a new strategy directory
├── fetch_earnings_calendar.py    # Fetch upcoming earnings dates via yfinance → earnings-calendar.json
├── daily_104.sh                   # Full run: renquant_104 FullTrainingPipeline + live trading pass (ACTIVE)
├── live_only_104.sh               # Sell-only pass for renquant_104 (no retrain, intraday stop checks)
├── train_104.py                   # Thin CLI wrapper over FullTrainingPipeline (baseline / panel / recalibrate)
├── compare_panel_vs_baseline.py  # A/B compare 103 baseline vs 104 panel scorer on the same sim
├── daily_103.sh                   # 103 daily run — kept for rollback
├── live_only_103.sh               # 103 sell-only — kept for rollback
└── daily_102.sh                   # 102 legacy
```
