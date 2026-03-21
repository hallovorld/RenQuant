# Usage Guide

## Workflow Overview

RenQuant uses a three-mode workflow:

| Mode | Notebook / Tool | When to use |
|------|-----------------|-------------|
| **Research** | `Notebooks/test_001_nvda.ipynb` | Train models and export JSON artifacts |
| **Validation** | QuantConnect LEAN (Docker) | Rigorous event-driven backtest |
| **Analysis** | `Notebooks/backtest_analysis.ipynb` | Visualise LEAN results |

---

## Research Mode (Daily Work)

Start JupyterLab:
```bash
conda activate renquant
jupyter lab
```

Open `Notebooks/test_001_nvda.ipynb`. The pipeline runs top-to-bottom:

1. **Cell 1 — Setup + baseline**: configures paths via `import common as rq`, trains a deprecated XGBClassifier baseline
2. **Cell 2 — Data + transitions**: fetches OHLCV for the configured date range, computes MACD/RSI/CCI, applies gate signals, builds RL transition tuples
3. **Cell 3 — FQI training**: runs Fitted Q-Iteration (8 iterations, γ=0.95) to train 3 XGBRegressor Q-value models; produces a state catalog showing the policy
4. **Cell 4 — Export**: saves `*-q-hold.json`, `*-q-buy.json`, `*-q-sell.json`, and `*-policy-metadata.json` to the strategy directory

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
  "initial_cash": 100000,
  "backtest_start": "2022-01-01",
  "backtest_end": "2023-01-01"
}
```

---

## Backtest Analysis

After a LEAN run, open `Notebooks/backtest_analysis.ipynb` to visualize results.

The notebook auto-loads the most recent backtest from `backtests/` and renders a 4-panel dashboard:

| Panel | Content |
|-------|---------|
| **Price + Signals** | Close price with model buy/sell gate signals (▲/▼); LEAN entry/exit points overlaid when trades exist |
| **Equity Curve** | Portfolio value over time with profit/loss shading; falls back to "no data" when LEAN produced 0 trades |
| **Drawdown** | Rolling drawdown (%) with max drawdown labelled |
| **Statistics** | Full performance table: win rate, Sharpe, Sortino, CAR, max drawdown, alpha, beta, fees |

The dashboard is also saved as `dashboard.png` in the backtest run directory.

---

## Adding a New Strategy

1. **Copy the notebook**
   ```bash
   cp Notebooks/test_001_nvda.ipynb Notebooks/<new_strategy>.ipynb
   ```

2. **Create a new LEAN strategy directory**
   ```bash
   cp -r backtesting/test_001_nvda backtesting/<new_strategy>
   ```

3. **Update `strategy_config.json`** in the new directory — change `model_name`, `stock_symbol`, and date range.

4. **Run the notebook** — in Cell 1, update `STRATEGY_DIR` to point at the new strategy directory, then run all cells. Models are exported directly into the strategy directory:
   - `<model_name>-q-hold.json`
   - `<model_name>-q-buy.json`
   - `<model_name>-q-sell.json`
   - `<model_name>-policy-metadata.json`

5. **Update `main.py`** — rename the class (e.g. `XGBoostAAPLStrategy`) and verify `CONFIG` loads the right `model_name`.

6. **Validate**:
   ```bash
   cd backtesting/<new_strategy>
   lean backtest .
   ```

---

## File Reference

```
common/                            # Shared library — import as `import common as rq`
├── config.py                      # load_strategy_config, build_model_path
├── data.py                        # fetch_ohlcv (OpenBB)
├── indicators.py                  # compute_macd / rsi / cci, add_indicators
├── features.py                    # add_gate_signals, build_transitions, STATE_COLUMNS
├── training.py                    # fitted_q_iteration, score_valid_actions
└── plotting.py                    # backtest_dashboard, load_latest_backtest, plot helpers

Notebooks/
├── test_001_nvda.ipynb            # Strategy research: data → FQI → export JSON models
└── backtest_analysis.ipynb        # Backtest visualisation: signals, equity, drawdown, stats

backtesting/<strategy>/
├── main.py                        # LEAN QCAlgorithm — loads models, runs daily inference
├── config.py                      # LEAN-local config loader (self-contained for Docker)
├── strategy_config.json           # Symbol, dates, cash — edit this to change backtest params
├── config.json                    # LEAN entry point (rarely needs editing)
├── <model>-policy-metadata.json   # State columns, indicator params, gate rules
├── <model>-q-hold.json            # XGBoost Q-value model for hold action
├── <model>-q-buy.json             # XGBoost Q-value model for buy action
├── <model>-q-sell.json            # XGBoost Q-value model for sell action
└── backtests/<timestamp>/         # LEAN output: result JSON, logs, dashboard.png
```
