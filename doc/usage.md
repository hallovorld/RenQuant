# Usage Guide

## Workflow Overview

RenQuant uses a two-mode workflow to keep iteration fast without sacrificing backtest rigor:

| Mode | Tool | When to use |
|------|------|-------------|
| **Research** | `backtesting.py` inside notebook | Exploring ideas, tuning parameters, fast feedback |
| **Validation** | QuantConnect LEAN (Docker) | Final verification before committing to a strategy |

---

## Research Mode (Daily Work)

Start JupyterLab:
```bash
conda activate renquant
jupyter lab
```

Open the relevant notebook in `Notebooks/`. The notebook pipeline runs top-to-bottom:

1. **Cell 1 — Data fetch**: pulls OHLCV history via OpenBB or yfinance
2. **Cell 2 — Supervised baseline**: trains an XGBClassifier on return direction (sanity check)
3. **Cell 3 — RL transitions**: generates `(state, action, reward, next_state)` tuples for all position/signal combinations
4. **Cell 4 — Fitted Q-Iteration**: trains 3 XGBRegressor models (one per action) over 8 iterations, exports JSON artifacts

For quick strategy iteration, use `backtesting.py` at the end of Cell 4 to run a fast in-notebook backtest before exporting to LEAN.

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

4. **Run the notebook** — update the ticker and dates at the top, then run all cells. This exports:
   - `<model_name>-q-hold.json`
   - `<model_name>-q-buy.json`
   - `<model_name>-q-sell.json`
   - `<model_name>-policy-metadata.json`

5. **Copy models** to the new strategy directory (the notebook exports them to a path configured in the notebook).

6. **Update `main.py`** — rename the class (e.g. `XGBoostAAPLStrategy`) and verify `CONFIG` loads the right `model_name`.

7. **Validate**:
   ```bash
   cd backtesting/<new_strategy>
   lean backtest .
   ```

---

## File Reference

```
backtesting/<strategy>/
├── main.py                        # LEAN QCAlgorithm — loads models, runs daily inference
├── config.py                      # Config loader utilities
├── strategy_config.json           # Symbol, dates, cash — edit this to change backtest params
├── config.json                    # LEAN entry point (rarely needs editing)
├── <model>-policy-metadata.json   # State columns, indicator params, gate rules
├── <model>-q-hold.json            # XGBoost Q-value model for hold action
├── <model>-q-buy.json             # XGBoost Q-value model for buy action
└── <model>-q-sell.json            # XGBoost Q-value model for sell action
```
