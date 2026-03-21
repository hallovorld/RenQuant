# Architecture

## Design Principle: Glass-Box Pipeline

RenQuant is built around **strict layer decoupling**. Each layer has one job, communicates via well-defined interfaces (JSON files), and can be developed or replaced independently. Every decision in the pipeline is inspectable — no end-to-end black boxes.

---

## Three-Layer Pipeline

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Research (Notebooks/)                     │
│  - Fetch OHLCV via OpenBB / yfinance                │
│  - Compute MACD, RSI, CCI                           │
│  - Generate RL transitions                          │
│  - Train XGBoost Q-models via Fitted Q-Iteration    │
│  - Export: 3x *.json models + policy-metadata.json  │
└───────────────────────┬─────────────────────────────┘
                        │ JSON artifacts
┌───────────────────────▼─────────────────────────────┐
│  Layer 2: Model (backtesting/<strategy>/*.json)     │
│  - Q(state, hold)  → XGBRegressor JSON              │
│  - Q(state, buy)   → XGBRegressor JSON              │
│  - Q(state, sell)  → XGBRegressor JSON              │
│  - policy-metadata → state cols, indicator params,  │
│                       gate rules, transaction cost   │
└───────────────────────┬─────────────────────────────┘
                        │ loaded at Initialize()
┌───────────────────────▼─────────────────────────────┐
│  Layer 3: Backtesting (LEAN / Docker)               │
│  - Load models at Initialize()                      │
│  - Each trading day:                                │
│    1. Fetch 60-day history                          │
│    2. Compute features (MACD, RSI, CCI)             │
│    3. Apply gate rules                              │
│    4. argmax Q(state, action) → order               │
│  - Results: backtests/<timestamp>/<id>.json         │
└─────────────────────────────────────────────────────┘
                        │ JSON results
┌───────────────────────▼─────────────────────────────┐
│  Analysis (Notebooks/backtest_analysis.ipynb)       │
│  - Load latest LEAN result via common.plotting      │
│  - Price chart + model buy/sell signals             │
│  - Equity curve + drawdown (from LEAN trades)       │
│  - Performance statistics table                     │
└─────────────────────────────────────────────────────┘
```

---

## Shared Library: `common/`

All reusable logic lives in `common/` and is imported by notebooks as `import common as rq`. It is **not** available inside the LEAN Docker container — the backtesting layer remains self-contained.

| Module | Contents |
|--------|----------|
| `common/config.py` | `load_strategy_config`, `split_date_parts`, `build_model_path` |
| `common/data.py` | `fetch_ohlcv` — OpenBB/yfinance data fetching |
| `common/indicators.py` | `compute_macd`, `compute_rsi`, `compute_cci`, `add_indicators` |
| `common/features.py` | `add_gate_signals`, `build_transitions`, `STATE_COLUMNS` |
| `common/training.py` | `fitted_q_iteration`, `score_valid_actions` |
| `common/plotting.py` | `backtest_dashboard`, `load_latest_backtest`, parse/plot utilities |

---

## Layer 1: Research

**Location**: `Notebooks/`
**Environment**: `renquant` conda env

The notebook is the only place where training happens. It follows a 4-step pipeline:

1. **Data ingestion** — fetches daily OHLCV for a given ticker and date range
2. **Supervised baseline** — trains an XGBClassifier on next-day return direction as a sanity check
3. **Transition generation** — for each trading day, computes `(state, action, reward, next_state)` tuples:
   - `state`: [macd_line, macd_signal, macd_hist, rsi, cci, position_flag]
   - `action`: one of {hold, buy, sell}
   - `reward`: immediate % return minus transaction cost (5 bps)
   - Actions gated by buy/sell signals (MACD crossover + RSI confirmation)
4. **Fitted Q-Iteration (FQI)** — trains Q-value estimators over 8 iterations:
   - Initializes Q(s,a) = immediate reward
   - Each iteration: `Q_target = reward + γ * max_a' Q(s', a')`, refits XGBRegressor
   - `γ = 0.95` (values future rewards)
   - Produces 3 separate XGBRegressor models, one per action

---

## Layer 2: Model Artifacts

**Location**: `backtesting/<strategy>/`

All artifacts are JSON (not pickle) — required for LEAN compatibility and human-readability.

| File | Contents |
|------|----------|
| `*-q-hold.json` | XGBoost model: Q(state, hold) |
| `*-q-buy.json` | XGBoost model: Q(state, buy) |
| `*-q-sell.json` | XGBoost model: Q(state, sell) |
| `*-policy-metadata.json` | State columns, indicator parameters, gate rules, γ, transaction cost |

The policy metadata acts as a contract between Layer 1 and Layer 3 — both must use identical indicator parameters.

---

## Layer 3: Backtesting

**Location**: `backtesting/<strategy>/main.py`
**Runtime**: QuantConnect LEAN engine (Docker)

`main.py` implements `QCAlgorithm`:

- **`Initialize()`** — reads `strategy_config.json`, loads policy metadata and 3 XGBoost models, sets 40-day warmup
- **`OnData()`** — called once per trading day:
  1. Fetches 60-day price history
  2. Computes MACD(12,26,9), RSI(14), CCI(20) inline
  3. Determines buy/sell gate signals (MACD crossover + RSI threshold)
  4. Scores eligible actions via `model.predict(state)`; ineligible actions scored `-inf`
  5. Executes `argmax Q(state, action)`: buy → `SetHoldings(1.0)`, sell → `Liquidate()`

**Gate rules** (from policy metadata):
- Buy eligible: MACD line crosses above signal AND RSI > 50 AND currently flat
- Sell eligible: MACD line crosses below signal AND RSI < 50 AND currently long
- Hold: always eligible (fallback)

---

## State Space

All strategies share the same 6-feature state vector:

| Feature | Description | Parameters |
|---------|-------------|------------|
| `macd_line` | EMA(12) − EMA(26) | fast=12, slow=26 |
| `macd_signal` | EMA(9) of macd_line | signal=9 |
| `macd_hist` | macd_line − macd_signal | — |
| `rsi` | Relative Strength Index | period=14 |
| `cci` | Commodity Channel Index | period=20 |
| `position_flag` | 1 if long, 0 if flat | — |

Parameters are stored in `policy-metadata.json` and must be identical in both the notebook (training) and `main.py` (inference).

---

## Data Flow Summary

```
yfinance / OpenBB
       ↓
  OHLCV DataFrame
       ↓
  MACD, RSI, CCI computation
       ↓
  RL transition tuples (s, a, r, s')
       ↓
  Fitted Q-Iteration (8 iterations, γ=0.95)
       ↓
  3x XGBRegressor → exported as JSON
       ↓
  LEAN loads at Initialize()
       ↓
  Each day: history → features → gate check → argmax Q → order
```
