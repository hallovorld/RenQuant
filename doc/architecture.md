# Architecture

## Design Principle: Glass-Box Pipeline

RenQuant is built around **strict layer decoupling**. Each layer has one job, communicates via well-defined interfaces (JSON files), and can be developed or replaced independently. Every decision in the pipeline is inspectable — no end-to-end black boxes.

---

## Four-Layer Pipeline

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Research (Notebooks/)                     │
│  - Fetch OHLCV (yfinance/IBKR, cached as Parquet)  │
│  - Compute indicators via registry                  │
│  - Train model (Manual/RF/QL/FQI/Optimization)      │
│  - Export: JSON model artifacts + policy-metadata    │
└───────────────────────┬─────────────────────────────┘
                        │ JSON artifacts
┌───────────────────────▼─────────────────────────────┐
│  Layer 2: Model Artifacts (backtesting/<strategy>/) │
│  - JSON models (XGBoost, Q-table, rules, RF trees)  │
│  - policy-metadata.json → state cols, indicator      │
│    params, gate rules, model type                    │
└────────────┬──────────────────────┬─────────────────┘
             │ LEAN backtest        │ live runner
┌────────────▼──────────┐  ┌───────▼─────────────────┐
│  Layer 3: Backtesting  │  │  Layer 3b: Live Trading  │
│  (LEAN / Docker)       │  │  (python -m live.runner)  │
│  - Loads JSON models   │  │  - PaperBroker / IBKR    │
│  - Daily inference     │  │  - Loads same artifacts   │
│  - Event-driven sim    │  │  - Scheduled or --once    │
└────────────┬──────────┘  └───────┬─────────────────┘
             │                     │
┌────────────▼─────────────────────▼─────────────────┐
│  Layer 4: Analysis (Notebooks/backtest_analysis)    │
│  - Load LEAN results or live logs                   │
│  - 4-panel dashboard + normalized performance chart │
│  - Performance statistics                           │
└─────────────────────────────────────────────────────┘
```

---

## Shared Library: `common/`

All reusable logic lives in `common/` and is imported by notebooks as `import common`. It is **not** available inside the LEAN Docker container — the backtesting layer remains self-contained.

| Module | Contents |
|--------|----------|
| `common/config.py` | `load_strategy_config`, `split_date_parts`, `build_model_path` |
| `common/data/` | `fetch_ohlcv` (Parquet cache + yfinance/IBKR sources), `DataSource` ABC, `LocalStore` |
| `common/indicators/` | `compute_indicators`, `add_indicators`, `list_indicators`, `@register` decorator; 8 indicators |
| `common/models/` | `BaseModel` ABC, 5 implementations: `ManualModel`, `ClassificationModel`, `QLearningModel`, `FQIModel`, `OptimizationModel`, `create_model` factory |
| `common/models/learners/` | `RTLearner`, `BagLearner`, `TabularQLearner` |
| `common/strategy.py` | `StrategyConfig` dataclass, `Strategy` class (composes data + indicators + model) |
| `common/portfolio.py` | `compute_portvals`, `portfolio_stats` — local portfolio simulator |
| `common/plotting.py` | `backtest_dashboard`, `plot_normalized_performance`, parse/plot utilities |

---

## Layer 1: Research

**Location**: `Notebooks/`
**Environment**: `renquant` conda env

The notebook is where training happens. The typical workflow:

1. **Data ingestion** — `common.fetch_ohlcv` fetches daily OHLCV (cached locally as Parquet)
2. **Indicator computation** — `common.compute_indicators` applies any combination of registered indicators
3. **Model training** — depends on model type:
   - **Manual**: define threshold rules, no training needed
   - **Classification**: label forward returns, train BagLearner(RTLearner) ensemble
   - **Q-Learning**: discretize states, train Q-table over epochs
   - **FQI**: build (s, a, r, s') transitions, run Fitted Q-Iteration with XGBoost
   - **Optimization**: Nelder-Mead searches indicator params, trains inner Classification model
4. **Export** — `model.save()` writes JSON artifacts to `backtesting/<strategy>/`

---

## Layer 2: Model Artifacts

**Location**: `backtesting/<strategy>/`

All artifacts are JSON (not pickle) — required for LEAN compatibility and human-readability.

| File | Contents |
|------|----------|
| `*-policy-metadata.json` | Model type, state columns, indicator parameters, gate rules, hyperparams |
| `*-q-hold/buy/sell.json` | XGBoost models (FQI) |
| `*-rf-trees.json` | Random Forest tree structure (Classification) |
| `*-qtable.json` + `*-bin-edges.json` | Q-table + discretization (Q-Learning) |
| `*-manual-rules.json` | Threshold rules (Manual) |

The policy metadata acts as a contract between research and execution — both must use identical indicator parameters.

---

## Layer 3: Backtesting (LEAN)

**Location**: `backtesting/<strategy>/main.py`
**Runtime**: QuantConnect LEAN engine (Docker)

`main.py` implements `QCAlgorithm`:

- **`Initialize()`** — reads `strategy_config.json`, loads policy metadata and model artifacts, sets warmup
- **`OnData()`** — called once per trading day:
  1. Fetches price history
  2. Computes indicators inline (duplicated from common/ — Docker constraint)
  3. Determines gate signals
  4. Scores eligible actions via model; executes argmax

**Important**: `main.py` is self-contained. It does **not** import `common/` because LEAN Docker cannot access it.

---

## Layer 3b: Live Trading

**Location**: `live/`
**Entry point**: `python -m live.runner --strategy <name> --broker paper|ibkr --once`

The live runner loads the same model artifacts as LEAN but executes via broker API:

- `PaperBroker` — simulates fills locally for testing
- `IBKRBroker` — connects to Interactive Brokers TWS/Gateway (stub, pending IBKR setup)
- Logs every signal and order to `live/logs/<strategy>/<date>.json`

---

## State Space

Strategies define their own state features via `state_columns` in policy metadata. The default FQI state vector:

| Feature | Description | Parameters |
|---------|-------------|------------|
| `macd_line` | EMA(12) − EMA(26) | fast=12, slow=26 |
| `macd_signal` | EMA(9) of macd_line | signal=9 |
| `macd_hist` | macd_line − macd_signal | — |
| `rsi` | Relative Strength Index | period=14 |
| `cci` | Commodity Channel Index | period=20 |
| `position_flag` | 1 if long, 0 if flat | — |

Other model types (Classification, Q-Learning) default to `rsi`, `macd_hist`, `cci` but can be configured with any registered indicators.

---

## Data Flow Summary

```
yfinance / IBKR
       ↓
  Parquet cache (data/ohlcv/)
       ↓
  OHLCV DataFrame
       ↓
  Indicator registry (compute_indicators)
       ↓
  Model training (Manual / RF / QL / FQI / Optimization)
       ↓
  JSON artifacts → backtesting/<strategy>/
       ↓
  ┌─────────────────────────────────┐
  │ LEAN backtest (Docker)          │
  │ Live trader (IBKR / paper)      │
  └─────────────────────────────────┘
       ↓
  Analysis dashboard + normalized performance chart
```
