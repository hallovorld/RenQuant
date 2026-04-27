# Tech Stack

## Overview

| Layer | Tool | Role |
|-------|------|------|
| Data | OpenBB + yfinance | OHLCV and financial data fetching |
| Data cache | Parquet (pyarrow) | Local storage for fetched data |
| Research | JupyterLab | Interactive development and model training |
| ML | XGBoost + scikit-learn | Q-value estimation, classification, preprocessing |
| Learners | RTLearner, BagLearner, TabularQLearner | Custom tree ensemble and Q-learning primitives |
| Optimization | SciPy (Nelder-Mead) | Indicator parameter search |
| Portfolio sim | common/portfolio.py | Local portfolio simulation for quick iteration |
| Final backtest | QuantConnect LEAN | Rigorous, production-grade event-driven backtesting |
| Live trading | Alpaca (via alpaca-py) + IBKR (stub) | Real-time order execution |
| Runtime | Miniconda (arm64) | Apple Silicon-native Python environment |
| Containers | Docker Desktop | LEAN engine isolation |

---

## Data: OpenBB + yfinance

**OpenBB** is a unified financial data platform with connectors for equities, crypto, futures, forex, macroeconomic indicators, and alternative data (earnings, sentiment). It is the primary data source and kept for future expansion.

**yfinance** is used as a lightweight fallback for simple OHLCV pulls during rapid prototyping. It requires no API key and is fast for single-ticker daily data.

**Parquet caching**: Fetched data is cached locally at `data/ohlcv/{SYMBOL}/1d.parquet`. Subsequent calls to `fetch_ohlcv` hit the cache first, avoiding redundant network calls.

**IBKR**: Interactive Brokers data source is stubbed out, pending TWS/Gateway setup. When configured, it will serve as both historical data provider and real-time data feed for live trading.

---

## Research: JupyterLab

JupyterLab provides an interactive environment for the entire research pipeline: data exploration, feature engineering, model training, and fast iteration. Notebooks serve as living documents that combine code, outputs, and reasoning in one place.

---

## ML: XGBoost + Custom Learners

**XGBoost** is used in two ways:
- **XGBoostModel** (renquant_103 tournament): two `XGBClassifier` instances (buy-vs-rest, sell-vs-rest) with L1/L2 regularisation, residual boosting, and native JSON serialisation for LEAN compatibility.
- **FQIModel** Q-value estimators (renquant_101/102 only): `XGBRegressor` per action in Fitted Q-Iteration.

**Custom learners** (ported from ML4T, cleaned up):
- **RTLearner**: Random decision tree for classification (leaf nodes store majority class)
- **BagLearner**: Bootstrap aggregation wrapper (Random Forest when wrapping RTLearner)
- **TabularQLearner**: Q-table with epsilon-greedy exploration and optional Dyna experience replay

Why XGBoost over alternatives:
- **Interpretable**: tree-based models support feature importance, partial dependence plots, and SHAP values — fits the glass-box design goal
- **Tabular data performance**: outperforms neural networks on small tabular datasets (typical for daily OHLCV strategies with limited history)
- **JSON serialization**: XGBoost natively exports/loads models as JSON, which is required for LEAN compatibility (LEAN runs in Docker and cannot access pickle files from the host)
- **Apple Silicon**: XGBoost natively uses multi-core CPU on M-series chips with no configuration

---

## Portfolio Simulation: common/portfolio.py

A local `compute_portvals()` function simulates portfolio value from trade schedules with commission and impact modeling. Used by:
- The `OptimizationModel` for in-sample objective evaluation
- Notebooks for quick backtesting without Docker/LEAN overhead

Does not replace LEAN — handles transaction costs less rigorously and lacks corporate action adjustments.

---

## Final Backtesting: QuantConnect LEAN

LEAN is an industrial-grade, event-driven backtesting engine used in production by QuantConnect. It handles:
- Dividend and split adjustments
- Realistic slippage and transaction cost modeling
- Minute/daily/tick resolution data
- Live trading bridge (Alpaca, Interactive Brokers, etc.)

Why LEAN despite the Docker overhead:
- **Production parity**: the same algorithm code runs in backtesting and live trading
- **Corporate action correctness**: dividend and split handling matters for multi-year backtests
- **Future-proofing**: as the project grows to multi-asset or live trading, LEAN scales

LEAN is used only for final validation, not for every iteration.

---

## Live Trading: Alpaca + IBKR

The `live/` package provides a broker abstraction for live order execution:
- **PaperBroker**: simulates fills locally for testing the full runner pipeline
- **AlpacaBroker**: connects to Alpaca Markets API via `alpaca-py` for both paper and live trading. Requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` environment variables. Supports market orders (DAY time-in-force), position queries, and account value lookups.
- **IBKRBroker**: connects to Interactive Brokers TWS/Gateway via `ib_insync` (stub, pending setup)

The live runner loads the same JSON model artifacts that LEAN uses, ensuring consistency between backtested and live behavior.

---

## Runtime: Miniconda (arm64)

RenQuant uses the Apple Silicon build of Miniconda and configures `conda-forge` with strict channel priority. That keeps the Python environment arm64-native, which means:
- XGBoost, NumPy, and Pandas use NEON SIMD instructions natively
- No emulation overhead on M-series chips
- Packages like `pyarrow` and `scipy` compile correctly without Rosetta compatibility issues

All dependencies are installed in a **single `renquant` environment**.

---

## Serialization: JSON over Pickle

All model artifacts are saved as `.json` files rather than `.pkl`.

- **LEAN compatibility**: LEAN runs Python in a Docker container. Pickle files carry Python version and library version constraints that can cause load failures across environments. JSON is version-agnostic.
- **Human-readable**: JSON model files can be inspected to verify tree structure, feature names, and hyperparameters
- **Portability**: JSON files can be loaded by any language with appropriate bindings

---

## What Was Considered and Not Used

| Tool | Reason not used |
|------|-----------------|
| FinRL | End-to-end RL framework — black box, hard to debug, contradicts glass-box goal |
| Deep learning (LSTM, Transformer) | Requires much more data than daily OHLCV; prone to overfitting on small datasets |
| Zipline | Unmaintained; poor Apple Silicon support |
| pickle for model serialization | Python/library version-sensitive; incompatible with LEAN Docker environment |
| Two conda environments | Eliminated — OpenBB and ML stack coexist fine in one environment |
