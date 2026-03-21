# Tech Stack

## Overview

| Layer | Tool | Role |
|-------|------|------|
| Data | OpenBB + yfinance | OHLCV and financial data fetching |
| Research | JupyterLab | Interactive development and model training |
| ML | XGBoost + scikit-learn | Q-value estimation, preprocessing |
| Fast backtest | backtesting.py | In-notebook iteration and validation |
| Final backtest | QuantConnect LEAN | Rigorous, production-grade event-driven backtesting |
| Runtime | Miniforge (arm64) | Apple Silicon-native Python environment |
| Containers | Docker Desktop | LEAN engine isolation |

---

## Data: OpenBB + yfinance

**OpenBB** is a unified financial data platform with connectors for equities, crypto, futures, forex, macroeconomic indicators, and alternative data (earnings, sentiment). It is the primary data source and kept for future expansion.

**yfinance** is used as a lightweight fallback for simple OHLCV pulls during rapid prototyping. It requires no API key and is fast for single-ticker daily data.

Why not just yfinance? yfinance covers only price/volume. When the project expands to fundamentals, macro indicators, or alternative data, OpenBB provides a single consistent interface across all data types.

---

## Research: JupyterLab

JupyterLab provides an interactive environment for the entire research pipeline: data exploration, feature engineering, model training, and fast iteration. Notebooks serve as living documents that combine code, outputs, and reasoning in one place — important for a first quant project where understanding each step matters.

---

## ML: XGBoost

XGBoost is used for both the supervised baseline (XGBClassifier) and the Q-value estimators (XGBRegressor) in Fitted Q-Iteration.

Why XGBoost over alternatives:
- **Interpretable**: tree-based models support feature importance, partial dependence plots, and SHAP values — fits the glass-box design goal
- **Tabular data performance**: outperforms neural networks on small tabular datasets (typical for daily OHLCV strategies with limited history)
- **JSON serialization**: XGBoost natively exports/loads models as JSON, which is required for LEAN compatibility (LEAN runs in Docker and cannot access pickle files from the host)
- **Apple Silicon**: XGBoost natively uses multi-core CPU on M-series chips with no configuration

---

## Fast Backtesting: backtesting.py

`backtesting.py` is a lightweight, pure-Python backtesting library that runs directly inside notebooks. It has no external dependencies, no Docker, and produces results in seconds.

Role in the workflow: **research-mode validation**. When tuning indicator parameters or testing a new signal idea, running LEAN for every iteration is too slow (Docker startup + full event-driven simulation). `backtesting.py` provides a fast sanity check before committing to LEAN validation.

It does not replace LEAN — it handles transaction costs and slippage less rigorously and lacks corporate action adjustments.

---

## Final Backtesting: QuantConnect LEAN

LEAN is an industrial-grade, event-driven backtesting engine used in production by QuantConnect. It handles:
- Dividend and split adjustments
- Realistic slippage and transaction cost modeling
- Minute/daily/tick resolution data
- Live trading bridge (Alpaca, Interactive Brokers, etc.)

Why LEAN despite the Docker overhead:
- **Production parity**: the same algorithm code runs in backtesting and live trading — no rewrite required when moving to live deployment
- **Corporate action correctness**: dividend and split handling matters for multi-year backtests; LEAN gets this right automatically
- **Future-proofing**: as the project grows to multi-asset or live trading, LEAN scales without changing the backtesting infrastructure

LEAN is used only for final validation, not for every iteration, which limits the overhead cost.

---

## Runtime: Miniforge (arm64)

Standard Anaconda/conda ships x86_64 packages that run under Rosetta 2 emulation on Apple Silicon. Miniforge ships arm64-native packages, which means:
- XGBoost, NumPy, and Pandas use NEON SIMD instructions natively
- No emulation overhead on M-series chips
- Packages like `pyarrow` and `scipy` compile correctly without Rosetta compatibility issues

All dependencies are installed in a **single `renquant` environment** — previously split into `renquant` and `openbb`, merged to reduce activation overhead and eliminate environment-switching friction.

---

## Serialization: JSON over Pickle

XGBoost models are saved as `.json` files rather than `.pkl`.

- **LEAN compatibility**: LEAN runs Python in a Docker container. The host filesystem is mounted, but pickle files carry Python version and library version constraints that can cause load failures across environments. JSON is version-agnostic.
- **Human-readable**: JSON model files can be inspected to verify tree structure, feature names, and hyperparameters
- **Portability**: JSON files can be loaded by any language with an XGBoost binding (Python, C++, Java, R) — useful if the backtesting layer is ever ported to C#

---

## What Was Considered and Not Used

| Tool | Reason not used |
|------|-----------------|
| FinRL | End-to-end RL framework — black box, hard to debug, contradicts glass-box goal |
| Deep learning (LSTM, Transformer) | Requires much more data than daily OHLCV; prone to overfitting on small datasets |
| Zipline | Unmaintained; poor Apple Silicon support |
| pickle for model serialization | Python/library version-sensitive; incompatible with LEAN Docker environment |
| Two conda environments | Eliminated — OpenBB and ML stack coexist fine in one environment |
