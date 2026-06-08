# Tech Stack

## Overview

| Layer | Tool | Role |
|-------|------|------|
| Data | yfinance + OpenBB + Alpaca IEX (intraday) | OHLCV + intraday bars + macro factors + news sentiment |
| Data cache | Parquet (pyarrow) | Local storage for fetched data (`data/ohlcv/`, `data/intraday/`, `data/macro/`, `data/news_sentiment/`) |
| Research | JupyterLab + `scripts/train_104.py` | Interactive 101/102/103; CLI-driven for 104 (`FullTrainingPipeline`) |
| Panel ranker | HF PatchTST (PRIMARY since 2026-06-05), XGBoost (previous primary, readonly shadow / rollback), legacy custom PatchTST (deprecated) | Cross-sectional learning-to-rank backends (104); registry at `kernel/panel_pipeline/model_registry.py` |
| Sequence model training | Hugging Face `transformers` 5.8.1 + `accelerate` 1.13.0 | HF Trainer-based PatchTST with multi-task head (rank + Student-t dist) + optional FiLM regime conditioning (Perez 2017) |
| Probabilistic head | NGBoost (Normal distn) + Student-t via `torch.distributions` in HF PatchTST | μ/σ residual estimator (NGB head promoted 2026-05-17; σ-wire dormant); Student-t NLL multi-task head in HF PatchTST replaces NGB σ wire long-term |
| Calibration | scikit-learn Platt scaling (switched from isotonic 2026-05-18) | Score-DB global calibrator with non-collapse gate (H2a) + IC-regression gate (H2b) |
| Per-symbol learners | RTLearner, BagLearner, TabularQLearner, XGBoost (`XGBClassifier`) | 101/102/103 tournament backends + 104 baseline |
| Portfolio QP | cvxpy + CLARABEL solver | Rotation under correlation + sector + concentration constraints + HIFO lot accounting + min_share_floor (104) |
| State store | SQLite (runs.db) + parquet | pipeline_runs / candidate_scores / trades / training_runs / challenger_decisions / etc. |
| Native acceleration | Rust (`rust/transformer_scorer/`) | Pre-2026-05-19 transformer-inference path (legacy); HF PatchTST now uses the HF runtime path |
| Calendar | pandas-market-calendars | NYSE holiday + early-close awareness for cron + trading-day math |
| Final backtest | QuantConnect LEAN (Docker) | Industrial-grade event-driven backtesting |
| Live trading | Alpaca (alpaca-py) + IBKR (stub) | Real-time order execution (PAPER for cron; LIVE for explicit `--broker alpaca`) |
| Notifications | macOS osascript + ntfy.sh | Banner + iPhone push for trade summaries / acceptance failures |
| Runtime | Project-local `.venv` (Python 3.10) on Apple Silicon M4 Pro 14c (10P+4E) / 48 GB / 20 GPU cores | Per `feedback_python_env` — NOT conda |
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

## ML: XGBoost + LightGBM + NGBoost + Transformer

**Per-symbol learners** (101/102/103 tournament, 104 baseline):
- **XGBoostModel**: two `XGBClassifier` instances (buy-vs-rest, sell-vs-rest) with L1/L2 regularisation, residual boosting, native JSON serialisation
- **FQIModel** Q-value estimators (101/102 only): `XGBRegressor` per action in Fitted Q-Iteration
- **RTLearner / BagLearner / TabularQLearner**: ported from ML4T, cleaned up — Random tree, bagging wrapper (Random Forest when wrapping RTLearner), Q-table with epsilon-greedy + optional Dyna experience replay

**Cross-sectional panel-LTR backends** (104 active):
- **HF PatchTST** (PRIMARY since 2026-06-05): `transformers.PatchTSTModel` backbone + dual head (rank_head Linear→1, dist_head Linear→3 for Student-t df/loc/scale). Margin Ranking loss + Student-t NLL multi-task. HF Trainer with `load_best_model_at_end=True` + PerRegimeICCallback (PRIME DIRECTIVE) + cosine LR + warmup. Optional FiLM regime conditioning via `--film-regime-cond`. Training: `scripts/patchtst_hf.py`. Plan: `doc/research/2026-05-19-patchtst-improvement-plan.md`.
- **XGBoost** (`rank:pairwise`, previous primary): readonly shadow / rollback. 172 features (alpha158 + 5 fund + 3 PEAD + 3 SUE + 3 sentiment). Artifact at `artifacts/prod/panel-ltr.alpha158_fund.json`.
- **Legacy custom PatchTST**: pre-2026-05-19 refactor (hand-rolled train loop); retained for old shadow checkpoints. Replaced by HF Trainer-based.
- **LightGBM** (`lambdarank`): RE-OPENED 2026-05-20 — GICS sector data unblock (`data/ticker_sectors.json` 304 tickers). E48 retest pending.
- **RegimeRouterScorer** (`regime_router_scorer.py`, commit `c52ad8d`): FROZEN as dormant baseline per arXiv 2603.13252 — hard routing + market-state gate AUROC < 0.5.

**Probabilistic head** (104):
- **NGBoost** (Normal distribution): trained 2026-05-17 + promoted to prod (val_IC +0.0352, σ-calib +0.274 vs XGB baseline +0.0294); quality gate refuses save below baseline. σ-wire to Kelly is dormant per 3-condition A/B all NULL/negative. μ available, σ not consumed.
- **Student-t multi-task head in HF PatchTST**: end-to-end calibrated σ via `torch.distributions.StudentT` NLL alongside Margin Ranking loss. Shared representation = no train/serve skew that broke NGB σ wire repeatedly.

**Calibration** (104):
- **scikit-learn isotonic regression**: monotonic mapping from raw panel scores → `rank_score ∈ [0, 1]`. Defended by acceptance gates G2 (≥5 unique probabilities) and G3 (pool_ic > 0).

**Portfolio optimization** (104):
- **cvxpy + OSQP**: QP solver for rotation under correlation, sector-concentration, and per-position Kelly constraints.

Why this mix:
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
| Zipline | Unmaintained; poor Apple Silicon support |
| pickle for model serialization | Python/library version-sensitive; incompatible with LEAN Docker environment |
| Two conda environments | Eliminated — OpenBB and ML stack coexist fine in one environment |

> **Note** — earlier versions of this doc said "Deep learning not used; prone to overfitting on small datasets". That was true for renquant_101/102 (per-symbol scope). Stage C-3 of renquant_104 (2026-04-26 round-7) ships a daily + hourly transformer backend with native Rust scorer for inference parity. See [`../components/transformer.md`](../components/transformer.md). The general claim still holds: don't use deep nets where the panel is small (< ~50K rows); for the cross-sectional 99-ticker × 753-day panel (~75K rows) it's borderline-acceptable behind acceptance gates.
