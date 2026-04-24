# Environment — Libraries + Tooling Versions

Single source of truth for "what's installed and at what version" so the project is reproducible end-to-end.

## Reproducing the env

```bash
# 1) Conda env (Apple Silicon arm64)
conda create -n renquant python=3.10
conda activate renquant

# 2) Full locked Python deps
pip install -r requirements.lock.txt
```

`requirements.lock.txt` (310 packages, regenerated via `pip freeze > requirements.lock.txt`) is the authoritative record — anything else in this doc is a curated summary.

---

## Python: 3.10.20

## macOS host: 26.4.1 (Darwin 25.4.0) on Apple Silicon M4 Pro (14 cores, 48 GB RAM)

## Critical libraries (live in `requirements.lock.txt`, summarised here)

| Domain | Library | Version | Why it matters |
|---|---|---|---|
| Core data | `pandas` | **2.3.3** | OHLCV + panel frames |
| Core data | `numpy` | **2.2.6** | arrays / numeric |
| Core data | `scipy` | **1.15.3** | stats / optimization |
| Core data | `pyarrow` | **23.0.1** | parquet cache I/O |
| ML — ranker | `xgboost` | **3.2.0** | panel-LTR primary backend (`rank:pairwise`) |
| ML — ranker | `lightgbm` | **4.6.0** | alternate backend (shelved per Plan B) |
| ML — uncertainty | `ngboost` | **0.5.10** | μ,σ head for panel scores |
| ML — classical | `scikit-learn` | **1.7.2** | isotonic calibration, splits, preprocessing |
| ML — transformer | `torch` | **2.11.0** | MPS-backed `PanelTransformerModel` (Plan H) |
| ML — trees | (internal) | — | `common/models/learners/RTLearner` + `BagLearner` — no external dep |
| Plotting | `matplotlib` | **3.10.8** | dashboards, per-symbol charts |
| Plotting | `seaborn` | **0.13.2** | correlation heatmaps, regime plots |
| Data sources | `yfinance` | **1.2.0** | OHLCV + earnings dates + short pct float |
| Data sources | `openbb` | installed (no `__version__` attr) | fundamentals (earnings yield, ROE, gross profitability, B/P) |
| Data sources | `alpaca-py` | **0.43.2** | intraday bars (Plan G hourly cache) + live broker |
| Backtester | `lean` (QuantConnect CLI) | **1.0.225** | Docker-backed backtest engine |
| Notebooks | `jupyterlab` | **4.5.6** | research notebook kernel |
| Test | `pytest` | **9.0.3** | 1037+ tests |

## Secondary but present

| Library | Version | Use |
|---|---|---|
| `autograd` | 1.8.0 | NGBoost internals |
| `autograd-gamma` | 0.5.0 | NGBoost internals |
| `cryptography` | 46.0.5 | Authlib / Alpaca auth |
| `arch` | 7.2.0 | ARCH/GARCH (regime layer, via common/indicators) |
| `cloudpickle` | 3.1.2 | ML object pickling |
| `beartype` | 0.22.9 | runtime type checks (OpenBB) |

## Non-Python tooling

| Tool | Version | Use |
|---|---|---|
| Docker | 29.3.1 | Runs the LEAN engine container |
| LEAN CLI | 1.0.225 | `cd backtesting/renquant_104 && lean backtest .` |
| launchd (macOS) | — | Scheduled runs (5 plists in `~/Library/LaunchAgents/com.renquant.*.plist`) |
| `terminal-notifier` | optional | macOS banner for `scripts/backtest_and_analyze.py` + scheduled runs (`brew install terminal-notifier`) |
| `ntfy.sh` | — | iPhone push notifications (topic: `renquant` by default) |
| SQLite CLI | 3.x (system) | `data/runs.db` inspection; decision-log + training-run audit |

## Environment variables

Stored in `.env` at repo root (gitignored). Required for live trading + intraday fetch:

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

Optional, used by specific scripts/torch paths:

```
PYTORCH_ENABLE_MPS_FALLBACK=1   # transformer A/B — CPU fallback for nested_tensor op
```

## Regenerating the lockfile

After installing or upgrading any package:

```bash
conda activate renquant
pip freeze > requirements.lock.txt
git add requirements.lock.txt doc/environment.md
git commit -m "env: pin <packages>"
```

Commit the lockfile same-day as any dep change — it's the authoritative record, not the `pip install` command you typed.

## Docker configuration for LEAN

LEAN is Docker-backed; Docker Desktop preferences must allocate ≥ 16 GB (project uses 24 GB for safety). Set via Docker Desktop → Resources → Memory. This is out-of-band from the repo.

## Minimum-viable dev loop (no broker / no LEAN)

You can run the panel-training pipeline and unit tests without Docker/Alpaca:

```bash
conda activate renquant
python -m pytest tests/                              # 1037 tests
python scripts/train_104.py --skip-baseline --force  # retrain panel only
```

This exercises `pandas/numpy/scipy/xgboost/ngboost/sklearn` and the parquet cache — enough to validate a code change before spinning up the full stack.
