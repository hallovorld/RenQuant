# Environment — Libraries + Tooling Versions

Single source of truth for "what's installed and at what version" so the project is reproducible end-to-end.

## Reproducing the env

```bash
# 1) Project .venv (Apple Silicon arm64; NOT conda per feedback_python_env)
python3.10 -m venv .venv
source .venv/bin/activate

# 2) Full locked Python deps
pip install -r requirements.lock.txt
```

`requirements.lock.txt` (regenerated via `pip freeze > requirements.lock.txt`) is the authoritative record — anything else in this doc is a curated summary.

---

## Python: 3.10.20

## macOS host: Apple Silicon M4 Pro (14 cores: 10P + 4E, 20 GPU cores, 48 GB unified RAM)

## Critical libraries (live in `requirements.lock.txt`, summarised here)

| Domain | Library | Version | Why it matters |
|---|---|---|---|
| Core data | `pandas` | **2.3.3** | OHLCV + panel frames |
| Core data | `numpy` | **2.2.6** | arrays / numeric |
| Core data | `scipy` | **1.15.3** | stats / optimization |
| Core data | `pyarrow` | **23.0.1** | parquet cache I/O |
| ML — ranker | `xgboost` | **3.2.0** | panel-LTR primary backend (`rank:pairwise`) on 172 features |
| ML — ranker | `lightgbm` | **4.6.0** | alternate backend; re-opened 2026-05-20 with GICS unblock |
| ML — uncertainty | `ngboost` | **0.5.10** | μ/σ head — trained + promoted 2026-05-17 (val_IC +0.0352); σ-wire dormant per A/B |
| ML — classical | `scikit-learn` | **1.7.2** | Platt-scaling calibration (switched from isotonic 2026-05-18), splits, preprocessing |
| ML — transformer backbone | `transformers` (HF) | **5.8.1** | HF PatchTST shadow path; `Trainer`, `TrainingArguments`, `TrainerCallback`, `PatchTSTModel` |
| ML — transformer training | `accelerate` (HF) | **1.13.0** | Required by `transformers.Trainer` (`accelerate>=1.1.0` minimum) |
| ML — transformer tensor | `torch` | **2.11.0** | MPS backend on M4 Pro 20 GPU cores |
| ML — serialization | `safetensors` | installed | Optional safe checkpoint format (used by `export_transformer_to_safetensors.py`) |
| Plotting | `matplotlib` | **3.10.8** | dashboards, per-symbol charts |
| Plotting | `seaborn` | **0.13.2** | correlation heatmaps, regime plots |
| Data sources | `yfinance` | **1.2.0** | OHLCV + earnings dates + short pct float |
| Data sources | `openbb` | installed | fundamentals (earnings yield, ROE, gross profitability, B/P) |
| Data sources | `alpaca-py` | **0.43.2** | intraday bars + live broker (LIVE for `--broker alpaca`, PAPER for cron) |
| Backtester | `lean` (QuantConnect CLI) | **1.0.225** | Docker-backed backtest engine |
| Notebooks | `jupyterlab` | **4.5.6** | research notebook kernel |
| Test | `pytest` | **9.0.3** | ~11.7k passing (12 pre-existing failures, 7897 skipped) |

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
| launchd (macOS) | — | Scheduled runs (11 active plists in `~/Library/LaunchAgents/com.renquant.*.plist`; see `doc/ops/schedule.md`) |
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
source .venv/bin/activate
pip freeze > requirements.lock.txt
git add requirements.lock.txt doc/ops/environment.md
git commit -m "env: pin <packages>"
```

Commit the lockfile same-day as any dep change — it's the authoritative record, not the `pip install` command you typed.

## Docker configuration for LEAN

LEAN is Docker-backed; Docker Desktop preferences must allocate ≥ 16 GB (project uses 24 GB for safety). Set via Docker Desktop → Resources → Memory. This is out-of-band from the repo.

## Minimum-viable dev loop (no broker / no LEAN)

You can run the panel-training pipeline and unit tests without Docker/Alpaca:

```bash
source .venv/bin/activate
python -m pytest tests/                              # ~11.7k tests
python scripts/train_104.py --skip-baseline --force  # daily STAGES only (per 2026-05-17 walk-forward gate enforcement)
```

This exercises `pandas/numpy/scipy/xgboost/ngboost/sklearn/transformers/accelerate` and the parquet cache — enough to validate a code change before spinning up the full stack.
