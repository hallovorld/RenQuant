# Setup Guide

Optimized for Apple Silicon M4 Pro (14 cores: 10P + 4E, 20 GPU cores, 48 GB unified RAM). All packages are arm64-native via the project `.venv` (not conda — per `feedback_python_env`).

## Prerequisites

### 1. Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Docker Desktop
- Download the **Apple Silicon** version from docker.com
- After install: Docker Settings → Resources → Memory → set to **16GB minimum**
  - This prevents LEAN from crashing during backtests. With 48 GB unified memory, 16 GB is safe.

### 3. QuantConnect Account
- Create a free account at quantconnect.com
- Go to Account → API Access → note your **User ID** and **API Token**

---

## Environment Setup

RenQuant uses a project-local `.venv` (Python 3.10) for everything — research, data, ML, and live trading. All dependencies pinned in `requirements.lock.txt`.

```bash
# From repo root:
python3.10 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies (xgboost, lightgbm, ngboost, pandas, numpy, scikit-learn,
# yfinance, jupyterlab, pyarrow, scipy, alpaca-py, lean, openbb, transformers >= 5.8.1,
# accelerate >= 1.1.0 for HF Trainer-based PatchTST, etc.)
pip install -r requirements.lock.txt

# Notifications (macOS + iPhone)
brew install terminal-notifier   # macOS banner notifications
# Install ntfy app on iPhone; subscribe to 'renquant' topic

# Authenticate LEAN CLI
lean login          # enter your QuantConnect User ID and API Token
```

Verify LEAN is connected:
```bash
lean whoami
```

---

## One-Time LEAN Workspace Init

This was already done in the repo. Only needed if starting from scratch:
```bash
lean init
```

---

## Daily Activation

```bash
source .venv/bin/activate
jupyter lab
```

That's it — one command to start working.

---

## HF PatchTST model

If working on the HF PatchTST path, the lockfile includes `transformers>=5.8.1` and `accelerate>=1.1.0` (required by HF `Trainer`). Verify:

```bash
.venv/bin/python -c "import transformers, accelerate; print(transformers.__version__, accelerate.__version__)"
```

Multi-task head (rank + Student-t dist), Margin Ranking loss, FiLM regime conditioning all live in `scripts/patchtst_hf.py`. See `doc/research/2026-05-19-patchtst-improvement-plan.md`.
