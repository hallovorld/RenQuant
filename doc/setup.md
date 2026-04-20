# Setup Guide

Optimized for Apple Silicon (M-Series). All packages are arm64-native via Miniconda with `conda-forge` configured as the primary channel.

## Prerequisites

### 1. Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Docker Desktop
- Download the **Apple Silicon** version from docker.com
- After install: Docker Settings → Resources → Memory → set to **16GB minimum**
  - This prevents LEAN from crashing during backtests. With 48GB unified memory, 16GB is safe.

### 3. QuantConnect Account
- Create a free account at quantconnect.com
- Go to Account → API Access → note your **User ID** and **API Token**

---

## Environment Setup

RenQuant uses a **single conda environment** for everything — research, data, ML, and live trading.

```bash
# Install Miniconda (Apple Silicon arm64 build)
brew install miniconda
source ~/miniconda3/bin/activate

# Configure channels
conda config --add channels conda-forge
conda config --set channel_priority strict

# Create the unified environment
conda create -n renquant python=3.10 -y
conda activate renquant

# Core dependencies
pip install pandas numpy matplotlib seaborn yfinance scikit-learn xgboost jupyterlab pyarrow

# OpenBB + optimization + backtesting
pip install "openbb[all]" openbb-cli backtesting scipy

# Live trading (Alpaca)
pip install alpaca-py

# Notifications (macOS + iPhone)
brew install terminal-notifier   # macOS banner notifications
# Install ntfy app on iPhone; subscribe to 'renquant' topic

# Install and authenticate LEAN CLI
pip install lean
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
conda activate renquant
jupyter lab
```

That's it — one command to start working.
