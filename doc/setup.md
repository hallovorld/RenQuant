# Setup Guide

Optimized for Apple Silicon (M-Series). All packages are arm64-native via Miniforge.

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

RenQuant uses a **single conda environment** for everything — research, data, and ML.

```bash
# Install Miniforge (arm64-optimized conda)
brew install miniconda
source ~/miniconda3/bin/activate

# Configure channels
conda config --add channels conda-forge
conda config --set channel_priority strict

# Create the unified environment
conda create -n renquant python=3.10 -y
conda activate renquant

# Install all dependencies
pip install pandas numpy matplotlib seaborn yfinance scikit-learn xgboost jupyterlab
pip install "openbb[all]" openbb-cli
pip install backtesting

# Build OpenBB extensions (takes a few minutes)
openbb-build

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
