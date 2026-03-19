# RenQuant Setup Guide ⚙️

This guide is specifically tailored for the **Apple Silicon (M4 Pro)** architecture. To avoid complex C++ compilation errors and package conflicts, we use `Miniforge` for base environment management and strictly isolate the research, modeling, and backtesting environments.

## 0. Prerequisites

1. **Install Homebrew**: The essential package manager for macOS.
2. **Install Docker Desktop**: 
   - Ensure you download the Apple Silicon version.
   - **⚠️ Crucial Performance Tweak**: Open Docker Settings -> `Resources` -> `Memory`, and **allocate at least 16GB** (leveraging the host's 48GB capacity). This prevents the LEAN engine from crashing during heavy backtests.

## 1. Base Environment Management (Miniforge)

Avoid the standard Anaconda distribution. Instead, install the `arm64`-optimized Miniforge:

```bash
# Configure miniconda3
brew install miniconda
source ~/miniconda3/bin/activate
conda config --add channels conda-forge
conda config --set channel_priority strict

# Create and activate the environment
conda create -n renquant python=3.10 -y
conda activate renquant

# Install core data processing and ML libraries
# XGBoost will automatically detect and utilize the Mac's multi-core CPU for acceleration
pip install pandas numpy matplotlib seaborn yfinance scikit-learn xgboost jupyterlab


# Create an isolated environment
conda create -n openbb python=3.10 -y
conda activate openbb

# Install OpenBB core and CLI interface
pip install "openbb[all]"
pip install openbb-cli
openbb-build

# Test the installation (this will launch the interactive terminal)
openbb


# Return to the base environment
conda deactivate

# Install the LEAN CLI tool
pip install lean

# Log into your QuantConnect account (create a free account on their website first to get an API Token)
lean login

# Initialize the LEAN workspace in your current RenQuant directory
lean init