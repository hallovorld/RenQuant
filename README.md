# RenQuant 🚀

RenQuant is a personal quantitative trading and research workstation built for high-performance local compute. This project bridges academic machine learning concepts (such as those from Georgia Tech's ML4T) with real-world financial data, utilizing a modular, high-fault-tolerance, "glass-box" pipeline.

## 🎯 Design Philosophy

Unlike highly encapsulated end-to-end Reinforcement Learning frameworks (like FinRL), RenQuant adopts a **"Glass-box" engineering architecture**. 
By strictly decoupling data ingestion, signal generation (Alpha), and backtest execution, we ensure that every trading decision is statistically interpretable and real-world risk is tightly managed.

## 🏗️ Tech Stack Architecture

The RenQuant pipeline consists of three core modules:

1. **Research & Data Layer**
   - **Core Tool**: [OpenBB](https://openbb.co/)
   - **Purpose**: Scrape and clean financial time-series data, fundamental earnings data, macroeconomic indicators, and alternative data (e.g., social media sentiment).
2. **Signal & Modeling Layer**
   - **Core Tools**: `Scikit-learn`, `XGBoost`, `Pandas`
   - **Purpose**: Run traditional supervised learning models and time-series analysis. Leverage Gradient Boosting Decision Trees (GBDT) to mine statistically significant Alpha signals and predict the probability of future asset movements.
3. **Backtesting & Execution Layer**
   - **Core Tool**: [QuantConnect LEAN Engine](https://github.com/QuantConnect/Lean) (Local Docker Deployment)
   - **Purpose**: An industrial-grade, event-driven backtesting engine. It strictly handles splits, dividends, slippage, and transaction fees. It can also bridge directly to major brokerages (e.g., Alpaca, IBKR) via APIs for paper or live trading.

## 💻 Hardware Environment

This project is highly optimized for the **Apple Silicon (M-Series)** architecture:
- **Dev Machine**: MacBook Pro 14-inch (M4 Pro)
- **Memory**: 48GB Unified Memory
- **Performance Edge**: The unified memory architecture allows us to load massive historical tick datasets directly into RAM for rapid feature engineering. XGBoost natively utilizes Apple Silicon's multi-threading, while ample memory allocation for Docker prevents Out-Of-Memory (OOM) errors during multi-factor cross-sectional backtesting in LEAN.

## 📂 Directory Structure

\`\`\`text
RenQuant/
├── common/                # Shared Python library (import as `import common as rq`)
│   ├── config.py          # Config loading and path utilities
│   ├── data.py            # OpenBB/yfinance OHLCV fetching
│   ├── indicators.py      # MACD, RSI, CCI computation
│   ├── features.py        # Gate signals, RL transition builder
│   ├── training.py        # Fitted Q-Iteration, action scoring
│   └── plotting.py        # Backtest dashboard and plot utilities
├── Notebooks/
│   ├── test_001_nvda.ipynb        # Strategy research: data → FQI → export models
│   └── backtest_analysis.ipynb   # Backtest visualisation: signals, equity, stats
├── backtesting/
│   └── test_001_nvda/     # NVDA strategy (LEAN main.py, config, JSON models)
├── doc/                   # Architecture, usage, setup, and tech-stack docs
└── README.md
\`\`\`

## 🚀 Quick Start

Please refer to [`doc/setup.md`](./doc/setup.md) to configure your local environment.

**1. Research — train and export models:**
```bash
conda activate renquant
jupyter lab  # open Notebooks/test_001_nvda.ipynb and run all cells
```

**2. Validation — run rigorous LEAN backtest:**
```bash
cd backtesting/test_001_nvda
lean backtest .
```

**3. Analysis — visualize backtest results:**
```bash
# open Notebooks/backtest_analysis.ipynb and run all cells
```