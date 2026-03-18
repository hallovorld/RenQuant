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
├── data/                  # Local cache for market data and feature sets
├── research/              # Jupyter Notebooks (EDA, Factor Mining, OpenBB experiments)
├── models/                # Trained machine learning models (.pkl or .json)
├── strategies/            # QuantConnect LEAN strategy scripts (Python/C#)
├── setup.md               # Environment configuration and deployment guide
└── README.md              # Project overview
\`\`\`

## 🚀 Quick Start

Please refer to [`setup.md`](./setup.md) to configure your local environment and install dependencies.