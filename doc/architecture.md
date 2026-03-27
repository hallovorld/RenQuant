# Architecture

## Design Principle: Glass-Box Pipeline

RenQuant is built around **strict layer decoupling**. Each layer has one job, communicates via well-defined interfaces (JSON files), and can be developed or replaced independently. Every decision in the pipeline is inspectable — no end-to-end black boxes.

---

## Four-Layer Pipeline

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Research (Notebooks/)                     │
│  - Fetch OHLCV (yfinance/IBKR, cached as Parquet)  │
│  - Compute indicators via registry                  │
│  - Relativize indicators vs SPY benchmark           │
│  - Train model (Manual/RF/QL/FQI/Optimization)      │
│  - Export: JSON model artifacts + policy-metadata    │
└───────────────────────┬─────────────────────────────┘
                        │ JSON artifacts
┌───────────────────────▼─────────────────────────────┐
│  Layer 2: Model Artifacts (backtesting/<strategy>/) │
│  - JSON models (XGBoost, Q-table, rules, RF trees)  │
│  - policy-metadata.json → state cols, indicator      │
│    params, gate rules, model type                    │
└────────────┬──────────────────────┬─────────────────┘
             │ LEAN backtest        │ live runner
┌────────────▼──────────┐  ┌───────▼─────────────────┐
│  Layer 3: Backtesting  │  │  Layer 3b: Live Trading  │
│  (LEAN / Docker)       │  │  (python -m live.runner)  │
│  - Loads JSON models   │  │  - PaperBroker / IBKR    │
│  - Daily inference     │  │  - Loads same artifacts   │
│  - Event-driven sim    │  │  - Scheduled or --once    │
└────────────┬──────────┘  └───────┬─────────────────┘
             │                     │
┌────────────▼─────────────────────▼─────────────────┐
│  Layer 4: Analysis (scripts/analyze_backtest.py)   │
│  - Load LEAN results or live logs                  │
│  - Dashboard + normalized performance chart        │
│  - Performance statistics + decision telemetry     │
└─────────────────────────────────────────────────────┘
```

---

## Shared Library: `common/`

All reusable logic lives in `common/` and is imported by notebooks as `import common`. It is **not** available inside the LEAN Docker container — the backtesting layer remains self-contained.

| Module | Contents |
|--------|----------|
| `common/config.py` | `load_strategy_config`, `split_date_parts`, `build_model_path` |
| `common/data/` | `fetch_ohlcv` (Parquet cache + yfinance/IBKR sources), `DataSource` ABC, `LocalStore` |
| `common/indicators/` | `compute_indicators`, `add_indicators`, `list_indicators`, `@register` decorator; 12 indicators across 4 categories |
| `common/models/` | `BaseModel` ABC, 5 implementations: `ManualModel`, `ClassificationModel`, `QLearningModel`, `FQIModel`, `OptimizationModel`, `create_model` factory |
| `common/models/learners/` | `RTLearner`, `BagLearner`, `TabularQLearner` |
| `common/strategy.py` | `StrategyConfig` dataclass, `Strategy` class (composes data + indicators + model) |
| `common/portfolio.py` | `compute_portvals`, `portfolio_stats` — local portfolio simulator |
| `common/plotting.py` | `backtest_dashboard`, `plot_normalized_performance`, parse/plot utilities |

---

## Layer 1: Research

**Location**: `Notebooks/`
**Environment**: `renquant` conda env

The notebook is where training happens. The typical workflow:

1. **Data ingestion** — `common.fetch_ohlcv` fetches daily OHLCV for both the target stock and SPY benchmark (cached locally as Parquet)
2. **Indicator computation** — `common.compute_indicators` applied to both stock and SPY
3. **Relative feature construction** — indicators are relativized against SPY:
   - **Ratio** (`stock / SPY`) for always-positive indicators: RSI, ADX
   - **Difference** (`stock - SPY`) for zero-crossing indicators: MACD hist, CCI, BBP, Williams %R, OBV slope
   - Additional trend-following features: `trend` (price/50EMA), `trend_long` (price/200EMA), `rel_mom_20d`, `rel_mom_60d`
4. **Model training** — depends on model type (see below)
5. **Comparison** — all models simulated with constraints (wash sale 30d, min hold 20d, max hold 150d), compared with stock and SPY buy-and-hold benchmarks
6. **Export** — best model by Sharpe ratio auto-exported to `backtesting/<strategy>/`

---

## Layer 2: Model Artifacts

**Location**: `backtesting/<strategy>/`

All artifacts are JSON (not pickle) — required for LEAN compatibility and human-readability.

| File | Contents |
|------|----------|
| `*-policy-metadata.json` | Model type, state columns, indicator parameters, gate rules, hyperparams |
| `*-q-hold/buy/sell.json` | XGBoost models (FQI) |
| `*-rf-trees.json` | Random Forest tree structure (Classification) |
| `*-qtable.json` + `*-bin-edges.json` | Q-table + discretization (Q-Learning) |
| `*-manual-rules.json` | Threshold rules (Manual) |

The policy metadata acts as a contract between research and execution — both must use identical indicator parameters.

---

## Layer 3: Backtesting (LEAN)

**Location**: `backtesting/<strategy>/main.py`
**Runtime**: QuantConnect LEAN engine (Docker)

`main.py` implements `QCAlgorithm`. renquant_101 supports Manual, Classification, and Q-Learning; renquant_102 supports Classification and Manual:

- **`Initialize()`** — reads `strategy_config.json`, loads policy metadata and model artifacts, sets warmup
- **`OnData()`** — called once per trading day:
  1. Fetches price history and computes indicators inline (duplicated from common/ — Docker constraint)
  2. Scores actions via the exported policy
  3. Applies trading constraints (wash sale, min/max hold)
  4. Applies position sizing (max position %, cash reserve) before submitting orders
  5. Logs decisions, plots telemetry, and submits orders when allowed

> **Note**: renquant_101's `main.py` feeds raw indicators to the model (not relative to SPY). This is a known gap for that strategy. renquant_102's `main.py` computes relative features by fetching history for both the stock and SPY and computing ratio/diff transforms inline.

**Important**: `main.py` is self-contained. It does **not** import `common/` because LEAN Docker cannot access it.

---

## Layer 3b: Live Trading

**Location**: `live/`
**Entry point**: `python -m live.runner --strategy <name> --broker paper|ibkr --once`

The live runner loads the same model artifacts as LEAN but executes via broker API:

- `PaperBroker` — simulates fills locally for testing
- `IBKRBroker` — connects to Interactive Brokers TWS/Gateway (stub, pending IBKR setup)
- Logs every signal and order to `live/logs/<strategy>/<date>.json`

---

## Relative Indicator Framework

All features are computed relative to SPY to answer "is the stock outperforming the market?" rather than "is the stock going up?"

| Transform | When to use | Features |
|-----------|-------------|----------|
| **Ratio** (`stock / SPY`) | Always-positive indicators | `rsi`, `adx` |
| **Difference** (`stock - SPY`) | Zero-crossing indicators | `macd_hist`, `cci`, `bbp`, `williams_r`, `obv_slope` |
| **Absolute trend** | Price vs moving average | `trend` (price/50EMA), `trend_long` (price/200EMA) |
| **Relative momentum** | Relative price changes | `rel_mom_20d`, `rel_mom_60d` |

This means models learn relative patterns (outperformance/underperformance vs market), not absolute patterns that break in bull/bear regime changes.

---

## Trading Constraints

All models are subject to execution constraints during both notebook simulation and LEAN backtesting:

| Constraint | Value | Purpose |
|------------|-------|---------|
| Wash sale avoidance | 30 calendar days | Cannot buy within 30 days of selling (IRS wash sale rule) |
| Minimum hold | 20 calendar days | Prevents excessive short-term trading |
| Maximum hold | 150 calendar days | Forces position review, prevents "buy and forget" |

## Position Sizing

Position sizing is configured in `strategy_config.json` under the `position_sizing` block and enforced during both notebook simulation and LEAN backtesting:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_position_pct` | 0.33 (33%) | No single stock can exceed 1/3 of total portfolio value |
| `cash_reserve_pct` | 0.10 (10%) | Always maintain 10% cash reserve |

**Rules:**
1. **Cash-only buys** — only use available cash for new positions; never sell existing holdings to fund a new buy
2. **Max position cap** — `target_pct = min(max_position_pct, (available_cash - cash_reserve) / portfolio_value)`
3. **Whole shares only** — notebook simulation buys whole shares; LEAN uses `SetHoldings` which handles this internally

These rules are used by both single-stock (renquant_101) and multi-stock (renquant_102) strategies. In multi-stock mode, each position is independently capped and the cash reserve is maintained across all positions.

---

## State Space

The NVDA strategy uses 7 shared relative indicator features for ML models (Classification, Q-Learning):

| Feature | Transform | Description |
|---------|-----------|-------------|
| `rsi` | ratio | Relative Strength Index (stock/SPY) |
| `macd_hist` | diff | MACD histogram (stock − SPY) |
| `cci` | diff | Commodity Channel Index (stock − SPY) |
| `bbp` | diff | Bollinger Band Percentage (stock − SPY) |
| `adx` | ratio | Average Directional Index (stock/SPY) |
| `williams_r` | diff | Williams %R (stock − SPY) |
| `obv_slope` | diff | OBV rate of change (stock − SPY) |

The Manual model uses trend-following features instead (see Strategy Details below).
Q-Learning uses a subset of 3 trend features to keep state space small.

All 12 registered indicators can be combined freely.

---

## Strategy Details

### renquant_102 — Multi-Stock Volume Z-Score Scanner

A 3-stage pipeline strategy: **DETECT** → **CONFIRM** → **EXECUTE**.

**Stage 1: DETECT** — compute rolling volume z-score `(today_vol - mean_N) / std_N` for each watchlist stock. A z-score above threshold (default 2.0σ, lookback 15 days) signals unusual institutional activity.

**Stage 2: CONFIRM** — on spike days, 4 approaches analyze 2 years of history to confirm direction:
1. Dual Momentum — trend-following rules (trend, relative momentum, MACD, OBV)
2. Classification — per-stock Random Forest on relative features
3. Mean Reversion — contrarian buy-the-dip on oversold conditions
4. Breakout — ride momentum on 20-day high breakouts

**Stage 3: EXECUTE** — best approach by Sharpe trades, max 3 concurrent positions.

**Config** uses `watchlist` (array of symbols) instead of `stock_symbol`:
```json
{
  "watchlist": ["NVDA", "TSLA", "AAPL", ...],
  "volume_zscore_lookback": 15,
  "volume_zscore_threshold": 2.0,
  "training_years": 2,
  "max_concurrent_positions": 3
}
```

**LEAN `main.py` flow** (`ZScoreScannerStrategy`):
1. `Initialize()` — `AddEquity` for each watchlist stock + SPY, load per-stock model artifacts
2. `OnData()` — process sells first (check models + constraints for held positions), then compute volume z-scores for non-held stocks, rank candidates by z-score, run per-stock models, execute buys (up to `max_concurrent_positions`)

**Artifact naming**: `{model_name}-{SYMBOL}-rf-trees.json` and `{model_name}-{SYMBOL}-policy-metadata.json` per stock. The existing `ClassificationModel.save(dir, "renquant-102-NVDA")` handles this with no changes to `common/models/`.

**Live runner**: auto-detects multi-stock strategies by checking for `"watchlist"` in config. Uses `run_once_multi()` which computes volume z-scores, checks sell signals, and executes buy orders across the watchlist.

### renquant_101 — Single-Stock Classification

Trains a single model on relative indicators (stock vs SPY) for one symbol. The notebook trains 3 model types (Manual/Dual Momentum, Classification/RF, Q-Learning), compares them with stock and SPY buy-and-hold benchmarks, and exports the best by Sharpe ratio. Config uses `stock_symbol` (single string).

### Manual — Dual Momentum + Trend Following

Based on Gary Antonacci's Dual Momentum principles. Uses **trend-following features** where threshold checks have clear meaning, unlike oscillator thresholds that whipsaw:

| Rule | Feature | Buy condition | Sell condition | Rationale |
|------|---------|---------------|----------------|-----------|
| Absolute trend (50d) | `trend` | > 1.0 | < 0.97 | Price above 50-day EMA = uptrend |
| Absolute trend (200d) | `trend_long` | > 1.0 | < 0.97 | Price above 200-day EMA = structural uptrend |
| Relative momentum (20d) | `rel_mom_20d` | > 0.0 | < -0.03 | Stock outperforming SPY over 20 days |
| Relative momentum (60d) | `rel_mom_60d` | > 0.0 | < -0.05 | Stock outperforming SPY over 60 days |
| MACD confirmation | `macd_hist` diff | > 0 | < 0 | Stock momentum leads SPY |
| Volume confirmation | `obv_slope` diff | > 0 | < 0 | Accumulation exceeds market |

**Entry**: 4 of 6 rules agree bullish (trend + relative momentum + confirmations).
**Exit**: 3 of 6 rules agree bearish (trend break + momentum loss).

### Classification — Bagged Random Forest

Uses all 7 relative indicator features. Labels each day by 10-day forward return (±4% threshold). BagLearner(RTLearner) ensemble with 15 bags and leaf_size=25. Buy/sell thresholds at ±0.1 (lowered from default ±0.5 for trending stocks). The RF learns nonlinear relationships between relative features automatically — it effectively discovers crossover patterns and conditional logic from the data.

### Q-Learning — Tabular RL with Relative Reward

Uses 3 trend-following features (`trend`, `rel_mom_20d`, `macd_hist`) with 5 bins = 375 states. Key design choice: **reward is relative price returns** (stock/SPY ratio changes), not raw stock returns. In a bull market, raw returns are always positive and the Q-learner learns "buy and never sell." Relative returns can go negative, giving the agent a reason to exit when the stock underperforms the market.

---

## Data Flow Summary

```
yfinance / IBKR
       ↓
  Parquet cache (data/ohlcv/)
       ↓
  OHLCV DataFrames (stock + SPY)
       ↓
  Indicator registry (compute_indicators) × 2
       ↓
  Relative features (ratio or diff vs SPY)
       ↓
  Model training (Manual / RF / QL / FQI / Optimization)
       ↓
  JSON artifacts → backtesting/<strategy>/
       ↓
  ┌─────────────────────────────────┐
  │ LEAN backtest (Docker)          │
  │ Live trader (IBKR / paper)      │
  └─────────────────────────────────┘
       ↓
  Analysis dashboard + normalized performance chart
  (stock vs SPY vs model equity)
```
