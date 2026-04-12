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
│  - Loads JSON models   │  │  - Paper / Alpaca / IBKR │
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
| `common/indicators/` | `compute_indicators`, `add_indicators`, `list_indicators`, `@register` decorator; 12 registered indicators across 4 categories; regime detection (non-registered): `compute_hurst`, `rolling_hurst`, `compute_cusum`, `rolling_cusum`, `build_gmm_features`, `RegimeGMM` |
| `common/models/` | `BaseModel` ABC, 6 implementations: `ManualModel`, `ClassificationModel`, `QLearningModel`, `FQIModel`, `OptimizationModel`, `XGBoostModel`, `create_model` factory |
| `common/models/learners/` | `RTLearner`, `BagLearner`, `TabularQLearner` |
| `common/strategy.py` | `StrategyConfig` dataclass, `Strategy` class (composes data + indicators + model) |
| `common/portfolio.py` | `compute_portvals`, `portfolio_stats` — local portfolio simulator |
| `common/tax.py` | `compute_trade_tax`, `compute_after_tax_pnl`, `load_tax_config`, `add_tax_columns`, `tax_rate_for_holding` — after-tax return analysis |
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
5. **Comparison** — all models simulated on the 30% OOS test set with constraints (wash sale 30d, min hold 20d, max hold 500d), compared with stock and SPY buy-and-hold benchmarks. Both OOS Sharpe and in-sample Sharpe are recorded.
6. **Export** — best model by OOS after-tax Sharpe auto-exported to `backtesting/<strategy>/` (Sharpe floor: 0.8 OOS for renquant_102). Orphan model directories for symbols not in the current watchlist are purged on each run.

---

## Layer 2: Model Artifacts

**Location**: `backtesting/<strategy>/`

All artifacts are JSON (not pickle) — required for LEAN compatibility and human-readability.

| File | Contents |
|------|----------|
| `*-policy-metadata.json` | Model type, state columns, indicator parameters, gate rules, hyperparams, `sharpe` (OOS), `sharpe_is` (in-sample, diagnostic), `trained_date`, `best_approach` |
| `*-q-hold/buy/sell.json` | XGBoost models (FQI) |
| `*-rf-trees.json` | Random Forest tree structure (Classification) |
| `*-qtable.json` + `*-bin-edges.json` | Q-table + discretization (Q-Learning) |
| `*-manual-rules.json` | Threshold rules (Manual) |
| `*-xgb-buy.json` + `*-xgb-sell.json` | XGBoost buy/sell classifiers (XGBoost model) |

The policy metadata acts as a contract between research and execution — both must use identical indicator parameters.

For multi-stock strategies (renquant_102, renquant_103), models are organized per-symbol under `models/{SYMBOL}/`:
```
backtesting/renquant_102/models/
  TSLA/TSLA-policy-metadata.json, TSLA-rf-trees.json
  AMZN/AMZN-policy-metadata.json, AMZN-manual-rules.json
  ...
```
Each symbol's model may be a different type (the notebook picks the best approach per symbol).

renquant_103 adds three additional strategy-level artifacts:
```
backtesting/renquant_103/
  spy-gmm-regime.json          # GMM parameters for 3-state regime classifier
  watchlist-correlation.json   # 120-day pairwise return correlations for correlation guard
  earnings-calendar.json       # Upcoming earnings dates per ticker (refreshed weekly)
```

---

## Layer 3: Backtesting (LEAN)

**Location**: `backtesting/<strategy>/main.py`
**Runtime**: QuantConnect LEAN engine (Docker)

`main.py` implements `QCAlgorithm`. The two strategies differ in their approach:

**renquant_101** (single-stock): Loads pre-exported model artifacts (Manual, Classification, or Q-Learning). `Initialize()` reads policy metadata and model files. `OnData()` computes indicators inline and scores actions via the exported policy.

> **Note**: renquant_101's `main.py` feeds raw indicators to the model (not relative to SPY). This is a known gap for that strategy.

**renquant_102** (multi-stock): Loads pre-trained models per symbol from `models/{SYMBOL}/`. `Initialize()` reads `strategy_config.json`, loads all per-symbol model artifacts (checking staleness), and sets up watchlist equities + SPY benchmark. `OnData()`:
  1. Processes sells first for held positions (apply per-stock model + constraints + max hold)
  2. Scans volume z-scores for non-held watchlist stocks (DETECT stage)
  3. For each spike candidate, computes 60-day indicators + relative features for stock and SPY (CONFIRM stage)
  4. Applies that stock's pre-trained model (classification, manual, or qlearning) to get buy/sell/hold signal
  5. Applies position sizing and trading constraints, executes orders (EXECUTE stage)

**renquant_103** (adaptive regime multi-stock): Extends 102's architecture with a 3-layer regime detector. `Initialize()` additionally loads `spy-gmm-regime.json`, `watchlist-correlation.json`, and `earnings-calendar.json`. `OnData()`:
  1. Accumulates SPY daily returns; runs Hurst (Layer 1), CUSUM (Layer 2), GMM (Layer 3) to classify regime and confidence
  2. Sets regime-adaptive parameters (stop-loss, position size, max hold, drawdown halt) — position sizing scales continuously with GMM confidence
  3. Processes sells (same as 102, but with regime-adaptive stop-loss and wider trailing stop in BULL_CALM: 35% trigger, 28% trail)
  4. DETECT: SPY velocity crash filter blocks new buys if SPY fell >3% in last 3 days; then regime-conditional scan — momentum (up-close) in BULL_CALM, capitulation in BULL_VOLATILE, divergence-from-SPY in CHOPPY, blocked in BEAR; defensives (GLD/TLT/XLV/XLU) allowed in BEAR (1 slot, 15%)
  5. CONFIRM: compute relative-strength score (vs sector ETF) + continuous model score; combined rank (50/50)
  6. EXECUTE: correlation-aware greedy selection (max pairwise correlation 0.70), then sector guard, then orders

**Important**: `main.py` is self-contained. It does **not** import `common/` because LEAN Docker cannot access it.

---

## Layer 3b: Live Trading

**Location**: `live/`
**Entry point**: `python -m live.runner --strategy <name> --broker paper|alpaca-paper|alpaca|ibkr --once`

The live runner loads the same model artifacts as LEAN but executes via broker API:

- `PaperBroker` — simulates fills locally for testing
- `AlpacaBroker` — connects to Alpaca Markets API for paper or live trading (requires `alpaca-py` and `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` env vars)
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
| Minimum hold | 20 days (all multi-stock strategies) | Prevents noise-driven model-signal exits during consolidation; stop-loss still triggers immediately |
| Maximum hold | 500 calendar days | Forces position review; allows long-term capital gains rate |

## Tax-Aware Returns

After-tax returns are computed at each sell event using configurable capital gains rates from the `tax` block in `strategy_config.json`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `short_term_rate` | 0.50 (50%) | Tax rate on gains held < `long_term_threshold_days` |
| `long_term_rate` | 0.32 (32%) | Tax rate on gains held ≥ `long_term_threshold_days` |
| `long_term_threshold_days` | 365 | Days to qualify for long-term rate |

Losses pass through untaxed (loss harvesting is not modeled). In notebooks, tax is deducted from cash at each sell, producing after-tax equity curves. LEAN strategies report tax as metadata via `SetRuntimeStatistic()` (LEAN equity stays gross). The analysis notebook uses `common.add_tax_columns()` to enrich LEAN trade data with per-trade tax breakdowns.

> **Note**: With `max_hold_days: 500`, trades held over 365 days qualify for the 32% long-term rate instead of 50% short-term.

## Position Sizing

Position sizing is configured in `strategy_config.json` under the `position_sizing` block and enforced during both notebook simulation and LEAN backtesting:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_position_pct` | 0.30 (30%) | No single stock can exceed 30% of total portfolio value |
| `cash_reserve_pct` | 0.00 (0%) | All capital available for positions |

**Rules:**
1. **Cash-only buys** — only use available cash for new positions; never sell existing holdings to fund a new buy
2. **Max position cap** — `target_pct = min(max_position_pct, (available_cash - cash_reserve) / portfolio_value)`
3. **Whole shares only** — notebook simulation buys whole shares; LEAN uses `SetHoldings` which handles this internally

These rules are used by single-stock (renquant_101) and multi-stock (renquant_102, renquant_103) strategies. In multi-stock mode, each position is independently capped and the cash reserve is maintained across all positions.

---

## State Space

renquant_102 uses 7 shared relative indicator features for ML models (Classification, Q-Learning) per symbol. renquant_103 adds 4 regime-context features (see below):

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
Q-Learning uses a subset of 5 indicator features (`rsi`, `macd_hist`, `cci`, `bbp`, `adx`) to keep state space tractable (5^5 × 3 = 9,375 states).

renquant_103 adds 4 regime-context features (computed from SPY, appended to the stock frame):

| Feature | Computation | Purpose |
|---------|------------|---------|
| `spy_realized_vol` | SPY 20d return std × √252 | Volatility regime signal |
| `spy_adx` | ADX(14) on SPY | Trend strength of the market |
| `spy_trend` | SPY close / SPY EMA50 | Market trend direction |
| `hurst_proxy` | Autocorr(lag=1) of SPY 20d returns | Fast momentum-persistence proxy |

All 12 registered indicators can be combined freely.

---

## Strategy Details

### renquant_102 — Multi-Stock Pre-Trained Scanner

A 3-stage pipeline strategy: **DETECT** → **CONFIRM** → **EXECUTE**.

**Notebook** (`renquant_102.ipynb`): Trains 3 approaches per symbol on a rolling 3-year window (`training_years=3`), picks the best by OOS after-tax Sharpe ratio, exports one model per symbol to `models/{SYMBOL}/` (minimum Sharpe floor: 0.8). After export, a portfolio-level simulation replicates the LEAN multi-stock logic in Python — scanning bullish volume z-scores, confirming with models, managing concurrent positions — and renders a 4-panel dashboard (equity vs SPY, drawdown, positions held, cash allocation). This enables parameter tuning (z-score threshold, lookback, position sizing) before running LEAN. The 3 approaches are:
1. Dual Momentum — trend-following ManualModel rules
2. Classification — BagLearner(RTLearner) random forest on relative features
3. Q-Learning — tabular RL with discretized trend features

Each symbol's best model may be a different type. The daily automation retrains all models via `daily_103.sh` (renquant_103 is the active live strategy). Models include a `trained_date` field; LEAN skips models older than `model_staleness_days` (default 60).

**Stage 1: DETECT** — compute adaptive per-stock volume score for each watchlist symbol. Default mode: **percentile** — triggers when today's volume is in the top 15% of the 20-day lookback window (P85). Legacy **zscore** mode (threshold 1.5σ) also supported via `volume_filter.mode` config. Bullish filter: only enter on up-close days.

**Stage 2: CONFIRM** — on bullish spike days, apply that stock's pre-trained model to 60-day relative feature history (indicators relativized vs SPY).

**Stage 3: EXECUTE** — if model says "buy" and all constraints pass, enter position. Max 5 concurrent positions.

**Risk management** (all configured in `strategy_config.json` under `risk`):
- **Stop-loss**: exit any position that falls 8% below entry price before waiting for model signal
- **Drawdown circuit breaker**: halt all new buys when portfolio is down ≥15% from its high-water mark
- **Regime filter**: configurable (currently disabled — relative features already encode market context vs SPY)
- **Sector guard**: max 3 concurrent positions per sector (prevents all 5 slots being correlated tech names)

**Config** uses `watchlist` (array of symbols) instead of `stock_symbol`:
```json
{
  "watchlist": ["TSLA", "AMZN", "GOOG", "MSFT", "AMD", "NFLX", "..."],
  "train_split": 0.70,
  "model_staleness_days": 60,
  "volume_zscore_lookback": 20,
  "volume_filter": {"mode": "percentile", "percentile_threshold": 85},
  "max_concurrent_positions": 5,
  "risk": {
    "stop_loss_pct": 0.08,
    "portfolio_drawdown_halt_pct": 0.15,
    "regime_filter": {"enabled": false, "symbol": "SPY", "sma_period": 200}
  },
  "sector_map": {"TSLA": "tech", "JPM": "finance", "...": "..."},
  "max_positions_per_sector": 3
}
```

**LEAN `main.py` flow** (`PreTrainedMultiStockStrategy`):
1. `Initialize()` — load pre-trained models from `models/{SYMBOL}/`, check staleness (log warning if <50% loaded), `AddEquity` for watchlist + SPY
2. `OnData()` — update portfolio high-water mark; process sells first (stop-loss check → max hold check → model signal + constraints); if drawdown circuit breaker triggered, skip all buys; check SPY regime filter; scan volume for non-held stocks (DETECT); rank candidates; apply sector guard; apply pre-trained model (CONFIRM); execute buys (EXECUTE)

**Live runner**: auto-detects multi-stock strategies by checking for `"watchlist"` in config. Uses `run_once_multi()` which computes volume z-scores, checks sell signals, and executes buy orders across the watchlist.

### renquant_103 — Adaptive Regime Multi-Stock

Successor to 102, built for volatile and choppy markets. See full design: [`doc/renquant_103_design.md`](renquant_103_design.md).

Key differences from 102:
- **3-layer regime detection** always running on SPY: Hurst (slow baseline) + CUSUM (fast transition trigger) + GMM (continuous confidence)
- **Regime-conditional entry**: momentum, capitulation, divergence, or blocked depending on regime
- **Stock selection**: earnings filter → volume scan → relative-strength ranking vs sector ETF → continuous model score → combined rank → correlation-aware selection
- **Regime-adaptive parameters**: all risk parameters (stop-loss, position size, cash reserve) adapt per regime and scale with GMM confidence; `max_hold_days` is 500 for most regimes (matching 102), staying 10 days only in CHOPPY
- **Relative-outperformance labels**: Classification model trained with `close = stock_close / spy_close × 100`, so the 5-day forward return label measures stock outperformance vs SPY (not raw return), preventing bull-market always-buy bias
- **Sharpe floor**: 0.8 (matching 102) — high-conviction models only; marginal models below 0.8 are excluded
- **min_hold_days: 20** — no position can be sold via model signal until held 20 days (stop-loss still immediate); extends avg hold to 150-200 days, improving LT tax treatment
- **Consecutive-sell filter**: requires 3 consecutive daily sell signals before exiting a position; eliminates one-day noise flips that would otherwise trigger short-term tax events
- **Defensive tickers**: GLD, TLT, XLV, XLU — triggered as counter-cyclical buys in BEAR/BULL_VOLATILE
- **Trailing stop**: active in BULL_CALM after 20% gain from entry, trails at 18% below the position's rolling high-water mark — locks in gains on large winners without capping upside during normal corrections
- **BEAR regime defensive buys**: when regime = BEAR, offensive buys are blocked but GLD/TLT may be bought (1 slot, max 15% of portfolio) if their model gives a buy signal — provides counter-cyclical hedge instead of going fully to cash
- **Simulation output**: includes trade log (buys/sells, avg hold, avg pnl per trade, total tax, win rate)

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

Uses relative indicator features (7 for renquant_102; 11 for renquant_103, adding 4 SPY regime-context columns). Labels each day by N-day forward return vs a threshold — default: `lookahead=10, threshold=±4%`. renquant_103 overrides to `lookahead=5, threshold=±3%` and uses a **relative close price** (`stock_close / spy_close × 100`) so the label becomes the stock's relative outperformance vs SPY, not its raw return. This prevents the bull-market bias where every stock looks like a buy. BagLearner(RTLearner) ensemble with 15 bags and `leaf_size=25`. Buy/sell thresholds at ±0.1 on the raw tree output. The RF learns nonlinear relationships between relative features automatically — it effectively discovers crossover patterns and conditional logic from the data.

### Q-Learning — Tabular RL with Relative Reward

Uses 5 indicator features (`rsi`, `macd_hist`, `cci`, `bbp`, `adx`) with 5 bins = 9,375 states (5^5 × 3 holding buckets). Key design choice: **reward is relative price returns** (stock/SPY ratio changes), not raw stock returns. In a bull market, raw returns are always positive and the Q-learner learns "buy and never sell." Relative returns can go negative, giving the agent a reason to exit when the stock underperforms the market. Training uses a deterministic per-ticker seed (`abs(hash(ticker)) % 2^32`) for reproducible results across daily retraining runs.

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
  │ Live trader (Alpaca / paper)    │
  └─────────────────────────────────┘
       ↓
  Analysis dashboard + normalized performance chart
  (stock vs SPY vs model equity)
```
