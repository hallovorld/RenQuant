# Model Types

All models implement `BaseModel` with a common interface:
- `train(df, **kwargs)` — train on indicator-enriched OHLCV data
- `predict(state)` — return `"hold"`, `"buy"`, or `"sell"` (string) for a single row or DataFrame (only processes row 0)
- `predict_bulk(df)` — return a Series of strings `"buy"/"hold"/"sell"` for all rows (vectorized); use `.map({"buy": 1, "hold": 0, "sell": -1})` to convert to integers for simulation/Sharpe computation
- `save(directory, model_name)` — export as JSON
- `load(directory, model_name)` — load from JSON

## Manual Model — Dual Momentum + Trend Following

**Type**: `manual` | **Module**: `common/models/manual.py`

Generic indicator-threshold voting. Each rule evaluates one column and contributes +1 (bullish) or -1 (bearish) to a total score. No ML training required.

**Parameters**: `score_rules` (list of rule dicts), `buy_threshold` (default: 2), `sell_threshold` (default: -2)

**Rule format** — each rule is a dict with `col` and one or more conditions:
- `buy_below`: value < threshold → score +1
- `buy_above`: value > threshold → score +1
- `sell_above`: value > threshold → score -1
- `sell_below`: value < threshold → score -1

### Strategy: Dual Momentum (NVDA)

The NVDA strategy implements Dual Momentum (Gary Antonacci) + trend following. This approach uses **trend-following features** where thresholds have clear meaning, unlike oscillator thresholds that whipsaw:

```python
model = create_model("manual", score_rules=[
    {"col": "trend",       "buy_above": 1.0,   "sell_below": 0.97},   # price > 50-day EMA
    {"col": "trend_long",  "buy_above": 1.0,   "sell_below": 0.97},   # price > 200-day EMA
    {"col": "rel_mom_20d", "buy_above": 0.0,   "sell_below": -0.03},  # outperforming SPY 20d
    {"col": "rel_mom_60d", "buy_above": 0.0,   "sell_below": -0.05},  # outperforming SPY 60d
    {"col": "macd_hist",   "buy_above": 0,      "sell_below": 0},      # momentum vs SPY
    {"col": "obv_slope",   "buy_above": 0,      "sell_below": 0},      # volume vs SPY
], buy_threshold=4, sell_threshold=-3)
```

**Why Dual Momentum works better than oscillator voting**: Simple threshold voting on oscillators (RSI, CCI, BBP) is the weakest form of technical analysis — oscillators whipsaw and all fire at once because they're correlated. Trend-following features (`trend`, `rel_mom`) have clear monotonic meaning: above 1.0 = uptrend, positive = outperforming. Thresholds on these features are structurally meaningful.

**When to use**: Interpretable baseline. Zero training time. Best with trend-following features rather than oscillators.

## Classification Model

**Type**: `classification` | **Module**: `common/models/classification.py`

Bagged Random Forest of RTLearners. Each day is labeled by its N-day forward return as +1 (long), -1 (short), or 0 (hold), with thresholds adjusted for market impact.

**Key params**: `feature_columns`, `lookahead` (10), `threshold` (0.04), `leaf_size` (25), `bags` (15), `buy_threshold` (0.1), `sell_threshold` (-0.1)

**Labels are built from `df["close"]`** — to get relative-outperformance labels (which prevent bull-market bias), pass a relative price series as `close`: e.g., `df["close"] = stock_close / spy_close × 100`. The model then labels each day by 5-day relative forward return vs the threshold. renquant_103 uses this technique with `lookahead=5, threshold=0.03`.

**With relative features**: The RF ensemble learns nonlinear relationships between relative indicators automatically. It effectively discovers crossover patterns, conditional logic, and regime changes from the data — capturing what simple threshold voting cannot express.

**Tuning note**: The ensemble averages predictions of {-1, 0, +1} across trees. For trending stocks, the average is often close to 0, so wide thresholds suppress signals entirely. The renquant_102 default is `±0.1` for active trading; raise toward `±0.5` to filter only high-confidence signals.

**When to use**: Best default choice. Fast, deterministic, handles high-dimensional relative features well. Consistently outperforms Manual and Q-Learning in backtests.

## Q-Learning Model

**Type**: `qlearning` | **Module**: `common/models/qlearning.py`

Tabular Q-learning with discretized indicator states. Continuous features are binned (quantile-based), then encoded with holding status into a single state integer. Trains over multiple epochs through the data.

**Key params**: `feature_columns`, `n_bins` (5), `n_epochs` (500), `alpha` (0.15), `gamma` (0.95), `rar` (0.99)

**State space**: `n_bins^n_features * 3` (3 holding buckets: short/flat/long)

### Design considerations with relative features

**Feature selection**: Use 5 indicator features (`rsi`, `macd_hist`, `cci`, `bbp`, `adx`) — these are the first 5 entries of the shared `feature_columns` list. With 5 bins: `5^5 * 3 = 9,375 states` — manageable for ~500+ training bars. Using all 11 features (including 4 regime-context columns) creates `5^11 * 3 = 146M` states — completely intractable.

**Reward signal**: Use `rel_price` (stock/SPY ratio) as the "close" column so the Q-learner optimizes **relative returns**. With raw stock returns in a bull market, the agent learns "buy and never sell" because returns are always positive. Relative returns can go negative (stock underperforms SPY), giving the agent a real incentive to exit.

**Reproducibility**: Training uses a deterministic per-ticker seed (`abs(hash(ticker)) % 2^32`) so daily retraining produces the same model for each symbol regardless of execution order.

**Scaling note**: With many features, reduce `n_bins` to keep state space manageable. E.g., 5 features with 5 bins = 9,375 states (good). 7 features with 5 bins = 234,375 states (too sparse).

**When to use**: Model-free RL exploration. Best with small feature sets and relative reward.

## FQI Model (Fitted Q-Iteration)

**Type**: `fqi` | **Module**: `common/models/fqi.py`

Trains one XGBRegressor per action (hold/buy/sell) using Fitted Q-Iteration with discount factor gamma. Requires gate signals (buy_signal/sell_signal) to define valid actions.

**Key params**: `n_iter` (8), `gamma` (0.95), `transaction_cost_bps` (5), `state_columns`

**When to use**: When the state space is too large for tabular Q and you need function approximation.

## Optimization Model

**Type**: `optimization` | **Module**: `common/models/optimization.py`

Meta-model: SciPy Nelder-Mead searches over indicator parameters while training an inner ClassificationModel. Objective is in-sample cumulative return via portfolio simulation.

**Key params**: `max_iter` (30), `optimizable_params` (list of indicator params to search)

**When to use**: When you suspect default indicator parameters are suboptimal for a given symbol/period.

## XGBoost Model

**Type**: `xgboost` | **Module**: `common/models/xgboost_model.py`

Gradient-boosted trees using two one-vs-rest XGBClassifier instances (buy probability and sell probability). Labels are the same as Classification (N-day forward relative return vs threshold), but the learning algorithm is fundamentally different: each tree corrects the residual errors of the previous trees (boosting), with L1/L2 regularisation and row/column subsampling to prevent overfitting.

**Key advantages over RTLearner/BagLearner:**
- Residual boosting corrects previous errors systematically (vs random forests that average independent noisy trees)
- L1/L2 regularisation + `min_child_weight` prevent overfitting on short time-series
- `scale_pos_weight` handles class imbalance (buy/sell labels are typically 20-30% of data)
- Feature importance scores (averaged over buy+sell models) give interpretable attribution
- Saves to JSON natively via XGBoost's own format

**Key params**: `feature_columns`, `lookahead` (5), `threshold` (0.03), `buy_threshold` (0.1), `sell_threshold` (0.1), `n_estimators` (200), `max_depth` (4), `learning_rate` (0.05), `subsample` (0.8), `colsample_bytree` (0.8), `min_child_weight` (10)

**Signal**: `P(buy) - P(sell)` → buy if score > buy_threshold-0.5, sell if score < -(sell_threshold-0.5), hold otherwise.

**Artifacts**: `{name}-xgb-buy.json` + `{name}-xgb-sell.json` + `{name}-policy-metadata.json`

**When to use**: Primary replacement for Classification when higher prediction quality is needed. Consistently outperforms RTLearner/BagLearner on out-of-sample Sharpe for complex multi-feature setups. Tournament in renquant_103 picks the best of Classification / QLearning / Manual / XGBoost per ticker.

**Note**: Requires `pip install xgboost`. Not yet supported in LEAN main.py (notebook simulation only until LEAN layer is extended).

## Decision Guide

```
Is your strategy rule-based?
  └─ Yes → Manual (use Dual Momentum, not oscillator voting)
  └─ No → Is the state space small (<15000 states)?
              └─ Yes → Q-Learning (use relative reward + 5 indicator features)
              └─ No → Do you have gate signals?
                        └─ Yes → FQI
                        └─ No → XGBoost (best default — gradient boosting, regularised)
                                  └─ Want interpretable fallback? → Classification (RTLearner)
                                  └─ Want auto-tuned params? → Optimization
```

## Trading Constraints

All models are subject to execution constraints during simulation and LEAN backtesting:

| Constraint | Value | Purpose |
|------------|-------|---------|
| Wash sale | 30 days | Cannot buy within 30 calendar days of selling |
| Min hold | 20 days (all multi-stock strategies) | Prevents noise-driven model-signal exits during early hold period |
| Max hold | 500 days | Forces position review (allows long-term tax rate) |

## Position Sizing

All models use position sizing rules from `strategy_config.json`:

| Parameter | renquant_103 value | Purpose |
|-----------|-------------------|---------|
| `max_position_pct` | 15% | No single stock exceeds 15% of portfolio (8 concurrent slots) |
| `cash_reserve_pct` | 0% | All capital available for positions |
| `max_concurrent_positions` | 8 | Diversification across 8 simultaneous positions |

Buy logic: `invest = min(max_position_pct * portfolio, available_cash - cash_reserve)`. Whole shares only in notebook simulation; LEAN uses `SetHoldings` with the capped percentage. Cash-only buys — never sell existing holdings to fund a new position.

## Exit Logic (renquant_103 priority order)

1. **Trailing stop** (BULL_CALM): activates once position gains ≥35% from entry; then trails 28% below rolling high-water mark. Allows winners like NVDA/PLTR to run through minor corrections.
2. **Hard stop-loss**: 5% from entry price (triggers immediately, no min-hold).
3. **SPY velocity crash filter** (entry gate): blocks all new buys if SPY has fallen >3% over the last 3 days — prevents entering into momentum crashes (tariff events, flash crashes).
4. **Max hold**: forced exit after 500 days (BULL_CALM/BEAR) or 10 days (CHOPPY).
5. **Model sell**: 3 consecutive daily sell signals with min 20-day hold.

## JSON Artifact Format

All models export to JSON (no pickle) for LEAN compatibility. Each model writes:
- `{name}-policy-metadata.json` — contract between research and backtesting
- Model-specific artifacts:
  - Classification: `{name}-rf-trees.json`
  - Q-Learning: `{name}-qtable.json` + `{name}-bin-edges.json`
  - Manual: `{name}-manual-rules.json`
  - XGBoost: `{name}-xgb-buy.json` + `{name}-xgb-sell.json`
