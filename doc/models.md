# Model Types

All models implement `BaseModel` with a common interface:
- `train(df, **kwargs)` — train on indicator-enriched OHLCV data
- `predict(state)` — return `"hold"`, `"buy"`, or `"sell"` for a single row
- `predict_bulk(df)` — return a Series of signals for all rows (vectorized, faster)
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

**Key params**: `feature_columns`, `lookahead` (10), `threshold` (0.04), `leaf_size` (25), `bags` (15), `buy_threshold` (0.5), `sell_threshold` (-0.5)

**With relative features**: The RF ensemble learns nonlinear relationships between relative indicators automatically. It effectively discovers crossover patterns, conditional logic, and regime changes from the data — capturing what simple threshold voting cannot express.

**Tuning note**: The ensemble averages predictions of {-1, 0, +1} across trees. For trending stocks, the average is often close to 0, so the default `±0.5` thresholds may suppress sell signals entirely. Lower to `±0.1` for more active trading.

**When to use**: Best default choice. Fast, deterministic, handles high-dimensional relative features well. Consistently outperforms Manual and Q-Learning in backtests.

## Q-Learning Model

**Type**: `qlearning` | **Module**: `common/models/qlearning.py`

Tabular Q-learning with discretized indicator states. Continuous features are binned (quantile-based), then encoded with holding status into a single state integer. Trains over multiple epochs through the data.

**Key params**: `feature_columns`, `n_bins` (10), `n_epochs` (100), `alpha` (0.2), `gamma` (0.9), `rar` (0.98)

**State space**: `n_bins^n_features * 3` (3 holding buckets: short/flat/long)

### Design considerations with relative features

**Feature selection**: Use 3 trend-following features (`trend`, `rel_mom_20d`, `macd_hist`) instead of 7 oscillators. With 5 bins: `5^3 * 3 = 375 states` — dense enough for ~1000 bars to get good coverage. Using all 7 features creates `5^7 * 3 = 234,375` states with <1 visit each.

**Reward signal**: Use `rel_price` (stock/SPY ratio) as the "close" column so the Q-learner optimizes **relative returns**. With raw stock returns in a bull market, the agent learns "buy and never sell" because returns are always positive. Relative returns can go negative (stock underperforms SPY), giving the agent a real incentive to exit.

**Scaling note**: With many features, reduce `n_bins` to keep state space manageable. E.g., 3 features with 5 bins = 375 states (good). 7 features with 4 bins = 49,152 states (too sparse).

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

## Decision Guide

```
Is your strategy rule-based?
  └─ Yes → Manual (use Dual Momentum, not oscillator voting)
  └─ No → Is the state space small (<15000 states)?
              └─ Yes → Q-Learning (use relative reward + 3 trend features)
              └─ No → Do you have gate signals?
                        └─ Yes → FQI
                        └─ No → Classification (best default)
                                  └─ Want auto-tuned params? → Optimization
```

## Trading Constraints

All models are subject to execution constraints during simulation and LEAN backtesting:

| Constraint | Value | Purpose |
|------------|-------|---------|
| Wash sale | 30 days | Cannot buy within 30 calendar days of selling |
| Min hold | 3 days | Prevents same-day flipping |
| Max hold | 500 days | Forces position review (allows long-term tax rate) |

## Position Sizing

All models use position sizing rules from `strategy_config.json`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_position_pct` | 30% | No single stock exceeds 30% of portfolio value |
| `cash_reserve_pct` | 0% | All capital available for positions |

Buy logic: `invest = min(max_position_pct * portfolio, available_cash - cash_reserve)`. Whole shares only in notebook simulation; LEAN uses `SetHoldings` with the capped percentage. Cash-only buys — never sell existing holdings to fund a new position.

## JSON Artifact Format

All models export to JSON (no pickle) for LEAN compatibility. Each model writes:
- `{name}-policy-metadata.json` — contract between research and backtesting
- Model-specific artifacts (e.g., `{name}-rf-trees.json` for Classification, `{name}-qtable.json` for Q-Learning)
