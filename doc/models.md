# Model Types

All models implement `BaseModel` with a common interface:
- `train(df, **kwargs)` — train on indicator-enriched OHLCV data
- `predict(state)` — return `"hold"`, `"buy"`, or `"sell"` for a single row
- `predict_bulk(df)` — return a Series of signals for all rows (vectorized, faster)
- `save(directory, model_name)` — export as JSON
- `load(directory, model_name)` — load from JSON

## Manual Model

**Type**: `manual` | **Module**: `common/models/manual.py`

Generic indicator-threshold voting. Each rule evaluates one indicator column and contributes +1 (bullish) or -1 (bearish) to a total score. No training required.

**Parameters**: `score_rules` (list of rule dicts), `buy_threshold` (default: 2), `sell_threshold` (default: -2)

**Rule format** — each rule is a dict with `col` and one or more conditions:
- `buy_below`: value < threshold → score +1
- `buy_above`: value > threshold → score +1
- `sell_above`: value > threshold → score -1
- `sell_below`: value < threshold → score -1

**Example**:
```python
model = create_model("manual", score_rules=[
    {"col": "rsi",        "buy_below": 40,  "sell_above": 65},
    {"col": "macd_hist",  "buy_above": 0,   "sell_below": 0},
    {"col": "adx",        "buy_above": 25},
    {"col": "obv_slope",  "buy_above": 0,   "sell_below": 0},
], buy_threshold=2, sell_threshold=-2)
```

**When to use**: Baseline strategy, fully interpretable, zero training time. Supports any registered indicator.

## Classification Model

**Type**: `classification` | **Module**: `common/models/classification.py`

Bagged Random Forest of RTLearners. Each day is labeled by its N-day forward return as +1 (long), -1 (short), or 0 (hold), with thresholds adjusted for market impact.

**Key params**: `feature_columns`, `lookahead` (10), `threshold` (0.04), `leaf_size` (25), `bags` (15), `buy_threshold` (0.5), `sell_threshold` (-0.5)

**Tuning note**: The ensemble averages predictions of {-1, 0, +1} across trees. For trending stocks, the average is often close to 0, so the default `±0.5` thresholds may suppress sell signals entirely. Lower to `±0.1` for more active trading.

**When to use**: Fast, deterministic, easy to explain. Good default for most strategies.

## Q-Learning Model

**Type**: `qlearning` | **Module**: `common/models/qlearning.py`

Tabular Q-learning with discretized indicator states. Continuous features are binned (quantile-based), then encoded with holding status into a single state integer. Trains over multiple epochs through the data.

**Key params**: `feature_columns`, `n_bins` (10), `n_epochs` (100), `alpha` (0.2), `gamma` (0.9), `rar` (0.98)

**State space**: `n_bins^n_features * 3` (3 holding buckets: short/flat/long)

**Scaling note**: With many features, reduce `n_bins` to keep state space manageable. E.g., 6 features with 4 bins = 4^6 * 3 = 12,288 states (fits in memory). 6 features with 10 bins = 3,000,000 states (does not).

**When to use**: Model-free RL, no function approximation. Works well with small state spaces.

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
  └─ Yes → Manual
  └─ No → Is the state space small (<15000 states)?
              └─ Yes → Q-Learning
              └─ No → Do you have gate signals?
                        └─ Yes → FQI
                        └─ No → Classification
                                  └─ Want auto-tuned params? → Optimization
```

## JSON Artifact Format

All models export to JSON (no pickle) for LEAN compatibility. Each model writes:
- `{name}-policy-metadata.json` — contract between research and backtesting
- Model-specific artifacts (e.g., `{name}-rf-trees.json` for Classification, `{name}-qtable.json` for Q-Learning)
