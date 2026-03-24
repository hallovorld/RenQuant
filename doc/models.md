# Model Types

All models implement `BaseModel` with a common interface:
- `train(df, **kwargs)` — train on indicator-enriched OHLCV data
- `predict(state)` — return `"hold"`, `"buy"`, or `"sell"`
- `save(directory, model_name)` — export as JSON
- `load(directory, model_name)` — load from JSON

## Manual Model

**Type**: `manual` | **Module**: `common/models/manual.py`

Rule-based scoring. Each indicator is evaluated against thresholds and contributes +1 or -1 to a vote. No training required.

**Default rules**: RSI <30 → +1, RSI >70 → -1; MACD hist >0 → +1, <0 → -1; CCI <-100 → +1, >50 → -1. Buy if score >= 2, sell if <= -2.

**When to use**: Baseline strategy, fully interpretable, zero training time.

## Classification Model

**Type**: `classification` | **Module**: `common/models/classification.py`

Bagged Random Forest of RTLearners. Each day is labeled by its N-day forward return as +1 (long), -1 (short), or 0 (hold), with thresholds adjusted for market impact.

**Key params**: `lookahead` (10), `threshold` (0.04), `leaf_size` (25), `bags` (15)

**When to use**: Fast, deterministic, easy to explain. Good default for most strategies.

## Q-Learning Model

**Type**: `qlearning` | **Module**: `common/models/qlearning.py`

Tabular Q-learning with discretized indicator states. Continuous features are binned (quantile-based), then encoded with holding status into a single state integer. Trains over multiple epochs through the data.

**Key params**: `n_bins` (10), `n_epochs` (100), `alpha` (0.2), `gamma` (0.9), `rar` (0.98)

**State space**: `n_bins^n_features * 3` (3 holding buckets: short/flat/long)

**When to use**: Model-free RL, no function approximation. Works well with small state spaces.

## FQI Model (Fitted Q-Iteration)

**Type**: `fqi` | **Module**: `common/models/fqi.py`

Trains one XGBRegressor per action (hold/buy/sell) using Fitted Q-Iteration with discount factor gamma. Requires gate signals (buy_signal/sell_signal) to define valid actions.

**Key params**: `n_iter` (8), `gamma` (0.95), `transaction_cost_bps` (5), `state_columns`

**When to use**: When the state space is too large for tabular Q. The existing NVDA strategy uses this.

## Optimization Model

**Type**: `optimization` | **Module**: `common/models/optimization.py`

Meta-model: SciPy Nelder-Mead searches over indicator parameters while training an inner ClassificationModel. Objective is in-sample cumulative return via portfolio simulation.

**Key params**: `max_iter` (30), `optimizable_params` (list of indicator params to search)

**When to use**: When you suspect default indicator parameters are suboptimal for a given symbol/period.

## Decision Guide

```
Is your strategy rule-based?
  └─ Yes → Manual
  └─ No → Is the state space small (<5000 states)?
              └─ Yes → Q-Learning
              └─ No → Do you have gate signals?
                        └─ Yes → FQI
                        └─ No → Classification
                                  └─ Want auto-tuned params? → Optimization
```

## JSON Artifact Format

All models export to JSON (no pickle) for LEAN compatibility. Each model writes:
- `{name}-policy-metadata.json` — contract between research and backtesting
- Model-specific artifacts (e.g., `{name}-q-hold.json` for FQI, `{name}-qtable.json` for Q-Learning)
