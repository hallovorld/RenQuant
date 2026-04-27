# Model Types

All models implement `BaseModel` with a common interface:
- `train(df, **kwargs)` — train on indicator-enriched OHLCV data
- `predict(state)` — return `"hold"`, `"buy"`, or `"sell"` (string) for a single row or DataFrame (only processes row 0)
- `predict_bulk(df)` — return a Series of strings `"buy"/"hold"/"sell"` for all rows (vectorized); use `.map({"buy": 1, "hold": 0, "sell": -1})` to convert to integers for simulation/Sharpe computation
- `predict_score_bulk(df)` — return a Series of continuous float scores in the model's native scale. Classification returns raw BagLearner output; Q-Learning returns Q(buy)−Q(sell); XGBoost returns P(buy)−P(sell); Manual returns raw vote count (positive = buy pressure, negative = sell pressure).
- `save(directory, model_name)` — export as JSON
- `load(directory, model_name)` — load from JSON

## Score Contract

Raw model scores are intentionally preserved in their native semantics for interpretability and debugging, but they are **not directly comparable across model families**. A raw score of `2.0` from a Manual model and a raw score of `0.25` from XGBoost do not mean the same thing.

The live runner therefore uses a two-layer contract:

- `raw_score` — the native output from `predict_score_bulk()`; logged for diagnostics and model introspection
- `rank_score` — a calibrated score used for portfolio filtering, cross-model ranking, and tier thresholds

Calibration is handled by `kernel/scoring.py` (renquant_103/104) or `common/models/scoring.py` (renquant_101/102):

- If `policy-metadata.json` contains `score_calibration`, that artifact is used directly.
- Otherwise the live runner fits a fallback isotonic calibration from recent symbol history.
- The calibration target is `P(5d relative return > threshold)` using the model's own training horizon/threshold when available.

This keeps one champion model per symbol while putting mixed model families onto a common ranking scale.

### renquant_104 — Panel-LTR override

In renquant_104 the per-ticker `rank_score` above is **overwritten cross-sectionally** by `PanelScoringJob` (see `doc/arch/strategy-104.md`) whenever `ranking.panel_scoring.enabled=true` in `strategy_config.json`. A single XGBoost learning-to-rank model, trained on neutralized forward excess returns across the whole watchlist, emits `panel_score` — which is also written into `rank_score` so every downstream consumer (ranking blend, tier thresholds, rotation advantage) sees a directly comparable cross-sectional score. Short-circuits to the per-ticker calibrated score when the flag is off.

## Manual Model — Dual Momentum + Trend Following

**Type**: `manual` | **Module**: `kernel/models.py` (renquant_103) / `common/models/manual.py` (101/102)

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

**Type**: `classification` | **Module**: `kernel/models.py` (renquant_103) / `common/models/classification.py` (101/102)

Bagged Random Forest of RTLearners. Each day is labeled by its N-day forward return as +1 (long), -1 (short), or 0 (hold), with thresholds adjusted for market impact.

**Key params**: `feature_columns`, `lookahead` (10), `threshold` (0.04), `leaf_size` (25), `bags` (15), `buy_threshold` (0.1), `sell_threshold` (-0.1)

**Labels are built from `df["close"]`** — to get relative-outperformance labels (which prevent bull-market bias), pass a relative price series as `close`: e.g., `df["close"] = stock_close / spy_close × 100`. The model then labels each day by 5-day relative forward return vs the threshold. renquant_103 uses this technique with `lookahead=5, threshold=0.03`.

**With relative features**: The RF ensemble learns nonlinear relationships between relative indicators automatically. It effectively discovers crossover patterns, conditional logic, and regime changes from the data — capturing what simple threshold voting cannot express.

**Tuning note**: The ensemble averages predictions of {-1, 0, +1} across trees. For trending stocks, the average is often close to 0, so wide thresholds suppress signals entirely. The renquant_102 default is `±0.1` for active trading; raise toward `±0.5` to filter only high-confidence signals.

**When to use**: Best default choice. Fast, deterministic, handles high-dimensional relative features well. Consistently outperforms Manual and Q-Learning in backtests.

## Q-Learning Model

**Type**: `qlearning` | **Module**: `kernel/models.py` (renquant_103) / `common/models/qlearning.py` (101/102)

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

**Type**: `fqi` | **Module**: `common/models/fqi.py` (101/102 only — not used in renquant_103)

Trains one XGBRegressor per action (hold/buy/sell) using Fitted Q-Iteration with discount factor gamma. Requires gate signals (buy_signal/sell_signal) to define valid actions.

**Key params**: `n_iter` (8), `gamma` (0.95), `transaction_cost_bps` (5), `state_columns`

**When to use**: When the state space is too large for tabular Q and you need function approximation.

## Optimization Model

**Type**: `optimization` | **Module**: `common/models/optimization.py` (101/102 only — not used in renquant_103)

Meta-model: SciPy Nelder-Mead searches over indicator parameters while training an inner ClassificationModel. Objective is in-sample cumulative return via portfolio simulation.

**Key params**: `max_iter` (30), `optimizable_params` (list of indicator params to search)

**When to use**: When you suspect default indicator parameters are suboptimal for a given symbol/period.

## XGBoost Model

**Type**: `xgboost` | **Module**: `kernel/models.py` (renquant_103) / `common/models/xgboost_model.py` (101/102)

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

**Note**: Requires `pip install xgboost`. Fully supported in LEAN `main.py` (renquant_103) — XGBoost artifacts are loaded and scored via a pure-Python tree traversal in `_xgb_predict()` / `_get_raw_model_score()`.

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
| Min hold | 5 days (renquant_104) / 30 days (renquant_103) | Prevents noise-driven model-signal exits during early hold period — tuned per strategy via `min_hold_days` in `strategy_config.json` |
| Max hold | 500 days (BULL / BEAR), 40 days (CHOPPY) | Forces position review (allows long-term tax rate) |

## Position Sizing

All models use position sizing rules from `strategy_config.json`:

| Parameter | renquant_103/104 value | Purpose |
|-----------|------------------------|---------|
| `max_position_pct` | 15% | No single stock exceeds 15% of portfolio (8 concurrent slots) |
| `cash_reserve_pct` | 0% | All capital available for positions |
| `max_concurrent_positions` | 8 | Diversification across 8 simultaneous positions |

Buy logic: `invest = min(max_position_pct * portfolio, available_cash - cash_reserve)`. Whole shares only in notebook simulation; LEAN uses `SetHoldings` with the capped percentage. Cash-only buys — never sell existing holdings to fund a new position.

**renquant_104 conviction multiplier**: when `ranking.panel_scoring.sizing.enabled=true`, `max_position_pct` is multiplied by `conviction_multiplier(panel_score)` before sizing. The multiplier maps panel score to a `[min_mult, 1.0]` range using config floor/ceiling bounds — higher-conviction candidates get up to the full `max_position_pct`, weaker ones are sized down to `min_mult × max_position_pct`. See `kernel/sizing.py::conviction_multiplier`.

### Kelly-optimal sizing (golden v4+)

When `ranking.kelly_sizing.enabled=true`, `ApplyKellySizingTask` computes a per-candidate target weight directly from μ/σ² (NGBoost predictions):

```
f* = μ / σ²
kelly_target = min(max_position_pct × confidence, max_concentration, fractional × f*)
```

Three decision surfaces share this target:

1. **`SizeAndEmitTask`** — new-buy size clamped to `kelly_target × conviction × sigma_mult` (or `kelly_target` alone when `disable_extra_multipliers=true`, per 2026-04-24 pure-Kelly flag)
2. **`TopUpHeldTask`** — adds to under-weight holdings when `kelly_target - current_pct > top_up_threshold` (default 0.05)
3. **`TrimHeldTask`** (**opt-in, `trim_enabled: false` default after 2026-04-24 A/B showed net regression**) — emits partial sell when `current_pct - kelly_target > trim_threshold`. Audit guards: skip when `kelly_target < trim_target_floor` (default 0.05, kelly signal too noisy) or `hs.mu <= 0` (model turned bearish — use full exit path).

### Multi-entry accumulation (`per_session_buy_cap`)

When set (e.g. 0.35), any single-bar BUY order is capped at that fraction regardless of `kelly_target`. Over multiple sessions, TopUpHeldTask can continue building toward the full target up to `max_concentration`. User spec 2026-04-24: "65% OK, but not from one session".

### Partial-sell infra

`ExitSignal` carries optional `quantity: float | None`:
- `None` (default) → full liquidation
- `0 < quantity < current_shares` → partial sell; position stays open with cost basis + tenure preserved
- `quantity ≥ current_shares` → full liquidation (same as None)

Adapter paths in `sim.py::_apply_sell` and `runner.py` both honour this; live runner also preserves `entry_dates` / `position_hwm` / `sell_streaks` on partial trims.

### Thesis-degradation rotation (opt-in, `thesis_rotation.enabled: false` default)

Rotation comparison of today's candidate vs held position's **fixed entry-time baseline** (not noisy today-vs-today Kelly deltas). Rules:

- `degradation = (held.entry_rank_score − held.today_rank_score) / held.entry_rank_score`
- `uplift = cand.today_rank_score − held.entry_rank_score`
- Swap only when `degradation ≥ degradation_pct AND uplift ≥ uplift_pct`

`HoldingState.entry_{rank,panel,kelly_target}_score` fields are stamped at buy time by all three adapters (sim, LEAN, live). Live runner persists them in `live_state.json::entry_signals`.

## Exit Logic (renquant_103/104 priority order)

1. **Trailing stop** (BULL_CALM only): activates once position's peak gain (HWM-based) reaches ≥20% from entry; then trails 18% below the rolling high-water mark. Stop stays armed even after pullbacks — uses peak gain, not current gain. Allows winners like NVDA/PLTR to run through minor corrections.
2. **Cumulative stop-loss**: 15% from entry in BULL_CALM; 5% in BULL_VOLATILE / CHOPPY / BEAR. Triggers immediately, no min-hold gating.
3. **Single-day loss gate** (BULL_CALM only): exits if today's close drops ≥10% from yesterday's close. Protects against gap-down days where a 20%+ single-session drop would escape the 15% cumulative stop until the next bar. Disabled in other regimes (5% cumulative stop is already tight).
4. **Max hold**: forced exit after 500 days (BULL_CALM/BULL_VOLATILE/BEAR) or 40 days (CHOPPY).
5. **Tax-aware hold gate** (`lt_hold_gate_days=330`, `lt_hold_min_gain=10%`): suppresses model-sell (but not hard stops) at days 330–364 if gain ≥10%, to preserve the upcoming long-term tax rate.
6. **Model sell**: 3 consecutive daily sell signals, gated by `min_hold_days` (30 in renquant_103; 5 in renquant_104).

## JSON Artifact Format

All models export to JSON (no pickle) for LEAN compatibility. Each model writes:
- `{name}-policy-metadata.json` — contract between research and execution; may also carry optional `score_calibration` metadata for live cross-model ranking
- Model-specific artifacts:
  - Classification: `{name}-rf-trees.json`
  - Q-Learning: `{name}-qtable.json` + `{name}-bin-edges.json`
  - Manual: `{name}-manual-rules.json`
  - XGBoost: `{name}-xgb-buy.json` + `{name}-xgb-sell.json`
