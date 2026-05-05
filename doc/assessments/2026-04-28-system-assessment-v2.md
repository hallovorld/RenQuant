# renquant_104 Professional System Assessment — v2
**Date:** 2026-04-28 (written 2026-04-29)
**Scope:** Code-only. Every claim cites the specific file + function/line that proves it. Claims not findable in code are marked _code not found / unverifiable_.
**Source tree root:** `backtesting/renquant_104/`

---

## 1. Executive Summary

| Dimension | Score (1–10) | One-sentence justification (code citation) |
|---|---|---|
| **CV correctness** | 6 | Three structural bugs (linspace drift, early-stop misalignment, calendar-day purge) were fixed on 2026-04-28; fixes are in `purged_cv.py` and confirmed by code, but all prior IC numbers were produced under the broken regime and cannot be trusted retrospectively. |
| **Label quality** | 8 | Labels are residualized, cross-sectionally Gaussianized, and purged correctly in bars (`labels.py::_rolling_beta_purged`, `labels.py::gaussianize_cross_section`); no obvious look-ahead bias found in label construction code. |
| **Feature engineering** | 7 | 27 well-diversified features spanning technicals, cross-sectional factors, intraday microstructure, and fundamentals; cross-sectional z-scoring is correct; minute-level features introduce a hard operational dependency that is not guarded at inference. |
| **Model architecture** | 5 | XGBoost rank:pairwise stops at `best_iter=19` (total shrinkage = 0.38) on a 77k-row panel; heavy regularization is intentional but the model is shallower than its capacity warrants, and train IC (0.115) is 3.3× OOS IC (0.035), a gap unexplained by regularization alone. |
| **OOS IC reliability** | 5 | CPCV mean IC = 0.0350 ± 0.030 (std is 85% of mean); Q05 is −0.011; one of the 15 CPCV paths is negative (−0.0052); IC is real but fragile (`panel-ltr.json` metadata fields `oos_mean_ic`, `oos_std_ic`, `oos_per_fold_ic`). |
| **NGBoost μ/σ head** | 4 | val_mu_ic = 0.021 (barely above noise), trained one day before the panel-LTR on a different data snapshot; serialized as a pickle blob, making it version-sensitive (`ngboost-head.json::regressor_pickle_b64`). |
| **Kelly/position sizing** | 5 | Formula is correct (`kelly.py::kelly_target_pct`), but `fractional=0.5` in production (`strategy_config.json` line 519) is double the classical quarter-Kelly safety margin; combined with NGBoost's noisy σ, position sizes are likely systematically overconfident. |
| **Risk management** | 6 | Drawdown circuit is correctly implemented (`task_drawdown.py`); BULL_CALM halt threshold of 35% is extremely permissive for a $10k account; halting buys does not force partial exits, allowing deep underwater positions to persist indefinitely. |
| **Live trading infra** | 7 | Pre-flight smoke test, config fingerprint guard, and ntfy alerting are all present and code-verified; crash behaviour (`runner.py` exception loop) is graceful; no retry/reconnect logic for broker API failures beyond the outer exception catch. |
| **Operational hygiene** | 5 | 83 artifact JSON files in the `artifacts/` directory with no lifecycle management; artifact filenames are the only protection against an experimental retrain overwriting production; no atomic swap. |

---

## 2. CV & Training Validity

### 2.1 Fold Construction

**File:** `training_panel/purged_cv.py`

`CombinatorialPurgedCV.split()` constructs fold boundaries with:

```python
# purged_cv.py — CombinatorialPurgedCV.split()
fold_size = n_dates // self.n_splits
fold_edges = [k * fold_size for k in range(self.n_splits + 1)]
fold_edges[-1] = n_dates
```

This is the **BUG-CV-1 fix** landed 2026-04-28. The prior code used `np.linspace(0, n_dates, n_splits+1, dtype=int)` which drifted fold edges by 1–2 positions on a daily rolling window. The fix is correct: integer division gives stable calendar-date boundaries regardless of `n_dates` variance.

**Production artifact parameters** (`panel-ltr.json` metadata fields):
- `cv_method: "cpcv"`, `cv_n_splits: 6`, `cv_n_test_groups: 2`
- → C(6, 2) = **15 CPCV paths** (confirmed by `len(oos_per_fold_ic) = 15`)
- `cv_embargo_days: 10`, `lookahead_days: 10`

**Important:** All prior CPCV IC numbers (including the +0.0418 referenced in CLAUDE.md) were produced under the broken linspace fold construction. They must be treated as unreliable. The current artifact (trained 2026-04-28) is the first clean measurement.

### 2.2 Purge Logic

**File:** `training_panel/purged_cv.py`, `PurgedKFold.split()` and `CombinatorialPurgedCV.split()`

The pre-2026-04-27 code purged by calendar days using `pd.Timedelta(days=L)`. Because labels are constructed by `close.shift(-lookahead)` (a bar shift), the correct purge unit is **trading bars**, not calendar days. A 10-bar lookahead spans ~14 calendar days; the old calendar-day purge of 10 days left ~3 trading bars of leakage.

The fix correctly operates in bar space:

```python
purge_lo  = max(0,       lo - int(self.lookahead_days))
embargo_hi = min(n_dates, hi + int(self.embargo_days))
purge_dates   = unique_dates[purge_lo:lo]
embargo_dates = unique_dates[hi:embargo_hi]
```

`unique_dates` contains only trading days, so indexing it directly gives the correct bar-count purge.

### 2.3 Label Construction

**File:** `training_panel/labels.py`

Forward return formula (from `labels.py`, confirmed by sub-agent read):

```python
fwd_returns[t] = c.shift(-lookahead) / c - 1.0  # lookahead = 10 bars in production
```

Beta computation is purged by `lookahead_days` (the `_rolling_beta_purged` function shifts both series by `purge` before the rolling window, ensuring the regression does not see future data):

```python
y_s = y.shift(purge)   # purge = lookahead_days = 10
x_s = x.shift(purge)
cov = y_s.rolling(window, min_periods=window).cov(x_s)
```

After residualization against SPY and sector betas, labels are Gaussianized per cross-section (per date) via rank → uniform → inverse-normal:

```python
# labels.py::gaussianize_cross_section
ranks = r[mask].rank(method="average")
u = ranks / (n + 1)
return norm.ppf(u.values)
```

**Assessment:** Label construction is methodologically sound. The purge-in-bars fix makes the purge correctly aligned with the bar-shift label.

**One edge case:** With `n=1` (single ticker on a date), the code returns `pd.Series(0.0)` — a scalar 0 instead of NaN. This will silently produce a "label = 0" for isolated single-ticker dates rather than dropping the row. Impact is small (rare condition) but not ideal.

### 2.4 Early Stopping

**File:** `training_panel/ltr_model.py`, `PanelLTRModel.train()`; `training_panel/pp_panel_training.py`, `FinalFitTask.run()`

Early stopping uses a **Python-level chunk training loop** (not XGBoost's native early stopping) because XGBoost 3.x's built-in `rank:ndcg` metric requires integer labels which Gaussianized labels cannot satisfy:

```python
# ltr_model.py::PanelLTRModel.train()
chunk_size = max(5, int(early_stopping_rounds) // 4)  # = max(5, 20//4) = 5 rounds/chunk
min_delta_ic = 1e-3
# trains in 5-round chunks, stops when patience exhausted
```

The eval set is aligned to CPCV fold size (**BUG-CV-3 fix**, `pp_panel_training.py::FinalFitTask.run()`):

```python
cv_splits_for_eval = int(cfg.get("cv_n_splits", 6))
n_eval = max(2, n_total // max(2, cv_splits_for_eval))   # = n_dates // 6 ≈ 125 dates
```

Prior to this fix, the eval set was a hardcoded 20% holdout — a different slice from the CPCV evaluation window — making early stopping and IC measurement incoherent.

**Production result:** `best_iter = 19` (`panel-ltr.json` field `best_iter`). With `eta = 0.02`:

> total shrinkage = 19 × 0.02 = **0.38**

The model has effectively 19 depth-3 trees. The **min_best_iter guard** (`FinalFitTask.run()`) would raise if `best_iter < 5`. At 19, it passes, but 19 rounds is still surprisingly shallow for a 77,353-row panel. Two hypotheses:
1. The eval IC genuinely saturates at round 19 (the panel IC ceiling is simply low and there is nothing more to fit).
2. The eval set construction (lookahead-purged last 1/6 of dates) coincidentally produces a high-noise eval slice.

The round-9-saturation diagnostic referenced in the code comments (`pp_panel_training.py` line 2090) confirms this is known and under investigation.

**The min_best_iter threshold of 5 is too permissive.** A model stopping at rounds 5, 6, 7, or 8 (total shrinkage 0.10–0.16) would be accepted to production. The original guard of 20 was lowered without a clear lower-bound justification for what constitutes "untrained." This needs an evidence-based floor, not an empirical floor derived from a single diagnostic run.

### 2.5 Training Window

**File:** `training_panel/pp_panel_training.py`; `panel-ltr.json` metadata

Panel shape from artifact: `{'rows': 77353, 'tickers': 103, 'dates': 751}`. 751 trading days ≈ 3 years of history. `min_history_days: 252` means tickers with < 1 year of history are dropped from training rows (panel_frame.py warmup logic).

The beta window is `beta_window: 252` (1 year rolling). This is correctly reflected in the artifact.

---

## 3. Feature Engineering

### 3.1 Complete Feature List (27 features — from `panel-ltr.json::feature_cols`)

**Technical oscillators (4) — raw values, not z-scored:**
| Feature | Description |
|---|---|
| `adx` | Average Directional Index — trend strength |
| `bbp` | Bollinger Band position |
| `cci` | Commodity Channel Index |
| `williams_r` | Williams %R oscillator |

**Trend indicators (2) — raw:**
| Feature | Description |
|---|---|
| `trend` | Short-term trend signal |
| `trend_long` | Longer-term trend signal |

**Cross-sectional momentum / factor z-scores (8) — z-scored per cross-section:**
| Feature | Description |
|---|---|
| `rel_mom_20d` | 20-day relative momentum (vs universe) |
| `rel_mom_60d` | 60-day relative momentum |
| `size_z` | Market cap factor |
| `mom_12_1_z` | 12-month minus 1-month momentum (standard Jegadeesh-Titman) |
| `beta_60d_z` | 60-day rolling beta to SPY |
| `resid_mom_z` | Momentum orthogonalized to SPY beta |
| `price_to_high_z` | Price relative to 52-week high |
| `realized_vol_z` | Historical realized volatility |

**Earnings / fundamental (5) — z-scored:**
| Feature | Description |
|---|---|
| `earnings_surprise_cum_z` | Trailing 4-quarter cumulative EPS surprise |
| `roe_z` | Return on equity |
| `gross_profitability_z` | Gross profit / total assets |
| `book_to_price_z` | Book-to-market ratio |
| `short_pct_float_z` | Short interest as % of float |

**Intraday hourly microstructure (3) — z-scored:**
| Feature | Description |
|---|---|
| `afternoon_drift_z` | Afternoon price drift vs open |
| `vwap_premium_z` | Hourly VWAP premium |
| `intraday_realized_vol_z` | Realized vol measured at hourly bars |

**Minute-level microstructure (5) — z-scored:**
| Feature | Description |
|---|---|
| `m_morning_30min_drift_z` | First 30-minute drift from open |
| `m_vwap_premium_z` | Minute-level VWAP premium |
| `m_intraday_realized_vol_z` | Realized vol at minute resolution |
| `m_overnight_gap_z` | Overnight gap (prev close → open) |
| `m_reversal_ratio_z` | Intraday reversal tendency |

### 3.2 Feature Correctness Assessment

**Technical oscillators:** Standard formulas applied per-ticker historically. No look-ahead risk detected.

**Cross-sectional z-scoring:** Performed per-date (groupby date), so each ticker's z-score only uses information from tickers present on that date. Correct.

**Momentum (mom_12_1_z, resid_mom_z):** Standard construction; the 1-month skip avoids short-term reversal contamination. Correct.

**Earnings surprise (earnings_surprise_cum_z):** This is the most look-ahead-prone feature. Point-in-time earnings data is notoriously difficult to get right — if the earnings calendar (`earnings-calendar.json` artifact) uses "as-of" dates incorrectly, this feature leaks. **The code in `factors.py` uses `yfinance earnings_dates` which updates step-wise at each announcement and is forward-filled.** This is the correct construction but relies on yfinance providing point-in-time data, which it does not guarantee. This is an unverified assumption.

**Short interest (short_pct_float_z):** Short interest is typically reported with a 2-week lag. `code not found / unverifiable` — the code does not show the lag correction for short interest data. If the current short interest reading is used without lag, this feature has ~10-trading-day look-ahead bias.

**Minute-level features:** These require minute bar data at inference time. `code not found / unverifiable` — no guard in the inference path was found for "what happens when minute data is unavailable?" If the minute data pipeline fails, features `m_*` will be NaN. The `{col}_is_missing` binary flag system (`panel_frame.py`) will catch this, but the model was trained with few such missing rows — inference on all-NaN minute features will produce out-of-distribution inputs.

**Cross-sectional neutralization:** `neutralize_features: True` in the artifact. This confirms features are neutralized before training, which reduces sector/factor concentration in the learned signal.

### 3.3 Look-Ahead Bias Assessment

No structural look-ahead bias found in the label construction (forward returns, beta purge, Gaussianization). The feature construction uses only lagged/trailing windows. **Two conditional concerns:**
1. Short interest lag correction: unverifiable from code.
2. Earnings surprise point-in-time: relies on yfinance data quality, not architecturally guaranteed.

---

## 4. Model Architecture

### 4.1 XGBoost Panel-LTR

**Objective:** `rank:pairwise` (confirmed: `panel-ltr.json::params::objective`).

**Production hyperparameters** (`panel-ltr.json::params`):

```json
{
  "objective": "rank:pairwise",
  "eta": 0.02,
  "max_depth": 3,
  "min_child_weight": 60,
  "subsample": 0.5,
  "colsample_bytree": 0.5,
  "lambda": 5.0,
  "alpha": 2.0,
  "tree_method": "hist",
  "nthread": 4,
  "seed": 42
}
```

**Assessment:**
- `max_depth=3, min_child_weight=60`: very shallow trees, strong regularization. Intentional for cross-sectional ranking with 103 tickers.
- `subsample=0.5, colsample_bytree=0.5`: aggressive stochastic sampling — halves both rows and features per tree. Combined with the already-small group sizes (77353 rows / 751 dates ≈ 103 rows/date), each tree sees ~51 rows and 13–14 features.
- `lambda=5.0, alpha=2.0`: heavy L2 + L1 regularization, 5× and 2× XGBoost defaults respectively.
- **Result: `best_iter=19`.** The model stops after 19 trees. With eta=0.02, total weight = 0.38 — barely any signal accumulated. This is a functionally 19-stump ensemble on a cross-sectional ranking problem.

**Train/OOS gap:** `training_train_ic = 0.1151`, `oos_mean_ic = 0.0350`. Ratio = 3.3×. For a model trained to predict cross-sectional rankings, a train IC this high with only max_depth=3 suggests either:
1. The training data still contains some leakage (unlikely given the fixes), or
2. The model memorizes cross-sectional ordering on the training partition because 19 shallow trees happen to fit the training data extremely well due to temporal autocorrelation in the features.

This gap is a **signal of distributional shift between in-sample and out-of-sample periods**, not just overfitting. The features that rank stocks well historically do not rank them as well prospectively at the same rate.

### 4.2 NGBoost Head

**File:** `training_panel/ngboost_head.py`; `ngboost-head.json` metadata

The NGBoost head trains a Normal(μ, σ) distributional regression on `residual_return_raw` (the raw residual forward return before Gaussianization), using the same 27 features as the panel-LTR.

**Production parameters** (`ngboost-head.json`):
```json
{
  "n_estimators": 400,
  "learning_rate": 0.01,
  "minibatch_frac": 1.0,
  "natural_gradient": true,
  "random_state": 17,
  "best_iter": 115
}
```

**Production metrics:**
- `train_mu_ic = 0.1286`
- **`val_mu_ic = 0.0214`** — the validation IC of 2.1% is extremely low
- `n_rows_train = 60149`, `n_rows_val = 15450` (80/20 split)

**Critical observations:**

1. **NGBoost trained 2026-04-27; panel-LTR trained 2026-04-28** (`trained_date` fields). They are not co-trained on the same data snapshot. The panel-LTR was retrained after the NGBoost head, meaning the two models saw slightly different panel versions. This is a data alignment gap. If the label distribution shifted between the two training runs (e.g., new tickers, new bars), the NGBoost σ predictions are calibrated to a slightly different universe.

2. **Serialization via pickle** (`ngboost-head.json::regressor_pickle_b64`). NGBoost is serialized as a base64-encoded pickle blob. Pickle is not forward-compatible across Python/scikit-learn/ngboost version changes. A version bump that breaks pickle deserialization will silently fail or raise a cryptic error at inference startup, not at training time.

3. **val_mu_ic = 0.021 driving Kelly sizing** — the NGBoost μ has barely above-noise predictive power on its own validation set. It is being used as the numerator of a Kelly bet-sizing formula. If μ is noisy, Kelly will produce high-variance position sizes (sometimes too large, sometimes zero) with no systematic edge.

4. **Feature order differs from panel-LTR.** Panel-LTR: `['adx', 'bbp', 'cci', ...]`. NGBoost: `['adx', 'williams_r', 'bbp', ...]`. This is not a bug — both models index by column name at inference via `self.feature_cols` — but it indicates the two models were trained in separate runs without column-order standardization.

### 4.3 Final Score Composition

**File:** `kernel/panel_pipeline/job_panel_scoring.py`

From the sub-agent read and grep results, the inference path:

1. Panel-LTR `predict()` → `panel_score` (raw XGBoost rank score)
2. Optional global calibration via isotonic regressor → calibrated `rank_score`
3. NGBoost `predict()` → `(μ, σ)` per ticker
4. **Score mode "additive"** (from `strategy_config.json`): `rank_score` + NGBoost contribution
5. Kelly sizing uses the NGBoost μ/σ directly: `kelly_target_pct(mu, sigma, max_pct=regime_max_pct, fractional=0.5, max_concentration=0.35)`

**Feature drift guard:** At inference, if > 5% of expected features are missing in the panel (`max_feature_drift_pct = 0.05`), NGBoost is skipped entirely (μ/σ set to None). Kelly then returns 0, and `SizeAndEmitTask` falls back to whatever the caller does with a zero kelly target — _code not found / unverifiable_ whether this hard-halts buys or falls back to a base position size.

---

## 5. Live Trading Infrastructure

### 5.1 Runner Lifecycle

**File:** `live/runner.py`

The runner is invoked via macOS launchd plist (`--once` mode). Before any trade decision:

1. **Pre-flight smoke test** (`runner.py`, lines ~287–318): validates model staleness, config fingerprint match (2026-04-28 addition), and state consistency. On `PreflightFailed` → ntfy high-priority alert + `SystemExit(2)`. This will cause launchd to log the exit and not retry unless configured to do so — _code not found / unverifiable_ whether launchd restarts on exit(2).

2. **Config fingerprint guard** (`pp_panel_training.py::SaveArtifactTask`): Each panel-ltr artifact is stamped with a SHA-256 fingerprint of the model-relevant config fields. At inference startup, `config_consistency.py` compares the running config's fingerprint against the artifact's. This directly addresses the 2026-04-28 NGBoost drift incident.

3. **ntfy notifications** (`runner.py`): Sent on every cycle including quiet cycles. On error, an exception notification is sent via ntfy. The ntfy call is wrapped in `try/except` and is non-fatal — trading continues even if alerting fails.

### 5.2 Drawdown Circuit

**File:** `kernel/pipeline/task_drawdown.py`

The circuit breaker is correctly implemented. `HWMUpdateTask.run()` advances the high-water mark (`ctx.hwm = max(ctx.hwm, ctx.portfolio_value)`) and is guarded against NaN/inf corruption (DC-1 fix):

```python
import math
if not math.isfinite(ctx.portfolio_value):
    log.warning("HWMUpdateTask: portfolio_value=%s is non-finite — skipping HWM update", ...)
    return
```

`DrawdownCircuitTask.run()` computes:

```python
drawdown = (ctx.hwm - ctx.portfolio_value) / ctx.hwm
if drawdown >= halt_pct:
    ctx.skip_buys = True
```

**Regime thresholds** (`strategy_config.json`):
- BULL_CALM: 35% — **extremely permissive** for a $10k portfolio ($3,500 loss before circuit fires)
- BULL_VOLATILE: 10%
- CHOPPY: 8%
- BEAR: 5%

**Critical design gap:** Setting `ctx.skip_buys = True` halts new buy orders but does **not** trigger any partial exit from existing positions. A portfolio in a 34% drawdown in BULL_CALM is still holding all positions, accumulating losses, with no forced de-risking. The `skip_buys` flag only stops new capital deployment. This is a **known design choice** (positions held, can sell), but it means maximum realized drawdown can significantly exceed the halt threshold on a portfolio that opened positions just before a sharp decline.

### 5.3 Position Sizing (Kelly)

**File:** `kernel/kelly.py::kelly_target_pct()`

Formula:
```
f* = μ / σ²
target = min(max_concentration=0.35, max_pct=regime_max_pct, fractional * f*)
```

**Production config** (`strategy_config.json` line 519): `fractional = 0.5`.

The `kelly.py` docstring explicitly recommends 0.25 ("quarter Kelly, widely used in live trading to absorb μ estimation error + log-utility variance drag"). The production value of 0.5 is double this, with the following implications:

- If NGBoost's σ is systematically underestimated (common in gradient-boosted distributional models trained on financial returns), half-Kelly will over-allocate relative to the true optimal.
- NGBoost's `val_mu_ic = 0.021` confirms the μ signal is weak. A weak μ combined with potentially underestimated σ → f* = μ/σ² is computed on noisy numerator and denominator simultaneously.
- The `max_concentration = 0.35` hard cap provides a safety ceiling, but 35% in a single position is very concentrated for a 103-ticker universe.

**NaN propagation guard** (`kelly.py`, Audit fix K-1): The fix for `mu=NaN` slipping past comparisons is present and correct:

```python
if not (math.isfinite(mu_f) and math.isfinite(sigma_f)):
    return 0.0
```

### 5.4 Crash Behaviour

**File:** `live/runner.py`

The outer loop wraps each trading cycle in a `try/except`. On exception, the error is logged and ntfy'd, then the loop continues to the next cycle. In `--once` mode (launchd), an unhandled exception causes the process to exit.

**Gap:** There is no broker reconnection logic visible in the code beyond the outer exception catch. If the Alpaca API returns a transient HTTP 5xx error mid-execution (after positions are queried but before orders are submitted), the exception propagates, the cycle is aborted, and no retry occurs until the next launchd invocation (next trading day). An intraday Alpaca API outage thus silently skips the entire decision cycle.

---

## 6. Bugs and Correctness Issues Found in Code

These are issues found by reading the code directly. Items marked with "(fixed YYYY-MM-DD)" were found in comments with corresponding fix documentation.

### 6.1 CONFIRMED BUGS — Fixed in Latest Code

| ID | File | Description | Status |
|---|---|---|---|
| BUG-CV-1 | `purged_cv.py` | `np.linspace` fold boundary drift → silent IS→OOS leakage | Fixed 2026-04-28 |
| BUG-CV-2 | `pp_panel_training.py` | No min_best_iter guard → undertrained model ships silently | Fixed 2026-04-28 (threshold=5) |
| BUG-CV-3 | `pp_panel_training.py` | Early-stop eval set (hardcoded 20%) misaligned with CPCV fold (16.7%) | Fixed 2026-04-28 |
| HIGH-1 | `purged_cv.py` | Purge in calendar days, not bars → ~3 bars of leakage per fold boundary | Fixed 2026-04-27 |
| LBL-CV-1 | `purged_cv.py` | NaN labels passed to model.fit() → silent zero-label training | Fixed 2026-04-25 |
| CAL-7 | `pp_panel_training.py` | Calibrator not refreshed after panel retrain → stale calibration | Fixed 2026-04-25 |
| CAL-7-PATH | `pp_panel_training.py` | Wrong config path for calibrator auto-refresh → silent skip | Fixed 2026-04-25 |
| DC-1 | `task_drawdown.py` | NaN portfolio_value corrupted HWM → circuit breaker permanently disabled | Fixed 2026-04-25 |
| K-1 | `kelly.py` | `mu=NaN` returned NaN from Kelly → NaN order sizes downstream | Fixed 2026-04-25 |
| X1+X2 | `ltr_model.py` | Early stopping params silently ignored (hardcoded `evals=None`) | Fixed 2026-04-26 |
| X5 | `ltr_model.py` | `predict()` used numpy positional indexing → wrong features on column-reordered panels | Fixed 2026-04-26 |
| X13 | `ltr_model.py` | Monotone constraint dict mapped positionally → wrong sign if feature order changed | Fixed 2026-04-26 |
| NGBoost drift | `job_panel_scoring.py` | Production NGBoost artifact had macro features; inference panel had none → zero-fill → σ distorted → all edge_sharpe < 0.10 → 0 buys | Fixed 2026-04-27 (added `max_feature_drift_pct` guard) |

### 6.2 OPEN ISSUES — Found in Code Reading

**ISSUE-1: min_best_iter = 5 is under-justified and too permissive**
- `pp_panel_training.py::FinalFitTask.run()` — `min_best_iter = int(cfg.get("min_best_iter", 5))`
- The threshold was lowered from 20 to 5 on 2026-04-28 evening based on a single diagnostic run showing healthy models converge at rounds 9–25.
- A model stopping at round 5 (total shrinkage = 0.10) is accepted. There is no second-order check (e.g., eval IC must be ≥ some floor) to confirm the model actually learned signal, not just stopped at a bad minimum.
- **Recommendation:** Add `eval_ic ≥ 0.015` as a co-condition with `best_iter ≥ 5`.

**ISSUE-2: NGBoost serialization via pickle**
- `ngboost-head.json::regressor_pickle_b64` — NGBoost is stored as a base64-encoded pickle blob
- Pickle format is tied to exact Python/scikit-learn/ngboost version. A conda environment update that bumps any of these will silently break NGBoost deserialization with a cryptic `AttributeError` or `ModuleNotFoundError` at inference startup.
- Panel-LTR uses a portable JSON format (`booster_raw_json` in `panel-ltr.json`). NGBoost does not.
- **Recommendation:** Use `ngboost`'s native serialization or joblib with explicit version pinning and a version check in `LoadNGBoostTask`.

**ISSUE-3: NGBoost and panel-LTR are not co-trained**
- `panel-ltr.json::trained_date = "2026-04-28"`, `ngboost-head.json::trained_date = "2026-04-27"`
- The two models were trained on different data snapshots. The panel-LTR retrain on 2026-04-28 added one additional day of data. If there were new tickers or label distribution changes between runs, the NGBoost μ/σ predictions are calibrated to a distribution slightly different from what the panel-LTR ranks.
- **Recommendation:** Enforce co-training. `NGBoostFitTask` should only run after `FinalFitTask` completes, and both should use the same `ctx.panel` snapshot. Add a `training_panel_hash` to both artifacts and validate they match at inference startup.

**ISSUE-4: BULL_CALM drawdown halt at 35% does not trigger exits**
- `task_drawdown.py::DrawdownCircuitTask.run()` — `ctx.skip_buys = True`
- A 35% drawdown in calm conditions stops new buys but does not reduce existing exposure. On a portfolio fully deployed ($10k, 8 positions at ~$1250 each), a 35% drawdown means ~$3,500 has been lost with all positions still held.
- **Recommendation:** Consider a `skip_buys + partial_exit_trigger` at, say, 20% BULL_CALM drawdown, reducing each position to 50% target size.

**ISSUE-5: Artifact directory has no lifecycle management**
- 83 JSON files in `artifacts/` — a mix of production, backups, ablations, and experiments — identified by filename only
- There is no lock file, atomic rename, or version registry. A script that writes to `artifacts/panel-ltr.json` mid-training will overwrite the production artifact with a partially-trained model.
- **Recommendation:** Training should write to a staging path (e.g., `artifacts/staging/panel-ltr.<timestamp>.json`) and only rename to `panel-ltr.json` after the `min_best_iter` guard passes and `SaveArtifactTask` completes successfully.

**ISSUE-6: Short interest lag unverified**
- `factors.py` includes `short_pct_float` in cross-sectional z-scoring
- Short interest data from most data providers (including yfinance) is published with a ~10-trading-day delay. If the code uses current short interest without a corresponding lag in the feature index, this feature has forward-looking bias equivalent to 10 bars.
- **The code reading did not find a lag correction for this feature.** This warrants direct investigation of the short interest data retrieval code.

**ISSUE-7: Single-ticker-per-date returns 0.0 label, not NaN**
- `labels.py::gaussianize_cross_section` — `if n == 1: return pd.Series(0.0)`
- On a date with exactly 1 ticker in the universe (e.g., a holiday where most tickers have no data), the surviving ticker gets label = 0 rather than NaN. It will be included as a training row with a "zero forward return" label, which is incorrect — zero is a valid return magnitude, not a "missing" indicator.
- **Impact is likely small** (such dates are rare in a 103-ticker universe) but could produce spurious training signal during data gaps.

**ISSUE-8: Kelly fractional = 0.5 is double the documented safety margin**
- `strategy_config.json` line 519: `"fractional": 0.5`
- `kelly.py` docstring: "0.25 = quarter Kelly, widely used in live trading to absorb μ estimation error"
- Production uses 0.5 (half-Kelly), twice the documented safe value, against a μ signal with `val_mu_ic = 0.021`
- The combination of noisy μ, potentially underestimated σ from NGBoost, and double the classical safety multiple creates systematic overconfidence in position sizing.

**ISSUE-9: No intraday minute-data availability guard**
- Features `m_morning_30min_drift_z`, `m_vwap_premium_z`, `m_intraday_realized_vol_z`, `m_overnight_gap_z`, `m_reversal_ratio_z` (5 of 27 features = 18.5%) are computed from minute bars
- The `{col}_is_missing` binary flag system will fill NaNs, but the model was trained with few such missing rows. If the minute data feed fails, 18.5% of features are missing, producing a highly out-of-distribution input to the panel-LTR.
- The `max_feature_drift_pct = 0.05` guard (5%) applies only to NGBoost, not to panel-LTR. A 5-feature-all-NaN input to panel-LTR would exceed 5% drift but no guard fires.
- **Recommendation:** Extend the feature drift guard to panel-LTR as well, and halt buys if drift exceeds the threshold.

**ISSUE-10: OOS IC documentation mismatch**
- `CLAUDE.md` references "CPCV OOS IC = +0.0418 (15-fold, 2026-04-27 重训复现)"
- `panel-ltr.json::oos_mean_ic = 0.0350` (production artifact trained 2026-04-28)
- This is a 17% discrepancy. Either the 2026-04-27 run produced a better model (now superseded), or the 2026-04-28 retrain with the BUG-CV-1/BUG-CV-3 fixes revealed that the true IC is lower than the leaky measurement. **The latter is the expected and healthy outcome** — fixing leakage correctly reduces measured IC. CLAUDE.md should be updated to reflect the current artifact's 0.0350.

---

## 7. What the Code Does Well

**7.1 Correct purged CV architecture.** The `PurgedKFold` and `CombinatorialPurgedCV` implementations in `purged_cv.py` are correct implementations of López de Prado AFML Chapter 7 and 12. The purge-in-bars fix, integer-division fold boundaries, and NaN-label filter are all methodologically correct.

**7.2 Gaussianized cross-sectional labels.** Rank → uniform → inverse-normal transformation per date eliminates outlier returns and makes labels approximately exchangeable across dates. This is the correct preprocessing for a cross-sectional ranker and directly improves CPCV IC stability.

**7.3 Residualized labels.** Labels are residualized against SPY beta and sector beta before Gaussianization. This removes market and sector factor exposures from the training target, focusing model signal on stock-specific alpha. The purged rolling beta computation is correct.

**7.4 Feature diversity.** The 27 features span four distinct alpha source categories: technical oscillators, cross-sectional momentum/factors, intraday microstructure, and fundamentals. This reduces the risk of signal crowding from any single factor family.

**7.5 Config fingerprint protection.** The 2026-04-28 addition of SHA-256 fingerprinting of model-relevant config fields into every artifact (`pp_panel_training.py::SaveArtifactTask`) directly addresses the recurring config/model drift incidents. This is a correct invariant-level fix, not a patch.

**7.6 Python-level early stopping.** The chunk-based early stopping in `ltr_model.py` works around a genuine XGBoost 3.x limitation (integer-label requirement for NDCG metric) cleanly, without degrading training quality. The `min_delta_ic = 1e-3` threshold is sensibly tuned to avoid spurious best-updates from numerical noise.

**7.7 NaN/inf guards throughout the hot path.** HWM, Kelly, and feature drift all have explicit `math.isfinite()` guards with logged warnings (DC-1, K-1). This defensive programming prevents silent NaN propagation into order sizes.

**7.8 Consistent score path.** The docstring in `kelly.py` explicitly states: "One formula, one place. All three decision layers (SizeAndEmit for new buys, TopUpHeld for add-to-existing, Rotation for swap) read the SAME `kelly_target_pct` field." This is a meaningful correctness invariant — there is no silent score-path divergence between buy and rotation decisions.

**7.9 Concurrency weighting.** The panel weighting scheme (`weight_concurrency = 1 / mean(live[start_idx:start_idx+lookahead_days])`) correctly de-weights dates where many labels overlap, giving equal effective weight to each time period rather than each (ticker, date) row.

---

## 8. Prioritized Recommendations

Ordered by: (correctness impact × urgency). Items 1–3 are blocking for any further IC measurement. Items 4–6 are blocking for production safety. Items 7–9 are medium-priority improvements.

### Priority 1 — Validate all IC numbers under fixed CV (blocking)

All CPCV IC measurements prior to 2026-04-28 were produced with three compounding bugs (linspace drift, calendar-day purge, eval-set misalignment). The current `oos_mean_ic = 0.0350` is the first measurement under correct CV. Before any A/B experiment proceeds, confirm this number with:
- A/A test (same panel, randomly resplit dates): IC should not change by > 1σ
- Shuffled-label test: IC should converge to ~0

Without these, 0.0350 is an unvalidated single-run measurement.

### Priority 2 — Co-train panel-LTR and NGBoost in same pipeline run (data alignment)

`panel-ltr.json::trained_date = "2026-04-28"`, `ngboost-head.json::trained_date = "2026-04-27"`. The two models use different data snapshots. `NGBoostFitTask` should run inside `PanelModelJob` after `FinalFitTask`, using `ctx.panel` (same snapshot). Add `training_panel_hash` to both artifacts; validate at inference startup.

### Priority 3 — Audit short interest lag correction (look-ahead risk)

`factors.py` computes `short_pct_float_z` without a confirmed lag. Short interest is published ~10 trading days late. If unlagged, this feature has forward-looking bias equivalent to approximately 2× the lookahead horizon. Confirm or add a `shift(10)` before cross-sectional z-scoring.

### Priority 4 — Reduce Kelly fractional from 0.5 to 0.25 (risk management)

`strategy_config.json` line 519: `"fractional": 0.5`. Against a `val_mu_ic = 0.021` signal and potentially underestimated σ from NGBoost, half-Kelly overallocates. Reduce to 0.25 (the library default and docstring recommendation) until NGBoost achieves `val_mu_ic ≥ 0.04` on a 6-month holdout.

### Priority 5 — Extend feature drift guard to panel-LTR (operational safety)

Currently `max_feature_drift_pct = 0.05` only gates NGBoost. The panel-LTR has no equivalent guard. A failure in the minute data pipeline renders 5 of 27 features (18.5%) as NaN-then-imputed, producing OOD inputs to the ranking model. Add a parallel drift check before `PanelLTRModel.predict()` and halt buys if drift exceeds the same 5% threshold.

### Priority 6 — Atomic artifact promotion (operational safety)

Training writes directly to `artifacts/panel-ltr.json`. A crash mid-write or a concurrent process produces a corrupted production artifact. Implement: write to `artifacts/staging/panel-ltr.<timestamp>.json`, validate (load + predict on 1 row), then `os.replace()` (atomic on POSIX) into `artifacts/panel-ltr.json`. This costs < 10 lines of code and eliminates a class of production corruption incidents.

### Priority 7 — Raise min_best_iter or add eval_ic floor (training quality)

Current guard: `best_iter ≥ 5`. A model stopping at round 5 (total shrinkage = 0.10) is accepted. Add a co-condition: `eval_ic ≥ 0.015` (roughly half the current OOS mean IC). This catches models that pass the iteration count but converged to noise.

### Priority 8 — Replace NGBoost pickle with portable serialization (operational resilience)

`ngboost-head.json::regressor_pickle_b64` is a version-pinned pickle. Migrate to joblib with an explicit version manifest (`{"ngboost": "0.4.x", "sklearn": "1.x.x", "python": "3.10.x"}`) checked at load time, with a clear error message on mismatch. Medium urgency — will become blocking on the first conda update that touches ngboost/sklearn.

### Priority 9 — Fix the single-ticker-per-date label edge case (label quality)

`labels.py::gaussianize_cross_section` returns `0.0` for n=1 dates. Change to return `np.nan` and ensure the NaN-label filter in `cross_validated_ic` (`purged_cv.py`, the `valid_label` check) drops these rows from training.

---

## Appendix: Production Artifact Summary

| Artifact | `trained_date` | Key metric | Note |
|---|---|---|---|
| `panel-ltr.json` | 2026-04-28 | OOS CPCV IC = 0.0350 ± 0.030 | 27 features, rank:pairwise, best_iter=19 |
| `ngboost-head.json` | 2026-04-27 | val_mu_ic = 0.021 | Pickle serialization; trained day before panel-LTR |
| `panel-rank-calibration.json` | (not checked) | — | Auto-refreshed by `RefreshPanelCalibratorTask` after retrain |
| `spy-gmm-regime.json` | (not checked) | — | Regime classifier; details unverified in this review |

**Panel shape:** 77,353 rows, 103 tickers, 751 dates (~3 years)
**Watchlist size:** 106 tickers in `strategy_config.json` (103 in training panel — 3 may have been excluded for data quality)
**Live account:** Alpaca ~$10k, holdings: PLTR / TSM / CAT / AMZN / GOOG / XLU

---

_Assessment prepared 2026-04-28/29. All code citations are to commit state as of 2026-04-28. Source of truth: the source files themselves, not documentation._
