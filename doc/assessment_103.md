# RenQuant-103: Fact-Based Technical Assessment

**Date:** 2026-04-16  
**Scope:** Architecture, model quality, statistical validity, live execution, regime detection.

---

## 1. What Works Well

**Architecture quality is high.** JSON artifacts for all model types (not pickle) — LEAN can reload them without any Python dependency chain. The stage separation (notebook → export → LEAN → live runner) is clean and failure modes are isolated. The `common/` library is properly scoped. The `feature.shift(1)` lookahead prevention is correctly applied in all three model types (`classification.py:68`, `qlearning.py:94`, `xgboost_model.py:113`).

**Test discipline is real.** 430 tests with paired NB/LEAN alignment tests (`test_policy_alignment.py`, 17 policy classes × 13 tests each). The CLAUDE.md rule requiring paired tests before any commit is the right enforcement mechanism.

**Post-Copilot calibration fixes are correct.** `StandardScaler` is now properly applied before the Platt logistic regression fit (`scoring.py:187`) and saved alongside the coefficients (`platt_scale_mean/std`) for correct inference-time transform. The method-selection logic (isotonic for n≥300, Platt for 120≤n<300, constant for n<120) is sound in principle. The daily recalibration pipeline is now wired into `daily_103.sh`.

**Risk architecture is layered and thoughtful.** Five exit conditions with strict priority ordering (trailing stop → cumulative stop → single-day gate → max hold → model streak), sector guard, correlation guard, earnings filter, wash-sale, and drawdown circuit breaker. The regime-conditional stop widths (15% BULL_CALM vs 5% CHOPPY/BEAR) make economic sense.

**Tax modeling is honest.** 50% short-term / 32% long-term rates force the strategy to work harder in net terms.

---

## 2. Critical Issues

### 2.1 OOS Sharpe Estimates Are Statistically Meaningless

Every model is evaluated on a fixed OOS window from 2024-01-01 to the training date. Current live models have `cal_n: 185` across all 24 symbols.

With n=185 daily OOS observations, the standard error of the Sharpe ratio estimate is:

```
SE(SR_annual) = sqrt((1 + 0.5 × SR²) / n) × sqrt(252) ≈ 1.17  for any SR in [0.8, 2.0]
```

**95% confidence intervals:**

| Reported SR | 95% CI |
|-------------|--------|
| 0.8 (floor) | [−1.49, +3.09] |
| 1.5 | [−0.79, +3.79] |
| 2.0 (PLTR, GLD, AMZN) | [−0.30, +4.30] |

The floor at 0.8 is indistinguishable from zero. PLTR reporting 2.00 and MSFT reporting 0.82 are statistically the same observation at this noise level.

**Tournament selection makes this worse.** The notebook trains 4 model types per ticker and exports the best OOS Sharpe. Under the null hypothesis of zero true alpha, the expected Sharpe of the *best* of 4 independent estimates on n=185 observations:

```
E[max of 4 zero-alpha models, n=185] = 1.21 annual SR units  (measured via 50,000-run simulation)
```

The 0.8 floor is passed by pure noise with high probability. All reported Sharpe numbers are inflated by approximately 1.2 SR units from tournament selection alone.

**Floor bypass:** XLV (reported 0.596) and UNH (reported 0.683) are below floor but present in the active model set. The real issue is not the notebook's chart-only "bootstrap" section — LEAN and the live runner load models directly from per-symbol metadata files if they exist on disk, with no sharpe floor check. Stale/below-floor model directories were not purged during notebook export, allowing below-floor artifacts to remain loadable. Fixed: both `_load_all_models()` (LEAN) and `_load_strategy_multi()` (runner) now check `metadata["sharpe"]` against `config["sharpe_floor"]` before loading.

---

### 2.2 CUSUM Detects Local Drift, Not Structural Breaks

The CUSUM implementation (`main.py:578-593`) normalizes by the **mean and standard deviation of the same 20-bar window being tested**:

```python
mu    = returns.mean()      # from the test window itself
sigma = returns.std(ddof=1) # from the test window itself
z     = (r - mu) / sigma
```

This is not a structural break test. A proper CUSUM uses reference parameters estimated from a stable in-control period. Normalizing by the test window's own statistics detects any path that deviates from the endpoint average — which fires for any non-flat trajectory.

**Measured false positive rate on a pure random walk (iid N(0,0.01), 20 bars, threshold=3.0, drift=0.5): 8.1%**

With daily trading over 2+ years (~500 bars), this fires approximately 500/20 × 8.1% ≈ 20 spurious transition signals per year. Each triggers a 3-bar uncertainty window blocking all new buys: ~60 buy-blocked days per year from noise alone.

---

### 2.3 Hurst with 63-Day Window Is Insufficient

R/S Hurst exponent requires at minimum 200 observations for a reliable estimate. With 63 bars and max_lag=40, the effective lag range is [2, 31]. The slope of log(R/S) vs log(lag) has high variance at this length.

The thresholds H>0.55 (MOMENTUM) and H<0.45 (REVERSION) are within the expected noise band for a true H=0.5 process at this sample size. BULL_CALM ↔ CHOPPY transitions are driven by noisy estimates, not genuine regime identification.

---

### 2.4 GMM Has 3 Components but Strategy Has 4 Regimes

The GMM artifact is trained with 3 components (`regime.py:177`). `_auto_label` assigns: BULL_CALM, BULL_VOLATILE, BEAR. GMM can never output CHOPPY.

CHOPPY is only reachable via the Hurst REVERSION path (`main.py:499`). This creates an **impossible constraint**: CHOPPY has `max_hold_days: 10` but `min_hold_days: 20`. The model-sell exit (gated at 20 days minimum) can never fire before max hold forces exit at 10 days. **In CHOPPY, the model sell path is structurally dead** — only stop-loss and max hold matter.

---

### 2.5 Live Runner Permanently Assumes BULL_CALM

`runner.py:419-425` hardcodes all regime parameters to `BULL_CALM`:

```python
bull_calm_rp   = regime_params.get("BULL_CALM", {})
stop_loss_pct  = float(bull_calm_rp.get("stop_loss_pct", ...))
```

No live GMM inference. The regime detection that drives all adaptive behavior in LEAN is absent in live trading.

**Impact in current market (April 2026, tariff turbulence):**
- Live trading uses 15% cumulative stop-loss (BULL_CALM) instead of 5% (BEAR/CHOPPY)
- Trailing stop is LEAN-only (`[EXIT 1] Trailing stop — not tracked in runner`, `runner.py:573`)
- Cash reserve is 0% regardless of actual regime
- This is the most consequential gap between backtest and live execution

---

### 2.6 Blend Weights Are [1.0, 0.0] — RS Has Been Silenced

`strategy_config.json` shows `"blend_weights": [1.0, 0.0]` as of 2026-04-16. The recalibrate_scores.py logistic regression assigned zero weight to RS across all 24 symbols.

Two interpretations: (a) RS genuinely adds no predictive value — plausible if both signals are correlated; (b) the regression on n×185 pooled rows with L2 regularization shrank the RS coefficient to zero due to insufficient signal, not genuine absence of predictive power.

If (b), the system has permanently discarded a potentially useful signal due to insufficient OOS data. The RS infrastructure is dead weight at current weights.

---

### 2.7 Wash-Sale Guard Bypassed in Live State

`live_state.json`:
- `entry_dates.AMZN: "2026-04-16"`
- `last_sell_dates.AMZN: "2026-04-15"`

AMZN was sold April 15 and re-entered April 16 — 1-day gap. The 30-day wash-sale guard (`runner.py:744-751`) checks `days_since_sell < 30` and should skip. Yet the position was entered.

Most likely mechanism: broker reconciliation (`runner.py:541-542`) only re-populates `last_sell_dates` for sells **today** — so the April 15 sell is not re-seeded on April 16, and if state was missing or `last_sell_dates` empty at run time, no guard fires. The state is inconsistent and should be investigated before treating the position as clean.

---

## 3. Moderate Concerns

**XGBoost buy threshold is effectively just signal direction.** `xgboost_model.py:192`: `score > buy_threshold - 0.5`. With `buy_threshold=0.55`, this becomes `score > 0.05`. This passes any row where P(buy) > P(sell) by even a tiny margin. The threshold adds no meaningful filtering.

**Calibration is fitted on model-selection data.** `recalibrate_scores.py` builds calibration on the full available history including the OOS window (2024-01-01+) used to select the winning model type. The calibrated `rank_score` is therefore over-optimistic on the same data that determined which model was exported.

**The "OOS" concept erodes daily.** With a fixed OOS cutoff of 2024-01-01 and daily retraining on all available data, every retrain expands training to include more of the period originally held out. The frozen Sharpe numbers in metadata become increasingly stale as OOS benchmarks.

**Tech-heavy sector map creates structural imbalance.** 12/24 tickers are "tech" with a max 3-position sector cap — at most 37.5% of the portfolio can be in tech. This structurally limits allocation to the apparently strongest signals (AMZN:2.05, PLTR:2.00 are both tech). Whether this is a feature or a bug depends on how much you trust those Sharpe numbers (see §2.1).

**GMM `_auto_label` may mislabel volatile regimes.** Assigning BULL_VOLATILE as "middle vol" is a statistical heuristic, not an economic one. In a crash/recovery environment (rapidly rising then falling vol), the middle cluster may correspond to a bear rally, not a volatile bull market.

---

## 4. Issue Summary Table

| # | Issue | Location | Severity |
|---|-------|----------|----------|
| 1 | OOS Sharpe SE=1.17; floor indistinguishable from noise | notebook, metadata | Critical |
| 2 | Tournament selection inflates Sharpe by +1.2 SR units | notebook training loop | Critical |
| 3 | CUSUM self-referencing mu/sigma; 8.1% FP on random walk | main.py:578, regime.py:83 | Critical |
| 4 | Hurst 63-day window too short for reliable R/S | main.py:545, regime.py:28 | High |
| 5 | GMM 3 components vs 4 regimes; CHOPPY model-sell is dead | main.py:_update_regime, regime.py:177 | High |
| 6 | Live runner hardcoded BULL_CALM; no trailing stop live | runner.py:419-425, 573 | Critical |
| 7 | Blend weights [1.0, 0.0]; RS signal silenced | strategy_config.json, recalibrate_scores.py | Moderate |
| 8 | AMZN wash-sale guard bypassed (1-day re-entry) | runner.py:744, live_state.json | Moderate |
| 9 | XGBoost buy threshold effectively direction-only | xgboost_model.py:192 | Low |
| 10 | Calibration fitted on model-selection data | recalibrate_scores.py | Low |
| 11 | XLV/UNH below Sharpe floor but active: stale artifact dirs not purged; LEAN/runner had no floor check | _load_all_models, _load_strategy_multi | Moderate |
| 12 | Fixed OOS cutoff erodes; Sharpes stale after daily retrain | notebook, metadata | Moderate |
| 13 | LEAN GMM inference ignored saved StandardScaler — regime probabilities mathematically misaligned | main.py:_gmm_predict | Critical |
| 14 | Live runner cash_avail never decremented in buy loop — multiple buys each sized off same cash snapshot | runner.py:buy loop | Critical |

---

## 5. The Core Structural Problem

The strategy has substantial engineering depth — regime layers, calibration, correlation guards, tax modeling, paired tests. But the statistical validation layer is too thin relative to the modeling complexity.

**185 OOS observations cannot distinguish alpha from noise at any reasonable confidence level, regardless of how many filters and layers sit on top.**

The tournament selection inflation alone (~1.2 annual SR) means the 0.8 Sharpe floor is passed by noise for the majority of symbols in the majority of retraining runs. Every symbol on the watchlist has approximately a 50%+ probability of passing the floor with zero true alpha, given the model zoo, the short OOS window, and the fixed cutoff that becomes stale.

**This is correctable without rebuilding:**
1. Require longer OOS windows (minimum 500 bars ≈ 2 years per symbol)
2. Use combinatorial purged cross-validation (CPCV) to account for overlapping labels and tournament selection
3. Validate regime detection signals independently before coupling them to position sizing
4. Close the live runner / LEAN regime gap (most urgent)
5. Fix the CUSUM to use a reference window, not the test window, for mu/sigma estimation

The live execution gap (regime always = BULL_CALM, no trailing stop) is the highest-priority fix. Everything else is risk already carried in exchange for the architectural investment already made.
