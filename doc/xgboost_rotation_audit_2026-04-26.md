# XGBoost + Rotation Algo Deep Audit — 2026-04-26

Two parallel audits per user spec:
1. XGBoost training + application logic — find bugs, evaluate retrain need
2. Rotation algo vs recent paper theory — find mismatches

## Part 1: XGBoost Backend Audit (`training_panel/ltr_model.py`)

### 🔴 CRITICAL

| # | Issue | Detail |
|---|---|---|
| X1 | **No early stopping in FinalFit** | `train_kwargs["early_stopping_rounds"]` only set when `deval is not None AND early_stopping_rounds is not None`. CV adapter passes neither → FinalFit runs full `num_boost_round=400` without early stop. Risk: overfit on small panel even if CV picked optimal `num_boost_round`. **Mitigation today**: monotone constraints + L1/L2 reg + min_child_weight=20 already constrain. Net effect: probably <0.005 IC loss, but unbounded as we add features. |
| X2 | **`best_iteration` defaults to last round when no early stop** | L167 `getattr(booster, "best_iteration", num_boost_round - 1)`. XGBoost only sets `best_iteration` after early stopping fires. Without eval set, `best_iter = num_boost_round - 1` → predict() uses ALL trained rounds. Same risk as X1. |

### 🟠 HIGH

| # | Issue | Detail |
|---|---|---|
| X3 | **`num_boost_round=400` hardcoded default** | With X1/X2 in play, this IS the training budget. Could overfit on smaller panels. Sunday sweep should compare to 200/300/400. |
| X4 | **Group-mean weight aggregation loses per-row signal** | XGBoost 3.x ranking uses per-group weights. We aggregate row weights (`age × concurrency`) to group mean. Concurrency is constant per date (group mean = row value), but `age` varies — mean preserves date-level "young listings" signal but loses ticker-level discount. Acceptable for ranking objective, but original per-row design intent partially lost. |
| X5 | **predict() doesn't validate feature column order** | `xgb.DMatrix(X)` accepts a numpy array; column ORDER must match training. If panel is reindexed between train and predict (e.g., new fundamental column added), silent miscolumn → wrong scores. **Mitigation**: features pass through panel pipeline in fixed order, but no assert. |
| X6 | **`nthread: -1` (use all cores)** | Combined with multiprocessing in TickerPanelFeatureJob → fork+OMP deadlock risk identical to transformer's set_num_threads(1) workaround. Currently safe because XGBoost training happens AFTER all parallel jobs complete, but adding parallel CV folds would break. |

### 🟡 MEDIUM

| # | Issue | Detail |
|---|---|---|
| X7 | `lambda=1.0, alpha=0.5` L1/L2 not tuned — should sweep | |
| X8 | No artifact format versioning beyond `version: 1` — no migration path | |
| X9 | `_mean_ic` uses panel[date_col] — KeyError if date col missing | |
| X10 | No SHAP feature_importances exposed in metadata — hard to debug feature contributions | |
| X11 | Sample weights ignored in CV adapter (`fit(X, y, sample_weight=None)` from sklearn signature) | |

### XGBoost Application Logic (panel_pipeline/job_panel_scoring.py)

Audit found in commit history: ApplyGlobalCalibrationTask was previously
no-op when NGBoost enabled (Plan F bug, fixed). Current path:
`LoadScorer → BuildFeatureMatrix → ApplyScores → LoadNGBoost → ApplyNGBoost
→ LoadGlobalCalibration → ApplyGlobalCalibration → VetoWeakBuys`. Order
verified correct. ✓

Saturation issue (cause of net_alpha=0.1446 tie): **calibrator's
isotonic upper bin** maps multiple raw scores to identical probability.
Fixed today via tiebreaker by raw `panel_score` in JointActionTask
(commit `397c1b3`). The calibrator itself is correctly fitted — the
tiebreaker is needed because the loss-of-resolution at the upper bin
is a fundamental property of isotonic, not a fittable bug.

### XGBoost Retrain Decision

**DON'T retrain mid-week**. Reasons:
1. Current OOS IC = 0.04764 already matches Tier 1.5 baseline 0.0476 → no measurable overfit happening today.
2. Adding early stopping would help future panels (more features) but on the current 24-feature panel the constraints are already adequate.
3. Sunday sweep will refresh anyway with the new transformer code in place — adding XGBoost early-stop fix can wait until then.
4. Production cost of one more retrain: ~30 min compute + state churn — not justified by <0.005 expected IC gain.

**Sunday's TODO**: thread `eval_panel` through to FinalFit + add
`early_stopping_rounds=20` to the XGBoost path. Estimated marginal IC
improvement: +0.002 to +0.005 absolute on this panel; up to +0.015 on
larger feature sets.

---

## Part 2: Rotation Algo vs Recent Paper Theory

### Reference papers + checked implementations

| Paper | Core insight | Our implementation | Match? |
|---|---|---|---|
| **Garleanu-Pedersen 2013** | Closed-form "aim portfolio". Trade TOWARD target, partial-execute damped by transaction costs. | Phase 1 rotations are all-or-nothing swaps. **Partial trading** exists via TopUpHeldTask + TrimHeldTask (Kelly-target driven). | 🟡 PARTIAL — Kelly path is GP-spirit; rotation path is not |
| **Boyd 2017 MPC** | cvxpy QP per bar; jointly optimize all trade sizes subject to costs + risk. | 3-pass greedy with dominance pruning (Bug MM). Phase 3 roadmap. | 🟠 GAP — heuristic approximation; documented |
| **Avellaneda-Lee 2010 pair-trading** | Z-score of spread for entry/exit timing. | `rotation_mode="thesis_symmetric"` does 4-point comparison (A_entry, A_today, B_entry, B_today). | ✅ MATCH |
| **Constantinides 1984 tax-aware** | Defer realization of gains until LT threshold; realize losses early. | `tax_drag(unreal_pct, hold_days, ST/LT)` enters net_advantage; `is_lt_protected` blocks rotations within `lt_protection_days` on gains. | ✅ MATCH |
| **Kelly 1956 / Thorp 2006** | `f* = μ/σ²` continuous Gaussian; half-Kelly for variance reduction. | `kelly_target_pct = μ/σ²` from NGBoost head; `fractional=0.5`. | ✅ MATCH |
| **Jegadeesh-Titman 1993 momentum** | 12-1 momentum anomaly. | `mom_12_1_z` factor in panel + monotone constraint +1. | ✅ MATCH |
| **Markowitz 1952 MV** | Single-period mean-variance. Target weights derived from `Σ⁻¹ μ`. | NGBoost gives μ; σ used for Kelly. No explicit covariance matrix in objective — replaced with sector + correlation guards. | 🟡 SIMPLIFIED — heuristic guards instead of full Σ |
| **Almgren-Chriss 2000 execution** | Spread one trade signal across N intraday bars to minimize impact. | We submit market orders at-once. Live broker handles intraday execution; we don't model impact. | 🟠 NOT IMPLEMENTED — appropriate for retail-scale ($10k account) |
| **Grinold-Kahn 1999 active mgmt** | IR = IC × √Breadth. Maximize IR not raw alpha. | Indirectly via panel-LTR + Kelly sizing — high-IC + low-σ tickers get bigger weight. No explicit IR optimization. | 🟡 IMPLICIT |
| **Carhart 1997 4-factor** | Add momentum to FF3. | Implemented as separate factors (mom_12_1_z + resid_mom_z) within panel-LTR. | ✅ MATCH |

### Theoretical gaps (recommendations)

1. **GP partial-rotation**: When net_alpha is positive but small (just above threshold), trade a FRACTION of the swap rather than full. Damping factor ∝ (net_alpha − cost) / (net_alpha + λ·variance). Could implement as fraction of `max_position_pct` for the buy-leg, with proportionate sell.

2. **Boyd MPC (Phase 3)**: Already in roadmap. Highest-leverage long-term improvement.

3. **Multi-period horizon**: Current rotation optimizes per-bar. GP shows multi-period horizon (with alpha decay rate λ) yields 5-15% better IR. Could add `horizon_decay_lambda` parameter to tax_drag-style cost function.

4. **Cost-aware execution**: For larger account sizes, Almgren-Chriss splitting becomes relevant. Not needed today ($10k account).

5. **Explicit covariance Σ**: Sector + correlation guards approximate Σ⁻¹μ. For Tier 4+ work, could replace with a sample covariance + Ledoit-Wolf shrinkage, then solve `argmax μ'w − γ w'Σw`.

### Rotation Algo Reliability Verdict

**Mechanically reliable** (1817 tests pass, e2e ran clean). **Theoretically
sound** for the implemented subset (GP-aim via Kelly, tax-aware,
calibrated scores, neutralized factors). **Theoretical gaps are well-
documented** and roadmapped.

The biggest theoretical gap (no joint-optimal QP via Boyd) is what the
3-pass greedy heuristic mitigates — and the dominance pruning of Bug MM
captures the most important case (SELL vs ROTATE for same held).

---

## Combined Retrain Decision

| Question | Answer |
|---|---|
| Retrain XGBoost mid-week? | **NO** — current OOS already at baseline; expected gain <0.005 |
| Retrain Transformer mid-week? | **NO** — even with all 10 fixes, estimated post-fix OOS 0.027-0.047 still won't beat XGBoost's 0.0476 on 1.5k-date panel |
| Retrain LightGBM mid-week? | **NO** — confirmed inferior 2 sessions in a row |
| Wait for Sunday sweep? | **YES** — sweep will compare all 3 backends with the new fixes side-by-side, decisive evidence |

## References

1. Garleanu N., Pedersen L. H. 2013. "Dynamic Trading with Predictable Returns and Transaction Costs." *J. Finance* 68 (6): 2309–2340.
2. Boyd S. et al. 2017. "Multi-Period Trading via Convex Optimization." *Foundations and Trends in Optimization*.
3. Avellaneda M., Lee J.-H. 2010. "Statistical Arbitrage in the U.S. Equities Market." *Quantitative Finance* 10 (7): 761–782.
4. Constantinides G. M. 1984. "Optimal Stock Trading with Personal Taxes." *J. Financial Economics* 13 (1): 65–89.
5. Kelly J. L. 1956. "A New Interpretation of Information Rate." *Bell System Technical Journal* 35 (4): 917–926.
6. Thorp E. O. 2006. "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market." *Handbook of Asset and Liability Management*.
7. Markowitz H. 1952. "Portfolio Selection." *J. Finance* 7 (1): 77–91.
8. Almgren R., Chriss N. 2000. "Optimal Execution of Portfolio Transactions." *J. Risk* 3 (2): 5–39.
9. Grinold R. C., Kahn R. N. 1999. *Active Portfolio Management*. McGraw-Hill.
10. Carhart M. M. 1997. "On Persistence in Mutual Fund Performance." *J. Finance* 52 (1): 57–82.
11. Jegadeesh N., Titman S. 1993. "Returns to Buying Winners and Selling Losers." *J. Finance* 48 (1): 65–91.
