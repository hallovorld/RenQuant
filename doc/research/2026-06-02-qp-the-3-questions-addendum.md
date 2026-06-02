# QP — the three questions, answered

**Date**: 2026-06-02
**Status**: Addendum to [`2026-06-02-qp-architecture-review-and-alternatives.md`](./2026-06-02-qp-architecture-review-and-alternatives.md)
**Author**: Claude
**Prompted by**: User question — *"is QP really the ultimate solution? what's its functional? why we introduced it?"*

This addendum answers the three questions directly from git history + memory, separate from the broader alternatives discussion in the parent memo. Read this first if you want the *why* before the *what's better*.

---

## Q1 — Why did we introduce QP?

Not "to optimize portfolios". The narrower story.

### The trigger event — April 2026 audit

**2026-04-27** commit `20155c4` (`docs(deep audit): T2-4 + macro v2 + XGB+macro = 21 actionable bugs`). The audit found:

- T2-4 (rotation logic): **5 HIGH + 2 MED + 3 LOW = 10 bugs**
- Macro v2: 2 HIGH + 4 MED + 1 LOW = 7 bugs
- XGB+Macro: 3 HIGH + 1 MED = 4 bugs

The legacy `kernel/rotation.py` was a 609-line file of hand-rolled "swap when net_advantage ≥ threshold" logic. Per-name decisions, no portfolio-level optimization, no risk-aware sizing. Every bug was a "logic that should have been there but wasn't" — wash-sale not respected, sector cap not enforced, turnover not budgeted, tax-drag misapplied.

The user's verbatim reaction (recorded in `memory/feedback_industry_leading_quality.md`):

> "Industry-leading quant systems don't iterate this way. They specify the math before coding (Almgren-Chriss, Garleanu-Pedersen, Brown-Smith, Ledoit-Wolf — these are reference implementations). Build the components with industry-grade tests. Compose into pipeline with clear interfaces. Validate ONCE end-to-end after the refactor is clean."

### The decision

**2026-04-27** commit `d284d52` (`feat(T2-4 phase A): kernel/rotation_convex.py — Boyd-style mean-variance QP`):

> "Per boyd-rotation-design.md Phase A. ConvexRotationSolver class:
> Objective: maximize μᵀΔw − γ·Δwᵀ Σ Δw − c·‖Δw‖₁
> Constraints: long-only, leverage cap, turnover cap, optional sector caps
> Two solver paths: cvxpy + OSQP (preferred), scipy SLSQP (fallback)"

The reference was **cvxportfolio (Boyd / Stanford)** — the canonical Boyd-Vandenberghe textbook implementation. Per CLAUDE.md §7.10 ("READ, not name-dropped"), this was the academically-sourced choice.

### The deeper why

The 2026-05-14 follow-up memory (`feedback_validate_with_mature_lib`) records the user's verbatim pivot point:

> "Before going too far! I still want to use mature 3rd party solution rather than building from scratch!"

So the chronology is:
1. Hand-rolled rotation logic → too many bugs
2. Audit found 21 bugs across 3 components
3. User: "I want mature 3rd party, industry-grade"
4. cvxportfolio (Boyd) selected as the reference
5. QP solver implemented matching the textbook
6. May 2026: progressively absorbed all advanced terms (CVaR, robust-μ, sqrt-impact, tax-aware, sector cap, correlation cap)
7. Today: 32 parameters, 5 objective terms, 5+3 constraint families, ~1100 LOC, 456 tests

**QP wasn't introduced because it was the best portfolio optimization in some abstract sense. It was introduced because (a) the predecessor had too many ad-hoc bugs and (b) the user explicitly mandated industrial-grade reference implementations over hand-rolled math.**

---

## Q2 — What is QP's functional role?

QP isn't doing "the portfolio decision". It's doing **one specific stage** in a multi-stage decision pipeline.

### Where QP sits in the InferencePipeline

Tracing `kernel/pipeline/pp_inference.py` end to end:

```
Phase 1: Regime detection (kernel/regime.py)
   ↓ outputs: regime label + confidence + drawdown state

Phase 2a: Sell-only logic
  - Drawdown halt (if portfolio DD > threshold → exit all)
  - Stop-loss / take-profit / σ-aware stop (per holding)
  - Wash-sale / earnings blackout (per holding)
   ↓ outputs: forced sells

Phase 2b: Buy scan (candidate generation)
  - Universe filter (watchlist + earnings + missing-sector + …)
  - Pre-trade risk gates (realized vol cap, sector quota, etc.)
   ↓ outputs: ~80-150 buy candidates

Phase 3: PanelScoring / Ranking (ML model)
  - LoadScorerTask → AssembleInferenceMatrix → ApplyScores
  - Calibrator: panel_score → expected_return (μ) + sigma (σ)
  - VetoWeakBuysTask: rank-score floor
  - LoadGlobalCalibrationTask → ApplyGlobalCalibrationTask
  - ApplyKellySizingTask → per-name Kelly fraction
   ↓ outputs: ranked candidates with calibrated (μ, σ)

Phase 4 = QP: Joint portfolio optimization
  - PrepareQPVectorsTask: build w_current, μ, σ, prices for {candidates ∪ holdings}
  - BuildSigmaTask: assemble Σ from corr_matrix + per-name σ + Ledoit-Wolf shrinkage
  - ComputeWashSaleMaskTask: wash-sale set
  - ComputeQPConstraintsTask: w_upper (regime × confidence × dw_max × cash_reserve)
  - ApplyExposureScalingTask: w_upper ×= vol_target × dd_scale
  - ApplyConvictionCapTask: w_upper ×= conviction_multiplier
  - ApplySectorMetadataGuardTask + ApplyExitOnlyTopupGuardTask
  - BuildSectorConstraintMatrixTask: hard sector linear constraint
  - SolveMarkowitzQPTask → solve_portfolio_qp(...)
  - EmitOrdersFromQPSolutionTask: Δw → orders (with min_dw filter, Davis-Norman band)
   ↓ outputs: actual buy/sell orders

Phase 5: TopUp logic (when QP solver=qp, this is delegated TO the QP)
Phase 6: RecordScoreDistributionTask + persist
```

### What QP TAKES

| Input | Source | Type |
|---|---|---|
| `w_current` | broker state via `_qp_w_current` | n-vector (current portfolio weights) |
| `μ` (`_qp_mu`) | ML model → calibrator → expected return | n-vector |
| `Σ` (`_qp_sigma`) | per-name σ (calibrator OR realized vol fallback) + correlation matrix + Ledoit-Wolf shrinkage | n × n PSD matrix |
| `w_upper` | regime × confidence × vol_target × dd_scale × conviction | n-vector |
| `w_lower` | 0 (longs only) or `-max_short_pct` (BEAR regime when hard_bear=True) | n-vector |
| `wash_sale_mask` | per-ticker {sold last 30d? loss?} | n-bools |
| `turnover_max` | per-config (prod 0.2, shadow None) | scalar |
| sector / correlation indicators | from sector_map + corr_matrix | matrix + list |

### What QP EMITS

| Output | Consumer | Type |
|---|---|---|
| `Δw` (delta weights per asset) | `EmitOrdersFromQPSolutionTask` | n-vector |
| `target_w` (post-trade weights) | logging + diagnostics | n-vector |
| `status` (`optimal` / `optimal_no_signal` / `infeasible`) | error handling | str |
| `diagnostics` (n_iter, binding hints, …) | logging | dict |

`EmitOrdersFromQPSolutionTask` then translates Δw into real `BUY`/`SELL` orders:
- Filter trades below `min_dw` (no micro-trades)
- Apply Davis-Norman closed-form no-trade band
- Convert weight Δw to share count (rounding, fee buffer)
- Apply benchmark sleeve credit
- Emit to broker

### QP's true functional name

**QP is the sizing-and-order-translation layer that converts a candidate set + signal vector + risk constraints into a concrete trade list.**

It does NOT:
- Generate the signal (that's the ML model)
- Select candidates (that's the gates)
- Make regime/timing decisions (that's the regime detector)
- Track positions / cash / NAV (that's the runner / SimAdapter)

It DOES:
- Cross-asset risk-aware sizing (`w'Σw` term)
- Transaction-cost-aware trade sizing (`κ‖Δw‖₁` term)
- Hard cap joint enforcement (sector + correlation + per-asset + turnover all interact)
- Wash-sale enforcement at the trade-emit level

That's it. **It's a sizing layer, not an alpha layer.**

---

## Q3 — Is QP the ultimate solution?

Crisply: **No.** Here's why, with what the ultimate actually looks like.

### What "ultimate" would mean

Portfolio decision-making has a well-defined theoretical hierarchy:

| Level | What it solves | Mathematical complexity | Example |
|---|---|---|---|
| 0 — Heuristic | Sort by signal, take top-K | O(n log n), closed-form | RenQuant 103 |
| 1 — **Single-period MV** | One-bar Markowitz optimum | Convex QP | **Current RenQuant 104 (this is us)** |
| 2 — Multi-period MV | T-bar Markowitz, finite horizon | Convex QP × T, DP | cvxportfolio's MultiPeriodOpt |
| 3 — Stochastic DP / HJB | Continuous-time optimal control | PDE solve | Merton 1969 |
| 4 — Robust dynamic | Min-max over uncertainty in (μ, Σ, frictions) | SOCP / robust DP | Lobo-Boyd 1998 |
| 5 — RL end-to-end | Learn the policy from data | Deep neural net + sim | Recent academic / hedge fund RL |

**We are at Level 1**. The "ultimate" would be Level 3 (Merton's HJB equation for continuous-time portfolio with transaction costs) or Level 4 (robust formulation accounting for noisy μ, Σ).

### Why Level 1 isn't ultimate

Three structural reasons:

#### 3a. Single-period assumes signals are stationary across bars

The QP solves: "given today's μ and Σ, what's the optimal target?"

This is **myopic**. It treats each bar independently. But real signals decay over time (Garleanu-Pedersen 2013 partial-move literature) — the model's μ at t+1 will reflect a partially-decayed version of μ at t. The single-period QP doesn't know that — it acts as if today's μ will persist forever.

Level 2 (multi-period) solves the trade-off: "should I take a 30% Δw today knowing the signal will decay by t+5, so it's better to spread the move over multiple bars?" Single-period can't ask that question.

For our 60-day-horizon label this matters less than for intraday — slow signals decay slowly — but it's still suboptimal.

#### 3b. Markowitz assumes (μ, Σ) are known precisely

Mean-variance optimization is **provably optimal under perfect estimates of μ and Σ**. With NOISY estimates (which we have):

> Michaud, R. O. (1989). "The Markowitz optimization enigma: Is 'optimized' optimal?" *Financial Analysts Journal* 45(1), 31–42.

Michaud showed that Markowitz portfolios are extremely sensitive to estimation error — small μ̂ perturbations cause large weight changes. He coined the term "error maximizer" for mean-variance optimization in noisy regimes.

> Chopra, V. K., & Ziemba, W. T. (1993). "The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice." *Journal of Portfolio Management* 19(2), 6–11.

Chopra & Ziemba's quantitative finding: **errors in μ̂ are ~10× more damaging than errors in Σ̂** for Markowitz portfolios. Since our μ̂ has IC ≈ 0.04 (high noise), this is exactly our regime.

Level 4 (robust optimization, Garlappi-Uppal-Wang 2007 — which we have as a parameter but with `robust_mu_kappa = 0.0`, i.e. disabled) would explicitly account for μ̂ uncertainty:

```
maximize_w  min_{μ ∈ confidence set} [μᵀw - γw'Σw - κ‖Δw‖₁]
```

This "robust μ" formulation hedges against the worst-case μ in a confidence set around μ̂. It's MORE conservative than Markowitz but typically performs better OOS when μ̂ is noisy.

**We have the parameter for this and aren't using it.** That's a hint: the QP architecture is more general than what we're actually exploiting.

#### 3c. The "one solver call per bar" assumes the constraint set is consistent

Today's daily-104 bug + PR #123 v1 regression demonstrated: when 4 separate Tasks compose the constraint vectors (`ComputeQPConstraintsTask` → `ApplyExposureScalingTask` → `ApplyConvictionCapTask` → `BuildSectorConstraintMatrixTask`), it's possible to construct mutually-contradictory constraints (`w_upper[i] < w_current[i]` from soft scaling), and the QP correctly reports "infeasible" — but the operator's intent was "hold flat", which IS feasible.

The architectural failure isn't QP's. It's **the constraint-composition design**. Each Task builds part of the constraint set with no knowledge of what later Tasks will do to it. The fact that the bug surfaces as "infeasible" at the solver IS QP doing its job — but the system as a whole produces a broken decision.

A higher-level abstraction (Level 2 multi-period or Level 4 robust) would still have the same problem because it still relies on consistent constraints. **The constraint-composition layer is the architectural debt, not the optimizer.**

### So what IS ultimate?

For RenQuant 104's actual problem class (long-only US equity, ~150-name universe, daily rebalance, 60-day-horizon signal, noisy μ̂), the published-literature answer to "ultimate" is somewhere on this spectrum:

1. **Practical apex** (DeMiguel 2009 + Kelly fractional + Davis-Norman bands): **1/N within top-K candidates, scaled by per-name fractional Kelly, with closed-form no-trade bands.** Stays at Level 0 but uses every tested-robust simplification. Closed-form, deterministic, no solver.

2. **Theoretical apex** (Merton 1969 + Davis-Norman 1990 + Almgren-Chriss 2000): **Continuous-time HJB equation with transaction-cost friction.** PDE solve per bar. Mathematically elegant; computationally expensive; rarely deployed in production because the closed-form solution requires unrealistic assumptions about asset dynamics.

3. **Industry-grade compromise** (Boyd-Busseti-Diamond-Kahn-Koh-Nystrup-Speth 2024 cvxportfolio multi-period): **Level 2 multi-period MV with explicit signal-decay model + transaction-cost smoothing.** This is what cvxportfolio's `MultiPeriodOpt` (vs the `SinglePeriodOpt` we're imitating) does. Still QP machinery, but at Level 2 instead of Level 1.

We are at Level 1 (cvxportfolio's `SinglePeriodOpt`). **The "ultimate" within the same architectural family would be Level 2 (cvxportfolio's `MultiPeriodOpt`)** — same Boyd reference, same DCP language, just a longer planning horizon.

But — and this is the honest answer — **for our actual problem (60-day-horizon signal, noisy μ̂, 4-name typical portfolio), the marginal value of moving to Level 2 is likely smaller than moving DOWN to Level 0 (closed-form Kelly + 1/N hybrid).**

DeMiguel 2009 + Michaud 1989 + Chopra-Ziemba 1993 all point in the same direction: **at our signal-to-noise ratio, optimizer complexity gives diminishing returns. The bug surface area + maintenance cost dominates the optimization gain.**

### The honest ranking for our specific problem

Ordered by likely OOS Sharpe given our (μ̂ noise, Σ̂ noise, 60-day-horizon, ~150-asset universe):

1. **(Probably best)** Closed-form Kelly per name, with regime-conditional caps, top-K = 4-8, equal-weight when caps don't bind, sector cap via greedy projection. Per-asset σ-aware no-trade bands (Davis-Norman). NO solver.
2. **(Probably tied)** Current QP at Level 1, but with all soft penalties disabled, only hard constraints active. Strip down to: budget + per-asset cap + sector cap + turnover. Drop CVaR, robust-μ, impact, tax penalty, cash-drag (all already 0 in config anyway). Reduces 32 → ~10 parameters.
3. **(Riskier)** HRP (López de Prado 2016) — robust to Σ̂ noise but ignores μ̂.
4. **(Risk)** Move UP to cvxportfolio MultiPeriodOpt — Level 2 — only justifiable if we have a published signal-decay model for `fwd_60d_excess` and want to spread Δw over multiple bars. Not obviously a win at our scale.
5. **(Currently)** Full QP with all 32 params. Theoretically most general, empirically over-engineered for our regime.

The QP architectural review memo's recommended Option F (Hybrid) is item 1, with QP as fallback for the rare hard-constraint case.

---

## Summary of the three answers

| Q | Short answer |
|---|---|
| Why introduce QP? | 21-bug audit (2026-04-27) of the predecessor + user mandate "industrial-grade not hand-rolled" → cvxportfolio (Boyd/Stanford) chosen as reference → implemented matching the textbook |
| What's its functional role? | **Sizing-and-order-translation layer**: converts {admitted candidates + (μ̂, Σ̂) signal + hard caps} into concrete buy/sell orders. NOT an alpha layer, NOT a candidate selector. |
| Is it the ultimate solution? | **No.** It's Level 1 (single-period MV) in a 5-level hierarchy. The "ultimate" within the same family is Level 2 (cvxportfolio MultiPeriodOpt). The "practical apex" for our regime is closer to Level 0 (Kelly + Davis-Norman). The current Level 1 sits between, and DeMiguel 2009 + Michaud 1989 + Chopra-Ziemba 1993 all suggest Level 0 likely wins at our SNR. |

The recommendation in [`2026-06-02-qp-architecture-review-and-alternatives.md §8`](./2026-06-02-qp-architecture-review-and-alternatives.md) (Hybrid Option F with 3-phase migration) is consistent with the "practical apex" finding above: it moves us toward Level 0 for the common case while keeping QP as the fallback for the rare hard-constraint case.

## References (in addition to parent memo)

- Michaud, R. O. (1989). "The Markowitz optimization enigma: Is 'optimized' optimal?" *Financial Analysts Journal* 45(1), 31–42.
- Chopra, V. K., & Ziemba, W. T. (1993). "The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice." *Journal of Portfolio Management* 19(2), 6–11.
- Merton, R. C. (1969). "Lifetime Portfolio Selection under Uncertainty: The Continuous-Time Case." *Review of Economics and Statistics* 51(3), 247–257.
- Lobo, M. S., & Boyd, S. (1998). "The worst-case risk of a portfolio." [Stanford EE technical note] — robust formulation precursor.
- Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P., & Speth, J. (2024). *Multi-Period Portfolio Optimization* (cvxportfolio textbook, 2nd ed.) — explicitly distinguishes SinglePeriodOpt (what we use) from MultiPeriodOpt (the within-family upgrade path).
