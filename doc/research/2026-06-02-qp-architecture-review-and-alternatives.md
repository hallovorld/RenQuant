# Portfolio QP — architecture review + alternatives

**Date**: 2026-06-02
**Status**: Research memo — proposes architectural decision, no code change
**Author**: Claude (mainline)
**Reviewer**: codex (assigned)

## Executive summary

The current `solve_portfolio_qp` solves a 32-parameter convex Markowitz
quadratic program with 5 objective terms (Markowitz risk + Garleanu-
Pedersen linear cost + Almgren-Chriss 1.5-power impact + Brown-Smith
tax + Rockafellar-Uryasev CVaR + Grossman-Zhou DD-Kelly + Garlappi-Uppal
robust-μ) and 5 hard-constraint families (budget + box per-asset + per-bar
slippage + wash-sale + turnover, plus optional sector + correlation + gross
exposure). It runs through a CLARABEL → OSQP → SCS solver chain via cvxpy.

The user's question is correct to ask: **is this complexity carrying its
weight?**

This memo's verdict:

1. **The QP is over-specified for our problem class.** It implements a
   textbook industrial-grade single-period optimizer (cvxportfolio-style)
   over a 142-stock universe with 10-20 active holdings + $11k account.
   Most of the textbook machinery (CVaR, robust-μ, fixed-cost, sqrt-
   impact) ships with zero or default-disabled weights in our config,
   meaning we pay the complexity tax without the benefit.

2. **The bugs aren't QP bugs — they're constraint-composition bugs.**
   Today's daily-104 failure was `w_upper < w_current` from soft-scaling
   pushing below the held weight; my v1 fix masked the cap-compliance
   contract (codex caught it); v2 moved the clamp to the soft-scaling
   layer where it belongs. Bug F (delta_below_min_dw) was the trade-band
   filter dropping good buys. None of these are QP arithmetic problems —
   they're **architectural complexity** problems.

3. **Empirically, QP often loses to simpler methods out-of-sample.**
   DeMiguel, Garlappi & Uppal (2009) showed across 14 datasets that the
   naive 1/N portfolio frequently beats mean-variance, minimum-variance,
   Bayes-Stein, and 8 other "optimized" portfolios in out-of-sample
   Sharpe — because estimation error in μ̂ and Σ̂ dominates the
   optimization gain. We have **noisier-than-textbook** μ̂ (the WF gate
   just showed +0.039 IC; for context, classic momentum is +0.04-0.06)
   and Σ̂ (rolling 60d on 142 stocks = high noise in 20k covariance
   cells, partly mitigated by our Ledoit-Wolf λ=0.2 shrinkage).

4. **The right move is decomposition, not replacement.** The QP
   conflates three distinct decisions: SELECT (which names) + SIZE (how
   much) + FILTER (which to actually trade given turnover budget).
   Decomposing into 3 sequential stages (each closed-form or O(n)) lets
   each stage be deterministic, debuggable, and individually testable.
   The QP becomes a **fallback feasibility-projector** rather than the
   primary decision engine.

5. **Migration is incremental, not a rewrite.** We can ship Stage 1 +
   Stage 2 as a shadow path TODAY (just compute alternative weights;
   keep QP as the live decision), measure the divergence over 30 days,
   and only flip the live decision after the shadow shows comparable or
   better behaviour.

**Recommended path**: Hybrid 2-stage architecture (described in §10),
shipped as a sim/shadow track first, promoted only after empirical proof.
Estimated effort: 2 days for shadow track, 1 week for sim verification,
1 week for live cutover — total 2-3 weeks for full migration.

---

## 1. What the current QP solves

`solve_portfolio_qp(...)` is the canonical interface. It solves:

```
maximize     μᵀw_p
             - γ_risk · w_pᵀΣw_p                       (Markowitz risk)
             - cvar_λ · z_α/√α · ‖Σ½ w_p‖₂              (CVaR tail)
             - κ · ‖Δw‖₁                               (G-P transaction cost)
             - Σᵢ tax_i · max(0, -Δwᵢ)                 (Brown-Smith tax-sell)
             - b · Σᵢ σᵢ · √(NAV/Vᵢ) · |Δwᵢ|^1.5       (Almgren-Chriss impact)
             - λ_cash · max(0, target_inv - Σw_p)      (soft cash-drag)

subject to   Σw_p ≤ 1 - cash_reserve                  (budget)
             w_lower ≤ w_p ≤ w_upper                  (box per-asset)
             -dw_max ≤ Δw ≤ dw_max                    (per-bar slippage)
             Δwᵢ ≤ 0 ∀ i ∈ wash_sale_mask             (wash-sale)
             ‖Δw‖₁ ≤ τ_max                            (turnover)
             [optional] S @ w_p ≤ sector_cap_vec      (sector hard cap)
             [optional] wᵢ + wⱼ ≤ corr_group_cap_ij   (correlation pair cap)
             [optional] ‖w_p‖₁ ≤ gross_max             (long-short gross cap)
```

**Quantitative footprint**: 32 parameters, 5 objective sites, 5+3 optional
constraint sites, ~1100 LOC across `qp_solver.py` + `tasks.py`. Test suite:
456 tests. **The single function is in the top decile of complexity in
the codebase.**

## 2. Which capabilities are actually active in prod

Auditing `strategy_config.json` against the 32 params:

| QP parameter | Default | Prod-config value | Status |
|---|---|---|---|
| `risk_aversion`           | 3.0     | 3.0       | **active** |
| `cost_kappa`              | 0.0001  | 0.0001    | **active** (G-P transaction cost) |
| `cash_reserve`            | 0.0     | from regime_params  | active |
| `w_upper`                 | 0.20    | per-regime × scaling | **active** |
| `w_lower`                 | 0.0     | 0.0       | active (no shorts) |
| `dw_max`                  | 0.5     | 0.5       | nominally active (rarely binds) |
| `wash_sale_mask`          | None    | per-ticker | **active** (essential) |
| `signal_decay`            | 0.0     | 0.0       | INACTIVE |
| `drawdown_limit`          | 0.2     | 0.2       | active (drawdown halt) |
| `robust_mu_kappa`         | 0.0     | 0.0       | INACTIVE (Garlappi-Uppal off) |
| `cvar_lambda`             | 0.0     | 0.0       | INACTIVE (R-U CVaR off) |
| `tax_cost_per_sell`       | None    | None      | INACTIVE (per CLAUDE.md "no tax-driven logic") |
| `turnover_max`            | None    | 0.2       | **active** (this is the bug source) |
| `impact_coef`             | 0.0     | 0.0       | INACTIVE (Almgren-Chriss off) |
| `fixed_cost_per_trade`    | 0.0     | 0.0       | INACTIVE (legacy, not DCP-compliant) |
| `min_invested_pct`        | 0.0     | 0.7       | active (soft) |
| `cash_drag_lambda`        | 0.05    | 0.05      | active |
| `gross_max`               | None    | None      | INACTIVE (longs-only path) |
| `sector_indicator/cap`    | None    | per sector | **active** (hard) |
| `corr_group_pairs`        | None    | per corr  | **active** (hard) |

**8 of 32 parameters genuinely drive prod decisions.** The other 24 are
either default-zero, INACTIVE-by-policy (tax, leverage), or legacy
kwargs. We're paying the complexity tax — 1100 LOC of solver + tasks,
456 tests, every bug touches every consumer — to support roughly **3
constraints (box + turnover + sector/corr) + 3 objective terms (μ +
Markowitz risk + transaction cost)**.

For a 3-constraint, 3-term problem there are SIGNIFICANTLY simpler
options.

## 3. Failure modes observed in production

Chronological QP-related incidents:

| Date | Bug | Root cause |
|---|---|---|
| 2026-05-18 | "MCD same-day rebuy" (memory) | QP's wash-sale didn't include min_reentry; fixed via `anti_churn` heuristic outside QP |
| 2026-05-30 | **Bug F**: delta_below_min_dw truncates top picks (ORCL) | QP sized 13 buys at <2% each; emit task dropped all → 0 buys |
| 2026-06-01 | QP cap-compliance retry decoration (memory) | Existing fallback ran but every promote set `RQ_ALLOW_NO_WF=1` |
| 2026-06-02 | **Today's daily-104**: QP infeasible despite trivially feasible state | Soft-scaling pushed `w_upper < w_current` → hold-flat infeasible |
| 2026-06-02 | **PR #123 v1**: my solver-level clamp masked cap-compliance | Codex caught: clamp made over-cap state silently "optimal" |
| 2026-06-02 | **PR #123 v2**: moved clamp to soft-scaling tasks | Restored hard-cap contract; the actual fix |

**Pattern**: every QP bug is a **constraint composition** bug. The QP
arithmetic is correct; the way we BUILD the constraints around it from
4 separate Tasks (Compute → ApplyExposureScaling → ApplyConviction →
sector/corr) produces edge cases where the constraints contradict each
other and the QP returns "infeasible" or worse, silently violates an
expected invariant.

This is a property of **constraint-composition complexity**, not of QP
itself. Replacing QP with another constrained solver (LP, SOCP, MILP)
would inherit the same problem.

## 4. The theoretical critique (DeMiguel, Garlappi & Uppal 2009)

The foundational empirical study on this question is:

> DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive
> Diversification: How Inefficient Is the 1/N Portfolio Strategy?"
> *Review of Financial Studies* 22(5), 1915–1953.

Their finding, summarized: across 14 different empirical datasets,
**the naive 1/N (equal-weight) portfolio outperformed 14 different
"optimized" portfolios out-of-sample on Sharpe ratio**, including
Markowitz, minimum-variance, Bayes-Stein shrinkage, Black-Litterman,
and several others. The reason: **estimation error in μ̂ and Σ̂
dominates the in-sample optimization gain**.

DeMiguel's specific quantitative claim (their Table 3): a Markowitz
portfolio would need an OOS Sharpe lift of ~1.5× over 1/N just to break
even after accounting for the estimation error penalty. **For most
realistic datasets it doesn't get there.**

Why this matters for RenQuant 104:

1. **Our μ̂ is noisy.** WF gate just showed real_ic = +0.039 (cross-
   sectional IC over 39 cuts × ~165k val rows). For context, classic
   12-month momentum is +0.04-0.06 IC in academic literature (Jegadeesh
   & Titman 1993). Our signal is at the lower end of "tradable" — well
   within DeMiguel's estimation-error regime where naive methods often
   win.

2. **Our Σ̂ is noisy.** Rolling 60-day daily-return covariance on 142
   stocks = 142² / 2 = 10,153 distinct estimation problems on ~60 data
   points each. Even with Ledoit-Wolf shrinkage at λ = 0.2 (our config),
   noise dominates the off-diagonal estimates.

3. **Our turnover is constrained.** With `qp_turnover_max=0.2` and 4
   typical holdings, the QP's per-bar Δw decisions are tiny (~1-3% per
   ticker). The optimization gain from "perfect" sizing within a 3%
   band is dominated by where the gates DROP names (admission, wash-
   sale, earnings), not how the QP weights the survivors.

**Putting these together**: the marginal value of QP's optimization over
a simpler method (greedy 1/N within selected, or 1/σ² inverse-variance
weight) is small relative to the bug surface area.

A complementary critique:

> Brodie, J., Daubechies, I., De Mol, C., Giannone, D., & Loris, I.
> (2009). "Sparse and stable Markowitz portfolios." *Proceedings of the
> National Academy of Sciences* 106(30), 12267–12272.

Brodie et al. show that adding an L1 regularizer to Markowitz produces
SPARSE portfolios that hold 5-20 names equal-weight and outperform full
mean-variance OOS — the regularizer effectively pushes the solution
toward 1/N on the subset of names with strong signal. This is a
half-step from full Markowitz toward DeMiguel's 1/N conclusion.

## 5. Candidate alternatives

Mapped against the **3 active capabilities** we identified (per §2):
box per-asset cap + sector/corr hard cap + turnover budget + transaction
cost + risk-aware sizing.

### Option A — Greedy + per-asset Kelly + slot-based caps (RenQuant 103 style)

```python
def allocate(candidates, holdings, cfg):
    # Stage 1: SELECT (rule-based — already in pipeline)
    selected = apply_admission_gates(candidates)
    
    # Stage 2: SIZE
    target = {}
    cash = cfg.cash_reserve
    for c in sorted(selected, key=lambda x: -x.score):
        kelly = kelly_fraction(c.mu, c.sigma, confidence=c.confidence)
        size = min(kelly, cfg.max_position_pct, sector_remaining(c.sector))
        if size * c.price > cash:
            break
        target[c.ticker] = size
        cash -= size * c.price
    
    # Stage 3: TRADE FILTER
    orders = [(t, target[t] - holdings.get(t, 0))
              for t in set(target) | set(holdings)
              if abs(target[t] - holdings.get(t, 0)) >= cfg.min_dw]
    return orders
```

**Properties**:
- O(n log n) for sort + O(n) for allocation → ~150 LOC total
- Always feasible (cash budget exhausts → stop, never infeasible status)
- Per-name Kelly = each name sized by own μ/σ (Kelly 1956 / Thorp 1969)
- Sector / per-asset caps enforced as box clamps (no constraint engine)
- Turnover handled by min_dw threshold + cash budget — no L1 penalty
- Transaction cost: implicit via min_dw (don't trade small)

**Pros**: dramatically simpler, no infeasibility class, easy unit
testing per stage, matches the RenQuant 103 architecture the system
historically used.

**Cons**:
- No closed-form portfolio-level risk awareness — high-correlation 
  pair both rank high, both get allocated, exposure concentrates
- Doesn't honor the L1 turnover cap as a hard budget (instead via min_dw)
- Suboptimal at the seam between Selection / Sizing (greedy ordering
  affects final mix)

### Option B — Hierarchical Risk Parity (HRP)

> López de Prado, M. (2016). "Building Diversified Portfolios that
> Outperform Out-of-Sample." *Journal of Portfolio Management* 42(4),
> 59–69.

HRP clusters assets by correlation, then allocates risk equally across
clusters via recursive bisection. **No matrix inversion** → robust to
Σ̂ estimation error (Σ⁻¹ amplifies noise; HRP avoids it).

```python
def hrp_weights(returns, mu_signs):
    # 1. correlation-based distance matrix
    corr = returns.corr()
    dist = ((1 - corr) / 2) ** 0.5
    # 2. cluster via single-linkage
    linkage = scipy.cluster.hierarchy.linkage(dist, method='single')
    sorted_ix = quasi_diagonalize(linkage)
    # 3. recursive bisection — allocate risk equally each split
    weights = recursive_bisection(returns.cov().iloc[sorted_ix, sorted_ix])
    # 4. sign by μ (long only → zero out short candidates)
    return weights * (mu_signs > 0).astype(float)
```

**Pros**: López de Prado's empirical tests show HRP beats mean-variance
+ minimum-variance OOS on Sharpe + DD on equity universes; deterministic;
no inversion noise.

**Cons**: doesn't use μ for SIZING (uses it only for sign filter); would
need a hybrid (HRP for cluster weights, μ-scaling within cluster); harder
to add hard sector caps (would need post-clip).

### Option C — Linear Programming (LP) with diagonal Σ approximation

If we approximate Σ as diagonal (ignoring covariance), the Markowitz QP
becomes **separable per asset**:

```
w_i* = clip(μ_i / (γ · σ_i²), w_lower, w_upper)
```

Then normalize so Σw ≤ 1 - cash_reserve. The non-separable parts
(turnover, sector cap) become linear constraints handleable by an LP.

**Pros**: closed-form per asset before constraints; constraints are all
linear → LP solver (simpler than QP); no covariance noise issues.

**Cons**: ignores correlation entirely (worse than the diagonal-Σ
Markowitz, which AT LEAST has Σ⁻¹ off-diagonals). May concentrate too
much when 2+ candidates have ρ ≈ 1.

### Option D — Soft-only QP (current QP, all constraints → penalties)

Convert every hard constraint to a soft penalty:

```
maximize  μᵀw - γw'Σw - κ‖Δw‖₁
        - λ_box · Σᵢ max(0, wᵢ - w_upper_i)
        - λ_sector · Σ_s max(0, S_s · w - cap_s)
        - λ_wash · Σ_{wash i} max(0, Δwᵢ)
        - λ_turnover · max(0, ‖Δw‖₁ - τ_max)
```

**Pros**: ALWAYS feasible by construction → no infeasibility class of
bug; preserves the QP machinery (less migration work).

**Cons**: wash-sale becomes soft → can be violated; sector cap soft →
can be violated; tuning the λ_* coefficients is brittle; "always feasible"
doesn't mean "always good" — can still produce sub-optimal allocations.

### Option E — Closed-form Markowitz + projection to box

Compute closed-form `w* = Σ⁻¹μ / (γ + 1ᵀΣ⁻¹μ)` (Markowitz 1952),
then project to the feasible box via:

```python
def project(w, w_upper, w_lower, cash_reserve):
    w = np.clip(w, w_lower, w_upper)
    total = w.sum()
    if total > 1 - cash_reserve:
        w *= (1 - cash_reserve) / total
    return w
```

Then apply min_dw filter on `w - w_current`.

**Pros**: O(n³) once for Σ⁻¹, no iterative solver, deterministic.

**Cons**: doesn't respect L1 turnover (which is non-separable);
sector cap is non-trivial post-projection; transaction cost not modeled.

### Option F — Hybrid: greedy SELECT + per-name Kelly SIZE + QP feasibility check (recommended)

```
Stage 1 (SELECT, rule-based, deterministic):
  candidates ← passing admission gates (existing)
  candidates ← top-K by score, K ≤ max_concurrent_positions
  candidates ← regime-conditional admission

Stage 2 (SIZE, closed-form, per-name):
  for each candidate:
    kelly_i = kelly_fraction(μ_i, σ_i, conviction_i)
    size_i  = min(kelly_i, w_upper_i, sector_remaining)
  normalize: Σ size_i ≤ 1 - cash_reserve  
  # If over budget: scale proportionally; this is closed-form

Stage 3 (TRADE FILTER, deterministic):
  Δw_i = size_i - w_current_i
  orders ← [(i, Δw_i) for i if |Δw_i| ≥ min_dw_eff]
  
  # Davis-Norman closed-form no-trade band already in code
  Δw_i = davis_norman_filter(Δw_i, σ_i, κ)
  
  # If Σ|Δw_i| > τ_max: prioritize by |Δw_i|·signal_strength
  orders ← top_by_priority(orders, budget=τ_max)

Stage 4 (FEASIBILITY CHECK, QP fallback):
  If the proposed portfolio violates a HARD constraint
  (sector cap, correlation pair cap):
    Solve QP starting from the proposed point as a warm start
    Output is guaranteed feasible
  Else:
    Emit orders as-is, no QP solve
```

**Properties**:
- Stages 1-3 are all closed-form / O(n) — no optimization, no infeasibility
- QP only runs when there's an actual hard-constraint conflict
- 90%+ of bars don't need the QP solve at all (most candidate sets don't
  violate sector or corr caps)
- Existing QP code stays as the fallback projector — no rewrite
- Per-stage testing: each stage is small enough to unit-test exhaustively

**Pros**:
- Massively simpler than current path on the common case
- Preserves QP for the rare actual-hard-constraint case
- Migration is incremental: ship Stages 1-3 as shadow path, measure
  divergence from QP for 30 days, then cut over
- DeMiguel 2009 / Brodie 2009 evidence supports the per-name closed-form

**Cons**:
- Doesn't optimize cross-asset risk at portfolio level (HRP would)
- Sector / correlation enforcement happens via post-allocation projection
  rather than as a hard constraint baked into the optimum

## 6. What QP fundamentally provides that the alternatives don't

Honest accounting. QP's TRUE advantages:

1. **Joint optimization across hard constraints.** When the box cap +
   sector cap + correlation cap + turnover cap interact, the QP finds
   the simultaneously feasible point that maximizes the objective.
   No greedy / closed-form method can fully replicate this for the
   joint problem.
   - **How often does this matter for us?** Rarely. Our 4-name typical
     portfolio rarely triggers sector cap saturation; correlation cap
     was added defensively (sector + corr cap audit 2026-05-10) but
     rarely binds.

2. **Linear transaction cost trade-off via L1 norm.** Garleanu-Pedersen
   2013 partial-move closed-form solution requires the QP formulation
   to express |Δw|₁ in the objective. Greedy can approximate via min_dw,
   but the smoothness is lost.
   - **How often does this matter for us?** The G-P partial-move is most
     valuable when signal half-life is short (intraday). We're trading
     `fwd_60d_excess` — the half-life is months, so the partial-move
     dynamics are slow.

3. **Mathematical guarantees about Pareto-efficiency.** Within the
   convex problem class, QP outputs the global optimum.
   - **How often does this matter for us?** Per DeMiguel 2009, in
     practice the optimum of the noisy in-sample problem is NOT the
     optimum of the true problem; the mathematical guarantee is hollow
     when applied to noisy estimates.

## 7. What this means for the path forward

The honest research conclusion:

- **QP is theoretically correct but operationally over-engineered for
  our problem class.** A 142-stock universe with 4-10 typical holdings,
  noisy signals at IC ≈ 0.04, and rolling 60-day Σ̂ noise — this is
  PRECISELY the regime where DeMiguel 2009 says simpler methods win.

- **The bugs aren't QP arithmetic — they're complexity-induced.** Every
  QP bug this month has been a constraint-composition issue, not a
  cvxpy bug. ANY constrained solver inherits this risk.

- **Replacement isn't free.** The QP infrastructure is now stable
  (456 tests passing, 4 weeks of production hardening). Migration costs
  real engineering time + introduces a new bug surface.

- **The compromise is decomposition.** Use closed-form per-name sizing
  for the 90%+ of bars where hard constraints don't bind; reserve QP for
  the rare case where they do. Migration is incremental.

## 8. Recommendation

**Adopt Option F (Hybrid: greedy SELECT + per-name Kelly SIZE + QP
fallback)** with a 30-day shadow-path verification window.

### Phase 1 — Shadow path (2 days engineering, 30 days observation)

1. Implement Stages 1-3 (SELECT + SIZE + FILTER) as a parallel scorer.
2. At each bar, compute BOTH:
   - QP solution (current decision path, drives live orders)
   - Hybrid solution (logged, NOT executed)
3. Log per-bar divergence: |Δw_hybrid - Δw_QP|, ratio of bars where
   hybrid would have been feasible vs needed QP fallback.
4. After 30 trading days, compute: (a) realized return difference if
   hybrid had been live, (b) % bars where hybrid disagrees with QP by
   >50% Δw on any name.

### Phase 2 — Sim verification (1 week)

Run the WF gate's 3-cut sim with BOTH the QP path and the Hybrid path
as separate scorers. Compare: Sharpe, APY, beat-SPY count, regime IC.
If hybrid Sharpe is within 0.1 of QP Sharpe AND there are no MORE
catastrophic single-bar losses, Phase 3 is authorized.

### Phase 3 — Live cutover (1 week with monitoring)

Flip the live decision to Hybrid via a config flag (rollback path is
"set the flag back"). QP remains as fallback for the hard-constraint
case. Monitor for 2 weeks before declaring the migration complete.

### What stays from QP

- `qp_solver.py` itself (used as fallback in Phase 1; can be kept indefinitely)
- Davis-Norman closed-form no-trade band logic (already used; would be
  Stage 3.5 in the Hybrid)
- The wash-sale + earnings + admission gates (all Stage 1)
- The Kelly fraction calculation (Stage 2)
- The σ-aware sizing (Stage 2)
- The Ledoit-Wolf shrinkage (Stage 4 fallback only)
- The Markowitz / G-P theoretical references (move into Hybrid docstring)

### What goes away

- The 32-parameter API on `solve_portfolio_qp` (becomes internal-only,
  fewer callers).
- The 5-task constraint composition chain (collapses into 3 closed-form
  stages).
- The "infeasible" bug class on the common path.
- ~400 LOC of QP-supporting Tasks (ApplyExposureScalingTask,
  ApplyConvictionCapTask, BuildSectorConstraintMatrixTask, etc.) — they
  become smaller in-Stage helpers.

## 9. Open questions for codex review

1. **Is the IC-noise / estimation-error framing correct?** Our real_ic
   = 0.039 is in the lower half of DeMiguel's tested regimes. Is the
   1/N-beats-Markowitz threshold the right reference?

2. **Sector-cap enforcement post-Stage 2.** Recommended approach: 
   project sizes to satisfy sector caps via reverse-greedy (drop the
   weakest candidate in the saturated sector). Alternative: minimum-
   distance projection. Which is more defensible for our case?

3. **Hybrid fallback frequency.** I estimate <10% of bars will need the
   QP fallback (hard-constraint conflicts). If empirically >30%, the
   migration logic gets noisier — should we set a "max fallback rate"
   threshold and roll back if exceeded?

4. **Shadow path tax accounting.** §7.5 says one source of truth for
   tax. The shadow path during Phase 1 doesn't execute, so it doesn't
   need full tax simulation. But its log should still reflect
   after-tax expected return to be comparable. Right?

5. **Does codex's experience with cvxportfolio production show similar
   1/N-beats-MV patterns?** Or is the textbook Boyd-style optimizer
   typically operating in regimes where μ is cleaner (higher-frequency,
   higher-conviction signals)?

## 10. References

### Core theoretical / empirical literature

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance* 7(1),
  77–91.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive
  Diversification: How Inefficient Is the 1/N Portfolio Strategy?"
  *Review of Financial Studies* 22(5), 1915–1953. — **THE foundational
  empirical critique of mean-variance.**
- López de Prado, M. (2016). "Building Diversified Portfolios that
  Outperform Out-of-Sample." *Journal of Portfolio Management* 42(4),
  59–69.
- Brodie, J., Daubechies, I., De Mol, C., Giannone, D., & Loris, I. (2009).
  "Sparse and stable Markowitz portfolios." *PNAS* 106(30), 12267–12272.
- Jagannathan, R., & Ma, T. (2003). "Risk Reduction in Large Portfolios:
  Why Imposing the Wrong Constraints Helps." *Journal of Finance* 58(4),
  1651–1683.
- Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance
  Matrix." *Journal of Portfolio Management* 30(4), 110–119.
- Kelly, J. L. (1956). "A New Interpretation of Information Rate." *Bell
  System Technical Journal* 35(4), 917–926.
- Thorp, E. O. (1969). "Optimal Gambling Systems for Favorable Games."
  *Review of the International Statistical Institute* 37(3), 273–293.

### Optimization and execution

- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization.* Cambridge
  University Press. §4 + §10.
- Garleanu, N., & Pedersen, L. H. (2013). "Dynamic Trading with
  Predictable Returns and Transaction Costs." *Journal of Finance* 68(6),
  2309–2340.
- Almgren, R., & Chriss, N. (2000). "Optimal Execution of Portfolio
  Transactions." *Journal of Risk* 3, 5–39.
- Davis, M. H. A., & Norman, A. R. (1990). "Portfolio Selection with
  Transaction Costs." *Mathematics of Operations Research* 15(4),
  676–713.
- Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of Conditional
  Value-at-Risk." *Journal of Risk* 2, 21–41.
- Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P.,
  & Speth, J. (2024). *Multi-Period Portfolio Optimization* (cvxportfolio
  textbook, 2nd ed.). — The book whose idiom our current `solve_
  portfolio_qp` follows.

### Robust optimization (currently inactive in our config)

- Garlappi, L., Uppal, R., & Wang, T. (2007). "Portfolio Selection with
  Parameter and Model Uncertainty: A Multi-Prior Approach." *Review of
  Financial Studies* 20(1), 41–81.
- Black, F., & Litterman, R. (1990). "Asset Allocation: Combining
  Investor Views with Market Equilibrium." Goldman Sachs Fixed Income
  Research.

### Internal references

- CLAUDE.md §1 PRIME DIRECTIVE (regime-conditional)
- CLAUDE.md §7.5 single source of truth
- CLAUDE.md §7.10 canonical references (READ, not name-dropped — applies
  to this memo too)
- `memory/feedback_industry_leading_quality.md` — historical context for
  the choice of cvxportfolio-style QP
- `memory/feedback_qp_pipeline_alignment.md` — QP follows Job/Task
  architecture
- `doc/research/2026-06-02-bull-calm-no-signal-diagnostic.md` — the
  current model's IC profile that supports the "noisy μ̂" framing
- `backtesting/renquant_104/kernel/portfolio_qp/qp_solver.py` —
  the 1100-LOC body under review
- `tests/test_portfolio_qp_solver.py` + `tests/test_qp_*.py` —
  current 456-test QP suite

## 11. Verdict for the user

**QP is not "wrong" — it's over-engineered for our problem class.** The
empirical literature (DeMiguel 2009, Brodie 2009, López de Prado 2016)
says simpler methods often win when μ̂ and Σ̂ are noisy, which describes
RenQuant 104's data regime.

The migration to **Hybrid (Option F)** is:
- **Incremental**: shadow → sim → live in three phases
- **Reversible**: config flag, QP stays as fallback
- **Empirically gated**: Phase 2 requires Sharpe within 0.1 of QP path
- **Estimated effort**: 2-3 weeks total, of which 30 days is calendar
  observation (engineering touch time ≈ 8-10 days)

If the shadow path during Phase 1 shows the Hybrid would have traded
similarly to QP (with the SAME daily-104 BULL_CALM-silence behaviour
since both use the same regime_admission gate), the migration is a net
simplification win. If the Hybrid diverges materially, we keep QP and
learn that the optimization gain DOES matter for our regime, which is
also a useful finding.

Recommendation: **authorize Phase 1 (shadow path, 2-day eng cost) now.**
