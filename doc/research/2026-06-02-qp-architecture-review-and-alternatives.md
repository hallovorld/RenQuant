# Portfolio QP — architecture review + alternatives

**Date**: 2026-06-02 · **Updated**: 2026-06-03 (final scope after two codex re-review rounds + gemini review)
**Author**: Claude (mainline) · **Reviewers**: codex, gemini (PR #125, 2026-06-02 / 03)
**Scope**: Research memo + open questions. **Does NOT request authorization
to migrate to any specific allocator (Hybrid / MPO / hard-only QP / …).**
The decision-grade artifact is the offline WF A/B replay + the
`ConstraintSnapshot` contract PR sketched in §8 — NOT this memo.

> **Resolution log** (so the reader can see what claims were rejected and
> why before reading the body):
>
> 1. **First codex review** flagged 7 load-bearing claims as stale or
>    unsupported. Param inventory was wrong (claimed "8 of 32 active"
>    with stale numeric values), DeMiguel 2009 was miscited as "14
>    datasets" (actual: 14 models × 7 datasets), `label_autocorr_60` was
>    the wrong observable for signal-decay claims, the Option F
>    fallback boundary enumerated only sector/correlation rather than
>    the full hard-constraint set, 30-day live shadow was framed as a
>    Sharpe gate (it isn't), and the Level-0 / Kelly framing missed the
>    μ-error guardrails (shrinkage, edge floors, fractional sizing).
>    All seven corrected in §2-§8 below.
> 2. **Second codex re-review** flagged 4 MED contradictions where
>    companion text still preserved the old recommendation. The
>    executive summary still recommended the 3-phase Hybrid migration;
>    §3 chronology still flagged #123 as in-flight; the thinking-log
>    methodology section still cited the stale "8 of 32" count as if
>    it had been the performed audit. All corrected in this revision.
> 3. **PR #123 v4 status**: MERGED 2026-06-03 (commit on `main`). The
>    immutable `_qp_w_upper_hard` snapshot is now stamped by
>    `ComputeQPConstraintsTask` and the hard-cap-aware
>    `_clamp_w_upper_at_w_current` restores the hard cap on over-cap
>    rows. Cap-compliance fallback observably fires on the codex repro
>    values (`tests/test_qp_soft_scaling_clamps.py::test_cap_compliance_
>    fallback_sells_to_hard_cap_not_soft_cap`).
>
> This means **Step 0 of the §8 plan is DONE**. The remaining sequence is
> Steps 1-5 (`ConstraintSnapshot` contract → μ̂ autocorr measurement →
> offline WF A/B replay → optional live shadow for telemetry).

## Executive summary

The current `solve_portfolio_qp` solves a 32-parameter convex Markowitz
quadratic program with up to 5 objective terms (Markowitz risk +
Gârleanu-Pedersen linear cost + Almgren-Chriss 1.5-power impact + CVaR
+ soft cash-drag) and up to 5 hard-constraint families (budget + box
per-asset + per-bar slippage + wash-sale + turnover, plus optional
sector + correlation + gross-exposure). It runs through a
CLARABEL → OSQP → SCS solver chain via cvxpy.

The user's question — **is this complexity carrying its weight?** — is
correct to ask. This memo's findings (each load-bearing claim is
expanded in the cited section):

1. **The textbook machinery is partly inactive in our config (§2).**
   Of 32 `qp_*` keys, 25 are active in some way, 7 are zero-weighted
   (`qp_cash_drag_lambda`, `qp_cvar_lambda`, `qp_min_invested_pct`,
   `qp_min_share_floor_pct`, `qp_robust_mu_kappa`, `qp_signal_decay`,
   `qp_tax_aware`). We pay the complexity tax — 9 hard constraints +
   3 objective terms + a 4-Task constraint-composition pipeline — for
   a substantively smaller working set than the API suggests.

2. **The bugs of the last six weeks are constraint-composition bugs,
   not solver-arithmetic bugs (§3).** Bug F (`delta_below_min_dw`
   truncating top picks), today's daily-104 (`w_upper < w_current`
   silently infeasible), and #123's three rejected revisions all
   surfaced where the per-asset box bound is *built* by 4 composed
   Tasks (`ComputeQPConstraintsTask → ApplyExposureScalingTask →
   ApplyConvictionCapTask → sector/corr`), not in `cvxpy` itself.
   v4 (merged 2026-06-03) closed the specific cap-compliance vector;
   the *class* is still open and is what codex's `ConstraintSnapshot`
   and gemini's `BuildQPConstraintsTask` recommendations target.

3. **The published-literature *mechanism* for "simpler may compete
   with MV" applies to us (§4); a quantitative IC-based ranking does
   not.** DeMiguel-Garlappi-Uppal (2009) found that across **14
   portfolio models on 7 empirical datasets**, none was consistently
   better than naive 1/N in OOS Sharpe, certainty-equivalent return,
   or turnover — because estimation error in μ̂ and Σ̂ dominates the
   optimization gain. The paper does NOT publish an IC threshold at
   which MV catches 1/N, so we cannot conclude RenQuant 104 (WF gate
   +0.039 IC, rolling 60d Σ̂ on 142 stocks with Ledoit-Wolf λ=0.2)
   sits below a known boundary. The mechanism is consistent with our
   observed Sharpe degradation under turnover pressure; the
   quantitative placement is unsupported.

4. **At least six candidate alternatives exist (§5).** Greedy +
   Kelly (Option A), Hierarchical Risk Parity (B), LP with diagonal
   Σ (C), soft-only QP (D), closed-form Markowitz + projection (E),
   Hybrid greedy-select + Kelly-size + QP-fallback (F). Each has
   distinct trade-offs against the active constraint set; **none is
   pre-selected by this memo**.

5. **The decision-grade artifact is offline measurement, not this
   memo (§8).** Codex (MED-6) and gemini both rejected an earlier
   "ship Hybrid shadow now, decide on 30-day Sharpe later" plan: 30
   trading days does not separate a Sharpe delta of 0.1, and a shared
   `ConstraintSnapshot` contract must be in place before any
   allocator comparison is trustworthy. The corrected sequence is
   Step 1 (`ConstraintSnapshot` refactor) → Step 2 (μ̂ autocorr per
   regime — closes the misuse of `label_autocorr_60` that this memo
   originally relied on) → Step 4 (offline WF A/B replay with 5
   baselines + DSR/PBO multiple-comparison correction) → Step 5
   (live shadow ONLY for operational telemetry).

**This PR's authorization request**: NONE. The next concrete
authorization request will come with the `ConstraintSnapshot` contract
PR (Step 1) and again with the offline A/B replay PR (Step 4).

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
459 tests (456 prior + 3 added in PR #123 v4 for the hard-cap contract).
**The single function is in the top decile of complexity in the
codebase.**

## 2. Which capabilities are actually active in prod

**Re-audited 2026-06-02 against committed `strategy_config.json` head
(`rotation.joint_actions.qp_*` + regime overlays + `_build_solver_kwargs`).**
This replaces the earlier stale claim that "8 of 32 are active".

### 2.1 — All 32 keys with actual prod values

| QP parameter (key) | Prod-config value | Status |
|---|---|---|
| `qp_admission_gate.enabled`               | True (min_rank_score=0.55, BULL_CALM min_expected_return=0.04) | **active** (alpha gate) |
| `qp_band_method`                          | `"davis_norman"`  | **active** (no-trade band) |
| `qp_c2_infeasible_policy`                 | `"strict"`        | **active** (infeasibility handling) |
| `qp_cash_drag_lambda`                     | 0.0               | INACTIVE (was claimed 0.05 in stale memo) |
| `qp_conviction_cap_enabled`               | True              | **active** (per-ticker scaling) |
| `qp_correlation_cap_enabled`              | True              | **active** (hard correlation group cap) |
| `qp_cost_kappa`                           | **0.002** (NOT 0.0001) | **active** (G-P transaction cost) |
| `qp_cost_kappa_floor_round_trip`          | True              | **active** (cost floor) |
| `qp_cvar_lambda`                          | 0.0               | INACTIVE (R-U CVaR off) |
| `qp_drawdown_limit`                       | 0.2               | **active** (drawdown halt) |
| `qp_dw_max`                               | 0.5               | nominally active (per-asset trade cap) |
| `qp_horizon_contract`                     | `"strict"`        | **active** (μ/Σ horizon match) |
| `qp_ledoit_wolf_lambda`                   | 0.2               | **active** (covariance shrinkage) |
| `qp_min_dw_pct`                           | 0.02              | **active** (rounding floor) |
| `qp_min_invested_edge_floor`              | 0.002             | **active** |
| `qp_min_invested_pct`                     | **0.0** (NOT 0.7) | INACTIVE |
| `qp_min_invested_requires_positive_edge`  | True              | **active** |
| `qp_min_share_floor_pct`                  | 0.0               | INACTIVE |
| `qp_mu_contract`                          | `"strict"`        | **active** |
| `qp_mu_horizon_days`                      | 60                | **active** |
| `qp_no_trade_band_cap`                    | 0.05              | **active** |
| `qp_no_trade_band_factor`                 | 1.0               | **active** |
| `qp_risk_aversion`                        | 3.0               | **active** (Markowitz quadratic term) |
| `qp_robust_mu_kappa`                      | 0.0               | INACTIVE (Garlappi-Uppal off) |
| `qp_sector_cap_enabled`                   | True (per sector caps) | **active** (hard sector cap) |
| `qp_sigma_horizon_mode`                   | `"match_mu"`      | **active** |
| `qp_sigma_unit`                           | `"annualized"`    | **active** |
| `qp_signal_decay`                         | 0.0               | INACTIVE |
| `qp_soft_sell_guard.enabled`              | True (BULL_CALM min_holding=60d) | **active** (thesis-age guard) |
| `qp_tax_aware`                            | False             | INACTIVE (per "no tax-driven logic" mandate) |
| `qp_tax_lot_method`                       | `"hifo"`          | **active** (tax-lot accounting) |
| `qp_turnover_max`                         | 0.2 / **BULL_CALM=0.15** (NOT flat 0.2) | **active** |

**Per-regime `max_position_pct` overlays** (also reach the solver as `w_upper`):
- BULL_CALM = 0.15, CHOPPY = 0.15, BULL_VOLATILE = 0.20, BEAR = 0.00.

### 2.2 — Active/inactive count + constraint/objective accounting

**Active: 25 of 32** (7 inactive: `cash_drag_lambda`, `cvar_lambda`,
`min_invested_pct`, `min_share_floor_pct`, `robust_mu_kappa`,
`signal_decay`, `tax_aware`).

**Constraints (hard) reaching the solver**:
1. Per-asset box bound `w_lower ≤ w ≤ w_upper` (regime-scaled,
   confidence-scaled, soft-scaled — and after PR #123 v4 [merged
   2026-06-03], with an immutable `_qp_w_upper_hard` snapshot for
   cap-compliance fallback).
2. Cash budget `Σwᵢ + cash_reserve = 1` (regime cash_reserve).
3. Turnover cap `‖Δw‖₁ ≤ qp_turnover_max` (BULL_CALM=0.15).
4. Sector cap (per sector group, hard).
5. Correlation-group cap (hard pairwise/cluster).
6. Wash-sale / no-rebuy mask (per-ticker hard zero-Δw bound).
7. No-trade band (Davis-Norman, `qp_no_trade_band_cap=0.05`).
8. Drawdown gate (`qp_drawdown_limit=0.2`, halts buys).
9. Admission gate (`qp_admission_gate` — rank/panel/expected-return
   minima, BULL_CALM 4% 60d expected excess).

**Objective terms**:
- `μᵀw` (panel-LTR → calibrator → expected-return)
- `-γ wᵀΣw` (Markowitz, γ=3.0)
- `-κ‖Δw‖₁` (Gârleanu-Pedersen, κ=0.002)

That's **9 hard constraints + 3 objective terms** (the earlier "3
constraints + 3 objective terms" claim was wrong). The complexity-tax
argument therefore needs to be re-stated: the count is higher, and
the per-asset box bound is itself produced by 4 composed Tasks
(Compute → ApplyExposureScaling → ApplyConviction → sector/corr) —
**this is exactly the constraint-composition layer that codex's
`ConstraintSnapshot` recommendation targets**.

## 3. Failure modes observed in production

Chronological QP-related incidents:

| Date | Bug | Root cause |
|---|---|---|
| 2026-05-18 | "MCD same-day rebuy" (memory) | QP's wash-sale didn't include min_reentry; fixed via `anti_churn` heuristic outside QP |
| 2026-05-30 | **Bug F**: delta_below_min_dw truncates top picks (ORCL) | QP sized 13 buys at <2% each; emit task dropped all → 0 buys |
| 2026-06-01 | QP cap-compliance retry decoration (memory) | Existing fallback ran but every promote set `RQ_ALLOW_NO_WF=1` |
| 2026-06-02 | **Today's daily-104**: QP infeasible despite trivially feasible state | Soft-scaling pushed `w_upper < w_current` → hold-flat infeasible |
| 2026-06-02 | **PR #123 v1**: solver-level clamp masked cap-compliance | Codex caught: clamp made over-cap state silently `optimal_no_signal` |
| 2026-06-02 | **PR #123 v2**: moved clamp to soft-scaling tasks | Codex caught: `ApplyConvictionCapTask` still raised the hard cap up to over-cap `w_current`. Same bug, different layer. |
| 2026-06-02 | **PR #123 v3**: separate `_qp_w_upper_hard` snapshot | Codex caught: over-cap branch returned soft-scaled `w_upper` (≈ 7.5% under low conviction) rather than the hard 15%. Cap-compliance fallback would have sold to the SOFT cap, not the RISK cap. |
| 2026-06-03 | **PR #123 v4** (merged) | Over-cap branch now restores `w_hard_arr` (discards soft-scaled value). Cap-compliance fallback observably sells back to the hard cap at exactly 15% on codex's repro values. Regression guarded by `test_cap_compliance_fallback_sells_to_hard_cap_not_soft_cap`. |

**Pattern**: each of these bugs surfaced in **constraint composition**,
not in `cvxpy` arithmetic. The way we BUILD the per-asset box bound
from 4 separate Tasks (`ComputeQPConstraintsTask →
ApplyExposureScalingTask → ApplyConvictionCapTask → sector/correlation`)
produces edge cases where the composed constraints contradict each
other and the QP either returns infeasible OR (worse) silently
violates an expected invariant. PR #123 v4 closed one specific vector
(over-cap holdings under low conviction); the *class* — multi-Task
constraint composition with no shared contract — remains open.

This is a property of **constraint-composition complexity**, not
specific to QP. Any constrained solver consuming the same composed
state (LP, SOCP, MILP, or the Hybrid Stage-4 fallback in §5 Option F)
would inherit it. That is why §8 Step 1 (a single `ConstraintSnapshot`
contract shared by every candidate allocator) is the convergent
recommendation from both codex and gemini, independent of which
allocator eventually wins the §8 Step 4 offline A/B replay.

## 4. The theoretical critique (DeMiguel, Garlappi & Uppal 2009)

The foundational empirical study on this question is:

> DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive
> Diversification: How Inefficient Is the 1/N Portfolio Strategy?"
> *Review of Financial Studies* 22(5), 1915–1953.

Their finding, summarized: across **7 empirical datasets they tested
14 portfolio models**, and found that **none of the 14 was consistently
better than naive 1/N (equal-weight) in OOS Sharpe ratio,
certainty-equivalent return, or turnover**. The models tested include
Markowitz, minimum-variance, Bayes-Stein shrinkage, Black-Litterman,
and several others. The reason: **estimation error in μ̂ and Σ̂
dominates the in-sample optimization gain**.

DeMiguel's specific quantitative claim (their Table 3): a Markowitz
portfolio would need an OOS Sharpe lift of ~1.5× over 1/N just to break
even after accounting for the estimation error penalty.

**Important caveats codex flagged on re-review** (these limits matter
for the recommendation in §10):
- DeMiguel does NOT provide an IC-threshold taxonomy for "MV-vs-1/N".
  We cannot therefore conclude RenQuant at +0.039 IC sits "below"
  some published boundary.
- The defensible claim is that the *mechanism* (μ̂-error damage
  dominating OOS optimization gains) is well-established and consistent
  with our observed Sharpe degradation under turnover pressure — NOT
  that RenQuant matches DeMiguel's tested regimes quantitatively.

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

**Putting these together** (mechanism only, NOT a pre-decided ranking):
the regime is consistent with simpler methods being competitive vs full
MV. Whether they actually are — for THIS data, signal, cost structure —
is what the §8 Step 4 offline A/B replay measures.

A complementary mechanism paper:

> Brodie, J., Daubechies, I., De Mol, C., Giannone, D., & Loris, I.
> (2009). "Sparse and stable Markowitz portfolios." *Proceedings of the
> National Academy of Sciences* 106(30), 12267–12272.

Brodie et al. show that adding an L1 regularizer to Markowitz produces
sparse portfolios that hold 5-20 names equal-weight and outperform full
mean-variance OOS in their tested cases. The regularizer pushes the
solution toward 1/N on the subset of names with strong signal. This is
ONE empirical demonstration of the DeMiguel mechanism, not an
independent threshold result; the same caveat applies (no published IC
threshold for RenQuant's regime).

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

### Option F — Hybrid: greedy SELECT + per-name Kelly SIZE + QP feasibility check

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
  If the proposed portfolio violates ANY hard execution constraint, fall
  through to the QP. The fallback boundary MUST enumerate the full hard
  constraint set, not just sector/correlation (codex re-review #125):
    • per-asset hard cap (incl. existing over-cap holdings → cap-compliance
      sell-down, NOT relaxation)
    • cash budget after share-rounding
    • wash-sale / no-rebuy mask
    • turnover cap (`qp_turnover_max`, BULL_CALM 0.15)
    • forced sells (drawdown, risk exits)
    • missing-sector guard (no sector → cannot increase weight)
    • broker min-share / fee buffer
    • soft-sell thesis-age guard (BULL_CALM min 60d)
    • sector cap / correlation-group cap
  If ALL hard constraints satisfied → emit Stage-3 orders as-is, no QP.

  NOTE: `solve_portfolio_qp()` does NOT currently accept a warm start.
  A Hybrid implementation either omits warm-start (QP solves from cold
  on the fallback path) OR adds warm-start support as new infrastructure.
  Earlier drafts of this memo implied warm-start was available — it is
  not.
```

**Properties**:
- Stages 1-3 are all closed-form / O(n) — no optimization, no infeasibility
- QP only runs when ANY hard constraint is violated (which is *broader*
  than the earlier sector/correlation framing implied — see above)
- The fallback rate is empirically unknown until measured; the "90% of
  bars don't need QP" estimate above is a guess and should NOT be cited
  until the offline replay (§ revised path) actually measures it.
- Existing QP code stays as the fallback projector — no rewrite
- Per-stage testing: each stage is small enough to unit-test exhaustively
- **Open structural concern (gemini #2)**: greedy Stage 1 selects on
  standalone score — if top candidates are highly correlated, greedy
  grabs all of them, and Stage 4 can only react after the pool is fixed.
  A true optimizer evaluates marginal risk *before* selection. The
  Hybrid path therefore needs explicit pre-selection correlation
  pruning OR an offline measurement of how often this matters.

**Pros**:
- Massively simpler than the current path on the common case
- Preserves QP for the rare actual-hard-constraint case
- DeMiguel 2009 / Brodie 2009 supply the mechanism (μ̂-error damage)
  that motivates a per-name closed-form sizer — they do NOT establish
  a quantitative IC threshold at which Hybrid beats Level-1 QP for
  RenQuant's data; that is what §8 Step 4 offline A/B measures.

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

- **QP is theoretically correct but operationally complex for our
  problem class.** A 142-stock universe with 4-10 typical holdings,
  noisy signals at IC ≈ 0.04, rolling 60-day Σ̂ noise on 20k covariance
  cells. The mechanism DeMiguel 2009 identifies (μ̂-error damage) is
  active here, but **the paper does NOT bound the IC at which 1/N
  catches MV**; we cannot conclude RenQuant sits below a known
  boundary. We can only say: the regime is consistent with simpler
  methods being competitive, not that they are guaranteed to win.

- **The bugs aren't QP arithmetic — they're complexity-induced.** Every
  QP bug this month has been a constraint-composition issue, not a
  cvxpy bug. ANY constrained solver inherits this risk.

- **Replacement isn't free.** The QP infrastructure is now stable
  (456 tests passing, 4 weeks of production hardening). Migration costs
  real engineering time + introduces a new bug surface.

- **The work order is decoupled from the allocator choice.** Even if
  the §8 Step 4 offline A/B replay selects current QP (or hard-only QP,
  or MPO) over Hybrid, the §8 Step 1 `ConstraintSnapshot` contract
  refactor still lands first — it's the bug-class fix. Allocator
  selection comes after measurement, not before.

## 8. Recommendation (REVISED 2026-06-02 after codex re-review)

**Codex + gemini both rejected the original 30-day-shadow-then-promote
plan as insufficient for a Sharpe decision.** The corrected sequence is
measurement-and-contract first, ordered to surface the hard-constraint
contract bug before committing to ANY allocator change.

The decision-grade artifact will be the offline A/B replay + the
`ConstraintSnapshot` contract PR — NOT this memo. Authorization for
Hybrid migration is NOT requested by this PR.

### Step 0 — PR #123 v4 (DONE, merged 2026-06-03)

The immutable `_qp_w_upper_hard` snapshot is stamped by
`ComputeQPConstraintsTask` and the hard-cap-aware
`_clamp_w_upper_at_w_current` restores the hard cap on over-cap rows.
Cap-compliance fallback observably fires; the regression is guarded by
`tests/test_qp_soft_scaling_clamps.py::test_cap_compliance_fallback_sells_to_hard_cap_not_soft_cap`.
This pre-requisite for any allocator comparison is satisfied.

### Step 1 — Single `ConstraintSnapshot` / `BuildQPConstraintsTask`

Shared by ALL allocator candidates (current QP, Hybrid, MPO, …). One
contract carrying:

- per-asset hard cap, soft target cap, soft-floor (hold-flat)
- sector cap / correlation-group cap
- turnover budget
- wash-sale / no-buy / no-rebuy masks
- cash budget + share-rounding
- forced sells (drawdown, risk exits)
- broker min-share / fee buffer + missing-sector guard
- soft-sell thesis-age guard (`qp_soft_sell_guard`, BULL_CALM 60d)

Codex's `ConstraintSnapshot` recommendation and gemini's
`BuildQPConstraintsTask` consolidation are the same idea. This is the
work that fixes the bug *class* (constraint composition), not just the
symptom.

### Step 2 — Measure forecast-state autocorrelation per regime

Per-regime measurement of `corr(μ̂_t, μ̂_{t+k})` on the calibrator
output, plus top-K overlap and expected-return half-life. This closes
codex's HIGH-4: realized forward-label autocorrelation is NOT the same
observable, and gemini's Level-2 attractiveness argument depends on
this measurement. Quantifies whether MultiPeriodOpt is even a
candidate.

### Step 3 — Re-run param inventory against committed config

Already done in §2 above. The "25 of 32 active" number is the input
to the complexity-tax argument.

### Step 4 — Offline WF A/B replay with paired daily returns + regime
buckets

Five baselines explicitly: (a) current QP, (b) simplified-QP (hard-only
constraints, no soft-scaling layer), (c) Hybrid (§5 Option F, with the
fallback-boundary fixes above), (d) inverse-vol top-K, (e) equal-weight
top-K. Optional (f): Level-2 MultiPeriodOpt if Step 2 shows fast μ̂
decay.

Metrics: net-of-cost return, Sharpe (with `Sharpe_raw / DSR / PBO` —
§7.3), MDD, turnover, fallback rate, cap violations, forced-sell
preservation, sector/corr concentration, per-regime stratified
attribution.

**Non-negotiable gate**: zero hard-constraint regressions vs Step 1's
`ConstraintSnapshot`.

### Step 5 — Live shadow IF a candidate dominates offline

Live shadow is for **operational telemetry + implementation parity** —
fallback-rate drift, broker-side rounding behavior, observability hooks.
NOT a Sharpe selection gate. 30 trading days is insufficient sample for
a Sharpe delta of 0.1 (codex MED-6).

### What stays from QP (only relevant IF Step 4 selects Hybrid)

The following "stays / goes" lists describe the END-STATE of a Hybrid
adoption — they only apply if the Step 4 offline A/B replay actually
selects Option F over the other 4 baselines. They are NOT a
pre-decided migration plan.

- `qp_solver.py` itself (would be the fallback under Hybrid; can be kept
  indefinitely if not selected, or kept as the live engine if A/B
  selects current-QP / hard-only QP / Level-2 MPO)
- Davis-Norman closed-form no-trade band logic (already used; would be
  Stage 3.5 in the Hybrid)
- The wash-sale + earnings + admission gates (all Stage 1)
- The Kelly fraction calculation (Stage 2)
- The σ-aware sizing (Stage 2)
- The Ledoit-Wolf shrinkage (Stage 4 fallback only)
- The Markowitz / G-P theoretical references (move into Hybrid docstring)

### What goes away (only IF Step 4 selects Hybrid)

- The 32-parameter API on `solve_portfolio_qp` (becomes internal-only,
  fewer callers).
- The 5-task constraint composition chain (collapses into 3 closed-form
  stages — but only after the Step 1 `ConstraintSnapshot` contract
  refactor, which lands regardless of which allocator wins).
- The "infeasible" bug class on the common path.
- ~400 LOC of QP-supporting Tasks (ApplyExposureScalingTask,
  ApplyConvictionCapTask, BuildSectorConstraintMatrixTask, etc.) — they
  become smaller in-Stage helpers.

## 9. Open questions for codex review

1. **Is the IC-noise / estimation-error framing correct?** Codex
   re-review (2026-06-02) correctly flagged that DeMiguel 2009 does
   NOT provide an IC-threshold taxonomy. The defensible claim is
   only the mechanism (μ̂-error damage). Is there a more recent paper
   that DOES bound the IC at which MV catches up to 1/N? If so the
   recommendation should re-weight.

2. **Sector-cap enforcement post-Stage 2.** Recommended approach: 
   project sizes to satisfy sector caps via reverse-greedy (drop the
   weakest candidate in the saturated sector). Alternative: minimum-
   distance projection. Which is more defensible for our case?

3. **Hybrid fallback frequency is empirically unknown.** §5 Option F
   notes any pre-measurement estimate is a guess. The §8 Step 4 offline
   A/B replay measures the actual rate as one of its primary metrics.
   Should we set a max-fallback-rate threshold above which we'd reject
   Hybrid even if Sharpe is competitive (because two-path maintenance
   cost dominates)?

4. **Tax accounting under Step 5 live shadow.** §7.5 says one source of
   truth for tax. The Step 5 live-shadow logs (operational telemetry,
   not Sharpe gate) don't execute, so they don't need full tax
   simulation. But their log should still reflect after-tax expected
   return to be comparable. Right?

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

## 11. Verdict for the user (REVISED 2026-06-02 after codex re-review)

**QP is not "wrong" — its constraint-composition layer is the bug
class, and the empirical literature (DeMiguel 2009, Michaud 1989,
Chopra-Ziemba 1993, Brodie 2009, López de Prado 2016) provides a
*mechanism* (μ̂-error damage) under which simpler methods may match
or beat MV. The literature does NOT bound the IC at which 1/N catches
MV, so we cannot conclude RenQuant 104 sits below a known boundary
without measurement.**

The earlier "verdict" in this section recommended **authorizing
Phase 1 (Hybrid shadow path, 2 engineering days, 30 days observation)
now**. Codex and gemini both rejected that recommendation on PR #125
re-review (2026-06-02):

- **Codex MED-6**: 30 trading days does not statistically separate a
  Sharpe delta of 0.1. Live shadow is operational telemetry, not a
  Sharpe selection gate.
- **Codex / gemini convergent**: the constraint-composition contract
  must be fixed BEFORE any allocator comparison can be trusted. A
  Hybrid that consumes contradictory constraint state can still
  produce contradictory outputs.

The revised verdict is **authorize the measurement-and-contract
sequence** in §8 (Steps 0–5):

1. ~~Land PR #123 v4 (hard-cap separation).~~ **DONE — merged
   2026-06-03.** Cap-compliance fallback observably fires.
2. Build the single `ConstraintSnapshot` / `BuildQPConstraintsTask`
   contract that all candidate allocators consume.
3. Measure μ̂ autocorrelation per regime (closes HIGH-4).
4. (Done) Param inventory rebuilt from current config.
5. Offline WF A/B replay with 5 baselines + DSR/PBO.
6. ONLY IF a candidate dominates offline → live shadow for
   operational telemetry + implementation parity.

The decision-grade artifact is the offline A/B result + the
`ConstraintSnapshot` contract PR — NOT this memo.

**Authorization request**: NONE for migration. The next concrete
authorization request will come with the offline A/B replay PR.
