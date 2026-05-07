# QP Refactor Plan — adopt cvxportfolio (Boyd/Stanford) reference

**Status:** design phase. Source read 2026-05-06.
**Reference:** github.com/cvxgrp/cvxportfolio @ main
- Boyd group, Stanford
- 7966 LOC, GPL-3 / Apache-2 (dual license)
- Powers BlackRock-funded research; cited 100+ times since 2016 paper

## What we have today (qp_solver.py + tasks.py)

**Strengths (adopted 2026-05):**
- Almgren-Chriss sqrt market impact ✓ (matches cvxportfolio TransactionCost)
- Ledoit-Wolf Σ shrinkage ✓ (matches cvxportfolio FullCovariance)
- CVaR via λ + α ✓
- Davis-Norman no-trade band ✓ (post-solve, not native to cvxportfolio)
- min_invested_pct soft floor + capacity clamp + warm-start ✓
- 14 bug fixes today (tax accounting, NaN handling, earnings blackout, etc.)

**Gaps vs cvxportfolio:**
1. **Backend**: scipy.optimize.minimize + SLSQP. Vs cvxpy + auto-select OSQP/SCS/MOSEK. SLSQP is general-purpose (slow on convex QP); OSQP is tuned for QP (10-100× faster typical).
2. **Cost composition**: rolled-our-own additive costs in `tasks.py`. Vs cvxportfolio's `Cost` class hierarchy with `compile_to_cvxpy` returning convex expressions and explicit `assert expression.is_convex()` checks.
3. **Numerical safety**: SLSQP has known issues with infeasibility detection ("Positive directional derivative for linesearch"). cvxpy has explicit `ProgramInfeasible` / `ProgramUnbounded` exceptions.
4. **Risk model abstraction**: our Σ is a numpy matrix passed around. Cvxportfolio has `FullCovariance`, `FactorModel`, `DiagonalCovariance`, `WorstCaseRisk` as composable Cost objects.
5. **Multi-period optimization**: cvxportfolio supports T-period look-ahead via `MultiPeriodOptimization`. Our solver is single-period.

## Empirical Phase A finding (2026-05-06)

Smoke test (`scripts/qp_cvxpy_smoke.py`) on n=8 problem, 100 instances:

- **Δw parity**: max diff 2e-6 (numerical precision) — both solvers solve same problem
- **Speed**: **SLSQP 0.7 ms vs CVXPY 3.2 ms** — SLSQP is **5× faster** at n≤290 scale
- CVXPY's OSQP/CLARABEL backend is faster than SLSQP only at n≥100; below that, parser+compile overhead dominates

**Conclusion**: original Phase A motivation (speed) was wrong. CVXPY's wins at our scale are stability + code clarity, not throughput. Revised plan below.

## Refactor strategy (multi-phase, low-risk)

### Phase A' — cvxpy as INFEASIBILITY FALLBACK (4-6 hours)

Keep SLSQP as default solver. Use cvxpy only when SLSQP fails. Pseudo:

```python
def solve_portfolio_qp(...):
    try:
        sol = _solve_via_slsqp(...)
        if sol.status == "optimal":
            return sol
    except (SlsqpInfeasibilityError, ValueError):
        log.warning("SLSQP infeasible → cvxpy fallback")
    return _solve_via_cvxpy(...)
```

Wins:
- Keeps 5× speed advantage on healthy paths
- Recovers from "Positive directional derivative" failures (cvxpy uses interior-point methods that detect infeasibility cleanly)
- Adds 35 LOC, no architectural change

Cost:
- 5-min implementation
- 30-min regression test (force infeasibility, verify cvxpy fallback recovers)

### Phase A — backend swap (1-2 days)

Replace scipy SLSQP with cvxpy compilation, keep all our domain logic (tax, no-trade band, etc.) unchanged. The QP problem we're solving:

```
maximize  μᵀΔw - λᵀ Δw·Σ·Δw - tx_cost(Δw)
subject to  Σ(w + Δw) ≤ 1 - cash_reserve     (LE)
            Σ(w + Δw) ≥ min_invested_pct      (GE) — new today
            lo ≤ Δw ≤ hi                        (box)
            |Δw| ≥ no_trade_threshold OR Δw=0   (post-solve)
```

is fully convex except the post-solve no-trade band (handled by zero-out).

Implementation:
1. Add `qp_backend = "cvxpy"` config option, default `"slsqp"` (existing).
2. New `_solve_via_cvxpy()` function in `qp_solver.py` that constructs:
   ```python
   import cvxpy as cp
   dw = cp.Variable(n)
   wp = w_current + dw
   objective = cp.Maximize(mu @ dw - lam * cp.quad_form(dw, Σ) - tx_cost_expr(dw))
   constraints = [
       cp.sum(wp) <= 1 - cash_reserve,
       cp.sum(wp) >= min_invested_pct,
       dw >= lo_bounds,
       dw <= hi_bounds,
   ]
   prob = cp.Problem(objective, constraints)
   prob.solve(solver=cp.OSQP)  # or cp.SCS for fallback
   return dw.value
   ```
3. Mirror tx_cost expression from cvxportfolio TransactionCost.compile_to_cvxpy.
4. A/B test SLSQP vs cvxpy on a fixed input set. Verify same Δw within 1e-4.
5. Ship cvxpy as default after parity confirmed.

**Acceptance**: identical Δw on 100 historic bars; ≤ 30% slower or faster on benchmarks; no infeasibility regressions.

### Phase B — Cost class refactor (3-5 days)

Refactor `tasks.py` so each cost component (tax, fixed cost, market impact, holding cost) is a `Cost` subclass with `compile_to_cvxpy(dw, w_plus, **kwargs)`. Pattern matches cvxportfolio.

This makes A/B testing different cost combinations trivial: compose costs at config-time.

### Phase C — risks.py-style covariance abstraction (3-5 days)

Make `FullCovariance`, `FactorModel`, `DiagonalCovariance` first-class. We already do Ledoit-Wolf shrinkage in `ComputeFullSigmaTask`; refactor as a class with `compile_to_cvxpy(w_plus)` returning the quad form.

This unblocks future factor-model risk (vs raw covariance) — Bryan Kelly EAP uses factor exposure betas.

### Phase D — Multi-period optimization (1-2 weeks; optional)

Look at cvxportfolio `MultiPeriodOptimization`. Useful for: (a) splitting orders across multiple bars to reduce impact, (b) anticipating regime change.

Probably YAGNI for now. Defer until single-period cvxpy is shipped.

## Acceptance gates per phase

- Phase A: SLSQP-cvxpy parity tests on 100 bars + 14 bug fixes pass + sim Sharpe matches.
- Phase B: each Cost class has dedicated unit test; total cost = sum of parts.
- Phase C: risk model A/B (Full vs Factor vs Diagonal).
- Phase D: walk-forward shows ≥ +0.5 pt APY from multi-period vs single.

## Order of operations

Phase A first because:
1. Highest ROI (cvxpy stability + speed)
2. Existing infeasibility issues we've patched today
3. Low blast radius — only replaces solver inside one Task

After Phase A ships and A/B passes for 2 weeks → Phase B refactor (clean abstraction; not blocking).

C/D as needed post-B.

## Estimated wallclock

- Phase A: 1-2 days incl. tests
- Phase B: 3-5 days
- Phase C: 3-5 days
- Phase D: 1-2 weeks (only if needed)

Total: ~2-4 weeks for A+B+C; D opportunistic.
