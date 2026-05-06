#!/usr/bin/env python
"""Smoke-test cvxpy as drop-in replacement for SLSQP in QP solver.

Goal: verify cvxpy + OSQP/CLARABEL gives same Δw as scipy SLSQP on a
small problem. If parity holds, justify Phase A backend swap.

Reference pattern from cvxportfolio (costs.py:TransactionCost.compile_to_cvxpy):
  - Use cp.Variable for trade weights
  - Use cp.quad_form for variance
  - Use cp.abs for L1 cost
  - cp.Problem(cp.Maximize(...), [constraints]).solve(solver=cp.CLARABEL)

Usage: python scripts/qp_cvxpy_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import cvxpy as cp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))
from kernel.portfolio_qp.qp_solver import solve_portfolio_qp  # noqa: E402


def cvxpy_solve(w_current, mu, Sigma, *,
                risk_aversion=3.0, cost_kappa=0.0001,
                cash_reserve=0.05, w_upper=0.20, w_lower=0.0,
                dw_max=0.50, min_invested_pct=0.0,
                solver=cp.CLARABEL):
    n = len(w_current)
    w_current = np.asarray(w_current, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    dw = cp.Variable(n)
    wp = w_current + dw

    # Markowitz: maximize mu' wp - gamma * wp' Σ wp
    objective = mu @ wp - risk_aversion * cp.quad_form(wp, cp.psd_wrap(Sigma))
    # Linear cost: -kappa * |dw|_1
    if cost_kappa > 0:
        objective = objective - cost_kappa * cp.norm(dw, 1)
    prob_obj = cp.Maximize(objective)

    # Box per position
    if np.isscalar(w_upper):
        w_upper = np.full(n, w_upper)
    if np.isscalar(w_lower):
        w_lower = np.full(n, w_lower)
    if np.isscalar(dw_max):
        dw_max_arr = np.full(n, dw_max)
    else:
        dw_max_arr = np.asarray(dw_max)

    constraints = [
        cp.sum(wp) <= 1.0 - cash_reserve,
        wp >= w_lower,
        wp <= w_upper,
        dw >= -dw_max_arr,
        dw <= dw_max_arr,
    ]
    if min_invested_pct > 0:
        constraints.append(cp.sum(wp) >= min_invested_pct)

    problem = cp.Problem(prob_obj, constraints)
    problem.solve(solver=solver, verbose=False)

    return {
        "delta_w": dw.value if dw.value is not None else np.zeros(n),
        "objective": float(problem.value) if problem.value is not None else float("nan"),
        "status": problem.status,
        "n_iter": problem.solver_stats.num_iters if problem.solver_stats else -1,
    }


def main() -> None:
    rng = np.random.default_rng(42)
    n = 8

    # Random PSD covariance
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T / n + 1e-3 * np.eye(n)
    mu = rng.normal(scale=0.05, size=n)
    w_current = rng.uniform(0, 0.10, size=n)
    w_current[-1] = 0.0  # leave some space

    print(f"=== Setup: n={n} ===")
    print(f"  mu mean={mu.mean():+.4f}  std={mu.std():.4f}")
    print(f"  Σ trace/n={np.trace(Sigma)/n:.6f}  cond={np.linalg.cond(Sigma):.1f}")
    print(f"  w_current sum={w_current.sum():.4f}")

    common_args = dict(
        risk_aversion=3.0, cost_kappa=0.0001,
        cash_reserve=0.05, w_upper=0.20, w_lower=0.0, dw_max=0.50,
    )

    # SLSQP
    t0 = time.time()
    sl = solve_portfolio_qp(
        w_current=w_current, mu=mu, Sigma=Sigma, **common_args
    )
    sl_time = time.time() - t0
    print()
    print(f"=== SLSQP ===")
    print(f"  status     = {sl.status}")
    print(f"  delta_w    = {sl.delta_w}")
    print(f"  objective  = {sl.objective:+.6f}")
    print(f"  n_iter     = {sl.n_iter}")
    print(f"  wallclock  = {sl_time*1000:.1f} ms")

    # CVXPY
    t0 = time.time()
    cvx = cvxpy_solve(w_current, mu, Sigma, **common_args)
    cvx_time = time.time() - t0
    print()
    print(f"=== CVXPY (CLARABEL) ===")
    print(f"  status     = {cvx['status']}")
    print(f"  delta_w    = {cvx['delta_w']}")
    print(f"  objective  = {cvx['objective']:+.6f}")
    print(f"  n_iter     = {cvx['n_iter']}")
    print(f"  wallclock  = {cvx_time*1000:.1f} ms")

    # Comparison
    print()
    diff = sl.delta_w - cvx['delta_w']
    print(f"=== Δw diff: max={np.max(np.abs(diff)):.6f}  l2={np.linalg.norm(diff):.6f} ===")

    # Now test min_invested_pct (Davis-Norman cash-drag fix)
    print("\n=== With min_invested_pct=0.7 ===")
    sl2 = solve_portfolio_qp(
        w_current=w_current, mu=mu, Sigma=Sigma, min_invested_pct=0.7, **common_args
    )
    cvx2 = cvxpy_solve(w_current, mu, Sigma, min_invested_pct=0.7, **common_args)
    print(f"  SLSQP    sum(wp) = {(w_current + sl2.delta_w).sum():.4f}  status={sl2.status}")
    print(f"  CVXPY    sum(wp) = {(w_current + cvx2['delta_w']).sum():.4f}  status={cvx2['status']}")

    # Speed test
    print("\n=== Speed test on 100 random problems ===")
    sl_total = 0.0
    cvx_total = 0.0
    for i in range(100):
        rng2 = np.random.default_rng(i)
        A = rng2.normal(size=(n, n))
        Sigma = A @ A.T / n + 1e-3 * np.eye(n)
        mu = rng2.normal(scale=0.05, size=n)
        w_current = rng2.uniform(0, 0.10, size=n)

        t0 = time.time()
        solve_portfolio_qp(w_current=w_current, mu=mu, Sigma=Sigma, **common_args)
        sl_total += time.time() - t0

        t0 = time.time()
        cvxpy_solve(w_current, mu, Sigma, **common_args)
        cvx_total += time.time() - t0

    print(f"  SLSQP : {sl_total*10:.1f} ms/problem")
    print(f"  CVXPY : {cvx_total*10:.1f} ms/problem")
    print(f"  speedup = {sl_total/cvx_total:.2f}× ({'cvxpy faster' if cvx_total < sl_total else 'slsqp faster'})")


if __name__ == "__main__":
    main()
