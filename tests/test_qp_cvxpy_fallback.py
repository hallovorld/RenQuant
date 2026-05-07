"""Regression tests for cvxpy fallback in solve_portfolio_qp (Phase A').

Closes the 2026-05-06 SLSQP "Positive directional derivative for
linesearch" failure mode by routing infeasibility cases to cvxpy +
CLARABEL/OSQP. Per CLAUDE.md §2: every bug fix ships with a regression
test that would fail before the fix.

Reference: doc/research/qp-cvxportfolio-refactor-plan.md Phase A'.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


class TestCvxpyFallback:
    """Phase A': cvxpy is invoked iff SLSQP fails on infeasibility."""

    def _basic_problem(self, n: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(n, n))
        Sigma = A @ A.T / n + 1e-3 * np.eye(n)
        mu = rng.normal(scale=0.05, size=n)
        return mu, Sigma

    def test_fallback_helper_solves_basic_problem(self):
        """_solve_via_cvxpy_fallback alone returns valid Δw."""
        from kernel.portfolio_qp.qp_solver import _solve_via_cvxpy_fallback

        n = 8
        mu, Sigma = self._basic_problem(n=n)
        w_current = np.zeros(n)
        dw = _solve_via_cvxpy_fallback(
            w_current=w_current, mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_lower_arr=np.zeros(n),
            w_upper_arr=np.full(n, 0.20),
            dw_max_arr=np.full(n, 0.50),
        )
        assert dw is not None, "cvxpy fallback returned None on solvable problem"
        assert dw.shape == (n,)
        assert np.all(np.isfinite(dw))
        # Constraints: cash reserve 0.05 → sum(w_current + dw) ≤ 0.95
        assert (w_current + dw).sum() <= 0.95 + 1e-6
        # Box: 0 ≤ w_current + dw ≤ 0.20
        assert ((w_current + dw) >= -1e-6).all()
        assert ((w_current + dw) <= 0.20 + 1e-6).all()

    def test_fallback_respects_min_invested_pct(self):
        """min_invested_pct hard floor is honored by cvxpy fallback."""
        from kernel.portfolio_qp.qp_solver import _solve_via_cvxpy_fallback

        n = 8
        mu, Sigma = self._basic_problem(n=n, seed=1)
        w_current = np.zeros(n)
        dw = _solve_via_cvxpy_fallback(
            w_current=w_current, mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_lower_arr=np.zeros(n),
            w_upper_arr=np.full(n, 0.20),
            dw_max_arr=np.full(n, 0.50),
            min_invested_pct=0.7,
        )
        assert dw is not None
        # Σ(w_current + dw) ≥ 0.7 (within numerical tolerance)
        assert (w_current + dw).sum() >= 0.70 - 1e-4

    def test_fallback_respects_turnover_cap(self):
        """turnover_max=0.2 → ∑|Δw| ≤ 0.2 in cvxpy solution."""
        from kernel.portfolio_qp.qp_solver import _solve_via_cvxpy_fallback

        n = 8
        mu, Sigma = self._basic_problem(n=n, seed=2)
        w_current = np.full(n, 0.05)
        dw = _solve_via_cvxpy_fallback(
            w_current=w_current, mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_lower_arr=np.zeros(n),
            w_upper_arr=np.full(n, 0.20),
            dw_max_arr=np.full(n, 0.50),
            turnover_max=0.2,
        )
        assert dw is not None
        assert np.abs(dw).sum() <= 0.20 + 1e-4

    def test_top_level_solver_with_high_cash_drag_deploys_to_target(self):
        """2026-05-06 refactor: min_invested_pct is now a SOFT target
        driven by `cash_drag_lambda`. With λ=10 (stiff), the soft penalty
        approximates the old hard floor — solver should deploy to ≥ target.

        Pre-2026-05-06 hard-floor test had a bug: it relied on SLSQP's
        slack relaxation to satisfy infeasible problems. New convex
        formulation makes that explicit (penalty grows linearly with the
        gap)."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

        n = 8
        mu, Sigma = self._basic_problem(n=n, seed=3)
        sol = solve_portfolio_qp(
            w_current=np.zeros(n), mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_upper=0.20, w_lower=0.0, dw_max=0.50,
            min_invested_pct=0.7,
            cash_drag_lambda=10.0,        # stiff penalty → behaves like hard floor
        )
        assert sol.status in ("optimal", "optimal_no_signal"), \
            f"unexpected status: {sol.status}"
        # With λ_cash=10, penalty for being below 0.7 is 10×(0.7 - Σwp);
        # only worth it if marginal μ < -10/n ≈ -1.25 per asset, which
        # never happens in our μ ~ N(0, 0.05).
        assert sol.target_w.sum() >= 0.69, \
            f"high cash_drag_lambda should approximate hard floor; "\
            f"got sum(wp)={sol.target_w.sum():.4f}"

    def test_top_level_solver_with_default_cash_drag_partial_deploy(self):
        """Default `cash_drag_lambda=0.05` is moderate — solver deploys
        when net signal beats the drag, stays partial otherwise.

        This is the cvxportfolio-textbook tradeoff: cash drag is a soft
        preference, not a hard rule. Expressing intent as a coefficient
        in the objective makes the trade-off explicit and tunable."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

        n = 8
        mu, Sigma = self._basic_problem(n=n, seed=3)
        sol = solve_portfolio_qp(
            w_current=np.zeros(n), mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_upper=0.20, w_lower=0.0, dw_max=0.50,
            min_invested_pct=0.7,
            # cash_drag_lambda defaults to 0.05
        )
        assert sol.status in ("optimal", "optimal_no_signal"), \
            f"unexpected status: {sol.status}"
        # Default penalty is moderate; some deployment but not the full
        # 0.7 target unless μ is strongly positive on average.
        assert 0.0 < sol.target_w.sum() < 0.7 + 1e-6, (
            f"Default cash_drag should enable partial deployment; "
            f"got sum(wp)={sol.target_w.sum():.4f}"
        )

    def test_solver_requires_cvxpy(self):
        """Post-2026-05-06: `solve_portfolio_qp` REQUIRES cvxpy. The old
        scipy.optimize.SLSQP path is gone. If cvxpy can't be imported,
        an ImportError propagates — there is no fallback to a non-convex
        scipy solver. (Industry-standard convex QP only.)"""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "cvxpy":
                raise ImportError("Mocked: cvxpy not available")
            return real_import(name, *args, **kwargs)

        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        with patch.object(builtins, "__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="cvxpy"):
                solve_portfolio_qp(
                    w_current=np.zeros(8), mu=np.zeros(8),
                    Sigma=np.eye(8) * 1e-3,
                    risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
                    w_upper=0.2, w_lower=0.0, dw_max=0.5,
                )

    def test_fallback_clamps_turnover_infeasible_floor(self):
        """REGRESSION (2026-05-06 V5 sim 0-trade): from-cash with
        min_invested_pct=0.7 + turnover_max=0.3 is mathematically
        infeasible — needs 70% turnover to satisfy floor, only 30%
        allowed. CLARABEL correctly reported infeasible until the
        turnover clamp landed.

        With the turnover clamp: floor = max(0, sum(w_current) +
        turnover_max - 0.01) = 0 + 0.30 - 0.01 = 0.29. Solver
        successfully invests up to 29% in one bar; subsequent bars
        ratchet up to the original 0.7 target."""
        from kernel.portfolio_qp.qp_solver import (
            _solve_via_cvxpy_fallback, _clamp_min_invested_floor,
        )

        n = 50   # capacity not the binding constraint
        mu, Sigma = self._basic_problem(n=n, seed=5)
        w_current = np.zeros(n)
        # Verify clamp helper picks the turnover clamp:
        floor, reason = _clamp_min_invested_floor(
            min_invested_pct=0.70,
            w_current=w_current,
            cash_reserve=0.05,
            hi_bounds=np.full(n, 0.075),     # 50 × 0.075 = 3.75 capacity (not binding)
            turnover_max=0.30,
        )
        assert reason == "turnover", (
            f"Expected turnover-clamp; got reason={reason!r}, floor={floor:.4f}"
        )
        assert 0.28 < floor < 0.30, f"Clamp floor {floor:.4f} not in expected range"

        # End-to-end: cvxpy fallback now succeeds where it used to fail.
        dw = _solve_via_cvxpy_fallback(
            w_current=w_current, mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_lower_arr=np.zeros(n),
            w_upper_arr=np.full(n, 0.075),
            dw_max_arr=np.full(n, 0.50),
            min_invested_pct=0.70,            # > 0.30 turnover cap
            turnover_max=0.30,
        )
        assert dw is not None, (
            "cvxpy fallback returned None — turnover clamp did not engage; "
            "from-cash + min_invested=0.7 + turnover=0.3 still infeasible."
        )
        wp = w_current + dw
        # Solution respects turnover cap and the relaxed (turnover) floor.
        assert np.abs(dw).sum() <= 0.30 + 1e-4, (
            f"turnover_max=0.30 violated: Σ|dw|={np.abs(dw).sum():.4f}"
        )
        assert wp.sum() >= 0.27, (
            f"After turnover-clamp the QP should still allocate ≈0.29; "
            f"got sum(wp)={wp.sum():.4f}"
        )

    def test_fallback_clamps_capacity_infeasible_floor(self):
        """REGRESSION (2026-05-06): when sum(per-asset hi-bounds) <
        min_invested_pct, the cvxpy fallback used to forward the raw floor
        unchanged → CLARABEL reported `infeasible` → V4 alpha158_linear sim
        produced 0 trades over 128 days. The SLSQP path applied a capacity
        clamp at lines 340-358, but the cvxpy path did not. Fix: shared
        `_clamp_min_invested_floor` helper applied in both paths.

        Construct a from-cash problem where 4 candidates × 0.15 cap = 0.60
        capacity, but min_invested_pct=0.70 — without the clamp the QP is
        infeasible; with the clamp the floor relaxes to 0.59 (capacity − ε)
        and CLARABEL solves cleanly."""
        from kernel.portfolio_qp.qp_solver import (
            _solve_via_cvxpy_fallback, _clamp_min_invested_floor,
        )

        n = 4   # 4 candidates × 0.15 = 0.60 capacity
        mu, Sigma = self._basic_problem(n=n, seed=4)
        w_current = np.zeros(n)
        # Verify clamp helper detects the capacity infeasibility:
        floor, reason = _clamp_min_invested_floor(
            min_invested_pct=0.70,
            w_current=w_current,
            cash_reserve=0.05,
            hi_bounds=np.full(n, 0.15),
        )
        assert reason == "capacity", (
            f"Expected capacity-clamp; got reason={reason!r}, floor={floor:.4f}"
        )
        assert 0.55 < floor < 0.60, f"Clamp floor {floor:.4f} not in expected range"

        # End-to-end: cvxpy fallback now succeeds where it used to fail.
        dw = _solve_via_cvxpy_fallback(
            w_current=w_current, mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_lower_arr=np.zeros(n),
            w_upper_arr=np.full(n, 0.15),       # tight per-asset cap
            dw_max_arr=np.full(n, 0.50),
            min_invested_pct=0.70,              # > 0.60 capacity
        )
        assert dw is not None, (
            "cvxpy fallback returned None — capacity clamp did not engage; "
            "this means a from-cash V4-shape problem still infeasible."
        )
        # Solution must respect the per-asset cap and the relaxed floor.
        wp = w_current + dw
        assert (wp <= 0.15 + 1e-4).all()
        assert wp.sum() >= 0.55, (
            f"After capacity-clamp the QP should still allocate ≈0.59; "
            f"got sum(wp)={wp.sum():.4f}"
        )

    def test_fallback_parity_with_slsqp_on_easy_problem(self):
        """On a problem both solvers can solve, results agree to 1e-3."""
        from kernel.portfolio_qp.qp_solver import (
            _solve_via_cvxpy_fallback, solve_portfolio_qp,
        )

        n = 5
        rng = np.random.default_rng(42)
        mu = rng.normal(scale=0.05, size=n)
        Sigma = np.diag(rng.uniform(0.001, 0.01, n))
        w_current = rng.uniform(0, 0.05, size=n)

        slsqp_sol = solve_portfolio_qp(
            w_current=w_current, mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_upper=0.20, w_lower=0.0, dw_max=0.50,
        )
        assert slsqp_sol.status == "optimal"

        cvx_dw = _solve_via_cvxpy_fallback(
            w_current=w_current, mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_lower_arr=np.zeros(n),
            w_upper_arr=np.full(n, 0.20),
            dw_max_arr=np.full(n, 0.50),
        )
        diff = np.abs(slsqp_sol.delta_w - cvx_dw).max()
        assert diff < 1e-3, f"SLSQP/cvxpy disagree by {diff:.6f}"
