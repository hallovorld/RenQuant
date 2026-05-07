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

    def test_top_level_solver_integrates_fallback(self):
        """solve_portfolio_qp's full path can invoke fallback path.

        Force it via the parameter combination that historically broke
        SLSQP: empty portfolio (w_current=0) + min_invested_pct=0.7.
        Should return optimal status (either via SLSQP success after
        warm-start, or via cvxpy fallback)."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

        n = 8
        mu, Sigma = self._basic_problem(n=n, seed=3)
        sol = solve_portfolio_qp(
            w_current=np.zeros(n), mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_upper=0.20, w_lower=0.0, dw_max=0.50,
            min_invested_pct=0.7,
        )
        assert sol.status in ("optimal", "optimal_no_signal"), \
            f"unexpected status: {sol.status}"
        # Either SLSQP succeeded with warm-start, or cvxpy stepped in.
        # Either way, sum(target_w) should be ≥ 0.7 - tolerance.
        assert sol.target_w.sum() >= 0.69, \
            f"min_invested floor violated: sum(wp)={sol.target_w.sum():.4f}"

    def test_fallback_returns_none_when_cvxpy_unavailable(self):
        """Graceful degradation if cvxpy module is missing."""
        # Block cvxpy import inside _solve_via_cvxpy_fallback
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "cvxpy":
                raise ImportError("Mocked: cvxpy not available")
            return real_import(name, *args, **kwargs)

        from kernel.portfolio_qp.qp_solver import _solve_via_cvxpy_fallback
        with patch.object(builtins, "__import__", side_effect=fake_import):
            result = _solve_via_cvxpy_fallback(
                w_current=np.zeros(8), mu=np.zeros(8),
                Sigma=np.eye(8) * 1e-3,
                risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
                w_lower_arr=np.zeros(8), w_upper_arr=np.full(8, 0.2),
                dw_max_arr=np.full(8, 0.5),
            )
            assert result is None, \
                "fallback should return None when cvxpy not installed"

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
