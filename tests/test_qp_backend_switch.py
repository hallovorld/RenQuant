"""Tests for the qp_solver_backend config switch in SolveMarkowitzQPTask.

Verifies that the orchestrator routes to the correct backend (cvxpy vs
cvxportfolio.SinglePeriodOpt) based on `rotation.joint_actions.qp_solver_backend`.

Pins:
  1. Default (no config) → cvxpy backend.
  2. "cvxpy" → cvxpy backend.
  3. "cvxportfolio" → cvxportfolio.SinglePeriodOpt backend (diagnostics
      tagged "backend":"cvxportfolio").
  4. Backend switch only changes the solver call, not upstream Tasks.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _make_ctx(n: int = 5, backend: str = "cvxpy") -> SimpleNamespace:
    """Build a minimal ctx that satisfies SolveMarkowitzQPTask's reads."""
    ctx = SimpleNamespace()
    ctx.config = {
        "rotation": {"joint_actions": {
            "qp_solver_backend": backend,
            "qp_risk_aversion": 3.0,
            "qp_cost_kappa":    0.0001,
            "qp_min_invested_pct": 0.0,
            "qp_cash_drag_lambda": 0.0,
        }},
    }
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T / n + 1e-3 * np.eye(n)
    ctx._qp_w_current     = np.zeros(n)
    ctx._qp_mu            = rng.normal(scale=0.005, size=n)
    ctx._qp_sigma         = None
    ctx._qp_Sigma_full    = Sigma
    ctx._qp_cash_reserve  = 0.05
    ctx._qp_w_upper       = np.full(n, 0.20)
    ctx._qp_w_lower       = 0.0
    ctx._qp_dw_max        = np.full(n, 0.50)
    ctx._qp_wash_mask     = None
    ctx._qp_drawdown      = 0.0
    ctx._qp_drawdown_limit = 0.20
    ctx._qp_tax_cost      = None
    ctx._qp_turnover_max  = None
    ctx._qp_v_daily_dollar = None
    ctx.portfolio_value   = 100_000.0
    ctx._qp_tickers       = [f"T{i}" for i in range(n)]
    return ctx


class TestBackendSwitch:
    def test_default_uses_cvxpy(self):
        from kernel.portfolio_qp.tasks import SolveMarkowitzQPTask
        ctx = _make_ctx()
        # Remove the explicit backend key → default ("cvxpy") should kick in
        del ctx.config["rotation"]["joint_actions"]["qp_solver_backend"]
        SolveMarkowitzQPTask().run(ctx)
        # cvxpy backend doesn't tag diagnostics["backend"]; cvxportfolio does
        assert ctx._qp_solution.diagnostics.get("backend") != "cvxportfolio"

    def test_cvxpy_explicit(self):
        from kernel.portfolio_qp.tasks import SolveMarkowitzQPTask
        ctx = _make_ctx(backend="cvxpy")
        SolveMarkowitzQPTask().run(ctx)
        assert ctx._qp_solution.diagnostics.get("backend") != "cvxportfolio"
        assert ctx._qp_solution.status in ("optimal", "optimal_no_signal")

    def test_cvxportfolio_route(self):
        from kernel.portfolio_qp.tasks import SolveMarkowitzQPTask
        ctx = _make_ctx(backend="cvxportfolio")
        SolveMarkowitzQPTask().run(ctx)
        assert ctx._qp_solution.diagnostics.get("backend") == "cvxportfolio"
        assert ctx._qp_solution.status in ("optimal", "optimal_no_signal")
        assert ctx._qp_solution.diagnostics.get("policy_class") == (
            "cvxportfolio.SinglePeriodOpt"
        )

    def test_case_insensitive(self):
        """Config string is normalised to lowercase — `CVXPortfolio`,
        `CVXPY`, etc. all route correctly."""
        from kernel.portfolio_qp.tasks import SolveMarkowitzQPTask
        ctx = _make_ctx(backend="CVXPortfolio")
        SolveMarkowitzQPTask().run(ctx)
        assert ctx._qp_solution.diagnostics.get("backend") == "cvxportfolio"

    def test_unknown_backend_falls_back_to_cvxpy(self):
        """Unknown backend strings fall back to cvxpy — defensive default
        rather than crashing the live runner with a ValueError."""
        from kernel.portfolio_qp.tasks import SolveMarkowitzQPTask
        ctx = _make_ctx(backend="some_typo_qp")
        SolveMarkowitzQPTask().run(ctx)
        assert ctx._qp_solution.diagnostics.get("backend") != "cvxportfolio"
        assert ctx._qp_solution.status in ("optimal", "optimal_no_signal")
