"""Multi-bar QP acceptance tests — Garleanu-Pedersen turnover-aware ramp.

Pins the canonical day-by-day deployment behaviour from cash to target
that the 2026-05-06 cvxpy refactor is supposed to deliver. These tests
were the missing acceptance gate that allowed the V4/V5 0-trade
infeasibility to ship without a regression test for *the actual
production failure mode* (multi-bar accumulation toward target).

Each test simulates ≥3 sequential QP calls, feeding the previous bar's
target_w as the next bar's w_current. A solver that handles the soft
cash-drag penalty correctly will:

  Bar 1 (from cash):     deploy up to turnover_max
  Bar 2 (mid-ramp):      deploy more, approaching target
  Bar 3 (near target):   small Δw to close the gap

References (read prior to test design):
- Garleanu & Pedersen 2013 §4 "Dynamic Trading with Predictable Returns
  and Transaction Costs" — formal derivation of the partial-move ramp
  under linear cost and quadratic risk.
- cvxportfolio 1.5 (Boyd/Stanford) `SinglePeriodOpt` — the reference
  policy follows the same per-bar greedy convex solve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _make_problem(n: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic mu/Sigma pair representative of the alpha158_linear
    inference output (μ in ±0.5%, σ ≈ 0.04, 50 candidates)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T / n + 1e-3 * np.eye(n)
    mu = rng.normal(scale=0.005, size=n)
    return mu, Sigma


class TestRampFromCash:
    """The actual V5 production failure mode — 50 candidates, w_upper=0.075,
    min_invested_pct=0.7, turnover_max=0.30 — must produce a 3-bar ramp
    rather than 0 trades."""

    def test_three_bar_ramp_reaches_target(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

        n = 50
        mu, Sigma = _make_problem(n=n, seed=1)

        common_kw = dict(
            mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_upper=0.075, w_lower=0.0, dw_max=0.50,
            min_invested_pct=0.70, cash_drag_lambda=10.0,   # stiff → push to target
            turnover_max=0.30,
        )

        # Bar 1: from cash
        w0 = np.zeros(n)
        sol1 = solve_portfolio_qp(w_current=w0, **common_kw)
        assert sol1.status == "optimal", f"bar 1 status={sol1.status}"
        bar1_invested = sol1.target_w.sum()
        assert bar1_invested >= 0.27, (
            f"Bar 1 should saturate the 0.30 turnover cap; got "
            f"sum(wp)={bar1_invested:.4f}"
        )
        assert bar1_invested <= 0.31

        # Bar 2: from previous target
        sol2 = solve_portfolio_qp(w_current=sol1.target_w, **common_kw)
        assert sol2.status == "optimal", f"bar 2 status={sol2.status}"
        bar2_invested = sol2.target_w.sum()
        assert bar2_invested > bar1_invested, (
            f"Bar 2 must increase deployment; got bar1={bar1_invested:.4f} "
            f"→ bar2={bar2_invested:.4f}"
        )

        # Bar 3: should reach or surpass target
        sol3 = solve_portfolio_qp(w_current=sol2.target_w, **common_kw)
        assert sol3.status == "optimal", f"bar 3 status={sol3.status}"
        bar3_invested = sol3.target_w.sum()
        assert bar3_invested >= 0.69, (
            f"Bar 3 must reach the 0.7 target; got sum(wp)={bar3_invested:.4f}"
        )

    def test_no_infeasibility_in_first_bar(self):
        """REGRESSION: V4/V5 alpha158_linear sim hit `cvxpy fallback
        status=infeasible` every bar from all-cash with min_invested=0.7
        + turnover=0.30. The new soft-penalty formulation must NEVER
        report `infeasible` from the all-cash starting state regardless
        of how aggressive the target is."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

        n = 50
        mu, Sigma = _make_problem(n=n, seed=2)

        # Aggressively-tight constraints — old SLSQP+fallback infeasible.
        for floor in [0.50, 0.70, 0.90, 1.0]:
            sol = solve_portfolio_qp(
                w_current=np.zeros(n), mu=mu, Sigma=Sigma,
                risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.0,
                w_upper=0.075, w_lower=0.0, dw_max=0.50,
                min_invested_pct=floor, cash_drag_lambda=10.0,
                turnover_max=0.30,
            )
            assert not sol.status.startswith("infeasible"), (
                f"Soft penalty must NEVER infeasible; floor={floor} "
                f"got status={sol.status}"
            )
            assert sol.target_w.sum() >= 0.27, (
                f"Bar must deploy at least the turnover cap; floor={floor} "
                f"got sum(wp)={sol.target_w.sum():.4f}"
            )


class TestRampNotRequired:
    """When constraints aren't tight, the QP should hit the target on the
    first bar — verifies the ramp emerges only from binding turnover, not
    from the soft-penalty formulation itself."""

    def test_unconstrained_one_bar_deployment(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

        n = 20
        mu, Sigma = _make_problem(n=n, seed=3)
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_upper=0.20, w_lower=0.0, dw_max=0.95,
            min_invested_pct=0.70, cash_drag_lambda=10.0,
            turnover_max=1.50,            # generous — non-binding
        )
        assert sol.status == "optimal"
        assert sol.target_w.sum() >= 0.69, (
            f"Without binding turnover, bar 1 should hit target; "
            f"got sum(wp)={sol.target_w.sum():.4f}"
        )


class TestSoftPenaltyTradeoff:
    """The soft penalty must respect the alpha-vs-drag tradeoff that's the
    whole point of the cvxportfolio formulation."""

    def test_negative_signal_does_not_force_deployment(self):
        """When all candidates have NEGATIVE μ and cash_drag_lambda is
        small, the QP should prefer cash. Pre-fix hard floor would force
        deployment regardless of signal — losing money to satisfy a
        constraint, which is the textbook violation of mean-variance."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

        n = 20
        rng = np.random.default_rng(4)
        A = rng.normal(size=(n, n))
        Sigma = A @ A.T / n + 1e-3 * np.eye(n)
        mu = -np.abs(rng.normal(scale=0.05, size=n))   # all negative

        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_upper=0.20, w_lower=0.0, dw_max=0.50,
            min_invested_pct=0.70,
            cash_drag_lambda=0.001,           # weak → signal wins
            turnover_max=1.0,
        )
        assert sol.status == "optimal"
        # Bearish signal + weak cash drag → solver should mostly stay in
        # cash (drag of 0.001 × 0.7 = 7e-4 vs negative mu of |μ|×0.7 ≈
        # 7e-2 → cash wins by 100×).
        assert sol.target_w.sum() < 0.10, (
            f"With weak cash_drag and strongly negative μ, solver should "
            f"prefer cash; got sum(wp)={sol.target_w.sum():.4f}"
        )

    def test_strong_drag_overrides_negative_signal(self):
        """Stiff cash_drag forces deployment even on bearish signal —
        approximates the user-intent of the old hard floor (deploy this
        much regardless of signal). cvxportfolio doc Tab. 5."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

        n = 20
        rng = np.random.default_rng(5)
        A = rng.normal(size=(n, n))
        Sigma = A @ A.T / n + 1e-3 * np.eye(n)
        mu = -np.abs(rng.normal(scale=0.001, size=n))   # mild negative

        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_upper=0.20, w_lower=0.0, dw_max=0.50,
            min_invested_pct=0.70,
            cash_drag_lambda=10.0,            # stiff
            turnover_max=1.0,
        )
        assert sol.status == "optimal"
        assert sol.target_w.sum() >= 0.69, (
            f"Stiff cash_drag must enforce target even on negative μ; "
            f"got sum(wp)={sol.target_w.sum():.4f}"
        )
