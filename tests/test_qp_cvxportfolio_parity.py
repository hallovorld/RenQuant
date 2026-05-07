"""Parity tests — cvxpy primary vs cvxportfolio.SinglePeriodOpt backend.

Verifies the two backends produce equivalent target weights to within
solver tolerance on representative inputs. The cvxportfolio backend is
the "industry-standard reference framework" path (Boyd/Stanford); the
cvxpy backend is our hand-rolled implementation that follows the same
mathematical formulation. They should agree.

References:
- Boyd-Busseti-Diamond-Kahn 2017 — single-period MV objective is well-
  defined enough that two correct implementations produce identical
  optima up to solver tolerance. Disagreement > 1% indicates one
  implementation has a bug.
- cvxportfolio 1.5 `policies.SinglePeriodOpt.execute(market_data=None)`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _make_problem(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T / n + 1e-3 * np.eye(n)
    mu = rng.normal(scale=0.005, size=n)
    return mu, Sigma


class TestParityBasicMV:
    """Plain Markowitz — no impact, no tax, no soft-floor — should match
    to within 1% per asset."""

    def test_basic_unconstrained_universe(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        from kernel.portfolio_qp.cvxportfolio_backend import (
            solve_portfolio_qp_cvxportfolio,
        )
        n = 10
        mu, Sigma = _make_problem(n=n, seed=1)

        common_kw = dict(
            w_current=np.full(n, 0.05), mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_upper=0.20, w_lower=0.0, dw_max=0.50,
        )
        cvxpy_sol = solve_portfolio_qp(**common_kw)
        cvxp_sol  = solve_portfolio_qp_cvxportfolio(**common_kw)

        assert cvxpy_sol.status == "optimal"
        assert cvxp_sol.status == "optimal"
        diff = np.abs(cvxpy_sol.target_w - cvxp_sol.target_w).max()
        assert diff < 0.01, (
            f"Backends disagree on basic MV by {diff:.4f}. "
            f"cvxpy target_w: {cvxpy_sol.target_w}\n"
            f"cvxportfolio target_w: {cvxp_sol.target_w}"
        )

    def test_per_asset_cap_enforced_both_backends(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        from kernel.portfolio_qp.cvxportfolio_backend import (
            solve_portfolio_qp_cvxportfolio,
        )
        n = 5
        mu, Sigma = _make_problem(n=n, seed=2)
        # Tight per-asset cap; check both solvers respect it
        common_kw = dict(
            w_current=np.zeros(n), mu=mu, Sigma=Sigma,
            risk_aversion=1.0, cost_kappa=0.0, cash_reserve=0.0,
            w_upper=0.15, w_lower=0.0, dw_max=0.50,
        )
        cvxpy_sol = solve_portfolio_qp(**common_kw)
        cvxp_sol  = solve_portfolio_qp_cvxportfolio(**common_kw)
        assert (cvxpy_sol.target_w <= 0.15 + 1e-4).all()
        assert (cvxp_sol.target_w  <= 0.15 + 1e-4).all()
        # Long-only for both
        assert (cvxpy_sol.target_w >= -1e-6).all()
        assert (cvxp_sol.target_w  >= -1e-6).all()

    def test_turnover_limit_respected_both_backends(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        from kernel.portfolio_qp.cvxportfolio_backend import (
            solve_portfolio_qp_cvxportfolio,
        )
        n = 10
        mu, Sigma = _make_problem(n=n, seed=3)
        common_kw = dict(
            w_current=np.zeros(n), mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_upper=0.30, w_lower=0.0, dw_max=0.50,
            turnover_max=0.20,
        )
        cvxpy_sol = solve_portfolio_qp(**common_kw)
        cvxp_sol  = solve_portfolio_qp_cvxportfolio(**common_kw)
        assert np.abs(cvxpy_sol.delta_w).sum() <= 0.20 + 1e-3
        assert np.abs(cvxp_sol.delta_w).sum()  <= 0.20 + 1e-3


class TestBackendIdentity:
    """The cvxportfolio backend should expose `backend=cvxportfolio` in
    diagnostics so the orchestrator can branch on it."""

    def test_diagnostics_identifies_backend(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        from kernel.portfolio_qp.cvxportfolio_backend import (
            solve_portfolio_qp_cvxportfolio,
        )
        n = 5
        mu, Sigma = _make_problem(n=n, seed=4)
        cvxpy_sol = solve_portfolio_qp(
            w_current=np.zeros(n), mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.0,
            w_upper=0.20, w_lower=0.0, dw_max=0.50,
        )
        cvxp_sol = solve_portfolio_qp_cvxportfolio(
            w_current=np.zeros(n), mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.0,
            w_upper=0.20, w_lower=0.0, dw_max=0.50,
        )
        # cvxpy backend doesn't tag, cvxportfolio backend does
        assert cvxp_sol.diagnostics["backend"] == "cvxportfolio"
        assert cvxp_sol.diagnostics["policy_class"] == (
            "cvxportfolio.SinglePeriodOpt"
        )
        # Common diagnostics shape
        for key in ("n_assets", "risk_aversion", "actual_turnover"):
            assert key in cvxp_sol.diagnostics
            assert key in cvxpy_sol.diagnostics


class TestV8LeverageBugFix:
    """REGRESSION (2026-05-07): V8 sim of cvxportfolio backend produced
    pathological leverage blowup (max DD 557%, APY 1276%, ann vol 1403%)
    over 128 days. Two structural bugs in the constraint set:

      1. cvxportfolio.TurnoverLimit(delta) constrains ½‖z‖₁ ≤ delta,
         not ‖z‖₁ ≤ delta. Passing turnover_max=0.30 doubled effective
         turnover to 0.60 per bar.

      2. cvxportfolio.LongOnly(applies_to_cash=False) (the default for
         our backend) lets cash go NEGATIVE = margin loan. Combined
         with bug 1, positions could compound with leverage.

    Fix: pass turnover_max/2 to TurnoverLimit, switch to
    LongOnly(applies_to_cash=True). These tests pin both fixes."""

    def test_turnover_bounded_by_passed_value_not_double(self):
        """Σ|delta_w| ≤ turnover_max (not 2x)."""
        from kernel.portfolio_qp.cvxportfolio_backend import (
            solve_portfolio_qp_cvxportfolio,
        )
        n = 50
        rng = np.random.default_rng(7)
        A = rng.normal(size=(n, n))
        Sigma = A @ A.T / n + 1e-3 * np.eye(n)
        mu = np.ones(n) * 0.05    # all-positive → saturate constraints

        sol = solve_portfolio_qp_cvxportfolio(
            w_current=np.zeros(n), mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
            w_upper=0.075, w_lower=0.0, dw_max=0.075,
            turnover_max=0.30,
        )
        assert sol.status == "optimal"
        # Pre-fix: this was 0.60 (the doubling bug). Post-fix: ≤ 0.30.
        assert np.abs(sol.delta_w).sum() <= 0.301, (
            f"V8 turnover-doubling bug regressed: Σ|delta_w| = "
            f"{np.abs(sol.delta_w).sum():.4f} > 0.30"
        )

    def test_no_negative_cash_under_aggressive_signal(self):
        """LongOnly with applies_to_cash=True prevents margin / leverage.

        With strong all-positive μ, cash_drag, and tight per-asset cap,
        the QP wants to deploy aggressively. The pre-fix backend let
        cash go negative (Σwp_stocks > 1). Post-fix: cash ≥ 0, so
        Σwp_stocks ≤ 1 - cash_reserve."""
        from kernel.portfolio_qp.cvxportfolio_backend import (
            solve_portfolio_qp_cvxportfolio,
        )
        n = 50
        rng = np.random.default_rng(8)
        A = rng.normal(size=(n, n))
        Sigma = A @ A.T / n + 1e-3 * np.eye(n)
        mu = np.abs(rng.normal(scale=0.05, size=n))  # all-positive bear

        # Run 6 bars; sum_w_stocks must never exceed (1 - cash_reserve)
        w_current = np.zeros(n)
        for bar in range(6):
            sol = solve_portfolio_qp_cvxportfolio(
                w_current=w_current, mu=mu, Sigma=Sigma,
                risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
                w_upper=0.075, w_lower=0.0, dw_max=0.10,
                turnover_max=0.30,
            )
            assert sol.status == "optimal", f"bar {bar} status={sol.status}"
            sum_stocks = sol.target_w.sum()
            assert sum_stocks <= 0.951, (
                f"bar {bar}: Σwp_stocks = {sum_stocks:.4f} > 0.95 "
                f"(implies negative cash, leverage violation)"
            )
            assert (sol.target_w >= -1e-6).all(), (
                f"bar {bar}: target_w has negative entry"
            )
            w_current = sol.target_w

    def test_garleanu_pedersen_ramp_emerges(self):
        """With turnover binding, the canonical 3-bar Garleanu-Pedersen
        ramp emerges: 30% → 60% → 90% → 95% from cash. This test pins
        the qualitative behaviour (post-V8 fix) and parallels the
        cvxpy backend's identical multi-bar test."""
        from kernel.portfolio_qp.cvxportfolio_backend import (
            solve_portfolio_qp_cvxportfolio,
        )
        n = 50
        rng = np.random.default_rng(9)
        A = rng.normal(size=(n, n))
        Sigma = A @ A.T / n + 1e-3 * np.eye(n)
        mu = np.ones(n) * 0.05

        w = np.zeros(n)
        sums = []
        for _ in range(4):
            sol = solve_portfolio_qp_cvxportfolio(
                w_current=w, mu=mu, Sigma=Sigma,
                risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.05,
                w_upper=0.075, w_lower=0.0, dw_max=0.075,
                turnover_max=0.30,
            )
            sums.append(sol.target_w.sum())
            w = sol.target_w

        # Bar 1: ~30% (turnover-saturated from cash)
        assert 0.27 <= sums[0] <= 0.31, f"bar 1 sum={sums[0]:.4f}"
        # Bar 2: ~60% (another 30% turnover)
        assert 0.55 <= sums[1] <= 0.61, f"bar 2 sum={sums[1]:.4f}"
        # Bar 3: ~90% (90% < 95% cap)
        assert 0.85 <= sums[2] <= 0.91, f"bar 3 sum={sums[2]:.4f}"
        # Bar 4: ~95% (LeverageLimit binding)
        assert 0.92 <= sums[3] <= 0.951, f"bar 4 sum={sums[3]:.4f}"


class TestBackendTickers:
    """cvxportfolio uses pandas Series labelled by ticker; verify the
    user can pass real ticker symbols."""

    def test_real_ticker_labels_pass_through(self):
        from kernel.portfolio_qp.cvxportfolio_backend import (
            solve_portfolio_qp_cvxportfolio,
        )
        n = 5
        mu, Sigma = _make_problem(n=n, seed=5)
        sol = solve_portfolio_qp_cvxportfolio(
            w_current=np.zeros(n), mu=mu, Sigma=Sigma,
            risk_aversion=3.0, cost_kappa=0.0001, cash_reserve=0.0,
            w_upper=0.20, w_lower=0.0, dw_max=0.50,
            tickers=["AAPL", "MSFT", "GOOG", "META", "NVDA"],
        )
        assert sol.status == "optimal"
        assert len(sol.target_w) == n
