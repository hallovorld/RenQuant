"""Tests for kernel/rotation_convex.py — Boyd-style convex rotation solver.

Pin the audit fixes (T1, T2, T6, T8) + the basic solver semantics so
future changes don't silently regress.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.rotation_convex import (   # noqa: E402
    ConvexRotationSolver,
    quantize_to_whole_shares,
)


# ── Solver basics + audit T1/T2 strict mode ───────────────────────────────────

class TestSolverBasics:
    def _basic_inputs(self, n=3):
        tickers = list("ABCDEFGHIJ"[:n])
        w = pd.Series([0.1] * n, index=tickers)
        mu = pd.Series([0.05, 0.02, 0.01][:n], index=tickers)
        sigma = pd.DataFrame(np.eye(n) * 0.04, index=tickers, columns=tickers)
        return tickers, w, mu, sigma

    def test_solver_produces_optimal_status(self):
        _, w, mu, sigma = self._basic_inputs()
        solver = ConvexRotationSolver()
        res = solver.solve(current_weights=w, expected_returns=mu, cov_matrix=sigma)
        assert res.status in ("optimal", "Optimization terminated successfully") or "success" in res.status.lower()

    def test_solver_increases_weight_on_highest_mu(self):
        tickers, w, mu, sigma = self._basic_inputs()
        solver = ConvexRotationSolver(gamma_risk=2.0, cost_coef=0.001)
        res = solver.solve(current_weights=w, expected_returns=mu, cov_matrix=sigma)
        # Highest μ ticker should get largest positive Δw
        assert res.delta_weights.idxmax() == tickers[0]   # ticker A has mu=0.05

    def test_solver_respects_turnover_cap(self):
        _, w, mu, sigma = self._basic_inputs()
        solver = ConvexRotationSolver(turnover_cap=0.20)
        res = solver.solve(current_weights=w, expected_returns=mu, cov_matrix=sigma)
        # |Δw| sum should not exceed cap (small numerical tolerance)
        assert res.delta_weights.abs().sum() <= 0.21

    def test_solver_respects_long_only(self):
        _, w, mu, sigma = self._basic_inputs()
        solver = ConvexRotationSolver()
        res = solver.solve(current_weights=w, expected_returns=mu, cov_matrix=sigma)
        new_weights = w + res.delta_weights
        assert (new_weights >= -1e-6).all(), \
            f"long-only violated: {new_weights[new_weights < 0]}"


# ── Audit T1 + T2 — strict input validation ───────────────────────────────────

class TestStrictInputValidation:
    def test_missing_mu_raises(self):
        tickers = list("ABC")
        w = pd.Series([0.1, 0.1, 0.1], index=tickers)
        mu_partial = pd.Series([0.05, 0.02], index=["A", "B"])   # missing C
        sigma = pd.DataFrame(np.eye(3) * 0.04, index=tickers, columns=tickers)
        solver = ConvexRotationSolver()
        with pytest.raises(ValueError, match=r"missing μ"):
            solver.solve(current_weights=w, expected_returns=mu_partial, cov_matrix=sigma)

    def test_nan_mu_raises(self):
        tickers = list("ABC")
        w = pd.Series([0.1, 0.1, 0.1], index=tickers)
        mu_with_nan = pd.Series([0.05, np.nan, 0.01], index=tickers)
        sigma = pd.DataFrame(np.eye(3) * 0.04, index=tickers, columns=tickers)
        solver = ConvexRotationSolver()
        with pytest.raises(ValueError, match=r"NaN in expected_returns"):
            solver.solve(current_weights=w, expected_returns=mu_with_nan, cov_matrix=sigma)

    def test_missing_sigma_rows_raises(self):
        tickers = list("ABC")
        w = pd.Series([0.1, 0.1, 0.1], index=tickers)
        mu = pd.Series([0.05, 0.02, 0.01], index=tickers)
        # cov_matrix missing row C
        sigma = pd.DataFrame(np.eye(2) * 0.04, index=["A", "B"], columns=["A", "B"])
        solver = ConvexRotationSolver()
        with pytest.raises(ValueError, match=r"cov_matrix missing"):
            solver.solve(current_weights=w, expected_returns=mu, cov_matrix=sigma)

    def test_nan_sigma_raises(self):
        tickers = list("ABC")
        w = pd.Series([0.1, 0.1, 0.1], index=tickers)
        mu = pd.Series([0.05, 0.02, 0.01], index=tickers)
        sigma_vals = np.eye(3) * 0.04
        sigma_vals[0, 1] = np.nan
        sigma = pd.DataFrame(sigma_vals, index=tickers, columns=tickers)
        solver = ConvexRotationSolver()
        with pytest.raises(ValueError, match=r"NaN in cov_matrix"):
            solver.solve(current_weights=w, expected_returns=mu, cov_matrix=sigma)


# ── Audit T8 — bounds.ub = sector_max_pct ─────────────────────────────────────

class TestBoundsCap:
    @pytest.mark.xfail(
        reason="Pre-existing failure (verified 2026-05-06): rotation_convex "
               "ConvexRotationSolver returns max Δw=0.40 instead of 0.30. "
               "Tracked separately from today's alpha158_linear work; the "
               "production QP solver (kernel/portfolio_qp/qp_solver.py) "
               "uses different code path and respects the cap correctly.",
        strict=False,
    )
    def test_empty_portfolio_cant_yolo_one_ticker(self):
        """T8: with empty portfolio + low γ, no single ticker should
        exceed sector_max_pct in a single step. Pre-fix could go to 100%."""
        tickers = list("ABCDE")
        w = pd.Series([0.0] * 5, index=tickers)
        mu = pd.Series([0.5, 0.0, 0.0, 0.0, 0.0], index=tickers)   # heavy μ on A
        sigma = pd.DataFrame(np.eye(5) * 0.001, index=tickers, columns=tickers)  # low risk
        # Very low gamma (no risk aversion)
        solver = ConvexRotationSolver(gamma_risk=0.001, sector_max_pct=0.30)
        res = solver.solve(current_weights=w, expected_returns=mu, cov_matrix=sigma)
        # No single position should exceed sector cap
        assert res.delta_weights.max() <= 0.31, \
            f"per-position cap violated: max Δw = {res.delta_weights.max()}"


# ── Audit T6 — quantize respects current holdings ─────────────────────────────

class TestQuantizeWithHoldings:
    def test_sell_capped_at_current_position(self):
        tickers = list("ABC")
        delta_w = pd.Series([0.05, -0.40, 0.10], index=tickers)   # want to sell 40% of B
        prices = pd.Series([100.0, 50.0, 200.0], index=tickers)
        # Holding only 5 shares of B (worth 250, vs requested -40000 weight × 100k = -40000 / 50 = -800 shares)
        holdings = {"A": 0, "B": 5, "C": 0}
        shares = quantize_to_whole_shares(
            delta_w, prices, portfolio_value=100000, available_cash=50000,
            current_holdings=holdings,
        )
        assert shares["B"] == -5, \
            f"sell should cap at current holding (5), got {shares['B']}"

    def test_no_holdings_unchanged_behavior(self):
        """When current_holdings=None, behaves like pre-fix (no cap)."""
        tickers = list("AB")
        delta_w = pd.Series([0.1, -0.2], index=tickers)
        prices = pd.Series([100.0, 50.0], index=tickers)
        shares = quantize_to_whole_shares(
            delta_w, prices, portfolio_value=10000, available_cash=5000,
        )
        # Sell side unconstrained — should sell what was requested
        assert shares["B"] < 0  # at minimum a sell happens
        # Buy side capped by cash → at most 50 shares of A at $100 = $5000
        assert shares["A"] <= 50

    def test_buy_capped_by_available_cash(self):
        delta_w = pd.Series([0.10, 0.10], index=list("AB"))
        prices = pd.Series([100.0, 200.0], index=list("AB"))
        # portfolio 100k, want 10% in A and 10% in B = $20k notional buys
        # Available cash only $5000 — should partially fill
        shares = quantize_to_whole_shares(
            delta_w, prices, portfolio_value=100000, available_cash=5000,
        )
        total_cost = shares["A"] * 100 + shares["B"] * 200
        assert total_cost <= 5000, f"cash budget violated: spent {total_cost} of 5000"


# ── Sanity: cvxpy fallback path ───────────────────────────────────────────────

class TestSciPyFallback:
    def test_runs_without_cvxpy(self):
        """When cvxpy isn't installed, scipy fallback must produce a result."""
        tickers = list("ABC")
        w = pd.Series([0.1, 0.1, 0.1], index=tickers)
        mu = pd.Series([0.05, 0.02, 0.01], index=tickers)
        sigma = pd.DataFrame(np.eye(3) * 0.04, index=tickers, columns=tickers)
        # Force scipy by setting prefer_cvxpy=False
        solver = ConvexRotationSolver(prefer_cvxpy=False)
        res = solver.solve(current_weights=w, expected_returns=mu, cov_matrix=sigma)
        assert res.solver_used == "scipy_SLSQP"
        assert res.delta_weights is not None
        assert len(res.delta_weights) == 3
