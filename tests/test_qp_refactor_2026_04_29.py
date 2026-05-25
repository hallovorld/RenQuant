"""QP refactor tests (2026-04-29) — tax basis + full Σ + turnover hard cap.

Three structural fixes from the QP audit:
  1. tax_cost_per_sell — discourage selling high-gain positions (per-asset)
  2. Sigma full matrix — accept off-diagonal correlations (no longer diag-only)
  3. turnover_max — hard constraint Σ|Δw| ≤ τ_max (vs the soft κ·‖Δw‖₁ alone)

Each test designed to FAIL on Stage-1 solver, PASS post-refactor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


# ── Fix 1: tax-basis cost ────────────────────────────────────────────────────

class TestTaxBasisCost:
    def test_high_gain_position_held_when_tax_drag_present(self):
        """A position with high tax drag is harder to liquidate vs zero-tax peer.

        Setup: two assets with identical μ = -0.001 (slightly negative), σ same.
        Both currently held at w=0.10. One has 25% unrealized gain × 30% tax
        rate = 7.5% tax cost per unit sold; the other zero.

        Expectation: pre-refactor (no tax_cost) — both get sold equally.
        Post-refactor — high-tax position is held longer (smaller sell)."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

        # No-tax case (baseline)
        sol_baseline = solve_portfolio_qp(
            w_current      = [0.10, 0.10],
            mu             = [-0.001, -0.001],
            sigma          = [0.05, 0.05],
            risk_aversion  = 3.0,
            cost_kappa     = 0.0001,
            w_upper        = 0.20,
        )

        # With tax cost: ticker 0 has heavy tax drag
        sol_tax = solve_portfolio_qp(
            w_current      = [0.10, 0.10],
            mu             = [-0.001, -0.001],
            sigma          = [0.05, 0.05],
            risk_aversion  = 3.0,
            cost_kappa     = 0.0001,
            w_upper        = 0.20,
            tax_cost_per_sell = [0.10, 0.0],   # 10% NAV-fraction tax drag
        )

        # Asset 0 should be held more strongly when tax cost is high
        assert sol_tax.delta_w[0] > sol_baseline.delta_w[0] - 1e-6, (
            f"high-tax asset should sell less. baseline Δw[0]={sol_baseline.delta_w[0]:+.4f}, "
            f"with-tax Δw[0]={sol_tax.delta_w[0]:+.4f}"
        )

    def test_tax_cost_no_effect_on_buys(self):
        """Tax cost only applies to sells — buys are not penalised."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        sol = solve_portfolio_qp(
            w_current      = [0.0],
            mu             = [0.05],
            sigma          = [0.05],
            risk_aversion  = 3.0,
            cost_kappa     = 0.0001,
            w_upper        = 0.20,
            tax_cost_per_sell = [0.50],  # huge tax — but we're BUYING
        )
        assert sol.delta_w[0] > 0, "buying should not be affected by tax_cost"

    def test_tax_cost_diagnostic_recorded(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        sol = solve_portfolio_qp(
            w_current      = [0.10],
            mu             = [-0.01],
            sigma          = [0.05],
            risk_aversion  = 3.0,
            w_upper        = 0.20,
            tax_cost_per_sell = [0.05],
        )
        assert sol.diagnostics["tax_cost_max"] == 0.05
        assert sol.diagnostics["tax_cost_mean"] == 0.05


# ── Fix 2: Full Σ matrix accepted ────────────────────────────────────────────

class TestFullSigmaMatrix:
    def test_off_diagonal_correlation_drives_diversification(self):
        """Two assets with identical μ, σ. Diagonal Σ → equal weights.
        Full Σ with high correlation → reduced total exposure."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        # Diagonal (independent) — should put weight on both
        sol_diag = solve_portfolio_qp(
            w_current=[0.0, 0.0], mu=[0.05, 0.05],
            sigma=[0.05, 0.05], risk_aversion=3.0, w_upper=0.30,
        )
        # Highly correlated — should reduce total
        Sigma_corr = np.array([
            [0.0025,  0.0024],   # ρ ≈ 0.96
            [0.0024,  0.0025],
        ])
        sol_corr = solve_portfolio_qp(
            w_current=[0.0, 0.0], mu=[0.05, 0.05],
            Sigma=Sigma_corr, risk_aversion=3.0, w_upper=0.30,
        )
        # Total exposure with correlated assets should be ≤ uncorrelated
        # (the variance of the sum is higher → optimizer takes less risk)
        sum_diag = float(np.sum(sol_diag.delta_w))
        sum_corr = float(np.sum(sol_corr.delta_w))
        assert sum_corr < sum_diag + 1e-6, (
            f"correlated case should have ≤ exposure: diag={sum_diag:.4f}, corr={sum_corr:.4f}"
        )
        # Diagnostics show off-diagonal nonzero count
        assert sol_corr.diagnostics["sigma_off_diag_nonzero"] >= 2

    def test_sigma_shape_validated(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        with pytest.raises(ValueError, match="Sigma shape"):
            solve_portfolio_qp(
                w_current=[0.0, 0.0], mu=[0.05, 0.05],
                Sigma=np.eye(3),  # wrong size
                risk_aversion=3.0, w_upper=0.30,
            )

    def test_zero_corr_is_not_replaced_by_reverse_lookup(self):
        """A real 0.0 corr must not fall through to a stale reverse value."""
        from types import SimpleNamespace

        from kernel.portfolio_qp.tasks import ComputeFullSigmaTask

        ctx = SimpleNamespace(
            config={"rotation": {"joint_actions": {"qp_use_full_sigma": True}}},
            corr_matrix={"A": {"B": 0.0}, "B": {"A": 0.95}},
            _qp_tickers=["A", "B"],
            _qp_sigma=np.array([0.20, 0.30]),
        )

        ComputeFullSigmaTask().run(ctx)

        assert ctx._qp_Sigma_full[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert ctx._qp_Sigma_full[1, 0] == pytest.approx(0.0, abs=1e-12)


# ── Fix 3: Turnover hard constraint ─────────────────────────────────────────

class TestTurnoverHardConstraint:
    def test_turnover_cap_binds(self):
        """Set turnover_max=0.1 with strong signals demanding 0.4+ trade."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        sol = solve_portfolio_qp(
            w_current      = [0.0, 0.0, 0.0, 0.0],
            mu             = [0.10, 0.10, 0.10, 0.10],
            sigma          = [0.05, 0.05, 0.05, 0.05],
            risk_aversion  = 3.0,
            cost_kappa     = 0.0001,
            w_upper        = 0.30,
            turnover_max   = 0.10,   # hard cap
        )
        actual = float(np.sum(np.abs(sol.delta_w)))
        assert actual <= 0.10 + 1e-4, (
            f"turnover {actual:.4f} exceeds cap 0.10"
        )

    def test_turnover_unconstrained_baseline_exceeds_cap(self):
        """Sanity: without turnover_max, the same problem trades more."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        sol = solve_portfolio_qp(
            w_current      = [0.0, 0.0, 0.0, 0.0],
            mu             = [0.10, 0.10, 0.10, 0.10],
            sigma          = [0.05, 0.05, 0.05, 0.05],
            risk_aversion  = 3.0,
            cost_kappa     = 0.0001,
            w_upper        = 0.30,
        )
        actual = float(np.sum(np.abs(sol.delta_w)))
        assert actual > 0.10, (
            f"without cap, expected turnover > 0.10, got {actual:.4f}"
        )

    def test_turnover_cap_diagnostic_recorded(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        sol = solve_portfolio_qp(
            w_current=[0.0, 0.0], mu=[0.10, 0.10],
            sigma=[0.05, 0.05], risk_aversion=3.0, w_upper=0.30,
            turnover_max=0.05,
        )
        assert sol.diagnostics["turnover_max"] == 0.05
        assert sol.diagnostics["actual_turnover"] <= 0.05 + 1e-4
