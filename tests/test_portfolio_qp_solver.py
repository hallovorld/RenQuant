"""Tests for kernel.portfolio_qp.qp_solver — Markowitz QP w/ linear cost.

Design: ``doc/unified_portfolio_action_design_2026-04-26.md`` §2.

Stage-0 contract: solver lands and is tested in isolation; not yet
wired into InferencePipeline (that's Stage-1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.qp_solver import QPSolution, solve_portfolio_qp  # noqa: E402


class TestBasicSolves:
    def test_zero_signal_with_zero_holdings_no_trade(self):
        """μ=0 + currently flat → optimum is no trade.

        (Note: μ=0 with current holdings WILL drive Δw<0 to reduce
        variance term γw'Σw; that's expected risk-averse behaviour.)
        """
        sol = solve_portfolio_qp(
            w_current=[0.0, 0.0, 0.0],
            mu=[0.0, 0.0, 0.0],
            sigma=[0.1, 0.1, 0.1],
            cost_kappa=0.001,
        )
        assert isinstance(sol, QPSolution)
        assert np.max(np.abs(sol.delta_w)) < 1e-3

    def test_positive_signal_buys(self):
        """μ > 0 with no current position → Δw > 0."""
        sol = solve_portfolio_qp(
            w_current=[0.0, 0.0, 0.0],
            mu=[0.05, 0.05, 0.05],
            sigma=[0.10, 0.10, 0.10],
            cost_kappa=0.0001,
            cash_reserve=0.5,
        )
        # All three should pick up positive position (subject to caps)
        assert (sol.delta_w > 0).all()
        # Cash constraint respected
        assert float(np.sum(sol.target_w)) <= 0.5 + 1e-6

    def test_negative_signal_sells(self):
        """μ < 0 on a held position → Δw < 0."""
        sol = solve_portfolio_qp(
            w_current=[0.15],
            mu=[-0.05],
            sigma=[0.10],
            cost_kappa=0.0001,
            w_lower=[0.0],
        )
        assert sol.delta_w[0] < 0.0
        assert sol.target_w[0] >= 0.0   # no shorts


class TestConstraints:
    def test_cash_reserve_respected(self):
        """Total weight cap = 1 - cash_reserve."""
        sol = solve_portfolio_qp(
            w_current=[0.0, 0.0, 0.0, 0.0],
            mu=[0.10, 0.10, 0.10, 0.10],
            sigma=[0.05, 0.05, 0.05, 0.05],
            cash_reserve=0.30,
            cost_kappa=0.0,
        )
        total = float(np.sum(sol.target_w))
        assert total <= 0.70 + 1e-6

    def test_position_cap_respected(self):
        sol = solve_portfolio_qp(
            w_current=[0.0],
            mu=[100.0],   # very strong signal
            sigma=[0.1],
            w_upper=0.20,
            cash_reserve=0.0,
            cost_kappa=0.0,
        )
        assert sol.target_w[0] <= 0.20 + 1e-6

    def test_lower_bound_no_shorts(self):
        sol = solve_portfolio_qp(
            w_current=[0.0],
            mu=[-100.0],   # extremely negative
            sigma=[0.10],
            w_lower=0.0,
        )
        assert sol.target_w[0] >= -1e-6   # bounded at 0

    def test_dw_max_respected(self):
        """Slippage cap |Δw| ≤ dw_max enforced."""
        sol = solve_portfolio_qp(
            w_current=[0.0],
            mu=[10.0],
            sigma=[0.1],
            dw_max=0.05,
            cash_reserve=0.0,
        )
        assert sol.delta_w[0] <= 0.05 + 1e-6

    def test_wash_sale_blocks_buying(self):
        """wash_sale_mask True → cannot increase position (Δw ≤ 0)."""
        sol = solve_portfolio_qp(
            w_current=[0.0],
            mu=[0.10],      # strong buy signal
            sigma=[0.10],
            wash_sale_mask=[True],   # blocked from buying
            cost_kappa=0.0,
        )
        assert sol.delta_w[0] <= 1e-6   # blocked from buy

    def test_wash_sale_allows_sell(self):
        """wash_sale_mask True doesn't block selling further."""
        sol = solve_portfolio_qp(
            w_current=[0.10],
            mu=[-0.10],
            sigma=[0.10],
            wash_sale_mask=[True],
            cost_kappa=0.0,
        )
        assert sol.delta_w[0] < 0.0   # selling permitted


class TestGarleanuPedersenIntuition:
    def test_higher_cost_smaller_trade(self):
        """As cost rises, trade size should shrink (G-P 2013 partial-move)."""
        kwargs = dict(
            w_current=[0.0, 0.0],
            mu=[0.05, 0.05],
            sigma=[0.10, 0.10],
            cash_reserve=0.0,
        )
        sol_low  = solve_portfolio_qp(cost_kappa=0.00001, **kwargs)
        sol_high = solve_portfolio_qp(cost_kappa=0.10,    **kwargs)
        assert float(np.sum(np.abs(sol_low.delta_w))) > \
               float(np.sum(np.abs(sol_high.delta_w)))

    def test_higher_gamma_smaller_position(self):
        """More risk-averse → smaller positions for same μ/σ."""
        kwargs = dict(
            w_current=[0.0],
            mu=[0.05],
            sigma=[0.10],
            cost_kappa=0.0,
            cash_reserve=0.0,
            w_upper=10.0,    # large enough that gamma matters
        )
        sol_low_g  = solve_portfolio_qp(risk_aversion=1.0, **kwargs)
        sol_high_g = solve_portfolio_qp(risk_aversion=10.0, **kwargs)
        assert sol_low_g.target_w[0] > sol_high_g.target_w[0]


class TestMatrixCovariance:
    def test_full_sigma_used_when_provided(self):
        """When Σ is given (with off-diagonal corr), solver uses it."""
        # Two perfectly-correlated assets — solver should split between them
        Sigma = np.array([[0.01, 0.01], [0.01, 0.01]])
        sol = solve_portfolio_qp(
            w_current=[0.0, 0.0],
            mu=[0.05, 0.05],
            Sigma=Sigma,
            cost_kappa=0.0001,
            cash_reserve=0.0,
        )
        # Solver should converge — exact split depends on penalty,
        # but both should be non-negative and total < 1
        assert (sol.target_w >= -1e-6).all()
        assert float(np.sum(sol.target_w)) <= 1.0

    def test_decorrelated_basket_diversifies(self):
        """Independent assets with same μ/σ → equal-weight target."""
        Sigma = np.diag([0.01, 0.01, 0.01])
        sol = solve_portfolio_qp(
            w_current=[0.0, 0.0, 0.0],
            mu=[0.05, 0.05, 0.05],
            Sigma=Sigma,
            cost_kappa=0.0,
            cash_reserve=0.0,
            w_upper=1.0,
        )
        # symmetric problem → near-equal weights
        assert np.std(sol.target_w) < 0.02


class TestEdgeCases:
    def test_nan_mu_treated_as_zero(self):
        """NaN in μ → asset contributes 0 to objective; solver doesn't crash."""
        sol = solve_portfolio_qp(
            w_current=[0.0, 0.10, 0.0],
            mu=[float("nan"), 0.05, 0.03],
            sigma=[0.10, 0.10, 0.10],
            cost_kappa=0.0001,
        )
        assert sol.status == "optimal"
        # Asset 0 with NaN μ should not be aggressively traded
        assert abs(sol.delta_w[0]) < 0.05

    def test_zero_sigma_floored(self):
        """Solver floors σ at 1e-6 to keep Σ positive-definite."""
        sol = solve_portfolio_qp(
            w_current=[0.0],
            mu=[0.05],
            sigma=[0.0],
            cost_kappa=0.0001,
        )
        assert sol.status == "optimal"

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match=r"len\(mu\)"):
            solve_portfolio_qp(
                w_current=[0.0, 0.0],
                mu=[0.05],
                sigma=[0.10],
            )

    def test_missing_sigma_and_Sigma_raises(self):
        with pytest.raises(ValueError, match="must provide either"):
            solve_portfolio_qp(
                w_current=[0.0],
                mu=[0.05],
            )


class TestSolutionShape:
    def test_solution_has_diagnostics(self):
        sol = solve_portfolio_qp(
            w_current=[0.0, 0.0],
            mu=[0.05, -0.02],
            sigma=[0.1, 0.1],
        )
        assert "n_assets" in sol.diagnostics
        assert sol.diagnostics["n_assets"] == 2
        assert "risk_aversion" in sol.diagnostics
        assert "cost_kappa" in sol.diagnostics
        assert sol.n_iter > 0

    def test_target_w_consistent_with_delta_w(self):
        w0 = np.array([0.10, 0.05])
        sol = solve_portfolio_qp(
            w_current=w0,
            mu=[0.05, -0.02],
            sigma=[0.10, 0.10],
        )
        np.testing.assert_allclose(
            sol.target_w, w0 + sol.delta_w, atol=1e-12,
        )


class TestPerformance:
    def test_100_asset_solve_under_500ms(self):
        """Solver must handle our universe scale (~101 assets) quickly."""
        import time
        n = 100
        rng = np.random.default_rng(42)
        w0 = rng.uniform(0, 0.05, size=n)
        mu = rng.normal(0.005, 0.01, size=n)
        sigma = rng.uniform(0.05, 0.15, size=n)
        t0 = time.time()
        sol = solve_portfolio_qp(
            w_current=w0, mu=mu, sigma=sigma,
            cost_kappa=0.0001, cash_reserve=0.10, w_upper=0.20,
        )
        elapsed_ms = (time.time() - t0) * 1000
        # Generous bound — production target <100ms; CI/test must hold <500ms
        assert elapsed_ms < 500, f"QP took {elapsed_ms:.0f}ms — too slow"
        assert sol.status == "optimal"


# ── Stage 2: Garleanu-Pedersen partial-move (signal_decay) ────────────────────

class TestStageTwoGarleanuPedersen:
    def test_persistent_signal_trades_more(self):
        """φ → 1 (persistent) → larger trade than φ = 0 (one-shot)."""
        kwargs = dict(
            w_current=[0.0],
            mu=[0.005],            # weak so neither hits cap
            sigma=[0.10],
            risk_aversion=10.0,    # high so γ_eff dominates
            cost_kappa=0.001,
            cash_reserve=0.0,
            w_upper=10.0,
        )
        sol_decay_low  = solve_portfolio_qp(signal_decay=0.0,  **kwargs)
        sol_decay_high = solve_portfolio_qp(signal_decay=0.8,  **kwargs)
        assert sol_decay_high.target_w[0] > sol_decay_low.target_w[0]

    def test_signal_decay_clipped_at_099(self):
        """φ ≥ 0.99 → clamped (avoid div-by-zero)."""
        sol = solve_portfolio_qp(
            w_current=[0.0],
            mu=[0.05],
            sigma=[0.10],
            signal_decay=1.5,   # invalid → clipped
        )
        assert sol.status == "optimal"


# ── Stage 4: Grossman-Zhou drawdown scaler ────────────────────────────────────

class TestStageFourDrawdownScaler:
    def test_higher_dd_smaller_position(self):
        """As DD approaches limit, position shrinks (γ_eff grows)."""
        kwargs = dict(
            w_current=[0.0],
            mu=[0.05],
            sigma=[0.10],
            risk_aversion=3.0,
            cost_kappa=0.0,
            cash_reserve=0.0,
            drawdown_limit=0.20,
            w_upper=10.0,        # large enough for γ_eff to matter
        )
        sol_no_dd  = solve_portfolio_qp(drawdown=0.0,  **kwargs)
        sol_at_dd  = solve_portfolio_qp(drawdown=0.18, **kwargs)
        assert sol_no_dd.target_w[0] > sol_at_dd.target_w[0]

    def test_dd_at_limit_forces_zero_position(self):
        """DD ≈ α → γ_eff → very large → target_w → 0."""
        sol = solve_portfolio_qp(
            w_current=[0.0],
            mu=[0.05],
            sigma=[0.10],
            risk_aversion=3.0,
            cost_kappa=0.0,
            drawdown=0.20,
            drawdown_limit=0.20,
            w_upper=10.0,
        )
        assert sol.target_w[0] < 0.05
        assert sol.diagnostics["dd_factor"] < 0.01

    def test_dd_diagnostics_surface(self):
        sol = solve_portfolio_qp(
            w_current=[0.0],
            mu=[0.05],
            sigma=[0.10],
            drawdown=0.05,
            drawdown_limit=0.20,
        )
        assert "gamma_effective" in sol.diagnostics
        assert "dd_factor" in sol.diagnostics
        assert sol.diagnostics["dd_factor"] == pytest.approx(0.75, rel=1e-6)


# ── Stage 5: Garlappi-Uppal-Wang robust μ ─────────────────────────────────────

class TestStageFiveRobustMu:
    def test_robust_mu_shrinks_position(self):
        """κ > 0 reduces effective μ → smaller position."""
        kwargs = dict(
            w_current=[0.0],
            mu=[0.05],
            sigma=[0.10],
            risk_aversion=3.0,
            cost_kappa=0.0,
            cash_reserve=0.0,
            w_upper=10.0,
        )
        sol_no_robust = solve_portfolio_qp(robust_mu_kappa=0.0,  **kwargs)
        sol_robust    = solve_portfolio_qp(robust_mu_kappa=0.5,  **kwargs)
        assert sol_robust.target_w[0] < sol_no_robust.target_w[0]

    def test_robust_kappa_one_neutralises_one_sigma(self):
        """κ=1 + μ=σ → effective μ ≈ 0 → trade ≈ 0."""
        sol = solve_portfolio_qp(
            w_current=[0.0],
            mu=[0.10],
            sigma=[0.10],
            risk_aversion=3.0,
            cost_kappa=0.0,
            robust_mu_kappa=1.0,
            cash_reserve=0.0,
            w_upper=10.0,
        )
        assert abs(sol.delta_w[0]) < 0.01

    def test_robust_diagnostics_surface(self):
        sol = solve_portfolio_qp(
            w_current=[0.0],
            mu=[0.05],
            sigma=[0.10],
            robust_mu_kappa=0.5,
        )
        assert sol.diagnostics["robust_kappa"] == pytest.approx(0.5)


# ── Stages combined ───────────────────────────────────────────────────────────

class TestStagesCombined:
    def test_all_stages_compose(self):
        """Stages 2 + 4 + 5 stack: persistent signal + DD halt + robust μ."""
        sol = solve_portfolio_qp(
            w_current=[0.0],
            mu=[0.05],
            sigma=[0.10],
            risk_aversion=3.0,
            cost_kappa=0.0001,
            cash_reserve=0.0,
            signal_decay=0.5,
            drawdown=0.10,
            drawdown_limit=0.20,
            robust_mu_kappa=0.3,
            w_upper=10.0,
        )
        assert sol.status == "optimal"
        # Each knob's diagnostic surfaced
        for k in ("signal_decay", "dd_factor",
                   "robust_kappa", "gamma_effective"):
            assert k in sol.diagnostics
