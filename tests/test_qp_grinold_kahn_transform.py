"""AUDIT REGRESSION GUARD — Grinold-Kahn α→μ transform (2026-05-12).

Pinned invariants:

1. Default (alpha_to_mu.enabled=false) → identity. Existing config
   produces same `_qp_mu` as before — backward compat for the entire
   prod baseline.

2. Enabled → μ_QP = IC × σ × z(score). Magnitude of μ_QP is ≈ IC × σ
   (since |z| has E[|z|] ≈ √(2/π) ≈ 0.8 for normal). With IC=0.10 and
   σ=0.05, μ_QP magnitude is ~5e-3 — matching the natural scale of
   Σ's diagonal and the QP risk penalty's calibration.

3. Scale invariance: scaling raw μ by any positive constant produces the
   SAME transformed μ_QP (because z-score is scale-invariant). This is
   the property that fixes the §5.13.10 NGBoost bug class: swapping LTR
   panel_score (±2) for NGBoost μ (±0.005) yields identical μ_QP after
   transform.

Reference: Grinold 1989 *J. Portfolio Management*; Grinold-Kahn 1999
*Active Portfolio Management* §5.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _make_ctx(mu, sigma, **cfg):
    ctx = SimpleNamespace()
    ctx._qp_mu = np.asarray(mu, dtype=float)
    ctx._qp_sigma = np.asarray(sigma, dtype=float)
    ctx.config = cfg
    return ctx


class TestGrinoldKahnTransform:

    def test_disabled_default_is_identity(self):
        """Backward compat: alpha_to_mu OFF → _qp_mu unchanged."""
        from kernel.portfolio_qp.tasks import ApplyGrinoldKahnTransformTask
        ctx = _make_ctx(mu=[1.5, -0.5, 0.0], sigma=[0.05, 0.07, 0.04])
        before = ctx._qp_mu.copy()
        ApplyGrinoldKahnTransformTask().run(ctx)
        np.testing.assert_array_equal(ctx._qp_mu, before)

    def test_enabled_transforms_to_sigma_scale(self):
        """μ_QP = IC × σ × z. With IC=0.10, scaled output is in σ-units."""
        from kernel.portfolio_qp.tasks import ApplyGrinoldKahnTransformTask
        ctx = _make_ctx(
            mu=[2.0, 0.0, -2.0], sigma=[0.10, 0.10, 0.10],
            ranking={"alpha_to_mu": {"enabled": True, "ic": 0.10}},
        )
        ApplyGrinoldKahnTransformTask().run(ctx)
        # z-scores: (2-0)/2=1, (0-0)/2=0, (-2-0)/2=-1
        # μ_QP = 0.10 × 0.10 × [1, 0, -1] = [0.01, 0, -0.01]
        assert ctx._qp_mu == pytest.approx([0.01, 0.0, -0.01], abs=1e-9)
        assert ctx._qp_mu_transformed is True

    def test_scale_invariance_fixes_ngboost_bug_class(self):
        """The §5.13.10 NGBoost guard: same RANK ordering produces same
        μ_QP regardless of input scale (panel_score ±2 vs NGBoost μ ±0.005)."""
        from kernel.portfolio_qp.tasks import ApplyGrinoldKahnTransformTask
        sigma = [0.05, 0.05, 0.05]
        # Panel-score scale (±2 z-score)
        ctx_panel = _make_ctx(
            mu=[+2.0, 0.0, -2.0], sigma=sigma,
            ranking={"alpha_to_mu": {"enabled": True, "ic": 0.10}},
        )
        # NGBoost-μ scale (±0.005 raw return, 400× smaller)
        ctx_ngb   = _make_ctx(
            mu=[+0.005, 0.0, -0.005], sigma=sigma,
            ranking={"alpha_to_mu": {"enabled": True, "ic": 0.10}},
        )
        ApplyGrinoldKahnTransformTask().run(ctx_panel)
        ApplyGrinoldKahnTransformTask().run(ctx_ngb)
        # Same rank order, same σ, same IC → bit-identical μ_QP.
        # This is the property that decouples QP risk-penalty calibration
        # from the upstream signal source.
        np.testing.assert_allclose(ctx_panel._qp_mu, ctx_ngb._qp_mu, atol=1e-12)

    def test_per_asset_sigma_scales(self):
        """Higher-σ assets get LARGER μ_QP magnitude for the same z."""
        from kernel.portfolio_qp.tasks import ApplyGrinoldKahnTransformTask
        ctx = _make_ctx(
            mu=[+1.0, -1.0], sigma=[0.05, 0.20],
            ranking={"alpha_to_mu": {"enabled": True, "ic": 0.10}},
        )
        ApplyGrinoldKahnTransformTask().run(ctx)
        # ddof=1 sample std on [+1, -1] → mean=0, std=√2
        # z = [+1/√2, -1/√2], μ_QP = 0.10 × [0.05, 0.20] × [+1/√2, -1/√2]
        expected = 0.10 * np.array([0.05, 0.20]) * np.array([+1, -1]) / np.sqrt(2)
        np.testing.assert_allclose(ctx._qp_mu, expected, atol=1e-9)
        # Pin the qualitative property: |μ_QP[1]| > |μ_QP[0]| because σ[1] > σ[0]
        assert abs(ctx._qp_mu[1]) > abs(ctx._qp_mu[0])

    def test_constant_mu_fail_open(self):
        """All-equal mu → z-score division by zero → fail-open (no transform)."""
        from kernel.portfolio_qp.tasks import ApplyGrinoldKahnTransformTask
        ctx = _make_ctx(
            mu=[0.5, 0.5, 0.5], sigma=[0.05, 0.05, 0.05],
            ranking={"alpha_to_mu": {"enabled": True, "ic": 0.10}},
        )
        before = ctx._qp_mu.copy()
        ApplyGrinoldKahnTransformTask().run(ctx)
        # σ_z=0 → no transform, original mu unchanged
        np.testing.assert_array_equal(ctx._qp_mu, before)

    def test_nan_inputs_passthrough(self):
        """NaN entries skipped from z-score; transform applied to rest.
        NaN positions get 0 (neutral signal)."""
        from kernel.portfolio_qp.tasks import ApplyGrinoldKahnTransformTask
        ctx = _make_ctx(
            mu=[+1.0, float("nan"), -1.0],
            sigma=[0.05, 0.05, 0.05],
            ranking={"alpha_to_mu": {"enabled": True, "ic": 0.10}},
        )
        ApplyGrinoldKahnTransformTask().run(ctx)
        # finite mu = [1, -1], mean=0, std=√2, z=[+1/√2, ?, -1/√2]
        # NaN position → z[1]=0 → μ_QP[1] = 0
        assert ctx._qp_mu[1] == pytest.approx(0.0, abs=1e-12)
        # Other two get IC × σ × z
        expected = 0.10 * 0.05 * (1.0 / np.sqrt(2))
        assert ctx._qp_mu[0] == pytest.approx(+expected, abs=1e-9)
        assert ctx._qp_mu[2] == pytest.approx(-expected, abs=1e-9)


class TestWiredIntoJointPortfolioQPJob:
    """Pin §5.13.2: prove the task is actually IN the prod pipeline."""

    def test_task_is_in_qp_job_after_mu_sigma_build(self):
        from kernel.portfolio_qp.job_qp import JointPortfolioQPJob
        job = JointPortfolioQPJob()
        names = [type(t).__name__ for t in job.tasks]
        assert "ApplyGrinoldKahnTransformTask" in names, (
            "ApplyGrinoldKahnTransformTask MUST be wired into JointPortfolioQPJob "
            "or the §5.13.10 NGBoost μ-scale-mismatch bug returns."
        )
        # Must run AFTER mu/sigma vectors are built (private classes
        # named with leading underscore in job_qp.py).
        idx_mu  = names.index("_BuildMuVectorTask")
        idx_sig = names.index("_BuildSigmaVectorTask")
        idx_gk  = names.index("ApplyGrinoldKahnTransformTask")
        assert idx_gk > idx_mu and idx_gk > idx_sig


class TestQPMuContract:
    """QP must not silently optimize raw rank scores as expected returns."""

    def test_warn_mode_records_raw_score_fallback(self):
        from kernel.portfolio_qp.tasks import ValidateQPMuContractTask
        src = {
            "AAA": SimpleNamespace(ticker="AAA", panel_score=1.2),
            "BBB": SimpleNamespace(ticker="BBB", mu=0.01, panel_score=0.5),
        }
        ctx = SimpleNamespace(
            _qp_tickers=["AAA", "BBB"],
            _qp_mu_source_map=src,
            counters={},
            config={"rotation": {"joint_actions": {"qp_mu_contract": "warn"}}},
        )

        assert ValidateQPMuContractTask().run(ctx) is None
        assert ctx.counters["qp_mu_contract_fallback"] == 1
        assert ctx._qp_mu_contract["ok"] is False

    def test_strict_mode_stops_when_raw_scores_are_untransformed(self):
        from kernel.portfolio_qp.tasks import ValidateQPMuContractTask
        src = {"AAA": SimpleNamespace(ticker="AAA", panel_score=1.2)}
        ctx = SimpleNamespace(
            _qp_tickers=["AAA"],
            _qp_mu_source_map=src,
            counters={},
            config={"rotation": {"joint_actions": {"qp_mu_contract": "strict"}}},
        )

        assert ValidateQPMuContractTask().run(ctx) is False
        assert ctx.counters["qp_mu_contract_block"] == 1

    def test_grinold_kahn_transform_satisfies_contract(self):
        from kernel.portfolio_qp.tasks import ValidateQPMuContractTask
        src = {"AAA": SimpleNamespace(ticker="AAA", panel_score=1.2)}
        ctx = SimpleNamespace(
            _qp_tickers=["AAA"],
            _qp_mu_source_map=src,
            _qp_mu_transformed=True,
            counters={},
            config={
                "ranking": {"alpha_to_mu": {"enabled": True, "ic": 0.10}},
                "rotation": {"joint_actions": {"qp_mu_contract": "strict"}},
            },
        )

        assert ValidateQPMuContractTask().run(ctx) is None
        assert ctx._qp_mu_contract["ok"] is True

    def test_task_is_wired_between_transform_and_weights(self):
        from kernel.portfolio_qp.job_qp import JointPortfolioQPJob
        job = JointPortfolioQPJob()
        names = [type(t).__name__ for t in job.tasks]
        idx_gk = names.index("ApplyGrinoldKahnTransformTask")
        idx_contract = names.index("ValidateQPMuContractTask")
        idx_weights = names.index("BuildWeightVectorTask")
        assert idx_gk < idx_contract < idx_weights


class TestQPHorizonContract:
    """QP μ/σ must describe the same single-period horizon.

    Calibrator μ is a forward-return estimate over panel_ltr.lookahead_days
    while realized-vol fallback is annualized. Markowitz optimization is
    single-period: expected return and covariance must share the same period.
    """

    def test_annualized_sigma_is_scaled_to_mu_horizon(self):
        from kernel.portfolio_qp.tasks import AlignQPHorizonUnitsTask
        ctx = _make_ctx(
            mu=[0.02, 0.01],
            sigma=[0.20, 0.10],
            panel_ltr={"lookahead_days": 63},
            rotation={"joint_actions": {
                "qp_sigma_horizon_mode": "match_mu",
                "qp_sigma_unit": "annualized",
                "qp_horizon_contract": "strict",
            }},
        )

        assert AlignQPHorizonUnitsTask().run(ctx) is None
        np.testing.assert_allclose(ctx._qp_sigma, [0.10, 0.05], atol=1e-12)
        assert ctx._qp_horizon_contract["sigma_unit"] == "annualized"
        assert ctx._qp_horizon_contract["mu_horizon_days"] == 63

    def test_horizon_sigma_is_not_rescaled(self):
        from kernel.portfolio_qp.tasks import AlignQPHorizonUnitsTask
        ctx = _make_ctx(
            mu=[0.02],
            sigma=[0.08],
            rotation={"joint_actions": {
                "qp_sigma_horizon_mode": "match_mu",
                "qp_sigma_unit": "horizon",
                "qp_mu_horizon_days": 60,
            }},
        )

        assert AlignQPHorizonUnitsTask().run(ctx) is None
        assert ctx._qp_sigma[0] == pytest.approx(0.08, abs=1e-12)
        assert ctx._qp_horizon_contract["scale"] == pytest.approx(1.0)

    def test_missing_horizon_strict_contract_blocks_qp(self):
        from kernel.portfolio_qp.tasks import AlignQPHorizonUnitsTask
        ctx = _make_ctx(
            mu=[0.02],
            sigma=[0.08],
            rotation={"joint_actions": {
                "qp_sigma_horizon_mode": "match_mu",
                "qp_sigma_unit": "annualized",
                "qp_horizon_contract": "strict",
            }},
        )

        assert AlignQPHorizonUnitsTask().run(ctx) is False
        assert ctx._qp_horizon_contract["ok"] is False

    def test_task_is_wired_between_mu_contract_and_covariance(self):
        from kernel.portfolio_qp.job_qp import JointPortfolioQPJob
        job = JointPortfolioQPJob()
        names = [type(t).__name__ for t in job.tasks]
        idx_contract = names.index("ValidateQPMuContractTask")
        idx_horizon = names.index("AlignQPHorizonUnitsTask")
        idx_sigma = names.index("ComputeFullSigmaTask")
        assert idx_contract < idx_horizon < idx_sigma


class TestQPCashDeploymentContract:
    """The cash-drag target must not force deployment into negative edge."""

    def _solver_kwargs(self, mu):
        from kernel.portfolio_qp.tasks import SolveMarkowitzQPTask
        ctx = _make_ctx(
            mu=mu,
            sigma=[0.10] * len(mu),
            rotation={"joint_actions": {
                "qp_min_invested_pct": 0.70,
                "qp_min_invested_requires_positive_edge": True,
                "qp_min_invested_edge_floor": 0.002,
            }},
        )
        n = len(mu)
        ctx._qp_w_current = np.zeros(n)
        ctx._qp_Sigma_full = None
        ctx._qp_cash_reserve = 0.0
        ctx._qp_w_upper = np.ones(n)
        ctx._qp_w_lower = np.zeros(n)
        ctx._qp_dw_max = np.ones(n)
        ctx._qp_wash_mask = np.zeros(n, dtype=bool)
        ctx._qp_tax_cost = np.zeros(n)
        ctx._qp_turnover_max = None
        ctx._qp_v_daily_dollar = None
        ctx._qp_sector_indicator = None
        ctx._qp_sector_cap_vec = None
        ctx._qp_corr_group_pairs = None
        ctx._qp_gross_max = None
        ctx.portfolio_value = 100_000.0
        return SolveMarkowitzQPTask._build_solver_kwargs(  # noqa: SLF001
            ctx, ctx.config["rotation"]["joint_actions"]
        ), ctx

    def test_min_invested_drops_to_zero_when_no_positive_edge(self):
        kwargs, ctx = self._solver_kwargs(mu=[-0.01, 0.001])
        assert kwargs["min_invested_pct"] == pytest.approx(0.0)
        assert ctx._qp_min_invested_contract["blocked"] is True

    def test_min_invested_survives_when_best_mu_clears_hurdle(self):
        kwargs, ctx = self._solver_kwargs(mu=[-0.01, 0.01])
        assert kwargs["min_invested_pct"] == pytest.approx(0.70)
        assert ctx._qp_min_invested_contract["blocked"] is False
