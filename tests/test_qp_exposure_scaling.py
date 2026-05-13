"""AUDIT REGRESSION GUARD — vol-target + DD-Kelly dead-path fix (2026-05-12).

Pinned invariant (CLAUDE.md §5.13.10):
    Vol-target and DD-Kelly scaling must affect QP `_qp_w_upper`
    INDEPENDENT of the Kelly sizing path. Specifically:

    ctx._qp_w_upper ≡ max_pos × confidence × vol_target_scale × dd_scale

    Previously these scales lived inside `ApplyKellySizingTask` and
    multiplied a local `max_pct` used only by `_kelly_with_reason()`.
    With NGB off (current prod), `mu is None` → Kelly returns 0 for
    every candidate → vol-target scale unused → bug.

Reference incident: doc/AUDIT_2026-05-12_dead_paths.md.
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


def _make_ctx(spy_returns=None, hwm=100_000.0, portfolio_value=100_000.0, **cfg):
    """Minimal ctx for testing the scaling task."""
    ctx = SimpleNamespace()
    ctx._qp_w_upper = np.array([0.20, 0.20, 0.20])
    ctx.config = cfg
    ctx.spy_returns = spy_returns or []
    ctx.hwm = hwm
    ctx.portfolio_value = portfolio_value
    return ctx


class TestVolTargetIndependentOfNGB:
    """The §5.13.10 regression guard: vol-target observable effect with NGB OFF."""

    def test_vol_target_shrinks_w_upper_in_high_vol_regime(self):
        """Spike SPY realized vol → vol_target_scale < 1 → w_upper shrinks."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask

        # Construct SPY return series: target_vol=0.05, realized ≈ 0.30 ann
        # → raw_scale ≈ 0.05/0.30 ≈ 0.17, clipped to floor=0.30
        big_moves = [0.02, -0.02] * 60  # high realized vol
        ctx = _make_ctx(
            spy_returns=big_moves,
            ranking={"kelly_sizing": {"vol_target": {
                "enabled": True, "target_vol": 0.05, "window_days": 60,
                "floor": 0.30, "ceiling": 1.50,
            }}},
        )
        ApplyExposureScalingTask().run(ctx)
        assert ctx._vol_target_scale == pytest.approx(0.30, rel=0.01), (
            f"Expected vol_target_scale ≈ 0.30 (floor), got {ctx._vol_target_scale}"
        )
        # w_upper must have observably shrunk
        assert np.all(ctx._qp_w_upper < 0.20)
        assert ctx._qp_w_upper[0] == pytest.approx(0.20 * 0.30, rel=0.01)

    def test_vol_target_disabled_no_effect(self):
        """Default config: vol_target.enabled=false → w_upper unchanged."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        ctx = _make_ctx()
        before = ctx._qp_w_upper.copy()
        ApplyExposureScalingTask().run(ctx)
        np.testing.assert_array_equal(ctx._qp_w_upper, before)
        assert ctx._vol_target_scale == 1.0
        assert ctx._dd_kelly_scale == 1.0

    def test_vol_target_fail_open_on_empty_spy(self):
        """No SPY history → scale=1.0 (fail-open per §5.13.11)."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        ctx = _make_ctx(
            spy_returns=[],  # empty
            ranking={"kelly_sizing": {"vol_target": {
                "enabled": True, "target_vol": 0.10, "window_days": 60,
            }}},
        )
        ApplyExposureScalingTask().run(ctx)
        assert ctx._vol_target_scale == 1.0


class TestDDKellyScaling:

    def test_dd_kelly_shrinks_w_upper_under_drawdown(self):
        """15% drawdown with dd_max=20%, exponent=1 → scale = 1 - 0.75 = 0.25."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        ctx = _make_ctx(
            hwm=100_000, portfolio_value=85_000,  # 15% DD
            ranking={"kelly_sizing": {"drawdown_scaling": {
                "enabled": True, "dd_max": 0.20, "exponent": 1.0,
            }}},
        )
        ApplyExposureScalingTask().run(ctx)
        assert ctx._dd_kelly_scale == pytest.approx(0.25, rel=0.01)
        assert ctx._qp_w_upper[0] == pytest.approx(0.20 * 0.25, rel=0.01)

    def test_dd_kelly_no_dd_no_effect(self):
        """At HWM, no drawdown → scale = 1.0, w_upper unchanged."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        ctx = _make_ctx(
            hwm=100_000, portfolio_value=100_000,
            ranking={"kelly_sizing": {"drawdown_scaling": {
                "enabled": True, "dd_max": 0.20,
            }}},
        )
        ApplyExposureScalingTask().run(ctx)
        assert ctx._dd_kelly_scale == 1.0
        np.testing.assert_array_equal(ctx._qp_w_upper, [0.20, 0.20, 0.20])


class TestVolTargetAndDDCompose:
    """Both scales must compose multiplicatively at the QP bound."""

    def test_combined_scaling_multiplicative(self):
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        big_moves = [0.02, -0.02] * 60  # high vol → vt floor 0.30
        ctx = _make_ctx(
            spy_returns=big_moves,
            hwm=100_000, portfolio_value=90_000,  # 10% DD
            ranking={"kelly_sizing": {
                "vol_target": {"enabled": True, "target_vol": 0.05,
                               "window_days": 60, "floor": 0.30},
                "drawdown_scaling": {"enabled": True, "dd_max": 0.20,
                                     "exponent": 1.0},
            }},
        )
        ApplyExposureScalingTask().run(ctx)
        # vt ≈ 0.30 (floor), dd = 1 - 0.10/0.20 = 0.50, combined = 0.15
        assert ctx._vol_target_scale == pytest.approx(0.30, rel=0.01)
        assert ctx._dd_kelly_scale   == pytest.approx(0.50, rel=0.01)
        assert ctx._qp_w_upper[0]    == pytest.approx(0.20 * 0.30 * 0.50, rel=0.01)


class TestNewConfigPathAlsoSupported:

    def test_top_level_exposure_scaling_config_works(self):
        """Future-clean config location: exposure_scaling.vol_target.*"""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        big_moves = [0.02, -0.02] * 60
        ctx = _make_ctx(
            spy_returns=big_moves,
            exposure_scaling={"vol_target": {
                "enabled": True, "target_vol": 0.05, "window_days": 60,
                "floor": 0.30,
            }},
        )
        ApplyExposureScalingTask().run(ctx)
        assert ctx._vol_target_scale == pytest.approx(0.30, rel=0.01)


class TestWiredIntoJointPortfolioQPJob:
    """Pin §5.13.2: prove the task is actually IN the prod pipeline."""

    def test_task_is_in_qp_job(self):
        from kernel.portfolio_qp.job_qp import JointPortfolioQPJob
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        job = JointPortfolioQPJob()
        names = [type(t).__name__ for t in job.tasks]
        assert "ApplyExposureScalingTask" in names, (
            "ApplyExposureScalingTask MUST be wired into JointPortfolioQPJob "
            "or the §5.13.10 dead-path bug returns."
        )
        # Must run AFTER ComputeQPConstraintsTask (it scales the bounds
        # that task produced), BEFORE ApplyConvictionCapTask (which also
        # multiplies _qp_w_upper — order doesn't matter mathematically
        # but pinning the order documents the invariant).
        idx_cqc = names.index("ComputeQPConstraintsTask")
        idx_exp = names.index("ApplyExposureScalingTask")
        assert idx_exp > idx_cqc, (
            "ApplyExposureScalingTask must run AFTER ComputeQPConstraintsTask"
        )
