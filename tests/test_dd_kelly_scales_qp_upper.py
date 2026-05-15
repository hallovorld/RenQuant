"""Audit-mandated regression test #2 (doc/AUDIT_2026-05-12_dead_paths.md
§"Mandatory regression tests when fix lands"):

> 2. test_dd_kelly_scales_qp_upper.py — set portfolio drawdown to 15%,
>    dd_scaling.dd_max=0.20, assert _qp_w_upper shrinks.

Pins the Grossman-Zhou 1993 drawdown-conditioned Kelly scaling fix:
when portfolio drawdown approaches dd_max, single-name caps shrink
continuously (not waiting for the binary halt_pct circuit).
Without this guard, the dead-path bug (scaling a Kelly local var QP
never reads) would silently re-enter on any future Kelly refactor.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _ctx_with_dd(*, hwm, portfolio_value, max_pos=0.20,
                 dd_max=0.20, exponent=1.0, regime="BULL_VOLATILE"):
    """Build minimal ctx for ApplyExposureScalingTask. Drawdown derived
    via compute_portfolio_drawdown(hwm, portfolio_value)."""
    ctx = SimpleNamespace()
    ctx._qp_w_upper = np.full(3, max_pos)
    ctx.spy_regime = regime
    ctx.regime = regime
    ctx.spy_returns = [0.005, -0.005] * 60  # calm so vol-target ≈ 1.0
    ctx.hwm = hwm
    ctx.portfolio_value = portfolio_value
    ctx.config = {
        "ranking": {"kelly_sizing": {
            "drawdown_scaling": {
                "enabled": True,
                "dd_max":   dd_max,
                "exponent": exponent,
            },
        }},
    }
    return ctx


class TestDDKellyScalesQPUpper:

    def test_15pct_drawdown_with_20pct_dd_max_shrinks_upper(self):
        """Audit doc verbatim case: portfolio_drawdown = 15%, dd_max = 20%
        → expected scale = max(0, 1 - (0.15/0.20)^1) = 0.25
        → _qp_w_upper shrinks from 0.20 to 0.20 × 0.25 = 0.05."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask

        ctx = _ctx_with_dd(hwm=100.0, portfolio_value=85.0,
                            max_pos=0.20, dd_max=0.20)
        ApplyExposureScalingTask().run(ctx)

        # ctx._dd_kelly_scale should be 0.25 (= 1 - 0.15/0.20)
        assert abs(ctx._dd_kelly_scale - 0.25) < 1e-6, (
            f"DD-Kelly scale mismatch: got {ctx._dd_kelly_scale}, "
            f"expected 0.25 (= 1 - 0.15/0.20)"
        )
        # _qp_w_upper must shrink
        assert (ctx._qp_w_upper < 0.20).all(), (
            f"DD-Kelly failed to shrink _qp_w_upper: {ctx._qp_w_upper}. "
            f"Original 0.20 × scale 0.25 should give 0.05."
        )
        np.testing.assert_allclose(ctx._qp_w_upper, 0.05, rtol=1e-6)

    def test_zero_drawdown_no_op(self):
        """No drawdown → scale = 1.0 → _qp_w_upper unchanged."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        ctx = _ctx_with_dd(hwm=100.0, portfolio_value=100.0,
                            max_pos=0.20, dd_max=0.20)
        ApplyExposureScalingTask().run(ctx)
        assert ctx._dd_kelly_scale == 1.0
        np.testing.assert_array_equal(ctx._qp_w_upper, [0.20, 0.20, 0.20])

    def test_drawdown_at_dd_max_zeros_upper(self):
        """drawdown == dd_max → scale = 0 → _qp_w_upper effectively 0
        (no new buys allowed)."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        ctx = _ctx_with_dd(hwm=100.0, portfolio_value=80.0,  # 20% dd
                            max_pos=0.20, dd_max=0.20)
        ApplyExposureScalingTask().run(ctx)
        assert ctx._dd_kelly_scale == 0.0
        np.testing.assert_allclose(ctx._qp_w_upper, 0.0, atol=1e-9)

    def test_disabled_no_op(self):
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        ctx = _ctx_with_dd(hwm=100.0, portfolio_value=85.0)
        ctx.config["ranking"]["kelly_sizing"]["drawdown_scaling"]["enabled"] = False
        ApplyExposureScalingTask().run(ctx)
        assert ctx._dd_kelly_scale == 1.0
        np.testing.assert_array_equal(ctx._qp_w_upper, [0.20, 0.20, 0.20])
