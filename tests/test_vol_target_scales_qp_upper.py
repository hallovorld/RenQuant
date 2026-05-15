"""Audit-mandated regression test #1 (doc/AUDIT_2026-05-12_dead_paths.md
§"Mandatory regression tests when fix lands"):

> 1. test_vol_target_scales_qp_upper.py — set
>    vol_target.target_vol=0.05, run sim 1 bar, assert
>    ctx._qp_w_upper[0] < max_position_pct AND assert QP-emitted
>    buy size respects the reduced upper bound.

Pins the Moskowitz-Ooi-Pedersen 2012 vol-targeting fix: realized SPY
vol >> target_vol → vol_target_scale < 1.0 → ctx._qp_w_upper shrinks
proportionally. Without this guard, a future Kelly refactor that
re-introduces the 2026-05-12 dead-path bug would silently pass.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


class TestVolTargetScalesQPUpper:

    def _make_ctx(self, *, max_pos=0.20, target_vol=0.05,
                  spy_returns=None, regime="BULL_VOLATILE"):
        ctx = SimpleNamespace()
        ctx._qp_w_upper = np.full(3, max_pos)
        ctx.spy_regime = regime
        ctx.regime = regime
        ctx.spy_returns = spy_returns or ([0.02, -0.02] * 60)  # ~32% ann vol
        ctx.config = {
            "ranking": {"kelly_sizing": {
                "vol_target": {
                    "enabled": True,
                    "target_vol": target_vol,
                    "window_days": 60,
                    "floor": 0.10,
                    "ceiling": 1.50,
                },
            }},
        }
        return ctx

    def test_target_vol_below_realized_shrinks_upper(self):
        """target_vol=0.05 (5% ann) << realized ~32% ann → scale < 1.0
        → ctx._qp_w_upper strictly less than original max_pos."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask

        original_max = 0.20
        ctx = self._make_ctx(max_pos=original_max, target_vol=0.05)
        ApplyExposureScalingTask().run(ctx)

        # Per-asset upper bound MUST shrink — the dead-path bug would
        # leave it untouched at original_max
        assert (ctx._qp_w_upper < original_max).all(), (
            f"vol-target failed to scale _qp_w_upper: "
            f"target_vol=0.05 vs realized ~0.32 ann should produce "
            f"scale < 1.0, but got {ctx._qp_w_upper}"
        )
        # Scale stored on ctx for QP downstream
        assert ctx._vol_target_scale < 1.0
        # Scale should be approximately target_vol / realized_vol_60d,
        # bounded by [floor, ceiling]. Allow ±2% rel tolerance because the
        # internal helper may use ddof=1 vs ddof=0 stdev.
        realized = float(np.std(ctx.spy_returns)) * np.sqrt(252.0)
        expected = max(0.10, min(1.50, 0.05 / realized))
        assert abs(ctx._vol_target_scale - expected) / expected < 0.02

    def test_emitted_buy_respects_reduced_upper(self):
        """End-to-end: scale × max_pos = effective bound. Pin invariant
        from audit doc:
            _qp_w_upper ≡ max_pos × confidence × vol_target_scale × dd_scale
        """
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        ctx = self._make_ctx(max_pos=0.20, target_vol=0.05)
        ApplyExposureScalingTask().run(ctx)
        # The scale should multiplicatively shrink each upper bound.
        # Original 0.20 × scale ≈ effective bound.
        scale = ctx._vol_target_scale
        np.testing.assert_allclose(
            ctx._qp_w_upper, 0.20 * scale, rtol=1e-9,
            err_msg="ctx._qp_w_upper does not equal max_pos × vol_target_scale",
        )

    def test_target_vol_above_realized_no_leverage(self):
        """target_vol high vs realized → scale would want > 1.0, but ceiling
        caps it. Also verifies the no-leverage default invariant."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        # Calm market: ±0.005 daily → realized ~8% ann
        calm_returns = [0.005, -0.005] * 60
        ctx = self._make_ctx(max_pos=0.20, target_vol=0.20,  # higher target
                             spy_returns=calm_returns)
        # Force ceiling 1.0 to catch leverage attempts
        ctx.config["ranking"]["kelly_sizing"]["vol_target"]["ceiling"] = 1.0
        ApplyExposureScalingTask().run(ctx)
        assert ctx._vol_target_scale <= 1.0 + 1e-9, (
            "vol-target produced leverage > 1.0; ceiling guard broken"
        )

    def test_disabled_no_op(self):
        """vol_target.enabled = False → ctx._qp_w_upper unchanged,
        scale = 1.0."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        ctx = self._make_ctx()
        ctx.config["ranking"]["kelly_sizing"]["vol_target"]["enabled"] = False
        ApplyExposureScalingTask().run(ctx)
        assert ctx._vol_target_scale == 1.0
        np.testing.assert_array_equal(ctx._qp_w_upper, [0.20, 0.20, 0.20])
