"""Audit-mandated regression test #3 (doc/AUDIT_2026-05-12_dead_paths.md
§"Mandatory regression tests when fix lands"):

> 3. test_vol_target_independent_of_ngb.py — NGB OFF + vol-target ON,
>    assert vol-target HAS an observable effect on _qp_w_upper. (The
>    regression guard against the dead-path bug.)
>
> Without test #3 we'd re-introduce the bug in any future Kelly refactor.

The dead-path bug (2026-05-12): vol-target lived inside
ApplyKellySizingTask and modified a `max_pct` LOCAL VARIABLE that the
QP optimizer never reads. That local was discarded as soon as Kelly
returned, and Kelly itself returned 0 for every candidate when NGB
was OFF (mu_none) → vol-target had ZERO observable effect.

The fix: hoist into ApplyExposureScalingTask which writes
ctx._vol_target_scale × ctx._qp_w_upper at the QP-bound layer,
INDEPENDENT of whether Kelly is functional.

This test pins that decoupling: NGB OFF is the natural production
configuration, and vol-target MUST still scale _qp_w_upper.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


class TestVolTargetIndependentOfNGB:

    def _ctx(self, *, ngb_enabled: bool):
        ctx = SimpleNamespace()
        ctx._qp_w_upper = np.full(3, 0.20)
        ctx.spy_regime = "BULL_VOLATILE"
        ctx.regime = "BULL_VOLATILE"
        ctx.spy_returns = [0.02, -0.02] * 60  # ~32% annualized
        # NGB ON or OFF — vol-target should fire either way
        ctx.config = {
            "ranking": {
                "ngboost": {"enabled": ngb_enabled},
                "kelly_sizing": {
                    "vol_target": {
                        "enabled": True,
                        "target_vol": 0.05,
                        "window_days": 60,
                        "floor": 0.10,
                        "ceiling": 1.50,
                    },
                },
            },
        }
        return ctx

    def test_vol_target_fires_with_ngb_off(self):
        """The DEAD-PATH bug regression guard. NGB OFF means Kelly
        returns 0 for every candidate (mu_none). The OLD code path
        scaled a Kelly local-var that QP never read — so vol-target
        had NO observable effect with NGB off.

        Post-fix, ApplyExposureScalingTask runs INSIDE the QP pipeline
        and writes ctx._qp_w_upper directly. Vol-target MUST shrink
        the upper bound regardless of Kelly's mu_none state.
        """
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask

        ctx = self._ctx(ngb_enabled=False)
        original_upper = ctx._qp_w_upper.copy()

        ApplyExposureScalingTask().run(ctx)

        # MUST observably shrink the upper bound.
        assert (ctx._qp_w_upper < original_upper).all(), (
            f"DEAD-PATH BUG REGRESSION: vol-target with NGB OFF failed "
            f"to scale ctx._qp_w_upper. Original={original_upper}, "
            f"got={ctx._qp_w_upper}. The 2026-05-12 bug class is back: "
            f"vol-target is no longer affecting QP bounds. Audit doc "
            f"AUDIT_2026-05-12_dead_paths.md §1 + §3 mandate."
        )
        # Scale stamped on ctx
        assert ctx._vol_target_scale < 1.0
        assert hasattr(ctx, "_vol_target_scale"), \
            "ctx._vol_target_scale not set — ApplyExposureScalingTask broken"

    def test_vol_target_fires_identically_with_ngb_on(self):
        """Sanity: NGB ON path produces SAME vol-target effect (the scale
        is independent of NGB state by design)."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask

        ctx_off = self._ctx(ngb_enabled=False)
        ctx_on  = self._ctx(ngb_enabled=True)
        ApplyExposureScalingTask().run(ctx_off)
        ApplyExposureScalingTask().run(ctx_on)

        assert ctx_off._vol_target_scale == ctx_on._vol_target_scale, (
            f"vol-target scale differs by NGB state — should be NGB-independent. "
            f"NGB OFF scale={ctx_off._vol_target_scale} "
            f"vs NGB ON scale={ctx_on._vol_target_scale}"
        )
        np.testing.assert_array_equal(ctx_off._qp_w_upper, ctx_on._qp_w_upper)

    def test_dd_kelly_also_independent_of_ngb(self):
        """Same invariant for DD-Kelly: it must scale _qp_w_upper
        regardless of NGB state."""
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask

        for ngb in (False, True):
            ctx = SimpleNamespace()
            ctx._qp_w_upper = np.full(3, 0.20)
            ctx.spy_regime = "BULL_CALM"
            ctx.regime = "BULL_CALM"
            ctx.spy_returns = [0.005, -0.005] * 60
            ctx.hwm = 100.0
            ctx.portfolio_value = 85.0  # 15% drawdown
            ctx.config = {
                "ranking": {
                    "ngboost": {"enabled": ngb},
                    "kelly_sizing": {
                        "drawdown_scaling": {
                            "enabled": True, "dd_max": 0.20, "exponent": 1.0,
                        },
                    },
                },
            }
            ApplyExposureScalingTask().run(ctx)
            assert (ctx._qp_w_upper < 0.20).all(), (
                f"DD-Kelly with NGB={ngb} failed to shrink _qp_w_upper. "
                f"Got: {ctx._qp_w_upper}"
            )
