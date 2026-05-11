"""R-04 regression guard — Grossman-Zhou drawdown-conditioned Kelly scaling.

Pins:
  (1) compute_kelly_dd_scale math: identity at DD=0; zero at DD>=dd_max;
      linear at exponent=1; quadratic at exponent=2; NaN/inf fail-open
      (returns 1.0).
  (2) ApplyKellySizingTaskTask integration: config-gated; off by default
      preserves golden behaviour exactly; on shrinks Kelly target when
      portfolio drawdown breaches the soft cap.

Reference: Grossman & Zhou 1993 JEEM 19(2):241-276 Eq. 8,
``f*(DD) = f_K × max(0, 1 - (DD/DD_max)^k)``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.kelly import compute_kelly_dd_scale, kelly_target_pct  # noqa: E402
from kernel.panel_pipeline.job_panel_scoring import ApplyKellySizingTask  # noqa: E402


class TestKellyDDScaleMath:
    """Pin the closed-form Grossman-Zhou taper."""

    def test_zero_drawdown_returns_one(self):
        assert compute_kelly_dd_scale(0.0, dd_max=0.30) == 1.0

    def test_negative_drawdown_returns_one(self):
        # NaN guard hits; even a clean negative (impossible per
        # compute_portfolio_drawdown) shouldn't oversize.
        assert compute_kelly_dd_scale(-0.05, dd_max=0.30) == 1.0

    def test_at_dd_max_returns_zero(self):
        assert compute_kelly_dd_scale(0.30, dd_max=0.30) == 0.0

    def test_above_dd_max_returns_zero(self):
        assert compute_kelly_dd_scale(0.50, dd_max=0.30) == 0.0

    def test_linear_taper_half_dd_half_size(self):
        # DD=0.15, dd_max=0.30, k=1 → 1 - 0.5 = 0.5
        scale = compute_kelly_dd_scale(0.15, dd_max=0.30, exponent=1.0)
        assert scale == pytest.approx(0.5, abs=1e-9)

    def test_quadratic_taper_defers_derisking(self):
        # DD=0.15, dd_max=0.30, k=2 → 1 - 0.25 = 0.75
        scale = compute_kelly_dd_scale(0.15, dd_max=0.30, exponent=2.0)
        assert scale == pytest.approx(0.75, abs=1e-9)
        # k=2 should be GENTLER at small DD than k=1.
        scale_k1 = compute_kelly_dd_scale(0.15, dd_max=0.30, exponent=1.0)
        assert scale > scale_k1

    def test_invalid_dd_max_is_noop(self):
        assert compute_kelly_dd_scale(0.10, dd_max=0.0) == 1.0
        assert compute_kelly_dd_scale(0.10, dd_max=-0.10) == 1.0

    def test_nan_inputs_failopen_to_one(self):
        # Fail-open per §5.13.11. DrawdownCircuitTask already fail-SAFEs
        # the buy halt; this layer is the soft-cap pre-stage and should
        # not double-penalise a transient NaN signal.
        assert compute_kelly_dd_scale(math.nan, dd_max=0.30) == 1.0
        assert compute_kelly_dd_scale(0.10, dd_max=math.nan) == 1.0
        assert compute_kelly_dd_scale(math.inf, dd_max=0.30) == 1.0


def _ctx(*, hwm=100.0, pv=100.0, regime="BULL_CALM",
         dd_scaling: dict | None = None):
    cfg = {
        "regime_params": {regime: {"max_position_pct": 0.20}},
        "ranking": {
            "kelly_sizing": {
                "enabled": True,
                "fractional": 0.25,
                "min_edge": 0.0,
                "max_concentration": 0.35,
            }
        },
    }
    if dd_scaling is not None:
        cfg["ranking"]["kelly_sizing"]["drawdown_scaling"] = dd_scaling
    return SimpleNamespace(
        config=cfg,
        regime=regime,
        confidence=1.0,
        hwm=hwm,
        portfolio_value=pv,
        candidates=[SimpleNamespace(
            ticker="AAPL", mu=0.06, sigma=0.20,
            kelly_target_pct=None,
        )],
        holdings={},
    )


class TestApplyKellySizingTaskDDIntegration:
    """R-04 regression guard at the Task layer."""

    def test_disabled_block_preserves_golden(self):
        # No drawdown_scaling key → same kelly target as before R-04.
        ctx_off = _ctx(hwm=100.0, pv=70.0)  # 30% DD ignored
        ApplyKellySizingTask().run(ctx_off)
        target_off = ctx_off.candidates[0].kelly_target_pct
        # mu/σ² = 0.06/0.04 = 1.5; 0.25 × 1.5 = 0.375; capped at
        # min(0.20 regime, 0.35 concentration) = 0.20.
        assert target_off == pytest.approx(0.20, abs=1e-9)

    def test_enabled_off_flag_preserves_golden(self):
        ctx_flag_off = _ctx(
            hwm=100.0, pv=70.0,
            dd_scaling={"enabled": False, "dd_max": 0.30},
        )
        ApplyKellySizingTask().run(ctx_flag_off)
        assert ctx_flag_off.candidates[0].kelly_target_pct == pytest.approx(
            0.20, abs=1e-9
        )

    def test_dd_scaling_shrinks_target_at_half_dd(self):
        # PV=85 on HWM=100 → DD=0.15; dd_max=0.30; k=1 → scale=0.5 →
        # max_pct 0.20 → 0.10.
        ctx_dd = _ctx(
            hwm=100.0, pv=85.0,
            dd_scaling={"enabled": True, "dd_max": 0.30, "exponent": 1.0},
        )
        ApplyKellySizingTask().run(ctx_dd)
        assert ctx_dd.candidates[0].kelly_target_pct == pytest.approx(
            0.10, abs=1e-9
        )

    def test_dd_scaling_zero_at_cap(self):
        ctx_cap = _ctx(
            hwm=100.0, pv=70.0,
            dd_scaling={"enabled": True, "dd_max": 0.30, "exponent": 1.0},
        )
        ApplyKellySizingTask().run(ctx_cap)
        assert ctx_cap.candidates[0].kelly_target_pct == 0.0

    def test_dd_scaling_quadratic_is_gentler_at_small_dd(self):
        # DD=0.10 on dd_max=0.30 → k=1 yields 0.667, k=2 yields ~0.889.
        ctx_k1 = _ctx(
            hwm=100.0, pv=90.0,
            dd_scaling={"enabled": True, "dd_max": 0.30, "exponent": 1.0},
        )
        ctx_k2 = _ctx(
            hwm=100.0, pv=90.0,
            dd_scaling={"enabled": True, "dd_max": 0.30, "exponent": 2.0},
        )
        ApplyKellySizingTask().run(ctx_k1)
        ApplyKellySizingTask().run(ctx_k2)
        t1 = ctx_k1.candidates[0].kelly_target_pct
        t2 = ctx_k2.candidates[0].kelly_target_pct
        assert t2 > t1  # quadratic defers de-risking at small DD
