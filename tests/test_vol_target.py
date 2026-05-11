"""R-02 regression guard — Moskowitz-Ooi-Pedersen 2012 vol-targeting.

Pins:
  (1) compute_vol_target_scale math: identity at realized=target; halves
      when realized=2×target; doubles when realized=0.5×target (within
      ceiling); clipped at floor / ceiling.
  (2) ApplyKellySizingTask integration: config-gated; off preserves
      golden exactly; on shrinks max_pct when SPY 60d σ > target.
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

from kernel.vol_target import compute_vol_target_scale  # noqa: E402
from kernel.panel_pipeline.job_panel_scoring import ApplyKellySizingTask  # noqa: E402


def _spy_returns_with_vol(target_vol: float, n: int = 60) -> list[float]:
    """Construct a deterministic SPY returns list whose annualised σ is
    EXACTLY `target_vol`. Alternating +d / −d returns yield daily σ = d,
    annualised = d × √252.
    """
    d = target_vol / math.sqrt(252.0)
    seq = []
    for i in range(n):
        seq.append(d if i % 2 == 0 else -d)
    return seq


class TestComputeVolTargetScaleMath:
    """Pin the closed-form Moskowitz-Ooi-Pedersen scale."""

    # Bessel correction: realized σ on n=60 with our constructor is
    # off by √(n/(n-1)) ≈ 1.0085 vs the target. Tolerance reflects that.

    def test_realized_equals_target_returns_one(self):
        rets = _spy_returns_with_vol(0.15, n=60)
        s = compute_vol_target_scale(rets, target_vol=0.15, window_days=60,
                                      floor=0.1, ceiling=3.0)
        # Within 1% (Bessel-correction inflation).
        assert s == pytest.approx(1.0, rel=0.02)

    def test_realized_double_target_halves_scale(self):
        rets = _spy_returns_with_vol(0.30, n=60)   # realized 30%
        s = compute_vol_target_scale(rets, target_vol=0.15, window_days=60,
                                      floor=0.01, ceiling=10.0)
        assert s == pytest.approx(0.5, rel=0.02)

    def test_realized_half_target_doubles_scale_within_ceiling(self):
        rets = _spy_returns_with_vol(0.075, n=60)  # realized 7.5%
        s = compute_vol_target_scale(rets, target_vol=0.15, window_days=60,
                                      floor=0.30, ceiling=5.0)
        assert s == pytest.approx(2.0, rel=0.02)

    def test_clipped_at_ceiling(self):
        rets = _spy_returns_with_vol(0.05, n=60)
        s = compute_vol_target_scale(rets, target_vol=0.15, window_days=60,
                                      floor=0.30, ceiling=1.50)
        assert s == pytest.approx(1.50, abs=1e-6)

    def test_clipped_at_floor(self):
        rets = _spy_returns_with_vol(0.80, n=60)
        s = compute_vol_target_scale(rets, target_vol=0.15, window_days=60,
                                      floor=0.30, ceiling=1.50)
        assert s == pytest.approx(0.30, abs=1e-6)

    def test_too_few_returns_failopen(self):
        s = compute_vol_target_scale([0.001] * 5, target_vol=0.15,
                                      window_days=60, floor=0.30,
                                      ceiling=1.50)
        assert s == 1.0

    def test_empty_returns_failopen(self):
        s = compute_vol_target_scale([], target_vol=0.15, window_days=60)
        assert s == 1.0

    def test_nan_target_failopen(self):
        rets = _spy_returns_with_vol(0.20, n=60)
        s = compute_vol_target_scale(rets, target_vol=math.nan, window_days=60)
        assert s == 1.0

    def test_skips_nonfinite_returns(self):
        rets = _spy_returns_with_vol(0.30, n=60)
        # Sprinkle nan/inf into the tail; the helper must skip them and
        # still report a valid scale instead of NaN.
        rets[-2] = math.nan
        rets[-3] = math.inf
        s = compute_vol_target_scale(rets, target_vol=0.15, window_days=60,
                                      floor=0.01, ceiling=10.0)
        assert math.isfinite(s) and 0.4 < s < 0.6


def _ctx(*, spy_returns, vt_cfg=None):
    cfg = {
        "regime_params": {"BULL_CALM": {"max_position_pct": 0.20}},
        "ranking": {
            "kelly_sizing": {
                "enabled": True,
                "fractional": 0.25,
                "min_edge": 0.0,
                "max_concentration": 0.50,
            }
        },
    }
    if vt_cfg is not None:
        cfg["ranking"]["kelly_sizing"]["vol_target"] = vt_cfg
    return SimpleNamespace(
        config=cfg,
        regime="BULL_CALM",
        confidence=1.0,
        hwm=100.0,
        portfolio_value=100.0,
        spy_returns=spy_returns,
        candidates=[SimpleNamespace(
            ticker="AAPL", mu=0.06, sigma=0.20,
            kelly_target_pct=None,
        )],
        holdings={},
    )


class TestApplyKellySizingVolTargetIntegration:

    def test_disabled_block_preserves_golden(self):
        rets = _spy_returns_with_vol(0.30, n=60)
        ctx_off = _ctx(spy_returns=rets)
        ApplyKellySizingTask().run(ctx_off)
        # f* = mu/σ² = 0.06/0.04 = 1.5; quarter Kelly = 0.375; capped to
        # min(0.20 regime, 0.50 concentration) = 0.20.
        assert ctx_off.candidates[0].kelly_target_pct == pytest.approx(0.20)

    def test_realized_double_target_halves_target(self):
        rets = _spy_returns_with_vol(0.30, n=60)  # vol = 30%
        ctx_on = _ctx(spy_returns=rets, vt_cfg={
            "enabled": True, "target_vol": 0.15, "window_days": 60,
            "floor": 0.01, "ceiling": 10.0,
        })
        ApplyKellySizingTask().run(ctx_on)
        # Scale≈0.5 (Bessel correction → ~0.496) → max_pct 0.20 → ~0.099.
        # Kelly cap also bounds it; allow 2% tolerance.
        assert ctx_on.candidates[0].kelly_target_pct == pytest.approx(0.10, rel=0.02)

    def test_realized_half_target_does_not_increase_kelly_above_concentration(self):
        rets = _spy_returns_with_vol(0.075, n=60)  # vol = 7.5%
        ctx_on = _ctx(spy_returns=rets, vt_cfg={
            "enabled": True, "target_vol": 0.15, "window_days": 60,
            "floor": 0.30, "ceiling": 5.0,
        })
        ApplyKellySizingTask().run(ctx_on)
        # Scale=2.0, max_pct → 0.40. Kelly raw=0.375; max_concentration=0.50.
        # Kelly cap = min(0.40, 0.50, 0.375) = 0.375.
        assert ctx_on.candidates[0].kelly_target_pct == pytest.approx(0.375)
