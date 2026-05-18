"""Tests for σ-wire hysteresis (2026-05-17).

Without hysteresis, the 5-day BEAR detector catches brief crises
(SVB/Aug-2024/DeepSeek 1-5 BEAR bars) correctly but σ-wire toggles
ON↔OFF mid-window → strategy churn. 5/17 A/B pre-hysteresis lost
-4.7pp pooled (W7 -21pp single window) where uniform σ-on won +3pp.

Hysteresis design:
  • When live per-regime overlay activates σ-wire → memo overlay +
    arm `sigma_wire_hysteresis_remaining = N` (default 10).
  • Subsequent bars: if no live trigger, decrement counter and keep
    using the memo overlay (sticky σ-wire).
  • After N consecutive non-trigger bars, counter at 0 → σ-wire
    reverts to global defaults (OFF in current config).
  • A retrigger anywhere resets the counter (and memo if changed).

State updates happen in RegimeFinalizeTask (once per bar). _ngb_cfg
is a pure read.
"""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.regime import RegimeState  # noqa: E402
from kernel.panel_pipeline.job_panel_scoring import _ngb_cfg  # noqa: E402
from kernel.pipeline.task_regime import RegimeFinalizeTask  # noqa: E402


def _make_ctx(regime: str, state: RegimeState, ngb_global: dict,
              regime_params: dict, hysteresis_bars: int = 10):
    """Mock ctx with everything RegimeFinalizeTask + _ngb_cfg need."""
    return SimpleNamespace(
        regime=regime,
        regime_state=state,
        config={
            "ranking": {"panel_scoring": {"ngboost": ngb_global}},
            "regime_params": regime_params,
            "regime": {"sigma_wire_hysteresis_bars": hysteresis_bars,
                       "transition_uncertainty_bars": 3,
                       "cusum_cooldown_days": 0.0},
        },
        regime_counts={},
        confidence=0.0,
        today=None,
        ohlcv={},
    )


REGIME_PARAMS = {
    "BEAR":   {"ngboost": {"enabled": True, "score_mode": "mu_minus_lambda_sigma", "lambda_sigma": 1.0}},
    "CHOPPY": {"ngboost": {"enabled": True, "score_mode": "mu_minus_lambda_sigma", "lambda_sigma": 1.0}},
}
NGB_GLOBAL = {"enabled": False, "score_mode": "additive", "lambda_sigma": 0.0}


def _finalize(ctx, new_regime: str):
    """Drive the hysteresis state-update step in isolation.

    RegimeFinalizeTask normally also computes regime — here we want to
    test ONLY the σ-wire state update so we set state.regime then run
    just the hysteresis tail. To do that simply, we invoke
    RegimeFinalizeTask.run with the pre-conditions it needs to resolve
    to `new_regime`.
    """
    # Force the resolution to match new_regime by setting hard_bear if BEAR
    # or using state.hurst_regime + vol_cluster signals.
    s = ctx.regime_state
    if new_regime == "BEAR":
        s.hard_bear = True
    elif new_regime == "CHOPPY":
        s.hard_bear = False; s.hurst_regime = "REVERSION"; s.vol_cluster_choppy = True
    elif new_regime == "BULL_CALM":
        s.hard_bear = False; s.hurst_regime = "MOMENTUM"; s.vol_cluster_choppy = False
    else:
        raise ValueError(new_regime)
    RegimeFinalizeTask().run(ctx)


class TestSigmaWireHysteresis:
    """RegimeFinalizeTask drives state; _ngb_cfg reads state."""

    def test_bear_bar_arms_hysteresis(self):
        s = RegimeState()
        ctx = _make_ctx("BEAR", s, NGB_GLOBAL, REGIME_PARAMS, hysteresis_bars=10)
        _finalize(ctx, "BEAR")
        assert s.sigma_wire_hysteresis_remaining == 10
        assert s.sigma_wire_overlay_memo.get("enabled") is True
        # _ngb_cfg picks up live overlay
        cfg = _ngb_cfg(ctx)
        assert cfg["enabled"] is True
        assert cfg["lambda_sigma"] == 1.0

    def test_hysteresis_persists_into_bull(self):
        """After 1 BEAR bar, next bar is BULL_CALM. σ wire should still
        be ON via memo + counter (decremented to 9)."""
        s = RegimeState()
        ctx = _make_ctx("BEAR", s, NGB_GLOBAL, REGIME_PARAMS)
        _finalize(ctx, "BEAR")
        assert s.sigma_wire_hysteresis_remaining == 10
        # Next bar — BULL_CALM (no live overlay)
        ctx.regime = "BULL_CALM"
        _finalize(ctx, "BULL_CALM")
        assert s.sigma_wire_hysteresis_remaining == 9
        cfg = _ngb_cfg(ctx)
        # _ngb_cfg picks up memo (since no live overlay activates here)
        assert cfg["enabled"] is True
        assert cfg["lambda_sigma"] == 1.0

    def test_hysteresis_decays_to_off_after_N_bars(self):
        """After N consecutive non-trigger bars, σ wire reverts."""
        s = RegimeState()
        ctx = _make_ctx("BEAR", s, NGB_GLOBAL, REGIME_PARAMS, hysteresis_bars=3)
        _finalize(ctx, "BEAR")
        assert s.sigma_wire_hysteresis_remaining == 3
        # 3 BULL_CALM bars in a row → counter 2, 1, 0
        ctx.regime = "BULL_CALM"
        _finalize(ctx, "BULL_CALM"); assert s.sigma_wire_hysteresis_remaining == 2
        _finalize(ctx, "BULL_CALM"); assert s.sigma_wire_hysteresis_remaining == 1
        _finalize(ctx, "BULL_CALM"); assert s.sigma_wire_hysteresis_remaining == 0
        # After counter at 0, _ngb_cfg returns base (σ-wire OFF globally)
        cfg = _ngb_cfg(ctx)
        assert cfg["enabled"] is False
        assert cfg["lambda_sigma"] == 0.0

    def test_retrigger_resets_counter(self):
        s = RegimeState()
        ctx = _make_ctx("BEAR", s, NGB_GLOBAL, REGIME_PARAMS, hysteresis_bars=5)
        _finalize(ctx, "BEAR"); assert s.sigma_wire_hysteresis_remaining == 5
        ctx.regime = "BULL_CALM"
        _finalize(ctx, "BULL_CALM"); assert s.sigma_wire_hysteresis_remaining == 4
        _finalize(ctx, "BULL_CALM"); assert s.sigma_wire_hysteresis_remaining == 3
        # Re-trigger BEAR
        ctx.regime = "BEAR"
        _finalize(ctx, "BEAR")
        assert s.sigma_wire_hysteresis_remaining == 5, "should reset to full N"

    def test_no_overlay_means_no_arming(self):
        """REGRESSION GUARD: if no regime has ngboost overlay, hysteresis
        never arms and σ-wire stays off (baseline behavior preserved)."""
        s = RegimeState()
        regime_params = {"BULL_CALM": {"stop_loss_pct": 0.07}}  # no ngboost
        ctx = _make_ctx("BEAR", s, NGB_GLOBAL, regime_params)
        _finalize(ctx, "BEAR")
        assert s.sigma_wire_hysteresis_remaining == 0
        assert s.sigma_wire_overlay_memo == {}
        cfg = _ngb_cfg(ctx)
        assert cfg["enabled"] is False

    def test_choppy_also_arms_hysteresis(self):
        s = RegimeState()
        ctx = _make_ctx("CHOPPY", s, NGB_GLOBAL, REGIME_PARAMS)
        _finalize(ctx, "CHOPPY")
        assert s.sigma_wire_hysteresis_remaining > 0
        assert s.sigma_wire_overlay_memo.get("enabled") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
