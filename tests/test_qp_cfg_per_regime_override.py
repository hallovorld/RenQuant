"""Test the per-regime override layer on `_qp_cfg`.

Pins the 2026-05-16 patch that lets `regime_params.<R>.<KEY>` override
the corresponding `rotation.joint_actions.<KEY>` for a curated set of QP
knobs (qp_cvar_lambda, qp_cvar_alpha, qp_turnover_max, qp_risk_aversion).

This is the kernel support for B-track per-regime CVaR experiments. Without
the override, regime_params.BEAR.qp_cvar_lambda was decorative — kernel
never read it. The fix matches the existing PRIME-DIRECTIVE pattern used
by stop_loss_pct / sdl_n_sigma / trailing_stop_trigger_pct in pp_inference.

INVARIANT (5.3): For any key K in _QP_PER_REGIME_KEYS, _qp_cfg(ctx)[K]
returns regime_params[ctx.regime][K] when present; otherwise the
joint_actions[K] global fallback; otherwise the function-default in
_build_solver_kwargs.
"""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.tasks import _qp_cfg, _QP_PER_REGIME_KEYS


def _ctx(regime: str | None, joint: dict, regime_params: dict) -> SimpleNamespace:
    return SimpleNamespace(
        regime=regime,
        config={
            "rotation": {"joint_actions": joint},
            "regime_params": regime_params,
        },
    )


class TestQpCfgPerRegimeOverride:
    """2026-05-16 patch — regime-conditional QP knobs."""

    def test_no_regime_falls_back_to_joint_actions(self):
        ctx = _ctx(regime=None,
                   joint={"qp_cvar_lambda": 0.0, "qp_turnover_max": 0.3},
                   regime_params={})
        cfg = _qp_cfg(ctx)
        assert cfg["qp_cvar_lambda"] == 0.0
        assert cfg["qp_turnover_max"] == 0.3

    def test_regime_with_no_override_falls_back(self):
        ctx = _ctx(regime="BULL_CALM",
                   joint={"qp_cvar_lambda": 0.0},
                   regime_params={"BULL_CALM": {"some_other_knob": 1}})
        cfg = _qp_cfg(ctx)
        assert cfg["qp_cvar_lambda"] == 0.0

    def test_regime_override_for_cvar_lambda(self):
        """BEAR has CVaR=0.25 overlay; CHOPPY has 0.15; BULL has nothing."""
        regime_params = {
            "BEAR":   {"qp_cvar_lambda": 0.25},
            "CHOPPY": {"qp_cvar_lambda": 0.15},
            # BULL_CALM intentionally omitted
        }
        joint = {"qp_cvar_lambda": 0.0}  # global default
        # BEAR → 0.25
        assert _qp_cfg(_ctx("BEAR",   joint, regime_params))["qp_cvar_lambda"] == 0.25
        # CHOPPY → 0.15
        assert _qp_cfg(_ctx("CHOPPY", joint, regime_params))["qp_cvar_lambda"] == 0.15
        # BULL_CALM → fallback 0.0
        assert _qp_cfg(_ctx("BULL_CALM", joint, regime_params))["qp_cvar_lambda"] == 0.0

    def test_all_per_regime_keys_overridable(self):
        """Every key in _QP_PER_REGIME_KEYS gets per-regime read."""
        regime_params = {
            "BEAR": {k: 0.42 for k in _QP_PER_REGIME_KEYS},
        }
        joint = {k: 0.0 for k in _QP_PER_REGIME_KEYS}
        cfg = _qp_cfg(_ctx("BEAR", joint, regime_params))
        for k in _QP_PER_REGIME_KEYS:
            assert cfg[k] == 0.42, f"{k} not overridden"

    def test_unlisted_key_not_overridden(self):
        """A regime_params key that ISN'T in _QP_PER_REGIME_KEYS must not
        leak through (would silently shadow joint_actions otherwise).

        Use a fund/stop-loss knob that lives in regime_params for a
        different reason — _qp_cfg should ignore it.
        """
        regime_params = {
            "BEAR": {
                "stop_loss_pct": 0.07,           # NOT a QP knob
                "qp_cvar_lambda": 0.25,          # IS a QP knob
            },
        }
        joint = {"qp_cvar_lambda": 0.0}
        cfg = _qp_cfg(_ctx("BEAR", joint, regime_params))
        assert cfg["qp_cvar_lambda"] == 0.25
        assert "stop_loss_pct" not in cfg, "non-QP knob leaked into _qp_cfg"

    def test_empty_regime_params_no_crash(self):
        ctx = _ctx(regime="BEAR", joint={"qp_cvar_lambda": 0.1}, regime_params={})
        cfg = _qp_cfg(ctx)
        assert cfg["qp_cvar_lambda"] == 0.1

    def test_missing_config_keys_no_crash(self):
        """Pre-existing behavior: missing rotation.joint_actions returns {}."""
        ctx = SimpleNamespace(regime="BEAR", config={})
        cfg = _qp_cfg(ctx)
        assert cfg == {}

    def test_baseline_unchanged_when_no_overlay(self):
        """REGRESSION GUARD: if no regime_params overlay set, behavior is
        bit-identical to the pre-patch implementation (returns
        joint_actions dict)."""
        joint = {
            "qp_cvar_lambda": 0.0,
            "qp_turnover_max": 0.5,
            "qp_risk_aversion": 3.0,
            "fee_pct": 0.0005,
        }
        ctx = _ctx("BULL_CALM", joint=joint, regime_params={})
        cfg = _qp_cfg(ctx)
        # Same content (dict ordering doesn't matter for ==).
        assert cfg == joint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
