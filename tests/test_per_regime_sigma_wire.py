"""Test the per-regime override layer on NGBoost σ-wire (2026-05-17).

Pins the patch that lets `regime_params.<R>.ngboost.<KEY>` override the
corresponding `ranking.panel_scoring.ngboost.<KEY>` for {enabled,
score_mode, lambda_sigma}.

Why (CLAUDE.md PRIME DIRECTIVE + 5/17 σ-wire A/B):
σ-wire ON globally lost pooled mean across regimes (5/9 E55 -3.78
APY pts; 5/17 dense panel saw -14pp on BULL windows vs +14pp on BEAR/
crisis windows). Per-regime gating lets us turn σ-wire ON in
{BEAR, CHOPPY, BULL_VOLATILE} where it helps, and keep it OFF in
{BULL_CALM, BULL_STRONG} where it hurts. Mirrors the B-track
_qp_cfg pattern (test_qp_cfg_per_regime_override.py).
"""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.panel_pipeline.job_panel_scoring import (  # noqa: E402
    _ngb_cfg, _NGB_PER_REGIME_KEYS,
)


def _ctx(regime: str | None, ngb_global: dict, regime_params: dict) -> SimpleNamespace:
    return SimpleNamespace(
        regime=regime,
        config={
            "ranking": {"panel_scoring": {"ngboost": ngb_global}},
            "regime_params": regime_params,
        },
    )


class TestNgbCfgPerRegimeOverride:
    """2026-05-17 patch — regime-conditional σ-wire activation."""

    def test_no_regime_falls_back_to_global(self):
        ctx = _ctx(regime=None,
                   ngb_global={"enabled": False, "lambda_sigma": 1.0},
                   regime_params={})
        cfg = _ngb_cfg(ctx)
        assert cfg["enabled"] is False
        assert cfg["lambda_sigma"] == 1.0

    def test_regime_with_no_override_falls_back(self):
        ctx = _ctx(regime="BULL_CALM",
                   ngb_global={"enabled": False, "lambda_sigma": 1.0},
                   regime_params={"BULL_CALM": {"some_other_knob": 1}})
        cfg = _ngb_cfg(ctx)
        assert cfg["enabled"] is False

    def test_regime_enables_in_bear_only(self):
        """BEAR overlay → enabled=True; BULL_CALM → fallback to disabled."""
        regime_params = {
            "BEAR":   {"ngboost": {"enabled": True, "score_mode": "mu_minus_lambda_sigma", "lambda_sigma": 1.0}},
            "CHOPPY": {"ngboost": {"enabled": True}},
            # BULL_CALM intentionally omitted → fallback
        }
        ngb_global = {"enabled": False, "score_mode": "additive", "lambda_sigma": 0.0}
        # BEAR → full activation
        c = _ngb_cfg(_ctx("BEAR", ngb_global, regime_params))
        assert c["enabled"] is True
        assert c["score_mode"] == "mu_minus_lambda_sigma"
        assert c["lambda_sigma"] == 1.0
        # CHOPPY → only enabled flipped; score_mode/lambda fall back
        c = _ngb_cfg(_ctx("CHOPPY", ngb_global, regime_params))
        assert c["enabled"] is True
        assert c["score_mode"] == "additive"     # not overridden
        assert c["lambda_sigma"] == 0.0          # not overridden
        # BULL_CALM → fallback to global (disabled)
        c = _ngb_cfg(_ctx("BULL_CALM", ngb_global, regime_params))
        assert c["enabled"] is False

    def test_all_per_regime_keys_overridable(self):
        """Verify all 3 per-regime keys propagate when overlay activates.
        Updated 2026-05-17: overlay only fires when enabled=True (per
        hysteresis-aware _ngb_cfg). Use real-shaped values to verify."""
        regime_params = {"BEAR": {"ngboost": {
            "enabled": True,
            "score_mode": "BEAR_score_mode",
            "lambda_sigma": 0.42,
        }}}
        ngb_global = {
            "enabled": False,
            "score_mode": "global_score_mode",
            "lambda_sigma": 0.0,
        }
        c = _ngb_cfg(_ctx("BEAR", ngb_global, regime_params))
        assert c["enabled"] is True
        assert c["score_mode"] == "BEAR_score_mode"
        assert c["lambda_sigma"] == 0.42

    def test_unlisted_ngb_subkey_not_overridden(self):
        """artifact_path / max_feature_drift_pct are NOT per-regime keys
        (loading is global, drift threshold is invariant)."""
        regime_params = {
            "BEAR": {"ngboost": {
                "enabled": True,
                "artifact_path": "/should/not/take/effect.json",
                "max_feature_drift_pct": 0.99,
            }},
        }
        ngb_global = {
            "enabled": False,
            "artifact_path": "artifacts/prod/ngboost-head.alpha158_fund.json",
            "max_feature_drift_pct": 0.05,
        }
        c = _ngb_cfg(_ctx("BEAR", ngb_global, regime_params))
        assert c["enabled"] is True                           # IS per-regime
        # artifact_path / drift NOT in _NGB_PER_REGIME_KEYS → stay global
        assert c["artifact_path"] == ngb_global["artifact_path"]
        assert c["max_feature_drift_pct"] == 0.05

    def test_baseline_unchanged_when_no_overlay(self):
        """REGRESSION GUARD: if no regime_params.<R>.ngboost overlay, the
        result equals the global config (bit-identical to pre-patch read)."""
        ngb_global = {
            "enabled": False,
            "score_mode": "additive",
            "lambda_sigma": 0.0,
            "artifact_path": "x.json",
        }
        # Multiple regimes, none have ngboost overlay
        ctx = _ctx("BULL_CALM", ngb_global, regime_params={
            "BULL_CALM": {"stop_loss_pct": 0.07},
            "BEAR": {"sdl_n_sigma": 2.0},
        })
        c = _ngb_cfg(ctx)
        assert c == ngb_global, f"expected baseline-identical, got {c}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
