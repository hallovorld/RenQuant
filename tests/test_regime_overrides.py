"""Tests for P1: ranking.<X>.regime_overrides config schema.

Pinned invariants:

1. No `regime_overrides` → base_cfg unchanged. Existing behavior preserved.
2. `regime_overrides` set + ctx.spy_regime=None → base_cfg unchanged.
3. `regime_overrides` set + ctx.spy_regime in overrides → override wins.
4. `regime_overrides` set + ctx.spy_regime NOT in overrides → base_cfg.
5. Override can DISABLE a feature in a toxic regime (`enabled: false`).
6. Override can CHANGE the IC parameter in a favorable regime.

Per doc/research/2026-05-12-findings-and-next.md: regime-conditional
deployment lets us run GK in HIGH_CALM (wins +18%/yr) but disable in
HIGH_SPIKED (loses -32%/yr) without affecting other regimes.

§5.13.10: this code path is ONLY active when SpyRegimeLabelTask is
enabled (ctx.spy_regime != None). When P0-A is off, regime_overrides
is dead code (falls back to base_cfg).
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


class TestResolveRegimeOverride:

    def test_no_overrides_returns_base(self):
        from kernel.portfolio_qp.tasks import _resolve_regime_override
        ctx = SimpleNamespace(spy_regime="HIGH_CALM")
        base = {"enabled": True, "ic": 0.094}
        assert _resolve_regime_override(base, ctx) == base

    def test_regime_none_returns_base(self):
        from kernel.portfolio_qp.tasks import _resolve_regime_override
        ctx = SimpleNamespace(spy_regime=None)
        base = {"enabled": True, "ic": 0.094,
                "regime_overrides": {"HIGH_CALM": {"ic": 0.20}}}
        # Override exists but no regime label set → fall back to base
        result = _resolve_regime_override(base, ctx)
        assert result["ic"] == 0.094

    def test_regime_matches_override_wins(self):
        from kernel.portfolio_qp.tasks import _resolve_regime_override
        ctx = SimpleNamespace(spy_regime="HIGH_CALM")
        base = {"enabled": True, "ic": 0.094,
                "regime_overrides": {"HIGH_CALM": {"ic": 0.20}}}
        result = _resolve_regime_override(base, ctx)
        assert result["ic"] == 0.20
        assert result["enabled"] is True   # not overridden, kept

    def test_regime_not_in_overrides_falls_back(self):
        from kernel.portfolio_qp.tasks import _resolve_regime_override
        ctx = SimpleNamespace(spy_regime="MED_NORMAL")
        base = {"enabled": True, "ic": 0.094,
                "regime_overrides": {"HIGH_CALM": {"ic": 0.20}}}
        result = _resolve_regime_override(base, ctx)
        assert result["ic"] == 0.094

    def test_override_can_disable_in_toxic_regime(self):
        """The key conditional-deployment use case: disable GK in HIGH_SPIKED."""
        from kernel.portfolio_qp.tasks import _resolve_regime_override
        ctx = SimpleNamespace(spy_regime="HIGH_SPIKED")
        base = {"enabled": True, "ic": 0.094,
                "regime_overrides": {"HIGH_SPIKED": {"enabled": False}}}
        result = _resolve_regime_override(base, ctx)
        assert result["enabled"] is False  # disabled in HIGH_SPIKED

    def test_missing_spy_regime_attr_safe(self):
        """ctx without spy_regime attr (legacy SimAdapter without P0-A) → base."""
        from kernel.portfolio_qp.tasks import _resolve_regime_override
        ctx = SimpleNamespace()
        # No spy_regime attribute set
        base = {"enabled": True, "ic": 0.094,
                "regime_overrides": {"HIGH_CALM": {"ic": 0.20}}}
        result = _resolve_regime_override(base, ctx)
        assert result["ic"] == 0.094  # base wins

    def test_non_dict_override_falls_back(self):
        from kernel.portfolio_qp.tasks import _resolve_regime_override
        ctx = SimpleNamespace(spy_regime="HIGH_CALM")
        base = {"enabled": True, "ic": 0.094,
                "regime_overrides": {"HIGH_CALM": "not_a_dict"}}
        result = _resolve_regime_override(base, ctx)
        assert result["ic"] == 0.094

    def test_empty_overrides_block(self):
        from kernel.portfolio_qp.tasks import _resolve_regime_override
        ctx = SimpleNamespace(spy_regime="HIGH_CALM")
        base = {"enabled": True, "ic": 0.094, "regime_overrides": {}}
        result = _resolve_regime_override(base, ctx)
        assert result["ic"] == 0.094


class TestGKTaskWithRegimeOverrides:
    """Pin §5.13.10: conditional deployment via regime_overrides must
    fire end-to-end through ApplyGrinoldKahnTransformTask."""

    def _make_ctx(self, spy_regime, gk_cfg):
        ctx = SimpleNamespace()
        ctx._qp_mu = np.array([+1.0, 0.0, -1.0])
        ctx._qp_sigma = np.array([0.05, 0.05, 0.05])
        ctx.spy_regime = spy_regime
        ctx.config = {"ranking": {"alpha_to_mu": gk_cfg}}
        return ctx

    def test_disabled_in_high_spiked_does_nothing(self):
        from kernel.portfolio_qp.tasks import ApplyGrinoldKahnTransformTask
        cfg = {"enabled": True, "ic": 0.094,
               "regime_overrides": {"HIGH_SPIKED": {"enabled": False}}}
        ctx = self._make_ctx("HIGH_SPIKED", cfg)
        before = ctx._qp_mu.copy()
        ApplyGrinoldKahnTransformTask().run(ctx)
        # Override disabled GK in HIGH_SPIKED → μ unchanged
        np.testing.assert_array_equal(ctx._qp_mu, before)

    def test_enabled_in_high_calm_with_override_ic(self):
        from kernel.portfolio_qp.tasks import ApplyGrinoldKahnTransformTask
        cfg = {"enabled": True, "ic": 0.094,
               "regime_overrides": {"HIGH_CALM": {"ic": 0.50}}}
        ctx = self._make_ctx("HIGH_CALM", cfg)
        ApplyGrinoldKahnTransformTask().run(ctx)
        # IC=0.50 (overridden) instead of 0.094 (base)
        # z = [+1, 0, -1] / std = [+1, 0, -1], so μ_QP = 0.50 × 0.05 × [+1,0,-1]
        assert ctx._qp_mu[0] == pytest.approx(0.025, abs=1e-9)
        assert ctx._qp_mu[2] == pytest.approx(-0.025, abs=1e-9)

    def test_fallback_to_base_when_regime_not_in_overrides(self):
        from kernel.portfolio_qp.tasks import ApplyGrinoldKahnTransformTask
        cfg = {"enabled": True, "ic": 0.10,
               "regime_overrides": {"HIGH_CALM": {"ic": 0.50}}}
        # ctx in MED_NORMAL — not in overrides → base IC=0.10
        ctx = self._make_ctx("MED_NORMAL", cfg)
        ApplyGrinoldKahnTransformTask().run(ctx)
        # base IC=0.10
        assert ctx._qp_mu[0] == pytest.approx(0.005, abs=1e-9)
        assert ctx._qp_mu[2] == pytest.approx(-0.005, abs=1e-9)


class TestExposureScalingWithRegimeOverrides:
    """vol_target / drawdown_scaling can also be regime-conditional."""

    def test_vol_target_disabled_in_specific_regime(self):
        from kernel.portfolio_qp.tasks import ApplyExposureScalingTask
        ctx = SimpleNamespace()
        ctx._qp_w_upper = np.array([0.20, 0.20])
        ctx.spy_regime = "HIGH_SPIKED"
        ctx.spy_returns = [0.02, -0.02] * 60  # high vol
        ctx.config = {
            "ranking": {"kelly_sizing": {
                "vol_target": {"enabled": True, "target_vol": 0.05,
                               "window_days": 60, "floor": 0.30,
                               "regime_overrides": {
                                   "HIGH_SPIKED": {"enabled": False},
                               }},
            }},
        }
        ApplyExposureScalingTask().run(ctx)
        # vol_target disabled in HIGH_SPIKED → scale = 1.0
        assert ctx._vol_target_scale == 1.0
        np.testing.assert_array_equal(ctx._qp_w_upper, [0.20, 0.20])
