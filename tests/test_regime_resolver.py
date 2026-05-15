"""Per-regime config knob resolver (P1a foundation).

Tests the resolution-order contract of `kernel.regime_resolver.resolve_regime_knob`:
  1. regime_params.<regime>.<knob>     (overlay wins)
  2. <top_section>.<knob>                (global default)
  3. <knob>                              (top-level, if no top_section)
  4. default                             (fallback)

This is the architectural pattern every regime-conditional knob uses.
A bug here silently breaks every P1 wiring downstream.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.regime_resolver import resolve_regime_knob  # noqa: E402


def _ctx(config: dict, regime: str | None = "BULL_CALM"):
    return SimpleNamespace(config=config, regime=regime)


class TestResolverPrecedence:

    def test_overlay_wins_over_top_section(self):
        """Overlay key for `long_short.enabled` is `long_short_enabled`
        (top_section + knob, underscore-joined to avoid key collisions)."""
        ctx = _ctx({
            "regime_params": {"BEAR": {"long_short_enabled": True}},
            "long_short": {"enabled": False},
        }, regime="BEAR")
        assert resolve_regime_knob(ctx, "long_short", "enabled", default=False) is True

    def test_top_section_wins_over_default(self):
        ctx = _ctx({"long_short": {"enabled": True}}, regime="BEAR")
        assert resolve_regime_knob(ctx, "long_short", "enabled", default=False) is True

    def test_default_returned_when_no_match(self):
        ctx = _ctx({}, regime="BEAR")
        assert resolve_regime_knob(ctx, "long_short", "enabled", default=False) is False
        assert resolve_regime_knob(ctx, "long_short", "max_short_pct", default=0.05) == 0.05

    def test_overlay_for_one_regime_not_other(self):
        ctx_bear = _ctx({
            "regime_params": {
                "BEAR":      {"long_short_enabled": True},
                "BULL_CALM": {"long_short_enabled": False},
            },
            "long_short": {"enabled": False},
        }, regime="BEAR")
        ctx_bull = _ctx(ctx_bear.config, regime="BULL_CALM")
        assert resolve_regime_knob(ctx_bear, "long_short", "enabled", default=False) is True
        assert resolve_regime_knob(ctx_bull, "long_short", "enabled", default=False) is False

    def test_overlay_with_falsy_value_still_wins(self):
        """Bug guard: overlay value of 0 / False / empty must override
        a non-zero global default. `if key in overlay` (not truthy check)."""
        ctx = _ctx({
            "regime_params": {"BULL_STRONG": {"long_short_max_short_pct": 0.0}},
            "long_short": {"max_short_pct": 0.05},
        }, regime="BULL_STRONG")
        assert resolve_regime_knob(
            ctx, "long_short", "max_short_pct", default=0.05,
        ) == 0.0

    def test_no_top_section_reads_toplevel_key(self):
        """When top_section=None, overlay key is just `knob` and we look up
        config[knob] directly."""
        ctx = _ctx({
            "regime_params": {"BEAR": {"max_position_pct": 0.0}},
            "max_position_pct": 0.20,
        }, regime="BEAR")
        assert resolve_regime_knob(ctx, None, "max_position_pct", default=0.20) == 0.0

    def test_none_regime_skips_overlay(self):
        ctx = _ctx({
            "regime_params": {"BEAR": {"long_short_enabled": True}},
            "long_short": {"enabled": False},
        }, regime=None)
        assert resolve_regime_knob(ctx, "long_short", "enabled", default=False) is False

    def test_explicit_regime_kwarg_overrides_ctx_regime(self):
        ctx = _ctx({
            "regime_params": {
                "BEAR":      {"long_short_enabled": True},
                "BULL_CALM": {"long_short_enabled": False},
            },
        }, regime="BULL_CALM")
        # Explicit override looks up BEAR's value despite ctx.regime=BULL_CALM
        assert resolve_regime_knob(
            ctx, "long_short", "enabled", default=None, regime="BEAR",
        ) is True

    def test_explicit_overlay_key_kwarg_overrides_default_construction(self):
        """For unusual cases where the overlay key doesn't follow the default
        `<top_section>_<knob>` convention."""
        ctx = _ctx({
            "regime_params": {"BEAR": {"shorts_on": True}},  # custom legacy key
        }, regime="BEAR")
        assert resolve_regime_knob(
            ctx, "long_short", "enabled", default=False, overlay_key="shorts_on",
        ) is True


class TestQPConstraintsP1aWiring:
    """The first per-regime knob wired into production: long_short.enabled
    plus the BEAR-OFFENSIVE / BEAR-DEFENSIVE hybrid routed by hard_bear."""

    def _ctx(self, *, regime, hard_bear=False, long_short=None,
              regime_params=None, rotation=None):
        from kernel.regime import RegimeState
        rs = RegimeState()
        rs.hard_bear = hard_bear
        rp = regime_params or {regime: {"max_position_pct": 0.20}}
        cfg = {
            "regime_params": rp,
            "rotation": rotation or {"joint_actions": {}},
        }
        if long_short is not None:
            cfg["long_short"] = long_short
        ctx = SimpleNamespace(
            config=cfg,
            regime=regime,
            regime_state=rs,
            confidence=1.0,
            candidates=[SimpleNamespace(ticker=f"T{i}") for i in range(5)],
            holdings={},
        )
        ctx._qp_tickers = ["AAA", "BBB", "CCC"]
        return ctx

    def test_global_shorts_off_no_regime_overlay_path_long_only(self):
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        ctx = self._ctx(regime="BULL_CALM",
                        long_short={"enabled": False})
        ComputeQPConstraintsTask().run(ctx)
        assert ctx._qp_w_lower == 0.0
        assert ctx._qp_gross_max is None

    def test_overlay_enables_shorts_in_choppy_only(self):
        """Overlay enables shorts in CHOPPY but global stays False."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        cfg = {
            "long_short": {"enabled": False, "max_short_pct": 0.05},
            "regime_params": {
                "CHOPPY":     {"max_position_pct": 0.15, "long_short_enabled": True},
                "BULL_STRONG":{"max_position_pct": 0.15},
            },
        }
        from kernel.regime import RegimeState
        rs = RegimeState()
        rs.hard_bear = False
        ctx_choppy = SimpleNamespace(
            config=cfg, regime="CHOPPY", regime_state=rs, confidence=1.0,
            candidates=[SimpleNamespace(ticker=f"T{i}") for i in range(3)],
            holdings={},
        )
        ctx_choppy._qp_tickers = ["AAA", "BBB", "CCC"]
        ctx_bull = SimpleNamespace(
            config=cfg, regime="BULL_STRONG", regime_state=rs, confidence=1.0,
            candidates=[SimpleNamespace(ticker=f"T{i}") for i in range(3)],
            holdings={},
        )
        ctx_bull._qp_tickers = ["AAA", "BBB", "CCC"]
        ComputeQPConstraintsTask().run(ctx_choppy)
        ComputeQPConstraintsTask().run(ctx_bull)
        assert ctx_choppy._qp_w_lower < 0, "CHOPPY overlay should enable shorts"
        assert ctx_bull._qp_w_lower == 0, "BULL_STRONG should stay long-only"

    def test_bear_slow_keeps_defensive_mode(self):
        """regime=BEAR + hard_bear=False (slow bear) → DEFENSIVE preserved
        (no shorts; existing bear_defensive_slots path picks up GLD/TLT)."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        ctx = self._ctx(
            regime="BEAR",
            hard_bear=False,
            long_short={"enabled": True, "max_short_pct": 0.05},
            regime_params={"BEAR": {"max_position_pct": 0.0}},
        )
        ComputeQPConstraintsTask().run(ctx)
        assert ctx._qp_w_lower == 0.0, "slow BEAR must not allow shorts"
        assert ctx._qp_gross_max is None

    def test_bear_sharp_routes_to_offensive_shorts(self):
        """regime=BEAR + hard_bear=True (sharp bear) → OFFENSIVE mode:
        longs already blocked by max_position_pct=0; shorts allowed at QP."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        ctx = self._ctx(
            regime="BEAR",
            hard_bear=True,
            long_short={"enabled": True, "max_short_pct": 0.08,
                        "max_gross_exposure": 1.0},
            regime_params={"BEAR": {"max_position_pct": 0.0}},
        )
        ComputeQPConstraintsTask().run(ctx)
        assert ctx._qp_w_lower < 0, (
            f"OFFENSIVE mode must enable shorts, got w_lower={ctx._qp_w_lower}"
        )
        assert abs(ctx._qp_w_lower - (-0.08)) < 1e-6, (
            f"max_short_pct=0.08 should produce w_lower=-0.08, got {ctx._qp_w_lower}"
        )
        # Longs already blocked at upper bound:
        assert ctx._qp_w_upper.max() == 0.0
        # Hardcap still enforced:
        assert ctx._qp_gross_max == 1.0

    def test_bull_strong_disable_via_overlay_kills_shorts_there(self):
        """Global enabled, but BULL_STRONG overlay turns it off — catastrophes
        Q07/Q11 (Δ_APY -20pt) are eliminated by this path."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        cfg = {
            "long_short": {"enabled": True, "max_short_pct": 0.05},
            "regime_params": {
                "BULL_STRONG": {"max_position_pct": 0.15, "long_short_enabled": False},
            },
        }
        from kernel.regime import RegimeState
        rs = RegimeState()
        rs.hard_bear = False
        ctx = SimpleNamespace(
            config=cfg, regime="BULL_STRONG", regime_state=rs, confidence=1.0,
            candidates=[SimpleNamespace(ticker=f"T{i}") for i in range(3)],
            holdings={},
        )
        ctx._qp_tickers = ["AAA", "BBB", "CCC"]
        ComputeQPConstraintsTask().run(ctx)
        assert ctx._qp_w_lower == 0.0, "BULL_STRONG overlay must veto shorts"

    def test_regime_specific_max_short_pct_overlay(self):
        """Different regimes get different short-position sizing via the
        `<top_section>_<knob>` overlay convention."""
        from kernel.portfolio_qp.tasks import ComputeQPConstraintsTask
        cfg = {
            "long_short": {"enabled": True, "max_short_pct": 0.05},
            "regime_params": {
                "CHOPPY":   {"max_position_pct": 0.15, "long_short_max_short_pct": 0.03},
                "BULL_VOL": {"max_position_pct": 0.15, "long_short_max_short_pct": 0.08},
            },
        }
        from kernel.regime import RegimeState
        rs = RegimeState()
        rs.hard_bear = False
        for regime, expected in [("CHOPPY", -0.03), ("BULL_VOL", -0.08)]:
            ctx = SimpleNamespace(
                config=cfg, regime=regime, regime_state=rs, confidence=1.0,
                candidates=[SimpleNamespace(ticker=f"T{i}") for i in range(3)],
                holdings={},
            )
            ctx._qp_tickers = ["AAA", "BBB", "CCC"]
            ComputeQPConstraintsTask().run(ctx)
            assert abs(ctx._qp_w_lower - expected) < 1e-6, (
                f"{regime}: expected w_lower={expected}, got {ctx._qp_w_lower}"
            )
