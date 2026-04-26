"""Behavior tests for PanelRankVetoTask (2026-04-26).

Per user: "GOOG/AMZN model_sell while panel says strong → architectural
flaw." This task vetoes model_sell exits when held.rank_score is
above threshold.
"""
from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.pipeline.task_panel_veto import (  # noqa: E402
    PanelRankVetoTask,
    RISK_EXIT_TYPES,
)


@dataclass
class _Hold:
    rank_score: float | None = None
    sell_streak: int = 0


@dataclass
class _Exit:
    should_exit: bool = True
    reason: str = ""
    exit_type: str = "model_sell"


@dataclass
class _Ctx:
    config: dict = field(default_factory=dict)
    holdings: dict = field(default_factory=dict)
    exits: list = field(default_factory=list)
    counters: dict = field(default_factory=dict)


def _cfg(enabled=True, threshold=0.50, vetoable=None):
    out = {"model_sell": {"panel_veto": {
        "enabled": enabled,
        "min_rank_score": threshold,
    }}}
    if vetoable:
        out["model_sell"]["panel_veto"]["vetoable_exit_types"] = vetoable
    return out


class TestPanelRankVetoFlagGate:
    def test_default_off_returns_false(self):
        ctx = _Ctx(config={"model_sell": {}})
        assert PanelRankVetoTask().run(ctx) is False

    def test_enabled_with_no_exits_returns_false(self):
        ctx = _Ctx(config=_cfg(enabled=True))
        assert PanelRankVetoTask().run(ctx) is False

    def test_explicitly_disabled(self):
        ctx = _Ctx(config=_cfg(enabled=False))
        ctx.holdings["GOOG"] = _Hold(rank_score=0.95)
        ctx.exits = [("GOOG", _Exit(exit_type="model_sell"))]
        result = PanelRankVetoTask().run(ctx)
        assert result is False
        # Exit NOT vetoed
        assert len(ctx.exits) == 1


class TestModelSellVetoed:
    def test_strong_panel_vetoes_model_sell(self):
        """GOOG-style: model_sell while panel rank_score > threshold → veto."""
        ctx = _Ctx(config=_cfg(threshold=0.50))
        ctx.holdings["GOOG"] = _Hold(rank_score=0.85)  # strong
        ctx.exits = [("GOOG", _Exit(exit_type="model_sell"))]
        PanelRankVetoTask().run(ctx)
        assert len(ctx.exits) == 0, "model_sell should have been vetoed"
        assert ctx.counters.get("model_sell_vetoed") == 1
        assert hasattr(ctx, "exits_vetoed")
        assert ctx.exits_vetoed[0]["ticker"] == "GOOG"
        assert ctx.exits_vetoed[0]["rank_score"] == 0.85
        assert ctx.exits_vetoed[0]["threshold"] == 0.50

    def test_weak_panel_does_not_veto(self):
        """held.rank_score below threshold → exit fires normally."""
        ctx = _Ctx(config=_cfg(threshold=0.50))
        ctx.holdings["WEAK"] = _Hold(rank_score=0.10)
        ctx.exits = [("WEAK", _Exit(exit_type="model_sell"))]
        PanelRankVetoTask().run(ctx)
        # Exit NOT vetoed
        assert len(ctx.exits) == 1
        assert ctx.counters.get("model_sell_vetoed", 0) == 0

    def test_at_threshold_does_not_veto(self):
        """rank_score exactly at threshold → still fires (strict >)."""
        ctx = _Ctx(config=_cfg(threshold=0.50))
        ctx.holdings["EQ"] = _Hold(rank_score=0.50)
        ctx.exits = [("EQ", _Exit(exit_type="model_sell"))]
        PanelRankVetoTask().run(ctx)
        assert len(ctx.exits) == 1, "tie should NOT veto (strict >)"


class TestRiskExitsNeverVetoed:
    """Stop-loss / trailing-stop / max-hold MUST fire regardless of panel."""

    @pytest.mark.parametrize("exit_type", sorted(RISK_EXIT_TYPES))
    def test_risk_exit_fires_even_when_strong(self, exit_type):
        ctx = _Ctx(config=_cfg(threshold=0.50))
        ctx.holdings["X"] = _Hold(rank_score=0.99)  # MAX strong
        ctx.exits = [("X", _Exit(exit_type=exit_type))]
        PanelRankVetoTask().run(ctx)
        assert len(ctx.exits) == 1, (
            f"{exit_type} is risk-driven, must NEVER be vetoed"
        )
        assert ctx.counters.get("model_sell_vetoed", 0) == 0


class TestNaNGuard:
    def test_nan_rank_score_does_not_veto(self):
        """Held with NaN rank_score → exit fires (safe default)."""
        ctx = _Ctx(config=_cfg(threshold=0.50))
        ctx.holdings["NAN"] = _Hold(rank_score=float("nan"))
        ctx.exits = [("NAN", _Exit(exit_type="model_sell"))]
        PanelRankVetoTask().run(ctx)
        assert len(ctx.exits) == 1, "NaN rank_score should not veto (safe default)"

    def test_none_rank_score_does_not_veto(self):
        ctx = _Ctx(config=_cfg(threshold=0.50))
        ctx.holdings["NONE"] = _Hold(rank_score=None)
        ctx.exits = [("NONE", _Exit(exit_type="model_sell"))]
        PanelRankVetoTask().run(ctx)
        assert len(ctx.exits) == 1


class TestMultipleExits:
    def test_mixed_veto_and_keep(self):
        """3 exits: strong model_sell vetoed, weak model_sell kept, stop_loss kept."""
        ctx = _Ctx(config=_cfg(threshold=0.50))
        ctx.holdings["STRONG"] = _Hold(rank_score=0.90)  # vetoed
        ctx.holdings["WEAK"]   = _Hold(rank_score=0.20)  # not vetoed
        ctx.holdings["STOP"]   = _Hold(rank_score=0.95)  # stop_loss → never vetoed
        ctx.exits = [
            ("STRONG", _Exit(exit_type="model_sell")),
            ("WEAK",   _Exit(exit_type="model_sell")),
            ("STOP",   _Exit(exit_type="stop_loss")),
        ]
        PanelRankVetoTask().run(ctx)
        kept_tickers = {t for t, _ in ctx.exits}
        assert kept_tickers == {"WEAK", "STOP"}
        assert ctx.counters.get("model_sell_vetoed") == 1


class TestVetoableSet:
    """Operator can extend vetoable_exit_types beyond default ['model_sell']."""

    def test_default_vetoable_is_model_sell(self):
        ctx = _Ctx(config=_cfg(threshold=0.50))
        ctx.holdings["X"] = _Hold(rank_score=0.95)
        ctx.exits = [("X", _Exit(exit_type="thesis_degradation"))]
        PanelRankVetoTask().run(ctx)
        # Not in vetoable set → exit fires
        assert len(ctx.exits) == 1

    def test_custom_vetoable_set(self):
        ctx = _Ctx(config=_cfg(
            threshold=0.50,
            vetoable=["model_sell", "thesis_degradation"],
        ))
        ctx.holdings["X"] = _Hold(rank_score=0.95)
        ctx.exits = [("X", _Exit(exit_type="thesis_degradation"))]
        PanelRankVetoTask().run(ctx)
        # Custom vetoable set → vetoed
        assert len(ctx.exits) == 0


class TestStreakCap:
    """PV-NEW-7: persistent weakness override after N consecutive sell signals."""

    def test_streak_below_cap_vetoed(self):
        """sell_streak=3, cap=5 → still vetoed."""
        cfg = {"model_sell": {"panel_veto": {
            "enabled": True, "min_rank_score": 0.50,
            "max_streak_to_veto": 5,
        }}}
        ctx = _Ctx(config=cfg)
        ctx.holdings["X"] = _Hold(rank_score=0.85, sell_streak=3)
        ctx.exits = [("X", _Exit(exit_type="model_sell"))]
        PanelRankVetoTask().run(ctx)
        assert len(ctx.exits) == 0, "still under streak cap → veto"

    def test_streak_at_cap_bypasses(self):
        """sell_streak=5, cap=5 → veto bypassed."""
        cfg = {"model_sell": {"panel_veto": {
            "enabled": True, "min_rank_score": 0.50,
            "max_streak_to_veto": 5,
        }}}
        ctx = _Ctx(config=cfg)
        ctx.holdings["X"] = _Hold(rank_score=0.85, sell_streak=5)
        ctx.exits = [("X", _Exit(exit_type="model_sell"))]
        PanelRankVetoTask().run(ctx)
        assert len(ctx.exits) == 1, "streak hit cap → exit fires"
        assert ctx.counters.get("model_sell_veto_bypassed") == 1

    def test_streak_above_cap_bypasses(self):
        """sell_streak=10, cap=5 → veto bypassed (streak >> cap)."""
        cfg = {"model_sell": {"panel_veto": {
            "enabled": True, "min_rank_score": 0.50,
            "max_streak_to_veto": 5,
        }}}
        ctx = _Ctx(config=cfg)
        ctx.holdings["X"] = _Hold(rank_score=0.99, sell_streak=10)
        ctx.exits = [("X", _Exit(exit_type="model_sell"))]
        PanelRankVetoTask().run(ctx)
        assert len(ctx.exits) == 1
        assert ctx.counters.get("model_sell_veto_bypassed") == 1

    def test_max_streak_zero_disables_cap(self):
        """max_streak_to_veto=0 → no cap; veto always fires when strong."""
        cfg = {"model_sell": {"panel_veto": {
            "enabled": True, "min_rank_score": 0.50,
            "max_streak_to_veto": 0,
        }}}
        ctx = _Ctx(config=cfg)
        ctx.holdings["X"] = _Hold(rank_score=0.85, sell_streak=999)
        ctx.exits = [("X", _Exit(exit_type="model_sell"))]
        PanelRankVetoTask().run(ctx)
        assert len(ctx.exits) == 0, "cap disabled → veto regardless of streak"


class TestVetoableSetStringInput:
    """PV-NEW-3: defend against operator passing string instead of list."""

    def test_string_vetoable_treated_as_single_element_list(self):
        ctx = _Ctx(config={"model_sell": {"panel_veto": {
            "enabled": True, "min_rank_score": 0.50,
            "vetoable_exit_types": "model_sell",  # STRING, not list
        }}})
        ctx.holdings["X"] = _Hold(rank_score=0.95)
        ctx.exits = [("X", _Exit(exit_type="model_sell"))]
        PanelRankVetoTask().run(ctx)
        # If treated as set("model_sell") = {'m','o','d','e','l',...},
        # the check `exit_type in vetoable_set` would fail and veto NOT fire.
        # Post-fix: wrapped to single-element list → veto fires.
        assert len(ctx.exits) == 0, (
            "string vetoable_exit_types should be treated as single-element list"
        )


class TestPipelineIntegration:
    def test_panel_rank_veto_job_should_skip_when_disabled(self):
        from kernel.pipeline.job_panel_veto import PanelRankVetoJob
        ctx = _Ctx(config={})
        assert PanelRankVetoJob().should_skip(ctx) is True

    def test_panel_rank_veto_job_should_skip_with_no_exits(self):
        from kernel.pipeline.job_panel_veto import PanelRankVetoJob
        ctx = _Ctx(config=_cfg(enabled=True))
        ctx.exits = []
        assert PanelRankVetoJob().should_skip(ctx) is True

    def test_panel_rank_veto_job_runs_when_enabled_with_exits(self):
        from kernel.pipeline.job_panel_veto import PanelRankVetoJob
        ctx = _Ctx(config=_cfg(enabled=True))
        ctx.exits = [("X", _Exit())]
        assert PanelRankVetoJob().should_skip(ctx) is False
