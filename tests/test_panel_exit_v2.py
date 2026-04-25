"""Panel conviction-exit V2 — AND/OR trigger mode.

V1 default AND (both panel<floor AND μ<=ceiling) never fired in the
A/B on current golden. V2 adds `trigger_mode` = "and" (default,
backwards-compat) | "or" to widen trigger.
"""
from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _tc(panel_score, mu, trigger_mode=None, floor=0.20, ceiling=0.0,
        enabled=True):
    cfg = {
        "risk": {
            "panel_exit": {
                "enabled": enabled,
                "panel_sell_floor": floor,
                "mu_sell_ceiling":  ceiling,
            }
        }
    }
    if trigger_mode is not None:
        cfg["risk"]["panel_exit"]["trigger_mode"] = trigger_mode

    # Post-2026-04-24-audit: task reads `rank_score` (calibrated probability),
    # not `panel_score` (raw LTR). The arg is named `panel_score` for
    # back-compat — set both fields so test descriptions still read clean.
    hs = SimpleNamespace(panel_score=panel_score, rank_score=panel_score, mu=mu)
    return SimpleNamespace(
        ticker       = "NVDA",
        config       = cfg,
        holding      = hs,
        exit_signal  = None,
    )


class TestAndMode:
    def test_both_conditions_fires(self):
        from kernel.pipeline.task_sell import PanelConvictionExitTask
        tc = _tc(panel_score=0.15, mu=-0.01)   # both below thresholds
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is not None
        assert tc.exit_signal.exit_type == "panel_conviction"

    def test_only_panel_below_blocks(self):
        from kernel.pipeline.task_sell import PanelConvictionExitTask
        tc = _tc(panel_score=0.15, mu=0.05)    # panel OK-bad, μ positive
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None

    def test_only_mu_below_blocks(self):
        from kernel.pipeline.task_sell import PanelConvictionExitTask
        tc = _tc(panel_score=0.30, mu=-0.02)   # panel ok, μ bad
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None


class TestOrMode:
    def test_only_panel_below_fires(self):
        from kernel.pipeline.task_sell import PanelConvictionExitTask
        tc = _tc(panel_score=0.15, mu=0.05, trigger_mode="or")
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is not None
        assert "or" in tc.exit_signal.reason

    def test_only_mu_below_fires(self):
        from kernel.pipeline.task_sell import PanelConvictionExitTask
        tc = _tc(panel_score=0.30, mu=-0.02, trigger_mode="or")
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is not None

    def test_both_above_blocks(self):
        from kernel.pipeline.task_sell import PanelConvictionExitTask
        tc = _tc(panel_score=0.30, mu=0.05, trigger_mode="or")
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None


class TestDisabledOrMissing:
    def test_disabled_no_fire(self):
        from kernel.pipeline.task_sell import PanelConvictionExitTask
        tc = _tc(panel_score=0.10, mu=-0.10, enabled=False)
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None

    def test_missing_panel_score_no_fire(self):
        from kernel.pipeline.task_sell import PanelConvictionExitTask
        tc = _tc(panel_score=None, mu=-0.10, trigger_mode="or")
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None

    def test_higher_priority_exit_preserved(self):
        """If tc.exit_signal already set, task must not override."""
        from kernel.exits import ExitSignal
        from kernel.pipeline.task_sell import PanelConvictionExitTask
        tc = _tc(panel_score=0.10, mu=-0.10)   # both below (AND mode fires)
        tc.exit_signal = ExitSignal(
            should_exit=True, reason="stop_loss fired",
            exit_type="stop_loss",
        )
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal.exit_type == "stop_loss"   # unchanged
