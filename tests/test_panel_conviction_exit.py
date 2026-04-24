"""PanelConvictionExitTask — panel/NGBoost-based sell trigger.

User spec 2026-04-24: "买卖换加减仓都要是 model+policy". Sell used to
only consult per-ticker tournament model + price rules; now also checks
the cross-sectional panel score + NGBoost μ/σ (persisted on HoldingState
from prior bar's PanelScoringJob).

Priority: this task runs LAST in TickerSellJob chain so higher-priority
rules (trailing/stop/SDL/max_hold/model-streak) always win.

Flag default off — users A/B before promoting.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.pipeline.task_sell import PanelConvictionExitTask  # noqa: E402
from kernel.pipeline.job_sell import TickerSellJob  # noqa: E402


def _hs(panel_score: float | None, mu: float | None):
    from kernel.exits import HoldingState
    h = HoldingState(
        entry_price=100.0, entry_date=datetime.date(2026, 1, 15),
        shares=10, high_watermark=100.0,
    )
    h.panel_score = panel_score
    h.mu = mu
    return h


def _tc(*, panel_score: float | None, mu: float | None,
        enabled: bool = True, already_exiting: bool = False,
        panel_sell_floor: float = 0.20, mu_sell_ceiling: float = 0.0):
    return SimpleNamespace(
        ticker      = "NVDA",
        holding     = _hs(panel_score, mu),
        exit_signal = "stop_loss" if already_exiting else None,
        config      = {"risk": {"panel_exit": {
            "enabled":          enabled,
            "panel_sell_floor": panel_sell_floor,
            "mu_sell_ceiling":  mu_sell_ceiling,
        }}},
    )


# ── Flag gating ───────────────────────────────────────────────────────────────

class TestFlagGating:
    def test_default_disabled_noop(self):
        tc = _tc(panel_score=0.10, mu=-0.05, enabled=False)
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None

    def test_exit_signal_already_set_noop(self):
        """Higher-priority rule already fired — don't override."""
        tc = _tc(panel_score=0.10, mu=-0.05, already_exiting=True)
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal == "stop_loss"


# ── Core trigger logic ────────────────────────────────────────────────────────

class TestTriggerLogic:
    def test_fires_when_panel_low_and_mu_nonpositive(self):
        """Panel 0.10 < 0.20 AND μ=-0.05 ≤ 0 → fire."""
        tc = _tc(panel_score=0.10, mu=-0.05)
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is not None
        assert tc.exit_signal.should_exit is True
        assert tc.exit_signal.exit_type == "panel_conviction"
        assert "panel=0.100" in tc.exit_signal.reason
        assert "-0.0500" in tc.exit_signal.reason

    def test_skips_when_panel_above_floor(self):
        """Panel 0.25 > 0.20 → don't fire even if μ negative."""
        tc = _tc(panel_score=0.25, mu=-0.05)
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None

    def test_skips_when_mu_positive(self):
        """μ > 0 → model still sees edge; don't fire even if panel low."""
        tc = _tc(panel_score=0.10, mu=0.02)
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None

    def test_skips_when_panel_score_missing(self):
        """First bar after buy or panel disabled → graceful no-op."""
        tc = _tc(panel_score=None, mu=-0.05)
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None

    def test_skips_when_mu_missing(self):
        tc = _tc(panel_score=0.10, mu=None)
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None


# ── Threshold tunability ─────────────────────────────────────────────────────

class TestThresholds:
    def test_stricter_panel_floor_suppresses_fires(self):
        """floor=0.05 instead of 0.20 → panel 0.10 > 0.05 → skip."""
        tc = _tc(panel_score=0.10, mu=-0.05, panel_sell_floor=0.05)
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None

    def test_stricter_mu_ceiling_suppresses_fires(self):
        """mu_ceiling=-0.10 → only fire when μ ≤ -0.10; μ=-0.05 > ceiling → skip."""
        tc = _tc(panel_score=0.10, mu=-0.05, mu_sell_ceiling=-0.10)
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None


# ── Task wiring into TickerSellJob ───────────────────────────────────────────

class TestJobWiring:
    def test_panel_conviction_is_last_in_chain(self):
        """Position matters: must run AFTER EvaluateExitsTask so higher
        priority rules always win."""
        tasks = TickerSellJob().tasks
        types = [type(t).__name__ for t in tasks]
        assert types[-1] == "PanelConvictionExitTask"
        # Sanity: order is Prepare → Score → Evaluate → PanelConviction
        assert types == ["PrepareHoldingTask", "ScoreModelTask",
                          "EvaluateExitsTask", "PanelConvictionExitTask"]
