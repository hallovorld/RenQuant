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
    """Build a HoldingState. The arg is named `panel_score` for back-compat
    with existing tests, but post-2026-04-24-audit the task reads
    `rank_score` (calibrated probability) — set both fields to keep tests
    descriptive.
    """
    from kernel.exits import HoldingState
    h = HoldingState(
        entry_price=100.0, entry_date=datetime.date(2026, 1, 15),
        shares=10, high_watermark=100.0,
    )
    h.panel_score = panel_score   # raw LTR (kept for descriptive symmetry)
    h.rank_score  = panel_score   # calibrated probability — what the task reads
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

    def test_legacy_disabled_noop(self):
        h = _hs(0.10, -0.05)
        tc = SimpleNamespace(
            ticker="NVDA",
            holding=h,
            exit_signal=None,
            config={"risk": {"panel_exit": {
                "enabled": True,
                "legacy_enabled": False,
                "panel_sell_floor": 0.20,
                "mu_sell_ceiling": 0.0,
            }}},
        )

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
        """rank_score 0.10 < 0.20 AND μ=-0.05 ≤ 0 → fire."""
        tc = _tc(panel_score=0.10, mu=-0.05)
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is not None
        assert tc.exit_signal.should_exit is True
        assert tc.exit_signal.exit_type == "panel_conviction"
        assert "rank=0.100" in tc.exit_signal.reason
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


class TestSoftExitGuards:
    def test_bull_calm_min_holding_days_suppresses_legacy_panel_exit(self):
        today = datetime.date(2026, 2, 1)
        h = _hs(0.10, -0.05)
        h.entry_date = today - datetime.timedelta(days=3)
        tc = SimpleNamespace(
            ticker="NVDA",
            holding=h,
            exit_signal=None,
            today=today,
            regime="BULL_CALM",
            price=95.0,
            config={"risk": {"panel_exit": {
                "enabled": True,
                "panel_sell_floor": 0.20,
                "mu_sell_ceiling": 0.0,
                "min_holding_days_by_regime": {"BULL_CALM": 10},
            }}},
        )
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None

    def test_tax_adjusted_gate_suppresses_marginal_short_term_gain_exit(self):
        today = datetime.date(2026, 2, 1)
        h = _hs(0.10, -0.02)
        h.entry_date = today - datetime.timedelta(days=30)
        tc = SimpleNamespace(
            ticker="NVDA",
            holding=h,
            exit_signal=None,
            today=today,
            regime="BULL_CALM",
            price=120.0,
            config={
                "lt_hold_gate_days": 330,
                "lt_hold_min_gain": 0.10,
                "tax": {
                    "short_term_rate": 0.50,
                    "long_term_rate": 0.32,
                    "long_term_threshold_days": 365,
                },
                "risk": {"panel_exit": {
                    "enabled": True,
                    "panel_sell_floor": 0.20,
                    "mu_sell_ceiling": 0.0,
                    "tax_adjusted_soft_exit": {"enabled": True},
                }},
            },
        )
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None


# ── Audit 2026-04-24: scale-correctness regression ───────────────────────────

class TestUnitMismatchAudit:
    """Pre-fix bug: the task compared `panel_sell_floor=0.20` (probability
    scale) against `hs.panel_score` (raw LTR ~N(0,1) or μ−λσ ~±0.05).

    In raw-LTR mode that meant ~58% of holdings had panel_score<0.20 and
    triggered. In NGBoost μ−λσ mode it meant 100% of holdings.

    Post-fix: the task reads `hs.rank_score` (calibrated probability)
    instead, so the 0.20 floor matches its intended scale.
    """

    def _build_holding(self, *, raw_panel: float, calibrated: float, mu: float):
        from kernel.exits import HoldingState
        h = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2026, 1, 15),
            shares=10, high_watermark=100.0,
        )
        h.panel_score = raw_panel       # raw LTR scale (~N(0,1))
        h.rank_score  = calibrated      # calibrated probability (0..1)
        h.mu = mu
        return h

    def test_does_not_fire_on_raw_panel_above_calibrated_floor(self):
        """Raw panel 0.05 (low Z-score) maps to calibrated prob 0.65
        (high) — pre-fix would have fired (0.05<0.20); post-fix doesn't
        because rank_score=0.65 > 0.20."""
        h = self._build_holding(raw_panel=0.05, calibrated=0.65, mu=-0.01)
        tc = SimpleNamespace(
            ticker="X", holding=h, exit_signal=None,
            config={"risk": {"panel_exit": {"enabled": True}}},
        )
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None, (
            "Pre-fix bug: raw panel < probability-scale floor caused "
            "spurious exits. Post-fix uses calibrated rank_score."
        )

    def test_fires_on_low_calibrated_rank(self):
        """Raw panel can be middling; what matters is the calibrated
        probability dropping below the (probability-scale) floor."""
        h = self._build_holding(raw_panel=0.50, calibrated=0.08, mu=-0.02)
        tc = SimpleNamespace(
            ticker="X", holding=h, exit_signal=None,
            config={"risk": {"panel_exit": {"enabled": True}}},
        )
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is not None
        assert tc.exit_signal.exit_type == "panel_conviction"

    def test_ngboost_mu_minus_lambda_sigma_does_not_spuriously_fire(self):
        """In μ−λσ mode, panel_score is overwritten with μ−λσ in range
        ~±0.05 — pre-fix this would fire on EVERY holding (always <
        0.20). Post-fix only the calibrated rank_score matters."""
        h = self._build_holding(raw_panel=-0.03, calibrated=0.55, mu=0.01)
        tc = SimpleNamespace(
            ticker="X", holding=h, exit_signal=None,
            config={"risk": {"panel_exit": {"enabled": True}}},
        )
        PanelConvictionExitTask().run(tc)
        assert tc.exit_signal is None


# ── Task wiring into TickerSellJob ───────────────────────────────────────────

class TestJobWiring:
    def test_panel_conviction_runs_after_all_exit_deciders(self):
        """Position matters: PCT must run AFTER the exit-decider chain so
        higher priority rules always win, but BEFORE EarningsBlackoutSell
        so a panel_conviction exit set by PCT is still subject to the
        event-blackout veto.

        Round-7 (2026-04-26): SellGateBTask was inserted between
        EvaluateExitsTask and PanelConvictionExitTask to add a μ/σ guard
        on model_sell. PCT then evaluated independently after SellGateB
        cleared model_sell.

        2026-05-01 trade-audit response: EarningsBlackoutSellTask appended
        as the final task. It vetoes both `model_sell` and
        `panel_conviction` exits when the holding sits inside the earnings
        event-blackout window — so PCT still fires conditionally, but its
        verdict is now subject to the same calendar-respect contract that
        guards model_sell.
        """
        tasks = TickerSellJob().tasks
        types = [type(t).__name__ for t in tasks]
        # Order: Prepare → Score → Evaluate → SellGateB → PanelConviction
        # → EarningsBlackoutSell.
        assert types == ["PrepareHoldingTask", "ScoreModelTask",
                          "EvaluateExitsTask", "SellGateBTask",
                          "PanelConvictionExitTask",
                          "EarningsBlackoutSellTask"]
        # Invariants this ordering protects:
        i_pct  = types.index("PanelConvictionExitTask")
        i_eb   = types.index("EarningsBlackoutSellTask")
        i_gate = types.index("SellGateBTask")
        i_eval = types.index("EvaluateExitsTask")
        assert i_eval < i_gate < i_pct, (
            "PCT must run AFTER SellGateB clears (or doesn't clear) model_sell"
        )
        assert i_pct < i_eb, (
            "EarningsBlackoutSell must run AFTER PCT so a panel_conviction "
            "exit set by PCT is still subject to event-blackout veto"
        )
