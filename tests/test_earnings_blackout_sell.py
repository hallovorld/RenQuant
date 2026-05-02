"""Tests for EarningsBlackoutSellTask + TopUp earnings guard.

Motivating incident (2026-05-01 trade audit):
  * CAT printed 2026-04-30 BMO (+19.9% EPS beat, +9.88% single-day);
    system fired streak=3 model_sell on 2026-05-01 morning, exiting
    into post-earnings strength.
  * FTNT was topped up on 2026-04-29 — one day before its 2026-04-30
    earnings print — because TopUpHeldTask did not consult the
    earnings calendar.

Invariant under test:
  Model-driven exits (`model_sell`, `panel_conviction`) are SUPPRESSED
  when the holding sits inside its earnings event-blackout window.
  Path-action exits (stop_loss / trailing_stop / single_day_loss /
  max_hold / kelly_trim / rotation) ALWAYS fire.
  TopUp adds to held positions and is treated as entry — must respect
  the same buy-side earnings buffer.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.exits import ExitSignal, HoldingState   # noqa: E402
from kernel.pipeline.context import (                 # noqa: E402
    InferenceContext, TickerInferenceContext,
)
from kernel.pipeline.task_sell import (   # noqa: E402
    EarningsBlackoutSellTask,
    MODEL_DRIVEN_EXIT_TYPES,
    PATH_DRIVEN_EXIT_TYPES,
)
from kernel.pipeline.task_topup import TopUpHeldTask  # noqa: E402
from kernel.pipeline.job_sell import TickerSellJob   # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

TODAY = datetime.date(2026, 5, 1)


def _holding(*, sell_streak: int = 3, shares: float = 10.0,
             entry_offset_days: int = 60) -> HoldingState:
    h = HoldingState(
        entry_price=100.0,
        entry_date=TODAY - datetime.timedelta(days=entry_offset_days),
        high_watermark=110.0,
        sell_streak=sell_streak,
        last_streak_inc_date=TODAY,
        shares=shares,
    )
    return h


def _exit_signal(exit_type: str, should_exit: bool = True) -> ExitSignal:
    return ExitSignal(
        should_exit=should_exit,
        reason=f"{exit_type} test",
        exit_type=exit_type,
    )


def _sell_tctx(*, ticker: str = "CAT",
               exit_signal: ExitSignal | None,
               earnings_calendar: dict | None = None,
               pre_days: int = 2,
               post_days: int = 5,
               holding: HoldingState | None = None,
               today: datetime.date = TODAY) -> TickerInferenceContext:
    cfg = {
        "regime": {
            "earnings_sell_buffer_pre_days":  pre_days,
            "earnings_sell_buffer_post_days": post_days,
        },
    }
    tc = TickerInferenceContext(
        ticker=ticker,
        ohlcv={},
        model=None,
        config=cfg,
        today=today,
        regime="BULL_CALM",
        regime_params={},
        exit_params={},
        holding=holding if holding is not None else _holding(),
        price=100.0,
        earnings_calendar=earnings_calendar,
    )
    tc.exit_signal = exit_signal
    return tc


# ── Path rules are EXEMPT (path-action exits always fire) ─────────────────────

class TestPathRulesExempt:
    """Path-action exits are price-driven and must always fire, even if
    earnings are tomorrow. Vetoing a stop_loss next to earnings would
    actively hurt — that's exactly when stop_loss matters most.
    """

    @pytest.mark.parametrize("exit_type", [
        "stop_loss", "trailing_stop", "single_day_loss",
        "max_hold", "kelly_trim", "rotation",
    ])
    def test_path_rule_not_vetoed_inside_window(self, exit_type):
        # Earnings yesterday — deep inside post-window
        cal = {"CAT": [(TODAY - datetime.timedelta(days=1)).isoformat()]}
        sig = _exit_signal(exit_type)
        tc = _sell_tctx(exit_signal=sig, earnings_calendar=cal)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is sig
        assert tc.exit_signal.should_exit is True


# ── Model-driven exits VETOED inside window ───────────────────────────────────

class TestModelDrivenExitVetoed:
    """The CAT regression: earnings on TODAY-1, model_sell streak=3 on TODAY,
    must be vetoed. Same for panel_conviction.
    """

    @pytest.mark.parametrize("exit_type", ["model_sell", "panel_conviction"])
    def test_vetoed_one_day_after_earnings(self, exit_type):
        cal = {"CAT": [(TODAY - datetime.timedelta(days=1)).isoformat()]}
        tc = _sell_tctx(exit_signal=_exit_signal(exit_type), earnings_calendar=cal)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is None

    def test_vetoed_on_earnings_day(self):
        cal = {"CAT": [TODAY.isoformat()]}
        tc = _sell_tctx(exit_signal=_exit_signal("model_sell"), earnings_calendar=cal)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is None

    def test_vetoed_one_day_before_earnings(self):
        cal = {"CAT": [(TODAY + datetime.timedelta(days=1)).isoformat()]}
        tc = _sell_tctx(exit_signal=_exit_signal("model_sell"), earnings_calendar=cal)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is None

    def test_cat_regression_recreates_2026_05_01_incident(self):
        """Reproduce CAT 2026-05-01: earnings 2026-04-30, model_sell streak=3
        on 2026-05-01 → must be vetoed."""
        cal = {"CAT": ["2026-04-30"]}
        tc = _sell_tctx(
            ticker="CAT",
            exit_signal=_exit_signal("model_sell"),
            earnings_calendar=cal,
            today=datetime.date(2026, 5, 1),
        )
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is None, (
            "CAT 2026-05-01 model_sell must be vetoed by post-earnings window"
        )


# ── Window boundaries (asymmetric pre/post) ───────────────────────────────────

class TestWindowBoundaries:
    """Default window is asymmetric: pre=2 (tighter, voluntary sized-down),
    post=5 (PEAD respect, Bernard-Thomas 1989).
    """

    def test_pre_window_inclusive_at_pre_days(self):
        cal = {"CAT": [(TODAY + datetime.timedelta(days=2)).isoformat()]}
        tc = _sell_tctx(exit_signal=_exit_signal("model_sell"),
                        earnings_calendar=cal, pre_days=2, post_days=5)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is None  # offset=+2, inside (0, +2]

    def test_pre_window_exclusive_outside(self):
        sig = _exit_signal("model_sell")
        cal = {"CAT": [(TODAY + datetime.timedelta(days=3)).isoformat()]}
        tc = _sell_tctx(exit_signal=sig,
                        earnings_calendar=cal, pre_days=2, post_days=5)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is sig  # offset=+3, outside (0, +2]

    def test_post_window_inclusive_at_post_days(self):
        cal = {"CAT": [(TODAY - datetime.timedelta(days=5)).isoformat()]}
        tc = _sell_tctx(exit_signal=_exit_signal("model_sell"),
                        earnings_calendar=cal, pre_days=2, post_days=5)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is None  # offset=-5, inside [-5, 0)

    def test_post_window_exclusive_outside(self):
        sig = _exit_signal("model_sell")
        cal = {"CAT": [(TODAY - datetime.timedelta(days=6)).isoformat()]}
        tc = _sell_tctx(exit_signal=sig,
                        earnings_calendar=cal, pre_days=2, post_days=5)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is sig  # offset=-6, outside [-5, 0)

    def test_far_future_earnings_no_veto(self):
        sig = _exit_signal("model_sell")
        cal = {"CAT": [(TODAY + datetime.timedelta(days=30)).isoformat()]}
        tc = _sell_tctx(exit_signal=sig, earnings_calendar=cal)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is sig


# ── Streak preservation (mirror SellGateB behaviour) ──────────────────────────

class TestStreakPreservation:
    def test_vetoed_model_sell_does_not_reset_streak(self):
        h = _holding(sell_streak=3)
        cal = {"CAT": [(TODAY - datetime.timedelta(days=1)).isoformat()]}
        tc = _sell_tctx(exit_signal=_exit_signal("model_sell"),
                        earnings_calendar=cal, holding=h)
        EarningsBlackoutSellTask().run(tc)
        assert h.sell_streak == 3, "streak preserved across veto"


# ── Defensive paths (no veto on bad inputs / disabled config) ─────────────────

class TestDefensive:
    def test_no_calendar_no_veto(self):
        sig = _exit_signal("model_sell")
        tc = _sell_tctx(exit_signal=sig, earnings_calendar=None)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is sig

    def test_empty_calendar_no_veto(self):
        sig = _exit_signal("model_sell")
        tc = _sell_tctx(exit_signal=sig, earnings_calendar={})
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is sig

    def test_ticker_not_in_calendar_no_veto(self):
        sig = _exit_signal("model_sell")
        # Calendar has AAPL but not CAT
        cal = {"AAPL": [TODAY.isoformat()]}
        tc = _sell_tctx(ticker="CAT", exit_signal=sig, earnings_calendar=cal)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is sig

    def test_malformed_date_skipped(self):
        sig = _exit_signal("model_sell")
        cal = {"CAT": ["not-a-date", (TODAY + datetime.timedelta(days=30)).isoformat()]}
        tc = _sell_tctx(exit_signal=sig, earnings_calendar=cal)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is sig  # malformed skipped, far-future doesn't trigger

    def test_disabled_via_zero_buffers(self):
        """pre=0 AND post=0 means operator opted out — no veto."""
        sig = _exit_signal("model_sell")
        cal = {"CAT": [TODAY.isoformat()]}
        tc = _sell_tctx(exit_signal=sig, earnings_calendar=cal,
                        pre_days=0, post_days=0)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is sig

    def test_no_exit_signal_no_op(self):
        tc = _sell_tctx(exit_signal=None,
                        earnings_calendar={"CAT": [TODAY.isoformat()]})
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is None

    def test_should_not_exit_signal_left_alone(self):
        sig = _exit_signal("model_sell", should_exit=False)
        cal = {"CAT": [TODAY.isoformat()]}
        tc = _sell_tctx(exit_signal=sig, earnings_calendar=cal)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is sig
        assert tc.exit_signal.should_exit is False


# ── Module-level taxonomy guard (M2 fix, 2026-05-01) ─────────────────────────

class TestExitTypeTaxonomy:
    """The MODEL_DRIVEN / PATH_DRIVEN sets are the single source of truth
    for "which exits respect the earnings blackout vs which always fire."
    Future code adding a new exit_type must classify it explicitly here;
    these guards catch silent bypass via missing classification.
    """

    def test_model_and_path_sets_disjoint(self):
        """No exit type can be in both sets — they're mutually exclusive
        by definition (model-driven OR price-driven, never both)."""
        overlap = MODEL_DRIVEN_EXIT_TYPES & PATH_DRIVEN_EXIT_TYPES
        assert overlap == set(), (
            f"exit type(s) classified in BOTH MODEL_ and PATH_DRIVEN: "
            f"{overlap}. Each must be exactly one or the other."
        )

    def test_known_model_driven_set_includes_at_least_model_sell(self):
        # Sanity check: regression guard against accidental empty set.
        assert "model_sell" in MODEL_DRIVEN_EXIT_TYPES
        assert "panel_conviction" in MODEL_DRIVEN_EXIT_TYPES

    def test_known_path_driven_set_includes_hard_risk_exits(self):
        for must_have in ("stop_loss", "trailing_stop",
                           "single_day_loss", "max_hold"):
            assert must_have in PATH_DRIVEN_EXIT_TYPES, (
                f"{must_have} must be PATH_DRIVEN — losing this "
                f"classification would let earnings blackout veto a "
                f"price-action stop, which is the OPPOSITE of safe behavior"
            )


# ── Wiring (TickerSellJob includes EarningsBlackoutSellTask, runs LAST) ───────

class TestWiring:
    def test_in_ticker_sell_job(self):
        names = [type(t).__name__ for t in TickerSellJob().tasks]
        assert "EarningsBlackoutSellTask" in names

    def test_runs_last_after_all_exit_deciders(self):
        """Order: …EvaluateExits → SellGateB → PanelConvictionExit →
        EarningsBlackoutSell. Must run AFTER PanelConvictionExit so it
        sees the FINAL exit_signal regardless of which decider set it.
        """
        names = [type(t).__name__ for t in TickerSellJob().tasks]
        i_eval   = names.index("EvaluateExitsTask")
        i_gate_b = names.index("SellGateBTask")
        i_pc     = names.index("PanelConvictionExitTask")
        i_earn   = names.index("EarningsBlackoutSellTask")
        assert i_eval < i_gate_b < i_pc < i_earn


# ── TopUp earnings guard ──────────────────────────────────────────────────────

class TestTopUpEarningsGuard:
    """TopUpHeldTask adds shares to existing positions when Kelly target
    exceeds current weight. Treated as entry — must respect the buy-side
    earnings_buffer_days. Pre-fix: FTNT topped up 2026-04-29, one day
    before earnings 2026-04-30.
    """

    @staticmethod
    def _make_holding_with_kelly(*, kelly_target_pct: float = 0.30,
                                  shares: float = 10.0,
                                  rank_score: float = 0.50) -> HoldingState:
        h = HoldingState(
            entry_price=100.0,
            entry_date=TODAY - datetime.timedelta(days=30),
            high_watermark=110.0,
            sell_streak=0,
            last_streak_inc_date=TODAY,
            shares=shares,
        )
        h.kelly_target_pct = kelly_target_pct
        h.rank_score = rank_score
        h.panel_score = 0.55
        h.sigma = 0.05
        h.mu    = 0.01
        return h

    @staticmethod
    def _make_ctx(*, earnings_calendar: dict | None,
                   buffer_days: int = 3,
                   today: datetime.date = TODAY) -> InferenceContext:
        cfg = {
            "regime": {"earnings_buffer_days": buffer_days},
            "ranking": {"kelly_sizing": {"enabled": True,
                                          "top_up_threshold": 0.05}},
        }
        ctx = InferenceContext(config=cfg, today=today)
        ctx.earnings_calendar = earnings_calendar
        ctx.holdings = {"FTNT": TestTopUpEarningsGuard._make_holding_with_kelly()}
        ctx.prices = {"FTNT": 85.0}
        ctx.cash = 10_000.0
        ctx.portfolio_value = 10_000.0
        return ctx

    def test_topup_blocked_one_day_before_earnings(self):
        """FTNT 2026-04-29 regression: earnings 2026-04-30, top-up blocked."""
        cal = {"FTNT": ["2026-04-30"]}
        ctx = self._make_ctx(earnings_calendar=cal,
                             today=datetime.date(2026, 4, 29))
        TopUpHeldTask().run(ctx)
        assert ctx.orders == [], "TopUp must skip earnings-imminent ticker"

    def test_topup_blocked_on_earnings_day(self):
        cal = {"FTNT": [TODAY.isoformat()]}
        ctx = self._make_ctx(earnings_calendar=cal)
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_topup_blocked_one_day_after_earnings(self):
        cal = {"FTNT": [(TODAY - datetime.timedelta(days=1)).isoformat()]}
        ctx = self._make_ctx(earnings_calendar=cal)
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_topup_allowed_outside_buffer(self):
        cal = {"FTNT": [(TODAY + datetime.timedelta(days=10)).isoformat()]}
        ctx = self._make_ctx(earnings_calendar=cal)
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1
        assert ctx.orders[0]["ticker"] == "FTNT"
        assert ctx.orders[0]["order_type"] == "TOP_UP"

    def test_topup_allowed_no_calendar(self):
        ctx = self._make_ctx(earnings_calendar=None)
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1

    def test_topup_allowed_ticker_not_in_calendar(self):
        cal = {"AAPL": [TODAY.isoformat()]}  # only AAPL has dates
        ctx = self._make_ctx(earnings_calendar=cal)
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1
        assert ctx.orders[0]["ticker"] == "FTNT"

    def test_topup_order_keeps_order_type_marker(self):
        """Distinguish TOP_UP from NEW_BUY downstream — the field that the
        trade log should preserve so audits can tell why a trade fired.
        """
        cal = {"FTNT": [(TODAY + datetime.timedelta(days=10)).isoformat()]}
        ctx = self._make_ctx(earnings_calendar=cal)
        TopUpHeldTask().run(ctx)
        assert ctx.orders[0]["order_type"] == "TOP_UP"
        assert "rank_score" in ctx.orders[0]


# ── Integration with the rest of the sell chain ───────────────────────────────

class TestIntegrationWithSellChain:
    """Verify the new task composes cleanly with SellGateB and
    PanelConvictionExit (the existing model-driven exit deciders) and
    with the buy-side EarningsFilter (single source of truth: same
    calendar, consistent semantics).
    """

    def test_panel_conviction_exit_can_be_vetoed_by_blackout(self):
        """PanelConvictionExit may set exit_signal to `panel_conviction`
        AFTER SellGateB cleared the original model_sell. EarningsBlackoutSell
        runs LAST — it must catch this case too.
        """
        cal = {"CAT": [(TODAY - datetime.timedelta(days=1)).isoformat()]}
        tc = _sell_tctx(exit_signal=_exit_signal("panel_conviction"),
                        earnings_calendar=cal)
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is None

    def test_sell_tctx_receives_earnings_calendar_via_pp_inference(self):
        """Smoke test the plumbing fix in pp_inference._make_sell_tctx —
        without this, the sell tctx would have earnings_calendar=None and
        the new task would silently fail open on every holding.
        """
        from kernel.pipeline.pp_inference import _make_sell_tctx  # noqa: PLC0415
        ctx = InferenceContext(config={}, today=TODAY)
        ctx.holdings = {"CAT": _holding()}
        ctx.prices = {"CAT": 100.0}
        ctx.regime = "BULL_CALM"
        ctx.earnings_calendar = {"CAT": ["2026-04-30"]}
        sell_tc = _make_sell_tctx(ctx, "CAT")
        assert sell_tc.earnings_calendar == {"CAT": ["2026-04-30"]}, (
            "sell tctx must receive earnings_calendar — without this "
            "EarningsBlackoutSellTask cannot see any earnings dates and "
            "fails open silently"
        )

    def test_no_double_handling_when_sellgateb_already_cleared(self):
        """SellGateB may clear model_sell first; EarningsBlackoutSell sees
        None and does nothing. No state corruption either way."""
        tc = _sell_tctx(exit_signal=None,
                        earnings_calendar={"CAT": [TODAY.isoformat()]})
        # Earnings on TODAY but no exit signal — must remain None
        EarningsBlackoutSellTask().run(tc)
        assert tc.exit_signal is None


# ── Logging contract (operator visibility into the decision tree) ─────────────

class TestLogging:
    """The veto must be observable. Without a clear log line per veto, an
    operator can't tell from `live/logs/*.json` why a model_sell did NOT
    fire on a given bar — and silent suppression is worse than the bug
    we're fixing.
    """

    def test_veto_logs_ticker_exit_type_offset_and_window(self, caplog):
        import logging as _log
        caplog.set_level(_log.INFO, logger="kernel.pipeline.sell")
        cal = {"CAT": ["2026-04-30"]}
        tc = _sell_tctx(exit_signal=_exit_signal("model_sell"),
                        earnings_calendar=cal,
                        today=datetime.date(2026, 5, 1))
        EarningsBlackoutSellTask().run(tc)
        msgs = [r.getMessage() for r in caplog.records]
        # The single veto log line must carry every piece of info an
        # operator needs to understand the suppression.
        assert any("EarningsBlackoutSellTask" in m for m in msgs)
        assert any("CAT" in m for m in msgs)
        assert any("model_sell" in m for m in msgs)
        assert any("2026-04-30" in m for m in msgs)
        assert any("VETO" in m for m in msgs)

    def test_topup_block_logs_ticker_and_buffer(self, caplog):
        import logging as _log
        caplog.set_level(_log.INFO, logger="kernel.pipeline.topup")
        cal = {"FTNT": ["2026-04-30"]}
        ctx = TestTopUpEarningsGuard._make_ctx(
            earnings_calendar=cal, today=datetime.date(2026, 4, 29),
        )
        TopUpHeldTask().run(ctx)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("FTNT" in m and "earnings" in m.lower() for m in msgs)
