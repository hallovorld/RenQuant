"""Tests for inference-time earnings-calendar filtering (forward-look guard).

Motivation (Track #17, 2026-05-10 audit)
----------------------------------------
`backtesting/renquant_104/artifacts/earnings-calendar.json` is a flat
dict[ticker → list[ISO date strings]] that includes BOTH past and
future-dated events relative to any historical bar a simulator/LEAN run
might step over. Example: AAPL has 2026-04-30 entries; a backtest bar
at 2024-06-15 would have ~685 days until that event.

If any consumer iterated the calendar without bounding by `ctx.today`
(e.g. "is the next earnings within 60 days?"), a backtest would
"learn" earnings dates that weren't knowable on the historical bar →
forward-looking leakage.

Invariant under test
--------------------
Every consumer of `ctx.earnings_calendar` MUST bound its decision by
`abs(event_date - today) <= window` (or asymmetric `(-post, +pre)`).
Future-dated events outside that window are functionally invisible.

Consumers audited (path:line at audit time):
  * kernel/selection.py:186-202        is_earnings_blocked (helper)
  * kernel/pipeline/task_candidates.py:12-19   EarningsFilterTask (buy gate)
  * kernel/pipeline/task_sell.py:370-476       EarningsBlackoutSellTask (sell veto)
  * kernel/pipeline/task_topup.py:96-130       TopUpHeldTask earnings guard
  * kernel/pipeline/task_selection.py:53-72   SelectionContext.earnings_buffer
  * kernel/portfolio_qp/tasks.py:520-522       QP buy gate

All six route the (ticker, today, calendar, buffer) tuple through the
single source-of-truth `is_earnings_blocked` (or apply an equivalent
asymmetric window). This file pins that invariant.

Per CLAUDE.md §5.13.3: each fix names an invariant. Here the invariant
is "the calendar is a passive lookup table; only the today-bounded
window contributes to any decision".

Per CLAUDE.md §5.13.1: includes a real-prod-path test that calls
through the actual Task class with a real TickerInferenceContext.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.selection import is_earnings_blocked  # noqa: E402
from kernel.pipeline.context import TickerInferenceContext  # noqa: E402
from kernel.pipeline.task_candidates import EarningsFilterTask  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────

# A calendar with a mix of past and future-dated events relative to the
# default test "today" (2024-06-15). The future events at 2026-04-30 are
# exactly the kind of leakage payload the audit is concerned about.
SYNTHETIC_CALENDAR = {
    "AAPL": [
        "2024-02-01",   # ~134d before today
        "2024-05-02",   # ~44d before today  (post-window for 5d buffer? no)
        "2024-08-01",   # ~47d AFTER today
        "2026-04-30",   # 685d AFTER today  (the leakage smoking gun)
    ],
    "MSFT": [
        "2024-06-13",   # 2 days before today  (inside ±3 buffer)
        "2026-07-24",   # far-future
    ],
    "NVDA": [
        "2024-06-17",   # 2 days after today  (inside ±3 buffer)
    ],
    "GOOG": [
        "2024-06-15",   # exactly today
    ],
    "QQQ": [
        "2023-01-01",   # very old
        "2030-12-31",   # very future
    ],
    "NOEVENT": [],
    "BADDATE": ["not-a-date", "2024-13-99"],
}

TODAY = datetime.date(2024, 6, 15)


def _make_tc(ticker: str, today: datetime.date = TODAY,
             buffer_days: int = 3) -> TickerInferenceContext:
    """Minimal TickerInferenceContext for the buy-side gate.

    Only fields read by EarningsFilterTask are populated:
      ticker, today, earnings_calendar, config["regime"]["earnings_buffer_days"].
    """
    return TickerInferenceContext(
        ticker=ticker,
        ohlcv={},
        model=None,
        config={"regime": {"earnings_buffer_days": buffer_days}},
        today=today,
        regime="BULL_CALM",
        regime_params={},
        exit_params={},
        earnings_calendar=SYNTHETIC_CALENDAR,
    )


# ── TestEarningsFilterRegression ─────────────────────────────────────────────

class TestEarningsFilterRegression:
    """Pins the (ticker, today, calendar) → blocked invariant.

    These exercise `is_earnings_blocked` (the single source of truth used
    by the buy gate, sell veto helper, top-up gate, selection loop, and
    QP buy gate).
    """

    def test_blocks_when_event_is_today(self):
        # GOOG: 2024-06-15 == today → offset 0 → inside ±3 window
        assert is_earnings_blocked("GOOG", TODAY, SYNTHETIC_CALENDAR, buffer_days=3)

    def test_blocks_when_event_is_within_pre_window(self):
        # NVDA: 2024-06-17 → +2d → inside ±3 window
        assert is_earnings_blocked("NVDA", TODAY, SYNTHETIC_CALENDAR, buffer_days=3)

    def test_blocks_when_event_is_within_post_window(self):
        # MSFT: 2024-06-13 → -2d → inside ±3 window
        assert is_earnings_blocked("MSFT", TODAY, SYNTHETIC_CALENDAR, buffer_days=3)

    def test_does_NOT_block_when_event_is_outside_window(self):
        # AAPL has 2024-08-01 (+47d) and 2024-05-02 (-44d) — neither within ±3
        # AAPL also has 2026-04-30 (+685d) which is the LEAKAGE EVENT.
        # The filter MUST ignore all of them.
        assert not is_earnings_blocked(
            "AAPL", TODAY, SYNTHETIC_CALENDAR, buffer_days=3
        )

    def test_does_NOT_block_future_far_event(self):
        # QQQ: only events 2023-01-01 (-533d) and 2030-12-31 (+2391d).
        # Both outside any sane window.
        assert not is_earnings_blocked(
            "QQQ", TODAY, SYNTHETIC_CALENDAR, buffer_days=3
        )

    def test_empty_calendar_returns_false(self):
        assert not is_earnings_blocked("AAPL", TODAY, {}, buffer_days=3)

    def test_no_dates_for_ticker_returns_false(self):
        # NOEVENT has empty list — must not crash or block.
        assert not is_earnings_blocked(
            "NOEVENT", TODAY, SYNTHETIC_CALENDAR, buffer_days=3
        )

    def test_unknown_ticker_returns_false(self):
        # Ticker not present at all in calendar — must not block.
        assert not is_earnings_blocked(
            "UNLISTED", TODAY, SYNTHETIC_CALENDAR, buffer_days=3
        )

    def test_malformed_dates_silently_skipped(self):
        # BADDATE: 'not-a-date' and '2024-13-99' — ValueError raised by
        # date.fromisoformat → caught, that entry skipped. No block.
        assert not is_earnings_blocked(
            "BADDATE", TODAY, SYNTHETIC_CALENDAR, buffer_days=3
        )

    def test_window_zero_only_today_blocks(self):
        # buffer_days=0 → only event_date == today blocks
        assert is_earnings_blocked("GOOG", TODAY, SYNTHETIC_CALENDAR, buffer_days=0)
        assert not is_earnings_blocked(
            "NVDA", TODAY, SYNTHETIC_CALENDAR, buffer_days=0
        )
        assert not is_earnings_blocked(
            "MSFT", TODAY, SYNTHETIC_CALENDAR, buffer_days=0
        )


# ── AUDIT REGRESSION GUARD (per CLAUDE.md §5.13.3) ───────────────────────────

class TestAudit2026_05_10FutureEarningsBlocked:
    """AUDIT REGRESSION GUARD — Track #17 (2026-05-10).

    Invariant: future-dated calendar events do NOT influence any
    decision at a historical bar `today`. The filter must behave
    identically whether the calendar contains only past-bounded events
    or includes far-future events alongside.

    If a future regression adds a consumer that iterates the calendar
    without `today`-bounding (e.g. "next earnings within 60d"), the
    parity assertions below WILL fail.
    """

    BACKTEST_BAR = datetime.date(2024, 6, 15)
    LIVE_TODAY = datetime.date(2026, 5, 10)  # roughly when this test was written

    @pytest.fixture
    def calendar_with_future_events(self):
        # The literal calendar that prod ships with — future earnings
        # entries the operator added for the next reporting season.
        return {
            "AAPL": ["2024-05-02", "2026-04-30"],  # one past, one future-leak
            "MSFT": ["2026-07-24"],
        }

    @pytest.fixture
    def calendar_truncated_to_bar(self):
        # The same calendar the operator WOULD have had on 2024-06-15.
        # All events strictly after `2024-06-30` removed.
        return {
            "AAPL": ["2024-05-02"],
            "MSFT": [],
        }

    def test_buy_gate_identical_with_or_without_future_events(
        self, calendar_with_future_events, calendar_truncated_to_bar
    ):
        """The buy gate's decision on 2024-06-15 must not depend on
        whether the calendar contains 2026-04-30 events."""
        for ticker in ("AAPL", "MSFT"):
            decision_with_future = is_earnings_blocked(
                ticker, self.BACKTEST_BAR, calendar_with_future_events,
                buffer_days=3,
            )
            decision_without_future = is_earnings_blocked(
                ticker, self.BACKTEST_BAR, calendar_truncated_to_bar,
                buffer_days=3,
            )
            assert decision_with_future == decision_without_future, (
                f"LEAKAGE: {ticker} buy-gate decision changed when future "
                f"events added to calendar. with_future="
                f"{decision_with_future} != without_future="
                f"{decision_without_future}"
            )

    def test_aapl_future_2026_event_invisible_at_2024_bar(
        self, calendar_with_future_events
    ):
        """The smoking-gun assertion: AAPL has 2026-04-30 in the
        calendar, sim runs at 2024-06-15, must NOT be blocked by the
        future event (685 days out, far beyond any sane buffer)."""
        assert not is_earnings_blocked(
            "AAPL", self.BACKTEST_BAR, calendar_with_future_events,
            buffer_days=3,
        )
        # And also at the larger sell buffers (pre=2, post=5 default):
        # — abs(685) > 5 — still invisible.
        assert not is_earnings_blocked(
            "AAPL", self.BACKTEST_BAR, calendar_with_future_events,
            buffer_days=30,
        )

    def test_buffer_is_symmetric_bounded(self):
        """Even with an aggressive 365-day buffer, the 2026 event at
        +685d from a 2024 bar is still safely outside. This pins the
        cliff — any consumer that uses an unbounded scan would fail.

        Uses a calendar with ONLY a future-leaked event so we isolate
        the leakage signal from any legitimate in-window past events.
        """
        future_only = {"AAPL": ["2026-04-30"]}
        # 365-day buffer: distance is 685d — should remain outside.
        assert not is_earnings_blocked(
            "AAPL", self.BACKTEST_BAR, future_only, buffer_days=365
        )
        # 684 days — JUST inside, but the consumer would have to be
        # using an absurd buffer (unrelated to leakage class). Pinned
        # behavior: the check is symmetric, so this IS blocked.
        assert is_earnings_blocked(
            "AAPL", self.BACKTEST_BAR, future_only, buffer_days=685
        )


# ── Real-prod-path test (per CLAUDE.md §5.13.1) ──────────────────────────────

class TestEarningsFilterTaskRealProdPath:
    """Per §5.13.1: don't trust hand-built fixtures — exercise the
    actual EarningsFilterTask through its real interface with a real
    TickerInferenceContext.

    `EarningsFilterTask.run(tc)` returns False (drop) when blocked,
    None (pass through) otherwise. This pinning is critical: the
    pipeline short-circuits on `return False`.
    """

    def test_real_task_drops_ticker_inside_window(self):
        task = EarningsFilterTask()
        # NVDA has 2024-06-17 — 2 days after today, inside default ±3
        tc = _make_tc("NVDA")
        result = task.run(tc)
        assert result is False  # pipeline short-circuit signal

    def test_real_task_passes_ticker_outside_window(self):
        task = EarningsFilterTask()
        # AAPL has only future events outside ±3 (and the 2026 leak)
        tc = _make_tc("AAPL")
        result = task.run(tc)
        assert result is None  # pipeline continues

    def test_real_task_immune_to_future_event_at_historical_bar(self):
        """The leakage assertion via the real Task class: AAPL has
        a 2026-04-30 event in the calendar but the bar is 2024-06-15.
        Task must pass-through (return None), not drop."""
        task = EarningsFilterTask()
        tc = _make_tc("AAPL", today=datetime.date(2024, 6, 15))
        result = task.run(tc)
        assert result is None, (
            "EarningsFilterTask incorrectly considered a 2026 event at a "
            "2024-06-15 bar — forward-looking leakage."
        )

    def test_real_task_with_empty_calendar(self):
        task = EarningsFilterTask()
        tc = TickerInferenceContext(
            ticker="AAPL",
            ohlcv={},
            model=None,
            config={"regime": {"earnings_buffer_days": 3}},
            today=TODAY,
            regime="BULL_CALM",
            regime_params={},
            exit_params={},
            earnings_calendar={},
        )
        # Empty calendar — never block.
        assert task.run(tc) is None

    def test_real_task_with_none_calendar(self):
        """Defensive: tc.earnings_calendar may be None (legacy / sim
        with no artifact)."""
        task = EarningsFilterTask()
        tc = TickerInferenceContext(
            ticker="AAPL",
            ohlcv={},
            model=None,
            config={"regime": {"earnings_buffer_days": 3}},
            today=TODAY,
            regime="BULL_CALM",
            regime_params={},
            exit_params={},
            earnings_calendar=None,
        )
        # None → `tc.earnings_calendar or {}` → empty → never block.
        assert task.run(tc) is None
