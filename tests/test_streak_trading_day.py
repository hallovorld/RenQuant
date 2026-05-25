"""Tests for STREAK-TRADING-DAY audit fix (2026-04-26 round-7).

User spec: sell_streak should ONLY count NYSE trading days. Pre-fix,
running e2e on a Sunday (calendar day, market closed) incremented
streak from 2 → 3 → triggered model_sell on GOOG/AMZN/BA inside 24
hours — even though these were not 3 actual trading-day signals.

Fix: only ++ streak when today is an NYSE trading day (excluded:
Sat/Sun + US market holidays). Reset on non-sell signal still requires
a trading day (otherwise a Sunday e2e would clear a legitimate streak).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.exits import (  # noqa: E402
    HoldingState,
    _is_nyse_trading_day,
    check_model_sell,
    nyse_trading_days_between,
)
decide_model_exit = check_model_sell   # alias for these tests


# ── _is_nyse_trading_day ──────────────────────────────────────────────────────

class TestIsNyseTradingDay:
    def test_monday_business_day_is_trading(self):
        # Monday 2026-04-27 (after today's Sunday)
        assert _is_nyse_trading_day(datetime.date(2026, 4, 27)) is True

    def test_sunday_not_trading(self):
        # Sunday 2026-04-26 — today
        assert _is_nyse_trading_day(datetime.date(2026, 4, 26)) is False

    def test_saturday_not_trading(self):
        # Saturday 2026-04-25
        assert _is_nyse_trading_day(datetime.date(2026, 4, 25)) is False

    def test_christmas_holiday(self):
        # Dec 25 — closed
        assert _is_nyse_trading_day(datetime.date(2026, 12, 25)) is False

    def test_new_years_day(self):
        # Jan 1 — closed
        assert _is_nyse_trading_day(datetime.date(2026, 1, 1)) is False


# ── STREAK-TRADING-DAY — sell streak respects NYSE calendar ───────────────────

def _holding(entry_days_ago: int = 60, sell_streak: int = 0,
              last_inc: datetime.date | None = None) -> HoldingState:
    today = datetime.date(2026, 4, 27)   # Monday
    return HoldingState(
        entry_price=100.0,
        entry_date=today - datetime.timedelta(days=entry_days_ago),
        high_watermark=110.0,
        sell_streak=sell_streak,
        last_streak_inc_date=last_inc,
        shares=10,
    )


class TestStreakTradingDay:
    def _first_date_with_n_trading_days(self, entry: datetime.date, n: int) -> datetime.date:
        cur = entry
        while nyse_trading_days_between(entry, cur) < n:
            cur += datetime.timedelta(days=1)
        return cur

    def test_sunday_does_not_increment(self):
        """User-found bug: e2e on Sunday should NOT inc streak."""
        sunday = datetime.date(2026, 4, 26)
        state = _holding(sell_streak=2)
        new_state, exit_sig = decide_model_exit(
            "sell", state, 3, 30, sunday,
        )
        assert new_state.sell_streak == 2, \
            "Sunday must NOT increment streak (user spec)"
        assert exit_sig.should_exit is False, \
            "model_sell must NOT trigger when streak unchanged"

    def test_monday_increments_normally(self):
        monday = datetime.date(2026, 4, 27)
        state = _holding(sell_streak=2)
        new_state, exit_sig = decide_model_exit(
            "sell", state, 3, 30, monday,
        )
        assert new_state.sell_streak == 3
        assert exit_sig.should_exit is True

    def test_saturday_does_not_increment(self):
        saturday = datetime.date(2026, 4, 25)
        state = _holding(sell_streak=2)
        new_state, exit_sig = decide_model_exit(
            "sell", state, 3, 30, saturday,
        )
        assert new_state.sell_streak == 2
        assert exit_sig.should_exit is False

    def test_holiday_does_not_increment(self):
        christmas = datetime.date(2026, 12, 25)
        state = _holding(sell_streak=2)
        new_state, exit_sig = decide_model_exit(
            "sell", state, 3, 30, christmas,
        )
        assert new_state.sell_streak == 2
        assert exit_sig.should_exit is False

    def test_sunday_does_not_reset_streak(self):
        """If model_action='hold' on Sunday, don't reset streak either —
        Sunday isn't a real trading-day signal."""
        sunday = datetime.date(2026, 4, 26)
        state = _holding(sell_streak=2)
        new_state, _ = decide_model_exit(
            "hold", state, 3, 30, sunday,
        )
        assert new_state.sell_streak == 2, \
            "Sunday hold must NOT reset streak (preserve until Mon)"

    def test_min_hold_uses_trading_days_not_calendar_days(self):
        today = datetime.date(2026, 6, 15)
        entry = today - datetime.timedelta(days=60)
        assert nyse_trading_days_between(entry, today) < 60
        state = HoldingState(
            entry_price=100.0,
            entry_date=entry,
            high_watermark=110.0,
            sell_streak=2,
            shares=10,
        )

        new_state, exit_sig = decide_model_exit("sell", state, 3, 60, today)

        assert new_state.sell_streak == 2
        assert exit_sig.should_exit is False

    def test_min_hold_allows_after_configured_trading_days(self):
        entry = datetime.date(2026, 4, 16)
        today = self._first_date_with_n_trading_days(entry, 60)
        state = HoldingState(
            entry_price=100.0,
            entry_date=entry,
            high_watermark=110.0,
            sell_streak=2,
            shares=10,
        )

        new_state, exit_sig = decide_model_exit("sell", state, 3, 60, today)

        assert nyse_trading_days_between(entry, today) == 60
        assert new_state.sell_streak == 3
        assert exit_sig.should_exit is True

    def test_monday_hold_resets_streak(self):
        """A real trading-day hold signal does reset the streak."""
        monday = datetime.date(2026, 4, 27)
        state = _holding(sell_streak=2)
        new_state, _ = decide_model_exit(
            "hold", state, 3, 30, monday,
        )
        assert new_state.sell_streak == 0

    def test_multiple_sundays_in_a_row_no_change(self):
        """Sun + Sat: 2 calendar days but 0 trading days. Streak unchanged."""
        state = _holding(sell_streak=1)
        for d in [datetime.date(2026, 4, 25),    # Sat
                   datetime.date(2026, 4, 26)]:   # Sun
            state, _ = decide_model_exit(
                "sell", state, 3, 30, d,
            )
        assert state.sell_streak == 1


# ── STREAK-TRADING-DAY ROUND 2 — model_sell does NOT fire on non-trading day
#    even when streak is already ≥ required (user spec: "还有streak sell")
# ──────────────────────────────────────────────────────────────────────────────

class TestStreakTradingDayRound2:
    """User spec 2026-04-26 round-7 (round 2): "怎么他妈的还有streak sell！"

    The original STREAK-TRADING-DAY fix prevented INCREMENT on non-trading
    days but didn't prevent FIRING when the streak was already at threshold
    (e.g. 3) from a prior buggy run that incorrectly bumped it on a Sunday.

    The 2026-04-26 17:18 live e2e demonstrated the gap: streak=3 was already
    persisted in live_state.json from a 16:31 e2e BEFORE the fix landed at
    16:42. The 17:18 run correctly didn't bump streak (still 3) but
    model_sell fired because streak ≥ required.

    Strengthening: model_sell must NOT FIRE on a non-trading day, period.
    Symmetric with the increment guard. Path rules (stop_loss/trailing/
    SDL/max_hold) are unaffected — they go through other branches in
    compute_exits and represent risk management that must always fire.
    """

    def test_sunday_with_streak_at_threshold_does_not_fire(self):
        """The bug seen live on 2026-04-26: streak=3 persisted, Sun fires."""
        sunday = datetime.date(2026, 4, 26)
        state = _holding(sell_streak=3, last_inc=datetime.date(2026, 4, 24))
        new_state, exit_sig = decide_model_exit(
            "sell", state, 3, 30, sunday,
        )
        assert exit_sig.should_exit is False, (
            "Sunday must NOT fire model_sell even with streak=3 (live bug "
            "from 2026-04-26 17:18 e2e — GOOG/PLTR sold despite Sunday)"
        )
        assert new_state.sell_streak == 3, "streak preserved for Mon"

    def test_saturday_with_streak_at_threshold_does_not_fire(self):
        saturday = datetime.date(2026, 4, 25)
        state = _holding(sell_streak=3, last_inc=datetime.date(2026, 4, 24))
        new_state, exit_sig = decide_model_exit(
            "sell", state, 3, 30, saturday,
        )
        assert exit_sig.should_exit is False
        assert new_state.sell_streak == 3

    def test_holiday_with_streak_at_threshold_does_not_fire(self):
        christmas = datetime.date(2026, 12, 25)   # NYSE closed
        state = _holding(sell_streak=3, last_inc=datetime.date(2026, 12, 24))
        new_state, exit_sig = decide_model_exit(
            "sell", state, 3, 30, christmas,
        )
        assert exit_sig.should_exit is False
        assert new_state.sell_streak == 3

    def test_monday_with_streak_at_threshold_DOES_fire(self):
        """Regression: real trading day fires on accumulated streak."""
        monday = datetime.date(2026, 4, 27)
        state = _holding(sell_streak=3, last_inc=datetime.date(2026, 4, 24))
        new_state, exit_sig = decide_model_exit(
            "sell", state, 3, 30, monday,
        )
        assert exit_sig.should_exit is True
        assert exit_sig.exit_type == "model_sell"

    def test_streak_above_required_also_blocked_on_sunday(self):
        """If streak somehow exceeded required (>3), Sunday still blocks."""
        sunday = datetime.date(2026, 4, 26)
        state = _holding(sell_streak=10, last_inc=datetime.date(2026, 4, 24))
        new_state, exit_sig = decide_model_exit(
            "sell", state, 3, 30, sunday,
        )
        assert exit_sig.should_exit is False

    def test_hold_signal_on_sunday_with_streak_no_fire_no_reset(self):
        """Sunday hold: streak preserved, no fire (was already true)."""
        sunday = datetime.date(2026, 4, 26)
        state = _holding(sell_streak=3, last_inc=datetime.date(2026, 4, 24))
        new_state, exit_sig = decide_model_exit(
            "hold", state, 3, 30, sunday,
        )
        assert exit_sig.should_exit is False
        assert new_state.sell_streak == 3   # not reset (Sun=non-trading)
