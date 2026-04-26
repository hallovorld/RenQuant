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
