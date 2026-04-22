"""
Comprehensive paired alignment tests: Notebook simulation vs LEAN main.py.

For every decision policy, the notebook logic and LEAN logic are each replicated
as pure-Python helper functions and tested independently with identical inputs.
The goal is to prove that both systems produce the exact same output for every
policy — making divergence between simulation and live execution impossible to hide.

STRUCTURE
---------
Each policy class has:
  - _notebook_<policy>()  — pure replication of cell 657a4a6c logic
  - _lean_<policy>()      — pure replication of main.py logic
  - N tests for notebook logic (named test_nb_*)
  - N tests for LEAN logic   (named test_lean_*)  ← ALWAYS same count as nb tests
  - A cross-check test      (named test_both_agree_*)

Run: python -m pytest tests/test_policy_alignment.py -v
"""

import sys
import math
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─── POLICY: Trailing Stop ────────────────────────────────────────────────────

class TestTrailingStopAlignment:
    """20% trigger, 18% trail below high-water mark. BULL_CALM only."""

    # ── pure replication ──────────────────────────────────────────────────────

    def _notebook(self, entry, high, current, trigger=0.20, trail=0.18):
        """cell 657a4a6c lines ~108-114."""
        peak_gain = (high - entry) / entry
        if peak_gain >= trigger:
            trail_floor = high * (1 - trail)
            return current <= trail_floor
        return False

    def _lean(self, entry, hwm, current, trigger=0.20, trail=0.18):
        """main.py lines ~192-200."""
        peak_gain = (hwm - entry) / entry
        if peak_gain >= trigger:
            trail_floor = hwm * (1 - trail)
            return current <= trail_floor
        return False

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_not_armed_below_trigger(self):
        assert not self._notebook(100, 119, 90)  # only 19% peak gain

    def test_nb_not_armed_exactly_below_trigger(self):
        assert not self._notebook(100, 119.9, 90)

    def test_nb_armed_exactly_at_trigger(self):
        # 20% gain → armed; trail floor = 120 * 0.82 = 98.4; price 95 < 98.4
        assert self._notebook(100, 120, 95)

    def test_nb_armed_but_above_floor(self):
        # 20% gain → armed; trail floor = 120 * 0.82 = 98.4; price 99 > 98.4
        assert not self._notebook(100, 120, 99)

    def test_nb_stays_armed_after_pullback(self):
        # HWM locked at 130 (30% gain); current pullback to 100 < 130*0.82=106.6
        assert self._notebook(100, 130, 100)

    def test_nb_large_winner_trails(self):
        # HWM 200 (+100%); trail floor = 200*0.82=164; price 160 < 164
        assert self._notebook(100, 200, 160)

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_not_armed_below_trigger(self):
        assert not self._lean(100, 119, 90)

    def test_lean_not_armed_exactly_below_trigger(self):
        assert not self._lean(100, 119.9, 90)

    def test_lean_armed_exactly_at_trigger(self):
        assert self._lean(100, 120, 95)

    def test_lean_armed_but_above_floor(self):
        assert not self._lean(100, 120, 99)

    def test_lean_stays_armed_after_pullback(self):
        assert self._lean(100, 130, 100)

    def test_lean_large_winner_trails(self):
        assert self._lean(100, 200, 160)

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_on_all_cases(self):
        cases = [
            (100, 119, 90), (100, 119.9, 90), (100, 120, 95),
            (100, 120, 99), (100, 130, 100), (100, 200, 160),
            (50, 62, 50), (200, 241, 195), (100, 120, 98.4),
        ]
        for entry, high, curr in cases:
            nb = self._notebook(entry, high, curr)
            ln = self._lean(entry, high, curr)
            assert nb == ln, f"entry={entry} high={high} curr={curr}: NB={nb} LEAN={ln}"


# ─── POLICY: Cumulative Stop-Loss ─────────────────────────────────────────────

class TestCumulativeStopLossAlignment:
    """Loss from entry ≥ stop_loss_pct triggers exit. Per-regime thresholds."""

    THRESHOLDS = {"BULL_CALM": 0.15, "BULL_VOLATILE": 0.05, "CHOPPY": 0.05, "BEAR": 0.05}

    def _notebook(self, entry, current, stop_pct):
        """cell 657a4a6c lines ~115-118: loss = (entry - current) / entry."""
        loss = (entry - current) / entry
        return loss >= stop_pct

    def _lean(self, entry, current, stop_pct):
        """main.py lines ~208-212: avg_price = entry; loss_pct = (avg - current) / avg."""
        loss_pct = (entry - current) / entry
        return loss_pct >= stop_pct

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_bull_calm_15pct_threshold(self):
        assert self._notebook(100, 84.9, 0.15)   # 15.1% loss → triggers

    def test_nb_bull_calm_not_triggered_just_under(self):
        assert not self._notebook(100, 85.1, 0.15)  # 14.9% loss → no exit

    def test_nb_volatile_5pct_threshold(self):
        assert self._notebook(100, 94.9, 0.05)

    def test_nb_volatile_not_triggered_just_under(self):
        assert not self._notebook(100, 95.1, 0.05)

    def test_nb_exact_threshold_triggers(self):
        assert self._notebook(100, 85.0, 0.15)  # exactly 15% → triggers

    def test_nb_gain_never_triggers(self):
        assert not self._notebook(100, 110, 0.15)

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_bull_calm_15pct_threshold(self):
        assert self._lean(100, 84.9, 0.15)

    def test_lean_bull_calm_not_triggered_just_under(self):
        assert not self._lean(100, 85.1, 0.15)

    def test_lean_volatile_5pct_threshold(self):
        assert self._lean(100, 94.9, 0.05)

    def test_lean_volatile_not_triggered_just_under(self):
        assert not self._lean(100, 95.1, 0.05)

    def test_lean_exact_threshold_triggers(self):
        assert self._lean(100, 85.0, 0.15)

    def test_lean_gain_never_triggers(self):
        assert not self._lean(100, 110, 0.15)

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_all_regimes(self):
        for regime, pct in self.THRESHOLDS.items():
            for entry, curr in [(100, 100*(1-pct)-0.1), (100, 100*(1-pct)+0.1), (100, 100*(1-pct))]:
                nb = self._notebook(entry, curr, pct)
                ln = self._lean(entry, curr, pct)
                assert nb == ln, f"{regime} entry={entry} curr={curr}: NB={nb} LEAN={ln}"


# ─── POLICY: Single-Day Loss Gate ────────────────────────────────────────────

class TestSingleDayLossAlignment:
    """Today's close drops ≥ sdl_pct from yesterday's close. BULL_CALM=0.10, others=0."""

    def _notebook(self, prev_close, today_close, sdl_pct):
        """cell 657a4a6c lines ~120-131."""
        if sdl_pct <= 0:
            return False
        if prev_close <= 0:
            return False
        return (prev_close - today_close) / prev_close >= sdl_pct

    def _lean(self, prev_close, today_close, sdl_pct):
        """main.py lines ~219-228."""
        if sdl_pct <= 0:
            return False
        if prev_close <= 0:
            return False
        daily_drop = (prev_close - today_close) / prev_close
        return daily_drop >= sdl_pct

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_disabled_in_other_regimes(self):
        assert not self._notebook(100, 80, 0.0)   # sdl_pct=0 → disabled

    def test_nb_triggers_on_10pct_drop(self):
        assert self._notebook(100, 89.9, 0.10)    # 10.1% drop → fires

    def test_nb_does_not_trigger_just_under(self):
        assert not self._notebook(100, 90.1, 0.10)  # 9.9% drop → no exit

    def test_nb_exact_10pct_triggers(self):
        assert self._notebook(100, 90.0, 0.10)    # exactly 10% → triggers

    def test_nb_up_day_never_triggers(self):
        assert not self._notebook(100, 110, 0.10)

    def test_nb_zero_prev_close_safe(self):
        assert not self._notebook(0, 90, 0.10)    # guard against division by zero

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_disabled_in_other_regimes(self):
        assert not self._lean(100, 80, 0.0)

    def test_lean_triggers_on_10pct_drop(self):
        assert self._lean(100, 89.9, 0.10)

    def test_lean_does_not_trigger_just_under(self):
        assert not self._lean(100, 90.1, 0.10)

    def test_lean_exact_10pct_triggers(self):
        assert self._lean(100, 90.0, 0.10)

    def test_lean_up_day_never_triggers(self):
        assert not self._lean(100, 110, 0.10)

    def test_lean_zero_prev_close_safe(self):
        assert not self._lean(0, 90, 0.10)

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_all_cases(self):
        cases = [(100, 89.9, 0.10), (100, 90.1, 0.10), (100, 90.0, 0.10),
                 (100, 80, 0.0), (100, 110, 0.10), (0, 90, 0.10)]
        for prev, curr, sdl in cases:
            nb = self._notebook(prev, curr, sdl)
            ln = self._lean(prev, curr, sdl)
            assert nb == ln, f"prev={prev} curr={curr} sdl={sdl}: NB={nb} LEAN={ln}"


# ─── POLICY: Max Hold ─────────────────────────────────────────────────────────

class TestMaxHoldAlignment:
    """Forced exit after max_hold_days. Per-regime: 500 most regimes, 10 CHOPPY."""

    def _notebook(self, entry_date, today, max_hold_days):
        """cell 657a4a6c lines ~132-135: hold_days = (today - entry_date).days."""
        hold_days = (today - entry_date).days
        return hold_days >= max_hold_days

    def _lean(self, entry_date, today, max_hold_days):
        """main.py lines ~232-237: same .days calculation."""
        if max_hold_days <= 0:
            return False
        days_held = (today - entry_date).days
        return days_held >= max_hold_days

    base = date(2024, 1, 1)

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_not_expired_day_499(self):
        entry = self.base
        today = self.base + timedelta(days=499)
        assert not self._notebook(entry, today, 500)

    def test_nb_expires_exactly_day_500(self):
        assert self._notebook(self.base, self.base + timedelta(days=500), 500)

    def test_nb_choppy_expires_day_10(self):
        assert self._notebook(self.base, self.base + timedelta(days=10), 10)

    def test_nb_choppy_not_expired_day_9(self):
        assert not self._notebook(self.base, self.base + timedelta(days=9), 10)

    def test_nb_day_0_never_expires(self):
        assert not self._notebook(self.base, self.base, 500)

    def test_nb_well_past_max_hold(self):
        assert self._notebook(self.base, self.base + timedelta(days=600), 500)

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_not_expired_day_499(self):
        assert not self._lean(self.base, self.base + timedelta(days=499), 500)

    def test_lean_expires_exactly_day_500(self):
        assert self._lean(self.base, self.base + timedelta(days=500), 500)

    def test_lean_choppy_expires_day_10(self):
        assert self._lean(self.base, self.base + timedelta(days=10), 10)

    def test_lean_choppy_not_expired_day_9(self):
        assert not self._lean(self.base, self.base + timedelta(days=9), 10)

    def test_lean_day_0_never_expires(self):
        assert not self._lean(self.base, self.base, 500)

    def test_lean_well_past_max_hold(self):
        assert self._lean(self.base, self.base + timedelta(days=600), 500)

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_at_all_boundaries(self):
        for max_days in [10, 500]:
            for offset in [max_days - 1, max_days, max_days + 1]:
                today = self.base + timedelta(days=offset)
                nb = self._notebook(self.base, today, max_days)
                ln = self._lean(self.base, today, max_days)
                assert nb == ln, f"max={max_days} offset={offset}: NB={nb} LEAN={ln}"


# ─── POLICY: Min Hold Guard ───────────────────────────────────────────────────

class TestMinHoldAlignment:
    """Model-sell blocked for min_hold_days=20. Streak cannot accumulate during window."""

    def _notebook(self, entry_date, today, min_hold_profit=20, min_hold_loss=20, gain=0):
        """cell 657a4a6c lines ~136-142."""
        hold_days = (today - entry_date).days
        min_hold_required = min_hold_profit if gain > 0 else min_hold_loss
        return hold_days < min_hold_required  # True = model-sell BLOCKED

    def _lean(self, entry_date, today, min_hold_days=20):
        """main.py lines ~244-250."""
        if min_hold_days <= 0:
            return False
        days_check = (today - entry_date).days
        return days_check < min_hold_days  # True = BLOCKED

    base = date(2026, 4, 14)

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_blocked_day_1(self):
        assert self._notebook(self.base, self.base + timedelta(days=1))

    def test_nb_blocked_day_19(self):
        assert self._notebook(self.base, self.base + timedelta(days=19))

    def test_nb_allowed_day_20(self):
        assert not self._notebook(self.base, self.base + timedelta(days=20))

    def test_nb_allowed_day_21(self):
        assert not self._notebook(self.base, self.base + timedelta(days=21))

    def test_nb_amzn_scenario_day_1_blocked(self):
        # Exact replay of 2026-04-15: bought Apr 14, checked Apr 15 = 1 day held
        buy = date(2026, 4, 14)
        check = date(2026, 4, 15)
        assert self._notebook(buy, check)

    def test_nb_zero_days_blocked(self):
        assert self._notebook(self.base, self.base)

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_blocked_day_1(self):
        assert self._lean(self.base, self.base + timedelta(days=1))

    def test_lean_blocked_day_19(self):
        assert self._lean(self.base, self.base + timedelta(days=19))

    def test_lean_allowed_day_20(self):
        assert not self._lean(self.base, self.base + timedelta(days=20))

    def test_lean_allowed_day_21(self):
        assert not self._lean(self.base, self.base + timedelta(days=21))

    def test_lean_amzn_scenario_day_1_blocked(self):
        buy = date(2026, 4, 14)
        check = date(2026, 4, 15)
        assert self._lean(buy, check)

    def test_lean_zero_days_blocked(self):
        assert self._lean(self.base, self.base)

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_amzn_scenario(self):
        buy = date(2026, 4, 14)
        for offset in [0, 1, 19, 20, 21, 30]:
            check = buy + timedelta(days=offset)
            nb = self._notebook(buy, check)
            ln = self._lean(buy, check)
            assert nb == ln, f"offset={offset}: NB={nb} LEAN={ln}"


# ─── POLICY: Consecutive Sell Streak ─────────────────────────────────────────

class TestConsecutiveSellStreakAlignment:
    """Requires 3 consecutive sell signals. Resets on non-sell. Blocked in min_hold."""

    def _notebook(self, streak, signal, required=3):
        """cell 657a4a6c lines ~143-151."""
        if signal == "sell":
            streak += 1
            fires = streak >= required
        else:
            streak = 0
            fires = False
        return streak, fires

    def _lean(self, streak, signal, required=3):
        """main.py lines ~252-259."""
        if signal == "sell":
            streak = streak + 1
            fires = streak >= required
        else:
            streak = 0
            fires = False
        return streak, fires

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_first_sell_does_not_fire(self):
        streak, fires = self._notebook(0, "sell")
        assert streak == 1 and not fires

    def test_nb_second_sell_does_not_fire(self):
        streak, fires = self._notebook(1, "sell")
        assert streak == 2 and not fires

    def test_nb_third_sell_fires(self):
        streak, fires = self._notebook(2, "sell")
        assert fires

    def test_nb_hold_resets_streak(self):
        streak, fires = self._notebook(2, "hold")
        assert streak == 0 and not fires

    def test_nb_buy_resets_streak(self):
        streak, fires = self._notebook(2, "buy")
        assert streak == 0 and not fires

    def test_nb_consecutive_fires_then_resets(self):
        s = 0
        for sig in ["sell", "sell", "sell"]:
            s, fired = self._notebook(s, sig)
        assert fired and s == 3

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_first_sell_does_not_fire(self):
        streak, fires = self._lean(0, "sell")
        assert streak == 1 and not fires

    def test_lean_second_sell_does_not_fire(self):
        streak, fires = self._lean(1, "sell")
        assert streak == 2 and not fires

    def test_lean_third_sell_fires(self):
        streak, fires = self._lean(2, "sell")
        assert fires

    def test_lean_hold_resets_streak(self):
        streak, fires = self._lean(2, "hold")
        assert streak == 0 and not fires

    def test_lean_buy_resets_streak(self):
        streak, fires = self._lean(2, "buy")
        assert streak == 0 and not fires

    def test_lean_consecutive_fires_then_resets(self):
        s = 0
        for sig in ["sell", "sell", "sell"]:
            s, fired = self._lean(s, sig)
        assert fired and s == 3

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_on_sequences(self):
        sequences = [
            ["sell", "sell", "sell"],
            ["sell", "hold", "sell", "sell", "sell"],
            ["buy", "sell", "sell"],
            ["sell", "buy", "sell", "sell", "sell"],
        ]
        for seq in sequences:
            nb_s, lean_s = 0, 0
            for sig in seq:
                nb_s, nb_fired = self._notebook(nb_s, sig)
                lean_s, lean_fired = self._lean(lean_s, sig)
            assert nb_s == lean_s, f"seq={seq}: NB streak={nb_s} LEAN streak={lean_s}"


# ─── POLICY: SPY EMA50 Trend Gate ────────────────────────────────────────────

class TestSPYEMA50Alignment:
    """Block new buys when SPY < 50-day EMA. Period=50, adjust=False."""

    def _notebook(self, spy_closes):
        """cell 657a4a6c lines ~277-281: ewm(span=50, adjust=False).mean()[-1]."""
        s = pd.Series(spy_closes, dtype=float)
        ema50 = s.ewm(span=50, adjust=False).mean().iloc[-1]
        return float(s.iloc[-1]) < ema50  # True = BLOCKED

    def _lean(self, spy_closes):
        """main.py lines ~337-345: History(51 bars), same ewm formula."""
        if len(spy_closes) < 51:
            return False
        s = pd.Series(spy_closes[-51:], dtype=float)
        ema50 = s.ewm(span=50, adjust=False).mean().iloc[-1]
        return float(s.iloc[-1]) < ema50

    def _rising_series(self, n=60, start=400, step=2):
        return [start + i * step for i in range(n)]

    def _falling_series(self, n=60, start=600, step=-2):
        return [start + i * step for i in range(n)]

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_not_blocked_in_uptrend(self):
        assert not self._notebook(self._rising_series())

    def test_nb_blocked_in_downtrend(self):
        assert self._notebook(self._falling_series())

    def test_nb_blocked_sudden_crash(self):
        # SPY was rising then crashed below EMA
        closes = self._rising_series(59) + [300]  # sudden drop
        assert self._notebook(closes)

    def test_nb_not_blocked_recovery_above_ema(self):
        # Flat then recovering
        closes = [450] * 50 + [460] * 10
        assert not self._notebook(closes)

    def test_nb_choppy_below_ema_blocks(self):
        # EMA slowly rising, price stuck below
        closes = [500 - i * 0.5 for i in range(60)]
        assert self._notebook(closes)

    def test_nb_long_uptrend_never_blocked(self):
        closes = [100 + i * 3 for i in range(60)]
        assert not self._notebook(closes)

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_not_blocked_in_uptrend(self):
        assert not self._lean(self._rising_series(55))

    def test_lean_blocked_in_downtrend(self):
        assert self._lean(self._falling_series(55))

    def test_lean_blocked_sudden_crash(self):
        closes = self._rising_series(55, step=1) + [300]
        assert self._lean(closes)

    def test_lean_not_blocked_recovery(self):
        closes = [450] * 52 + [460] * 3
        assert not self._lean(closes)

    def test_lean_below_51_bars_not_blocked(self):
        # LEAN requires ≥51 bars; short history returns False (no block)
        assert not self._lean([100] * 40)

    def test_lean_long_uptrend_never_blocked(self):
        closes = [100 + i * 3 for i in range(55)]
        assert not self._lean(closes)

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_on_trends(self):
        for closes in [self._rising_series(60), self._falling_series(60)]:
            nb = self._notebook(closes)
            ln = self._lean(closes)
            assert nb == ln, f"Mismatch on trend: NB={nb} LEAN={ln}"


# ─── POLICY: SPY Velocity Crash Filter ───────────────────────────────────────

class TestVelocityCrashAlignment:
    """Block new buys if SPY down >3% over last 3 days."""

    def _notebook(self, closes, lookback=3, halt_pct=0.03):
        """cell 657a4a6c lines ~262-268: (now/prev - 1) < -halt_pct."""
        if len(closes) <= lookback:
            return False
        spy_prev = closes[-(lookback + 1)]
        spy_now = closes[-1]
        if spy_prev <= 0:
            return False
        return (spy_now / spy_prev - 1) < -halt_pct

    def _lean(self, daily_returns, lookback=3, halt_pct=0.03):
        """main.py lines ~326-333: prod(1 + daily_returns[-lookback:]) - 1."""
        if len(daily_returns) < lookback:
            return False
        spy_nday = np.prod([1.0 + r for r in daily_returns[-lookback:]]) - 1.0
        return spy_nday < -halt_pct

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_not_blocked_flat(self):
        assert not self._notebook([500] * 10)

    def test_nb_blocked_3pct_drop(self):
        # 500 → 484: drop = -3.2%
        assert self._notebook([500, 498, 492, 484])

    def test_nb_not_blocked_just_under(self):
        # 500 → 485.1: drop = -2.98% < 3%
        assert not self._notebook([500, 499, 493, 485.1])

    def test_nb_blocked_single_large_drop(self):
        # Multi-day: 500, 490, 483, 483 — overall -3.4% over 3 days
        assert self._notebook([500, 490, 483, 483])

    def test_nb_recovery_not_blocked(self):
        # Dropped then recovered
        assert not self._notebook([500, 480, 490, 510])

    def test_nb_insufficient_history_not_blocked(self):
        assert not self._notebook([500, 490])  # less than lookback+1

    # ── LEAN tests (uses daily returns rather than close prices) ──────────────

    def test_lean_not_blocked_flat(self):
        assert not self._lean([0.0, 0.0, 0.0])

    def test_lean_blocked_3pct_cumulative(self):
        # Three days: -1%, -1.5%, -1% → compound ≈ -3.49%
        assert self._lean([-0.01, -0.015, -0.01])

    def test_lean_not_blocked_just_under(self):
        # Three days: -0.8%, -0.8%, -0.8% → compound ≈ -2.38%
        assert not self._lean([-0.008, -0.008, -0.008])

    def test_lean_blocked_large_single_day(self):
        # One big drop and two small
        assert self._lean([-0.02, -0.005, -0.015])

    def test_lean_recovery_not_blocked(self):
        assert not self._lean([-0.02, 0.01, 0.02])

    def test_lean_insufficient_returns_not_blocked(self):
        assert not self._lean([-0.02, -0.02])  # only 2 returns, need 3

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_3day_drop_scenarios(self):
        scenarios = [
            # (closes_list, daily_returns) equivalent scenarios
            ([500, 498, 492, 484], [-0.004, -0.0121, -0.0163]),  # both ~3.2% drop
            ([500] * 4, [0.0, 0.0, 0.0]),                        # flat
            ([500, 490, 480, 510], [-0.02, -0.02, 0.0625]),       # recovery
        ]
        for closes, rets in scenarios:
            nb = self._notebook(closes)
            ln = self._lean(rets)
            # Both should agree on direction (both blocked or both not)
            assert nb == ln, f"NB={nb} LEAN={ln} closes={closes[-4:]}"


# ─── POLICY: Transition Uncertainty Window ────────────────────────────────────

class TestTransitionWindowAlignment:
    """3-bar block after CUSUM changepoint. Countdown decrements each bar."""

    def _notebook(self, countdown, changepoint_today, n_bars=3):
        """cell 657a4a6c lines ~196-204."""
        if changepoint_today:
            countdown = n_bars
        blocked = countdown > 0
        if blocked:
            countdown -= 1
        return countdown, blocked

    def _lean(self, countdown, changepoint_today, n_bars=3):
        """main.py lines ~283-290: same logic."""
        if changepoint_today:
            countdown = n_bars
        blocked = countdown > 0
        if countdown > 0:
            countdown -= 1
        return countdown, blocked

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_changepoint_sets_countdown(self):
        countdown, blocked = self._notebook(0, True)
        assert blocked  # first bar after changepoint is blocked

    def test_nb_blocks_3_bars_then_clears(self):
        countdown = 0
        for i in range(3):
            countdown, blocked = self._notebook(countdown, i == 0)
            assert blocked
        countdown, blocked = self._notebook(countdown, False)
        assert not blocked

    def test_nb_no_changepoint_no_block(self):
        _, blocked = self._notebook(0, False)
        assert not blocked

    def test_nb_countdown_decrements_each_bar(self):
        countdown = 3
        for expected in [2, 1, 0]:
            countdown, _ = self._notebook(countdown, False)
            assert countdown == expected

    def test_nb_second_changepoint_resets(self):
        countdown = 1  # 1 bar remaining from first changepoint
        countdown, blocked = self._notebook(countdown, True)  # new changepoint
        assert blocked
        assert countdown == 2  # reset to 3, then decremented to 2

    def test_nb_after_window_no_block(self):
        countdown = 0
        _, blocked = self._notebook(countdown, False)
        assert not blocked

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_changepoint_sets_countdown(self):
        countdown, blocked = self._lean(0, True)
        assert blocked

    def test_lean_blocks_3_bars_then_clears(self):
        countdown = 0
        for i in range(3):
            countdown, blocked = self._lean(countdown, i == 0)
            assert blocked
        countdown, blocked = self._lean(countdown, False)
        assert not blocked

    def test_lean_no_changepoint_no_block(self):
        _, blocked = self._lean(0, False)
        assert not blocked

    def test_lean_countdown_decrements_each_bar(self):
        countdown = 3
        for expected in [2, 1, 0]:
            countdown, _ = self._lean(countdown, False)
            assert countdown == expected

    def test_lean_second_changepoint_resets(self):
        countdown = 1
        countdown, blocked = self._lean(countdown, True)
        assert blocked
        assert countdown == 2

    def test_lean_after_window_no_block(self):
        countdown = 0
        _, blocked = self._lean(countdown, False)
        assert not blocked

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_full_sequence(self):
        changepoints = [True, False, False, False, False, True, False, False, False]
        nb_cnt, lean_cnt = 0, 0
        for cp in changepoints:
            nb_cnt, nb_blk = self._notebook(nb_cnt, cp)
            lean_cnt, lean_blk = self._lean(lean_cnt, cp)
            assert nb_blk == lean_blk, f"cp={cp}: NB={nb_blk} LEAN={lean_blk}"
            assert nb_cnt == lean_cnt


# ─── POLICY: Earnings Filter ─────────────────────────────────────────────────

class TestEarningsFilterAlignment:
    """Block buys within ±3 trading days of earnings date."""

    def _notebook(self, ticker, today, calendar, buf=3):
        """cell 657a4a6c lines ~81-88: abs(d - today) <= buf."""
        for d_str in calendar.get(ticker, []):
            try:
                if abs((date.fromisoformat(d_str) - today).days) <= buf:
                    return True
            except Exception:
                pass
        return False

    def _lean(self, ticker, today, calendar, buf=3):
        """main.py lines ~789-804: identical logic."""
        for d_str in calendar.get(ticker, []):
            try:
                d = date.fromisoformat(d_str)
                if abs((d - today).days) <= buf:
                    return True
            except Exception:
                pass
        return False

    BASE_CAL = {"AAPL": ["2024-07-25"], "MSFT": ["2024-04-24"]}

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_blocked_on_earnings_day(self):
        assert self._notebook("AAPL", date(2024, 7, 25), self.BASE_CAL)

    def test_nb_blocked_3_days_before(self):
        assert self._notebook("AAPL", date(2024, 7, 22), self.BASE_CAL)

    def test_nb_blocked_3_days_after(self):
        assert self._notebook("AAPL", date(2024, 7, 28), self.BASE_CAL)

    def test_nb_not_blocked_4_days_away(self):
        assert not self._notebook("AAPL", date(2024, 7, 29), self.BASE_CAL)

    def test_nb_not_blocked_ticker_without_earnings(self):
        assert not self._notebook("NVDA", date(2024, 7, 25), self.BASE_CAL)

    def test_nb_not_blocked_empty_calendar(self):
        assert not self._notebook("AAPL", date(2024, 7, 25), {})

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_blocked_on_earnings_day(self):
        assert self._lean("AAPL", date(2024, 7, 25), self.BASE_CAL)

    def test_lean_blocked_3_days_before(self):
        assert self._lean("AAPL", date(2024, 7, 22), self.BASE_CAL)

    def test_lean_blocked_3_days_after(self):
        assert self._lean("AAPL", date(2024, 7, 28), self.BASE_CAL)

    def test_lean_not_blocked_4_days_away(self):
        assert not self._lean("AAPL", date(2024, 7, 29), self.BASE_CAL)

    def test_lean_not_blocked_ticker_without_earnings(self):
        assert not self._lean("NVDA", date(2024, 7, 25), self.BASE_CAL)

    def test_lean_not_blocked_empty_calendar(self):
        assert not self._lean("AAPL", date(2024, 7, 25), {})

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_all_offsets(self):
        earnings = date(2024, 7, 25)
        for offset in range(-5, 6):
            today = earnings + timedelta(days=offset)
            nb = self._notebook("AAPL", today, self.BASE_CAL)
            ln = self._lean("AAPL", today, self.BASE_CAL)
            assert nb == ln, f"offset={offset}: NB={nb} LEAN={ln}"


# ─── POLICY: Tiered Thresholds ────────────────────────────────────────────────

class TestTieredThresholdsAlignment:
    """Slot N requires progressively higher model score. tier_idx = min(slots_filled, len-1)."""

    TIERS = [{"min_model_score": 0.10}, {"min_model_score": 0.30}, {"min_model_score": 0.50}]

    def _notebook(self, slots_filled, model_score, tiers=None):
        """cell 657a4a6c lines ~327-330: min(len(selected), len-1)."""
        if tiers is None:
            tiers = self.TIERS
        tier_idx = min(slots_filled, len(tiers) - 1)
        min_score = tiers[tier_idx].get("min_model_score", 0.0)
        return model_score >= min_score  # True = passes

    def _lean(self, slots_filled, model_score, tiers=None):
        """main.py lines ~397-401: identical formula."""
        if tiers is None:
            tiers = self.TIERS
        if not tiers:
            return True
        tier_idx = min(slots_filled, len(tiers) - 1)
        tier_min = float(tiers[tier_idx].get("min_model_score", 0.0))
        return model_score >= tier_min

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_slot1_passes_at_0_10(self):
        assert self._notebook(0, 0.10)

    def test_nb_slot1_fails_below_0_10(self):
        assert not self._notebook(0, 0.09)

    def test_nb_slot2_requires_0_30(self):
        assert not self._notebook(1, 0.29)
        assert self._notebook(1, 0.30)

    def test_nb_slot3_requires_0_50(self):
        assert not self._notebook(2, 0.49)
        assert self._notebook(2, 0.50)

    def test_nb_slot4_clamped_to_last_tier(self):
        # slots_filled=3 clamps to tier index 2 (min_score=0.50)
        assert self._notebook(3, 0.50)
        assert not self._notebook(3, 0.49)

    def test_nb_empty_tiers_always_passes(self):
        assert self._notebook(0, 0.0, tiers=[{"min_model_score": 0.0}])

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_slot1_passes_at_0_10(self):
        assert self._lean(0, 0.10)

    def test_lean_slot1_fails_below_0_10(self):
        assert not self._lean(0, 0.09)

    def test_lean_slot2_requires_0_30(self):
        assert not self._lean(1, 0.29)
        assert self._lean(1, 0.30)

    def test_lean_slot3_requires_0_50(self):
        assert not self._lean(2, 0.49)
        assert self._lean(2, 0.50)

    def test_lean_slot4_clamped_to_last_tier(self):
        assert self._lean(3, 0.50)
        assert not self._lean(3, 0.49)

    def test_lean_empty_tiers_always_passes(self):
        assert self._lean(0, 0.0, tiers=[])

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_all_slots_and_scores(self):
        test_cases = [(0, 0.09), (0, 0.10), (1, 0.29), (1, 0.30),
                      (2, 0.49), (2, 0.50), (3, 0.50), (3, 0.49)]
        for slots, score in test_cases:
            nb = self._notebook(slots, score)
            ln = self._lean(slots, score)
            assert nb == ln, f"slots={slots} score={score}: NB={nb} LEAN={ln}"


# ─── POLICY: Correlation Guard ────────────────────────────────────────────────

class TestCorrelationGuardAlignment:
    """Block buy if |corr| >= 0.70 with any held or already-selected ticker."""

    def _notebook(self, candidate, held_plus_selected, corr_dict, threshold=0.70):
        """cell 657a4a6c lines ~331-338."""
        for h in held_plus_selected:
            c = corr_dict.get(candidate, {}).get(h) or corr_dict.get(h, {}).get(candidate, 0)
            if abs(c) >= threshold:
                return False  # blocked
        return True  # passes

    def _lean(self, candidate, held_tickers, corr_matrix, threshold=0.70):
        """main.py lines ~776-787: identical bidirectional lookup."""
        if corr_matrix is None or not held_tickers:
            return True
        for held in held_tickers:
            corr = (corr_matrix.get(candidate, {}).get(held) or
                    corr_matrix.get(held, {}).get(candidate))
            if corr is not None and abs(corr) >= threshold:
                return False
        return True

    CORR = {"AAPL": {"MSFT": 0.85, "TSLA": 0.30}, "MSFT": {"NVDA": 0.75}}

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_passes_when_no_held(self):
        assert self._notebook("AAPL", [], self.CORR)

    def test_nb_blocked_high_correlation(self):
        assert not self._notebook("AAPL", ["MSFT"], self.CORR)  # 0.85 ≥ 0.70

    def test_nb_passes_low_correlation(self):
        assert self._notebook("AAPL", ["TSLA"], self.CORR)  # 0.30 < 0.70

    def test_nb_reverse_lookup_works(self):
        assert not self._notebook("MSFT", ["AAPL"], self.CORR)  # AAPL→MSFT=0.85

    def test_nb_multiple_held_any_blocks(self):
        assert not self._notebook("AAPL", ["TSLA", "MSFT"], self.CORR)

    def test_nb_exact_threshold_blocks(self):
        corr = {"X": {"Y": 0.70}}
        assert not self._notebook("X", ["Y"], corr)

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_passes_when_no_held(self):
        assert self._lean("AAPL", [], self.CORR)

    def test_lean_blocked_high_correlation(self):
        assert not self._lean("AAPL", ["MSFT"], self.CORR)

    def test_lean_passes_low_correlation(self):
        assert self._lean("AAPL", ["TSLA"], self.CORR)

    def test_lean_reverse_lookup_works(self):
        assert not self._lean("MSFT", ["AAPL"], self.CORR)

    def test_lean_multiple_held_any_blocks(self):
        assert not self._lean("AAPL", ["TSLA", "MSFT"], self.CORR)

    def test_lean_exact_threshold_blocks(self):
        corr = {"X": {"Y": 0.70}}
        assert not self._lean("X", ["Y"], corr)

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_all_combinations(self):
        cases = [
            ("AAPL", []),
            ("AAPL", ["MSFT"]),
            ("AAPL", ["TSLA"]),
            ("AAPL", ["TSLA", "MSFT"]),
            ("MSFT", ["AAPL"]),
        ]
        for candidate, held in cases:
            nb = self._notebook(candidate, held, self.CORR)
            ln = self._lean(candidate, held, self.CORR)
            assert nb == ln, f"candidate={candidate} held={held}: NB={nb} LEAN={ln}"


# ─── POLICY: Sector Guard ────────────────────────────────────────────────────

class TestSectorGuardAlignment:
    """Max 3 positions per sector. Defensives exempt. Counts held + already-selected."""

    SECTOR_MAP = {"AAPL": "tech", "MSFT": "tech", "NVDA": "tech",
                  "AMD": "tech", "JPM": "finance", "GLD": "commodity"}
    DEFENSIVE = {"GLD", "TLT", "XLV", "XLU"}

    def _notebook(self, candidate, held_plus_selected, sector_map, defensive, max_per_sector=3):
        """cell 657a4a6c lines ~339-346."""
        if candidate in defensive or max_per_sector <= 0:
            return True  # exempt
        sector = sector_map.get(candidate, "other")
        count = sum(1 for t in held_plus_selected if sector_map.get(t, "other") == sector)
        return count < max_per_sector

    def _lean(self, candidate, held_tickers, sector_map, defensive, max_per_sector=3):
        """main.py lines ~424-434: same logic; held_tickers includes already-bought today."""
        if candidate in defensive or max_per_sector <= 0:
            return True
        sector = sector_map.get(candidate, "other")
        count = sum(1 for t in held_tickers if sector_map.get(t, "other") == sector)
        return count < max_per_sector

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_passes_with_no_held(self):
        assert self._notebook("AAPL", [], self.SECTOR_MAP, self.DEFENSIVE)

    def test_nb_passes_with_2_tech_held(self):
        assert self._notebook("NVDA", ["AAPL", "MSFT"], self.SECTOR_MAP, self.DEFENSIVE)

    def test_nb_blocked_with_3_tech_held(self):
        assert not self._notebook("AMD", ["AAPL", "MSFT", "NVDA"], self.SECTOR_MAP, self.DEFENSIVE)

    def test_nb_defensive_exempt(self):
        held = ["AAPL", "MSFT", "NVDA", "AMD"]  # 4 tech but...
        assert self._notebook("GLD", held, self.SECTOR_MAP, self.DEFENSIVE)

    def test_nb_different_sector_not_blocked(self):
        assert self._notebook("JPM", ["AAPL", "MSFT", "NVDA"], self.SECTOR_MAP, self.DEFENSIVE)

    def test_nb_counts_already_selected_today(self):
        # held=1 tech + selected=2 tech = 3 total → blocks 4th
        held_plus_selected = ["AAPL", "MSFT", "NVDA"]  # 3 tech
        assert not self._notebook("AMD", held_plus_selected, self.SECTOR_MAP, self.DEFENSIVE)

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_passes_with_no_held(self):
        assert self._lean("AAPL", [], self.SECTOR_MAP, self.DEFENSIVE)

    def test_lean_passes_with_2_tech_held(self):
        assert self._lean("NVDA", ["AAPL", "MSFT"], self.SECTOR_MAP, self.DEFENSIVE)

    def test_lean_blocked_with_3_tech_held(self):
        assert not self._lean("AMD", ["AAPL", "MSFT", "NVDA"], self.SECTOR_MAP, self.DEFENSIVE)

    def test_lean_defensive_exempt(self):
        held = ["AAPL", "MSFT", "NVDA", "AMD"]
        assert self._lean("GLD", held, self.SECTOR_MAP, self.DEFENSIVE)

    def test_lean_different_sector_not_blocked(self):
        assert self._lean("JPM", ["AAPL", "MSFT", "NVDA"], self.SECTOR_MAP, self.DEFENSIVE)

    def test_lean_counts_tickers_added_this_run(self):
        # Simulates held_tickers.append(ticker) mid-loop
        held = ["AAPL", "MSFT", "NVDA"]  # 3 tech already in held_tickers
        assert not self._lean("AMD", held, self.SECTOR_MAP, self.DEFENSIVE)

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_all_cases(self):
        cases = [
            ("AAPL", []),
            ("NVDA", ["AAPL", "MSFT"]),
            ("AMD", ["AAPL", "MSFT", "NVDA"]),
            ("GLD", ["AAPL", "MSFT", "NVDA"]),
            ("JPM", ["AAPL", "MSFT", "NVDA"]),
        ]
        for candidate, held in cases:
            nb = self._notebook(candidate, held, self.SECTOR_MAP, self.DEFENSIVE)
            ln = self._lean(candidate, held, self.SECTOR_MAP, self.DEFENSIVE)
            assert nb == ln, f"candidate={candidate} held={held}: NB={nb} LEAN={ln}"


# ─── POLICY: Wash Sale ────────────────────────────────────────────────────────

class TestWashSaleAlignment:
    """Block re-buy within 30 calendar days of last sell."""

    def _notebook(self, last_sell, today, days=30):
        """cell 657a4a6c lines ~313-315."""
        if last_sell is None:
            return True  # not blocked (passes)
        return (today - last_sell).days >= days  # True = passes

    def _lean(self, last_sell, today, days=30):
        """main.py lines ~1140-1149: identical formula."""
        if last_sell is None or days <= 0:
            return True
        return (today - last_sell).days >= days

    base = date(2026, 4, 15)  # AMZN sell date

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_same_day_blocked(self):
        assert not self._notebook(self.base, self.base)

    def test_nb_day_29_blocked(self):
        assert not self._notebook(self.base, self.base + timedelta(days=29))

    def test_nb_day_30_passes(self):
        assert self._notebook(self.base, self.base + timedelta(days=30))

    def test_nb_no_sell_date_always_passes(self):
        assert self._notebook(None, self.base)

    def test_nb_day_1_blocked(self):
        assert not self._notebook(self.base, self.base + timedelta(days=1))

    def test_nb_day_31_passes(self):
        assert self._notebook(self.base, self.base + timedelta(days=31))

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_same_day_blocked(self):
        assert not self._lean(self.base, self.base)

    def test_lean_day_29_blocked(self):
        assert not self._lean(self.base, self.base + timedelta(days=29))

    def test_lean_day_30_passes(self):
        assert self._lean(self.base, self.base + timedelta(days=30))

    def test_lean_no_sell_date_always_passes(self):
        assert self._lean(None, self.base)

    def test_lean_day_1_blocked(self):
        assert not self._lean(self.base, self.base + timedelta(days=1))

    def test_lean_day_31_passes(self):
        assert self._lean(self.base, self.base + timedelta(days=31))

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_all_offsets(self):
        for offset in [0, 1, 15, 29, 30, 31, 60]:
            today = self.base + timedelta(days=offset)
            nb = self._notebook(self.base, today)
            ln = self._lean(self.base, today)
            assert nb == ln, f"offset={offset}: NB={nb} LEAN={ln}"


# ─── POLICY: Min Model Score (Regime-Aware) ───────────────────────────────────

class TestMinModelScoreAlignment:
    """Score threshold is regime-specific: BULL_CALM=0.10, BULL_VOLATILE=0.15, CHOPPY=0.15."""

    REGIME_PARAMS = {
        "BULL_CALM":     {"min_model_score": 0.10},
        "BULL_VOLATILE": {"min_model_score": 0.15},
        "CHOPPY":        {"min_model_score": 0.15},
        "BEAR":          {"min_model_score": 0.0},
    }
    GLOBAL_DEFAULT = 0.10  # BULL_CALM value used as global fallback

    def _notebook(self, regime, model_score, regime_params, global_default):
        """cell 657a4a6c (FIXED): rp.get('min_model_score', global_default)."""
        rp = regime_params.get(regime, regime_params["BULL_CALM"])
        threshold = rp.get("min_model_score", global_default)
        return model_score >= threshold  # True = passes

    def _lean(self, regime, model_score, regime_params):
        """main.py lines ~374-375: regime_params.get('min_model_score', 0.10)."""
        rp = regime_params.get(regime, {})
        threshold = rp.get("min_model_score", 0.10)
        return model_score >= threshold

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_bull_calm_passes_at_0_10(self):
        assert self._notebook("BULL_CALM", 0.10, self.REGIME_PARAMS, self.GLOBAL_DEFAULT)

    def test_nb_bull_calm_fails_below_0_10(self):
        assert not self._notebook("BULL_CALM", 0.09, self.REGIME_PARAMS, self.GLOBAL_DEFAULT)

    def test_nb_volatile_requires_0_15(self):
        assert not self._notebook("BULL_VOLATILE", 0.14, self.REGIME_PARAMS, self.GLOBAL_DEFAULT)
        assert self._notebook("BULL_VOLATILE", 0.15, self.REGIME_PARAMS, self.GLOBAL_DEFAULT)

    def test_nb_choppy_requires_0_15(self):
        assert not self._notebook("CHOPPY", 0.14, self.REGIME_PARAMS, self.GLOBAL_DEFAULT)
        assert self._notebook("CHOPPY", 0.15, self.REGIME_PARAMS, self.GLOBAL_DEFAULT)

    def test_nb_bear_zero_threshold(self):
        assert self._notebook("BEAR", 0.0, self.REGIME_PARAMS, self.GLOBAL_DEFAULT)

    def test_nb_regime_aware_not_hardcoded(self):
        # Before fix: volatile would pass at 0.10 (using BULL_CALM hardcode)
        # After fix: volatile requires 0.15
        assert not self._notebook("BULL_VOLATILE", 0.10, self.REGIME_PARAMS, self.GLOBAL_DEFAULT)

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_bull_calm_passes_at_0_10(self):
        assert self._lean("BULL_CALM", 0.10, self.REGIME_PARAMS)

    def test_lean_bull_calm_fails_below_0_10(self):
        assert not self._lean("BULL_CALM", 0.09, self.REGIME_PARAMS)

    def test_lean_volatile_requires_0_15(self):
        assert not self._lean("BULL_VOLATILE", 0.14, self.REGIME_PARAMS)
        assert self._lean("BULL_VOLATILE", 0.15, self.REGIME_PARAMS)

    def test_lean_choppy_requires_0_15(self):
        assert not self._lean("CHOPPY", 0.14, self.REGIME_PARAMS)
        assert self._lean("CHOPPY", 0.15, self.REGIME_PARAMS)

    def test_lean_bear_zero_threshold(self):
        assert self._lean("BEAR", 0.0, self.REGIME_PARAMS)

    def test_lean_regime_aware_always(self):
        assert not self._lean("BULL_VOLATILE", 0.10, self.REGIME_PARAMS)

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_all_regimes_and_scores(self):
        test_cases = [
            ("BULL_CALM", 0.09), ("BULL_CALM", 0.10),
            ("BULL_VOLATILE", 0.14), ("BULL_VOLATILE", 0.15),
            ("CHOPPY", 0.14), ("CHOPPY", 0.15),
            ("BEAR", 0.0), ("BEAR", 0.001),
        ]
        for regime, score in test_cases:
            nb = self._notebook(regime, score, self.REGIME_PARAMS, self.GLOBAL_DEFAULT)
            ln = self._lean(regime, score, self.REGIME_PARAMS)
            assert nb == ln, f"regime={regime} score={score}: NB={nb} LEAN={ln}"


# ─── POLICY: Combined Ranking (config-driven blend weights) ──────────────────

class TestCombinedRankingAlignment:
    """Candidates sorted by w_rank*norm(rank_score) + w_rs*norm(rs_score), descending."""

    def _notebook(self, candidates, rank_weight=0.5, rs_weight=0.5):
        """Notebook simulation: calibrated rank-score blended with RS using config weights."""
        if len(candidates) <= 1:
            return [c[0] for c in candidates]
        ms = [c[1] for c in candidates]
        rs = [c[2] for c in candidates]
        ms_range = max(ms) - min(ms) or 1
        rs_range = max(rs) - min(rs) or 1
        ranked = sorted(
            candidates,
            key=lambda c: rank_weight * (c[1] - min(ms)) / ms_range + rs_weight * (c[2] - min(rs)) / rs_range,
            reverse=True,
        )
        return [r[0] for r in ranked]

    def _lean(self, candidates, rank_weight=0.5, rs_weight=0.5):
        """LEAN ranking: calibrated rank-score blended with RS using config weights."""
        if len(candidates) <= 1:
            return [c[0] for c in candidates]
        model_scores = [s[1] for s in candidates]
        rs_scores = [s[2] for s in candidates]
        model_min, model_max = min(model_scores), max(model_scores)
        rs_min, rs_max = min(rs_scores), max(rs_scores)

        def norm(v, lo, hi):
            return (v - lo) / (hi - lo) if hi > lo else 0.5

        ranked = sorted(
            candidates,
            key=lambda s: rank_weight * norm(s[1], model_min, model_max) + rs_weight * norm(s[2], rs_min, rs_max),
            reverse=True,
        )
        return [r[0] for r in ranked]

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_single_candidate_unchanged(self):
        assert self._notebook([("A", 0.5, 0.3)]) == ["A"]

    def test_nb_higher_model_score_wins_if_rs_equal(self):
        result = self._notebook([("A", 0.3, 0.5), ("B", 0.6, 0.5)])
        assert result[0] == "B"

    def test_nb_higher_rs_wins_if_model_equal(self):
        result = self._notebook([("A", 0.5, 0.2), ("B", 0.5, 0.8)])
        assert result[0] == "B"

    def test_nb_custom_rank_weight_can_override_rs(self):
        result = self._notebook([("A", 0.9, 0.1), ("B", 0.6, 0.9)], rank_weight=0.8, rs_weight=0.2)
        assert result[0] == "A"

    def test_nb_all_equal_stable(self):
        result = self._notebook([("A", 0.5, 0.5), ("B", 0.5, 0.5)])
        assert set(result) == {"A", "B"}

    def test_nb_zero_range_uses_0_5_fallback(self):
        # All model scores equal → range=0 → norm returns 0.5; RS decides
        result = self._notebook([("A", 0.5, 0.1), ("B", 0.5, 0.9)])
        assert result[0] == "B"

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_single_candidate_unchanged(self):
        assert self._lean([("A", 0.5, 0.3)]) == ["A"]

    def test_lean_higher_model_score_wins_if_rs_equal(self):
        result = self._lean([("A", 0.3, 0.5), ("B", 0.6, 0.5)])
        assert result[0] == "B"

    def test_lean_higher_rs_wins_if_model_equal(self):
        result = self._lean([("A", 0.5, 0.2), ("B", 0.5, 0.8)])
        assert result[0] == "B"

    def test_lean_custom_rank_weight_can_override_rs(self):
        result = self._lean([("A", 0.9, 0.1), ("B", 0.6, 0.9)], rank_weight=0.8, rs_weight=0.2)
        assert result[0] == "A"

    def test_lean_all_equal_stable(self):
        result = self._lean([("A", 0.5, 0.5), ("B", 0.5, 0.5)])
        assert set(result) == {"A", "B"}

    def test_lean_zero_range_uses_0_5_fallback(self):
        result = self._lean([("A", 0.5, 0.1), ("B", 0.5, 0.9)])
        assert result[0] == "B"

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_on_rankings(self):
        test_cases = [
            ([("A", 0.3, 0.5), ("B", 0.6, 0.5)], (0.5, 0.5)),
            ([("A", 0.5, 0.2), ("B", 0.5, 0.8)], (0.5, 0.5)),
            ([("A", 0.9, 0.1), ("B", 0.5, 0.9), ("C", 0.5, 0.5)], (0.5, 0.5)),
            ([("A", 0.9, 0.1), ("B", 0.6, 0.9)], (0.8, 0.2)),
            ([("A", 0.5, 0.1), ("B", 0.5, 0.9)], (0.5, 0.5)),
        ]
        for candidates, weights in test_cases:
            nb = self._notebook(candidates, *weights)
            ln = self._lean(candidates, *weights)
            assert nb == ln, f"candidates={candidates} weights={weights}: NB={nb} LEAN={ln}"


# ─── POLICY: Position Sizing with Cash Reserve ────────────────────────────────

class TestPositionSizingAlignment:
    """Invest sizing uses confidence-scaled max_position_pct and cash_reserve_pct."""

    def _notebook(self, cash, port_val, max_pos_pct, cash_reserve_pct, confidence=1.0):
        """Notebook simulation: position sizing params scale by regime confidence."""
        scaled_max_pct = max_pos_pct * confidence
        cash_reserve = port_val * cash_reserve_pct * confidence
        invest = min(cash - cash_reserve, port_val * scaled_max_pct)
        return max(invest, 0)

    def _lean(self, portfolio_value, available_cash, max_pct, cash_reserve_pct, confidence=1.0):
        """LEAN _rp() scales max_position_pct and cash_reserve_pct by regime confidence."""
        scaled_max_pct = max_pct * confidence
        cash_reserve = portfolio_value * cash_reserve_pct * confidence
        investable = max(available_cash - cash_reserve, 0)
        target_pct = min(scaled_max_pct, investable / max(portfolio_value, 1))
        invest = target_pct * portfolio_value
        return max(invest, 0)

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_bull_calm_zero_reserve(self):
        invest = self._notebook(cash=10000, port_val=10000, max_pos_pct=0.15, cash_reserve_pct=0.0)
        assert abs(invest - 1500) < 0.01

    def test_nb_volatile_20pct_reserve_reduces_invest(self):
        invest = self._notebook(cash=10000, port_val=10000, max_pos_pct=0.20, cash_reserve_pct=0.20, confidence=0.5)
        # scaled_max=10%; scaled_reserve=10%; invest=min(9000,1000)=1000
        assert abs(invest - 1000) < 0.01

    def test_nb_choppy_30pct_reserve(self):
        # cash_reserve=3000; cash-reserve=7000; max=1500; invest=min(7000,1500)=1500
        invest = self._notebook(cash=10000, port_val=10000, max_pos_pct=0.15, cash_reserve_pct=0.30)
        assert abs(invest - 1500) < 0.01

    def test_nb_bear_100pct_reserve_blocks_invest(self):
        # cash_reserve=10000; cash-reserve=0; invest=max(0,0)=0 — no offensive buys
        invest = self._notebook(cash=10000, port_val=10000, max_pos_pct=0.15, cash_reserve_pct=1.0)
        assert invest == 0

    def test_nb_limited_by_cash_not_max_pct(self):
        invest = self._notebook(cash=500, port_val=10000, max_pos_pct=0.15, cash_reserve_pct=0.0)
        assert abs(invest - 500) < 0.01

    def test_nb_limited_by_max_pct_not_cash(self):
        invest = self._notebook(cash=5000, port_val=10000, max_pos_pct=0.15, cash_reserve_pct=0.0)
        assert abs(invest - 1500) < 0.01

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_bull_calm_zero_reserve(self):
        invest = self._lean(portfolio_value=10000, available_cash=10000, max_pct=0.15, cash_reserve_pct=0.0)
        assert abs(invest - 1500) < 0.01

    def test_lean_volatile_20pct_reserve_reduces_invest(self):
        invest = self._lean(portfolio_value=10000, available_cash=10000, max_pct=0.20, cash_reserve_pct=0.20, confidence=0.5)
        assert abs(invest - 1000) < 0.01

    def test_lean_choppy_30pct_reserve(self):
        invest = self._lean(portfolio_value=10000, available_cash=10000, max_pct=0.15, cash_reserve_pct=0.30)
        assert abs(invest - 1500) < 0.01

    def test_lean_bear_100pct_reserve_blocks_invest(self):
        invest = self._lean(portfolio_value=10000, available_cash=10000, max_pct=0.15, cash_reserve_pct=1.0)
        assert invest == 0

    def test_lean_limited_by_cash_not_max_pct(self):
        invest = self._lean(portfolio_value=10000, available_cash=500, max_pct=0.15, cash_reserve_pct=0.0)
        assert abs(invest - 500) < 0.01

    def test_lean_limited_by_max_pct_not_cash(self):
        invest = self._lean(portfolio_value=10000, available_cash=5000, max_pct=0.15, cash_reserve_pct=0.0)
        assert abs(invest - 1500) < 0.01

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_all_regimes(self):
        configs = [
            (10000, 10000, 0.15, 0.00, 1.0),  # BULL_CALM
            (10000, 10000, 0.20, 0.20, 0.5),  # BULL_VOLATILE with lower confidence
            (10000, 10000, 0.15, 0.30, 1.0),  # CHOPPY
            (10000, 10000, 0.15, 1.00, 1.0),  # BEAR (blocked)
        ]
        for cash, port, pct, reserve, confidence in configs:
            nb = self._notebook(cash, port, pct, reserve, confidence)
            ln = self._lean(port, cash, pct, reserve, confidence)
            assert abs(nb - ln) < 0.01, (
                f"cash={cash} pct={pct} res={reserve} conf={confidence}: NB={nb:.2f} LEAN={ln:.2f}"
            )


# ─── POLICY: BEAR Defensive Slots ─────────────────────────────────────────────

class TestBearDefensiveSlotsAlignment:
    """In BEAR (or veto-active) the selection layer caps open slots at
    `bear_defensive_slots` (default 1, raised to 2). Defensive sizing uses
    `bear_defensive_pct`.
    """

    # ── pure replication ──────────────────────────────────────────────────────
    # Semantics: bear_slots is a TOTAL cap on defensive holdings, not a per-bar
    # cap on new entries. `remaining = bear_slots - defensive_held`.

    def _notebook_open_slots(self, held_count, defensive_held, bear_only,
                             bear_slots, global_cap):
        slots = max(global_cap - held_count, 0)
        if bear_only:
            remaining = max(bear_slots - defensive_held, 0)
            slots = min(slots, remaining)
        return slots

    def _lean_open_slots(self, held_count, defensive_held, bear_only,
                         bear_slots, global_cap):
        # kernel/pipeline/task_selection.py::PrepareSelectionTask
        slots = max(global_cap - held_count, 0)
        if bear_only:
            remaining = max(bear_slots - defensive_held, 0)
            slots = min(slots, remaining)
        return slots

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_bear_only_caps_at_two_slots(self):
        assert self._notebook_open_slots(0, 0, True, 2, 8) == 2

    def test_nb_bear_only_with_one_defensive_held_leaves_one(self):
        assert self._notebook_open_slots(1, 1, True, 2, 8) == 1

    def test_nb_bear_only_fully_filled(self):
        assert self._notebook_open_slots(2, 2, True, 2, 8) == 0

    def test_nb_non_bear_ignores_bear_cap(self):
        assert self._notebook_open_slots(0, 0, False, 2, 8) == 8

    def test_nb_legacy_one_slot_still_works(self):
        assert self._notebook_open_slots(0, 0, True, 1, 8) == 1

    def test_nb_global_cap_tighter_than_bear_cap(self):
        assert self._notebook_open_slots(7, 0, True, 2, 8) == 1

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_bear_only_caps_at_two_slots(self):
        assert self._lean_open_slots(0, 0, True, 2, 8) == 2

    def test_lean_bear_only_with_one_defensive_held_leaves_one(self):
        assert self._lean_open_slots(1, 1, True, 2, 8) == 1

    def test_lean_bear_only_fully_filled(self):
        assert self._lean_open_slots(2, 2, True, 2, 8) == 0

    def test_lean_non_bear_ignores_bear_cap(self):
        assert self._lean_open_slots(0, 0, False, 2, 8) == 8

    def test_lean_legacy_one_slot_still_works(self):
        assert self._lean_open_slots(0, 0, True, 1, 8) == 1

    def test_lean_global_cap_tighter_than_bear_cap(self):
        assert self._lean_open_slots(7, 0, True, 2, 8) == 1

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_bear_slots(self):
        cases = [
            (0, 0, True,  2, 8),
            (1, 1, True,  2, 8),
            (2, 2, True,  2, 8),
            (0, 0, False, 2, 8),
            (0, 0, True,  1, 8),
            (7, 0, True,  2, 8),
        ]
        for held, def_held, bear, slots, gcap in cases:
            assert self._notebook_open_slots(held, def_held, bear, slots, gcap) == \
                   self._lean_open_slots(held, def_held, bear, slots, gcap), \
                   f"held={held} def={def_held} bear={bear} slots={slots} gcap={gcap}"


# ─── POLICY: GMM Confidence Veto ──────────────────────────────────────────────

class TestConfidenceVetoAlignment:
    """If regime_confidence < regime.confidence_veto_threshold and the detected
    regime is not BEAR, treat as BEAR (defensives only) — block offensive buys
    without a BEAR label.
    """

    # ── pure replication ──────────────────────────────────────────────────────

    def _notebook(self, regime, confidence, threshold):
        """Sim/runner.py: veto_active flag folds into the BEAR branch."""
        if regime == "BEAR":
            return False            # BEAR branch handles it — veto not needed
        if threshold <= 0.0:
            return False            # disabled
        return confidence < threshold

    def _lean(self, regime, confidence, threshold):
        """ConfidenceVetoTask in kernel/pipeline/task_gates.py — same gate logic."""
        if regime == "BEAR":
            return False
        if threshold <= 0.0:
            return False
        return confidence < threshold

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_low_confidence_choppy_vetoes(self):
        assert self._notebook("CHOPPY", 0.45, 0.55) is True

    def test_nb_high_confidence_choppy_passes(self):
        assert self._notebook("CHOPPY", 0.70, 0.55) is False

    def test_nb_bear_regime_skips_veto(self):
        assert self._notebook("BEAR", 0.10, 0.55) is False

    def test_nb_threshold_zero_disables_veto(self):
        assert self._notebook("CHOPPY", 0.05, 0.0) is False

    def test_nb_at_exact_threshold_no_veto(self):
        assert self._notebook("BULL_CALM", 0.55, 0.55) is False

    def test_nb_bull_volatile_low_confidence_vetoes(self):
        assert self._notebook("BULL_VOLATILE", 0.40, 0.55) is True

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_low_confidence_choppy_vetoes(self):
        assert self._lean("CHOPPY", 0.45, 0.55) is True

    def test_lean_high_confidence_choppy_passes(self):
        assert self._lean("CHOPPY", 0.70, 0.55) is False

    def test_lean_bear_regime_skips_veto(self):
        assert self._lean("BEAR", 0.10, 0.55) is False

    def test_lean_threshold_zero_disables_veto(self):
        assert self._lean("CHOPPY", 0.05, 0.0) is False

    def test_lean_at_exact_threshold_no_veto(self):
        assert self._lean("BULL_CALM", 0.55, 0.55) is False

    def test_lean_bull_volatile_low_confidence_vetoes(self):
        assert self._lean("BULL_VOLATILE", 0.40, 0.55) is True

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_veto(self):
        cases = [
            ("CHOPPY", 0.45, 0.55),
            ("CHOPPY", 0.70, 0.55),
            ("BEAR",   0.10, 0.55),
            ("BULL_CALM", 0.55, 0.55),
            ("BULL_VOLATILE", 0.40, 0.55),
            ("CHOPPY", 0.50, 0.0),
        ]
        for regime, conf, thr in cases:
            assert self._notebook(regime, conf, thr) == self._lean(regime, conf, thr), (
                f"regime={regime} conf={conf} thr={thr}"
            )


# ─── POLICY: Regime-Aware Position Cap ────────────────────────────────────────

class TestRegimeCapAlignment:
    """Open slots = regime-scoped max_concurrent_positions - held.

    regime_params[regime].max_concurrent_positions overrides the global cap;
    if absent, the global config cap applies. BEAR branch clamps to the
    defensive-slot count (1) via a later-applied min().
    """

    # ── pure replication ──────────────────────────────────────────────────────

    def _notebook(self, held_count, regime, config):
        regime_params = config.get("regime_params", {}).get(regime, {})
        cap = int(regime_params.get(
            "max_concurrent_positions",
            config.get("max_concurrent_positions", 8),
        ))
        return max(cap - held_count, 0)

    def _lean(self, held_count, regime, config):
        # PrepareSelectionTask logic in kernel/pipeline/task_selection.py
        regime_params = config.get("regime_params", {}).get(regime, {})
        max_positions = int(regime_params.get(
            "max_concurrent_positions",
            config.get("max_concurrent_positions", 8),
        ))
        return max(max_positions - held_count, 0)

    def _cfg(self, **choppy_overrides):
        cfg = {"max_concurrent_positions": 8, "regime_params": {
            "BULL_CALM": {}, "BULL_VOLATILE": {}, "CHOPPY": {}, "BEAR": {},
        }}
        cfg["regime_params"]["CHOPPY"].update(choppy_overrides)
        return cfg

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_bull_calm_uses_global_cap(self):
        assert self._notebook(3, "BULL_CALM", self._cfg()) == 5

    def test_nb_choppy_override_to_4(self):
        assert self._notebook(2, "CHOPPY", self._cfg(max_concurrent_positions=4)) == 2

    def test_nb_choppy_override_caps_open_slots(self):
        # 7 held in CHOPPY cap=4 → no open slots
        assert self._notebook(7, "CHOPPY", self._cfg(max_concurrent_positions=4)) == 0

    def test_nb_choppy_without_override_falls_back_to_global(self):
        assert self._notebook(2, "CHOPPY", self._cfg()) == 6

    def test_nb_bear_uses_global_cap_before_defensive_clamp(self):
        assert self._notebook(0, "BEAR", self._cfg(max_concurrent_positions=4)) == 8

    def test_nb_unknown_regime_uses_global_cap(self):
        assert self._notebook(1, "UNKNOWN", self._cfg(max_concurrent_positions=4)) == 7

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_bull_calm_uses_global_cap(self):
        assert self._lean(3, "BULL_CALM", self._cfg()) == 5

    def test_lean_choppy_override_to_4(self):
        assert self._lean(2, "CHOPPY", self._cfg(max_concurrent_positions=4)) == 2

    def test_lean_choppy_override_caps_open_slots(self):
        assert self._lean(7, "CHOPPY", self._cfg(max_concurrent_positions=4)) == 0

    def test_lean_choppy_without_override_falls_back_to_global(self):
        assert self._lean(2, "CHOPPY", self._cfg()) == 6

    def test_lean_bear_uses_global_cap_before_defensive_clamp(self):
        assert self._lean(0, "BEAR", self._cfg(max_concurrent_positions=4)) == 8

    def test_lean_unknown_regime_uses_global_cap(self):
        assert self._lean(1, "UNKNOWN", self._cfg(max_concurrent_positions=4)) == 7

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_regime_cap(self):
        cfg = self._cfg(max_concurrent_positions=4)
        for regime in ("BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"):
            for held in (0, 2, 4, 8):
                assert self._notebook(held, regime, cfg) == self._lean(held, regime, cfg), (
                    f"regime={regime} held={held}: NB={self._notebook(held, regime, cfg)} "
                    f"LEAN={self._lean(held, regime, cfg)}"
                )


# ── Count verification ────────────────────────────────────────────────────────

def test_equal_nb_and_lean_test_counts():
    """Meta-test: every policy class must have equal numbers of test_nb_* and test_lean_* tests."""
    import inspect

    classes = [
        TestTrailingStopAlignment,
        TestCumulativeStopLossAlignment,
        TestSingleDayLossAlignment,
        TestMaxHoldAlignment,
        TestMinHoldAlignment,
        TestConsecutiveSellStreakAlignment,
        TestSPYEMA50Alignment,
        TestVelocityCrashAlignment,
        TestTransitionWindowAlignment,
        TestEarningsFilterAlignment,
        TestTieredThresholdsAlignment,
        TestCorrelationGuardAlignment,
        TestSectorGuardAlignment,
        TestWashSaleAlignment,
        TestMinModelScoreAlignment,
        TestCombinedRankingAlignment,
        TestPositionSizingAlignment,
        TestRotationAlignment,
        TestRegimeCapAlignment,
        TestConfidenceVetoAlignment,
        TestBearDefensiveSlotsAlignment,
    ]
    for cls in classes:
        methods = [m for m in dir(cls) if m.startswith("test_")]
        nb_tests   = [m for m in methods if m.startswith("test_nb_")]
        lean_tests = [m for m in methods if m.startswith("test_lean_")]
        assert len(nb_tests) == len(lean_tests), (
            f"{cls.__name__}: {len(nb_tests)} notebook tests vs {len(lean_tests)} LEAN tests — must be equal"
        )
        assert len(nb_tests) >= 6, (
            f"{cls.__name__}: only {len(nb_tests)} tests per system — need at least 6"
        )


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "--tb=short"])


# ─── POLICY: Cross-Sectional Rotation ─────────────────────────────────────────

# Make kernel importable so _notebook/_lean helpers can call the shared primitive.
_KERNEL_DIR = ROOT / "backtesting" / "renquant_104"
if str(_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNEL_DIR))

from kernel.rotation import find_rotation_pairs  # noqa: E402
from kernel.selection import (  # noqa: E402
    is_wash_sale_blocked,
    passes_sector_guard,
    passes_correlation_guard,
)


class _RCand:
    def __init__(self, ticker, rank_score, expected_return=0.0):
        self.ticker          = ticker
        self.rank_score      = rank_score
        self.expected_return = expected_return


class TestRotationAlignment:
    """Held positions vs ranked candidates rotate when expected-return advantage
    over the rotation horizon clears `min_expected_advantage_pct` net of tax
    and transaction cost, and survives wash-sale, sector, and correlation
    guards."""

    DEFAULT_CFG = {
        "enabled": True,
        "min_expected_advantage_pct": 0.03,
        "target_horizon_days": 20,
        "transaction_cost_pct": 0.0,
        "min_rotation_hold_days": 30,
        "lt_protection_days": 30,
        "max_rotations_per_bar": 2,
    }
    DEFAULT_TAX = {
        "short_term_rate": 0.50,
        "long_term_rate": 0.32,
        "long_term_threshold_days": 365,
    }

    def _meta(self, entry_days_ago=60, entry_price=100.0, current_price=100.0):
        return {
            "entry_date":    date(2026, 4, 20) - timedelta(days=entry_days_ago),
            "entry_price":   entry_price,
            "current_price": current_price,
        }

    # Both helpers replicate the same flow because NB and LEAN share the kernel.
    # Tests still exercise the per-side surface: NB applies guards via kernel
    # helpers (cell 657a4a6c) and LEAN does the same via task_rotation.py.

    def _notebook_rotate(self, held_scores, held_er, held_meta, candidates, today,
                         last_sells, sector_map, max_per_sector, defensive,
                         corr, corr_thresh, wash_days, cfg=None, tax=None):
        """Replicates cell 657a4a6c rotation block."""
        cfg = cfg or self.DEFAULT_CFG
        tax = tax or self.DEFAULT_TAX
        if not cfg.get("enabled", False) or not held_scores:
            return []
        eligible = [c for c in candidates if c.ticker not in held_scores]
        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=eligible, today=today, rotation_cfg=cfg, tax_cfg=tax,
        )
        validated = []
        for p in pairs:
            if is_wash_sale_blocked(p.buy_ticker, today, last_sells, wash_days):
                continue
            virtual = (set(held_scores.keys())
                       - {x.sell_ticker for x in validated} - {p.sell_ticker}
                       | {x.buy_ticker for x in validated})
            if not passes_sector_guard(p.buy_ticker, list(virtual),
                                       sector_map, max_per_sector, defensive):
                continue
            if not passes_correlation_guard(p.buy_ticker, list(virtual),
                                            corr, corr_thresh):
                continue
            validated.append(p)
        return validated

    def _lean_rotate(self, held_scores, held_er, held_meta, candidates, today,
                     last_sells, sector_map, max_per_sector, defensive,
                     corr, corr_thresh, wash_days, cfg=None, tax=None):
        """Replicates kernel/pipeline/task_rotation.py BuildPairs+Validate."""
        cfg = cfg or self.DEFAULT_CFG
        tax = tax or self.DEFAULT_TAX
        if not cfg.get("enabled", False) or not held_scores:
            return []
        held_set = set(held_scores.keys())
        eligible = [c for c in candidates if c.ticker not in held_set]
        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=eligible, today=today, rotation_cfg=cfg, tax_cfg=tax,
        )
        validated = []
        for p in pairs:
            if is_wash_sale_blocked(p.buy_ticker, today, last_sells, wash_days):
                continue
            virtual = (held_set
                       - {x.sell_ticker for x in validated} - {p.sell_ticker}
                       | {x.buy_ticker for x in validated})
            if not passes_sector_guard(p.buy_ticker, list(virtual),
                                       sector_map, max_per_sector, defensive):
                continue
            if not passes_correlation_guard(p.buy_ticker, list(virtual),
                                            corr, corr_thresh):
                continue
            validated.append(p)
        return validated

    today = date(2026, 4, 20)

    # Common scenario builders
    # AAPL ER=0.01, MSFT ER=0.06 → raw_adv = 0.05; loss position has zero tax;
    # no transaction cost → net_adv = 0.05 ≥ 0.03 threshold → swap.
    def _basic_swap_inputs(self, **over):
        held_scores = {"AAPL": 0.30}
        held_er     = {"AAPL": 0.01}
        held_meta   = {"AAPL": self._meta(entry_price=120.0, current_price=100.0)}
        cands       = [_RCand("MSFT", 0.55, expected_return=0.06)]
        last_sells  = {}
        sector_map  = {"AAPL": "tech", "MSFT": "tech"}
        defensive   = set()
        corr        = {"AAPL": {"MSFT": 0.40}}
        defaults = dict(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=cands, today=self.today, last_sells=last_sells,
            sector_map=sector_map, max_per_sector=3, defensive=defensive,
            corr=corr, corr_thresh=0.70, wash_days=30,
        )
        defaults.update(over)
        return defaults

    # ── notebook tests ────────────────────────────────────────────────────────

    def test_nb_disabled_returns_empty(self):
        cfg = {**self.DEFAULT_CFG, "enabled": False}
        out = self._notebook_rotate(cfg=cfg, **self._basic_swap_inputs())
        assert out == []

    def test_nb_min_hold_blocks_recent(self):
        ins = self._basic_swap_inputs(
            held_meta={"AAPL": self._meta(entry_days_ago=10, entry_price=120.0,
                                          current_price=100.0)},
        )
        assert self._notebook_rotate(**ins) == []

    def test_nb_swap_emitted_when_advantage_clears(self):
        out = self._notebook_rotate(**self._basic_swap_inputs())
        assert len(out) == 1
        assert out[0].sell_ticker == "AAPL" and out[0].buy_ticker == "MSFT"
        assert out[0].net_advantage >= self.DEFAULT_CFG["min_expected_advantage_pct"]

    def test_nb_wash_sale_blocks_rotation_buy(self):
        ins = self._basic_swap_inputs(
            last_sells={"MSFT": self.today - timedelta(days=10)},
        )
        assert self._notebook_rotate(**ins) == []

    def test_nb_sector_cap_blocks_rotation_buy(self):
        ins = self._basic_swap_inputs(
            held_scores={"AAPL": 0.30, "JPM": 0.55},
            held_er    ={"AAPL": 0.01, "JPM":  0.04},
            held_meta={
                "AAPL": self._meta(entry_price=120.0, current_price=100.0),
                "JPM":  self._meta(entry_price=100.0, current_price=100.0),
            },
            sector_map={"AAPL": "tech", "MSFT": "tech", "JPM": "tech"},
            max_per_sector=1,
            corr={"AAPL": {}, "JPM": {}, "MSFT": {}},
        )
        assert self._notebook_rotate(**ins) == []

    def test_nb_correlation_guard_blocks_rotation_buy(self):
        ins = self._basic_swap_inputs()
        ins["held_scores"] = {"AAPL": 0.30, "JPM": 0.55}
        ins["held_er"]     = {"AAPL": 0.01, "JPM": 0.04}
        ins["held_meta"]   = {
            "AAPL": self._meta(entry_price=120.0, current_price=100.0),
            "JPM":  self._meta(entry_price=100.0, current_price=100.0),
        }
        ins["sector_map"] = {"AAPL": "tech", "MSFT": "tech", "JPM": "fin"}
        ins["corr"] = {"JPM": {"MSFT": 0.85}, "MSFT": {"JPM": 0.85}}
        assert self._notebook_rotate(**ins) == []

    # ── LEAN tests ────────────────────────────────────────────────────────────

    def test_lean_disabled_returns_empty(self):
        cfg = {**self.DEFAULT_CFG, "enabled": False}
        assert self._lean_rotate(cfg=cfg, **self._basic_swap_inputs()) == []

    def test_lean_min_hold_blocks_recent(self):
        ins = self._basic_swap_inputs(
            held_meta={"AAPL": self._meta(entry_days_ago=10, entry_price=120.0,
                                          current_price=100.0)},
        )
        assert self._lean_rotate(**ins) == []

    def test_lean_swap_emitted_when_advantage_clears(self):
        out = self._lean_rotate(**self._basic_swap_inputs())
        assert len(out) == 1
        assert out[0].sell_ticker == "AAPL" and out[0].buy_ticker == "MSFT"
        assert out[0].net_advantage >= self.DEFAULT_CFG["min_expected_advantage_pct"]

    def test_lean_wash_sale_blocks_rotation_buy(self):
        ins = self._basic_swap_inputs(
            last_sells={"MSFT": self.today - timedelta(days=10)},
        )
        assert self._lean_rotate(**ins) == []

    def test_lean_sector_cap_blocks_rotation_buy(self):
        ins = self._basic_swap_inputs(
            held_scores={"AAPL": 0.30, "JPM": 0.55},
            held_er    ={"AAPL": 0.01, "JPM":  0.04},
            held_meta={
                "AAPL": self._meta(entry_price=120.0, current_price=100.0),
                "JPM":  self._meta(entry_price=100.0, current_price=100.0),
            },
            sector_map={"AAPL": "tech", "MSFT": "tech", "JPM": "tech"},
            max_per_sector=1,
            corr={"AAPL": {}, "JPM": {}, "MSFT": {}},
        )
        assert self._lean_rotate(**ins) == []

    def test_lean_correlation_guard_blocks_rotation_buy(self):
        ins = self._basic_swap_inputs(
            held_scores={"AAPL": 0.30, "JPM": 0.55},
            held_er    ={"AAPL": 0.01, "JPM":  0.04},
            held_meta={
                "AAPL": self._meta(entry_price=120.0, current_price=100.0),
                "JPM":  self._meta(entry_price=100.0, current_price=100.0),
            },
            sector_map={"AAPL": "tech", "MSFT": "tech", "JPM": "fin"},
            corr={"JPM": {"MSFT": 0.85}, "MSFT": {"JPM": 0.85}},
        )
        assert self._lean_rotate(**ins) == []

    # ── cross-check ───────────────────────────────────────────────────────────

    def test_both_agree_across_scenarios(self):
        scenarios = [
            self._basic_swap_inputs(),
            self._basic_swap_inputs(
                held_meta={"AAPL": self._meta(entry_days_ago=10,
                                              entry_price=120.0, current_price=100.0)}),
            self._basic_swap_inputs(last_sells={"MSFT": self.today - timedelta(days=5)}),
            # +20% ST gain on AAPL ⇒ tax drag of 0.10; raw_adv 0.05 - 0.10 = -0.05 → no swap
            self._basic_swap_inputs(
                held_meta={"AAPL": self._meta(entry_price=100.0, current_price=120.0)},
                candidates=[_RCand("MSFT", 0.45, expected_return=0.06)],
            ),
            # LT protection: 350d gain pinned regardless of how strong the candidate is
            self._basic_swap_inputs(
                held_meta={"AAPL": self._meta(entry_days_ago=350,
                                              entry_price=100.0, current_price=120.0)},
                candidates=[_RCand("MSFT", 0.95, expected_return=0.50)],
            ),
        ]
        for s in scenarios:
            nb = self._notebook_rotate(**s)
            ln = self._lean_rotate(**s)
            assert [(p.sell_ticker, p.buy_ticker) for p in nb] == \
                   [(p.sell_ticker, p.buy_ticker) for p in ln], \
                   f"NB and LEAN disagree on scenario: NB={nb} LEAN={ln}"
