"""Invariant tests for the extracted sim NAV mark-to-market (sim_nav).

Pins adapters/sim_nav.portfolio_value (sim.py decomposition). NAV feeds the
equity curve / vol / APY, and the function carries three hard-won correctness
fixes — this locks them:
  * SA-1: a non-finite price contributes 0, never NaN-poisons the total.
  * Bug #C: T+N pending-settlement proceeds are part of NAV.
  * Bug 25: a missing price falls back to the last close ON OR BEFORE
            today_ts (no lookahead), not the last historical bar.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.sim_nav import portfolio_value  # noqa: E402


def _q(total):
    return NS(pending_total=lambda: total)


class TestNavIdentity:
    def test_cash_plus_pending_plus_positions(self):
        nav = portfolio_value(
            {"AAPL": 10.0, "MSFT": 20.0}, cash=1000.0, t2_queue=_q(250.0),
            pos_shares={"AAPL": 5, "MSFT": 3}, ohlcv={})
        assert nav == 1000.0 + 250.0 + 5 * 10.0 + 3 * 20.0  # 1360

    def test_no_positions_is_cash_plus_pending(self):
        assert portfolio_value({}, cash=500.0, t2_queue=_q(100.0),
                               pos_shares={}, ohlcv={}) == 600.0

    def test_no_queue_is_cash_plus_positions(self):
        assert portfolio_value({"X": 2.0}, cash=100.0, t2_queue=None,
                               pos_shares={"X": 10}, ohlcv={}) == 120.0


class TestSA1NonFiniteSkip:
    def test_nan_price_contributes_zero_not_nan(self):
        nav = portfolio_value(
            {"AAPL": float("nan"), "MSFT": 20.0}, cash=1000.0, t2_queue=None,
            pos_shares={"AAPL": 5, "MSFT": 3}, ohlcv={})
        # AAPL skipped (no ohlcv fallback) → 1000 + 0 + 60
        assert nav == 1060.0

    def test_inf_price_skipped(self):
        nav = portfolio_value({"X": float("inf")}, cash=100.0, t2_queue=None,
                              pos_shares={"X": 5}, ohlcv={})
        assert nav == 100.0  # not inf

    def test_non_finite_pending_skipped(self):
        nav = portfolio_value({}, cash=100.0, t2_queue=_q(float("nan")),
                              pos_shares={}, ohlcv={})
        assert nav == 100.0


class TestBug25LookaheadSafeFallback:
    def _frame(self):
        idx = pd.to_datetime(["2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13"])
        return pd.DataFrame({"close": [10.0, 11.0, 12.0, 99.0]}, index=idx)

    def test_missing_price_uses_close_on_or_before_today(self):
        # today = 2026-06-12 → fallback must be 12.0 (that day's close),
        # NOT 99.0 (the last historical bar = future data).
        nav = portfolio_value(
            {}, today_ts=pd.Timestamp("2026-06-12"), cash=0.0, t2_queue=None,
            pos_shares={"X": 1}, ohlcv={"X": self._frame()})
        assert nav == 12.0

    def test_no_today_ts_uses_last_bar(self):
        # without a truncation hint the caller owns lookahead safety → last bar
        nav = portfolio_value(
            {}, cash=0.0, t2_queue=None, pos_shares={"X": 1},
            ohlcv={"X": self._frame()})
        assert nav == 99.0

    def test_today_before_all_bars_skips_position(self):
        nav = portfolio_value(
            {}, today_ts=pd.Timestamp("2026-06-01"), cash=50.0, t2_queue=None,
            pos_shares={"X": 1}, ohlcv={"X": self._frame()})
        assert nav == 50.0  # no close on/before → skipped

    def test_present_price_preferred_over_fallback(self):
        nav = portfolio_value(
            {"X": 5.0}, today_ts=pd.Timestamp("2026-06-12"), cash=0.0,
            t2_queue=None, pos_shares={"X": 1}, ohlcv={"X": self._frame()})
        assert nav == 5.0  # uses the live price, not the ohlcv fallback


class TestDelegateParity:
    def test_delegate_matches_pure_function(self):
        from adapters.sim import SimAdapter

        a = SimAdapter.__new__(SimAdapter)
        a._cash = 1000.0
        a._t2_queue = _q(200.0)
        a._pos_shares = {"AAPL": 4}
        a._ohlcv = {}
        prices = {"AAPL": 25.0}
        assert a._portfolio_value(prices) == portfolio_value(
            prices, cash=a._cash, t2_queue=a._t2_queue,
            pos_shares=a._pos_shares, ohlcv=a._ohlcv)
        assert a._portfolio_value(prices) == 1000.0 + 200.0 + 4 * 25.0  # 1300
