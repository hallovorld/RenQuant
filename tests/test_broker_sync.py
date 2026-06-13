"""runner.py decomposition slice 2 — broker_sync contract tests."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.broker_sync import (  # noqa: E402
    gc_recent_sell_orders,
    override_no_trade_streak_from_broker,
)


class _Ctx:
    today = dt.date(2026, 6, 12)


class _Broker:
    def __init__(self, fills):
        self._fills = fills

    def get_filled_orders(self, after):
        return self._fills


class TestNoTradeStreakOverride:

    def test_broker_truth_overrides_counter(self):
        state = {"monitor_state": {"no_trade_streak": 32}}
        fills = [{"filled_at": "2026-06-10T15:00:00Z"}]
        override_no_trade_streak_from_broker(_Broker(fills), state, _Ctx())
        mon = state["monitor_state"]
        # NYSE days in (fill, today] inclusive of today: 06-11 + 06-12 = 2
        # (the original loop counts today — verbatim-preserved behavior).
        assert mon["no_trade_streak"] == 2
        assert mon["no_trade_streak_source"] == "broker_filled_orders"
        assert mon["last_fill_date"] == "2026-06-10"

    def test_no_fills_caps_at_lookback(self):
        state = {}
        override_no_trade_streak_from_broker(_Broker([]), state, _Ctx())
        assert state["monitor_state"]["no_trade_streak"] == 120
        assert state["monitor_state"]["last_fill_date"] is None

    def test_runner_emission_fields_untouched(self):
        # codex PR #84 contract: last_activity_date / first_trade_date are
        # runner-emission semantics and must NOT be clobbered.
        state = {"monitor_state": {"last_activity_date": "2026-06-01",
                                   "first_trade_date": "2026-04-23"}}
        override_no_trade_streak_from_broker(
            _Broker([{"filled_at": "2026-06-10T15:00:00Z"}]), state, _Ctx())
        assert state["monitor_state"]["last_activity_date"] == "2026-06-01"
        assert state["monitor_state"]["first_trade_date"] == "2026-04-23"

    def test_broker_failure_leaves_state(self):
        class _Boom:
            def get_filled_orders(self, after):
                raise RuntimeError("down")

        state = {"monitor_state": {"no_trade_streak": 5}}
        override_no_trade_streak_from_broker(_Boom(), state, _Ctx())
        assert state["monitor_state"]["no_trade_streak"] == 5

    def test_broker_without_api_noop(self):
        state = {"monitor_state": {"no_trade_streak": 5}}
        override_no_trade_streak_from_broker(object(), state, _Ctx())
        assert state["monitor_state"]["no_trade_streak"] == 5


class TestGcRecentSellOrders:

    def test_prunes_older_than_six_days(self):
        bar = dt.date(2026, 6, 12)
        orders = {"old": {"submitted_at": "2026-06-01"},
                  "new": {"submitted_at": "2026-06-10T09:30:00"}}
        kept = gc_recent_sell_orders(orders, bar)
        assert set(kept) == {"new"}

    def test_unparseable_stamp_kept_fail_open(self):
        kept = gc_recent_sell_orders({"x": {"submitted_at": "garbage"}},
                                     dt.date(2026, 6, 12))
        assert "x" in kept

    def test_empty_and_none(self):
        assert gc_recent_sell_orders({}, dt.date(2026, 6, 12)) == {}
        assert gc_recent_sell_orders(None, dt.date(2026, 6, 12)) == {}
