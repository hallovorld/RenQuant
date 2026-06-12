"""G2 agent breaker tests — adapter-level order admission caps.

Design: doc/research/2026-06-12-engineering-architecture-deep-plan.md
(renquant-orchestrator) §0 Week-0, §III.4 "Disaster guards"; prototype
property proofs: scripts/engineering/agent_breaker_prototype.py (PR #112).

Invariants pinned here:
- TRADING_OFF file dominates everything (checked before caps, every call)
- daily order-count cap binds at exactly max_orders
- daily notional cap binds; notional=None consumes a slot but not notional
- day roll resets both counters
- a tripped breaker does NOT consume a slot (rejection is not an order)
- AlpacaBroker.place_order admits through the breaker BEFORE any API call
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from live.agent_breaker import AgentBreaker, BreakerTripped

D1 = dt.date(2026, 6, 12)
D2 = dt.date(2026, 6, 13)


def _breaker(tmp_path, **kw):
    kw.setdefault("off_flag", tmp_path / "TRADING_OFF")
    return AgentBreaker(**kw)


class TestTradingOffFlag:

    def test_flag_blocks_all_orders(self, tmp_path):
        b = _breaker(tmp_path)
        (tmp_path / "TRADING_OFF").touch()
        with pytest.raises(BreakerTripped, match="TRADING_OFF"):
            b.admit(symbol="AAPL", notional=100.0, today=D1)

    def test_flag_dominates_even_with_zero_usage(self, tmp_path):
        # Flag is checked before caps — fresh breaker with full headroom
        # still refuses.
        b = _breaker(tmp_path, max_orders_per_day=1000)
        (tmp_path / "TRADING_OFF").touch()
        with pytest.raises(BreakerTripped):
            b.admit(symbol="MSFT", notional=None, today=D1)

    def test_flag_removal_reenables(self, tmp_path):
        b = _breaker(tmp_path)
        flag = tmp_path / "TRADING_OFF"
        flag.touch()
        with pytest.raises(BreakerTripped):
            b.admit(symbol="AAPL", notional=10.0, today=D1)
        flag.unlink()
        b.admit(symbol="AAPL", notional=10.0, today=D1)  # must not raise


class TestOrderCountCap:

    def test_cap_binds_exactly(self, tmp_path):
        b = _breaker(tmp_path, max_orders_per_day=3)
        for _ in range(3):
            b.admit(symbol="AAPL", notional=None, today=D1)
        with pytest.raises(BreakerTripped, match="order cap"):
            b.admit(symbol="AAPL", notional=None, today=D1)

    def test_trip_does_not_consume_slot(self, tmp_path):
        # After a notional trip, count-only orders must still pass: the
        # rejected order was never admitted.
        b = _breaker(tmp_path, max_orders_per_day=2, max_notional_per_day=50.0)
        with pytest.raises(BreakerTripped):
            b.admit(symbol="AAPL", notional=100.0, today=D1)
        b.admit(symbol="AAPL", notional=40.0, today=D1)
        b.admit(symbol="AAPL", notional=None, today=D1)
        assert b._orders == 2

    def test_day_roll_resets(self, tmp_path):
        b = _breaker(tmp_path, max_orders_per_day=1)
        b.admit(symbol="AAPL", notional=None, today=D1)
        with pytest.raises(BreakerTripped):
            b.admit(symbol="AAPL", notional=None, today=D1)
        b.admit(symbol="AAPL", notional=None, today=D2)  # new day: must pass


class TestNotionalCap:

    def test_cap_binds(self, tmp_path):
        b = _breaker(tmp_path, max_notional_per_day=1000.0)
        b.admit(symbol="AAPL", notional=600.0, today=D1)
        with pytest.raises(BreakerTripped, match="notional cap"):
            b.admit(symbol="MSFT", notional=500.0, today=D1)

    def test_exact_boundary_admitted(self, tmp_path):
        b = _breaker(tmp_path, max_notional_per_day=1000.0)
        b.admit(symbol="AAPL", notional=1000.0, today=D1)  # == cap: allowed
        with pytest.raises(BreakerTripped):
            b.admit(symbol="AAPL", notional=0.01, today=D1)

    def test_none_notional_consumes_slot_only(self, tmp_path):
        b = _breaker(tmp_path, max_orders_per_day=5, max_notional_per_day=100.0)
        b.admit(symbol="AAPL", notional=None, today=D1)
        assert b._orders == 1
        assert b._notional == 0.0
        b.admit(symbol="AAPL", notional=100.0, today=D1)  # full cap still free

    def test_negative_notional_counted_abs(self, tmp_path):
        # Sells/shorts consume notional headroom too — the cap bounds
        # gross activity, not net direction.
        b = _breaker(tmp_path, max_notional_per_day=100.0)
        b.admit(symbol="AAPL", notional=-80.0, today=D1)
        with pytest.raises(BreakerTripped):
            b.admit(symbol="AAPL", notional=30.0, today=D1)


class TestAlpacaBrokerWiring:
    """place_order must admit through the breaker BEFORE touching the
    trading client, and BreakerTripped must propagate (not be swallowed)."""

    def _broker(self, tmp_path):
        from live.alpaca_broker import AlpacaBroker

        b = AlpacaBroker(api_key="k", secret_key="s", paper=True)

        class _ExplodingClient:
            def __getattr__(self, name):
                raise AssertionError(
                    f"trading client touched ({name}) — breaker must trip first")

        b._trading_client = _ExplodingClient()
        b._g2_breaker = AgentBreaker(off_flag=tmp_path / "TRADING_OFF")
        # Pin price so notional accounting is deterministic and no data
        # client is constructed.
        b.get_last_price = lambda symbol: 10.0
        return b

    def test_trading_off_blocks_before_api(self, tmp_path):
        b = self._broker(tmp_path)
        (tmp_path / "TRADING_OFF").touch()
        with pytest.raises(BreakerTripped, match="TRADING_OFF"):
            b.place_order("AAPL", "BUY", 1)

    def test_order_cap_blocks_before_api(self, tmp_path):
        b = self._broker(tmp_path)
        b._g2_breaker._day = dt.date.today()
        b._g2_breaker._orders = b._g2_breaker.max_orders
        with pytest.raises(BreakerTripped, match="order cap"):
            b.place_order("AAPL", "BUY", 1)

    def test_notional_cap_uses_last_price(self, tmp_path):
        b = self._broker(tmp_path)  # price pinned at $10
        b._g2_breaker.max_notional = 50.0
        with pytest.raises(BreakerTripped, match="notional cap"):
            b.place_order("AAPL", "BUY", 6)  # 6 × $10 = $60 > $50

    def test_price_failure_degrades_to_count_only(self, tmp_path):
        # Data outage must not block trading entirely — slot is consumed,
        # notional is not, and admission proceeds past the breaker (the
        # exploding client then proves we reached the API layer).
        b = self._broker(tmp_path)

        def _boom(symbol):
            raise RuntimeError("data feed down")

        b.get_last_price = _boom
        # place_order wraps the account-check probe in RuntimeError; either
        # way, reaching the trading client proves the breaker admitted.
        with pytest.raises(RuntimeError, match="account check failed"):
            b.place_order("AAPL", "BUY", 1)
        assert b._g2_breaker._orders == 1
        assert b._g2_breaker._notional == 0.0
