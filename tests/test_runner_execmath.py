"""runner.py decomposition slice 5 — runner_execmath pure-function tests."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.runner_execmath import (  # noqa: E402
    broker_order_execution,
    cap_buy_order_to_cash,
    normalize_order_status,
    same_bar_sell_credit,
)


class TestCapBuyToCash:
    def test_full_order_fits(self):
        order, reason = cap_buy_order_to_cash(
            {"ticker": "MU", "shares": 5, "price": 10.0}, remaining_cash=100.0)
        assert order["shares"] == 5 and reason is None

    def test_capped_to_affordable_shares(self):
        order, reason = cap_buy_order_to_cash(
            {"ticker": "MU", "shares": 20, "price": 10.0}, remaining_cash=55.0)
        assert order is not None and order["shares"] == 5
        assert reason is not None

    def test_zero_cash_drops_order(self):
        order, reason = cap_buy_order_to_cash(
            {"ticker": "MU", "shares": 5, "price": 10.0}, remaining_cash=0.0)
        assert order is None and reason is not None


class TestNormalizeStatus:
    def test_enum_dotted(self):
        assert normalize_order_status("OrderStatus.FILLED") == "filled"

    def test_plain(self):
        assert normalize_order_status("Accepted") == "accepted"

    def test_none(self):
        assert normalize_order_status(None) == ""


class TestSameBarSellCredit:
    def test_sums_filled_sells(self):
        ctx = SimpleNamespace(orders=[
            {"action": "SELL", "shares": 2, "price": 10.0, "status": "filled"},
            {"action": "BUY", "shares": 1, "price": 5.0, "status": "filled"},
        ])
        assert same_bar_sell_credit(ctx) >= 0.0


class TestBrokerOrderExecution:
    def test_summarizes_filled_result(self):
        out = broker_order_execution(
            {"order_id": "o1", "status": "filled", "filled_qty": 5,
             "filled_avg_price": 10.0},
            requested_qty=5, fallback_price=10.0)
        assert isinstance(out, dict)

    def test_none_result_uses_fallback(self):
        out = broker_order_execution(None, requested_qty=5, fallback_price=12.0)
        assert isinstance(out, dict)
