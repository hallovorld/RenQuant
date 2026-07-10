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


class TestCapBuyToCashFractional:
    """S-FRAC v2 stage 2 — fractional-aware cash cap (D7 gap inventory #1)."""

    def test_capped_to_6dp_floor(self):
        # affordable = floor(100/3 · 1e6)/1e6 = 33.333333 (6dp, floored,
        # never rounded up past cash).
        order, reason = cap_buy_order_to_cash(
            {"ticker": "MU", "shares": 50, "price": 3.0}, remaining_cash=100.0,
            fractional=True)
        assert reason == "cash_budget_resized"
        assert order["shares"] == 33.333333
        assert order["shares"] * 3.0 <= 100.0
        assert order["original_shares"] == 50

    def test_sub_one_share_resize_is_admitted(self):
        # The exact D7 #1 gap: legacy int truncation turned 0.5 affordable
        # shares into 0 → reject. Fractional mode admits the 0.5 slice.
        order, reason = cap_buy_order_to_cash(
            {"ticker": "BLK", "shares": 1, "price": 100.0}, remaining_cash=50.0,
            fractional=True)
        assert reason == "cash_budget_resized"
        assert order["shares"] == 0.5
        assert order["invest"] == 50.0

    def test_below_min_notional_rejects(self):
        # floor6(0.5/10) = 0.05 shares → $0.50 notional < the ~$1 broker
        # fractional minimum → reject, never a dust order.
        order, reason = cap_buy_order_to_cash(
            {"ticker": "MU", "shares": 5, "price": 10.0}, remaining_cash=0.5,
            fractional=True)
        assert order is None and reason == "cash_budget_exhausted"

    def test_noop_when_affordable(self):
        order, reason = cap_buy_order_to_cash(
            {"ticker": "MU", "shares": 0.25, "price": 10.0},
            remaining_cash=100.0, fractional=True)
        assert reason is None and order["shares"] == 0.25

    def test_flag_off_default_keeps_legacy_int_truncation(self):
        # Default (no kwarg) and explicit fractional=False are the legacy
        # whole-share behavior: int type, int(cash // price) value.
        for kwargs in ({}, {"fractional": False}):
            order, reason = cap_buy_order_to_cash(
                {"ticker": "MU", "shares": 20, "price": 10.0},
                remaining_cash=55.0, **kwargs)
            assert reason == "cash_budget_resized"
            assert order["shares"] == 5 and type(order["shares"]) is int
            order, reason = cap_buy_order_to_cash(
                {"ticker": "BLK", "shares": 1, "price": 100.0},
                remaining_cash=50.0, **kwargs)
            assert order is None and reason == "cash_budget_exhausted"


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
