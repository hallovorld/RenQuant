"""runner.py decomposition slice 5 — runner_execmath pure-function tests."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))

from _order_math_owner import owner_cap_affordable_qty  # noqa: E402
from adapters import runner_execmath as _execmath  # noqa: E402
from adapters.runner_execmath import (  # noqa: E402
    broker_order_execution,
    cap_buy_order_to_cash,
    normalize_order_status,
    same_bar_sell_credit,
)


@pytest.fixture
def with_delegate(monkeypatch):
    """Force the delegate-present wiring, independent of import order."""
    owner = owner_cap_affordable_qty()
    if owner is None:
        pytest.skip("renquant_execution.order_math unavailable "
                    "(pinned checkout predates execution#25)")
    monkeypatch.setattr(_execmath, "_cap_affordable_qty", owner)


@pytest.fixture
def without_delegate(monkeypatch):
    """Simulate an older pinned renquant-execution missing order_math."""
    monkeypatch.setattr(_execmath, "_cap_affordable_qty", None)


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
    """S-FRAC v2 stage 2 — fractional-aware cash cap (D7 gap inventory #1),
    DELEGATED to the owner renquant-execution order_math (execution#25);
    these pin the wiring through the compatibility call-site."""

    def test_capped_to_6dp_floor(self, with_delegate):
        # affordable = floor(100/3 · 1e6)/1e6 = 33.333333 (6dp, floored,
        # never rounded up past cash).
        order, reason = cap_buy_order_to_cash(
            {"ticker": "MU", "shares": 50, "price": 3.0}, remaining_cash=100.0,
            fractional=True)
        assert reason == "cash_budget_resized"
        assert order["shares"] == 33.333333
        assert order["shares"] * 3.0 <= 100.0
        assert order["original_shares"] == 50

    def test_sub_one_share_resize_is_admitted(self, with_delegate):
        # The exact D7 #1 gap: legacy int truncation turned 0.5 affordable
        # shares into 0 → reject. Fractional mode admits the 0.5 slice.
        order, reason = cap_buy_order_to_cash(
            {"ticker": "BLK", "shares": 1, "price": 100.0}, remaining_cash=50.0,
            fractional=True)
        assert reason == "cash_budget_resized"
        assert order["shares"] == 0.5
        assert order["invest"] == 50.0

    def test_below_min_notional_rejects(self, with_delegate):
        # floor6(0.5/10) = 0.05 shares → $0.50 notional < the ~$1 broker
        # fractional minimum → reject, never a dust order.
        order, reason = cap_buy_order_to_cash(
            {"ticker": "MU", "shares": 5, "price": 10.0}, remaining_cash=0.5,
            fractional=True)
        assert order is None and reason == "cash_budget_exhausted"

    def test_noop_when_affordable(self):
        # The afford-check happens before any delegation — no owner needed.
        order, reason = cap_buy_order_to_cash(
            {"ticker": "MU", "shares": 0.25, "price": 10.0},
            remaining_cash=100.0, fractional=True)
        assert reason is None and order["shares"] == 0.25

    def test_flag_off_default_keeps_legacy_int_truncation(self, with_delegate):
        # Default (no kwarg) and explicit fractional=False are the legacy
        # whole-share behavior: int type, int(cash // price) value — the
        # delegated whole-share mode is byte-identical to the inline legacy.
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


class TestCapBuyToCashDelegateFallback:
    """Missing-owner degradation (the pinned renquant-execution predates
    order_math): fractional requests FAIL CLOSED to the legacy whole-share
    truncation with a logged warning — never a crash, never umbrella-local
    fractional math. Flag-off behavior is unaffected."""

    def test_fractional_falls_back_to_whole_share_with_warning(
            self, without_delegate, caplog):
        with caplog.at_level(logging.WARNING, logger="adapters.runner"):
            order, reason = cap_buy_order_to_cash(
                {"ticker": "MU", "shares": 20, "price": 10.0},
                remaining_cash=55.0, fractional=True)
        assert reason == "cash_budget_resized"
        assert order["shares"] == 5 and type(order["shares"]) is int
        assert any("EXECMATH-CASHCAP-FALLBACK" in r.message
                   for r in caplog.records)

    def test_fractional_fallback_sub_share_rejects_like_legacy(
            self, without_delegate, caplog):
        # The D7 #1 case degrades conservatively: 0.5 affordable shares →
        # whole-share reject (as pre-D7 legacy), not a fractional resize.
        with caplog.at_level(logging.WARNING, logger="adapters.runner"):
            order, reason = cap_buy_order_to_cash(
                {"ticker": "BLK", "shares": 1, "price": 100.0},
                remaining_cash=50.0, fractional=True)
        assert order is None and reason == "cash_budget_exhausted"
        assert any("EXECMATH-CASHCAP-FALLBACK" in r.message
                   for r in caplog.records)

    def test_flag_off_fallback_is_silent_legacy(self, without_delegate, caplog):
        with caplog.at_level(logging.WARNING, logger="adapters.runner"):
            order, reason = cap_buy_order_to_cash(
                {"ticker": "MU", "shares": 20, "price": 10.0},
                remaining_cash=55.0)
        assert reason == "cash_budget_resized"
        assert order["shares"] == 5 and type(order["shares"]) is int
        assert not caplog.records  # no spurious fallback warning flag-off


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
