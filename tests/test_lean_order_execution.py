"""Unit tests for the extracted LEAN order-ticket execution helpers.

Pins adapters/lean_order.py (lean.py decomposition slice 3) at the module
boundary — the LEAN counterpart to runner_execmath.broker_order_execution.
These pure functions read a QCAlgorithm OrderTicket (or a list of tickets)
into the uniform (filled, qty, avg_price, status) tuple, FAIL-CLOSED so an
unconfirmed order can never mutate LEAN state/tax as if it filled.

REGRESSION GUARD: the helpers (and the _positive_finite_price fill-price
validator that moved with them) must remain importable from BOTH
adapters.lean_order (canonical) and adapters.lean (back-compat re-export) as
the SAME object — make_context/commit and the price-resolution helpers call
them by the re-exported name.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters import lean as _lean  # noqa: E402
from adapters import lean_order as _lo  # noqa: E402


class TestReexportIdentity:
    def test_same_objects(self):
        for n in ("_positive_finite_price", "_lean_ticket_status_text",
                  "_lean_ticket_float", "_lean_order_execution"):
            assert getattr(_lean, n) is getattr(_lo, n), n


class TestPositiveFinitePrice:
    def test_positive_finite_passes(self):
        assert _lo._positive_finite_price(12.5) == 12.5
        assert _lo._positive_finite_price("3.0") == 3.0

    def test_zero_negative_nonfinite_unparseable_are_none(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), None, "x"):
            assert _lo._positive_finite_price(bad) is None, bad


class TestTicketStatusText:
    def test_lowercased(self):
        assert _lo._lean_ticket_status_text(NS(Status="Filled")) == "filled"
        assert _lo._lean_ticket_status_text(NS(Status="PartiallyFilledThenCanceled")) \
            == "partiallyfilledthencanceled"

    def test_missing_status_is_empty(self):
        assert _lo._lean_ticket_status_text(NS()) == ""
        assert _lo._lean_ticket_status_text(NS(Status=None)) == ""


class TestTicketFloat:
    def test_first_finite_name_wins(self):
        t = NS(QuantityFilled=float("nan"), AbsoluteQuantityFilled=5.0)
        assert _lo._lean_ticket_float(t, "QuantityFilled", "AbsoluteQuantityFilled") == 5.0

    def test_callable_attribute_invoked(self):
        t = NS(FillPrice=lambda: 42.0)
        assert _lo._lean_ticket_float(t, "FillPrice") == 42.0

    def test_none_when_nothing_finite(self):
        t = NS(a=float("inf"), b=None)
        assert _lo._lean_ticket_float(t, "a", "b", "missing") is None


class TestOrderExecution:
    def test_filled_ticket_reports_fill(self):
        t = NS(Status="Filled", QuantityFilled=10.0, AverageFillPrice=100.0)
        assert _lo._lean_order_execution(t, requested_qty=10, fallback_price=99.0) \
            == (True, 10.0, 100.0, "filled")

    def test_missing_ticket_fails_closed(self):
        assert _lo._lean_order_execution(None, requested_qty=5, fallback_price=50.0) \
            == (False, 0.0, 50.0, "missing_order_ticket")

    def test_rejected_status_not_filled(self):
        for s in ("Rejected", "Canceled", "Invalid", "Error 500"):
            ok, qty, _, status = _lo._lean_order_execution(
                NS(Status=s), requested_qty=5, fallback_price=50.0)
            assert ok is False and qty == 0.0, s

    def test_filled_status_without_qty_uses_requested(self):
        # status says filled but no QuantityFilled attr → trust requested qty.
        t = NS(Status="Filled", AverageFillPrice=20.0)
        ok, qty, px, status = _lo._lean_order_execution(
            t, requested_qty=7, fallback_price=99.0)
        assert ok is True and qty == 7.0 and px == 20.0

    def test_price_falls_back_when_missing(self):
        t = NS(Status="Filled", QuantityFilled=3.0)  # no fill price
        ok, qty, px, _ = _lo._lean_order_execution(
            t, requested_qty=3, fallback_price=55.0)
        assert ok is True and qty == 3.0 and px == 55.0

    def test_unknown_status_no_qty_not_filled(self):
        t = NS(Status="Submitted")
        ok, qty, _, status = _lo._lean_order_execution(
            t, requested_qty=4, fallback_price=10.0)
        assert ok is False and qty == 0.0 and status == "submitted"

    def test_list_of_tickets_aggregates_quantity_weighted_price(self):
        # two partial fills: 4@10 and 6@20 → 10 shares, vwap 16.
        t1 = NS(Status="Filled", QuantityFilled=4.0, AverageFillPrice=10.0)
        t2 = NS(Status="Filled", QuantityFilled=6.0, AverageFillPrice=20.0)
        ok, qty, px, status = _lo._lean_order_execution(
            [t1, t2], requested_qty=10, fallback_price=99.0)
        assert ok is True and qty == 10.0
        assert px == (4 * 10 + 6 * 20) / 10  # == 16.0

    def test_list_all_rejected_not_filled(self):
        t1 = NS(Status="Rejected")
        t2 = NS(Status="Canceled")
        ok, qty, _, _ = _lo._lean_order_execution(
            [t1, t2], requested_qty=10, fallback_price=99.0)
        assert ok is False and qty == 0.0
