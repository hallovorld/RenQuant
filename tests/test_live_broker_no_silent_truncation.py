"""S-FRAC chain step 2: the LIVE broker never silently truncates a quantity.

Pre-change, ``live/alpaca_broker.py`` built every order with
``qty=int(quantity)`` (``place_order`` and ``place_stop_order``): a
fractional intent was SILENTLY truncated — 0.435578 became 0 shares, 7.5
became 7, and the stop for a 7.5-share position was armed for 7. This file
carries the burden for the replacement discipline (ported semantics of
renquant-execution pin 91c7bf88 ``broker.py::is_whole_share`` /
``validate_fractional_order`` and ``alpaca_broker.py::place_order``):

  1. Whole-share (eps-integral) quantities: the SDK payload, the returned
     dict, the log line and the client I/O are BYTE-IDENTICAL to the legacy
     path — pinned against a verbatim legacy oracle
     (``TestIntegralPathByteIdentical``).
  2. Fractional quantities are never truncated: refused with
     ``FractionalOrderRefused`` (no submit, one WARNING, no G2 slot, no
     account read) unless the asset is CONFIRMED fractionable and the order
     is MARKET + DAY, in which case the exact qty (9dp grid, rounded DOWN)
     is submitted (``TestFractionalRefusals``, ``TestFractionalSubmission``).
  3. Broker-side stops refuse every fractional qty (``TestStopPath``).
  4. The runner absorbs the refusal as a no-submit outcome on its existing
     surfaces — BUY → ``ctx.orders_skipped``, SELL → ``ctx.exits_failed``,
     Z9 → warning + no stop — and the run continues
     (``TestRunnerMapping``: end-to-end through the real
     ``RunnerAdapter.commit`` where the strategy deps are importable, plus
     a static pin of the handling sites that runs everywhere).

Everything is offline: no credentials, no network, no ``connect()``. The
broker is built with ``AlpacaBroker.__new__`` (the pattern of
``test_live_broker_fractional_contract.py``) and the alpaca SDK request /
enum modules are replaced with recorders so the payload is captured as the
exact kwargs the broker passes (the CI job for this file installs only
pytest — no alpaca-py).
"""
from __future__ import annotations

import logging
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = REPO_ROOT / "backtesting" / "renquant_104"
for _p in (str(REPO_ROOT), str(_STRATEGY), str(REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live import broker as live_broker  # noqa: E402
from live.agent_breaker import AgentBreaker  # noqa: E402
from live.alpaca_broker import AlpacaBroker  # noqa: E402
from live.broker import (  # noqa: E402
    BELOW_MIN_NOTIONAL_STATUS,
    FRACTIONABLE_LOOKUP_FAILED_STATUS,
    INVALID_FRACTIONAL_ORDER_STATUS,
    NON_FRACTIONABLE_STATUS,
    FractionalOrderRefused,
    is_no_submit_status,
    is_whole_share,
    snap_qty_to_broker_grid,
)

FRACTIONAL_QTY = 0.435578  # the design's E2E audit quantity
SYMBOL = "AAPL"
PRICE = 100.0


# ── Fake alpaca SDK: records the request kwargs verbatim ────────────────────

class _Request:
    def __init__(self, kind: str, **kwargs):
        self.kind = kind
        self.kwargs = dict(kwargs)


@pytest.fixture
def sdk(monkeypatch):
    """Replace ``alpaca.trading.requests`` / ``.enums`` with recorders.

    The broker imports them lazily inside the order methods, so swapping
    the ``sys.modules`` entries is sufficient and is restored afterwards.
    """
    requests = types.ModuleType("alpaca.trading.requests")
    requests.MarketOrderRequest = lambda **kw: _Request("market", **kw)
    requests.StopOrderRequest = lambda **kw: _Request("stop", **kw)
    enums = types.ModuleType("alpaca.trading.enums")
    enums.OrderSide = SimpleNamespace(BUY="BUY", SELL="SELL")
    enums.TimeInForce = SimpleNamespace(DAY="DAY", GTC="GTC")
    alpaca = types.ModuleType("alpaca")
    trading = types.ModuleType("alpaca.trading")
    alpaca.trading = trading
    trading.requests = requests
    trading.enums = enums
    for name, mod in (
        ("alpaca", alpaca), ("alpaca.trading", trading),
        ("alpaca.trading.requests", requests), ("alpaca.trading.enums", enums),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return SimpleNamespace(requests=requests, enums=enums)


class _Client:
    """TradingClient stand-in: account read, asset lookup, order submit.

    ``fractionable`` is a bool verdict or an Exception to raise from
    ``get_asset``. Every call is recorded so I/O can be asserted exactly.
    """

    def __init__(self, fractionable=None, status="ACTIVE"):
        self.fractionable = fractionable
        self.status = status
        self.submitted: list[_Request] = []
        self.account_calls = 0
        self.asset_calls: list[str] = []

    def get_account(self):
        self.account_calls += 1
        return SimpleNamespace(status=self.status)

    def get_asset(self, symbol):
        self.asset_calls.append(symbol)
        if isinstance(self.fractionable, Exception):
            raise self.fractionable
        return SimpleNamespace(fractionable=self.fractionable)

    def submit_order(self, request):
        self.submitted.append(request)
        return SimpleNamespace(
            id="ord-1", status="accepted", filled_qty=None,
            filled_avg_price=None, submitted_at="2026-08-28T20:55:00Z",
            filled_at=None,
        )


def _broker(client, tmp_path, price=PRICE) -> AlpacaBroker:
    """Offline AlpacaBroker: no __init__, no credentials, no connect()."""
    b = AlpacaBroker.__new__(AlpacaBroker)
    b._trading_client = client
    b._order_counter = 0
    b._g2_breaker = AgentBreaker(off_flag=tmp_path / "TRADING_OFF")
    # The G2 daily notional cap is not under test here (test_agent_breaker.py
    # owns it); lift it so 2500 × $100 admits.
    b._g2_breaker.max_notional = 1e12
    if isinstance(price, Exception):
        def _boom(symbol):
            raise price
        b.get_last_price = _boom
    else:
        b.get_last_price = lambda symbol: price
    return b


def _warnings(caplog, logger="live.alpaca_broker"):
    return [r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == logger]


def _infos(caplog, logger="live.alpaca_broker"):
    return [r for r in caplog.records
            if r.levelno == logging.INFO and r.name == logger]


# ── Legacy oracle: the pre-change construction, verbatim ────────────────────
#
# live/alpaca_broker.py before this change (RenQuant#610 numbering):
#   place_order      :318-343  MarketOrderRequest(symbol, qty=int(quantity),
#                              side, time_in_force=DAY); log "%d shares";
#                              result["quantity"] = int(quantity)
#   place_stop_order :399-417  StopOrderRequest(symbol, qty=int(quantity),
#                              SELL, stop_price=round(..., 2), GTC);
#                              log "%d @ stop"; result["quantity"] = int(quantity)

def _legacy_market_payload(symbol, action, quantity):
    return {
        "symbol": symbol,
        "qty": int(quantity),
        "side": "BUY" if action.upper() == "BUY" else "SELL",
        "time_in_force": "DAY",
    }


def _legacy_market_result(order, action, symbol, quantity):
    return {
        "order_id": str(order.id),
        "status": str(order.status),
        "action": action,
        "symbol": symbol,
        "quantity": int(quantity),
        "filled_qty": float(getattr(order, "filled_qty", 0) or 0),
        "filled_avg_price": float(getattr(order, "filled_avg_price", 0) or 0),
        "submitted_at": str(getattr(order, "submitted_at", "") or ""),
        "filled_at": str(getattr(order, "filled_at", "") or ""),
    }


def _legacy_market_log(order, action, symbol, quantity):
    return "Order %s: %s %s %d shares — status=%s" % (
        order.id, action, symbol, int(quantity), order.status,
    )


def _legacy_stop_payload(symbol, quantity, stop_price):
    return {
        "symbol": symbol,
        "qty": int(quantity),
        "side": "SELL",
        "stop_price": round(float(stop_price), 2),
        "time_in_force": "GTC",
    }


def _legacy_stop_result(order, symbol, quantity, stop_price):
    return {
        "order_id": str(order.id),
        "status": str(order.status),
        "symbol": symbol,
        "quantity": int(quantity),
        "stop_price": float(stop_price),
    }


def _legacy_stop_log(order, symbol, quantity, stop_price):
    return "Stop order %s: SELL %s %d @ stop=$%.2f — status=%s" % (
        order.id, symbol, int(quantity), stop_price, order.status,
    )


_ORDER = SimpleNamespace(
    id="ord-1", status="accepted", filled_qty=None, filled_avg_price=None,
    submitted_at="2026-08-28T20:55:00Z", filled_at=None,
)

# Exact-integer inputs (every quantity today's flag-off sizing emits) plus
# eps-noise ABOVE the integer, where legacy int() and the snap agree.
INTEGRAL_INPUTS = (1, 5, 100, 2500, 5.0, 12.0, 3.0000000004)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Whole-share path: byte-identical to the legacy construction
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegralPathByteIdentical:
    @pytest.mark.parametrize("action", ["BUY", "SELL"])
    @pytest.mark.parametrize("qty", INTEGRAL_INPUTS)
    def test_market_payload_result_log_and_io_unchanged(
        self, sdk, tmp_path, caplog, action, qty,
    ):
        client = _Client(fractionable=AssertionError("must not look up"))
        b = _broker(client, tmp_path)
        with caplog.at_level(logging.INFO, logger="live.alpaca_broker"):
            result = b.place_order(SYMBOL, action, qty)

        assert len(client.submitted) == 1
        req = client.submitted[0]
        assert req.kind == "market"
        assert req.kwargs == _legacy_market_payload(SYMBOL, action, qty)
        assert type(req.kwargs["qty"]) is int
        assert result == _legacy_market_result(_ORDER, action, SYMBOL, qty)
        assert type(result["quantity"]) is int
        assert "requested_quantity" not in result
        assert [r.getMessage() for r in _infos(caplog)] == [
            _legacy_market_log(_ORDER, action, SYMBOL, qty),
        ]
        assert _warnings(caplog) == []
        # I/O unchanged: one pre-trade account read, NO asset lookup, one
        # G2 slot consumed.
        assert client.account_calls == 1
        assert client.asset_calls == []
        assert b._g2_breaker._orders == 1

    def test_eps_noise_below_integer_snaps_to_nearest(self, sdk, tmp_path):
        """The ONE deliberate deviation from the legacy cast, made explicit:
        2.9999999996 is eps-integral (whole-share = 3) and the snap submits
        3. Legacy ``int()`` floored it to 2 — a silent one-share truncation
        of float noise. Such inputs cannot occur on today's flag-off path
        (sizing emits exact ints, see ``INTEGRAL_INPUTS``)."""
        q = 2.9999999996
        assert int(q) == 2  # the legacy cast's answer, for the record
        assert is_whole_share(q)
        client = _Client(fractionable=AssertionError("must not look up"))
        b = _broker(client, tmp_path)
        result = b.place_order(SYMBOL, "BUY", q)
        assert client.submitted[0].kwargs["qty"] == 3
        assert type(client.submitted[0].kwargs["qty"]) is int
        assert result["quantity"] == 3
        assert client.asset_calls == []

    @pytest.mark.parametrize("qty", (7, 7.0, 40, 3.0000000004))
    def test_stop_payload_result_and_log_unchanged(
        self, sdk, tmp_path, caplog, qty,
    ):
        client = _Client(fractionable=AssertionError("must not look up"))
        b = _broker(client, tmp_path)
        with caplog.at_level(logging.INFO, logger="live.alpaca_broker"):
            result = b.place_stop_order(SYMBOL, qty, 90.004)

        req = client.submitted[0]
        assert req.kind == "stop"
        assert req.kwargs == _legacy_stop_payload(SYMBOL, qty, 90.004)
        assert type(req.kwargs["qty"]) is int
        assert result == _legacy_stop_result(_ORDER, SYMBOL, qty, 90.004)
        assert type(result["quantity"]) is int
        assert [r.getMessage() for r in _infos(caplog)] == [
            _legacy_stop_log(_ORDER, SYMBOL, qty, 90.004),
        ]
        assert _warnings(caplog) == []
        assert client.account_calls == 1
        assert client.asset_calls == []

    def test_legacy_guards_on_the_stop_path_are_untouched(self, sdk, tmp_path):
        b = _broker(_Client(True), tmp_path)
        with pytest.raises(ValueError, match="quantity must be positive"):
            b.place_stop_order(SYMBOL, 0, 90.0)
        with pytest.raises(ValueError, match="stop_price must be positive"):
            b.place_stop_order(SYMBOL, 5, 0)

    def test_source_no_longer_carries_the_truncating_cast(self):
        src = (REPO_ROOT / "live" / "alpaca_broker.py").read_text()
        assert "qty=int(quantity)" not in src
        assert '"quantity": int(quantity)' not in src
        assert '"quantity":  int(quantity)' not in src


# ═════════════════════════════════════════════════════════════════════════════
# 2. Fractional intents: refused — never truncated, never submitted
# ═════════════════════════════════════════════════════════════════════════════

def _assert_refused(exc_info, *, symbol, quantity, status):
    exc = exc_info.value
    assert isinstance(exc, FractionalOrderRefused)
    assert isinstance(exc, ValueError)
    assert exc.symbol == symbol
    assert exc.quantity == quantity
    assert exc.status == status
    assert is_no_submit_status(exc.status) is True
    assert status in live_broker.NO_SUBMIT_STATUSES
    assert symbol in str(exc) and repr(quantity) in str(exc)
    assert exc.reason in str(exc)


def _assert_one_refusal_warning(caplog, *, symbol, quantity, status):
    warns = _warnings(caplog)
    assert len(warns) == 1, [w.getMessage() for w in warns]
    msg = warns[0].getMessage()
    assert "FRACTIONAL ORDER REFUSED (no submit)" in msg
    assert symbol in msg and repr(quantity) in msg and status in msg


class TestFractionalRefusals:
    @pytest.mark.parametrize("action", ["BUY", "SELL"])
    def test_not_fractionable_is_refused_nothing_submitted(
        self, sdk, tmp_path, caplog, action,
    ):
        client = _Client(fractionable=False)
        b = _broker(client, tmp_path)
        with caplog.at_level(logging.INFO, logger="live.alpaca_broker"):
            with pytest.raises(FractionalOrderRefused) as ei:
                b.place_order(SYMBOL, action, FRACTIONAL_QTY)
        _assert_refused(ei, symbol=SYMBOL, quantity=FRACTIONAL_QTY,
                        status=NON_FRACTIONABLE_STATUS)
        assert "not fractionable" in ei.value.reason
        assert "NOT floored" in ei.value.reason
        _assert_one_refusal_warning(caplog, symbol=SYMBOL,
                                    quantity=FRACTIONAL_QTY,
                                    status=NON_FRACTIONABLE_STATUS)
        assert _infos(caplog) == []
        # No submit, no account read, no G2 slot burnt; the (confirmed)
        # verdict was looked up exactly once and cached.
        assert client.submitted == []
        assert client.account_calls == 0
        assert b._g2_breaker._orders == 0
        assert client.asset_calls == [SYMBOL]
        assert b._fractionable_cache == {SYMBOL: False}

    def test_lookup_failure_is_refused_and_not_cached(
        self, sdk, tmp_path, caplog,
    ):
        client = _Client(fractionable=RuntimeError("asset API down"))
        b = _broker(client, tmp_path)
        with caplog.at_level(logging.INFO, logger="live.alpaca_broker"):
            with pytest.raises(FractionalOrderRefused) as ei:
                b.place_order(SYMBOL, "BUY", FRACTIONAL_QTY)
        _assert_refused(ei, symbol=SYMBOL, quantity=FRACTIONAL_QTY,
                        status=FRACTIONABLE_LOOKUP_FAILED_STATUS)
        assert "asset API down" in ei.value.reason
        # Exactly ONE warning (the refusal) — the lookup goes through
        # ``_lookup_fractionable`` directly, not ``is_fractionable``'s own
        # warning path — and the failure is not remembered as a verdict.
        _assert_one_refusal_warning(caplog, symbol=SYMBOL,
                                    quantity=FRACTIONAL_QTY,
                                    status=FRACTIONABLE_LOOKUP_FAILED_STATUS)
        assert b._fractionable_cache == {}
        assert client.submitted == []
        assert client.account_calls == 0
        assert b._g2_breaker._orders == 0

    def test_not_connected_is_refused(self, sdk, tmp_path):
        b = _broker(None, tmp_path)
        with pytest.raises(FractionalOrderRefused) as ei:
            b.place_order(SYMBOL, "BUY", FRACTIONAL_QTY)
        _assert_refused(ei, symbol=SYMBOL, quantity=FRACTIONAL_QTY,
                        status=FRACTIONABLE_LOOKUP_FAILED_STATUS)
        assert b._g2_breaker._orders == 0

    def test_known_notional_below_one_dollar_is_refused_before_lookup(
        self, sdk, tmp_path, caplog,
    ):
        client = _Client(fractionable=True)
        b = _broker(client, tmp_path, price=PRICE)  # 0.005 × $100 = $0.50
        with caplog.at_level(logging.WARNING, logger="live.alpaca_broker"):
            with pytest.raises(FractionalOrderRefused) as ei:
                b.place_order(SYMBOL, "BUY", 0.005)
        _assert_refused(ei, symbol=SYMBOL, quantity=0.005,
                        status=BELOW_MIN_NOTIONAL_STATUS)
        assert "$0.5000" in ei.value.reason
        _assert_one_refusal_warning(caplog, symbol=SYMBOL, quantity=0.005,
                                    status=BELOW_MIN_NOTIONAL_STATUS)
        assert client.asset_calls == []  # rule preflight precedes the lookup
        assert client.submitted == []

    def test_unknown_notional_does_not_block_a_fractionable_order(
        self, sdk, tmp_path, caplog,
    ):
        """Price feed down: notional is unknown (G2 count-only accounting,
        unchanged), so the $1 floor cannot be checked here — the broker
        remains the authority for that rejection. Mirrors the owner: the
        qty preflight has no notional to validate."""
        client = _Client(fractionable=True)
        b = _broker(client, tmp_path, price=RuntimeError("data feed down"))
        with caplog.at_level(logging.WARNING, logger="live.alpaca_broker"):
            result = b.place_order(SYMBOL, "BUY", FRACTIONAL_QTY)
        assert client.submitted[0].kwargs["qty"] == FRACTIONAL_QTY
        assert result["quantity"] == FRACTIONAL_QTY
        assert [w.getMessage() for w in _warnings(caplog)] == [
            "G2: last price unavailable for AAPL (data feed down) — "
            "count-only accounting for this order",
        ]

    @pytest.mark.parametrize("order_type,tif", [
        ("limit", "day"),
        ("market", "gtc"),
        ("market", "ioc"),
        ("stop", "gtc"),
        ("stop_limit", "day"),
        ("", ""),
    ])
    def test_non_market_or_non_day_is_refused(
        self, sdk, tmp_path, caplog, order_type, tif,
    ):
        client = _Client(fractionable=True)
        b = _broker(client, tmp_path)
        with caplog.at_level(logging.WARNING, logger="live.alpaca_broker"):
            with pytest.raises(FractionalOrderRefused) as ei:
                b._resolve_submit_qty(
                    SYMBOL, "BUY", FRACTIONAL_QTY,
                    order_type=order_type, time_in_force=tif,
                )
        _assert_refused(ei, symbol=SYMBOL, quantity=FRACTIONAL_QTY,
                        status=INVALID_FRACTIONAL_ORDER_STATUS)
        assert repr(order_type) in ei.value.reason
        assert repr(tif) in ei.value.reason
        _assert_one_refusal_warning(caplog, symbol=SYMBOL,
                                    quantity=FRACTIONAL_QTY,
                                    status=INVALID_FRACTIONAL_ORDER_STATUS)
        assert client.asset_calls == []  # refused before any I/O
        assert client.submitted == []

    def test_market_day_is_the_only_accepted_shape(self, sdk, tmp_path):
        b = _broker(_Client(fractionable=True), tmp_path)
        for order_type, tif in (("market", "day"), ("MARKET", "DAY"), (" market ", "Day")):
            assert b._resolve_submit_qty(
                SYMBOL, "BUY", FRACTIONAL_QTY,
                order_type=order_type, time_in_force=tif,
            ) == FRACTIONAL_QTY

    @pytest.mark.parametrize("bad", [
        -0.5, float("nan"), float("inf"), -float("inf"), "abc", None,
    ])
    def test_non_positive_or_non_finite_fractional_is_refused(
        self, sdk, tmp_path, caplog, bad,
    ):
        client = _Client(fractionable=True)
        b = _broker(client, tmp_path)
        with caplog.at_level(logging.WARNING, logger="live.alpaca_broker"):
            with pytest.raises(FractionalOrderRefused) as ei:
                b._resolve_submit_qty(
                    SYMBOL, "BUY", bad, order_type="market", time_in_force="day",
                )
        assert ei.value.status == INVALID_FRACTIONAL_ORDER_STATUS
        assert "finite and positive" in ei.value.reason
        assert client.asset_calls == [] and client.submitted == []

    def test_nan_through_place_order_is_refused_not_submitted(
        self, sdk, tmp_path,
    ):
        client = _Client(fractionable=True)
        b = _broker(client, tmp_path)
        with pytest.raises(FractionalOrderRefused):
            b.place_order(SYMBOL, "BUY", float("nan"))
        assert client.submitted == [] and client.account_calls == 0

    def test_smallest_non_integral_qty_snaps_onto_the_grid_not_to_zero(
        self, sdk, tmp_path,
    ):
        """Every value within 1e-9 of an integer is whole-share (the legacy
        branch), so the smallest fractional value is > 1e-9 and floors onto
        the 1e-9 grid — the "rounds to zero" refusal in the preflight is a
        defensive guard that this pins as unreachable for finite input."""
        client = _Client(fractionable=True)
        b = _broker(client, tmp_path)
        assert is_whole_share(5e-10) and not is_whole_share(1.5e-9)
        assert b._resolve_submit_qty(
            SYMBOL, "BUY", 1.5e-9, order_type="market", time_in_force="day",
            notional=1e3,
        ) == 1e-9


# ═════════════════════════════════════════════════════════════════════════════
# 3. Fractional intents on a confirmed-fractionable asset: exact submission
# ═════════════════════════════════════════════════════════════════════════════

class TestFractionalSubmission:
    @pytest.mark.parametrize("action", ["BUY", "SELL"])
    def test_exact_qty_market_day(self, sdk, tmp_path, caplog, action):
        client = _Client(fractionable=True)
        b = _broker(client, tmp_path)
        with caplog.at_level(logging.INFO, logger="live.alpaca_broker"):
            result = b.place_order(SYMBOL, action, FRACTIONAL_QTY)

        req = client.submitted[0]
        assert req.kind == "market"
        assert req.kwargs == {
            "symbol": SYMBOL,
            "qty": FRACTIONAL_QTY,
            "side": action,
            "time_in_force": "DAY",
        }
        assert type(req.kwargs["qty"]) is float
        assert result["quantity"] == FRACTIONAL_QTY
        assert type(result["quantity"]) is float
        assert result["requested_quantity"] == FRACTIONAL_QTY
        assert result["order_id"] == "ord-1" and result["status"] == "accepted"
        assert [r.getMessage() for r in _infos(caplog)] == [
            f"Order ord-1: {action} AAPL 0.435578 shares — status=accepted",
        ]
        assert _warnings(caplog) == []
        assert client.asset_calls == [SYMBOL]
        assert client.account_calls == 1
        assert b._g2_breaker._orders == 1

    def test_confirmed_verdict_is_cached_across_orders(self, sdk, tmp_path):
        client = _Client(fractionable=True)
        b = _broker(client, tmp_path)
        b.place_order(SYMBOL, "BUY", FRACTIONAL_QTY)
        b.place_order(SYMBOL, "SELL", 0.25)
        assert client.asset_calls == [SYMBOL]
        assert [r.kwargs["qty"] for r in client.submitted] == [FRACTIONAL_QTY, 0.25]

    @pytest.mark.parametrize("requested,submitted", [
        (0.1234567891234, 0.123456789),   # 13dp → 9dp, floor
        (0.5999999999, 0.599999999),      # floor, never up to 0.6
        (1.0000000019, 1.000000001),      # never up past the intent
        (0.435578, 0.435578),             # already on-grid: verbatim
        (7.5, 7.5),
        (1e-05, 1e-05),
    ])
    def test_qty_snaps_down_to_9dp_never_past_the_intent(
        self, sdk, tmp_path, requested, submitted,
    ):
        client = _Client(fractionable=True)
        # High price so the $1 floor is not the blocker for 1e-05 shares.
        b = _broker(client, tmp_path, price=1e7)
        result = b.place_order(SYMBOL, "BUY", requested)
        qty = client.submitted[0].kwargs["qty"]
        assert qty == submitted
        assert qty <= requested
        assert round(qty, 9) == qty
        assert result["quantity"] == submitted
        assert result["requested_quantity"] == requested

    def test_real_sdk_accepts_a_float_qty(self):
        """The actual alpaca-py request model must accept the fractional
        float this path now submits (guards an SDK contract, not our code)."""
        alpaca = pytest.importorskip("alpaca")
        from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: PLC0415
        from alpaca.trading.requests import MarketOrderRequest  # noqa: PLC0415
        req = MarketOrderRequest(
            symbol=SYMBOL, qty=FRACTIONAL_QTY, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        assert float(req.qty) == FRACTIONAL_QTY, alpaca


# ═════════════════════════════════════════════════════════════════════════════
# 3b. Broker-side stops: whole-share only
# ═════════════════════════════════════════════════════════════════════════════

class TestStopPath:
    @pytest.mark.parametrize("qty", [7.5, FRACTIONAL_QTY, 0.99999999])
    def test_fractional_stop_is_refused_before_any_io(
        self, sdk, tmp_path, caplog, qty,
    ):
        client = _Client(fractionable=True)  # fractionable does NOT help
        b = _broker(client, tmp_path)
        with caplog.at_level(logging.INFO, logger="live.alpaca_broker"):
            with pytest.raises(FractionalOrderRefused) as ei:
                b.place_stop_order(SYMBOL, qty, 90.0)
        _assert_refused(ei, symbol=SYMBOL, quantity=qty,
                        status=INVALID_FRACTIONAL_ORDER_STATUS)
        assert "software-stop layer" in ei.value.reason
        assert "order_type='stop'" in ei.value.reason
        _assert_one_refusal_warning(caplog, symbol=SYMBOL, quantity=qty,
                                    status=INVALID_FRACTIONAL_ORDER_STATUS)
        assert _infos(caplog) == []
        assert client.submitted == []
        assert client.account_calls == 0
        assert client.asset_calls == []

    def test_capability_probe_and_stop_path_agree(self, sdk, tmp_path):
        """supports_broker_side_stops(symbol, qty) already answers False for
        a fractional qty; the stop path now refuses the same set, so the Z9
        router and the broker can never disagree about protectability."""
        b = _broker(_Client(fractionable=True), tmp_path)
        for q in (1, 7.0, 3.0000000004, 2.9999999996):
            assert b.supports_broker_side_stops(SYMBOL, q) is True
            assert isinstance(b._resolve_submit_qty(
                SYMBOL, "SELL", q, order_type="stop", time_in_force="gtc"), int)
        for q in (7.5, FRACTIONAL_QTY, 0.99999999):
            assert b.supports_broker_side_stops(SYMBOL, q) is False
            with pytest.raises(FractionalOrderRefused):
                b._resolve_submit_qty(
                    SYMBOL, "SELL", q, order_type="stop", time_in_force="gtc")


# ═════════════════════════════════════════════════════════════════════════════
# 4. live/broker.py helpers + owner-constant drift tripwire
# ═════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    @pytest.mark.parametrize("q,expected", [
        (5, True), (5.0, True), (0, True), (3.0000000004, True),
        (2.9999999996, True), (5e-10, True), (0.999999999, True),
        (FRACTIONAL_QTY, False), (7.5, False), (1.5e-9, False),
        (0.99999999, False), (float("nan"), False), (float("inf"), False),
        ("abc", False), (None, False),
    ])
    def test_is_whole_share(self, q, expected):
        assert is_whole_share(q) is expected

    @pytest.mark.parametrize("q,expected", [
        (0.435578, 0.435578), (0.1234567891234, 0.123456789),
        (0.9999999999, 0.999999999), (1.0000000019, 1.000000001),
        (7.5, 7.5), (1e-05, 1e-05), (1.5e-9, 1e-9), (5e-10, 0.0),
    ])
    def test_snap_rounds_down_on_the_9dp_grid(self, q, expected):
        s = snap_qty_to_broker_grid(q)
        assert s == expected
        assert s <= q
        assert round(s, 9) == s

    def test_snap_rejects_non_finite(self):
        with pytest.raises(ValueError):
            snap_qty_to_broker_grid(float("nan"))

    def test_exception_shape(self):
        exc = FractionalOrderRefused("MSFT", 1.5, "because", status=NON_FRACTIONABLE_STATUS)
        assert isinstance(exc, ValueError)
        assert (exc.symbol, exc.quantity, exc.reason, exc.status) == (
            "MSFT", 1.5, "because", NON_FRACTIONABLE_STATUS)
        assert str(exc) == (
            "fractional order refused (no submit) for MSFT qty=1.5: because "
            "[status=rejected_non_fractionable]"
        )
        assert FractionalOrderRefused("X", 0.5, "r").status == INVALID_FRACTIONAL_ORDER_STATUS

    def test_every_refusal_status_is_no_submit_vocabulary(self):
        for status in (NON_FRACTIONABLE_STATUS, FRACTIONABLE_LOOKUP_FAILED_STATUS,
                       BELOW_MIN_NOTIONAL_STATUS, INVALID_FRACTIONAL_ORDER_STATUS):
            assert status in live_broker.NO_SUBMIT_STATUSES
            assert status in live_broker._FALLBACK_NO_SUBMIT_STATUSES
            assert is_no_submit_status(status) is True
            assert AlpacaBroker.is_no_submit_status(status) is True

    def test_constants_and_helpers_match_the_owner(self):
        """Drift tripwire against the pinned renquant-execution checkout
        (subrepos.lock.json), resolved the same way as
        test_live_broker_fractional_contract.py."""
        from _order_math_owner import _inject_sibling_src_paths  # noqa: PLC0415
        _inject_sibling_src_paths()
        try:
            from renquant_execution import broker as owner  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            pytest.skip("renquant_execution.broker unavailable (no sibling checkout)")
        assert live_broker.QTY_INTEGRAL_EPS == owner.QTY_INTEGRAL_EPS
        assert live_broker.MAX_ORDER_DECIMAL_PLACES == owner.MAX_ORDER_DECIMAL_PLACES
        assert live_broker.MIN_FRACTIONAL_NOTIONAL_USD == owner.MIN_FRACTIONAL_NOTIONAL_USD
        assert live_broker.FRACTIONAL_TIME_IN_FORCE == owner.FRACTIONAL_TIME_IN_FORCE
        assert live_broker.FRACTIONAL_ORDER_TYPE in owner.FRACTIONAL_ORDER_TYPES
        assert live_broker.NON_FRACTIONABLE_STATUS == owner.NON_FRACTIONABLE_STATUS
        assert (live_broker.FRACTIONABLE_LOOKUP_FAILED_STATUS
                == owner.FRACTIONABLE_LOOKUP_FAILED_STATUS)
        assert live_broker.BELOW_MIN_NOTIONAL_STATUS == owner.BELOW_MIN_NOTIONAL_STATUS
        assert (live_broker.INVALID_FRACTIONAL_ORDER_STATUS
                == owner.INVALID_FRACTIONAL_ORDER_STATUS)
        for q in (5, 5.0, 3.0000000004, 2.9999999996, FRACTIONAL_QTY, 7.5,
                  1.5e-9, float("nan"), float("inf")):
            assert is_whole_share(q) is owner.is_whole_share(q), q


# ═════════════════════════════════════════════════════════════════════════════
# 5. Runner mapping: the refusal is a no-submit outcome, never a crash
# ═════════════════════════════════════════════════════════════════════════════

RUNNER_SRC = (_STRATEGY / "adapters" / "runner.py").read_text()
Z9_SRC = (_STRATEGY / "adapters" / "z9_stops.py").read_text()


class TestRunnerMappingStatic:
    """Runs everywhere (no strategy deps): the three handling sites wrap the
    broker call in ``except Exception`` and route to the existing skip /
    fail surfaces. The end-to-end proof is ``TestRunnerMapping`` below."""

    def test_buy_site_routes_exceptions_to_orders_skipped(self):
        i = RUNNER_SRC.index('result = broker.place_order(ticker, "BUY", shares)')
        block = RUNNER_SRC[i:i + 400]
        assert "except Exception as exc:" in block
        assert 'ctx.orders_skipped.append' in block
        assert '"skip_reason": f"broker_error:{type(exc).__name__}"' in block
        assert "continue" in block

    def test_sell_site_routes_exceptions_to_exits_failed(self):
        import re  # noqa: PLC0415
        i = RUNNER_SRC.index('result = broker.place_order(ticker, "SELL", sell_qty)')
        block = RUNNER_SRC[i:i + 600]
        assert "except Exception as exc:" in block
        assert "ctx.exits_failed.append" in block
        assert re.search(r'"error":\s+str\(exc\)', block)
        assert "continue" in block

    def test_z9_site_routes_exceptions_to_warning_and_no_stop(self):
        i = Z9_SRC.index("result = broker.place_stop_order(ticker, qty, target)")
        block = Z9_SRC[i:i + 300]
        assert "except Exception as exc:" in block
        assert "return" in block


def _runner_harness():
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    import test_s_frac_stage0_commit_contract as h  # noqa: PLC0415
    return h


class TestRunnerMapping:
    """End-to-end through the REAL ``RunnerAdapter.commit`` (the stage-0
    harness of test_s_frac_stage0_commit_contract.py) with a broker whose
    order method raises ``FractionalOrderRefused`` exactly as the live
    broker now does."""

    @staticmethod
    def _refusing_broker(h, **kw):
        class RefusingBroker(h.FakeBroker):
            def place_order(self, ticker, side, qty):
                self.place_order_calls.append((ticker, side, qty))
                raise FractionalOrderRefused(
                    ticker, qty,
                    f"{ticker} is not fractionable at the broker; fractional "
                    f"qty {qty!r} is refused, NOT floored",
                    status=NON_FRACTIONABLE_STATUS,
                )
        return RefusingBroker(**kw)

    def test_buy_refusal_lands_in_orders_skipped_and_run_continues(
        self, tmp_path, caplog,
    ):
        h = _runner_harness()
        config = h._config(fractional=True)
        broker = self._refusing_broker(h, fills={}, fractional_contract=True)
        ra = h._make_adapter(tmp_path, config=config, broker=broker,
                             software_stops=h.ArmedSoftwareStops())
        ctx = h._make_ctx(
            config,
            orders=[{"ticker": "BLK", "shares": FRACTIONAL_QTY, "price": 100.0}],
            prices={"BLK": 100.0}, cash=1_000.0,
        )
        with caplog.at_level(logging.ERROR, logger="live.runner"):
            ra.commit(ctx)  # must NOT raise

        # The intent reached the broker (gate + stop-routing passed) and
        # was refused there — recorded on the existing no-submit surface.
        assert broker.place_order_calls == [("BLK", "BUY", FRACTIONAL_QTY)]
        assert ctx.orders_placed == []
        assert ctx.orders_pending == []
        assert [(o["ticker"], o["skip_reason"]) for o in ctx.orders_skipped] == [
            ("BLK", "broker_error:FractionalOrderRefused"),
        ]
        assert any("BUY failed for BLK" in r.getMessage()
                   and "fractional order refused (no submit)" in r.getMessage()
                   for r in caplog.records)
        # Nothing mutated: no entry, no stop, state persisted cleanly.
        assert broker.place_stop_calls == []
        state = h._saved_state(tmp_path)
        assert "BLK" not in state["entry_dates"]
        assert state["stop_orders"] == {}
        assert h._journal_records(tmp_path, action="BUY") == []

    def test_sell_refusal_lands_in_exits_failed_and_position_is_kept(
        self, tmp_path, caplog,
    ):
        h = _runner_harness()
        import datetime  # noqa: PLC0415
        from kernel.exits import ExitSignal, HoldingState  # noqa: PLC0415

        config = h._config(fractional=False)  # exits are never gated
        broker = self._refusing_broker(
            h, fills={}, positions={"BLK": FRACTIONAL_QTY},
        )
        ra = h._make_adapter(
            tmp_path, config=config, broker=broker,
            positions={"BLK": {"qty": FRACTIONAL_QTY,
                               "qty_available": FRACTIONAL_QTY,
                               "avg_entry_price": 100.0}},
            entry_dates={"BLK": "2026-06-20"},
            position_hwm={"BLK": 105.0},
        )
        hs = HoldingState(entry_price=100.0,
                          entry_date=datetime.date(2026, 6, 20),
                          high_watermark=105.0)
        sig = ExitSignal(should_exit=True, reason="model sell",
                         exit_type="model_sell")
        ctx = h._make_ctx(config, exits=[("BLK", sig)], holdings={"BLK": hs},
                          prices={"BLK": 101.0})
        with caplog.at_level(logging.ERROR, logger="live.runner"):
            ra.commit(ctx)  # must NOT raise

        assert broker.place_order_calls == [("BLK", "SELL", FRACTIONAL_QTY)]
        assert ctx.exits_placed == []
        assert len(ctx.exits_failed) == 1
        failed = ctx.exits_failed[0]
        assert failed["ticker"] == "BLK"
        assert failed["qty"] == FRACTIONAL_QTY
        assert "fractional order refused (no submit)" in failed["error"]
        assert NON_FRACTIONABLE_STATUS in failed["error"]
        assert any("SELL failed for BLK" in r.getMessage() for r in caplog.records)
        # The position was NOT reaped (nothing was sold — no truncated
        # partial exit either) and the run persisted state normally.
        state = h._saved_state(tmp_path)
        assert state["entry_dates"]["BLK"] == "2026-06-20"
        assert h._journal_records(tmp_path, action="SELL") == []

    def test_z9_refusal_is_a_warning_and_no_stop_is_recorded(self, caplog):
        """Defence in depth: even a broker whose capability probe lies
        (answers True for a fractional qty) cannot make Z9 record a stop
        that the stop path refused."""
        h = _runner_harness()
        from adapters.z9_stops import place_or_replace_stop  # noqa: PLC0415

        class LyingBroker(h.FakeBroker):
            def supports_broker_side_stops(self, symbol=None, qty=None):
                return True

            def place_stop_order(self, symbol, quantity, stop_price):
                self.place_stop_calls.append((symbol, quantity, stop_price))
                raise FractionalOrderRefused(
                    symbol, quantity, "broker-side GTC stops are whole-share only",
                )

        broker = LyingBroker()
        stop_orders: dict = {}
        with caplog.at_level(logging.WARNING, logger="adapters.z9_stops"):
            place_or_replace_stop(broker, stop_orders, "BLK", 7.5, 100.0,
                                  "2026-08-28", ctx_pct=0.06)
        assert broker.place_stop_calls == [("BLK", 7.5, 94.0)]
        assert stop_orders == {}
        assert any("Z9: place_stop_order(BLK, qty=7.5" in r.getMessage()
                   and "fractional order refused" in r.getMessage()
                   for r in caplog.records)
