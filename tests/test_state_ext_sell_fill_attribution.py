"""Regression coverage for STATE-EXT-SELL fill attribution (issue #71).

Audit #5: STATE-EXT-SELL log used to only carry the ticker name, so the
operator couldn't tell whether the position was closed by a Z9 broker-side
stop firing, a manual liquidation at Alpaca, or a corporate action.

This commit adds two helpers on ``RunnerAdapter``:

  * ``_lookup_ext_sell_fills(ctx, disappeared)`` — fetches the most recent
    SELL fill per disappeared ticker from ``broker.get_filled_orders``.
  * ``_attribute_ext_sell(ticker, fills)`` — classifies as ``z9_stop`` when
    the fill's ``order_id`` matches a tracked Z9 stop order; otherwise
    ``external_or_manual``. Falls back to ``no_broker_fill_record`` when
    the broker can't be queried.

These tests pin those classifications.
"""
from __future__ import annotations

import datetime as _dt
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY) not in sys.path:
    sys.path.insert(0, str(_STRATEGY))


def _adapter_skeleton():
    """Construct a `RunnerAdapter` shell whose only loaded surface is the
    two helpers we want to exercise. Avoids full adapter init (which pulls
    broker connection, state-file IO, etc.)."""
    from adapters.runner import RunnerAdapter  # noqa: E402, PLC0415

    ra = RunnerAdapter.__new__(RunnerAdapter)
    ra._stop_orders = {}  # populated per test  # noqa: SLF001
    ra._broker = None  # patched per test  # noqa: SLF001
    return ra


def _fake_ctx(today: _dt.date = _dt.date(2026, 6, 1)) -> types.SimpleNamespace:
    return types.SimpleNamespace(today=today)


# ── _lookup_ext_sell_fills ──────────────────────────────────────────────────

def test_lookup_returns_empty_when_disappeared_empty():
    ra = _adapter_skeleton()
    assert ra._lookup_ext_sell_fills(_fake_ctx(), []) == {}


def test_lookup_returns_empty_when_broker_lacks_filled_orders():
    ra = _adapter_skeleton()

    class _BrokerWithoutFills:
        # no get_filled_orders attribute
        pass

    ra._broker = _BrokerWithoutFills()
    assert ra._lookup_ext_sell_fills(_fake_ctx(), ["GE"]) == {}


def test_lookup_swallows_broker_exception():
    ra = _adapter_skeleton()

    class _BrokerThatRaises:
        def get_filled_orders(self, after=None):
            raise RuntimeError("broker offline")

    ra._broker = _BrokerThatRaises()
    assert ra._lookup_ext_sell_fills(_fake_ctx(), ["GE"]) == {}


def test_lookup_picks_latest_sell_per_ticker():
    ra = _adapter_skeleton()

    fills = [
        # different ticker — ignored
        {"symbol": "FOO", "side": "sell",
         "order_id": "f1", "filled_at": "2026-06-01T16:00:00Z"},
        # GE earlier sell
        {"symbol": "GE", "side": "sell",
         "order_id": "ge-old", "fill_price": 100.0, "filled_qty": 5,
         "filled_at": "2026-05-30T18:00:00Z"},
        # GE later sell — must win
        {"symbol": "GE", "side": "sell",
         "order_id": "ge-new", "fill_price": 110.0, "filled_qty": 7,
         "filled_at": "2026-06-01T19:30:00Z"},
        # GE BUY — must be ignored (wrong side)
        {"symbol": "GE", "side": "buy",
         "order_id": "ge-buy", "filled_at": "2026-06-01T20:00:00Z"},
    ]

    class _BrokerWithFills:
        def get_filled_orders(self, after=None):
            return fills

    ra._broker = _BrokerWithFills()

    result = ra._lookup_ext_sell_fills(_fake_ctx(), ["GE", "DUK"])
    assert "GE" in result
    assert result["GE"]["order_id"] == "ge-new"
    assert result["GE"]["fill_price"] == 110.0
    assert result["GE"]["fill_qty"] == 7
    # DUK had no fills in the window
    assert "DUK" not in result


# ── _attribute_ext_sell ─────────────────────────────────────────────────────

def test_attribution_no_fill_record():
    ra = _adapter_skeleton()
    assert ra._attribute_ext_sell("GE", {}) == "no_broker_fill_record"


def test_attribution_z9_stop_when_order_ids_match():
    ra = _adapter_skeleton()
    ra._stop_orders = {"GE": {"order_id": "stop-abc"}}
    fills = {"GE": {"order_id": "stop-abc", "fill_price": 95.0,
                    "fill_qty": 10, "filled_at": "2026-06-01T19:30:00Z"}}
    text = ra._attribute_ext_sell("GE", fills)
    assert "source=z9_stop" in text
    assert "order_id=stop-abc" in text
    assert "price=95.0" in text
    assert "qty=10" in text


def test_attribution_external_when_order_ids_differ():
    """Z9 stop tracked but fill came from a different order_id → manual."""
    ra = _adapter_skeleton()
    ra._stop_orders = {"GE": {"order_id": "stop-abc"}}
    fills = {"GE": {"order_id": "manual-xyz", "fill_price": 100.0,
                    "fill_qty": 10, "filled_at": "2026-06-01T15:00:00Z"}}
    text = ra._attribute_ext_sell("GE", fills)
    assert "source=external_or_manual" in text
    assert "order_id=manual-xyz" in text


def test_attribution_external_when_no_z9_stop_tracked():
    """No Z9 stop for this ticker → external by definition."""
    ra = _adapter_skeleton()
    ra._stop_orders = {}
    fills = {"GE": {"order_id": "any-id", "fill_price": 100.0,
                    "fill_qty": 5, "filled_at": "2026-06-01T15:00:00Z"}}
    text = ra._attribute_ext_sell("GE", fills)
    assert "source=external_or_manual" in text


def test_attribution_handles_missing_fields_gracefully():
    """A broker that omits fields shouldn't crash the log emission."""
    ra = _adapter_skeleton()
    ra._stop_orders = {}
    fills = {"GE": {"order_id": None, "fill_price": None,
                    "fill_qty": None, "filled_at": None}}
    text = ra._attribute_ext_sell("GE", fills)
    assert "source=external_or_manual" in text
    # Each missing field renders as "?" — single-line, no exceptions
    assert text.count("?") >= 4
