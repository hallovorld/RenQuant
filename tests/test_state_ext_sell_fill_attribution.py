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
    ra._recent_sell_orders = {}  # runner-submitted SELL order_ids  # noqa: SLF001
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
    """Original generic ``side`` schema (kept for back-compat)."""
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
    # Codex #76: normalized keys are ``price`` + ``qty`` (not the broker-
    # specific ``fill_price`` / ``filled_qty``).
    assert result["GE"]["price"] == 110.0
    assert result["GE"]["qty"] == 7
    assert result["GE"]["side"] == "sell"
    # DUK had no fills in the window
    assert "DUK" not in result


# ── Codex #76: live-broker schema (action + avg_price) ─────────────────────

def test_lookup_handles_live_alpaca_broker_schema():
    """Umbrella ``live/alpaca_broker.py::get_filled_orders`` returns:
        {order_id, symbol, action ("BUY"/"SELL"), qty, filled_at, avg_price, partial}

    Codex #76 found the original code looked for ``side`` only, accepting
    a BUY fill silently. This test pins the live-schema path."""
    ra = _adapter_skeleton()

    fills = [
        # GE SELL via action key + avg_price + filled_qty path
        {"order_id": "stop-abc", "symbol": "GE", "action": "SELL",
         "qty": 10, "filled_at": "2026-06-01T19:30:00Z",
         "avg_price": 95.0, "partial": False},
        # GE BUY via action — must be excluded
        {"order_id": "buy-xyz", "symbol": "GE", "action": "BUY",
         "qty": 5, "filled_at": "2026-06-01T20:00:00Z",
         "avg_price": 96.0, "partial": False},
    ]

    class _LiveAlpacaBroker:
        def get_filled_orders(self, after=None):
            return fills

    ra._broker = _LiveAlpacaBroker()
    ra._stop_orders = {"GE": {"order_id": "stop-abc"}}

    result = ra._lookup_ext_sell_fills(_fake_ctx(), ["GE"])
    assert "GE" in result, "live-broker SELL must be captured"
    assert result["GE"]["order_id"] == "stop-abc"
    assert result["GE"]["price"] == 95.0
    assert result["GE"]["qty"] == 10
    assert result["GE"]["side"] == "sell"

    # And attribution correctly classifies as z9_stop now.
    attribution = ra._attribute_ext_sell("GE", result)
    assert "source=z9_stop" in attribution
    assert "order_id=stop-abc" in attribution
    assert "price=95.0" in attribution
    assert "qty=10" in attribution


def test_lookup_handles_execution_subrepo_schema():
    """``renquant-execution/alpaca_broker.py::get_filled_orders`` returns:
        {order_id, status, symbol, filled_qty, filled_avg_price, filled_at, ...}
    No ``side`` / ``action`` field. The lookup must accept the row (absence
    of a side field is not "this is a buy") and read ``filled_avg_price`` /
    ``filled_qty``."""
    ra = _adapter_skeleton()

    fills = [
        {"order_id": "stop-def", "symbol": "DUK",
         "filled_qty": 20, "filled_avg_price": 75.0,
         "filled_at": "2026-06-01T18:00:00Z",
         "status": "filled"},
    ]

    class _ExecutionSubrepoBroker:
        def get_filled_orders(self, after=None):
            return fills

    ra._broker = _ExecutionSubrepoBroker()
    ra._stop_orders = {"DUK": {"order_id": "stop-def"}}

    result = ra._lookup_ext_sell_fills(_fake_ctx(), ["DUK"])
    assert "DUK" in result
    assert result["DUK"]["order_id"] == "stop-def"
    assert result["DUK"]["price"] == 75.0
    assert result["DUK"]["qty"] == 20
    # No side field on the input → normalized side = ""
    assert result["DUK"]["side"] == ""

    attribution = ra._attribute_ext_sell("DUK", result)
    assert "source=z9_stop" in attribution


def test_lookup_rejects_buy_action_under_live_schema():
    """Regression: previously, an ``action=BUY`` fill was accepted because
    the code only checked ``side``. Now it must be filtered out."""
    ra = _adapter_skeleton()

    fills = [
        # Only BUY — should produce no result
        {"order_id": "buy-only", "symbol": "GE", "action": "BUY",
         "qty": 5, "filled_at": "2026-06-01T20:00:00Z",
         "avg_price": 96.0, "partial": False},
    ]

    class _LiveAlpacaBroker:
        def get_filled_orders(self, after=None):
            return fills

    ra._broker = _LiveAlpacaBroker()
    result = ra._lookup_ext_sell_fills(_fake_ctx(), ["GE"])
    assert result == {}, f"BUY-only fills must not be captured; got {result}"


# ── _attribute_ext_sell ─────────────────────────────────────────────────────

def test_attribution_no_fill_record():
    ra = _adapter_skeleton()
    assert ra._attribute_ext_sell("GE", {}) == "no_broker_fill_record"


def test_attribution_z9_stop_when_order_ids_match():
    ra = _adapter_skeleton()
    ra._stop_orders = {"GE": {"order_id": "stop-abc"}}
    fills = {"GE": {"order_id": "stop-abc", "price": 95.0,
                    "qty": 10, "filled_at": "2026-06-01T19:30:00Z"}}
    text = ra._attribute_ext_sell("GE", fills)
    assert "source=z9_stop" in text
    assert "order_id=stop-abc" in text
    assert "price=95.0" in text
    assert "qty=10" in text


def test_attribution_external_when_order_ids_differ():
    """Z9 stop tracked but fill came from a different order_id → manual."""
    ra = _adapter_skeleton()
    ra._stop_orders = {"GE": {"order_id": "stop-abc"}}
    fills = {"GE": {"order_id": "manual-xyz", "price": 100.0,
                    "qty": 10, "filled_at": "2026-06-01T15:00:00Z"}}
    text = ra._attribute_ext_sell("GE", fills)
    assert "source=external_or_manual" in text
    assert "order_id=manual-xyz" in text


def test_attribution_external_when_no_z9_stop_tracked():
    """No Z9 stop for this ticker → external by definition."""
    ra = _adapter_skeleton()
    ra._stop_orders = {}
    fills = {"GE": {"order_id": "any-id", "price": 100.0,
                    "qty": 5, "filled_at": "2026-06-01T15:00:00Z"}}
    text = ra._attribute_ext_sell("GE", fills)
    assert "source=external_or_manual" in text


def test_attribution_handles_missing_fields_gracefully():
    """A broker that omits fields shouldn't crash the log emission."""
    ra = _adapter_skeleton()
    ra._stop_orders = {}
    fills = {"GE": {"order_id": None, "price": None,
                    "qty": None, "filled_at": None}}
    text = ra._attribute_ext_sell("GE", fills)
    assert "source=external_or_manual" in text
    # Each missing field renders as "?" — single-line, no exceptions
    assert text.count("?") >= 4


# ── runner-submitted fill attribution (2026-06-03 HON incident) ──────────────

def test_attribution_runner_sell_when_order_id_was_submitted():
    """The HON regression: a runner single_day_loss sell that filled must NOT
    be mislabeled external_or_manual on the next tick's reconciliation."""
    ra = _adapter_skeleton()
    ra._stop_orders = {}  # not a Z9 stop
    ra._recent_sell_orders = {
        "d98d2cbc": {
            "ticker": "HON",
            "exit_type": "single_day_loss",
            "qty": 2.0,
            "submitted_at": "2026-06-03",
        }
    }
    fills = {"HON": {"order_id": "d98d2cbc", "price": 223.83,
                     "qty": 2, "filled_at": "2026-06-03T19:24:23Z"}}
    text = ra._attribute_ext_sell("HON", fills)
    assert "source=runner_single_day_loss" in text
    assert "external_or_manual" not in text
    assert "order_id=d98d2cbc" in text


def test_attribution_runner_sell_without_exit_type_falls_back():
    ra = _adapter_skeleton()
    ra._recent_sell_orders = {"oid-1": {"ticker": "GE", "exit_type": "",
                                        "qty": 3.0, "submitted_at": "2026-06-03"}}
    fills = {"GE": {"order_id": "oid-1", "price": 50.0,
                    "qty": 3, "filled_at": "2026-06-03T15:00:00Z"}}
    text = ra._attribute_ext_sell("GE", fills)
    assert "source=runner_sell" in text


def test_attribution_z9_stop_wins_over_runner_record():
    """If a fill matches BOTH a Z9 stop and a recorded runner order, the Z9
    classification takes precedence (it's the more specific source)."""
    ra = _adapter_skeleton()
    ra._stop_orders = {"GE": {"order_id": "shared-oid"}}
    ra._recent_sell_orders = {"shared-oid": {"ticker": "GE",
                                             "exit_type": "trailing_stop",
                                             "qty": 1.0,
                                             "submitted_at": "2026-06-03"}}
    fills = {"GE": {"order_id": "shared-oid", "price": 100.0,
                    "qty": 1, "filled_at": "2026-06-03T15:00:00Z"}}
    text = ra._attribute_ext_sell("GE", fills)
    assert "source=z9_stop" in text


def test_attribution_external_when_order_id_not_runner_submitted():
    """A genuinely external fill (no Z9, no runner record) stays external."""
    ra = _adapter_skeleton()
    ra._stop_orders = {}
    ra._recent_sell_orders = {"runner-oid": {"ticker": "GE",
                                             "exit_type": "model_sell",
                                             "qty": 5.0,
                                             "submitted_at": "2026-06-03"}}
    fills = {"GE": {"order_id": "broker-external-oid", "price": 100.0,
                    "qty": 5, "filled_at": "2026-06-03T15:00:00Z"}}
    text = ra._attribute_ext_sell("GE", fills)
    assert "source=external_or_manual" in text


# ── _gc_recent_sell_orders ───────────────────────────────────────────────────

def test_gc_drops_orders_older_than_window():
    ra = _adapter_skeleton()
    ra._recent_sell_orders = {
        "fresh": {"ticker": "A", "exit_type": "x", "qty": 1,
                  "submitted_at": "2026-06-01"},   # 0 days old
        "stale": {"ticker": "B", "exit_type": "x", "qty": 1,
                  "submitted_at": "2026-05-20"},   # 12 days old → dropped
    }
    kept = ra._gc_recent_sell_orders(_fake_ctx(today=_dt.date(2026, 6, 1)))
    assert "fresh" in kept
    assert "stale" not in kept
    # mutates in place too
    assert ra._recent_sell_orders == kept


def test_gc_keeps_unparseable_timestamp_fail_open():
    ra = _adapter_skeleton()
    ra._recent_sell_orders = {
        "weird": {"ticker": "A", "exit_type": "x", "qty": 1,
                  "submitted_at": "not-a-date"},
    }
    kept = ra._gc_recent_sell_orders(_fake_ctx(today=_dt.date(2026, 6, 1)))
    assert "weird" in kept   # fail-open: never lose an order we might attribute


def test_gc_boundary_keeps_exactly_six_days():
    ra = _adapter_skeleton()
    ra._recent_sell_orders = {
        "edge": {"ticker": "A", "exit_type": "x", "qty": 1,
                 "submitted_at": "2026-05-26"},   # exactly 6 days → kept (>= cutoff)
    }
    kept = ra._gc_recent_sell_orders(_fake_ctx(today=_dt.date(2026, 6, 1)))
    assert "edge" in kept
