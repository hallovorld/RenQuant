"""Tests for Z9 — broker-side stop orders.

Z9 (2026-04-28): NVTS post-mortem showed the polled stop_loss check is
gated by 30-min cron cadence; price can crash 12% between ticks.
Broker-side stops trigger in ms regardless of poll cadence.

Layered tests:
  1. BaseBroker default impls fail loudly (no silent no-op).
  2. PaperBroker.place_stop_order + simulated fills behave correctly.
  3. PaperBroker.cancel_order behaves correctly.
  4. AlpacaBroker structural — supports_broker_side_stops + StopOrderRequest
     wiring (no real network calls in tests).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from live.broker import BaseBroker  # noqa: E402
from live.paper_broker import PaperBroker  # noqa: E402


# ── BaseBroker default impls ──────────────────────────────────────────────

class _MinimalBroker(BaseBroker):
    """Concrete subclass of BaseBroker that doesn't override stops."""
    def connect(self): pass
    def disconnect(self): pass
    def get_position(self, symbol): return 0.0
    def get_account_value(self): return 0.0
    def place_order(self, symbol, action, quantity): return {}


class TestBaseBrokerDefaults:
    def test_default_supports_returns_false(self):
        b = _MinimalBroker()
        assert b.supports_broker_side_stops() is False

    def test_default_place_stop_raises(self):
        b = _MinimalBroker()
        with pytest.raises(NotImplementedError, match="does not support"):
            b.place_stop_order("AAPL", 100, 150.0)

    def test_default_cancel_raises(self):
        b = _MinimalBroker()
        with pytest.raises(NotImplementedError, match="cancel_order"):
            b.cancel_order("XYZ")


# ── PaperBroker: place + simulate + cancel ────────────────────────────────

class TestPaperBrokerStops:
    def _seed_position(self, b: PaperBroker) -> None:
        """Seed AAPL @ 150 × 100 shares so we have something to stop on."""
        b.set_price("AAPL", 150.0)
        b.place_order("AAPL", "BUY", 100, price=150.0)

    def test_supports_flag(self):
        b = PaperBroker(initial_cash=100_000)
        assert b.supports_broker_side_stops() is True

    def test_place_stop_returns_order_id(self):
        b = PaperBroker(initial_cash=100_000)
        self._seed_position(b)
        out = b.place_stop_order("AAPL", 100, 142.50)
        assert out["status"] == "accepted"
        assert out["order_id"].startswith("PAPER-STP-")
        assert out["stop_price"] == 142.50

    def test_place_stop_rejects_zero_qty(self):
        b = PaperBroker(initial_cash=100_000)
        with pytest.raises(ValueError, match="quantity"):
            b.place_stop_order("AAPL", 0, 142.50)

    def test_place_stop_rejects_zero_price(self):
        b = PaperBroker(initial_cash=100_000)
        self._seed_position(b)
        with pytest.raises(ValueError, match="stop_price"):
            b.place_stop_order("AAPL", 50, 0.0)

    def test_place_stop_rejects_qty_above_held(self):
        b = PaperBroker(initial_cash=100_000)
        self._seed_position(b)   # 100 shares
        with pytest.raises(ValueError, match="exceeds held"):
            b.place_stop_order("AAPL", 200, 140.0)

    def test_stop_does_not_fire_above_stop_price(self):
        """When last_price > stop_price the stop stays armed."""
        b = PaperBroker(initial_cash=100_000)
        self._seed_position(b)
        b.place_stop_order("AAPL", 100, 142.50)
        b.set_price("AAPL", 148.0)   # still above stop
        triggered = b._check_stops()
        assert triggered == []
        assert b.get_position("AAPL") == 100   # untouched

    def test_stop_fires_at_stop_price(self):
        """Sell-stop fires when last_price <= stop_price (NVTS scenario)."""
        b = PaperBroker(initial_cash=100_000)
        self._seed_position(b)
        cash_before = b.get_cash()
        b.place_stop_order("AAPL", 100, 142.50)
        # Price drops below stop — broker-side stop fires immediately,
        # NOT waiting for our next poll. This is the whole point.
        b.set_price("AAPL", 142.0)
        triggered = b._check_stops()
        assert len(triggered) == 1
        assert triggered[0]["symbol"] == "AAPL"
        assert triggered[0]["quantity"] == 100
        assert b.get_position("AAPL") == 0
        # Cash credited at fill price (last_price).
        assert b.get_cash() == pytest.approx(cash_before + 100 * 142.0)

    def test_stop_fires_on_gap_down(self):
        """Price gaps below stop — fill happens at the gap price."""
        b = PaperBroker(initial_cash=100_000)
        self._seed_position(b)
        b.place_stop_order("AAPL", 100, 142.50)
        # Gap down to 130 — well below stop. Real brokers fill at gap
        # (or worse on a runaway). Paper sim fills at last_price.
        b.set_price("AAPL", 130.0)
        triggered = b._check_stops()
        assert len(triggered) == 1
        assert triggered[0]["fill_price"] == 130.0

    def test_stop_clipped_when_position_partially_sold(self):
        """If we sold half the position separately, stop only fires for
        the remaining shares."""
        b = PaperBroker(initial_cash=100_000)
        self._seed_position(b)
        b.place_stop_order("AAPL", 100, 142.50)
        # Sell half manually
        b.place_order("AAPL", "SELL", 50, price=148.0)
        # Now drop below stop
        b.set_price("AAPL", 142.0)
        triggered = b._check_stops()
        assert len(triggered) == 1
        assert triggered[0]["quantity"] == 50
        assert b.get_position("AAPL") == 0

    def test_idempotent_check_when_no_stops_armed(self):
        b = PaperBroker(initial_cash=100_000)
        b.set_price("AAPL", 150.0)
        assert b._check_stops() == []

    def test_cancel_known_stop(self):
        b = PaperBroker(initial_cash=100_000)
        self._seed_position(b)
        out = b.place_stop_order("AAPL", 100, 142.50)
        assert b.cancel_order(out["order_id"]) is True
        # After cancel, no fire
        b.set_price("AAPL", 142.0)
        assert b._check_stops() == []

    def test_cancel_unknown_order_returns_false(self):
        b = PaperBroker(initial_cash=100_000)
        assert b.cancel_order("PAPER-STP-9999") is False


# ── AlpacaBroker structural (no live network calls) ───────────────────────

class TestAlpacaBrokerStructural:
    """Source-level checks — full integration test would require an
    Alpaca paper account + network; we keep these as string contracts."""

    def test_alpaca_supports_flag_set(self):
        from live.alpaca_broker import AlpacaBroker  # noqa: PLC0415
        # Don't connect — just check the method
        assert AlpacaBroker.supports_broker_side_stops(None) is True  # type: ignore

    def test_alpaca_uses_stop_order_request(self):
        src = (REPO_ROOT / "live" / "alpaca_broker.py").read_text()
        assert "StopOrderRequest" in src
        assert "TimeInForce.GTC" in src   # invariant: stops survive across days
        assert "OrderSide.SELL" in src

    def test_alpaca_account_status_check_in_stop_path(self):
        """Stop path must reuse the same ALPACA-ACCT-STATUS check as place_order
        (raise on non-ACTIVE accounts). Otherwise stops get silently rejected
        by the API and our state thinks they're armed when they're not."""
        src = (REPO_ROOT / "live" / "alpaca_broker.py").read_text()
        # Find the place_stop_order block and assert the status check is in it
        idx = src.find("def place_stop_order")
        assert idx >= 0
        end = src.find("\n    def ", idx + 1)
        block = src[idx:end if end > 0 else len(src)]
        assert "get_account()" in block, "stop path missing account-status check"
        assert 'status not in ("ACTIVE"' in block

    def test_alpaca_cancel_uses_alpaca_api(self):
        src = (REPO_ROOT / "live" / "alpaca_broker.py").read_text()
        idx = src.find("def cancel_order")
        assert idx >= 0
        end = src.find("\n    def ", idx + 1)
        block = src[idx:end if end > 0 else len(src)]
        assert "cancel_order_by_id" in block
