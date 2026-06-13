"""Tests for NaN/inf guards in broker + adapter price/order paths.

Bugs found 2026-05-09 evening audit (RU-PRICE-1, RU-PRICE-2, PB-NaN-1):
- runner.py: micro-qty (e.g. 1e-7 fractional shares) passed `qty > 0`
  guard then `mkt / qty` produced inflated prices
- runner.py: NaN close from `df["close"].iloc[-1]` propagated silently
  into ctx.prices
- paper_broker.place_order: NaN quantity propagated through invest →
  cash → positions, corrupting all future get_account_value calls

These tests pin the invariants — any regression in NaN/dust handling
trips the AUDIT REGRESSION GUARD.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))   # so 'from live.paper_broker import PaperBroker' works


# ── PaperBroker NaN/inf guards ──────────────────────────────────────────────

class TestPaperBrokerNaNGuards:

    def test_nan_quantity_rejected(self):
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        result = b.place_order("AAPL", "BUY", float("nan"), price=100.0)
        assert result["status"] == "rejected", \
            f"NaN quantity must be rejected, not silently propagated. Got: {result}"
        assert "non-finite" in result.get("reject_reason", "").lower() or \
               "non-finite" in result.get("reject_reason", ""), \
            f"Reject reason should mention 'non-finite', got: {result.get('reject_reason')}"
        # State must NOT be corrupted
        assert math.isfinite(b.get_cash()), "Cash corrupted by NaN order"
        assert b.get_position("AAPL") == 0, "Position corrupted by NaN order"

    def test_inf_quantity_rejected(self):
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        result = b.place_order("AAPL", "BUY", float("inf"), price=100.0)
        assert result["status"] == "rejected"
        assert math.isfinite(b.get_cash())

    def test_negative_quantity_rejected(self):
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        result = b.place_order("AAPL", "BUY", -10, price=100.0)
        assert result["status"] == "rejected"

    def test_zero_quantity_rejected(self):
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        result = b.place_order("AAPL", "BUY", 0, price=100.0)
        assert result["status"] == "rejected"

    def test_nan_price_does_not_corrupt_cash(self):
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        result = b.place_order("AAPL", "BUY", 10, price=float("nan"))
        # Order may proceed (BaseBroker contract) but cash must not corrupt
        assert math.isfinite(b.get_cash()), \
            f"Cash became non-finite after NaN-price order: {b.get_cash()}"
        # And subsequent get_account_value must be finite
        assert math.isfinite(b.get_account_value()), \
            f"Account value corrupted by NaN-price order"

    def test_valid_order_succeeds(self):
        """Sanity check — guards don't break the happy path."""
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        result = b.place_order("AAPL", "BUY", 10, price=100.0)
        assert result["status"] == "filled"
        assert b.get_position("AAPL") == 10
        assert b.get_cash() == 100_000 - 1000  # 100k - 10*100


# ── Runner price extraction guards ──────────────────────────────────────────

class TestRunnerPriceExtractionGuards:
    """The runner builds ctx.prices from broker positions (mkt/qty) and
    OHLCV close. Both paths now have isfinite + sanity-bound guards."""

    def test_runner_micro_qty_dust_does_not_inflate_price(self):
        """Pre-fix: pos with qty=1e-7, mkt=$100 → price=$1e9. AUDIT REGRESSION
        GUARD: any qty < 0.5 is treated as dust and yields NO price.

        The price-computation logic moved to adapters/runner_prices.py
        (S2 make_context decomposition); the guard invariant is scanned
        across the runner adapter package so it follows the relocation."""
        adapters = (REPO / "backtesting" / "renquant_104" / "adapters")
        # The guard now lives in runner_prices.py (S2 make_context
        # decomposition); scan the runner adapter package for it.
        block = "".join(
            p.read_text() for p in
            [adapters / "runner.py", adapters / "runner_prices.py"])
        assert "RU-PRICE-1" in block
        # Must have isfinite check + qty floor
        assert "isfinite(qty)" in block, \
            "AUDIT REGRESSION (RU-PRICE-1): runner.py no longer guards qty " \
            "with isfinite — micro-qty dust will inflate prices."
        assert "qty >= 0.5" in block or "qty > 0.5" in block, \
            "AUDIT REGRESSION (RU-PRICE-1): runner.py no longer floors qty " \
            "to >= 0.5 share — sub-share dust will produce inflated prices."

    def test_runner_ohlcv_close_isfinite_guard(self):
        """Pre-fix: NaN close on delisted ticker silently propagated to
        ctx.prices. Now: isfinite + > 0 guard."""
        src = (REPO / "backtesting" / "renquant_104" / "adapters"
               / "runner.py").read_text()
        # Find the OHLCV close block
        idx = src.find("Fill prices from OHLCV last close")
        assert idx > 0
        block = src[idx:idx + 800]
        assert "isfinite(close_val)" in block, \
            "AUDIT REGRESSION (RU-PRICE-2): runner.py OHLCV close path no " \
            "longer guards with isfinite — NaN closes will propagate to " \
            "ctx.prices and silently corrupt downstream Kelly/HWM calcs."
