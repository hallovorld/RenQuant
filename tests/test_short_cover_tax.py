"""§1233 regression — short-cover P&L is taxed at short-term rate.

Bug found in 2026-05-14 self-audit: _apply_buy did not compute tax
when covering a short (old_shares < 0). All short P&L silently flowed
to cash without §1233 short-term tax applied. This pins the fix.
"""
from __future__ import annotations

import datetime
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _mk_adapter_with_short(short_shares: int, short_entry: float,
                            cover_price: float, *, cash: float = 100_000.0,
                            tax_cfg: dict | None = None):
    """Build a minimal SimAdapter via __new__ with an existing short
    position, ready for a cover (buy) order."""
    from adapters.sim import SimAdapter
    from kernel.exits import HoldingState
    adapter = SimAdapter.__new__(SimAdapter)
    adapter._cash = cash
    adapter._total_fees = 0.0
    adapter._exec_enabled = False  # skip slippage/fees for clean P&L
    adapter._trade_log = []
    today = pd.Timestamp("2024-06-15")
    df = pd.DataFrame({"close": [cover_price]}, index=pd.DatetimeIndex([today]))
    adapter._ohlcv = {"AAPL": df}
    adapter._holdings = {
        "AAPL": HoldingState(
            shares=-short_shares,
            entry_price=short_entry,
            entry_date=datetime.date(2024, 5, 1),
            high_watermark=short_entry,
        )
    }
    adapter._pos_shares = {"AAPL": -short_shares}
    return adapter, today


class TestShortCoverTax:

    def test_short_cover_at_profit_pays_st_tax(self):
        """Short AAPL at $150, cover at $120 → P&L = $30 × shares.
        ST tax @ 50% applied; cash debited."""
        from adapters.sim import SimAdapter
        adapter, today = _mk_adapter_with_short(
            short_shares=100, short_entry=150.0, cover_price=120.0,
        )
        ctx = SimpleNamespace(config={"tax": {
            "short_term_rate": 0.50,
            "long_term_rate": 0.32,
            "long_term_threshold_days": 365,
        }})
        cash_before = adapter._cash
        order = {"ticker": "AAPL", "shares": 100, "price": 120.0}
        adapter._apply_buy(order, today, ctx)

        # Cash flow:
        # -100×120 invest = -12,000 (buy to cover)
        # -50% × 100×30 = -1,500 ST tax
        # Net: cash_before - 12,000 - 1,500 = $86,500
        short_pnl = 100 * (150.0 - 120.0)  # $3,000
        expected_tax = short_pnl * 0.50  # $1,500
        expected_cash = cash_before - 100 * 120.0 - expected_tax
        assert abs(adapter._cash - expected_cash) < 0.01, (
            f"cash={adapter._cash}, expected {expected_cash}; "
            f"short_pnl={short_pnl}, tax={expected_tax}"
        )
        assert "AAPL" not in adapter._holdings
        assert "AAPL" not in adapter._pos_shares
        assert [e["action"] for e in adapter._trade_log] == ["short_cover"]
        assert adapter._trade_log[0]["gross_pnl"] == pytest.approx(short_pnl)
        assert adapter._trade_log[0]["tax"] == pytest.approx(expected_tax)

    def test_short_cover_at_loss_no_tax(self):
        """Short AAPL at $150, cover at $180 → loss. No tax owed.
        Cash debit = cover cost only (no tax adjustment)."""
        from adapters.sim import SimAdapter
        adapter, today = _mk_adapter_with_short(
            short_shares=100, short_entry=150.0, cover_price=180.0,
        )
        ctx = SimpleNamespace(config={"tax": {
            "short_term_rate": 0.50, "long_term_rate": 0.32,
            "long_term_threshold_days": 365,
        }})
        cash_before = adapter._cash
        order = {"ticker": "AAPL", "shares": 100, "price": 180.0}
        adapter._apply_buy(order, today, ctx)
        # Cash = cash_before - 100×180 = cash_before - 18,000 (no tax)
        expected = cash_before - 18000.0
        assert abs(adapter._cash - expected) < 0.01, (
            f"cash={adapter._cash}, expected {expected} (no tax on short loss)"
        )
        assert "AAPL" not in adapter._holdings
        assert [e["action"] for e in adapter._trade_log] == ["short_cover"]
        assert adapter._trade_log[0]["gross_pnl"] == pytest.approx(-3000.0)
        assert adapter._trade_log[0]["tax"] == pytest.approx(0.0)

    def test_short_cover_uses_st_rate_regardless_of_hold(self):
        """§1233: shorts always short-term even if held > 365 days.
        Compute_trade_tax called with hold_days=0 forces ST rate."""
        from adapters.sim import SimAdapter
        from kernel.exits import HoldingState
        adapter = SimAdapter.__new__(SimAdapter)
        adapter._cash = 100_000.0
        adapter._total_fees = 0.0
        adapter._exec_enabled = False
        adapter._trade_log = []
        # Short held > 1 year
        today = pd.Timestamp("2026-06-15")
        old_short_date = datetime.date(2025, 1, 1)  # ~530 days
        df = pd.DataFrame({"close": [80.0]}, index=pd.DatetimeIndex([today]))
        adapter._ohlcv = {"AAPL": df}
        adapter._holdings = {
            "AAPL": HoldingState(
                shares=-100, entry_price=150.0,
                entry_date=old_short_date, high_watermark=150.0,
            )
        }
        adapter._pos_shares = {"AAPL": -100}

        ctx = SimpleNamespace(config={"tax": {
            "short_term_rate": 0.50, "long_term_rate": 0.20,
            "long_term_threshold_days": 365,
        }})
        cash_before = adapter._cash
        adapter._apply_buy(
            {"ticker": "AAPL", "shares": 100, "price": 80.0}, today, ctx,
        )
        # Short P&L = (150 - 80) × 100 = $7,000; ST rate forced = 50%
        # Tax = $3,500 (not $1,400 if LT rate were applied)
        expected_tax = 7000.0 * 0.50
        expected_cash = cash_before - 100 * 80.0 - expected_tax
        assert abs(adapter._cash - expected_cash) < 0.01, (
            f"cash={adapter._cash}, expected {expected_cash}. "
            f"§1233 violated if LT rate applied (would give {7000*0.20})"
        )

    def test_partial_short_cover_keeps_remaining_short_without_long_lot(self):
        adapter, today = _mk_adapter_with_short(
            short_shares=100, short_entry=150.0, cover_price=120.0,
        )
        ctx = SimpleNamespace(config={"tax": {
            "short_term_rate": 0.50,
            "long_term_rate": 0.32,
            "long_term_threshold_days": 365,
        }})
        cash_before = adapter._cash

        adapter._apply_buy({"ticker": "AAPL", "shares": 40, "price": 120.0}, today, ctx)

        expected_tax = (150.0 - 120.0) * 40 * 0.50
        assert adapter._cash == pytest.approx(cash_before - 40 * 120.0 - expected_tax)
        assert adapter._holdings["AAPL"].shares == pytest.approx(-60.0)
        assert adapter._holdings["AAPL"].entry_price == pytest.approx(150.0)
        assert adapter._holdings["AAPL"].lots == []
        assert [e["action"] for e in adapter._trade_log] == ["short_cover"]
        assert adapter._trade_log[0]["partial"] is True

    def test_over_cover_closes_short_and_opens_clean_long_residual(self):
        adapter, today = _mk_adapter_with_short(
            short_shares=100, short_entry=150.0, cover_price=120.0,
        )
        ctx = SimpleNamespace(config={"tax": {
            "short_term_rate": 0.50,
            "long_term_rate": 0.32,
            "long_term_threshold_days": 365,
        }})

        adapter._apply_buy({"ticker": "AAPL", "shares": 125, "price": 120.0}, today, ctx)

        assert adapter._holdings["AAPL"].shares == pytest.approx(25.0)
        assert adapter._holdings["AAPL"].entry_price == pytest.approx(120.0)
        assert len(adapter._holdings["AAPL"].lots) == 1
        assert adapter._holdings["AAPL"].lots[0].shares == pytest.approx(25.0)
        assert [e["action"] for e in adapter._trade_log] == ["short_cover", "buy"]

    def test_long_buy_not_affected_no_short_tax(self):
        """Regular long buy (no existing short) doesn't trigger short tax."""
        from adapters.sim import SimAdapter
        adapter = SimAdapter.__new__(SimAdapter)
        adapter._cash = 100_000.0
        adapter._total_fees = 0.0
        adapter._exec_enabled = False
        adapter._trade_log = []
        adapter._ohlcv = {"AAPL": pd.DataFrame(
            {"close": [120.0]}, index=pd.DatetimeIndex([pd.Timestamp("2024-06-15")])
        )}
        adapter._holdings = {}  # fresh — no prior short
        adapter._pos_shares = {}

        ctx = SimpleNamespace(config={"tax": {"short_term_rate": 0.50}})
        cash_before = adapter._cash
        adapter._apply_buy(
            {"ticker": "AAPL", "shares": 50, "price": 120.0},
            pd.Timestamp("2024-06-15"), ctx,
        )
        # Fresh long: only invest deducted, no tax
        expected = cash_before - 50 * 120.0
        assert abs(adapter._cash - expected) < 0.01
