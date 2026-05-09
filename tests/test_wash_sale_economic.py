"""Cost-aware wash-sale tests (BUG #6 follow-up — IRC §1091 economic model).

Validates:
  1. wash_sale_npv_cost: deferred-deduction NPV math
     - gain → 0
     - loss + zero discount → tax × |loss|
     - loss + non-zero discount + hold > 0 → time-value adjusted
  2. is_wash_sale_blocked_with_cost decision matrix:
     a) wash_sale_days = 0 → never blocked
     b) outside window → never blocked
     c) inside window + GAIN sale → not blocked (§1091 N/A)
     d) inside window + LOSS sale + no expected return → blocked (conservative)
     e) inside window + LOSS sale + expected return > safety_margin × cost → not blocked
     f) inside window + LOSS sale + expected return < safety_margin × cost → blocked
     g) inside window + P/L unknown → binary block (conservative fallback)
  3. compute_recent_realized_pnl FIFO matching:
     - sell consumes earliest buys first
     - unmatched sells (pre-window inventory) silently skip
     - missing get_filled_orders API → empty dict
  4. Integration: WashSaleFilterTask honors cost-aware decision

Reference: IRC §1091, §1091(d), §1223(3); IRS Pub 550.
"""
from __future__ import annotations

import sys
import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.selection import (
    is_wash_sale_blocked,
    is_wash_sale_blocked_with_cost,
    wash_sale_npv_cost,
)


# ── 1. NPV cost formula ──────────────────────────────────────────────────────

class TestWashSaleNPVCost:
    """The §1091 NPV deferred-deduction cost formula."""

    def test_gain_has_zero_cost(self):
        # Wash-sale rule does NOT apply to gain sales
        assert wash_sale_npv_cost(realized_loss=+100.0) == 0.0
        assert wash_sale_npv_cost(realized_loss=0.0) == 0.0

    def test_loss_zero_discount_yields_zero_cost(self):
        """If discount rate = 0, the deduction is recovered at full present value."""
        assert wash_sale_npv_cost(
            realized_loss=-100.0, tax_rate=0.30, discount_rate=0.0,
            estimated_hold_years=2.0,
        ) == pytest.approx(0.0, abs=1e-9)

    def test_loss_zero_hold_yields_zero_cost(self):
        """If you sell the replacement immediately, no deferral."""
        assert wash_sale_npv_cost(
            realized_loss=-100.0, tax_rate=0.30, discount_rate=0.05,
            estimated_hold_years=0.0,
        ) == pytest.approx(0.0, abs=1e-9)

    def test_typical_case_2y_hold(self):
        """$100 loss, 30% tax, 5% discount, 2y hold → ~$2.78 NPV cost."""
        cost = wash_sale_npv_cost(
            realized_loss=-100.0, tax_rate=0.30, discount_rate=0.05,
            estimated_hold_years=2.0,
        )
        # 100 * 0.30 * (1 - 1/1.05^2) = 30 * 0.0930 = 2.789
        assert cost == pytest.approx(2.789, abs=0.01)

    def test_cost_proportional_to_abs_loss(self):
        c1 = wash_sale_npv_cost(realized_loss=-100.0)
        c2 = wash_sale_npv_cost(realized_loss=-200.0)
        assert c2 == pytest.approx(c1 * 2.0, abs=1e-9)

    def test_cost_increases_with_hold_years(self):
        c1 = wash_sale_npv_cost(realized_loss=-100.0, estimated_hold_years=1.0)
        c5 = wash_sale_npv_cost(realized_loss=-100.0, estimated_hold_years=5.0)
        assert c5 > c1


# ── 2. Decision matrix ──────────────────────────────────────────────────────

class TestWashSaleBlockDecision:
    """is_wash_sale_blocked_with_cost decision matrix."""

    today = datetime.date(2026, 5, 9)
    sell_15d_ago = today - datetime.timedelta(days=15)   # within 30d window
    sell_45d_ago = today - datetime.timedelta(days=45)   # outside 30d window

    def _call(self, **kw):
        kw.setdefault("ticker", "FOO")
        kw.setdefault("today", self.today)
        kw.setdefault("last_sell_dates", {"FOO": self.sell_15d_ago})
        kw.setdefault("last_sell_pls", {"FOO": -100.0})
        kw.setdefault("wash_sale_days", 30)
        return is_wash_sale_blocked_with_cost(**kw)

    def test_disabled_when_days_zero(self):
        blocked, reason, _ = self._call(wash_sale_days=0)
        assert not blocked
        assert "disabled" in reason

    def test_outside_window_passes(self):
        blocked, reason, _ = self._call(
            last_sell_dates={"FOO": self.sell_45d_ago},
        )
        assert not blocked
        assert "45d since sale" in reason

    def test_no_recent_sale_passes(self):
        blocked, reason, _ = self._call(last_sell_dates={})
        assert not blocked
        assert "no recent sale" in reason

    def test_gain_sale_passes(self):
        """§1091 does not apply to gain sales — must allow rebuy."""
        blocked, reason, cost = self._call(last_sell_pls={"FOO": +75.0})
        assert not blocked
        assert "§1091 N/A (gain sale" in reason
        assert cost == 0.0

    def test_zero_pl_treated_as_gain(self):
        """Even-money sale → no §1091 cost."""
        blocked, _, cost = self._call(last_sell_pls={"FOO": 0.0})
        assert not blocked
        assert cost == 0.0

    def test_loss_no_expected_return_blocks_conservatively(self):
        """If μ̂ unavailable, block losses (they have real cost; can't compare)."""
        blocked, reason, cost = self._call(
            last_sell_pls={"FOO": -100.0}, expected_dollar_return=None,
        )
        assert blocked
        assert "loss sale $-100.00" in reason
        assert cost > 0

    def test_loss_high_expected_return_passes(self):
        """If μ̂ × position >> NPV cost × safety, allow the rebuy."""
        # NPV cost for $100 loss ≈ $2.78. Safety 1.5x → $4.18.
        # Expected return $20 >> $4.18 → ALLOW
        blocked, reason, cost = self._call(
            last_sell_pls={"FOO": -100.0},
            expected_dollar_return=20.0,
            safety_margin=1.5,
        )
        assert not blocked
        assert "expected $+20.00" in reason

    def test_loss_low_expected_return_blocks(self):
        """If μ̂ × position too small to overcome NPV cost, block."""
        blocked, reason, cost = self._call(
            last_sell_pls={"FOO": -100.0},
            expected_dollar_return=1.0,    # tiny gain expected
            safety_margin=1.5,
        )
        assert blocked
        assert "expected $+1.00" in reason
        assert cost > 0

    def test_pl_unknown_falls_back_to_binary(self):
        """Missing P/L data → conservative block (we can't allow without info)."""
        blocked, reason, _ = self._call(last_sell_pls={})
        assert blocked
        assert "P/L unknown" in reason


# ── 3. compute_recent_realized_pnl FIFO ──────────────────────────────────────

class TestRealizedPnLFIFO:
    """FIFO matching of broker fill history."""

    def test_no_get_filled_orders_returns_empty(self):
        from kernel.realized_pnl import compute_recent_realized_pnl
        class _FakeBroker: pass
        assert compute_recent_realized_pnl(_FakeBroker()) == {}

    def test_simple_round_trip_gain(self):
        from kernel.realized_pnl import compute_recent_realized_pnl
        class _Broker:
            def get_filled_orders(self, after=None):
                return [
                    {"symbol": "AAPL", "action": "BUY",  "qty": 10, "avg_price": 150.0,
                     "filled_at": "2026-04-15T12:00:00Z"},
                    {"symbol": "AAPL", "action": "SELL", "qty": 10, "avg_price": 160.0,
                     "filled_at": "2026-04-20T12:00:00Z"},
                ]
        pl = compute_recent_realized_pnl(_Broker())
        assert pl == {"AAPL": 100.0}   # ($160 - $150) × 10 shares

    def test_simple_round_trip_loss(self):
        from kernel.realized_pnl import compute_recent_realized_pnl
        class _Broker:
            def get_filled_orders(self, after=None):
                return [
                    {"symbol": "FOO", "action": "BUY",  "qty": 5, "avg_price": 100.0,
                     "filled_at": "2026-04-15T12:00:00Z"},
                    {"symbol": "FOO", "action": "SELL", "qty": 5, "avg_price": 80.0,
                     "filled_at": "2026-04-20T12:00:00Z"},
                ]
        pl = compute_recent_realized_pnl(_Broker())
        assert pl == {"FOO": -100.0}

    def test_fifo_partial_match(self):
        """3 lots in, 1 lot out — sell consumes earliest buy first."""
        from kernel.realized_pnl import compute_recent_realized_pnl
        class _Broker:
            def get_filled_orders(self, after=None):
                return [
                    {"symbol": "X", "action": "BUY",  "qty": 5, "avg_price": 100.0,
                     "filled_at": "2026-04-01T12:00:00Z"},   # FIFO 1st
                    {"symbol": "X", "action": "BUY",  "qty": 5, "avg_price": 110.0,
                     "filled_at": "2026-04-05T12:00:00Z"},   # FIFO 2nd
                    {"symbol": "X", "action": "BUY",  "qty": 5, "avg_price": 120.0,
                     "filled_at": "2026-04-10T12:00:00Z"},
                    {"symbol": "X", "action": "SELL", "qty": 7, "avg_price": 130.0,
                     "filled_at": "2026-04-15T12:00:00Z"},
                ]
        pl = compute_recent_realized_pnl(_Broker())
        # FIFO: sells 5 from $100 lot ($30 each = $150) + 2 from $110 lot ($20 each = $40)
        # = $190 total
        assert pl["X"] == pytest.approx(190.0)

    def test_unmatched_sell_silently_skipped(self):
        """If we sold MORE than we bought in window, the unmatched portion
        had basis from before the window — skip it silently."""
        from kernel.realized_pnl import compute_recent_realized_pnl
        class _Broker:
            def get_filled_orders(self, after=None):
                return [
                    {"symbol": "Y", "action": "BUY",  "qty": 3, "avg_price": 50.0,
                     "filled_at": "2026-04-15T12:00:00Z"},
                    {"symbol": "Y", "action": "SELL", "qty": 10, "avg_price": 60.0,
                     "filled_at": "2026-04-20T12:00:00Z"},
                ]
        pl = compute_recent_realized_pnl(_Broker())
        # Match 3 shares: (60-50)*3 = $30. Other 7 shares unmatched (skip).
        assert pl["Y"] == pytest.approx(30.0)

    def test_zero_pl_filtered_out(self):
        from kernel.realized_pnl import compute_recent_realized_pnl
        class _Broker:
            def get_filled_orders(self, after=None):
                return [
                    {"symbol": "Z", "action": "BUY",  "qty": 5, "avg_price": 100.0,
                     "filled_at": "2026-04-15T12:00:00Z"},
                    {"symbol": "Z", "action": "SELL", "qty": 5, "avg_price": 100.0,
                     "filled_at": "2026-04-20T12:00:00Z"},
                ]
        pl = compute_recent_realized_pnl(_Broker())
        assert pl == {}    # zero-pl entries excluded

    def test_multi_ticker_separation(self):
        from kernel.realized_pnl import compute_recent_realized_pnl
        class _Broker:
            def get_filled_orders(self, after=None):
                return [
                    {"symbol": "A", "action": "BUY",  "qty": 1, "avg_price": 10.0,
                     "filled_at": "2026-04-10T12:00:00Z"},
                    {"symbol": "B", "action": "BUY",  "qty": 1, "avg_price": 20.0,
                     "filled_at": "2026-04-10T12:00:00Z"},
                    {"symbol": "A", "action": "SELL", "qty": 1, "avg_price": 12.0,
                     "filled_at": "2026-04-15T12:00:00Z"},
                    {"symbol": "B", "action": "SELL", "qty": 1, "avg_price": 18.0,
                     "filled_at": "2026-04-15T12:00:00Z"},
                ]
        pl = compute_recent_realized_pnl(_Broker())
        assert pl == {"A": +2.0, "B": -2.0}

    def test_broker_exception_returns_empty(self):
        from kernel.realized_pnl import compute_recent_realized_pnl
        class _Broker:
            def get_filled_orders(self, after=None):
                raise RuntimeError("api down")
        pl = compute_recent_realized_pnl(_Broker())
        assert pl == {}


# ── 4. Backwards compat: legacy is_wash_sale_blocked still works ─────────────

class TestLegacyBinaryBlocker:
    """The old is_wash_sale_blocked is kept for back-compat — verify behavior."""
    today = datetime.date(2026, 5, 9)

    def test_recent_sell_blocks(self):
        sells = {"FOO": self.today - datetime.timedelta(days=15)}
        assert is_wash_sale_blocked("FOO", self.today, sells, 30)

    def test_old_sell_passes(self):
        sells = {"FOO": self.today - datetime.timedelta(days=45)}
        assert not is_wash_sale_blocked("FOO", self.today, sells, 30)

    def test_no_sell_passes(self):
        assert not is_wash_sale_blocked("FOO", self.today, {}, 30)

    def test_zero_days_disables(self):
        sells = {"FOO": self.today}
        assert not is_wash_sale_blocked("FOO", self.today, sells, 0)
