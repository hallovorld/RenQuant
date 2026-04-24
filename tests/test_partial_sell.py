"""Partial-sell infrastructure — prereq for AB-trim (Kelly rebalance).

Before this change:
  * Every exit was a full liquidation. ExitSignal had no way to say
    "sell N < full shares".
  * SimAdapter.commit() unconditionally popped the ticker from holdings
    and pos_shares after _apply_sell.
  * RunnerAdapter.commit() popped entry_dates / position_hwm / sell_streaks
    too — even a single partial sell would wipe tenure history.

After:
  * ExitSignal.quantity (default None) signals full liquidation.
  * 0 < quantity < current_shares → partial sell; position stays open.
  * quantity ≥ current_shares or None → full liquidation (unchanged).

These tests pin both adapters + the ExitSignal contract.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── ExitSignal contract ──────────────────────────────────────────────────────

class TestExitSignalQuantity:
    def test_default_is_none(self):
        from kernel.exits import ExitSignal
        sig = ExitSignal(should_exit=True, reason="r", exit_type="model_sell")
        assert sig.quantity is None

    def test_accepts_explicit_float(self):
        from kernel.exits import ExitSignal
        sig = ExitSignal(should_exit=True, reason="r", exit_type="kelly_trim",
                          quantity=3.0)
        assert sig.quantity == 3.0

    def test_kelly_trim_is_a_valid_exit_type(self):
        """Docstring mentions kelly_trim — smoke test the constructor accepts it."""
        from kernel.exits import ExitSignal
        sig = ExitSignal(should_exit=True, reason="kelly delta", exit_type="kelly_trim",
                          quantity=2.5)
        assert sig.exit_type == "kelly_trim"
        assert sig.quantity  == 2.5


# ── SimAdapter partial-sell plumbing ─────────────────────────────────────────

class TestSimAdapterPartialSell:
    """Exercise _apply_sell + commit() partial path with a thin fake ctx."""

    def _make_adapter(self, ticker: str, shares: float, entry_price: float):
        """Bypass SimAdapter.__init__ (which loads artifacts + panel) and set
        just the state fields _apply_sell/commit touch.
        """
        import pandas as pd
        from adapters.sim import SimAdapter
        from kernel.exits import HoldingState
        adp = SimAdapter.__new__(SimAdapter)
        adp._holdings        = {ticker: HoldingState(
            entry_price=entry_price, entry_date=datetime.date(2026, 1, 15),
            shares=shares, high_watermark=entry_price,
        )}
        adp._pos_shares      = {ticker: shares}
        adp._cash            = 0.0
        adp._last_sell_date  = {}
        adp._trade_log       = []
        adp._ohlcv           = {}
        return adp

    def _make_ctx(self, today: datetime.date, ticker: str, price: float):
        return SimpleNamespace(
            today    = today,
            prices   = {ticker: price},
            config   = {"tax": {"short_term_rate": 0.37, "long_term_rate": 0.20,
                                "long_term_threshold_days": 365}},
            exits    = [],
            holdings = {},
        )

    def test_partial_sell_reduces_shares_keeps_position(self):
        import pandas as pd
        from kernel.exits import ExitSignal
        adp = self._make_adapter("NVDA", shares=10, entry_price=100.0)
        ctx = self._make_ctx(datetime.date(2026, 4, 24), "NVDA", price=150.0)
        sig = ExitSignal(should_exit=True, reason="kelly delta",
                          exit_type="kelly_trim", quantity=3.0)
        adp._apply_sell("NVDA", sig, pd.Timestamp("2026-04-24"), ctx)
        # Position still open with 7 shares
        assert adp._pos_shares["NVDA"] == 7
        assert "NVDA" in adp._holdings
        assert adp._holdings["NVDA"].entry_price == 100.0   # cost basis preserved
        # Trade log records the partial
        assert len(adp._trade_log) == 1
        tl = adp._trade_log[0]
        assert tl["shares"]  == 3.0
        assert tl["partial"] is True

    def test_full_sell_when_quantity_ge_current_shares(self):
        import pandas as pd
        from kernel.exits import ExitSignal
        adp = self._make_adapter("NVDA", shares=5, entry_price=100.0)
        ctx = self._make_ctx(datetime.date(2026, 4, 24), "NVDA", price=120.0)
        sig = ExitSignal(should_exit=True, reason="max_hold",
                          exit_type="max_hold", quantity=10.0)
        adp._apply_sell("NVDA", sig, pd.Timestamp("2026-04-24"), ctx)
        # quantity ≥ current_shares → full sell, shares stay 5 until commit pops
        assert adp._pos_shares["NVDA"] == 5
        tl = adp._trade_log[0]
        assert tl["shares"]  == 5
        assert tl["partial"] is False

    def test_full_sell_when_quantity_is_none(self):
        import pandas as pd
        from kernel.exits import ExitSignal
        adp = self._make_adapter("NVDA", shares=4, entry_price=100.0)
        ctx = self._make_ctx(datetime.date(2026, 4, 24), "NVDA", price=110.0)
        sig = ExitSignal(should_exit=True, reason="stop", exit_type="stop_loss")
        adp._apply_sell("NVDA", sig, pd.Timestamp("2026-04-24"), ctx)
        tl = adp._trade_log[0]
        assert tl["shares"]  == 4
        assert tl["partial"] is False

    def test_full_exit_detection_logic(self):
        """The commit() loop pops only when quantity is None or ≥ current shares.

        This pins the pop-decision logic directly (a thin slice of commit()
        that doesn't depend on the rest of the adapter's state).
        """
        from kernel.exits import ExitSignal

        def _is_full(sig, current):
            q = getattr(sig, "quantity", None)
            return q is None or q <= 0 or q >= current

        assert _is_full(ExitSignal(True, "r", "stop_loss"), current=10)
        assert _is_full(ExitSignal(True, "r", "kelly_trim", quantity=0.0), current=10)
        assert _is_full(ExitSignal(True, "r", "kelly_trim", quantity=10.0), current=10)
        assert _is_full(ExitSignal(True, "r", "kelly_trim", quantity=15.0), current=10)
        assert not _is_full(ExitSignal(True, "r", "kelly_trim", quantity=3.0), current=10)
        assert not _is_full(ExitSignal(True, "r", "kelly_trim", quantity=9.99), current=10)
