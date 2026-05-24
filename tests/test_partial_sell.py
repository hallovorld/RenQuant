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

    def test_lot_disposed_basis_used_for_tax_not_avg_entry(self):
        """Bug 6 fix (2026-05-05 wl183 incident): tax was computed off
        weighted-avg entry_price, but FIFO/HIFO lot disposal uses a
        DIFFERENT cost basis. Pre-fix, gross_pnl = sell_shares ×
        (price − avg) — under FIFO with rising prices, oldest (cheapest)
        lots dispose first, so true realized gain > computed → tax
        UNDER-collected → cash inflated → APY/Sharpe over-reported.

        Test scenario: 2 lots — 5sh @ $100 (oldest) + 5sh @ $200. Avg
        entry = $150. Sell 5 shares @ $250 with FIFO.

          legacy (avg-cost):  gross_pnl = 5 × ($250 − $150) = $500
          fixed  (FIFO basis): gross_pnl = 5 × $250 − 5×$100 = $750

        ST tax @ 50% on $750 = $375 (true) vs $250 (legacy). Cash after
        sell:
          legacy:  0 + 5×$250 − $250 = $1000  (over-stated by $125)
          fixed:   0 + 5×$250 − $375 = $875  (correct)
        """
        import pandas as pd
        from adapters.sim import SimAdapter
        from kernel.exits import HoldingState, TaxLot, ExitSignal

        adp = SimAdapter.__new__(SimAdapter)
        # Build a 2-lot holding directly (oldest cheap, newest expensive)
        hs = HoldingState(
            entry_price=150.0,    # avg = (100*5 + 200*5)/10 = 150
            entry_date=datetime.date(2026, 1, 1),
            shares=10, high_watermark=200.0,
        )
        hs.lots = [
            TaxLot(shares=5, price=100.0, date=datetime.date(2026, 1, 1)),
            TaxLot(shares=5, price=200.0, date=datetime.date(2026, 3, 1)),
        ]
        adp._holdings        = {"NVDA": hs}
        adp._pos_shares      = {"NVDA": 10}
        adp._cash            = 0.0
        adp._last_sell_date  = {}
        adp._trade_log       = []
        adp._ohlcv           = {}

        ctx = SimpleNamespace(
            today    = datetime.date(2026, 4, 24),
            prices   = {"NVDA": 250.0},
            config   = {
                "tax": {"short_term_rate": 0.50, "long_term_rate": 0.20,
                        "long_term_threshold_days": 365},
                "rotation": {"joint_actions": {"qp_tax_lot_method": "fifo"}},
            },
            exits    = [],
            holdings = {},
        )
        sig = ExitSignal(should_exit=True, reason="kelly trim",
                          exit_type="kelly_trim", quantity=5.0)
        adp._apply_sell("NVDA", sig, pd.Timestamp("2026-04-24"), ctx)

        tl = adp._trade_log[0]
        # Tax should be on the FIFO-disposed gain ($750), not avg ($500).
        # ST tax @ 50% on $750 = $375.
        assert abs(tl["tax"] - 375.0) < 1e-6, (
            f"Bug 6 reopened: tax={tl['tax']}, expected $375 (FIFO gain "
            f"$750 × 50% ST). Pre-fix would compute $250 (avg-cost gain "
            f"$500 × 50%)."
        )
        # Cash = 5×$250 − $375 = $875
        assert abs(adp._cash - 875.0) < 1e-6, (
            f"Bug 6 reopened: cash={adp._cash}, expected $875"
        )

    def test_nan_quantity_treated_as_full_liquidation(self):
        """Bug 8 fix (2026-05-05): NaN/inf sig.quantity slipped through
        the partial-vs-full check (NaN < total → False; NaN > 0 → False),
        flowed into sell_shares = NaN, then propagated to gross_pnl =
        NaN → tax=0 (guarded) → cash += NaN → equity curve poisoned.
        Post-fix: non-finite quantity is treated as full liquidation."""
        import math
        import pandas as pd
        from adapters.sim import SimAdapter
        from kernel.exits import HoldingState, TaxLot, ExitSignal

        adp = SimAdapter.__new__(SimAdapter)
        hs = HoldingState(
            entry_price=100.0,
            entry_date=datetime.date(2026, 1, 1),
            shares=10, high_watermark=120.0,
        )
        hs.lots = [TaxLot(shares=10, price=100.0,
                            date=datetime.date(2026, 1, 1))]
        adp._holdings        = {"NVDA": hs}
        adp._pos_shares      = {"NVDA": 10}
        adp._cash            = 0.0
        adp._last_sell_date  = {}
        adp._trade_log       = []
        adp._ohlcv           = {}

        ctx = SimpleNamespace(
            today    = datetime.date(2026, 4, 24),
            prices   = {"NVDA": 110.0},
            config   = {
                "tax": {"short_term_rate": 0.50, "long_term_rate": 0.20,
                        "long_term_threshold_days": 365},
                "rotation": {"joint_actions": {"qp_tax_lot_method": "fifo"}},
            },
            exits    = [],
            holdings = {},
        )
        # Quantity = NaN — must NOT corrupt cash.
        sig_nan = ExitSignal(should_exit=True, reason="bug",
                              exit_type="kelly_trim", quantity=float("nan"))
        adp._apply_sell("NVDA", sig_nan, pd.Timestamp("2026-04-24"), ctx)

        # Cash must be finite, equity curve safe.
        assert math.isfinite(adp._cash), (
            f"Bug 8 reopened: NaN quantity corrupted cash → {adp._cash}"
        )
        # Treated as full liquidation: 10 × $110 − tax_on_($1100−$1000) ST 50%
        # = 1100 − 50 = $1050
        assert abs(adp._cash - 1050.0) < 1e-6, (
            f"NaN quantity must be treated as full sell at FIFO basis; "
            f"cash={adp._cash}, expected $1050"
        )
        tl = adp._trade_log[0]
        assert tl["partial"] is False, (
            "NaN quantity must produce a full-liquidation event"
        )
        assert tl["shares"] == 10, (
            f"shares={tl['shares']}, expected 10 (all)"
        )

    def test_commit_full_exit_logic_matches_apply_sell_on_nan(self):
        """Bug 11 fix (2026-05-05, follow-on to bug 8): commit()'s
        partial-vs-full classification must match _apply_sell's exactly.
        Pre-fix: commit() used `q is None or q <= 0 or q >= cur` (NaN
        → not full); _apply_sell post-bug-8 used isfinite (NaN → full).
        Mismatch → ghost position with shares=0 stayed in _holdings.

        Source-level check: both code paths use the same classification
        predicate. Easier to verify via inspection than full commit() run
        because commit() pulls in many adapter state fields."""
        import inspect
        from adapters.sim import SimAdapter
        src = inspect.getsource(SimAdapter)
        # Both _apply_sell and commit must use the same shared predicate.
        assert src.count("is_full_liquidate_signal") >= 2, (
            "bug 11 requires _apply_sell and commit() to share the "
            "partial/full classification predicate"
        )

    def test_pnl_pct_uses_disposed_basis_not_surviving_avg(self):
        """Bug 7 fix (2026-05-05 wl183 incident, follow-on to bug 6):
        on partial trims, hs.entry_price gets refreshed to the
        SURVIVING-lot weighted avg before pnl_pct is computed. Pre-fix,
        pnl_pct = (price − surviving_avg) / surviving_avg — that's
        unrealized P&L on the *remaining* position, not realized P&L
        on what was sold.

        Scenario: 2 lots — 5sh @ $100 (FIFO oldest) + 5sh @ $200.
        Sell 5 @ $250 under FIFO:
          disposed cost = $100/sh, realized gain = ($250 − $100) / $100 = +150%
          surviving avg = $200/sh
          pre-fix pnl_pct = (250 − 200) / 200 = +25%   ← WRONG (under-reports)
          fixed pnl_pct  = (250 − 100) / 100 = +150%   ← realized

        Win-rate impact: a profitable FIFO partial trim could be
        classified as 25% gain or worse — sometimes flipping to a loss
        if the surviving lots are recent expensive ones."""
        import pandas as pd
        from adapters.sim import SimAdapter
        from kernel.exits import HoldingState, TaxLot, ExitSignal

        adp = SimAdapter.__new__(SimAdapter)
        hs = HoldingState(
            entry_price=150.0,
            entry_date=datetime.date(2026, 1, 1),
            shares=10, high_watermark=200.0,
        )
        hs.lots = [
            TaxLot(shares=5, price=100.0, date=datetime.date(2026, 1, 1)),
            TaxLot(shares=5, price=200.0, date=datetime.date(2026, 3, 1)),
        ]
        adp._holdings        = {"NVDA": hs}
        adp._pos_shares      = {"NVDA": 10}
        adp._cash            = 0.0
        adp._last_sell_date  = {}
        adp._trade_log       = []
        adp._ohlcv           = {}

        ctx = SimpleNamespace(
            today    = datetime.date(2026, 4, 24),
            prices   = {"NVDA": 250.0},
            config   = {
                "tax": {"short_term_rate": 0.50, "long_term_rate": 0.20,
                        "long_term_threshold_days": 365},
                "rotation": {"joint_actions": {"qp_tax_lot_method": "fifo"}},
            },
            exits    = [],
            holdings = {},
        )
        sig = ExitSignal(should_exit=True, reason="kelly trim",
                          exit_type="kelly_trim", quantity=5.0)
        adp._apply_sell("NVDA", sig, pd.Timestamp("2026-04-24"), ctx)

        tl = adp._trade_log[0]
        # FIFO disposed basis = $100/sh, gain = ($250-$100)/$100 = 1.50
        assert abs(tl["pnl_pct"] - 1.50) < 1e-6, (
            f"Bug 7 reopened: pnl_pct={tl['pnl_pct']}, expected +1.50 "
            f"(realized FIFO gain). Pre-fix would compute +0.25 against "
            f"surviving $200 avg."
        )

    def test_full_exit_detection_logic(self):
        """The commit() loop pops only when quantity is None or ≥ current shares.

        This pins the pop-decision logic directly (a thin slice of commit()
        that doesn't depend on the rest of the adapter's state).
        """
        from kernel.exits import ExitSignal
        from kernel.pipeline.task_execution import is_full_liquidate_signal

        assert is_full_liquidate_signal(ExitSignal(True, "r", "stop_loss"), 10)
        assert is_full_liquidate_signal(
            ExitSignal(True, "r", "kelly_trim", quantity=0.0), 10,
        )
        assert is_full_liquidate_signal(
            ExitSignal(True, "r", "kelly_trim", quantity=10.0), 10,
        )
        assert is_full_liquidate_signal(
            ExitSignal(True, "r", "kelly_trim", quantity=15.0), 10,
        )
        assert not is_full_liquidate_signal(
            ExitSignal(True, "r", "kelly_trim", quantity=3.0), 10,
        )
        assert not is_full_liquidate_signal(
            ExitSignal(True, "r", "kelly_trim", quantity=9.99), 10,
        )
