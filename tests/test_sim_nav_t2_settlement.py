"""AUDIT REGRESSION GUARD — Bug #C (2026-05-11)

`SimAdapter._portfolio_value` previously omitted the pending-settlement
queue pending balance. On a sell day, shares drop but `_cash` doesn't
move (proceeds queued); on settlement day the queue drains and cash
jumps. Each sell event creates phantom ±sale_amount returns in the
equity curve, inflating measured ann_vol.

Concrete observed regression: W1_maxpos08 sim showed APY=+3% with
Vol=516% / MaxDD=86% / Sharpe=1.54 — internally inconsistent.

The invariant pinned here (CLAUDE.md §5.3):
    NAV(t) ≡ free_cash(t) + pending_settle(t) + Σ(shares(t) × price(t))

All three are real claim against portfolio economic value. Skipping
pending_settle treats unsettled proceeds as "lost" until settlement then
"found".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


class TestSimNavT2SettlementGuard:
    """Invariant: SimAdapter._portfolio_value must include pending T+N queue."""

    def test_pending_t2_proceeds_included_in_nav(self):
        """Construct a minimal SimAdapter, manually populate _cash + _t2_queue
        + _pos_shares + _ohlcv, call _portfolio_value, verify result includes
        the pending balance."""
        from adapters.sim import SimAdapter
        from kernel.execution.t2_settlement import T2CashQueue

        # __new__ avoids __init__ side effects (LEAN data, scorer loads, etc.).
        # We're testing only the _portfolio_value method's accounting logic.
        sim = SimAdapter.__new__(SimAdapter)
        sim._cash = 50_000.0
        sim._pos_shares = {"AAPL": 100}   # 100 × $150 = $15,000
        sim._ohlcv = {}                    # not needed when prices given
        sim._t2_queue = T2CashQueue(settlement_days=2)

        # Simulate: sold $5,000 worth yesterday, pending settle
        today = pd.Timestamp("2025-01-15")
        sim._t2_queue.add_pending(today, 5_000.0)
        assert sim._t2_queue.pending_total() == pytest.approx(5_000.0)

        prices = {"AAPL": 150.0}
        nav = sim._portfolio_value(prices, today_ts=today)
        # Expected: 50_000 cash + 5_000 pending + 15_000 stock = 70_000
        assert nav == pytest.approx(70_000.0), (
            f"NAV must include pending settlement balance. "
            f"Got {nav}, expected 70_000."
        )

    def test_empty_t2_queue_no_change_to_nav(self):
        """Sanity: when queue is empty, NAV = cash + position MTM (unchanged
        from pre-fix behavior). This pins the no-regression direction."""
        from adapters.sim import SimAdapter
        from kernel.execution.t2_settlement import T2CashQueue

        sim = SimAdapter.__new__(SimAdapter)
        sim._cash = 80_000.0
        sim._pos_shares = {"MSFT": 50}    # 50 × $400 = 20_000
        sim._ohlcv = {}
        sim._t2_queue = T2CashQueue(settlement_days=2)
        # No add_pending — queue empty

        nav = sim._portfolio_value({"MSFT": 400.0}, today_ts=pd.Timestamp("2025-01-15"))
        assert nav == pytest.approx(100_000.0)

    def test_missing_t2_queue_attr_safe_default(self):
        """Defensive: __new__-constructed test fixtures might skip _t2_queue.
        The fix must not crash in that case."""
        from adapters.sim import SimAdapter

        sim = SimAdapter.__new__(SimAdapter)
        sim._cash = 100_000.0
        sim._pos_shares = {}
        sim._ohlcv = {}
        # Deliberately do NOT set _t2_queue

        nav = sim._portfolio_value({}, today_ts=pd.Timestamp("2025-01-15"))
        assert nav == pytest.approx(100_000.0)

    def test_pending_nan_safe_default(self):
        """If pending_total returns NaN/inf, NAV should ignore it (don't
        propagate corruption)."""
        from adapters.sim import SimAdapter
        from kernel.execution.t2_settlement import T2CashQueue

        sim = SimAdapter.__new__(SimAdapter)
        sim._cash = 100_000.0
        sim._pos_shares = {}
        sim._ohlcv = {}

        # Stub queue that returns NaN
        class StubQueue:
            def pending_total(self): return float("nan")
        sim._t2_queue = StubQueue()

        nav = sim._portfolio_value({}, today_ts=pd.Timestamp("2025-01-15"))
        # NaN pending should not corrupt NAV — fallback to cash only
        assert nav == pytest.approx(100_000.0)

    def test_nav_invariant_across_sell_and_settle(self):
        """The core regression: sell event must leave NAV unchanged
        (just relocates value cash → pending; settle moves pending → cash).

        Build a 3-bar sequence: pre-sell, sell-day, settle-day. NAV must
        be the SAME in all three (modulo the small fee/slippage which
        we mock as 0)."""
        from adapters.sim import SimAdapter
        from kernel.execution.t2_settlement import T2CashQueue

        sim = SimAdapter.__new__(SimAdapter)
        sim._pos_shares = {"AAPL": 100}     # 100 × $200 = 20_000
        sim._cash = 80_000.0
        sim._ohlcv = {}
        sim._t2_queue = T2CashQueue(settlement_days=2)

        prices_day1 = {"AAPL": 200.0}
        nav_day1 = sim._portfolio_value(prices_day1, today_ts=pd.Timestamp("2025-01-13"))
        assert nav_day1 == pytest.approx(100_000.0)

        # Sell 50 shares @ $200 = $10_000. Proceeds queued until settlement.
        sim._pos_shares["AAPL"] = 50          # shares drop
        sim._t2_queue.add_pending(pd.Timestamp("2025-01-13"), 10_000.0)
        # Cash unchanged (proceeds queued, not credited)

        nav_day1_after_sell = sim._portfolio_value(prices_day1, today_ts=pd.Timestamp("2025-01-13"))
        # MUST still be 100_000 — value just moved from shares to pending,
        # not lost. This is the regression pin.
        assert nav_day1_after_sell == pytest.approx(100_000.0), (
            "NAV invariant violated: sell relocates value, doesn't destroy it. "
            f"Got {nav_day1_after_sell}, expected 100_000."
        )

        # Day +2: drain settles
        settled = sim._t2_queue.drain(pd.Timestamp("2025-01-15"))
        assert settled == pytest.approx(10_000.0)
        sim._cash += settled

        nav_day3 = sim._portfolio_value(prices_day1, today_ts=pd.Timestamp("2025-01-15"))
        assert nav_day3 == pytest.approx(100_000.0), (
            f"NAV invariant violated on settlement day. Got {nav_day3}."
        )
