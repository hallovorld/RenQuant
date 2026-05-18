"""Tests for kernel/execution/smart_orders.py (roadmap C4 / item #8 VWAP).

Per CLAUDE.md §5.13.2, this module is dead until prod imports it. Wiring
is GATED on these tests + a separate integration test in alpaca_broker.
This file pins the public-API math.

References tested:
  - Almgren-Chriss 2000 §3 (uniform-volume even-spacing)
  - Almgren-Chriss 2000 §4 (max 1% ADV stealth threshold)
  - Cont-Stoikov-Talreja 2010 §3 (limit-order price monotonic in fill)
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.execution.smart_orders import (  # noqa: E402
    ChildOrder, compute_limit_price, slice_into_children,
    child_arrival_schedule, plan_execution,
)


class TestComputeLimitPrice:
    """Half-spread pricing per Cont-Stoikov-Talreja 2010."""

    def test_passive_buy_sits_at_bid(self):
        # aggressiveness=0 → buy at bid (mid - half_spread)
        # spread=20bps → half_spread = 10bps = 0.001
        p = compute_limit_price("BUY", mid=100.0, spread_bps=20.0,
                                aggressiveness=0.0)
        assert p == pytest.approx(99.90, abs=0.01)

    def test_aggressive_buy_crosses_to_ask(self):
        # aggressiveness=1 → buy at ask
        p = compute_limit_price("BUY", mid=100.0, spread_bps=20.0,
                                aggressiveness=1.0)
        assert p == pytest.approx(100.10, abs=0.01)

    def test_neutral_buy_at_mid(self):
        # aggressiveness=0.5 → buy at mid
        p = compute_limit_price("BUY", mid=100.0, spread_bps=20.0,
                                aggressiveness=0.5)
        assert p == pytest.approx(100.00, abs=0.01)

    def test_sell_mirror(self):
        # SELL: aggressiveness=0 sits at ask, =1 crosses to bid
        p0 = compute_limit_price("SELL", mid=100.0, spread_bps=20.0,
                                 aggressiveness=0.0)
        p1 = compute_limit_price("SELL", mid=100.0, spread_bps=20.0,
                                 aggressiveness=1.0)
        assert p0 == pytest.approx(100.10, abs=0.01)
        assert p1 == pytest.approx(99.90, abs=0.01)

    def test_zero_spread_returns_mid(self):
        for side in ("BUY", "SELL"):
            for agg in (0.0, 0.5, 1.0):
                p = compute_limit_price(side, mid=200.0, spread_bps=0.0,
                                        aggressiveness=agg)
                assert p == 200.00

    def test_rounds_to_cent(self):
        # 4-decimal price → snapped to 2 decimals
        p = compute_limit_price("BUY", mid=123.456, spread_bps=10.0,
                                aggressiveness=0.3)
        assert p == round(p, 2)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="mid"):
            compute_limit_price("BUY", mid=0, spread_bps=10, aggressiveness=0.5)
        with pytest.raises(ValueError, match="aggressiveness"):
            compute_limit_price("BUY", mid=100, spread_bps=10, aggressiveness=1.5)
        with pytest.raises(ValueError, match="spread_bps"):
            compute_limit_price("BUY", mid=100, spread_bps=-1, aggressiveness=0.5)
        with pytest.raises(ValueError, match="side"):
            compute_limit_price("LONG", mid=100, spread_bps=10, aggressiveness=0.5)


class TestSliceIntoChildren:
    """Almgren-Chriss §4: max 1% ADV per child for stealth."""

    def test_small_order_single_child(self):
        # 100 shares, 1M ADV, 1% cap → 10k cap → single child of 100
        children = slice_into_children(100, adv_shares=1_000_000, max_pct_adv=0.01)
        assert children == [100]

    def test_large_order_splits_into_n_children(self):
        # 50k shares, 1M ADV, 1% cap = 10k → 5 children of 10k each
        children = slice_into_children(50_000, adv_shares=1_000_000,
                                       max_pct_adv=0.01)
        assert sum(children) == 50_000
        assert all(c <= 10_000 for c in children)
        assert len(children) == 5

    def test_remainder_on_first_child(self):
        # 23k shares, 1M ADV, 1% = 10k → ceil(23/10) = 3 children
        # base = 23000 // 3 = 7666, rem = 23000 - 7666*3 = 23000-22998 = 2
        # first = 7668, rest = 7666
        children = slice_into_children(23_000, adv_shares=1_000_000,
                                       max_pct_adv=0.01)
        assert sum(children) == 23_000
        assert children[0] >= children[1]

    def test_zero_or_neg_parent_returns_empty(self):
        assert slice_into_children(0, 1_000_000) == []
        assert slice_into_children(-5, 1_000_000) == []

    def test_no_adv_info_returns_single_child(self):
        # adv=0 or max_pct=0 → submit as single child (no slicing possible)
        assert slice_into_children(1000, adv_shares=0) == [1000]
        assert slice_into_children(1000, adv_shares=100, max_pct_adv=0.0) == [1000]


class TestChildArrivalSchedule:
    """Almgren-Chriss §3 uniform-volume even-spacing."""

    def test_zero_children_empty(self):
        assert child_arrival_schedule(0, 1000) == []

    def test_one_child_immediate(self):
        assert child_arrival_schedule(1, 1000) == [0.0]

    def test_n_children_evenly_spaced_over_horizon(self):
        # 5 children over 1800s → [0, 450, 900, 1350, 1800]
        t = child_arrival_schedule(5, 1800)
        assert len(t) == 5
        assert t[0] == 0.0
        assert t[-1] == pytest.approx(1800)
        # Equal spacing
        diffs = [t[i+1] - t[i] for i in range(4)]
        assert all(d == pytest.approx(diffs[0]) for d in diffs)

    def test_negative_inputs_raise(self):
        with pytest.raises(ValueError):
            child_arrival_schedule(-1, 100)
        with pytest.raises(ValueError):
            child_arrival_schedule(3, -100)


class TestPlanExecution:
    """End-to-end: parent + ADV + horizon → list of ChildOrder."""

    def test_composes_slicing_and_scheduling(self):
        plan = plan_execution(
            side="BUY",
            parent_qty=30_000,
            mid=50.0,
            spread_bps=10.0,
            adv_shares=1_000_000,   # 1% cap = 10k
            horizon_seconds=900,    # 15 min
            max_pct_adv=0.01,
        )
        # 30k / 10k = 3 children
        assert len(plan) == 3
        assert sum(c.quantity for c in plan) == 30_000
        # Even spacing
        assert plan[0].arrival_offset_seconds == 0.0
        assert plan[-1].arrival_offset_seconds == pytest.approx(900)

    def test_small_order_single_immediate_child(self):
        plan = plan_execution(
            side="BUY", parent_qty=100, mid=50.0, spread_bps=10.0,
            adv_shares=1_000_000, horizon_seconds=900,
        )
        assert len(plan) == 1
        assert plan[0] == ChildOrder(quantity=100, arrival_offset_seconds=0.0)

    def test_zero_parent_returns_empty(self):
        plan = plan_execution(
            side="BUY", parent_qty=0, mid=50.0, spread_bps=10.0,
            adv_shares=1_000_000,
        )
        assert plan == []


class TestWiringStatus:
    """CLAUDE.md §5.13.2 — code is dead until prod imports it. This
    test exists to mark the current wiring state explicitly so the next
    audit catches if it stays unwired indefinitely."""

    def test_module_not_yet_wired_to_prod(self):
        """As of 2026-05-18 ship: smart_orders is a self-contained
        helper. live/alpaca_broker.py still uses MarketOrderRequest
        directly (line ~145). Wiring is gated on (a) integration
        test with mock Alpaca SDK, (b) ADV data source for live
        symbols, (c) cancel-on-stop interaction logic. Mark as TODO
        for next session."""
        import subprocess
        result = subprocess.run(
            ["grep", "-rln", "smart_orders", "live/", "adapters/",
             "backtesting/renquant_104/adapters/"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        files = [f for f in result.stdout.strip().split("\n") if f]
        # If any of these reference smart_orders, wiring has been done —
        # delete this test in that commit.
        assert not files, (
            f"smart_orders is wired into {files} — delete this test"
        )
