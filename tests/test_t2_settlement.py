"""Regression tests for T+N cash-settlement queue.

Pins:
- default sell on T → drain on T+1 returns proceeds
- explicit legacy T+2 remains supported
- multi-pending entries drain in date order
- NYSE holidays correctly skip (Christmas 2024 case)

Per CLAUDE.md §5.13.3 — names the invariant. Per §5.13.5 — single source
of truth for T+N settlement arithmetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.execution.t2_settlement import T2CashQueue, PendingCashEntry  # noqa: E402


class TestTnBasicSettlement:
    """AUDIT REGRESSION GUARD — sell proceeds materialize at T+N, not T+0.

    Pre-fix sim ran with T+0 settlement, inflating intraweek buying
    power → APY inflated 0.5-1.5%/yr through compounding. This is the
    settlement-day invariant.
    """

    def test_default_sell_on_monday_settles_tuesday(self):
        q = T2CashQueue()
        # 2025-04-07 is Monday; current default T+1 = Tuesday 2025-04-08
        sale = pd.Timestamp("2025-04-07")
        q.add_pending(sale, 1_000.0)
        # T+0: no cash
        assert q.drain(sale) == 0.0
        # T+1 (Tue): drained
        settled = q.drain(pd.Timestamp("2025-04-08"))
        assert settled == pytest.approx(1_000.0)
        # No double-settle
        assert q.drain(pd.Timestamp("2025-04-09")) == 0.0

    def test_explicit_legacy_t2_sell_on_monday_settles_wednesday(self):
        q = T2CashQueue(settlement_days=2)
        sale = pd.Timestamp("2025-04-07")
        q.add_pending(sale, 1_000.0)
        assert q.drain(sale) == 0.0
        assert q.drain(pd.Timestamp("2025-04-08")) == 0.0
        assert q.drain(pd.Timestamp("2025-04-09")) == pytest.approx(1_000.0)

    def test_drain_idempotent_on_quiet_day(self):
        q = T2CashQueue()
        d = pd.Timestamp("2025-04-15")
        assert q.drain(d) == 0.0
        assert q.drain(d) == 0.0  # idempotent

    def test_pending_total_tracks_unsettled(self):
        q = T2CashQueue()
        q.add_pending(pd.Timestamp("2025-04-07"), 500.0)
        q.add_pending(pd.Timestamp("2025-04-08"), 300.0)
        assert q.pending_total() == pytest.approx(800.0)
        q.drain(pd.Timestamp("2025-04-08"))  # settles the first
        assert q.pending_total() == pytest.approx(300.0)


class TestT2MultiEntryOrdering:
    """Multi-pending entries drain in settle-date order."""

    def test_multiple_sales_drain_in_date_order(self):
        q = T2CashQueue(settlement_days=2)
        # Three sales on consecutive days
        q.add_pending(pd.Timestamp("2025-04-07"), 100.0)  # settles 4/9
        q.add_pending(pd.Timestamp("2025-04-08"), 200.0)  # settles 4/10
        q.add_pending(pd.Timestamp("2025-04-09"), 400.0)  # settles 4/11
        assert len(q) == 3
        # Drain on 4/9 → only first settles
        s1 = q.drain(pd.Timestamp("2025-04-09"))
        assert s1 == pytest.approx(100.0)
        assert len(q) == 2
        # Drain on 4/11 → remaining two
        s2 = q.drain(pd.Timestamp("2025-04-11"))
        assert s2 == pytest.approx(600.0)
        assert len(q) == 0

    def test_drain_at_far_future_clears_all(self):
        q = T2CashQueue(settlement_days=2)
        for i, amt in enumerate([100.0, 200.0, 300.0]):
            q.add_pending(pd.Timestamp("2025-04-01") + pd.Timedelta(days=i), amt)
        total = q.drain(pd.Timestamp("2026-01-01"))
        assert total == pytest.approx(600.0)
        assert len(q) == 0


class TestT2HolidaySkip:
    """AUDIT REGRESSION GUARD — NYSE holidays skip correctly.

    Christmas 2024: Wed 12/25 is a market holiday.
    - Sell Mon 12/23 → T+2 = Fri 12/27 (skip Wed 12/25)
    - Sell Tue 12/24 → T+2 = Mon 12/30 (skip Wed 12/25 + weekend)

    Naïve +2 calendar-day arithmetic gets these wrong by 1-3 days,
    breaking intra-week cash math during holiday weeks.
    """

    def test_christmas_eve_settles_dec_27(self):
        # Sell Tuesday 2024-12-24 (Christmas Eve): T+1 = 12/26 (Thu, skip
        # Wed 12/25 holiday), T+2 = 12/27 (Fri). Naïve calendar arithmetic
        # would have predicted 12/26 (off by one). NYSE-aware: 12/27.
        q = T2CashQueue(settlement_days=2)
        q.add_pending(pd.Timestamp("2024-12-24"), 1_000.0)
        assert q.drain(pd.Timestamp("2024-12-26")) == 0.0  # only T+1
        settled = q.drain(pd.Timestamp("2024-12-27"))
        assert settled == pytest.approx(1_000.0)

    def test_dec_23_settles_dec_26(self):
        # Sell Monday 2024-12-23: T+1 = 12/24 (Tue), T+2 = 12/26 (Thu,
        # skipping Wed 12/25 holiday). Naïve +2 calendar would predict
        # 12/25 (Wed) which is closed.
        q = T2CashQueue(settlement_days=2)
        q.add_pending(pd.Timestamp("2024-12-23"), 500.0)
        assert q.drain(pd.Timestamp("2024-12-24")) == 0.0  # T+1
        settled = q.drain(pd.Timestamp("2024-12-26"))
        assert settled == pytest.approx(500.0)


class TestT2DefensiveGuards:
    """§5.13.11 — NaN / negative amounts silently dropped."""

    def test_nan_amount_dropped(self):
        q = T2CashQueue()
        q.add_pending(pd.Timestamp("2025-04-07"), float("nan"))
        assert len(q) == 0
        assert q.pending_total() == 0.0

    def test_zero_or_negative_amount_dropped(self):
        q = T2CashQueue()
        q.add_pending(pd.Timestamp("2025-04-07"), 0.0)
        q.add_pending(pd.Timestamp("2025-04-07"), -500.0)
        assert len(q) == 0
