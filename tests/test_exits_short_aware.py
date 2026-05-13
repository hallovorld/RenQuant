"""Long-Short Phase 2A — stop-loss is short-aware (price moves UP → exit short).

Pinned invariants:
1. LONG position (shares > 0): exit when price DROPS by stop_pct from entry.
2. SHORT position (shares < 0): exit when price RISES by stop_pct from entry.
3. Threshold magnitude is the same — only the direction flips.
4. Reason string contains "[SHORT]" for short stops (for ntfy debugging).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _make_state(shares, entry_price, entry_date=None):
    from kernel.exits import HoldingState
    return HoldingState(
        shares=shares,
        entry_price=entry_price,
        entry_date=entry_date or datetime.date(2025, 1, 1),
        high_watermark=entry_price,
    )


class TestShortAwareStopLoss:

    def test_long_position_stop_on_price_drop(self):
        """Long: stop fires when price drops 15% from entry."""
        from kernel.exits import check_stop_loss
        state = _make_state(shares=+100, entry_price=100.0)
        # Drop 15% → loss = +15% → fires
        sig = check_stop_loss(current_price=85.0, state=state, stop_pct=0.15)
        assert sig.should_exit
        assert "[SHORT]" not in sig.reason

    def test_long_position_no_stop_on_price_rise(self):
        """Long: no stop on price rising."""
        from kernel.exits import check_stop_loss
        state = _make_state(shares=+100, entry_price=100.0)
        sig = check_stop_loss(current_price=115.0, state=state, stop_pct=0.15)
        assert not sig.should_exit

    def test_short_position_stop_on_price_rise(self):
        """Short: stop fires when price RISES 15% from entry (opposite of long)."""
        from kernel.exits import check_stop_loss
        state = _make_state(shares=-100, entry_price=100.0)
        # Rise 15% → loss for short = +15% → fires
        sig = check_stop_loss(current_price=115.0, state=state, stop_pct=0.15)
        assert sig.should_exit
        assert "[SHORT]" in sig.reason

    def test_short_position_no_stop_on_price_drop(self):
        """Short: no stop on price dropping (drop = profit for short)."""
        from kernel.exits import check_stop_loss
        state = _make_state(shares=-100, entry_price=100.0)
        sig = check_stop_loss(current_price=85.0, state=state, stop_pct=0.15)
        assert not sig.should_exit

    def test_threshold_magnitude_symmetric(self):
        """Same stop_pct fires at same magnitude move, opposite direction."""
        from kernel.exits import check_stop_loss
        long_state = _make_state(shares=+100, entry_price=100.0)
        short_state = _make_state(shares=-100, entry_price=100.0)
        # 10% adverse moves
        long_fires = check_stop_loss(90.0, long_state, stop_pct=0.10).should_exit
        short_fires = check_stop_loss(110.0, short_state, stop_pct=0.10).should_exit
        assert long_fires
        assert short_fires
        # 9% adverse moves (just below threshold)
        long_no = check_stop_loss(91.5, long_state, stop_pct=0.10).should_exit
        short_no = check_stop_loss(108.5, short_state, stop_pct=0.10).should_exit
        assert not long_no
        assert not short_no
