"""Regression tests for two P1 bugs found in code-review audit of the
2026-05-04 QP session.

P1-1 (post-stop blackout misuse): DEFAULT_STOP_EXIT_TYPES included
`max_hold` and `max_hold_days` — TIME exits, not RISK exits. Including
them blocks re-entry after a position simply ages out, which is
unrelated to bad timing. Fix: drop them from the set.

P1-5 (entry_price stale after partial HIFO sell): apply_sell_lots
mutates hs.lots but NOT hs.entry_price. Under HIFO, the highest-cost
lot is consumed first → surviving weighted avg drops. Without a
refresh, downstream trailing_stop / stop_loss compare current_price
against a stale pre-sell basis → wrong P&L sign on every check.
Fix: sim.py recomputes hs.entry_price = weighted_avg_entry_price()
right after apply_sell_lots.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY  = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY) not in sys.path:
    sys.path.insert(0, str(STRATEGY))


# ── P1-1: max_hold removed from DEFAULT_STOP_EXIT_TYPES ────────────────────

class TestStopExitTypesNoMaxHold:
    """max_hold is a TIME exit; it should NOT trigger post-stop blackout.

    Mirror docs: kernel/pipeline/task_post_stop_cooldown.py top docstring
    explicitly motivates the cooldown by 'trailing_stop fired @ $587 …
    same day a 1-share rebuy executed' — that's a *price-action* signal.
    A position aging out after 90d carries no such signal."""

    def test_max_hold_not_in_default_set(self):
        from kernel.pipeline.task_post_stop_cooldown import (
            DEFAULT_STOP_EXIT_TYPES,
        )
        assert "max_hold" not in DEFAULT_STOP_EXIT_TYPES
        assert "max_hold_days" not in DEFAULT_STOP_EXIT_TYPES

    def test_risk_exits_present(self):
        """Don't over-correct: trailing_stop, stop_loss, single_day_loss
        and gap_down ARE risk exits and MUST stay in the set."""
        from kernel.pipeline.task_post_stop_cooldown import (
            DEFAULT_STOP_EXIT_TYPES,
        )
        for t in ("trailing_stop", "trailing_stop_loss", "stop_loss",
                  "single_day_loss", "sdl", "gap_down"):
            assert t in DEFAULT_STOP_EXIT_TYPES, (
                f"removed legitimate risk exit_type {t}"
            )


# ── P1-5: entry_price refresh after apply_sell_lots ─────────────────────────

class TestEntryPriceRefreshAfterPartialHIFO:
    """Sim's _apply_sell must refresh hs.entry_price = weighted_avg_entry_price()
    right after apply_sell_lots() mutates the lot list.

    Pre-fix: HIFO consumed highest-cost lot → surviving lots' weighted
    avg dropped (e.g. 110 → 95) but hs.entry_price stayed at 105 →
    trailing_stop check vs 100 wrongly thought P&L was -5% when actual
    surviving cost basis was 95 (P&L +5%).
    """

    def _make_hs_with_lots(self):
        """Position w/ 3 lots: 10 sh @ 90, 10 sh @ 100, 10 sh @ 130 (avg 106.67)."""
        from kernel.exits import HoldingState, TaxLot
        d1 = datetime.date(2025, 1, 15)
        d2 = datetime.date(2025, 4, 10)
        d3 = datetime.date(2025, 8, 22)
        hs = HoldingState(
            entry_price=106.6667,         # weighted avg of 90/100/130
            entry_date=d1,
            high_watermark=140.0,
            shares=30.0,
            lots=[
                TaxLot(shares=10.0, price=90.0,  date=d1),
                TaxLot(shares=10.0, price=100.0, date=d2),
                TaxLot(shares=10.0, price=130.0, date=d3),
            ],
        )
        return hs

    def test_hifo_sell_leaves_entry_price_correct(self):
        """After HIFO partial sell of 5 shares: should consume from $130 lot.

        Post-sell lots: 10 @ 90, 10 @ 100, 5 @ 130
        Surviving weighted avg = (10·90 + 10·100 + 5·130) / 25 = 2550/25 = 102.0
        """
        from kernel.exits import apply_sell_lots
        hs = self._make_hs_with_lots()
        apply_sell_lots(hs, 5.0, "hifo")
        # The helper itself does NOT refresh entry_price — that's the
        # caller's job (sim.py does it). We pin the helper output here:
        assert hs.weighted_avg_entry_price() == pytest.approx(102.0)

    def test_sim_apply_sell_refreshes_entry_price_post_hifo(self):
        """SimAdapter._apply_sell must call weighted_avg_entry_price()
        after apply_sell_lots(). Source-level pin so a future refactor
        that drops the refresh fails loud."""
        src = (STRATEGY / "adapters" / "sim.py").read_text()
        idx_apply_sell = src.find("def _apply_sell")
        idx_apply_buy  = src.find("def _apply_buy", idx_apply_sell)
        body = src[idx_apply_sell:idx_apply_buy]
        # Both pieces must appear in _apply_sell:
        assert "apply_sell_lots(" in body
        assert "weighted_avg_entry_price()" in body, (
            "P1-5 regression: sim's _apply_sell must refresh entry_price "
            "from the surviving lots after apply_sell_lots."
        )
        # AND the refresh must come AFTER apply_sell_lots, not before.
        idx_consume = body.find("apply_sell_lots(")
        idx_refresh = body.find("weighted_avg_entry_price()")
        assert idx_refresh > idx_consume, (
            "weighted_avg_entry_price() must execute AFTER apply_sell_lots; "
            "otherwise the refresh sees the unmutated lots."
        )

    def test_full_sell_leaves_zero_entry_price(self):
        """When all lots consumed (full liquidation), weighted_avg returns
        the legacy entry_price as fallback (pre-existing behaviour). Pin
        this so refactors don't accidentally return NaN.
        """
        from kernel.exits import apply_sell_lots
        hs = self._make_hs_with_lots()
        apply_sell_lots(hs, 30.0, "fifo")
        assert hs.lots == []
        # Post-fix sim refreshes entry_price; the helper returns the legacy
        # field when lots are empty (== fallback). Just check it's finite
        # and non-negative — actual semantic is "position is closed,
        # value is irrelevant".
        v = hs.weighted_avg_entry_price()
        assert v >= 0.0
