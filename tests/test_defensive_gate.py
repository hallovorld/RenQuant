"""Plan O regression tests — defensive tickers are only eligible in BEAR.

The 2026-04-20 live log captured a real bug: XLU (defensive utilities
ETF) was bought in BULL_VOLATILE regime at $961 — the `defensive_tickers`
gate was only applied *inside* the BEAR branch (capping slots), not
*outside* it (excluding from offensive buys). And the sector-guard
bypass for defensives made it easier, not harder, for them to sneak in.

This suite pins:
  1. `run_selection_loop` rejects defensives when `bear_only=False`.
  2. BEAR branch (`bear_only=True`) still allows defensives.
  3. `blocks["defensive_non_bear"]` counter fires in the right case.
  4. `PrepareSelectionTask` propagates `ctx.bear_only` into SelectionContext.
  5. End-to-end: XLU top-ranked in BULL_VOLATILE → NOT selected;
     XLU top-ranked in BEAR-only → selected.
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

from kernel.selection import (  # noqa: E402
    CandidateResult,
    SelectionContext,
    run_selection_loop,
)
from kernel.pipeline.task_selection import PrepareSelectionTask  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ranked(tickers_scores: list[tuple[str, float]]) -> list[CandidateResult]:
    return [
        CandidateResult(
            ticker=t, raw_score=s * 10, rank_score=s,
            rs_score=0.0, detail="",
        )
        for t, s in tickers_scores
    ]


def _sel_ctx(*, bear_only: bool, defensive_set: set[str],
             open_slots: int = 8) -> SelectionContext:
    return SelectionContext(
        today             = datetime.date(2026, 4, 23),
        held_tickers      = [],
        last_sell_dates   = {},
        earnings_calendar = {},
        corr_matrix       = None,
        sector_map        = {},
        defensive_set     = defensive_set,
        wash_sale_days    = 0,
        earnings_buffer   = 0,
        corr_threshold    = 1.0,
        max_per_sector    = 0,
        tiered_thresholds = [],
        open_slots        = open_slots,
        bear_only         = bear_only,
    )


def _low_corr_matrix(tickers: list[str]) -> dict[str, dict[str, float]]:
    return {
        a: {b: (1.0 if a == b else 0.1) for b in tickers}
        for a in tickers
    }


# ── Core guarantee ───────────────────────────────────────────────────────────

class TestDefensiveOnlyInBear:
    def test_non_bear_rejects_defensives(self):
        """BULL_VOLATILE / BULL_CALM / CHOPPY — defensives never selected."""
        ranked = _ranked([("XLU", 0.50), ("CAT", 0.40), ("GLD", 0.38)])
        ctx = _sel_ctx(bear_only=False, defensive_set={"XLU", "GLD", "TLT", "XLV"})
        selected, blocks = run_selection_loop(ranked, ctx)
        assert "XLU" not in selected, "XLU must not be bought in non-BEAR"
        assert "GLD" not in selected
        assert selected == ["CAT"]
        assert blocks["defensive_non_bear"] == 2   # XLU + GLD

    def test_bear_branch_allows_defensives(self):
        """BEAR — defensives are exactly what we want to buy."""
        ranked = _ranked([("XLU", 0.50), ("GLD", 0.30)])
        ctx = _sel_ctx(bear_only=True, defensive_set={"XLU", "GLD", "TLT", "XLV"},
                        open_slots=2)
        selected, blocks = run_selection_loop(ranked, ctx)
        assert "XLU" in selected
        assert blocks["defensive_non_bear"] == 0

    def test_non_bear_still_selects_offensives(self):
        """Non-defensives pass through normally in BULL_CALM."""
        ranked = _ranked([("CAT", 0.45), ("GOOG", 0.42), ("NVDA", 0.40)])
        ctx = _sel_ctx(bear_only=False, defensive_set={"XLU", "GLD"})
        ctx.corr_matrix = _low_corr_matrix(["CAT", "GOOG", "NVDA"])
        selected, _ = run_selection_loop(ranked, ctx)
        assert selected == ["CAT", "GOOG", "NVDA"]

    def test_bear_only_flag_default_false(self):
        """Default SelectionContext has bear_only=False (safe default —
        defensives are blocked unless the caller explicitly opts in)."""
        ctx = SelectionContext(
            today             = datetime.date(2026, 4, 23),
            held_tickers      = [],
            last_sell_dates   = {},
            earnings_calendar = {},
            corr_matrix       = None,
            sector_map        = {},
            defensive_set     = {"XLU"},
            wash_sale_days    = 0,
            earnings_buffer   = 0,
            corr_threshold    = 1.0,
            max_per_sector    = 0,
            tiered_thresholds = [],
            open_slots        = 5,
        )
        assert ctx.bear_only is False
        selected, _ = run_selection_loop(_ranked([("XLU", 0.9)]), ctx)
        assert "XLU" not in selected

    def test_bear_empty_defensive_set_still_allows(self):
        """If the watchlist has no defensives configured, bear_only has
        no effect — non-defensive tickers still get through."""
        ranked = _ranked([("CAT", 0.45)])
        ctx = _sel_ctx(bear_only=True, defensive_set=set())
        selected, _ = run_selection_loop(ranked, ctx)
        assert selected == ["CAT"]


# ── Block-counter audit ──────────────────────────────────────────────────────

class TestBlockCounterAudit:
    def test_counter_fires_only_when_non_bear_and_defensive(self):
        """The counter should only fire for defensive candidates in non-BEAR."""
        ranked = _ranked([
            ("XLU", 0.50),    # defensive, non-BEAR → block
            ("CAT", 0.40),    # not defensive → pass
            ("GLD", 0.38),    # defensive → block
            ("NVDA", 0.30),   # not defensive → pass
        ])
        ctx = _sel_ctx(bear_only=False, defensive_set={"XLU", "GLD"})
        _, blocks = run_selection_loop(ranked, ctx)
        assert blocks["defensive_non_bear"] == 2

    def test_counter_zero_in_bear(self):
        """In BEAR, defensives shouldn't increment the counter."""
        ranked = _ranked([("XLU", 0.50), ("GLD", 0.30)])
        ctx = _sel_ctx(bear_only=True, defensive_set={"XLU", "GLD"}, open_slots=2)
        _, blocks = run_selection_loop(ranked, ctx)
        assert blocks["defensive_non_bear"] == 0


# ── PrepareSelectionTask propagates bear_only ────────────────────────────────

class TestPrepareSelectionTaskWiresBearOnly:
    def _stub_ctx(self, bear_only: bool):
        ctx = MagicMock()
        ctx.config = {
            "regime": {},
            "regime_params": {"BULL_VOLATILE": {"max_concurrent_positions": 8}},
            "wash_sale_days":          0,
            "max_positions_per_sector": 0,
            "sector_map":              {},
            "defensive_tickers":       ["XLU", "GLD"],
            "tiered_thresholds":       [],
        }
        ctx.regime = "BEAR" if bear_only else "BULL_VOLATILE"
        ctx.bear_only = bear_only
        ctx.holdings = {}
        ctx.rotations = []
        ctx.last_sell_dates = {}
        ctx.earnings_calendar = {}
        ctx.corr_matrix = None
        ctx.today = datetime.date(2026, 4, 23)
        ctx._sel_ctx = None
        return ctx

    def test_propagates_bear_only_true(self):
        ctx = self._stub_ctx(bear_only=True)
        PrepareSelectionTask().run(ctx)
        assert ctx._sel_ctx is not None
        assert ctx._sel_ctx.bear_only is True

    def test_propagates_bear_only_false(self):
        ctx = self._stub_ctx(bear_only=False)
        PrepareSelectionTask().run(ctx)
        assert ctx._sel_ctx is not None
        assert ctx._sel_ctx.bear_only is False


# ── The 2026-04-20 XLU incident — regression replay ──────────────────────────

class TestXLUIncident20260420:
    """Replay: XLU was top-ranked in BULL_VOLATILE and got bought.
    With the fix, it must be rejected and the counter must fire.
    """
    def test_xlu_rejected_in_bull_volatile(self):
        ranked = _ranked([
            ("XLU",  0.4256),   # Platt-calibrated, the exact live rank_score
            ("CAT",  0.4287),
            ("GOOG", 0.4144),
        ])
        ctx = _sel_ctx(bear_only=False,
                        defensive_set={"GLD", "TLT", "XLV", "XLU"})
        ctx.corr_matrix = _low_corr_matrix(["XLU", "CAT", "GOOG"])
        selected, blocks = run_selection_loop(ranked, ctx)
        assert "XLU" not in selected, "XLU 2026-04-20 bug must stay fixed"
        assert "CAT" in selected and "GOOG" in selected
        assert blocks["defensive_non_bear"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
