"""Plan P regression tests — candidate_scores.blocked_by is populated.

Before the fix:
  * `candidate_scores.blocked_by` column existed but stayed empty — impossible
    to answer "why was X not selected" from the decision-trace DB.
  * `run_selection_loop` produced only aggregate counts (`blocks` dict), never
    per-ticker reasons.

After the fix:
  * `run_selection_loop` accepts an optional `blocked_by_ticker` out-param dict
    and populates it in-place with ticker → rejection reason.
  * `RunSelectionTask` opts in and stores on `ctx._blocked_by_ticker`.
  * `SimAdapter` / `RunnerAdapter` pass `blocked_map=` through to
    `record_candidate_scores`, which writes the `blocked_by` column.

Five rejection reasons are covered: tier, wash_sale, sector, correlation,
defensive_non_bear.
"""
from __future__ import annotations

import datetime
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.selection import (  # noqa: E402
    CandidateResult,
    SelectionContext,
    run_selection_loop,
)


def _ranked(items: list[tuple[str, float]]) -> list[CandidateResult]:
    return [
        CandidateResult(ticker=t, raw_score=s * 10, rank_score=s,
                        rs_score=0.0, detail="")
        for t, s in items
    ]


def _sel_ctx(**overrides) -> SelectionContext:
    defaults = dict(
        today             = datetime.date(2026, 4, 24),
        held_tickers      = [],
        last_sell_dates   = {},
        earnings_calendar = {},
        corr_matrix       = None,
        sector_map        = {},
        defensive_set     = set(),
        wash_sale_days    = 0,
        earnings_buffer   = 0,
        corr_threshold    = 1.0,
        max_per_sector    = 10,
        tiered_thresholds = [],
        open_slots        = 8,
        bear_only         = False,
    )
    defaults.update(overrides)
    return SelectionContext(**defaults)


# ── Direct behaviour of run_selection_loop out-param ──────────────────────────

class TestOutParamPopulation:
    def test_tier_rejection_is_recorded(self):
        """Tier-threshold rejection writes 'tier' into the ticker→reason map."""
        ranked = _ranked([("AAA", 0.05), ("BBB", 0.50)])   # AAA below, BBB above
        ctx = _sel_ctx(tiered_thresholds=[{"min_model_score": 0.10}])
        blocked: dict[str, str] = {}
        selected, blocks = run_selection_loop(ranked, ctx, blocked_by_ticker=blocked)
        assert "BBB" in selected
        assert blocked == {"AAA": "tier"}
        assert blocks["tier"] == 1

    def test_wash_sale_rejection_is_recorded(self):
        """Wash-sale block writes 'wash_sale'."""
        today = datetime.date(2026, 4, 24)
        ranked = _ranked([("NVDA", 0.5)])
        ctx = _sel_ctx(
            today          = today,
            last_sell_dates= {"NVDA": today - datetime.timedelta(days=2)},
            wash_sale_days = 30,
        )
        blocked: dict[str, str] = {}
        run_selection_loop(ranked, ctx, blocked_by_ticker=blocked)
        assert blocked == {"NVDA": "wash_sale"}

    def test_sector_rejection_is_recorded(self):
        """Sector cap block writes 'sector'."""
        ranked = _ranked([("MSFT", 0.5)])
        ctx = _sel_ctx(
            held_tickers   = ["AAPL", "GOOG", "AMZN"],
            sector_map     = {"MSFT": "Tech", "AAPL": "Tech", "GOOG": "Tech", "AMZN": "Tech"},
            max_per_sector = 3,
        )
        blocked: dict[str, str] = {}
        run_selection_loop(ranked, ctx, blocked_by_ticker=blocked)
        assert blocked == {"MSFT": "sector"}

    def test_correlation_rejection_is_recorded(self):
        """Correlation-guard block writes 'correlation'."""
        ranked = _ranked([("AMD", 0.5)])
        ctx = _sel_ctx(
            held_tickers    = ["NVDA"],
            corr_matrix     = {"AMD": {"NVDA": 0.95}},
            corr_threshold  = 0.70,
        )
        blocked: dict[str, str] = {}
        run_selection_loop(ranked, ctx, blocked_by_ticker=blocked)
        assert blocked == {"AMD": "correlation"}

    def test_defensive_non_bear_rejection_is_recorded(self):
        """Defensive-in-non-BEAR block writes 'defensive_non_bear'."""
        ranked = _ranked([("XLU", 0.8), ("CAT", 0.5)])
        ctx = _sel_ctx(defensive_set={"XLU", "GLD"}, bear_only=False)
        blocked: dict[str, str] = {}
        run_selection_loop(ranked, ctx, blocked_by_ticker=blocked)
        assert blocked == {"XLU": "defensive_non_bear"}
        # CAT was selected, not blocked

    def test_multi_reason_mixed_run(self):
        """Different tickers, different reasons — all land in one map."""
        ranked = _ranked([
            ("XLU", 0.9),    # defensive in non-BEAR
            ("AAA", 0.05),   # tier floor
            ("MSFT", 0.5),   # sector cap
            ("CAT", 0.5),    # selected
        ])
        ctx = _sel_ctx(
            held_tickers      = ["AAPL", "GOOG", "AMZN"],
            sector_map        = {"MSFT": "Tech", "AAPL": "Tech", "GOOG": "Tech",
                                 "AMZN": "Tech", "CAT": "Industrials"},
            max_per_sector    = 3,
            tiered_thresholds = [{"min_model_score": 0.10}],
            defensive_set     = {"XLU"},
        )
        blocked: dict[str, str] = {}
        selected, _ = run_selection_loop(ranked, ctx, blocked_by_ticker=blocked)
        assert selected == ["CAT"]
        assert blocked == {
            "XLU": "defensive_non_bear",
            "AAA": "tier",
            "MSFT": "sector",
        }

    def test_opt_in_default_is_none(self):
        """Callers that don't pass the dict get legacy 2-tuple behaviour."""
        ranked = _ranked([("AAA", 0.05)])
        ctx = _sel_ctx(tiered_thresholds=[{"min_model_score": 0.10}])
        selected, blocks = run_selection_loop(ranked, ctx)   # legacy signature
        assert selected == []
        assert blocks["tier"] == 1


# ── RunSelectionTask plumbing ────────────────────────────────────────────────

class TestTaskPopulatesCtx:
    def test_task_stashes_blocked_by_ticker_on_ctx(self):
        from kernel.pipeline.task_selection import RunSelectionTask
        task = RunSelectionTask()

        # Minimal fake InferenceContext — task only reads .ranked + ._sel_ctx
        # and writes ._selected / ._blocks / ._blocked_by_ticker + counters.
        class _FakeCtx:
            ranked   = _ranked([("XLU", 0.9), ("CAT", 0.5)])
            _sel_ctx = _sel_ctx(defensive_set={"XLU"}, bear_only=False)
            counters: dict = {}
        ctx = _FakeCtx()
        task.run(ctx)
        assert ctx._blocked_by_ticker == {"XLU": "defensive_non_bear"}
        assert ctx._selected == ["CAT"]


# ── End-to-end DB write: record_candidate_scores persists blocked_by ─────────

class TestDBWriteThrough:
    def test_blocked_by_column_populated_end_to_end(self, tmp_path):
        """record_candidate_scores writes the blocked_map into blocked_by."""
        from kernel.persistence import (
            get_connection, record_pipeline_run, record_candidate_scores,
        )
        db_path = tmp_path / "runs.db"
        conn = get_connection({"persistence": {"enabled": True,
                                                "db_path": str(db_path)}})
        run_id = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 24),
            strategy="renquant_104",
        )
        candidates = [
            SimpleNamespace(ticker="CAT", raw_score=5.0, rank_score=0.5,
                            rs_score=0.0, panel_score=0.5, mu=None, sigma=None),
            SimpleNamespace(ticker="XLU", raw_score=9.0, rank_score=0.9,
                            rs_score=0.0, panel_score=0.9, mu=None, sigma=None),
            SimpleNamespace(ticker="MSFT", raw_score=5.0, rank_score=0.5,
                            rs_score=0.0, panel_score=0.5, mu=None, sigma=None),
        ]
        blocked_map = {"XLU": "defensive_non_bear", "MSFT": "sector"}
        record_candidate_scores(
            conn, run_id,
            candidates, holdings={}, selected_tickers={"CAT"},
            blocked_map=blocked_map,
        )
        rows = dict(conn.execute(
            "SELECT ticker, blocked_by FROM candidate_scores WHERE run_id=?",
            (run_id,),
        ).fetchall())
        assert rows == {"CAT": None, "XLU": "defensive_non_bear", "MSFT": "sector"}
