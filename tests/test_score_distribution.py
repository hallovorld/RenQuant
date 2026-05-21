"""Behavior tests for RecordScoreDistributionTask + percentile lookup."""
from __future__ import annotations

import datetime
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.persistence import _SCHEMA_SQL  # noqa: E402
from kernel.pipeline.task_score_distribution import (  # noqa: E402
    RecordScoreDistributionTask,
    get_score_percentile_threshold,
)


@dataclass
class _Cand:
    ticker: str
    panel_score: float | None = None
    rank_score: float | None = None
    mu: float | None = None
    sigma: float | None = None


@dataclass
class _Hold:
    panel_score: float | None = None
    rank_score: float | None = None
    mu: float | None = None
    sigma: float | None = None


@dataclass
class _Ctx:
    config: dict = field(default_factory=dict)
    today: datetime.date = datetime.date(2026, 4, 26)
    run_id: str = "2026-04-26-test-a"
    regime: str = "BULL_CALM"
    candidates: list = field(default_factory=list)
    holdings: dict = field(default_factory=dict)


def _make_db():
    db = sqlite3.connect(":memory:")
    db.executescript(_SCHEMA_SQL)
    return db


def _cfg_on():
    return {"score_db": {"enabled": True}}


class TestFlagGate:
    def test_disabled_returns_false(self):
        ctx = _Ctx(config={"score_db": {"enabled": False}})
        ctx._db = _make_db()
        assert RecordScoreDistributionTask().run(ctx) is False

    def test_no_db_returns_false(self):
        ctx = _Ctx(config=_cfg_on())
        # No _db attr set
        assert RecordScoreDistributionTask().run(ctx) is False

    def test_no_cands_or_holdings_returns_false(self):
        ctx = _Ctx(config=_cfg_on())
        ctx._db = _make_db()
        assert RecordScoreDistributionTask().run(ctx) is False


class TestRecordCandidates:
    def test_writes_one_row_per_candidate(self):
        ctx = _Ctx(config=_cfg_on())
        ctx._db = _make_db()
        ctx.candidates = [
            _Cand("A", panel_score=0.5, rank_score=0.42, mu=0.01, sigma=0.05),
            _Cand("B", panel_score=0.8, rank_score=0.65, mu=0.02, sigma=0.06),
            _Cand("C", panel_score=0.2, rank_score=0.15, mu=-0.01, sigma=0.04),
        ]
        RecordScoreDistributionTask().run(ctx)
        cur = ctx._db.cursor()
        cur.execute("SELECT ticker, rank_score, is_holding FROM score_distribution ORDER BY ticker")
        rows = cur.fetchall()
        assert len(rows) == 3
        assert rows[0] == ("A", 0.42, 0)

    def test_writes_holdings_with_is_holding_flag(self):
        ctx = _Ctx(config=_cfg_on())
        ctx._db = _make_db()
        ctx.holdings = {
            "X": _Hold(panel_score=0.6, rank_score=0.55),
        }
        RecordScoreDistributionTask().run(ctx)
        cur = ctx._db.cursor()
        cur.execute("SELECT ticker, rank_score, is_holding FROM score_distribution")
        row = cur.fetchone()
        assert row == ("X", 0.55, 1)

    def test_idempotent_via_replace(self):
        """Running twice in the same run doesn't duplicate rows."""
        ctx = _Ctx(config=_cfg_on())
        ctx._db = _make_db()
        ctx.candidates = [_Cand("A", rank_score=0.5)]
        RecordScoreDistributionTask().run(ctx)
        ctx.candidates = [_Cand("A", rank_score=0.7)]   # changed score
        RecordScoreDistributionTask().run(ctx)
        cur = ctx._db.cursor()
        cur.execute("SELECT COUNT(*), MAX(rank_score) FROM score_distribution")
        n, max_score = cur.fetchone()
        assert n == 1, "INSERT OR REPLACE should keep one row per (run_id, ticker)"
        assert max_score == 0.7

    def test_same_date_ticker_preserved_across_runs(self):
        db = _make_db()
        ctx_a = _Ctx(config=_cfg_on(), run_id="2026-04-26-live-a")
        ctx_a._db = db
        ctx_a.candidates = [_Cand("A", rank_score=0.5)]
        ctx_b = _Ctx(config=_cfg_on(), run_id="2026-04-26-live-b")
        ctx_b._db = db
        ctx_b.candidates = [_Cand("A", rank_score=0.7)]
        RecordScoreDistributionTask().run(ctx_a)
        RecordScoreDistributionTask().run(ctx_b)
        out = db.execute(
            "SELECT COUNT(*), MIN(rank_score), MAX(rank_score) "
            "FROM score_distribution WHERE date=? AND ticker=?",
            ("2026-04-26", "A"),
        ).fetchone()
        assert out[0] == 2
        assert out[1] == pytest.approx(0.5)
        assert out[2] == pytest.approx(0.7)


class TestPercentilesAggregation:
    def test_writes_percentile_row_with_all_columns(self):
        ctx = _Ctx(config=_cfg_on())
        ctx._db = _make_db()
        ctx.candidates = [
            _Cand(f"T{i}", rank_score=0.1 * i) for i in range(1, 11)
        ]
        RecordScoreDistributionTask().run(ctx)
        cur = ctx._db.cursor()
        cur.execute(
            """SELECT n_cands, p05, p50, p95, score_min, score_max
               FROM score_percentiles_daily WHERE date = ?""",
            (ctx.today.isoformat(),),
        )
        row = cur.fetchone()
        assert row is not None
        n_cands, p05, p50, p95, smin, smax = row
        assert n_cands == 10
        assert smin == pytest.approx(0.1)
        assert smax == pytest.approx(1.0)
        assert p05 < p50 < p95   # ordered

    def test_percentile_skipped_when_no_finite_cand_scores(self):
        """All cands have None rank_score → no percentile row written."""
        ctx = _Ctx(config=_cfg_on())
        ctx._db = _make_db()
        ctx.candidates = [_Cand("A", rank_score=None)]
        RecordScoreDistributionTask().run(ctx)
        cur = ctx._db.cursor()
        cur.execute("SELECT COUNT(*) FROM score_percentiles_daily")
        assert cur.fetchone()[0] == 0


class TestPercentileLookup:
    def test_get_threshold_returns_mean_over_lookback(self):
        db = _make_db()
        cur = db.cursor()
        # Insert 3 days of percentile rows with different p85 values
        for i, day in enumerate(["2026-04-24", "2026-04-25", "2026-04-26"]):
            cur.execute(
                """INSERT INTO score_percentiles_daily
                   (run_id, date, n_cands, p85) VALUES (?, ?, ?, ?)""",
                (f"{day}-test", day, 10, 0.4 + 0.05 * i),  # 0.40, 0.45, 0.50
            )
        db.commit()
        # lookback 3 days from today → mean(0.40, 0.45, 0.50) = 0.45
        v = get_score_percentile_threshold(db, "2026-04-26", 85, lookback_days=3)
        assert v == pytest.approx(0.45)

    def test_get_threshold_returns_none_when_no_rows(self):
        db = _make_db()
        v = get_score_percentile_threshold(db, "2026-04-26", 85)
        assert v is None

    def test_get_threshold_invalid_percentile_raises(self):
        db = _make_db()
        with pytest.raises(ValueError, match="Unsupported percentile"):
            get_score_percentile_threshold(db, "2026-04-26", 42)


class TestPipelineIntegration:
    def test_score_distribution_job_should_skip_when_disabled(self):
        from kernel.pipeline.job_score_distribution import ScoreDistributionJob
        ctx = _Ctx(config={})
        ctx.candidates = [_Cand("A")]
        assert ScoreDistributionJob().should_skip(ctx) is True

    def test_score_distribution_job_runs_when_enabled(self):
        from kernel.pipeline.job_score_distribution import ScoreDistributionJob
        ctx = _Ctx(config=_cfg_on())
        ctx.candidates = [_Cand("A")]
        assert ScoreDistributionJob().should_skip(ctx) is False
