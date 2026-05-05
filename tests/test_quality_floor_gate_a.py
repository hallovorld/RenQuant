"""Tests for QualityFloorTask Gate A (Distribution-relative percentile floor).

Design: ``doc/buy_logic_redesign_2026-04-26.md`` §2 / Gate A.

Gate A reads the trailing-N-day p_X cutoff from `score_percentiles_daily`
(populated by RecordScoreDistributionTask) and rejects candidates whose
`panel_score` falls below it. Adapts to whatever scale the panel
emits — fix for calibrator saturation periods.

Stage 0 contract: defaults all OFF — bit-for-bit parity preserved.
"""
from __future__ import annotations

import datetime
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.panel_pipeline.task_quality_floor import (  # noqa: E402
    QualityFloorTask,
    _gate_a_distribution_floor,
)
from kernel.persistence import _SCHEMA_SQL  # noqa: E402


@dataclass
class _Cand:
    ticker: str
    panel_score: float | None = None
    rank_score:  float | None = None
    mu:    float | None = None
    sigma: float | None = None


@dataclass
class _Ctx:
    config: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    holdings:   dict = field(default_factory=dict)
    today: datetime.date = datetime.date(2026, 4, 26)


def _on_a(percentile: int = 85, lookback: int = 20,
          min_history: int = 5) -> dict:
    return {
        "ranking": {
            "panel_scoring": {
                "quality_floor": {
                    "enabled": True,
                    "distribution_floor": {
                        "enabled":     True,
                        "percentile":  percentile,
                        "lookback_days": lookback,
                        "min_history_days": min_history,
                    },
                },
            },
        },
    }


def _seed_percentiles(db: sqlite3.Connection,
                       p_values: list[tuple[str, float]],
                       n_cands_per_day: int = 10) -> None:
    """Insert (date, p85) pairs into score_percentiles_daily."""
    cur = db.cursor()
    for d, p85 in p_values:
        cur.execute(
            """INSERT INTO score_percentiles_daily
                  (date, n_cands, p85) VALUES (?, ?, ?)""",
            (d, n_cands_per_day, p85),
        )
    db.commit()


# ── Pure function ─────────────────────────────────────────────────────────────

class TestGateAPure:
    def test_passes_when_above_threshold(self):
        c = _Cand("A", rank_score=0.05)
        ok, reason = _gate_a_distribution_floor(c, threshold=0.02)
        assert ok is True and reason is None

    def test_rejects_when_below_threshold(self):
        c = _Cand("A", rank_score=0.005)
        ok, reason = _gate_a_distribution_floor(c, threshold=0.02)
        assert ok is False
        assert "rank_score" in reason

    def test_no_threshold_passes(self):
        """No history yet → threshold=None → don't gate."""
        c = _Cand("A", rank_score=-5.0)
        ok, reason = _gate_a_distribution_floor(c, threshold=None)
        assert ok is True and reason is None

    def test_missing_panel_score_passes(self):
        c = _Cand("A", rank_score=None)
        ok, _ = _gate_a_distribution_floor(c, threshold=0.05)
        assert ok is True

    def test_nan_panel_score_rejected(self):
        c = _Cand("A", rank_score=float("nan"))
        ok, reason = _gate_a_distribution_floor(c, threshold=0.05)
        assert ok is False
        assert reason == "rank_score_nan"


# ── Task integration ──────────────────────────────────────────────────────────

class TestGateAIntegration:
    def test_gate_a_off_preserves_candidates(self):
        ctx = _Ctx(config={})
        ctx.candidates = [_Cand("A", rank_score=0.001)]
        QualityFloorTask().run(ctx)
        assert len(ctx.candidates) == 1

    def test_gate_a_no_db_attached_no_op(self):
        """Gate A enabled but no DB → no threshold → no-op + no crash."""
        ctx = _Ctx(config=_on_a())
        ctx.candidates = [_Cand("WEAK", rank_score=-1.0)]
        QualityFloorTask().run(ctx)
        assert len(ctx.candidates) == 1   # no-op

    def test_gate_a_insufficient_history_no_op(self):
        ctx = _Ctx(config=_on_a(min_history=10))
        db = sqlite3.connect(":memory:")
        db.executescript(_SCHEMA_SQL)
        # only 3 rows of history < min_history=10 → no-op
        _seed_percentiles(db, [
            ("2026-04-23", 0.05),
            ("2026-04-24", 0.05),
            ("2026-04-25", 0.05),
        ])
        ctx._db = db                    # noqa: SLF001
        ctx.candidates = [_Cand("WEAK", rank_score=-1.0)]
        QualityFloorTask().run(ctx)
        assert len(ctx.candidates) == 1   # no-op

    def test_gate_a_rejects_below_p85(self):
        ctx = _Ctx(config=_on_a(percentile=85, lookback=5,
                                min_history=3))
        db = sqlite3.connect(":memory:")
        db.executescript(_SCHEMA_SQL)
        # 5 days of history, p85 averages to 0.05
        _seed_percentiles(db, [
            ("2026-04-21", 0.04),
            ("2026-04-22", 0.05),
            ("2026-04-23", 0.05),
            ("2026-04-24", 0.05),
            ("2026-04-25", 0.06),
        ])
        ctx._db = db                    # noqa: SLF001
        strong = _Cand("STRONG", rank_score=0.10)   # > p85
        weak   = _Cand("WEAK",   rank_score=0.01)   # < p85
        ctx.candidates = [strong, weak]
        QualityFloorTask().run(ctx)
        kept = {c.ticker for c in ctx.candidates}
        assert kept == {"STRONG"}

    def test_gate_a_blocked_reason_surfaces(self):
        ctx = _Ctx(config=_on_a(percentile=85, lookback=5,
                                min_history=3))
        db = sqlite3.connect(":memory:")
        db.executescript(_SCHEMA_SQL)
        _seed_percentiles(db, [
            ("2026-04-23", 0.10),
            ("2026-04-24", 0.10),
            ("2026-04-25", 0.10),
        ])
        ctx._db = db                    # noqa: SLF001
        ctx.candidates = [_Cand("WEAK", rank_score=0.01)]
        QualityFloorTask().run(ctx)
        blocked = getattr(ctx, "_blocked_by_ticker", {})
        assert "WEAK" in blocked
        assert blocked["WEAK"].startswith("quality_floor:gate_a:")

    def test_combined_gate_a_then_b(self):
        """When both gates enabled, gate_a fires first; b only on survivors."""
        ctx = _Ctx(config={
            "ranking": {"panel_scoring": {"quality_floor": {
                "enabled": True,
                "distribution_floor": {
                    "enabled": True, "percentile": 85,
                    "lookback_days": 5, "min_history_days": 3,
                },
                "edge_sharpe_floor": {"enabled": True, "threshold": 0.30},
            }}}
        })
        db = sqlite3.connect(":memory:")
        db.executescript(_SCHEMA_SQL)
        _seed_percentiles(db, [
            ("2026-04-23", 0.05),
            ("2026-04-24", 0.05),
            ("2026-04-25", 0.05),
        ])
        ctx._db = db                    # noqa: SLF001
        # GATE_A_FAIL: panel below p85 — rejected at gate A
        # GATE_B_FAIL: panel above p85 BUT edge_sharpe<0.3 — rejected at B
        # PASS_BOTH:   panel above + edge_sharpe>0.3
        gate_a_fail = _Cand("GA_FAIL",  rank_score=0.001,
                             mu=0.10, sigma=0.10)
        gate_b_fail = _Cand("GB_FAIL",  rank_score=0.10,
                             mu=0.01, sigma=0.10)        # 0.1 < 0.3
        pass_both   = _Cand("PASS",     rank_score=0.10,
                             mu=0.10, sigma=0.10)        # 1.0 > 0.3
        ctx.candidates = [gate_a_fail, gate_b_fail, pass_both]
        QualityFloorTask().run(ctx)
        kept = {c.ticker for c in ctx.candidates}
        assert kept == {"PASS"}
        blocked = getattr(ctx, "_blocked_by_ticker", {})
        assert "gate_a" in blocked.get("GA_FAIL", "")
        assert "gate_b" in blocked.get("GB_FAIL", "")
