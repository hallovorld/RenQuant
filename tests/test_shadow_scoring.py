"""Regression tests for ApplyShadowScoringTask.

Pin the shadow model pattern (records what alt models WOULD do without
affecting primary orders). Verifies:
  1. No-op when no shadow_models configured (safe default)
  2. Shadow Task is registered in PanelScoringJob
  3. DB schema init creates proper table
  4. Persist rows works (INSERT OR REPLACE on PK conflict)
  5. Source-level pins on key behavior strings
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))


@pytest.fixture(scope="module")
def shadow_mod():
    from kernel.panel_pipeline import shadow_scoring
    return shadow_scoring


class TestSourceContracts:
    """Pin behavior strings so future refactors can't silently change semantics."""

    def test_apply_shadow_task_registered_in_job(self):
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "job_panel_scoring.py").read_text()
        assert "from kernel.panel_pipeline.shadow_scoring import ApplyShadowScoringTask" in src
        assert "ApplyShadowScoringTask()" in src

    def test_shadow_does_not_submit_orders(self, shadow_mod):
        """Shadow Task must NOT contain order-placement code paths."""
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "shadow_scoring.py").read_text()
        # Forbidden — these would mean shadow places orders
        assert "place_order" not in src
        assert "submit_order" not in src
        assert "BUY" not in src   # no buy emission
        assert "broker." not in src

    def test_default_db_path(self, shadow_mod):
        assert shadow_mod._DB_PATH_DEFAULT == "data/shadow_scores.db"

    def test_2026_05_18_marker(self, shadow_mod):
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "shadow_scoring.py").read_text()
        assert "2026-05-18" in src


class TestDBSchema:
    """Pin the shadow_scores DB schema."""

    def test_init_creates_table(self, tmp_path, shadow_mod):
        db = tmp_path / "test_shadow.db"
        shadow_mod._init_shadow_db(db)
        assert db.exists()
        conn = sqlite3.connect(str(db))
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            assert "shadow_scores" in tables
        finally:
            conn.close()

    def test_schema_has_required_columns(self, tmp_path, shadow_mod):
        db = tmp_path / "test_shadow.db"
        shadow_mod._init_shadow_db(db)
        conn = sqlite3.connect(str(db))
        try:
            cur = conn.execute("PRAGMA table_info(shadow_scores)")
            cols = {r[1] for r in cur.fetchall()}
            required = {"as_of_date", "ticker", "shadow_name", "shadow_kind",
                        "primary_score", "shadow_score", "diff",
                        "primary_rank", "shadow_rank", "rank_diff",
                        "inserted_at"}
            assert required.issubset(cols), \
                f"Missing columns: {required - cols}"
        finally:
            conn.close()

    def test_pk_constraint(self, tmp_path, shadow_mod):
        """(as_of_date, ticker, shadow_name) is PK — duplicate insert replaces."""
        db = tmp_path / "test_shadow.db"
        shadow_mod._init_shadow_db(db)
        row1 = {"as_of_date": "2026-05-18", "ticker": "AAPL",
                "shadow_name": "patchtst_v1", "shadow_kind": "patchtst",
                "primary_score": 0.1, "shadow_score": 0.05, "diff": -0.05,
                "primary_rank": 1, "shadow_rank": 3, "rank_diff": 2}
        row2 = dict(row1, shadow_score=0.2)  # update for same PK
        shadow_mod._persist_shadow_rows(db, [row1])
        shadow_mod._persist_shadow_rows(db, [row2])
        conn = sqlite3.connect(str(db))
        try:
            cur = conn.execute(
                "SELECT shadow_score FROM shadow_scores WHERE ticker='AAPL'")
            scores = [r[0] for r in cur.fetchall()]
            assert scores == [0.2], "PK should REPLACE on conflict"
        finally:
            conn.close()


class TestNoOpWhenNoShadow:
    """When config has no shadow_models, Task is no-op (no DB created)."""

    def test_task_runs_silently_with_empty_config(self, shadow_mod, tmp_path):
        from kernel.panel_pipeline.shadow_scoring import ApplyShadowScoringTask
        class MockCtx:
            def __init__(self):
                self.config = {"ranking": {"panel_scoring": {}}}
                self.candidates = []
                self.holdings = []
                self.today = None
        ctx = MockCtx()
        task = ApplyShadowScoringTask()
        result = task.run(ctx)
        # Returns None silently, no exception
        assert result is None or result is False


class TestPersistShadowRows:
    """Verify writing real rows works."""

    def test_persist_and_query(self, tmp_path, shadow_mod):
        db = tmp_path / "shadow_persist.db"
        shadow_mod._init_shadow_db(db)
        rows = [
            {"as_of_date": "2026-05-18", "ticker": t,
             "shadow_name": "patchtst_seed42", "shadow_kind": "patchtst",
             "primary_score": i * 0.01, "shadow_score": i * 0.012,
             "diff": i * 0.002, "primary_rank": i, "shadow_rank": i + 1,
             "rank_diff": 1}
            for i, t in enumerate(["A", "B", "C", "D", "E"], start=1)
        ]
        shadow_mod._persist_shadow_rows(db, rows)
        conn = sqlite3.connect(str(db))
        try:
            cur = conn.execute("SELECT COUNT(*) FROM shadow_scores")
            assert cur.fetchone()[0] == 5
            cur = conn.execute(
                "SELECT ticker, diff FROM shadow_scores ORDER BY primary_rank")
            res = list(cur.fetchall())
            assert res[0] == ("A", 0.002)
            assert res[4] == ("E", 0.010)
        finally:
            conn.close()
