"""Regression tests for decision-trace invariant repair."""
from __future__ import annotations

import sqlite3

from scripts.repair_decision_trace_invariants import (
    clear_selected_blockers,
    count_selected_blockers,
    main,
)


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE candidate_scores (
            run_id TEXT,
            ticker TEXT,
            selected INTEGER,
            blocked_by TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE ticker_daily_state (
            run_id TEXT,
            ticker TEXT,
            selected INTEGER,
            blocked_by TEXT
        )"""
    )
    conn.executemany(
        "INSERT INTO candidate_scores VALUES (?, ?, ?, ?)",
        [
            ("r1", "AAA", 1, "kelly_zero:mu_none"),
            ("r1", "BBB", 0, "tier"),
            ("r1", "CCC", 1, None),
        ],
    )
    conn.executemany(
        "INSERT INTO ticker_daily_state VALUES (?, ?, ?, ?)",
        [
            ("r1", "AAA", 1, "kelly_zero:mu_none"),
            ("r1", "BBB", 0, "tier"),
        ],
    )
    conn.commit()
    return conn


def test_count_and_clear_selected_blockers(tmp_path):
    db = tmp_path / "runs.db"
    conn = _make_db(db)

    assert count_selected_blockers(conn) == {
        "candidate_scores": 1,
        "ticker_daily_state": 1,
    }
    assert clear_selected_blockers(conn) == {
        "candidate_scores": 1,
        "ticker_daily_state": 1,
    }
    conn.commit()
    assert count_selected_blockers(conn) == {
        "candidate_scores": 0,
        "ticker_daily_state": 0,
    }
    assert conn.execute(
        "SELECT blocked_by FROM candidate_scores WHERE ticker='BBB'"
    ).fetchone()[0] == "tier"
    conn.close()


def test_cli_check_fails_before_apply_and_passes_after(tmp_path, capsys):
    db = tmp_path / "runs.db"
    conn = _make_db(db)
    conn.close()

    assert main([str(db), "--check"]) == 1
    assert "kelly_zero" not in capsys.readouterr().out
    assert main([str(db), "--apply", "--check"]) == 0
