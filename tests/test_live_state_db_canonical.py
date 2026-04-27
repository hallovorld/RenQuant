"""Tests for #144 LIVE-STATE-DB-CANONICAL (2026-04-26 round-7).

User spec 2026-04-26: "live state json应该至少备份在db里"

Today: live_state_snapshots table is WRITTEN per-bar by
`record_live_state_snapshot` (kernel/persistence.py:732). Gap was
the READ path — if live_state.json was missing/corrupt, the runner
silently reset to defaults (lose streaks, HWM, regime).

Fix:
1. New `load_latest_live_state(conn, strategy, max_age_days)` reader
   in kernel/persistence.py.
2. Hook in adapters/runner.py::make_context — if JSON missing/corrupt,
   restore from latest db snapshot, write back to JSON.

These tests cover the reader directly (the runner integration is
exercised end-to-end in adapters tests; here we test the building
blocks).
"""
from __future__ import annotations

import datetime
import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.persistence import (  # noqa: E402
    _SCHEMA_SQL,
    load_latest_live_state,
    record_live_state_snapshot,
    record_pipeline_run,
)


def _db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA_SQL)
    return conn


def _record_snapshot(conn, *, run_id, run_date, strategy, state):
    """Helper: write a pipeline_runs row + a live_state_snapshots row.

    record_pipeline_run auto-generates a run_id; we want a deterministic
    one for ordering tests, so insert directly.
    """
    conn.execute(
        """INSERT INTO pipeline_runs
              (run_id, run_date, run_type, strategy)
           VALUES (?, ?, 'live', ?)""",
        (run_id, run_date.isoformat(), strategy),
    )
    record_live_state_snapshot(
        conn, run_id=run_id, run_date=run_date, strategy=strategy,
        state=state,
    )


# ── Basic load ────────────────────────────────────────────────────────────────

class TestLoadLatestLiveState:
    def test_returns_none_on_empty_db(self):
        conn = _db()
        assert load_latest_live_state(conn) is None

    def test_returns_none_on_none_conn(self):
        assert load_latest_live_state(None) is None

    def test_returns_state_dict(self):
        conn = _db()
        today = datetime.date(2026, 4, 26)
        state = {
            "regime": "BULL_CALM",
            "high_water_mark": 10250.18,
            "sell_streaks": {"GOOG": 0, "PLTR": 1},
        }
        _record_snapshot(conn, run_id="r1", run_date=today,
                         strategy="renquant_104", state=state)
        loaded = load_latest_live_state(conn)
        assert loaded == state

    def test_returns_latest_snapshot_when_multiple(self):
        conn = _db()
        d1 = datetime.date(2026, 4, 24)
        d2 = datetime.date(2026, 4, 26)
        _record_snapshot(conn, run_id="r1", run_date=d1,
                         strategy="renquant_104",
                         state={"regime": "BULL_CALM",  "tag": "old"})
        _record_snapshot(conn, run_id="r2", run_date=d2,
                         strategy="renquant_104",
                         state={"regime": "BULL_CALM",  "tag": "new"})
        loaded = load_latest_live_state(conn)
        assert loaded["tag"] == "new"


# ── Strategy filter ───────────────────────────────────────────────────────────

class TestStrategyFilter:
    def test_filters_by_strategy(self):
        conn = _db()
        today = datetime.date(2026, 4, 26)
        _record_snapshot(conn, run_id="r-104", run_date=today,
                         strategy="renquant_104",
                         state={"tag": "104"})
        _record_snapshot(conn, run_id="r-103", run_date=today,
                         strategy="renquant_103",
                         state={"tag": "103"})
        loaded = load_latest_live_state(conn, strategy="renquant_104")
        assert loaded["tag"] == "104"

    def test_returns_none_when_strategy_has_no_snapshots(self):
        conn = _db()
        today = datetime.date(2026, 4, 26)
        _record_snapshot(conn, run_id="r1", run_date=today,
                         strategy="renquant_104", state={"tag": "104"})
        assert load_latest_live_state(conn, strategy="ghost") is None


# ── max_age_days guard ────────────────────────────────────────────────────────

class TestMaxAgeGuard:
    def test_returns_none_when_snapshot_too_old(self):
        conn = _db()
        # Insert a snapshot 30 days old; max_age_days=14 → reject.
        old = datetime.date.today() - datetime.timedelta(days=30)
        _record_snapshot(conn, run_id="ancient", run_date=old,
                         strategy="renquant_104", state={"tag": "old"})
        assert load_latest_live_state(conn, max_age_days=14) is None

    def test_returns_state_when_snapshot_within_age(self):
        conn = _db()
        recent = datetime.date.today() - datetime.timedelta(days=2)
        _record_snapshot(conn, run_id="recent", run_date=recent,
                         strategy="renquant_104", state={"tag": "recent"})
        loaded = load_latest_live_state(conn, max_age_days=14)
        assert loaded is not None
        assert loaded["tag"] == "recent"

    def test_no_age_guard_returns_any_age(self):
        """max_age_days=None → no age check (default behavior)."""
        conn = _db()
        old = datetime.date.today() - datetime.timedelta(days=365)
        _record_snapshot(conn, run_id="ancient", run_date=old,
                         strategy="renquant_104", state={"tag": "year-old"})
        loaded = load_latest_live_state(conn)
        assert loaded is not None
        assert loaded["tag"] == "year-old"


# ── Defensive paths ───────────────────────────────────────────────────────────

class TestDefensive:
    def test_corrupt_json_blob_returns_none(self):
        """If the state_json blob in db is malformed, return None
        (don't propagate JSONDecodeError to caller — the caller's
        fallback is empty state)."""
        conn = _db()
        # Manually insert a row with bad JSON. Use the helper to write
        # the pipeline_runs row, then a hand-crafted snapshot row.
        conn.execute(
            """INSERT INTO pipeline_runs
                  (run_id, run_date, run_type, strategy)
               VALUES ('bad', '2026-04-26', 'live', 'renquant_104')""",
        )
        conn.execute(
            """INSERT INTO live_state_snapshots
                  (run_id, run_date, strategy, state_json)
               VALUES (?, ?, ?, ?)""",
            ("bad", "2026-04-26", "renquant_104", "{not valid json"),
        )
        assert load_latest_live_state(conn) is None

    def test_table_missing_returns_none(self):
        """Fresh db with no live_state_snapshots table → None, not crash."""
        conn = sqlite3.connect(":memory:")   # no schema!
        assert load_latest_live_state(conn) is None

    def test_streak_recovery_round_trip(self):
        """End-to-end: streaks written via record_live_state_snapshot
        roundtrip cleanly through load_latest_live_state."""
        conn = _db()
        today = datetime.date.today()
        original = {
            "sell_streaks": {"GOOG": 3, "PLTR": 2, "AMZN": 0},
            "high_water_mark": 10250.18,
            "entry_dates": {"GOOG": "2026-04-20", "PLTR": "2026-04-17"},
            "regime": "BULL_CALM",
            "regime_confidence": 0.54,
        }
        _record_snapshot(conn, run_id="recovery-test", run_date=today,
                         strategy="renquant_104", state=original)
        recovered = load_latest_live_state(
            conn, strategy="renquant_104", max_age_days=1,
        )
        assert recovered == original
        # Specifically: streaks survived
        assert recovered["sell_streaks"]["GOOG"] == 3
