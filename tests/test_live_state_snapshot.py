"""Plan S — live_state_snapshots append-only audit table.

Roadmap decision: keep live_state.json as the source of truth (fast
bootstrap, human-editable) AND append every bar's snapshot to
live_state_snapshots so historical queries like "what was
high_water_mark on 2026-04-20?" are answerable via SQL.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_conn(tmp_path: Path):
    from kernel.persistence import get_connection
    return get_connection(
        {"persistence": {"enabled": True, "db_path": str(tmp_path / "runs.db")}},
    )


class TestSchema:
    def test_table_exists(self, tmp_path):
        conn = _make_conn(tmp_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(live_state_snapshots)")}
        assert cols == {
            "run_id", "run_date", "strategy", "regime", "confidence",
            "high_water_mark", "cash", "portfolio_value", "n_holdings",
            "state_json", "created_at",
        }

    def test_primary_key_is_run_id(self, tmp_path):
        conn = _make_conn(tmp_path)
        pk = [r[1] for r in conn.execute("PRAGMA table_info(live_state_snapshots)")
              if r[5]]
        assert pk == ["run_id"]


class TestRecordLiveStateSnapshot:
    def test_insert_and_read_back(self, tmp_path):
        from kernel.persistence import (
            record_live_state_snapshot, record_pipeline_run,
        )
        conn = _make_conn(tmp_path)
        rid = record_pipeline_run(
            conn, run_type="live", run_date=datetime.date(2026, 4, 24),
            strategy="renquant_104",
        )
        record_live_state_snapshot(
            conn, rid,
            run_date = datetime.date(2026, 4, 24),
            strategy = "renquant_104",
            state    = {
                "regime":            "BULL_CALM",
                "regime_confidence": 0.82,
                "high_water_mark":   125_000.0,
                "entry_dates":       {"NVDA": "2026-03-01"},
                "regime_state":      {"countdown": 0, "in_transition": False},
            },
            cash            = 12_500.0,
            portfolio_value = 125_000.0,
            n_holdings      = 6,
        )
        row = conn.execute(
            """SELECT regime, confidence, high_water_mark, cash,
                      portfolio_value, n_holdings
                 FROM live_state_snapshots WHERE run_id=?""",
            (rid,),
        ).fetchone()
        assert row == ("BULL_CALM", 0.82, 125_000.0, 12_500.0, 125_000.0, 6)

    def test_state_json_roundtrip(self, tmp_path):
        """Full state dict must be retrievable from state_json for deep audit."""
        from kernel.persistence import (
            record_live_state_snapshot, record_pipeline_run,
        )
        conn = _make_conn(tmp_path)
        rid = record_pipeline_run(
            conn, run_type="live", run_date=datetime.date(2026, 4, 24),
        )
        original = {
            "regime":            "CHOPPY",
            "regime_confidence": 0.48,
            "high_water_mark":   99_000.0,
            "entry_dates":       {"NVDA": "2026-03-01", "AAPL": "2026-03-15"},
            "sell_streaks":      {"MSFT": 2},
            "regime_state":      {"cusum_pos": 0.021, "cusum_neg": -0.015,
                                   "countdown": 1, "in_transition": True},
            "monitor_state":     {"no_trade_streak": 4},
        }
        record_live_state_snapshot(
            conn, rid,
            run_date = datetime.date(2026, 4, 24),
            state    = original,
        )
        row = conn.execute(
            "SELECT state_json FROM live_state_snapshots WHERE run_id=?",
            (rid,),
        ).fetchone()
        round_tripped = json.loads(row[0])
        assert round_tripped == original

    def test_none_conn_noop(self):
        from kernel.persistence import record_live_state_snapshot
        # Just shouldn't error — returns None
        record_live_state_snapshot(
            None, "rid",
            run_date=datetime.date(2026, 4, 24),
            state={"regime": "BULL_CALM"},
        )

    def test_none_run_id_noop(self, tmp_path):
        from kernel.persistence import record_live_state_snapshot
        conn = _make_conn(tmp_path)
        record_live_state_snapshot(
            conn, None,
            run_date=datetime.date(2026, 4, 24),
            state={"regime": "BULL_CALM"},
        )
        # Table should be empty
        n = conn.execute("SELECT COUNT(*) FROM live_state_snapshots").fetchone()[0]
        assert n == 0

    def test_upsert_overwrites_same_run_id(self, tmp_path):
        """If the same pipeline run writes twice (retries / testing), the
        last snapshot wins."""
        from kernel.persistence import (
            record_live_state_snapshot, record_pipeline_run,
        )
        conn = _make_conn(tmp_path)
        rid = record_pipeline_run(
            conn, run_type="live", run_date=datetime.date(2026, 4, 24),
        )
        record_live_state_snapshot(conn, rid,
            run_date=datetime.date(2026, 4, 24),
            state={"regime": "BULL_CALM", "high_water_mark": 100_000})
        record_live_state_snapshot(conn, rid,
            run_date=datetime.date(2026, 4, 24),
            state={"regime": "BULL_VOLATILE", "high_water_mark": 105_000})
        n = conn.execute("SELECT COUNT(*) FROM live_state_snapshots").fetchone()[0]
        assert n == 1
        regime = conn.execute(
            "SELECT regime FROM live_state_snapshots WHERE run_id=?",
            (rid,)).fetchone()[0]
        assert regime == "BULL_VOLATILE"


class TestHistoricalQueries:
    """End-to-end: the audit use case — query HWM by date."""

    def test_query_hwm_by_date(self, tmp_path):
        from kernel.persistence import (
            record_live_state_snapshot, record_pipeline_run,
        )
        conn = _make_conn(tmp_path)
        for d, hwm in [(datetime.date(2026, 4, 20), 120_000.0),
                       (datetime.date(2026, 4, 21), 122_000.0),
                       (datetime.date(2026, 4, 22), 118_000.0),
                       (datetime.date(2026, 4, 23), 125_000.0)]:
            rid = record_pipeline_run(conn, run_type="live", run_date=d)
            record_live_state_snapshot(
                conn, rid,
                run_date = d,
                state    = {"high_water_mark": hwm, "regime": "BULL_CALM"},
            )
        # "What was HWM on 2026-04-20?"
        row = conn.execute(
            "SELECT high_water_mark FROM live_state_snapshots WHERE run_date=?",
            ("2026-04-20",),
        ).fetchone()
        assert row[0] == 120_000.0
