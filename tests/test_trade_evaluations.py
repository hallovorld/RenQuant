"""Tests for trade_evaluations schema + record helper + backfill script.

Roadmap §2026-04-26 Phase 1+2.

Coverage:
  * Schema migration creates the table idempotently.
  * `record_trade_evaluations` writes rows + handles missing/invalid input.
  * Primary-key (run_id, ticker, action, horizon_days) prevents duplicates.
  * INSERT OR REPLACE updates an existing row (re-eval scenario).
  * Backfill script's `_build_eval_rows` joins trades + forward returns
    and computes relative return + is_winner correctly.
  * Horizons not in DB schema are skipped, not crashed.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.persistence import (   # noqa: E402
    _SCHEMA_SQL,   # noqa: F401  (used to create test DBs)
    record_trade_evaluations,
)


@pytest.fixture
def conn():
    """Fresh in-memory SQLite with the full schema applied."""
    c = sqlite3.connect(":memory:")
    c.executescript(_SCHEMA_SQL)
    yield c
    c.close()


# ── Schema ────────────────────────────────────────────────────────────────────

class TestSchema:
    def test_table_exists_after_schema_apply(self, conn):
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trade_evaluations'"
        ).fetchall()
        assert len(rows) == 1

    def test_schema_columns_match_design(self, conn):
        cols = {
            row[1]: row[2] for row in
            conn.execute("PRAGMA table_info(trade_evaluations)").fetchall()
        }
        expected = {
            "run_id", "ticker", "action", "horizon_days",
            "fwd_return", "fwd_return_spy", "relative_return",
            "is_winner", "n_trade_rows", "created_at",
        }
        assert expected.issubset(set(cols.keys())), (
            f"missing cols: {expected - set(cols.keys())}"
        )

    def test_primary_key_prevents_duplicate_horizon_per_trade(self, conn):
        record_trade_evaluations(conn, [{
            "run_id": "r1", "ticker": "AAPL", "action": "buy", "horizon_days": 5,
            "fwd_return": 0.02, "fwd_return_spy": 0.01,
        }])
        # Second row with same composite key — should UPSERT not error
        record_trade_evaluations(conn, [{
            "run_id": "r1", "ticker": "AAPL", "action": "buy", "horizon_days": 5,
            "fwd_return": 0.03, "fwd_return_spy": 0.01,
        }])
        n = conn.execute(
            "SELECT COUNT(*) FROM trade_evaluations "
            "WHERE run_id='r1' AND ticker='AAPL' AND action='buy' AND horizon_days=5"
        ).fetchone()[0]
        assert n == 1, "PRIMARY KEY must prevent duplicate (run, ticker, action, horizon)"
        # New value should win after upsert
        v = conn.execute(
            "SELECT fwd_return FROM trade_evaluations "
            "WHERE run_id='r1' AND ticker='AAPL'"
        ).fetchone()[0]
        assert v == pytest.approx(0.03)


# ── record_trade_evaluations helper ──────────────────────────────────────────

class TestRecordHelper:
    def test_basic_insert(self, conn):
        n = record_trade_evaluations(conn, [{
            "run_id": "r1", "ticker": "AAPL", "action": "buy",
            "horizon_days": 5,
            "fwd_return": 0.02, "fwd_return_spy": 0.01,
            "relative_return": 0.01, "is_winner": 1, "n_trade_rows": 1,
        }])
        assert n == 1
        row = conn.execute("SELECT * FROM trade_evaluations").fetchone()
        assert row is not None

    def test_skips_row_missing_required_keys(self, conn):
        # Missing ticker
        n = record_trade_evaluations(conn, [{
            "run_id": "r1", "action": "buy", "horizon_days": 5,
        }])
        assert n == 0
        # Missing action
        n2 = record_trade_evaluations(conn, [{
            "run_id": "r1", "ticker": "AAPL", "horizon_days": 5,
        }])
        assert n2 == 0

    def test_skips_invalid_action(self, conn):
        n = record_trade_evaluations(conn, [{
            "run_id": "r1", "ticker": "AAPL", "action": "REBALANCE",  # not buy/sell
            "horizon_days": 5,
        }])
        assert n == 0

    def test_skips_non_positive_horizon(self, conn):
        n = record_trade_evaluations(conn, [{
            "run_id": "r1", "ticker": "AAPL", "action": "buy", "horizon_days": 0,
        }])
        assert n == 0
        n2 = record_trade_evaluations(conn, [{
            "run_id": "r1", "ticker": "AAPL", "action": "buy", "horizon_days": -3,
        }])
        assert n2 == 0

    def test_handles_none_returns(self, conn):
        """Caller may pass None for fwd_return etc. — schema allows NULL."""
        n = record_trade_evaluations(conn, [{
            "run_id": "r1", "ticker": "X", "action": "sell",
            "horizon_days": 1,
            # No fwd_return / fwd_return_spy / etc.
        }])
        assert n == 1
        row = conn.execute(
            "SELECT fwd_return, fwd_return_spy, relative_return, is_winner "
            "FROM trade_evaluations WHERE run_id='r1'"
        ).fetchone()
        assert all(v is None for v in row)

    def test_none_conn_returns_zero_no_crash(self):
        assert record_trade_evaluations(None, [{
            "run_id": "r", "ticker": "X", "action": "buy", "horizon_days": 5,
        }]) == 0


# ── Backfill script — _build_eval_rows ───────────────────────────────────────

class TestBackfillBuildRows:
    """Script: scripts/backfill_trade_evaluations.py"""

    @pytest.fixture
    def script_module(self):
        spec = importlib.util.spec_from_file_location(
            "_backfill", REPO_ROOT / "scripts" / "backfill_trade_evaluations.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # type: ignore[union-attr]
        return mod

    @pytest.fixture
    def populated_conn(self, conn):
        """Add some pipeline_runs + trades + ticker_forward_returns rows
        so the JOIN has something to compute."""
        # Use a single date so SPY + ticker forward returns line up
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, run_date, regime, run_type) "
            "VALUES (?, ?, ?, ?)",
            ("r1", "2026-04-01", "BULL_CALM", "live"),
        )
        conn.execute(
            "INSERT INTO trades (run_id, ticker, action, shares, price, invest) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("r1", "AAPL", "buy", 10, 200.0, 2000.0),
        )
        # Forward returns at 5d horizon
        conn.execute(
            "INSERT INTO ticker_forward_returns "
            "(as_of_date, ticker, fwd_1d, fwd_5d) VALUES (?, ?, ?, ?)",
            ("2026-04-01", "AAPL", 0.005, 0.04),
        )
        conn.execute(
            "INSERT INTO ticker_forward_returns "
            "(as_of_date, ticker, fwd_1d, fwd_5d) VALUES (?, ?, ?, ?)",
            ("2026-04-01", "SPY", 0.002, 0.015),
        )
        conn.commit()
        return conn

    def test_join_produces_correct_relative_return(self, populated_conn, script_module):
        rows = script_module._build_eval_rows(
            populated_conn, horizons=[5], since=None, benchmark="SPY",
        )
        assert len(rows) == 1
        r = rows[0]
        assert r["ticker"] == "AAPL"
        assert r["action"] == "buy"
        assert r["horizon_days"] == 5
        assert r["fwd_return"] == pytest.approx(0.04)
        assert r["fwd_return_spy"] == pytest.approx(0.015)
        # relative = ticker - benchmark = 0.04 - 0.015 = 0.025
        assert r["relative_return"] == pytest.approx(0.025)
        assert r["is_winner"] == 1   # ticker beat benchmark

    def test_unsupported_horizon_skipped(self, populated_conn, script_module):
        # Horizon 7d is not in DB schema → skipped
        rows = script_module._build_eval_rows(
            populated_conn, horizons=[7, 14, 28], since=None, benchmark="SPY",
        )
        assert rows == []   # all unsupported horizons → no rows

    def test_supports_5d_only_when_mixed_with_unsupported(
        self, populated_conn, script_module,
    ):
        # Mix of supported + unsupported → only supported rows produced
        rows = script_module._build_eval_rows(
            populated_conn, horizons=[5, 7, 10], since=None, benchmark="SPY",
        )
        # Only horizons 5 + 10 are supported in schema. AAPL has fwd_5d set
        # but no fwd_10d → only horizon=5 produces a row.
        horizons = sorted({r["horizon_days"] for r in rows})
        assert horizons == [5]

    def test_skips_trade_with_no_fwd_return(self, populated_conn, script_module):
        """If ticker has no forward return on the run_date (too recent),
        the row is skipped entirely, not written with NULLs."""
        # Add a trade for a different date with no forward returns
        populated_conn.execute(
            "INSERT INTO pipeline_runs (run_id, run_date, regime, run_type) "
            "VALUES (?, ?, ?, ?)",
            ("r2", "2026-04-30", "BULL_CALM", "live"),
        )
        populated_conn.execute(
            "INSERT INTO trades (run_id, ticker, action, shares) "
            "VALUES (?, ?, ?, ?)",
            ("r2", "MSFT", "buy", 5),
        )
        populated_conn.commit()
        rows = script_module._build_eval_rows(
            populated_conn, horizons=[5], since=None, benchmark="SPY",
        )
        # Only AAPL r1 produces a row; MSFT r2 has no fwd_5d → skipped
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"

    def test_since_filter_excludes_older_trades(self, populated_conn, script_module):
        rows = script_module._build_eval_rows(
            populated_conn, horizons=[5], since="2030-01-01",
            benchmark="SPY",
        )
        assert rows == []   # all trades older than far-future since
