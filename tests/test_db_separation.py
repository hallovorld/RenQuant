"""DB separation — sim and live write to separate SQLite files.

User architecture 2026-04-24:
  * data/runs.db       — permanent production data (live runner + LEAN reads)
  * data/sim_runs.db   — ephemeral notebook sim; TRUNCATEd at start of
                         each run_backtest so the 100th notebook session
                         of the day is the only one whose rows survive

Rationale: the model is evolving. Yesterday's sim decisions are not
meaningful ground truth for analyzing today's live decisions. Physical
separation prevents cross-pollution in AA statistics.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── get_connection routes by role ────────────────────────────────────────────

class TestRoleRouting:
    def _cfg(self, tmp_path):
        return {"persistence": {
            "enabled":      True,
            "db_path":      str(tmp_path / "runs.db"),
            "sim_db_path":  str(tmp_path / "sim_runs.db"),
        }}

    def test_live_role_opens_main_db(self, tmp_path):
        from kernel.persistence import get_connection
        conn = get_connection(self._cfg(tmp_path), role="live")
        assert conn is not None
        assert (tmp_path / "runs.db").exists()
        assert not (tmp_path / "sim_runs.db").exists()

    def test_sim_role_opens_sim_db(self, tmp_path):
        from kernel.persistence import get_connection
        conn = get_connection(self._cfg(tmp_path), role="sim")
        assert conn is not None
        assert (tmp_path / "sim_runs.db").exists()
        assert not (tmp_path / "runs.db").exists()

    def test_default_role_is_live(self, tmp_path):
        from kernel.persistence import get_connection
        conn = get_connection(self._cfg(tmp_path))
        assert conn is not None
        assert (tmp_path / "runs.db").exists()

    def test_sim_db_fallback_when_sim_path_missing(self, tmp_path):
        """If persistence.sim_db_path is not set, sim role falls back to
        the default `data/sim_runs.db` (relative to cwd), not silently
        writing into the live DB."""
        cfg = {"persistence": {"enabled": True,
                                "db_path": str(tmp_path / "runs.db")}}
        from kernel.persistence import _db_path
        p = _db_path(cfg, role="sim")
        assert p.name == "sim_runs.db"   # never the live filename


# ── clear_sim_tables ──────────────────────────────────────────────────────────

class TestClearSimTables:
    def _seed(self, tmp_path):
        """Populate the sim DB with rows across the 7 tables, return conn."""
        from kernel.persistence import (
            get_connection, record_pipeline_run, record_candidate_scores,
            record_trades, record_training_run, record_live_state_snapshot,
            record_forward_returns,
        )
        from types import SimpleNamespace
        conn = get_connection({"persistence": {
            "enabled":     True,
            "sim_db_path": str(tmp_path / "sim_runs.db"),
        }}, role="sim")
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 24),
            strategy="renquant_104",
        )
        cand = SimpleNamespace(ticker="NVDA", raw_score=5.0, rank_score=0.5,
                                rs_score=0.0, panel_score=0.5, mu=0.02, sigma=0.01)
        record_candidate_scores(conn, rid, [cand], {}, selected_tickers={"NVDA"})
        record_trades(conn, rid, [{"ticker": "NVDA", "action": "buy",
                                     "shares": 10, "price": 100}])
        record_live_state_snapshot(
            conn, rid,
            run_date=datetime.date(2026, 4, 24),
            state={"regime": "BULL_CALM"},
        )
        # DERIVED tables — these should survive the reset
        record_forward_returns(conn, [{
            "as_of_date": datetime.date(2026, 4, 24), "ticker": "NVDA",
            "close_price": 100.0, "fwd_1d": 0.01, "fwd_5d": 0.03,
            "fwd_10d": 0.05, "fwd_20d": 0.09,
        }])
        record_training_run(
            conn, strategy="renquant_104", artifact_type="panel-ltr",
            elapsed_sec=5.0, trigger="manual", also_log_jsonl=False,
        )
        return conn

    def test_resets_decision_tables(self, tmp_path):
        from kernel.persistence import clear_sim_tables
        conn = self._seed(tmp_path)
        # Sanity: all 5 decision tables populated pre-reset
        for table in ("pipeline_runs", "candidate_scores", "trades",
                      "live_state_snapshots"):
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert n > 0, f"seed failed for {table}"

        deleted = clear_sim_tables(conn)
        assert deleted > 0

        for table in ("pipeline_runs", "candidate_scores", "trades",
                      "live_state_snapshots", "rotations"):
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert n == 0, f"{table} should be empty after clear_sim_tables"

    def test_preserves_derived_tables(self, tmp_path):
        """forward_returns + training_runs are derived / audit-log —
        they're reused across sim sessions, not wiped."""
        from kernel.persistence import clear_sim_tables
        conn = self._seed(tmp_path)

        fr_before = conn.execute(
            "SELECT COUNT(*) FROM ticker_forward_returns"
        ).fetchone()[0]
        tr_before = conn.execute(
            "SELECT COUNT(*) FROM training_runs"
        ).fetchone()[0]

        clear_sim_tables(conn)

        fr_after = conn.execute(
            "SELECT COUNT(*) FROM ticker_forward_returns"
        ).fetchone()[0]
        tr_after = conn.execute(
            "SELECT COUNT(*) FROM training_runs"
        ).fetchone()[0]

        assert fr_after == fr_before
        assert tr_after == tr_before

    def test_none_conn_noop(self):
        from kernel.persistence import clear_sim_tables
        assert clear_sim_tables(None) == 0

    def test_idempotent_on_empty_db(self, tmp_path):
        """Calling twice in a row is safe."""
        from kernel.persistence import clear_sim_tables, get_connection
        conn = get_connection({"persistence": {
            "enabled":     True,
            "sim_db_path": str(tmp_path / "sim_runs.db"),
        }}, role="sim")
        assert clear_sim_tables(conn) == 0
        assert clear_sim_tables(conn) == 0
