"""runner.py decomposition slice 1 — state_store contract tests.

The sim replay does not cover the live adapter, so this move is gated
by tests: JSON-first, corrupt-JSON → DB fallback, DB age cap honored by
delegation, hot-cache write-back, atomic save (LS-ATOM).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.state_store import load_live_state, save_live_state_atomic  # noqa: E402


class TestLoad:

    def test_json_first(self, tmp_path):
        sf = tmp_path / "live_state.alpaca.json"
        sf.write_text(json.dumps({"regime": "BULL_CALM"}))
        state = load_live_state(sf, {}, tmp_path)
        assert state["regime"] == "BULL_CALM"

    def test_missing_file_empty_state_when_db_disabled(self, tmp_path):
        sf = tmp_path / "live_state.alpaca.json"
        state = load_live_state(sf, {}, tmp_path)  # persistence disabled
        assert state == {}

    def test_corrupt_json_falls_back_to_db(self, tmp_path):
        import datetime as dt

        from kernel.persistence import (
            get_connection, record_live_state_snapshot,
        )

        config = {"persistence": {"enabled": True,
                                  "db_path": str(tmp_path / "runs.db")},
                  "_strategy_name": "renquant_104"}
        conn = get_connection(config)
        conn.execute("INSERT INTO pipeline_runs (run_id, run_date, run_type,"
                     " strategy) VALUES ('r1', ?, 'live', 'renquant_104')",
                     (dt.date.today().isoformat(),))
        record_live_state_snapshot(conn, "r1", run_date=dt.date.today(),
                                   strategy="renquant_104",
                                   state={"regime": "BEAR", "entry_dates": {}})
        conn.close()
        sf = tmp_path / "live_state.alpaca.json"
        sf.write_text('{"regime": "BULL')  # corrupt
        state = load_live_state(sf, config, None)
        assert state["regime"] == "BEAR"
        # hot-cache write-back
        assert json.loads(sf.read_text())["regime"] == "BEAR"

    def test_db_failure_proceeds_empty(self, tmp_path):
        sf = tmp_path / "absent.json"
        config = {"persistence": {"enabled": True,
                                  "db_path": "/nonexistent/dir/x.db"}}
        state = load_live_state(sf, config, None)
        assert state == {}


class TestSave:

    def test_atomic_write_round_trip(self, tmp_path):
        sf = tmp_path / "live_state.alpaca.json"
        save_live_state_atomic(sf, {"regime": "CHOPPY", "entry_dates": {}})
        assert json.loads(sf.read_text())["regime"] == "CHOPPY"
        assert not sf.with_suffix(".json.tmp").exists()

    def test_crash_before_rename_preserves_prior(self, tmp_path, monkeypatch):
        sf = tmp_path / "live_state.alpaca.json"
        save_live_state_atomic(sf, {"v": 1})

        def _boom(self, target):
            raise OSError("simulated crash")

        monkeypatch.setattr(Path, "replace", _boom)
        try:
            save_live_state_atomic(sf, {"v": 2})
        except OSError:
            pass
        assert json.loads(sf.read_text())["v"] == 1, "LS-ATOM invariant"
