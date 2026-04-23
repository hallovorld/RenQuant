"""Training-run audit tests (SQLite + JSONL) — user spec.

Per the April 22 session:
    "每次训练消耗的时间和数据量要留下记录，写到数据库里，而且还要有log."

Covers:
  * Schema includes the Round-5 audit columns (elapsed_sec, trigger,
    n_tickers, n_dates, n_features, device, deterministic,
    training_window_years, notes).
  * record_training_run writes both SQLite and a daily JSONL file.
  * Disabled persistence still produces a JSONL log (operator visibility).
  * Deterministic bool coerces to 0/1 in SQLite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_conn(tmp_path: Path):
    from kernel.persistence import get_connection
    cfg = {"persistence": {"enabled": True, "db_path": str(tmp_path / "runs.db")}}
    return get_connection(cfg)


class TestSchema:
    def test_new_columns_present(self, tmp_path):
        conn = _make_conn(tmp_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(training_runs)")}
        for required in ("elapsed_sec", "trigger", "n_tickers", "n_dates",
                          "n_features", "device", "deterministic",
                          "training_window_years", "notes"):
            assert required in cols, f"missing column: {required}"


class TestRecordTrainingRun:
    def test_sqlite_round_trip_with_new_fields(self, tmp_path):
        from kernel.persistence import record_training_run
        conn = _make_conn(tmp_path)
        rid = record_training_run(
            conn,
            strategy             = "renquant_104",
            artifact_type        = "panel-transformer",
            oos_mean_ic          = 0.078,
            train_ic             = 0.22,
            n_rows               = 80000,
            n_tickers            = 38,
            n_dates              = 1260,
            n_features           = 25,
            elapsed_sec          = 156.4,
            trigger              = "anomaly_spy_2pct",
            device               = "mps",
            deterministic        = True,
            training_window_years= 5.0,
            notes                = "VIX jump 6.4%",
            also_log_jsonl       = False,
        )
        assert rid is not None
        row = conn.execute(
            """SELECT artifact_type, oos_mean_ic, train_ic, n_rows,
                      n_tickers, n_dates, n_features, elapsed_sec, trigger,
                      device, deterministic, training_window_years, notes
                 FROM training_runs WHERE run_id = ?""",
            (rid,),
        ).fetchone()
        assert row == (
            "panel-transformer", 0.078, 0.22, 80000, 38, 1260, 25,
            156.4, "anomaly_spy_2pct", "mps", 1,  # True → 1
            5.0, "VIX jump 6.4%",
        )

    def test_jsonl_log_is_written_alongside_sqlite(self, tmp_path):
        from kernel.persistence import record_training_run
        log_dir = tmp_path / "logs"
        conn = _make_conn(tmp_path)

        record_training_run(
            conn,
            strategy      = "renquant_104",
            artifact_type = "ngboost-head",
            n_rows        = 80000,
            n_features    = 25,
            elapsed_sec   = 270.1,
            trigger       = "scheduled_weekly",
            device        = "cpu",
            also_log_jsonl= True,
            jsonl_dir     = log_dir,
        )
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1
        row = json.loads(files[0].read_text().strip())
        assert row["artifact_type"] == "ngboost-head"
        assert row["elapsed_sec"]   == pytest.approx(270.1)
        assert row["trigger"]       == "scheduled_weekly"

    def test_jsonl_written_even_when_db_disabled(self, tmp_path):
        """Operators should still see a line in JSONL even if the DB is off."""
        from kernel.persistence import record_training_run
        log_dir = tmp_path / "logs"
        record_training_run(
            conn=None,
            strategy="renquant_104", artifact_type="panel-ltr",
            elapsed_sec=12.3, trigger="manual",
            also_log_jsonl=True, jsonl_dir=log_dir,
        )
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1
        row = json.loads(files[0].read_text().strip())
        assert row["trigger"] == "manual"

    def test_multiple_runs_append_to_same_daily_file(self, tmp_path):
        from kernel.persistence import record_training_run
        log_dir = tmp_path / "logs"
        conn = _make_conn(tmp_path)
        for i in range(3):
            record_training_run(
                conn, strategy="renquant_104", artifact_type="panel-ltr",
                elapsed_sec=float(i), trigger="manual",
                also_log_jsonl=True, jsonl_dir=log_dir,
            )
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1, "all 3 runs land in one daily file"
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 3
