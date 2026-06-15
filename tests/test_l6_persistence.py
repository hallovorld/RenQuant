"""Round-trip tests for the umbrella L6 audit persistence (mirror of pipeline).

Covers the score-drift audit log and the alert-incident book added to
kernel.persistence so the L6 audit sidecar can run against the umbrella runs
DB: score_drift_audits accrues drift history, alert_incidents persists the
escalation state machine across restarts (the whole point of the lifecycle).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace as NS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.alert_lifecycle import AlertBook  # noqa: E402
from kernel.persistence import (  # noqa: E402
    get_connection,
    load_alert_book,
    record_score_drift_audit,
    save_alert_book,
)

RUN_DATE = dt.date(2026, 6, 14)


def _conn(tmp_path):
    return get_connection({"persistence": {"enabled": True,
                                           "db_path": str(tmp_path / "runs.db")}})


class TestSchemaCreated:
    def test_l6_tables_exist_on_fresh_db(self, tmp_path):
        conn = _conn(tmp_path)
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "score_drift_audits" in names
        assert "alert_incidents" in names


class TestScoreDriftAudit:
    def test_append_and_read_back(self, tmp_path):
        conn = _conn(tmp_path)
        rep = NS(psi=0.27, severity="WARN", n_baseline=140, n_current=140)
        assert record_score_drift_audit(conn, run_id="r1", run_date=RUN_DATE, report=rep) == 1
        row = conn.execute(
            "SELECT run_id, severity, psi, n_baseline FROM score_drift_audits").fetchone()
        assert row == ("r1", "WARN", 0.27, 140)

    def test_nan_psi_stored_as_null(self, tmp_path):
        conn = _conn(tmp_path)
        rep = NS(psi=float("nan"), severity="CRITICAL", n_baseline=10, n_current=10)
        record_score_drift_audit(conn, run_id="r1", run_date=RUN_DATE, report=rep)
        assert conn.execute("SELECT psi FROM score_drift_audits").fetchone()[0] is None

    def test_none_conn_and_none_report_are_noops(self, tmp_path):
        assert record_score_drift_audit(None, run_id="r", run_date=RUN_DATE, report=NS()) == 0
        conn = _conn(tmp_path)
        assert record_score_drift_audit(conn, run_id="r", run_date=RUN_DATE, report=None) == 0

    def test_append_only_accrues_history(self, tmp_path):
        conn = _conn(tmp_path)
        for i in range(3):
            record_score_drift_audit(
                conn, run_id=f"r{i}", run_date=RUN_DATE + dt.timedelta(days=i),
                report=NS(psi=0.1 * i, severity="INFO", n_baseline=5, n_current=5))
        assert conn.execute("SELECT COUNT(*) FROM score_drift_audits").fetchone()[0] == 3


class TestAlertBookPersistence:
    def test_save_then_load_round_trips_incident(self, tmp_path):
        conn = _conn(tmp_path)
        book = AlertBook(escalate_after_days=5)
        book.observe("score_drift", "panel", "CRITICAL:psi~0.5", RUN_DATE)
        assert save_alert_book(conn, book) == 1

        restored = load_alert_book(conn, escalate_after_days=5)
        a = restored.alerts[("score_drift", "panel", "CRITICAL:psi~0.5")]
        assert a.state == "WARN" and a.notifications == 1
        assert a.first_seen == RUN_DATE and a.last_seen == RUN_DATE

    def test_upsert_continues_incident_across_restart(self, tmp_path):
        conn = _conn(tmp_path)
        # day 1: incident opens
        book = AlertBook(escalate_after_days=3)
        book.observe("score_drift", "panel", "psi~0.5", RUN_DATE)
        save_alert_book(conn, book)
        # restart: reload, re-observe over the escalation window
        for i in range(1, 5):
            book = load_alert_book(conn, escalate_after_days=3)
            book.observe("score_drift", "panel", "psi~0.5", RUN_DATE + dt.timedelta(days=i))
            save_alert_book(conn, book)
        final = load_alert_book(conn, escalate_after_days=3)
        a = final.alerts[("score_drift", "panel", "psi~0.5")]
        assert a.state == "CRITICAL"          # escalated, not reset-to-NEW each restart
        assert a.first_seen == RUN_DATE        # original open date preserved
        # exactly one row — upsert, not duplicate inserts
        assert conn.execute("SELECT COUNT(*) FROM alert_incidents").fetchone()[0] == 1

    def test_resolved_incident_recurrence_resets_first_seen_after_reload(self, tmp_path):
        conn = _conn(tmp_path)
        book = AlertBook(escalate_after_days=5)
        book.observe("score_drift", "panel", "psi~0.5", RUN_DATE)
        save_alert_book(conn, book)

        book.resolve_audit_scope("score_drift", "panel", RUN_DATE + dt.timedelta(days=1))
        save_alert_book(conn, book)

        recurrence_date = RUN_DATE + dt.timedelta(days=20)
        book.observe("score_drift", "panel", "psi~0.5", recurrence_date)
        save_alert_book(conn, book)

        restored = load_alert_book(conn, escalate_after_days=5)
        alert = restored.alerts[("score_drift", "panel", "psi~0.5")]
        assert alert.first_seen == recurrence_date
        assert alert.state == "WARN"
        assert alert.notifications == 1

        next_day = restored.observe(
            "score_drift",
            "panel",
            "psi~0.5",
            recurrence_date + dt.timedelta(days=1),
        )
        assert next_day.state == "WARN"
        assert next_day.notifications == 1

    def test_load_empty_book_when_no_rows(self, tmp_path):
        conn = _conn(tmp_path)
        book = load_alert_book(conn)
        assert book.alerts == {}

    def test_none_conn_noops(self, tmp_path):
        assert save_alert_book(None, AlertBook()) == 0
        assert load_alert_book(None).alerts == {}
