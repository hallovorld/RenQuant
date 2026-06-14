"""Post-scoring drift audit integration tests (L6 sidecar)."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_STRATEGY_DIR = _Path(__file__).resolve().parent.parent / 'backtesting' / 'renquant_104'
if str(_STRATEGY_DIR) not in _sys.path:
    _sys.path.insert(0, str(_STRATEGY_DIR))

import datetime as dt
import sqlite3

import numpy as np

from kernel.alert_lifecycle import AlertBook
from kernel.persistence import get_connection
from kernel.score_audit import run_score_drift_audit

RUN_DATE = dt.date(2026, 6, 13)


def _db_with(tmp_path, runs):
    conn = get_connection({"persistence": {"enabled": True,
                                           "db_path": str(tmp_path / "runs.db")}})
    for rid, scores in runs:
        # candidate_scores.run_id FKs pipeline_runs — seed the parent row.
        run_date = rid.split("-full")[0] if "-full" in rid else "2026-06-01"
        conn.execute("INSERT OR IGNORE INTO pipeline_runs "
                     "(run_id, run_date, run_type) VALUES (?, ?, 'sim')",
                     (rid, run_date))
        conn.executemany("INSERT INTO candidate_scores (run_id, rank_score) "
                         "VALUES (?, ?)", [(rid, s) for s in scores])
    conn.commit()
    return conn


def _stable_runs(n=5, size=140, seed=0):
    rng = np.random.RandomState(seed)
    return [(f"2026-06-{d:02d}-full", rng.normal(0.5, 0.1, size).tolist())
            for d in range(1, n + 1)]


def _collapse_runs():
    runs = _stable_runs(4)
    runs.append(("2026-06-06-full", [0.5] * 140))
    return runs


class TestAuditLoop:
    def test_measures_and_persists(self, tmp_path):
        conn = _db_with(tmp_path, _stable_runs())
        res = run_score_drift_audit(conn, run_id="r1", run_date=RUN_DATE)
        assert res.report is not None and res.persisted
        n = conn.execute("SELECT COUNT(*) FROM score_drift_audits").fetchone()[0]
        assert n == 1

    def test_critical_escalates_in_book(self, tmp_path):
        conn = _db_with(tmp_path, _collapse_runs())
        book = AlertBook(escalate_after_days=5)
        res = run_score_drift_audit(conn, run_id="r1", run_date=RUN_DATE, book=book)
        assert res.report.severity == "CRITICAL"
        assert res.alert_state == "WARN"          # first sighting → WARN (new incident)
        assert res.notifications == 1

    def test_repeated_critical_is_one_incident(self, tmp_path):
        conn = _db_with(tmp_path, _collapse_runs())
        book = AlertBook(escalate_after_days=3)
        states = []
        for i in range(6):
            res = run_score_drift_audit(conn, run_id=f"r{i}",
                                        run_date=RUN_DATE + dt.timedelta(days=i),
                                        book=book)
            states.append((res.alert_state, res.notifications))
        # escalates to CRITICAL once, total 2 notifications across the run
        assert states[-1][0] == "CRITICAL"
        assert states[-1][1] == 2                  # NEW + escalation, not 6

    def test_info_resolves_incident(self, tmp_path):
        book = AlertBook(escalate_after_days=5)
        # seed an open incident
        book.observe("score_drift", "panel", "CRITICAL:psi~0.5", RUN_DATE)
        conn = _db_with(tmp_path, _stable_runs())  # stable → INFO
        res = run_score_drift_audit(conn, run_id="r1",
                                    run_date=RUN_DATE + dt.timedelta(days=1),
                                    book=book)
        assert res.report.ok
        assert res.alert_state == "RESOLVED"

    def test_info_does_not_resolve_unrelated_incidents(self, tmp_path):
        # REGRESSION (PR #137 review): a clean score-drift run must resolve
        # ONLY its own (audit, scope) incident. Incidents from other audits,
        # and other scopes of the SAME audit, must survive untouched —
        # otherwise a stable panel score keeps silently clearing active
        # broker/reconciliation/shadow alarms.
        book = AlertBook(escalate_after_days=5)
        book.observe("score_drift", "panel", "CRITICAL:psi~0.5", RUN_DATE)
        book.observe("broker_reconciliation", "book", "EXT_SELL:NVDA", RUN_DATE)
        book.observe("score_drift", "shadow", "WARN:psi~0.3", RUN_DATE)
        conn = _db_with(tmp_path, _stable_runs())  # stable → INFO
        res = run_score_drift_audit(conn, run_id="r1", scope="panel",
                                    run_date=RUN_DATE + dt.timedelta(days=1),
                                    book=book)
        assert res.report.ok
        assert res.alert_state == "RESOLVED"   # panel score_drift cleared
        open_keys = {a.key for a in book.open_incidents()}
        # unrelated audit + different scope of the same audit stay OPEN
        assert ("broker_reconciliation", "book", "EXT_SELL:NVDA") in open_keys
        assert ("score_drift", "shadow", "WARN:psi~0.3") in open_keys
        # the targeted incident is the only one resolved
        assert ("score_drift", "panel", "CRITICAL:psi~0.5") not in open_keys

    def test_insufficient_data_noop(self, tmp_path):
        conn = _db_with(tmp_path, [("2026-06-01-full", [0.5] * 40)])
        res = run_score_drift_audit(conn, run_id="r1", run_date=RUN_DATE)
        assert res.report is None and not res.persisted

    def test_none_conn(self):
        res = run_score_drift_audit(None, run_id="r", run_date=RUN_DATE)
        assert res.report is None
