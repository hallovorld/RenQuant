"""Tests for the L6 score-drift audit sidecar wired into runner commit().

run_l6_score_audit_sidecar is the umbrella entry point that activates the L6
audit against the live runs DB after candidate_scores are persisted. It must
be AUDIT-ONLY and DEGRADE-SAFE: it persists drift history + escalates the
alert book on the happy path, and on any failure it returns None and never
raises (so it can never block a live commit()).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.runner_l6 import run_l6_score_audit_sidecar  # noqa: E402
from kernel.persistence import get_connection  # noqa: E402

RUN_DATE = dt.date(2026, 6, 14)


def _db(tmp_path, runs):
    conn = get_connection({"persistence": {"enabled": True,
                                           "db_path": str(tmp_path / "runs.db")}})
    for rid, scores in runs:
        run_date = rid.split("-full")[0] if "-full" in rid else "2026-06-01"
        conn.execute("INSERT OR IGNORE INTO pipeline_runs "
                     "(run_id, run_date, run_type) VALUES (?, ?, 'live')",
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
    runs.append(("2026-06-06-full", [0.5] * 140))  # degenerate → high PSI
    return runs


class TestActivePath:
    def test_persists_drift_row_and_returns_result(self, tmp_path):
        conn = _db(tmp_path, _stable_runs())
        res = run_l6_score_audit_sidecar(conn, run_id="2026-06-05-full",
                                         run_date=RUN_DATE)
        assert res is not None and res.report is not None
        n = conn.execute("SELECT COUNT(*) FROM score_drift_audits").fetchone()[0]
        assert n == 1

    def test_critical_drift_escalates_and_saves_book(self, tmp_path):
        conn = _db(tmp_path, _collapse_runs())
        res = run_l6_score_audit_sidecar(conn, run_id="2026-06-06-full",
                                         run_date=RUN_DATE, scope="panel")
        assert res is not None and res.report.severity == "CRITICAL"
        # the alert book was persisted (one open incident for the panel scope)
        rows = conn.execute(
            "SELECT scope, state FROM alert_incidents WHERE audit='score_drift'"
        ).fetchall()
        assert ("panel", "WARN") in rows or any(r[0] == "panel" for r in rows)


class TestDegradeSafe:
    def test_none_conn_returns_none(self):
        assert run_l6_score_audit_sidecar(None, run_id="r", run_date=RUN_DATE) is None

    def test_insufficient_data_is_noop_not_error(self, tmp_path):
        conn = _db(tmp_path, [("2026-06-01-full", [0.5] * 30)])  # too few runs
        res = run_l6_score_audit_sidecar(conn, run_id="2026-06-01-full",
                                         run_date=RUN_DATE)
        # not enough baseline → no report; must not raise
        assert res is None or res.report is None

    def test_audit_failure_never_raises(self, tmp_path, monkeypatch):
        conn = _db(tmp_path, _stable_runs())
        import adapters.runner_l6 as mod

        def _boom(*a, **k):
            raise RuntimeError("simulated audit failure")

        # force the inner audit to explode; the sidecar must swallow it.
        import kernel.score_audit as sa
        monkeypatch.setattr(sa, "run_score_drift_audit", _boom)
        assert run_l6_score_audit_sidecar(conn, run_id="x", run_date=RUN_DATE) is None

    def test_missing_l6_stack_degrades_to_none(self, tmp_path, monkeypatch):
        # simulate the lagging-pin case: the kernel import fails.
        conn = _db(tmp_path, _stable_runs())
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *a, **k):
            if name in ("kernel.score_audit", "kernel.persistence"):
                raise ImportError("L6 stack absent (lagging pin)")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        assert run_l6_score_audit_sidecar(conn, run_id="x", run_date=RUN_DATE) is None
