"""Score-distribution drift audit tests (eng plan L6 sidecar item 3)."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_STRATEGY_DIR = _Path(__file__).resolve().parent.parent / 'backtesting' / 'renquant_104'
if str(_STRATEGY_DIR) not in _sys.path:
    _sys.path.insert(0, str(_STRATEGY_DIR))

import sqlite3

import numpy as np

from kernel.score_drift import (
    DriftReport,
    load_score_drift_from_db,
    psi,
    score_drift_report,
    severity,
)


class TestPsi:
    def test_identical_distributions_near_zero(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0.5, 0.1, 1000)
        assert psi(x, x) < 1e-9

    def test_collapse_is_large(self):
        rng = np.random.RandomState(0)
        base = rng.normal(0.5, 0.1, 1000)
        collapsed = np.full(100, np.median(base))
        assert psi(base, collapsed) > 0.25

    def test_out_of_range_actual_handled(self):
        base = np.linspace(0, 1, 1000)
        shifted = np.linspace(5, 6, 100)   # entirely outside baseline support
        assert psi(base, shifted) > 0.25   # no crash, large drift


class TestSeverity:
    def test_bands(self):
        assert severity(0.05) == "INFO"
        assert severity(0.10) == "WARN"      # boundary is WARN
        assert severity(0.24) == "WARN"
        assert severity(0.25) == "CRITICAL"
        assert severity(2.0) == "CRITICAL"


class TestReport:
    def test_stable_is_ok(self):
        rng = np.random.RandomState(0)
        base = rng.normal(0.5, 0.1, 600)
        cur = np.random.RandomState(1).normal(0.5, 0.1, 80)
        r = score_drift_report(base, cur)
        assert isinstance(r, DriftReport)
        assert r.severity == "INFO" and r.ok

    def test_collapse_is_critical_finding(self):
        rng = np.random.RandomState(0)
        base = rng.normal(0.5, 0.1, 600)
        r = score_drift_report(base, np.full(80, np.median(base)))
        assert r.severity == "CRITICAL" and not r.ok

    def test_too_few_baseline_is_warn_not_pass(self):
        # "could not measure stability" is a finding, never a silent pass.
        r = score_drift_report(np.array([0.1, 0.2, 0.3]), np.array([0.5]))
        assert r.severity == "WARN" and not r.ok
        assert np.isnan(r.psi)

    def test_empty_current_is_warn(self):
        r = score_drift_report(np.linspace(0, 1, 100), np.array([]))
        assert not r.ok


class TestDbLoader:
    def _db(self, runs):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE candidate_scores (run_id TEXT, rank_score REAL)")
        for rid, scores in runs:
            conn.executemany("INSERT INTO candidate_scores VALUES (?, ?)",
                             [(rid, s) for s in scores])
        conn.commit()
        return conn

    def test_latest_vs_trailing(self):
        rng = np.random.RandomState(0)
        runs = [(f"2026-06-{d:02d}-r", rng.normal(0.5, 0.1, 140).tolist())
                for d in range(1, 6)]
        conn = self._db(runs)
        r = load_score_drift_from_db(conn, trailing=20)
        assert r is not None and r.severity == "INFO"

    def test_none_when_too_few_full_runs(self):
        conn = self._db([("2026-06-01-r", [0.5] * 40),
                         ("2026-06-02-r", [0.5] * 5)])   # 2nd is partial
        assert load_score_drift_from_db(conn) is None

    def test_partial_runs_excluded_from_baseline(self):
        rng = np.random.RandomState(0)
        runs = [(f"2026-06-{d:02d}-full", rng.normal(0.5, 0.1, 140).tolist())
                for d in range(1, 5)]
        runs.append(("2026-06-05-partial", [0.9] * 10))   # sell-only, excluded
        runs.append(("2026-06-06-full", rng.normal(0.5, 0.1, 140).tolist()))
        conn = self._db(runs)
        r = load_score_drift_from_db(conn)
        assert r is not None and r.n_current == 140  # latest FULL run, not partial
