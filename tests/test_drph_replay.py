"""DRPH replay executor tests (eng plan §IV + S2 item 5).

End-to-end determinism is proven operationally (three independent
single-day sims of 2026-06-10 produced identical case id
30243d8ba8b74306 / decisions_sha a68475943e13bac5 at PR time — the
committed corpus case below IS that capture). These tests pin the
cheap-but-load-bearing pieces: the data-drift fingerprint semantics,
schema-vintage resilience of the decision extraction, case-kind
separation, and corpus integrity.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))
sys.path.insert(0, str(REPO / "scripts"))

from drph_capture import TICKER_DECISION_COLS, extract_decisions  # noqa: E402
from drph_replay import _ohlcv_fingerprint  # noqa: E402
from kernel.drph import ReplayCase  # noqa: E402

SIM_CASE = REPO / "tests" / "drph_corpus" / "sim_2026-06-10"


def _frame(closes, start="2026-06-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"close": closes, "high": closes, "low": closes},
                        index=idx)


class TestOhlcvFingerprint:

    def test_stable_across_identical_copies(self):
        a = {"MU": _frame([1.0, 2.0, 3.0, 4.0])}
        b = {"MU": _frame([1.0, 2.0, 3.0, 4.0])}
        d = "2026-06-04"
        assert _ohlcv_fingerprint(a, d) == _ohlcv_fingerprint(b, d)

    def test_detects_restatement_before_date(self):
        a = {"MU": _frame([1.0, 2.0, 3.0, 4.0])}
        b = {"MU": _frame([1.0, 2.5, 3.0, 4.0])}  # restated bar
        d = "2026-06-04"
        assert _ohlcv_fingerprint(a, d) != _ohlcv_fingerprint(b, d)

    def test_ignores_appends_after_date(self):
        # Ordinary forward appends must NOT trip DATA-DRIFT.
        a = {"MU": _frame([1.0, 2.0, 3.0])}
        b = {"MU": _frame([1.0, 2.0, 3.0, 99.0, 98.0])}
        d = "2026-06-03"  # third business day
        assert _ohlcv_fingerprint(a, d) == _ohlcv_fingerprint(b, d)


class TestSchemaVintageResilience:
    """Fresh sim dbs use the umbrella DDL, which lacks columns live dbs
    gained via migration (found live: model_admission_ok et al.). The
    snapshot surface must stay fixed regardless of db vintage."""

    def _narrow_db(self, tmp_path):
        db = tmp_path / "narrow.db"
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE pipeline_runs (
            run_id TEXT PRIMARY KEY, run_date DATE, run_type TEXT,
            regime TEXT, confidence REAL, buy_blocked INTEGER,
            skip_buys INTEGER, bear_only INTEGER, n_candidates INTEGER,
            n_exits INTEGER, n_buys INTEGER, counters_json TEXT,
            run_bundle_json TEXT)""")
        # Deliberately NARROW: only a subset of TICKER_DECISION_COLS.
        conn.execute("""CREATE TABLE ticker_daily_state (
            run_id TEXT, date TEXT, ticker TEXT, regime TEXT,
            confidence REAL, selected INTEGER, blocked_by TEXT)""")
        conn.execute("INSERT INTO pipeline_runs VALUES "
                     "('r1','2026-06-10','sim','BULL_CALM',0.8,0,0,0,"
                     "1,0,1,'{}','{}')")
        conn.execute("INSERT INTO ticker_daily_state VALUES "
                     "('r1','2026-06-10','MU','BULL_CALM',0.8,1,NULL)")
        conn.commit()
        return conn

    def test_missing_columns_read_as_none(self, tmp_path):
        conn = self._narrow_db(tmp_path)
        decisions = extract_decisions(conn, "r1")
        row = decisions["tickers"][0]
        assert set(row) == set(TICKER_DECISION_COLS), \
            "snapshot surface must be fixed across db vintages"
        assert row["selected"] == 1
        assert row["model_admission_ok"] is None
        assert row["kelly_target_pct"] is None


class TestCaseKindSeparation:
    """Live-captured cases are forensic anchors; only sim_replay cases
    may gate refactors."""

    def test_verify_rejects_live_case(self, tmp_path, monkeypatch):
        import drph_replay

        case = ReplayCase(tmp_path / "live_case")
        case.write(inputs={"capture_meta": {"kind": "live"},
                           "ohlcv_fingerprint": {}},
                   expected_decisions={"book": {}, "tickers": []})

        class _Args:
            case = str(tmp_path / "live_case")

        with pytest.raises(SystemExit, match="forensic anchors"):
            drph_replay.cmd_verify(_Args())


class TestCommittedSimCase:

    def test_sim_case_integrity(self):
        assert SIM_CASE.exists(), "committed sim gate case missing"
        assert ReplayCase(SIM_CASE).check_integrity() == []

    def test_sim_case_meta(self):
        meta = json.loads((SIM_CASE / "inputs" / "capture_meta.json").read_text())
        assert meta["kind"] == "sim_replay"
        assert meta["date"] == "2026-06-10"
        assert meta["seed"] == 44
        expected = ReplayCase(SIM_CASE).expected()
        assert len(expected["tickers"]) == 142
