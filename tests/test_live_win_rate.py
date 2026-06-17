"""Tests for the live-only win-rate tracker (separating LIVE from SIM)."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from live_win_rate import compute, report  # noqa: E402


def _make_db(tmp_path) -> str:
    db = tmp_path / "runs.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE pipeline_runs (run_id TEXT, run_type TEXT, run_date TEXT)")
    con.execute("CREATE TABLE trades (run_id TEXT, action TEXT, pnl_pct REAL, "
                "trade_date TEXT, exit_reason TEXT, hold_days REAL)")
    runs = [("L1", "live", "2026-06-01"), ("S1", "sim", "2026-06-01")]
    con.executemany("INSERT INTO pipeline_runs VALUES (?,?,?)", runs)
    trades = [
        # live: 3 wins (+5,+3,+4) 1 loss (-10) -> win 75%, payoff small
        ("L1", "sell", 0.05, "2026-06-02", "model_sell", 8),
        ("L1", "sell", 0.03, None,         "qp_sell", 13),   # NULL trade_date -> run_date
        ("L1", "sell", 0.04, "2026-06-03", "model_sell", 9),
        ("L1", "sell", -0.10, "2026-06-04", "stop_loss", 40),
        ("L1", "buy",  None, "2026-06-01", None, None),       # non-sell ignored
        # sim: 1 win 1 loss
        ("S1", "sell", 0.20, "2026-06-02", "max_hold", 60),
        ("S1", "sell", -0.05, "2026-06-03", "stop_loss", 40),
    ]
    con.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?)", trades)
    con.commit()
    con.close()
    return str(db)


def test_compute_basic():
    s = compute([0.05, 0.03, 0.04, -0.10])
    assert s.n == 4
    assert abs(s.win_rate - 0.75) < 1e-9
    assert abs(s.avg_win - 0.04) < 1e-9
    assert abs(s.avg_loss + 0.10) < 1e-9
    assert abs(s.payoff - 0.4) < 1e-9          # 0.04 / 0.10
    assert abs(s.expectancy - (0.75 * 0.04 + 0.25 * -0.10)) < 1e-9


def test_compute_empty():
    assert compute([]) is None
    assert compute([None, None]) is None


def test_report_splits_live_from_sim(tmp_path):
    db = _make_db(tmp_path)
    r = report(db, by_exit=True)
    assert r["live"]["n"] == 4          # the buy row excluded
    assert abs(r["live"]["win_rate"] - 0.75) < 1e-9
    assert r["sim"]["n"] == 2
    # sim payoff (0.20/0.05=4) must NOT leak into live (0.04/0.10=0.4)
    assert r["sim"]["payoff"] > r["live"]["payoff"]


def test_null_trade_date_falls_back_to_run_date(tmp_path):
    db = _make_db(tmp_path)
    r = report(db)
    # the qp_sell row had NULL trade_date; date_max should still be populated
    assert r["live"]["date_min"] is not None
    assert r["live"]["date_max"] is not None


def test_by_exit_reason_present(tmp_path):
    db = _make_db(tmp_path)
    r = report(db, by_exit=True)
    reasons = {e["exit_reason"] for e in r["live_by_exit"]}
    assert {"model_sell", "qp_sell", "stop_loss"} <= reasons
