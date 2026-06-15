"""Round-trip tests for the umbrella reconciliation_actions audit log.

Completes the L6 audit-persistence trio (gate_verdicts + score_drift_audits +
reconciliation_actions). The table is the append-only position-divergence log:
"when did a position vanish externally / get quarantined / force-covered" must
be a SQL query. record_reconciliation_actions persists only NON-OK actions.
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

from kernel.persistence import (  # noqa: E402
    get_connection,
    record_reconciliation_actions,
)

RUN_DATE = dt.date(2026, 6, 15)


def _conn(tmp_path):
    return get_connection({"persistence": {"enabled": True,
                                           "db_path": str(tmp_path / "runs.db")}})


def _action(kind, ticker, detail="", state_qty=None, broker_qty=None):
    return NS(kind=kind, ticker=ticker, detail=detail,
              state_qty=state_qty, broker_qty=broker_qty)


class TestSchema:
    def test_table_exists_on_fresh_db(self, tmp_path):
        conn = _conn(tmp_path)
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "reconciliation_actions" in names


class TestRecord:
    def test_persists_non_ok_actions(self, tmp_path):
        conn = _conn(tmp_path)
        actions = [
            _action("EXT_SELL", "NVDA", "stop filled", 10.0, 0.0),
            _action("QUARANTINE", "MU", "unknown qty", 5.0, 7.0),
        ]
        assert record_reconciliation_actions(
            conn, run_id="r1", run_date=RUN_DATE, actions=actions) == 2
        rows = conn.execute(
            "SELECT kind, ticker, state_qty, broker_qty FROM reconciliation_actions "
            "ORDER BY ticker").fetchall()
        assert rows == [("QUARANTINE", "MU", 5.0, 7.0),
                        ("EXT_SELL", "NVDA", 10.0, 0.0)]

    def test_ok_actions_skipped(self, tmp_path):
        conn = _conn(tmp_path)
        actions = [_action("OK", "AAPL"), _action("OK", "MSFT"),
                   _action("FORCED_COVER", "TSLA", "short forced", 0.0, -3.0)]
        # only the non-OK one persists
        assert record_reconciliation_actions(
            conn, run_id="r1", run_date=RUN_DATE, actions=actions) == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM reconciliation_actions").fetchone()[0] == 1
        assert conn.execute(
            "SELECT kind FROM reconciliation_actions").fetchone()[0] == "FORCED_COVER"

    def test_all_ok_writes_nothing(self, tmp_path):
        conn = _conn(tmp_path)
        assert record_reconciliation_actions(
            conn, run_id="r1", run_date=RUN_DATE,
            actions=[_action("OK", "AAPL")]) == 0

    def test_noops(self, tmp_path):
        conn = _conn(tmp_path)
        assert record_reconciliation_actions(
            None, run_id="r1", run_date=RUN_DATE, actions=[_action("EXT_SELL", "X")]) == 0
        assert record_reconciliation_actions(
            conn, run_id=None, run_date=RUN_DATE, actions=[_action("EXT_SELL", "X")]) == 0
        assert record_reconciliation_actions(
            conn, run_id="r1", run_date=RUN_DATE, actions=[]) == 0

    def test_append_only_accrues(self, tmp_path):
        conn = _conn(tmp_path)
        for i in range(3):
            record_reconciliation_actions(
                conn, run_id=f"r{i}", run_date=RUN_DATE + dt.timedelta(days=i),
                actions=[_action("EXT_SELL", "NVDA", f"day{i}", 1.0, 0.0)])
        assert conn.execute(
            "SELECT COUNT(*) FROM reconciliation_actions").fetchone()[0] == 3
