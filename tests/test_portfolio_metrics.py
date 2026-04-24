"""Portfolio daily metrics — Sharpe/vol/DD/VaR/beta tracking.

Target-critical: user set goal APY=1.41 / Sharpe=2.0 on the golden
config. This table is the source-of-truth for measuring progress
toward the goal.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_conn(tmp_path: Path):
    from kernel.persistence import get_connection
    return get_connection(
        {"persistence": {"enabled": True, "db_path": str(tmp_path / "runs.db")}},
    )


class TestSchema:
    def test_table_exists_with_expected_columns(self, tmp_path):
        conn = _make_conn(tmp_path)
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(portfolio_daily_metrics)")}
        for required in ("as_of_date", "run_type", "strategy",
                          "portfolio_value", "daily_return",
                          "sharpe_21d", "sharpe_63d", "sharpe_252d",
                          "realized_vol_21d", "realized_vol_252d",
                          "max_drawdown_252d",
                          "var_95_21d", "var_99_21d",
                          "beta_spy_252d", "computed_at"):
            assert required in cols, f"missing column: {required}"

    def test_primary_key_is_date_runtype_strategy(self, tmp_path):
        conn = _make_conn(tmp_path)
        pk = [r[1] for r in conn.execute(
            "PRAGMA table_info(portfolio_daily_metrics)") if r[5]]
        assert pk == ["as_of_date", "run_type", "strategy"]


class TestRecordPortfolioMetrics:
    def test_insert_and_read_back(self, tmp_path):
        from kernel.persistence import record_portfolio_metrics
        conn = _make_conn(tmp_path)
        n = record_portfolio_metrics(conn, [
            {"as_of_date": datetime.date(2026, 4, 24),
             "run_type": "live", "strategy": "renquant_104",
             "portfolio_value": 125_000.0,
             "daily_return": 0.01,
             "sharpe_21d": 1.8, "sharpe_63d": 1.9, "sharpe_252d": 1.5,
             "realized_vol_21d": 0.18, "realized_vol_252d": 0.22,
             "max_drawdown_252d": -0.15,
             "var_95_21d": -0.02, "var_99_21d": -0.035,
             "beta_spy_252d": 1.1},
        ])
        assert n == 1
        row = conn.execute(
            "SELECT sharpe_252d, beta_spy_252d FROM portfolio_daily_metrics"
        ).fetchone()
        assert row == (1.5, 1.1)

    def test_upsert_on_same_key(self, tmp_path):
        """Same (date, run_type, strategy) → last write wins."""
        from kernel.persistence import record_portfolio_metrics
        conn = _make_conn(tmp_path)
        key = {"as_of_date": datetime.date(2026, 4, 24),
               "run_type": "live", "strategy": "renquant_104"}
        record_portfolio_metrics(conn, [{**key, "sharpe_252d": 1.5}])
        record_portfolio_metrics(conn, [{**key, "sharpe_252d": 1.8}])
        n = conn.execute("SELECT COUNT(*) FROM portfolio_daily_metrics").fetchone()[0]
        assert n == 1
        sharpe = conn.execute(
            "SELECT sharpe_252d FROM portfolio_daily_metrics"
        ).fetchone()[0]
        assert sharpe == 1.8

    def test_null_merge(self, tmp_path):
        """Partial row doesn't wipe existing non-null fields."""
        from kernel.persistence import record_portfolio_metrics
        conn = _make_conn(tmp_path)
        key = {"as_of_date": datetime.date(2026, 4, 24),
               "run_type": "live", "strategy": "renquant_104"}
        record_portfolio_metrics(conn, [
            {**key, "sharpe_252d": 1.5, "realized_vol_252d": 0.22}
        ])
        # Second call only updates one field — others should be preserved
        record_portfolio_metrics(conn, [
            {**key, "sharpe_252d": 1.8}
        ])
        row = conn.execute(
            "SELECT sharpe_252d, realized_vol_252d FROM portfolio_daily_metrics"
        ).fetchone()
        assert row == (1.8, 0.22)

    def test_none_conn_noop(self):
        from kernel.persistence import record_portfolio_metrics
        assert record_portfolio_metrics(None, [{
            "as_of_date": "2026-04-24", "run_type": "live",
        }]) == 0

    def test_empty_rows_noop(self, tmp_path):
        from kernel.persistence import record_portfolio_metrics
        conn = _make_conn(tmp_path)
        assert record_portfolio_metrics(conn, []) == 0


class TestSeparatesLiveFromSim:
    def test_same_date_different_run_types_coexist(self, tmp_path):
        from kernel.persistence import record_portfolio_metrics
        conn = _make_conn(tmp_path)
        d = datetime.date(2026, 4, 24)
        record_portfolio_metrics(conn, [
            {"as_of_date": d, "run_type": "live", "strategy": "renquant_104",
             "sharpe_252d": 1.5},
            {"as_of_date": d, "run_type": "sim", "strategy": "renquant_104",
             "sharpe_252d": 1.8},
        ])
        n = conn.execute("SELECT COUNT(*) FROM portfolio_daily_metrics").fetchone()[0]
        assert n == 2
