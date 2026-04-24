"""Plan AA — ticker_forward_returns table + record_forward_returns upsert.

Also exercises the backfill script end-to-end against a tiny synthetic
DB + OHLCV parquet cache.
"""
from __future__ import annotations

import datetime
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_conn(tmp_path: Path):
    from kernel.persistence import get_connection
    return get_connection(
        {"persistence": {"enabled": True, "db_path": str(tmp_path / "runs.db")}},
    )


class TestSchema:
    def test_ticker_forward_returns_table_exists(self, tmp_path):
        conn = _make_conn(tmp_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ticker_forward_returns)")}
        assert cols == {
            "as_of_date", "ticker", "close_price",
            "fwd_1d", "fwd_5d", "fwd_10d", "fwd_20d",
            "updated_at",
        }

    def test_primary_key_is_date_ticker(self, tmp_path):
        conn = _make_conn(tmp_path)
        pk = [r[1] for r in conn.execute("PRAGMA table_info(ticker_forward_returns)") if r[5]]
        assert pk == ["as_of_date", "ticker"]


class TestRecordForwardReturns:
    def test_insert_and_read_back(self, tmp_path):
        from kernel.persistence import record_forward_returns
        conn = _make_conn(tmp_path)
        n = record_forward_returns(conn, [
            {"as_of_date": datetime.date(2026, 4, 1), "ticker": "NVDA",
             "close_price": 100.0, "fwd_1d": 0.01, "fwd_5d": 0.03,
             "fwd_10d": 0.05, "fwd_20d": 0.09},
            {"as_of_date": datetime.date(2026, 4, 1), "ticker": "AAPL",
             "close_price": 200.0, "fwd_1d": -0.01, "fwd_5d": -0.02,
             "fwd_10d": -0.01, "fwd_20d": 0.00},
        ])
        assert n == 2
        rows = conn.execute(
            "SELECT ticker, close_price, fwd_10d FROM ticker_forward_returns ORDER BY ticker",
        ).fetchall()
        assert rows == [("AAPL", 200.0, -0.01), ("NVDA", 100.0, 0.05)]

    def test_upsert_merges_null_with_existing(self, tmp_path):
        """Second call filling fwd_20d must not wipe earlier fwd_10d."""
        from kernel.persistence import record_forward_returns
        conn = _make_conn(tmp_path)
        record_forward_returns(conn, [
            {"as_of_date": datetime.date(2026, 4, 1), "ticker": "NVDA",
             "close_price": 100.0, "fwd_1d": 0.01, "fwd_5d": 0.03,
             "fwd_10d": 0.05, "fwd_20d": None},
        ])
        record_forward_returns(conn, [
            {"as_of_date": datetime.date(2026, 4, 1), "ticker": "NVDA",
             "close_price": None, "fwd_1d": None, "fwd_5d": None,
             "fwd_10d": None, "fwd_20d": 0.09},
        ])
        row = conn.execute(
            "SELECT close_price, fwd_1d, fwd_10d, fwd_20d FROM ticker_forward_returns",
        ).fetchone()
        assert row == (100.0, 0.01, 0.05, 0.09)

    def test_upsert_overwrites_non_null(self, tmp_path):
        """Second call with a non-null value must overwrite (fresh data wins)."""
        from kernel.persistence import record_forward_returns
        conn = _make_conn(tmp_path)
        record_forward_returns(conn, [
            {"as_of_date": datetime.date(2026, 4, 1), "ticker": "NVDA",
             "close_price": 100.0, "fwd_1d": 0.01, "fwd_5d": 0.03,
             "fwd_10d": 0.05, "fwd_20d": 0.09},
        ])
        record_forward_returns(conn, [
            {"as_of_date": datetime.date(2026, 4, 1), "ticker": "NVDA",
             "close_price": 101.0, "fwd_1d": 0.02, "fwd_5d": 0.03,
             "fwd_10d": 0.06, "fwd_20d": 0.10},
        ])
        row = conn.execute(
            "SELECT close_price, fwd_1d, fwd_10d FROM ticker_forward_returns",
        ).fetchone()
        assert row == (101.0, 0.02, 0.06)

    def test_none_conn_noop(self):
        from kernel.persistence import record_forward_returns
        assert record_forward_returns(None, [{"as_of_date": "2026-04-01", "ticker": "X"}]) == 0

    def test_empty_rows_noop(self, tmp_path):
        from kernel.persistence import record_forward_returns
        conn = _make_conn(tmp_path)
        assert record_forward_returns(conn, []) == 0


class TestBackfillScriptEndToEnd:
    """Run `scripts/backfill_forward_returns.py` against a toy DB + parquet cache."""

    def _seed_parquet(self, cache_root: Path, ticker: str, closes: list[float],
                      start: datetime.date) -> None:
        import pandas as pd  # noqa: PLC0415
        dates = pd.bdate_range(start=start, periods=len(closes))
        df = pd.DataFrame({"close": closes, "open": closes, "high": closes,
                           "low": closes, "volume": [1_000_000] * len(closes)},
                          index=dates)
        df.index.name = "Date"
        out = cache_root / ticker / "1d.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)

    def _seed_db(self, db_path: Path, date_str: str, ticker: str) -> None:
        """Insert a minimal pipeline_run + candidate_score row."""
        from kernel.persistence import (
            get_connection, record_pipeline_run, record_candidate_scores,
        )
        from types import SimpleNamespace
        conn = get_connection({"persistence": {"enabled": True, "db_path": str(db_path)}})
        rid = record_pipeline_run(
            conn, run_type="sim",
            run_date=datetime.date.fromisoformat(date_str),
            strategy="renquant_104",
        )
        cand = SimpleNamespace(ticker=ticker, raw_score=5.0, rank_score=0.5,
                               rs_score=0.0, panel_score=0.5, mu=None, sigma=None)
        record_candidate_scores(conn, rid, [cand], {}, selected_tickers={ticker})
        conn.commit()
        conn.close()

    def test_backfill_computes_forward_returns(self, tmp_path):
        """End-to-end: seed DB + parquet → run script → rows appear."""
        db_path = tmp_path / "runs.db"
        cache_root = tmp_path / "ohlcv"
        # Seed parquet: 25 trading days, starting 2026-04-01, closes 100 → 125
        closes = [100.0 + i for i in range(25)]
        self._seed_parquet(cache_root, "NVDA", closes, datetime.date(2026, 4, 1))
        # Decision day: day 0 (close=100.0)
        self._seed_db(db_path, "2026-04-01", "NVDA")

        script = REPO_ROOT / "scripts" / "backfill_forward_returns.py"
        result = subprocess.run(
            [sys.executable, str(script),
             "--db", str(db_path.relative_to(REPO_ROOT)) if db_path.is_relative_to(REPO_ROOT)
                     else str(db_path),
             "--cache-root", str(cache_root)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT close_price, fwd_1d, fwd_5d, fwd_10d, fwd_20d FROM ticker_forward_returns",
        ).fetchone()
        assert row is not None
        close, f1, f5, f10, f20 = row
        assert close == 100.0
        # closes[1]/closes[0] - 1 = 101/100 - 1 = 0.01
        assert f1  == pytest.approx(0.01)
        assert f5  == pytest.approx(0.05)
        assert f10 == pytest.approx(0.10)
        assert f20 == pytest.approx(0.20)
