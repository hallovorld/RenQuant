"""Behavior tests for record_ticker_daily_state writer.

Per user spec round-5 (2026-04-26): every watchlist ticker gets a row
per bar — including those filtered out at universe floor / broker
pre-check / no-model-signal gates. This test pins the writer's API +
upsert semantics + NULL handling. The runner.commit() wiring is
validated end-to-end via e2e.
"""
from __future__ import annotations

import datetime
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.persistence import _SCHEMA_SQL, record_ticker_daily_state  # noqa: E402


def _db():
    db = sqlite3.connect(":memory:")
    db.executescript(_SCHEMA_SQL)
    return db


class TestNoOpGuards:
    def test_none_conn_returns_zero(self):
        assert record_ticker_daily_state(None, run_date=datetime.date(2026, 4, 26), rows=[]) == 0

    def test_empty_rows_returns_zero(self):
        db = _db()
        assert record_ticker_daily_state(db, run_date=datetime.date(2026, 4, 26), rows=[]) == 0

    def test_row_missing_ticker_skipped(self):
        db = _db()
        n = record_ticker_daily_state(db, run_date=datetime.date(2026, 4, 26),
                                      rows=[{"ticker": None, "regime": "BULL_CALM"},
                                            {"ticker": "AAPL", "regime": "BULL_CALM"}])
        assert n == 1


class TestWriteAndRead:
    def test_full_row_round_trip(self):
        db = _db()
        rd = datetime.date(2026, 4, 26)
        row = {
            "ticker": "AAPL", "regime": "BULL_CALM", "confidence": 0.54,
            "in_watchlist": 1, "in_universe": 1, "pending_at_broker": 0,
            "has_position": 1, "position_qty": 5.0, "position_pct": 0.07,
            "model_type": "XGBoost", "model_action": "hold",
            "sell_streak": 0, "panel_score": 0.42, "rank_score": 0.31,
            "expected_return": 0.012, "expected_return_horizon_days": 60,
            "kelly_target_pct": 0.08,
            "mu": 0.01, "mu_horizon_days": 60, "sigma": 0.05, "in_candidates": 0,
            "selected": 0, "blocked_by": None, "sector": "Tech",
            "qp_delta_w": -0.015, "qp_target_w": 0.055, "qp_status": "optimal",
        }
        record_ticker_daily_state(db, run_date=rd, rows=[row])
        cur = db.execute(
            """SELECT date, ticker, regime, confidence, in_watchlist, in_universe,
                      has_position, position_qty, model_type, model_action,
                      sell_streak, rank_score, kelly_target_pct, blocked_by, sector,
                      qp_delta_w, qp_target_w, qp_status,
                      expected_return_horizon_days, mu_horizon_days
               FROM ticker_daily_state WHERE ticker = ?""",
            ("AAPL",),
        )
        out = cur.fetchone()
        assert out[0] == "2026-04-26"
        assert out[1] == "AAPL"
        assert out[2] == "BULL_CALM"
        assert out[3] == pytest.approx(0.54)
        assert out[4] == 1 and out[5] == 1
        assert out[6] == 1
        assert out[7] == pytest.approx(5.0)
        assert out[8] == "XGBoost"
        assert out[9] == "hold"
        assert out[10] == 0
        assert out[11] == pytest.approx(0.31)
        assert out[12] == pytest.approx(0.08)
        assert out[13] is None
        assert out[14] == "Tech"
        assert out[15] == pytest.approx(-0.015)
        assert out[16] == pytest.approx(0.055)
        assert out[17] == "optimal"
        assert out[18] == 60
        assert out[19] == 60

    def test_universe_filtered_ticker_minimal_row(self):
        db = _db()
        rd = datetime.date(2026, 4, 26)
        record_ticker_daily_state(db, run_date=rd, rows=[
            {"ticker": "ZS", "in_watchlist": 1, "in_universe": 0,
             "blocked_by": "universe_floor"},
        ])
        out = db.execute(
            "SELECT in_universe, blocked_by, model_type, panel_score "
            "FROM ticker_daily_state WHERE ticker = ?", ("ZS",),
        ).fetchone()
        assert out == (0, "universe_floor", None, None)


class TestUpsert:
    def test_replace_on_same_date_ticker(self):
        db = _db()
        rd = datetime.date(2026, 4, 26)
        record_ticker_daily_state(db, run_date=rd, rows=[
            {"ticker": "AAPL", "model_action": "buy", "rank_score": 0.10},
        ])
        record_ticker_daily_state(db, run_date=rd, rows=[
            {"ticker": "AAPL", "model_action": "hold", "rank_score": 0.55},
        ])
        out = db.execute(
            "SELECT COUNT(*), MAX(rank_score), MAX(model_action) "
            "FROM ticker_daily_state WHERE ticker = ?", ("AAPL",),
        ).fetchone()
        assert out[0] == 1
        assert out[1] == pytest.approx(0.55)
        assert out[2] == "hold"

    def test_same_date_ticker_preserved_across_runs(self):
        db = _db()
        rd = datetime.date(2026, 4, 26)
        record_ticker_daily_state(
            db, run_date=rd, run_id="2026-04-26-live-a",
            rows=[{"ticker": "AAPL", "model_action": "buy", "rank_score": 0.10}],
        )
        record_ticker_daily_state(
            db, run_date=rd, run_id="2026-04-26-live-b",
            rows=[{"ticker": "AAPL", "model_action": "hold", "rank_score": 0.55}],
        )
        out = db.execute(
            "SELECT COUNT(*), MIN(rank_score), MAX(rank_score) "
            "FROM ticker_daily_state WHERE date=? AND ticker=?",
            ("2026-04-26", "AAPL"),
        ).fetchone()
        assert out[0] == 2
        assert out[1] == pytest.approx(0.10)
        assert out[2] == pytest.approx(0.55)

    def test_two_dates_two_rows(self):
        db = _db()
        record_ticker_daily_state(db, run_date=datetime.date(2026, 4, 25),
                                  rows=[{"ticker": "AAPL", "rank_score": 0.1}])
        record_ticker_daily_state(db, run_date=datetime.date(2026, 4, 26),
                                  rows=[{"ticker": "AAPL", "rank_score": 0.2}])
        n = db.execute(
            "SELECT COUNT(*) FROM ticker_daily_state WHERE ticker = ?", ("AAPL",),
        ).fetchone()[0]
        assert n == 2


class TestNumericSafety:
    def test_nan_inf_become_null(self):
        db = _db()
        rd = datetime.date(2026, 4, 26)
        record_ticker_daily_state(db, run_date=rd, rows=[
            {"ticker": "AAPL", "rank_score": float("nan"),
             "panel_score": float("inf"), "mu": float("-inf"),
             "sigma": 0.05},
        ])
        out = db.execute(
            "SELECT rank_score, panel_score, mu, sigma "
            "FROM ticker_daily_state WHERE ticker = ?", ("AAPL",),
        ).fetchone()
        assert out[0] is None
        assert out[1] is None
        assert out[2] is None
        assert out[3] == pytest.approx(0.05)

    def test_string_run_date_accepted(self):
        db = _db()
        n = record_ticker_daily_state(db, run_date="2026-04-26",
                                      rows=[{"ticker": "AAPL"}])
        assert n == 1
        out = db.execute("SELECT date FROM ticker_daily_state WHERE ticker=?", ("AAPL",)).fetchone()
        assert out[0] == "2026-04-26"


class TestBatch:
    def test_batch_with_universe_and_pending_and_held(self):
        """Realistic batch: 100 watchlist tickers, ~50 in_universe, 4 pending, ~7 held."""
        db = _db()
        rd = datetime.date(2026, 4, 26)
        rows = []
        for i in range(100):
            t = f"T{i:03d}"
            in_u = 1 if i < 50 else 0
            pending = 1 if t in {"T010", "T020", "T030", "T040"} else 0
            held    = 1 if i in {0, 1, 2, 3, 4, 5, 6} else 0
            rows.append({
                "ticker": t, "regime": "BULL_CALM", "confidence": 0.5,
                "in_watchlist": 1, "in_universe": in_u,
                "pending_at_broker": pending, "has_position": held,
                "model_type": "XGBoost" if in_u else None,
                "model_action": "hold" if in_u else None,
                "blocked_by": ("universe_floor" if not in_u else
                               "broker_pending" if pending else None),
            })
        n = record_ticker_daily_state(db, run_date=rd, rows=rows)
        assert n == 100
        # Sanity
        held_count = db.execute(
            "SELECT COUNT(*) FROM ticker_daily_state WHERE has_position = 1"
        ).fetchone()[0]
        assert held_count == 7
        floor_count = db.execute(
            "SELECT COUNT(*) FROM ticker_daily_state WHERE blocked_by = 'universe_floor'"
        ).fetchone()[0]
        assert floor_count == 50
