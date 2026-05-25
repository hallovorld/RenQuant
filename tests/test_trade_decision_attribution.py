from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analyze_trade_decision_attribution import (  # noqa: E402
    build_round_trips,
    load_trade_rows,
    summarize_round_trips,
)


def test_build_round_trips_preserves_entry_decision_payload() -> None:
    trades = pd.DataFrame([
        {
            "run_id": "r1",
            "trade_rowid": 1,
            "date": pd.Timestamp("2026-05-01"),
            "ticker": "AAA",
            "action": "buy",
            "shares": 10,
            "price": 100.0,
            "invest": 1000.0,
            "order_type": "QP_BUY",
            "source": "JointPortfolioQPJob.JointPortfolioQPTask",
            "source_job": "JointPortfolioQPJob",
            "source_task": "JointPortfolioQPTask",
            "order_source": "JointPortfolioQPJob.JointPortfolioQPTask",
            "attribution_version": "order_attribution_v1",
            "score_snapshot_json": json.dumps({
                "rank_score": 0.61,
                "panel_score": 0.58,
                "mu": 0.014,
                "mu_horizon_days": 60,
                "sigma": 0.032,
                "kelly_target_pct": 0.08,
                "expected_return": 0.025,
                "expected_return_horizon_days": 60,
                "model_type": "xgb",
                "sector": "tech",
                "confidence": 0.72,
                "regime": "BULL_CALM",
            }),
            "decision_inputs_json": json.dumps({
                "acceptance_reason": "qp_target_weight_increase",
                "target_w": 0.08,
            }),
            "run_regime": "BULL_CALM",
            "run_confidence": 0.72,
        },
        {
            "run_id": "r2",
            "trade_rowid": 2,
            "date": pd.Timestamp("2026-05-11"),
            "ticker": "AAA",
            "action": "sell",
            "shares": 10,
            "price": 110.0,
            "exit_reason": "model_sell",
            "pnl_pct": 0.10,
            "hold_days": 10,
            "tax": 12.0,
            "order_type": "SELL_model_sell",
            "source": "ExitPipeline",
            "source_job": "TickerSellJob",
            "source_task": "ModelExitTask",
            "order_source": "TickerSellJob.ModelExitTask",
            "attribution_version": "exit_decision_v1",
            "score_snapshot_json": json.dumps({
                "rank_score": 0.52,
                "regime": "BULL_CALM",
            }),
            "decision_inputs_json": json.dumps({
                "acceptance_reason": "model_sell_streak",
            }),
            "run_regime": "BULL_CALM",
        },
    ])

    out = build_round_trips(trades)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["entry_order_type"] == "QP_BUY"
    assert row["entry_source_job"] == "JointPortfolioQPJob"
    assert row["entry_regime"] == "BULL_CALM"
    assert row["entry_rank_score"] == pytest.approx(0.61)
    assert row["entry_mu_horizon_days"] == 60
    assert row["entry_expected_return"] == pytest.approx(0.025)
    assert row["entry_expected_return_horizon_days"] == 60
    assert row["entry_model_type"] == "xgb"
    assert row["entry_sector"] == "tech"
    assert row["entry_acceptance_reason"] == "qp_target_weight_increase"
    assert row["exit_order_type"] == "SELL_model_sell"
    assert row["exit_source_job"] == "TickerSellJob"
    assert row["exit_acceptance_reason"] == "model_sell_streak"
    assert row["gross_pnl"] == pytest.approx(100.0)
    assert row["tax"] == pytest.approx(12.0)
    assert row["net_pnl"] == pytest.approx(88.0)
    assert row["net_return"] == pytest.approx(0.088)


def test_build_round_trips_allocates_tax_only_to_winning_lots() -> None:
    trades = pd.DataFrame([
        {
            "run_id": "r1",
            "trade_rowid": 1,
            "date": pd.Timestamp("2026-05-01"),
            "ticker": "AAA",
            "action": "buy",
            "shares": 1,
            "price": 100.0,
            "invest": 100.0,
            "run_regime": "BULL_CALM",
        },
        {
            "run_id": "r2",
            "trade_rowid": 2,
            "date": pd.Timestamp("2026-05-02"),
            "ticker": "AAA",
            "action": "buy",
            "shares": 1,
            "price": 130.0,
            "invest": 130.0,
            "run_regime": "BULL_CALM",
        },
        {
            "run_id": "r3",
            "trade_rowid": 3,
            "date": pd.Timestamp("2026-05-20"),
            "ticker": "AAA",
            "action": "sell",
            "shares": 2,
            "price": 120.0,
            "exit_reason": "qp_sell",
            "tax": 5.0,
            "run_regime": "BULL_CALM",
        },
    ])

    out = build_round_trips(trades).sort_values("entry_price").reset_index(drop=True)

    assert out.loc[0, "gross_pnl"] == pytest.approx(20.0)
    assert out.loc[0, "tax"] == pytest.approx(5.0)
    assert out.loc[1, "gross_pnl"] == pytest.approx(-10.0)
    assert out.loc[1, "tax"] == pytest.approx(0.0)
    assert not ((out["gross_pnl"] <= 0) & (out["tax"] > 0)).any()
    assert not ((out["gross_pnl"] > 0) & (out["tax"] > out["gross_pnl"])).any()


def test_build_round_trips_handles_legacy_sell_without_share_quantity() -> None:
    trades = pd.DataFrame([
        {
            "run_id": "r1",
            "trade_rowid": 1,
            "date": pd.Timestamp("2026-05-01"),
            "ticker": "AAA",
            "action": "buy",
            "shares": 5,
            "price": 100.0,
            "invest": 500.0,
            "rank_score": 0.4,
            "run_regime": "CHOPPY",
        },
        {
            "run_id": "r2",
            "trade_rowid": 2,
            "date": pd.Timestamp("2026-05-03"),
            "ticker": "AAA",
            "action": "sell",
            "shares": None,
            "price": 95.0,
            "exit_reason": "stop_loss",
            "pnl_pct": -0.05,
            "hold_days": 2,
            "tax": None,
            "run_regime": "CHOPPY",
        },
    ])

    out = build_round_trips(trades)

    assert len(out) == 1
    assert out.iloc[0]["shares"] == pytest.approx(5)
    assert out.iloc[0]["entry_rank_score"] == pytest.approx(0.4)
    assert out.iloc[0]["entry_regime"] == "CHOPPY"
    assert out.iloc[0]["net_pnl"] == pytest.approx(-25.0)


def test_summarize_round_trips_reports_profit_factor_and_win_rate() -> None:
    trips = pd.DataFrame({
        "gross_return": [0.10, -0.05, 0.02],
        "net_return": [0.09, -0.05, 0.02],
        "gross_pnl": [100.0, -50.0, 20.0],
        "tax": [10.0, 0.0, 0.0],
        "net_pnl": [90.0, -50.0, 20.0],
        "hold_days": [10, 3, 20],
    })

    summary = summarize_round_trips(trips)

    assert summary["n_round_trips"] == 3
    assert summary["win_rate"] == pytest.approx(2 / 3)
    assert summary["net_pnl"] == pytest.approx(60.0)
    assert summary["profit_factor"] == pytest.approx(110.0 / 50.0)


def test_load_trade_rows_tolerates_legacy_schema(tmp_path: Path) -> None:
    db = tmp_path / "runs.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE pipeline_runs (
            run_id TEXT PRIMARY KEY,
            run_date DATE,
            run_type TEXT,
            strategy TEXT,
            regime TEXT,
            confidence REAL,
            portfolio_value REAL,
            cash REAL,
            n_candidates INTEGER,
            n_exits INTEGER,
            n_rotations INTEGER,
            n_buys INTEGER
        );
        CREATE TABLE trades (
            run_id TEXT,
            ticker TEXT,
            action TEXT,
            shares REAL,
            price REAL,
            invest REAL,
            pnl_pct REAL,
            hold_days INTEGER,
            rank_score REAL
        );
        """
    )
    conn.execute(
        "INSERT INTO pipeline_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("r1", "2026-05-01", "live", "renquant_104", "BULL_CALM",
         0.7, 10000, 1000, 5, 0, 0, 1),
    )
    conn.execute(
        "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("r1", "AAA", "buy", 1, 100.0, 100.0, None, None, 0.6),
    )
    conn.commit()
    conn.close()

    rows = load_trade_rows(db, run_type="live")

    assert len(rows) == 1
    assert "score_snapshot_json" in rows.columns
    assert rows.iloc[0]["score_snapshot_json"] is None
    assert rows.iloc[0]["run_regime"] == "BULL_CALM"
    assert rows.iloc[0]["date"] == pd.Timestamp("2026-05-01")
