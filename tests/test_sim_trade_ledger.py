from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from sim_trade_ledger import (  # noqa: E402
    build_forensic_report,
    round_trips_from_trade_log,
    write_trade_outputs,
)


def test_round_trips_fifo_matches_partial_sell_and_allocates_tax() -> None:
    trade_log = [
        {
            "action": "buy",
            "ticker": "AAPL",
            "date": pd.Timestamp("2024-01-02"),
            "price": 100.0,
            "shares": 10,
            "invest": 1000.0,
            "regime": "BULL_CALM",
            "rank_score": 0.72,
            "mu": 0.03,
            "sigma": 0.12,
        },
        {
            "action": "buy",
            "ticker": "AAPL",
            "date": pd.Timestamp("2024-01-05"),
            "price": 110.0,
            "shares": 5,
            "invest": 550.0,
            "regime": "BULL_VOLATILE",
            "rank_score": 0.61,
            "mu": 0.02,
            "sigma": 0.18,
        },
        {
            "action": "sell",
            "ticker": "AAPL",
            "date": pd.Timestamp("2024-02-01"),
            "price": 120.0,
            "shares": 12,
            "tax": 24.0,
            "pnl_pct": 0.20,
            "exit_reason": "qp_sell",
            "partial": True,
        },
    ]

    trips = round_trips_from_trade_log(trade_log, end_prices={"AAPL": 130.0})

    closed = trips[trips["status"] == "closed"]
    open_lots = trips[trips["status"] == "open"]
    assert len(closed) == 2
    assert len(open_lots) == 1
    assert list(closed["shares"]) == [10, 2]
    assert closed["tax"].round(6).tolist() == [20.0, 4.0]
    assert closed.iloc[0]["entry_regime"] == "BULL_CALM"
    assert closed.iloc[1]["entry_regime"] == "BULL_VOLATILE"
    assert open_lots.iloc[0]["shares"] == 3
    assert open_lots.iloc[0]["gross_pnl"] == 60.0


def test_forensic_report_groups_by_regime_and_exit_reason() -> None:
    trips = pd.DataFrame([
        {
            "status": "closed",
            "ticker": "AAPL",
            "entry_date": "2024-01-02",
            "exit_date": "2024-02-01",
            "entry_regime": "BULL_CALM",
            "exit_reason": "qp_sell",
            "shares": 10.0,
            "entry_price": 100.0,
            "exit_price": 90.0,
            "gross_pnl": -100.0,
            "tax": 0.0,
            "net_pnl_after_tax": -100.0,
            "pnl_pct": -0.10,
            "hold_days": 30,
            "entry_rank_score": 0.70,
            "entry_mu": 0.02,
            "entry_sigma": 0.10,
        }
    ])
    raw = pd.DataFrame([{"action": "buy", "ticker": "AAPL", "date": "2024-01-02"}])

    report = build_forensic_report(
        raw_trades=raw,
        round_trips=trips,
        metrics={"config": "unit", "n_buys": 1, "n_sells": 1},
        title="Unit Report",
    )

    assert "Theoretical Frame" in report
    assert "By entry_regime" in report
    assert "By exit_reason" in report
    assert "Worst 25 Closed Round Trips" in report


def test_write_trade_outputs_creates_raw_roundtrip_and_report_files(tmp_path) -> None:
    trade_log = [
        {
            "action": "buy",
            "ticker": "AAPL",
            "date": pd.Timestamp("2024-01-02"),
            "price": 100.0,
            "shares": 1,
            "invest": 100.0,
            "regime": "BULL_CALM",
            "rank_score": 0.70,
        },
        {
            "action": "sell",
            "ticker": "AAPL",
            "date": pd.Timestamp("2024-01-03"),
            "price": 90.0,
            "shares": 1,
            "tax": 0.0,
            "pnl_pct": -0.10,
            "exit_reason": "stop_loss",
        },
    ]
    result = SimpleNamespace(
        trade_log=trade_log,
        buys=[trade_log[0]],
        sells=[trade_log[1]],
        final_value=99.0,
        total_return=-0.01,
        apy=-0.10,
        sharpe=-1.0,
        max_dd=0.02,
        win_rate=0.0,
    )

    written = write_trade_outputs(
        result=result,
        trade_json=tmp_path / "trades.json",
        trade_csv=tmp_path / "trades.csv",
        round_trips_csv=tmp_path / "round.csv",
        report_md=tmp_path / "report.md",
    )

    assert set(written) == {
        "trade_json",
        "trade_csv",
        "round_trips_csv",
        "report_md",
    }
    assert (tmp_path / "trades.json").exists()
    assert (tmp_path / "trades.csv").exists()
    assert (tmp_path / "round.csv").exists()
    assert "Sim Trade Forensics" in (tmp_path / "report.md").read_text()

