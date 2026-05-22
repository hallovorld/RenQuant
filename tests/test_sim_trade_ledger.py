from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from sim_trade_ledger import (  # noqa: E402
    annual_net_tax_summary,
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
            "regime": "BULL_CALM",
            "confidence": 0.8,
            "exit_stop_loss_pct": 0.15,
            "exit_max_single_day_loss_pct": 0.0,
            "exit_sdl_n_sigma": 3.0,
            "exit_trailing_stop_trigger_pct": 0.12,
            "exit_trailing_stop_trail_pct": 0.25,
            "exit_max_hold_days": 500,
        },
    ]

    trips = round_trips_from_trade_log(trade_log, end_prices={"AAPL": 130.0})

    closed = trips[trips["status"] == "closed"]
    open_lots = trips[trips["status"] == "open"]
    assert len(closed) == 2
    assert len(open_lots) == 1
    assert list(closed["shares"]) == [10, 2]
    assert closed["tax"].round(6).tolist() == [21.818182, 2.181818]
    assert closed.iloc[0]["entry_regime"] == "BULL_CALM"
    assert closed.iloc[1]["entry_regime"] == "BULL_VOLATILE"
    assert closed.iloc[0]["exit_regime"] == "BULL_CALM"
    assert closed.iloc[0]["exit_stop_loss_pct"] == 0.15
    assert open_lots.iloc[0]["shares"] == 3
    assert open_lots.iloc[0]["gross_pnl"] == 60.0


def test_round_trip_tax_allocation_does_not_tax_losing_lots() -> None:
    trade_log = [
        {
            "action": "buy",
            "ticker": "AAPL",
            "date": pd.Timestamp("2024-01-02"),
            "price": 100.0,
            "shares": 1,
            "invest": 100.0,
        },
        {
            "action": "buy",
            "ticker": "AAPL",
            "date": pd.Timestamp("2024-01-03"),
            "price": 130.0,
            "shares": 1,
            "invest": 130.0,
        },
        {
            "action": "sell",
            "ticker": "AAPL",
            "date": pd.Timestamp("2024-02-01"),
            "price": 120.0,
            "shares": 2,
            "tax": 5.0,
            "pnl_pct": 10.0 / 230.0,
            "exit_reason": "qp_sell",
        },
    ]

    trips = round_trips_from_trade_log(trade_log)
    closed = trips[trips["status"] == "closed"].reset_index(drop=True)

    assert closed.loc[0, "gross_pnl"] == 20.0
    assert closed.loc[0, "tax"] == 5.0
    assert closed.loc[1, "gross_pnl"] == -10.0
    assert closed.loc[1, "tax"] == 0.0
    assert not ((closed["gross_pnl"] <= 0) & (closed["tax"] > 0)).any()
    assert not ((closed["gross_pnl"] > 0) & (closed["tax"] > closed["gross_pnl"])).any()


def test_forensic_report_groups_by_regime_and_exit_reason() -> None:
    trips = pd.DataFrame([
        {
            "status": "closed",
            "ticker": "AAPL",
            "entry_date": "2024-01-02",
            "exit_date": "2024-02-01",
            "entry_regime": "BULL_CALM",
            "exit_reason": "qp_sell",
            "exit_regime": "BULL_CALM",
            "exit_stop_loss_pct": 0.15,
            "exit_max_single_day_loss_pct": 0.0,
            "exit_sdl_n_sigma": 3.0,
            "exit_trailing_stop_trigger_pct": 0.12,
            "exit_trailing_stop_trail_pct": 0.25,
            "exit_max_hold_days": 500,
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
    assert "By exit_regime" in report
    assert "By exit_reason" in report
    assert "Annual Net Tax" not in report  # heading is Tax Stress
    assert "Tax Stress" in report
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
            "regime": "BULL_CALM",
            "exit_stop_loss_pct": 0.15,
            "exit_max_single_day_loss_pct": 0.0,
            "exit_sdl_n_sigma": 3.0,
            "exit_trailing_stop_trigger_pct": 0.12,
            "exit_trailing_stop_trail_pct": 0.25,
            "exit_max_hold_days": 500,
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


def test_write_trade_outputs_enriches_missing_buy_regime_and_marks_open_lots(tmp_path) -> None:
    trade_log = [
        {
            "action": "buy",
            "ticker": "AAPL",
            "date": pd.Timestamp("2024-01-02"),
            "price": 100.0,
            "shares": 2,
            "invest": 200.0,
            "rank_score": 0.70,
            # QP orders historically omitted regime; result.equity_df must
            # backfill it so per-regime forensic reports are not empty.
            "regime": None,
        },
    ]
    result = SimpleNamespace(
        trade_log=trade_log,
        buys=trade_log,
        sells=[],
        final_value=220.0,
        total_return=0.10,
        apy=0.10,
        sharpe=1.0,
        max_dd=0.0,
        win_rate=0.0,
        equity_df=pd.DataFrame(
            [{"portfolio": 200.0, "regime": "BULL_CALM"}],
            index=[pd.Timestamp("2024-01-02")],
        ),
    )

    write_trade_outputs(
        result=result,
        trade_csv=tmp_path / "trades.csv",
        round_trips_csv=tmp_path / "round.csv",
        end_prices={"AAPL": 110.0},
    )

    raw = pd.read_csv(tmp_path / "trades.csv")
    trips = pd.read_csv(tmp_path / "round.csv")
    assert raw.loc[0, "regime"] == "BULL_CALM"
    assert trips.loc[0, "status"] == "open"
    assert trips.loc[0, "entry_regime"] == "BULL_CALM"
    assert trips.loc[0, "gross_pnl"] == 20.0


def test_annual_net_tax_summary_nets_same_year_wins_and_losses() -> None:
    trips = pd.DataFrame([
        {
            "status": "closed",
            "exit_date": "2024-02-01",
            "hold_days": 30,
            "gross_pnl": 100.0,
        },
        {
            "status": "closed",
            "exit_date": "2024-03-01",
            "hold_days": 20,
            "gross_pnl": -80.0,
        },
    ])

    summary = annual_net_tax_summary(
        trips,
        {"short_term_rate": 0.50, "long_term_rate": 0.32, "long_term_threshold_days": 365},
    )

    assert summary["total_estimated_tax"] == 10.0
    assert summary["years"][0]["short_term_net"] == 20.0
