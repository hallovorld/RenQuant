from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import analyze_wf_trade_forensics as wf_forensics  # noqa: E402
from analyze_wf_trade_forensics import analyze_trace  # noqa: E402


def test_analyze_trace_uses_configured_hifo_lot_method(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    trades = [
        {
            "action": "buy",
            "ticker": "AAA",
            "date": str(pd.Timestamp("2024-01-02")),
            "price": 100.0,
            "shares": 1,
            "invest": 100.0,
            "rank_score": 0.60,
            "mu": 0.02,
            "panel_score": 0.1,
            "source_job": "JointPortfolioQPJob",
            "regime": "BULL_CALM",
        },
        {
            "action": "buy",
            "ticker": "AAA",
            "date": str(pd.Timestamp("2024-01-03")),
            "price": 135.0,
            "shares": 1,
            "invest": 135.0,
            "rank_score": 0.61,
            "mu": 0.03,
            "panel_score": 0.2,
            "source_job": "JointPortfolioQPJob",
            "regime": "BULL_CALM",
        },
        {
            "action": "sell",
            "ticker": "AAA",
            "date": str(pd.Timestamp("2024-02-01")),
            "price": 145.0,
            "shares": 1,
            "tax": 5.0,
            "tax_cash_debited": 0.0,
            "tax_cash_debit_mode": "reporting_only",
            "exit_reason": "qp_sell",
            "source_job": "JointPortfolioQPJob",
            "regime": "BULL_CALM",
        },
    ]
    (trace_dir / "cut1.trades.json").write_text(json.dumps(trades))
    (trace_dir / "cut1.equity.json").write_text(
        json.dumps({
            "event_level_apy": 0.1,
            "event_level_sharpe": 1.0,
            "annual_net_apy": 0.08,
            "annual_net_sharpe": 0.8,
            "tax_cash_debit_mode": "reporting_only",
        })
    )
    config = {
        "rotation": {
            "joint_actions": {
                "qp_tax_lot_method": "hifo",
            }
        }
    }

    payload = analyze_trace(trace_dir, config=config)

    assert payload["tax_lot_method"] == "hifo"
    assert payload["overall"]["gross_pnl"] == 10.0
    assert payload["overall"]["tax"] == 5.0
    assert payload["tax_integrity"]["positive_rows_with_tax_gt_gross"] == 0
    assert payload["tax_integrity"]["losing_rows_with_positive_tax"] == 0
    assert payload["n_rows"]["open"] == 1


def test_alpha_vs_benchmark_measures_same_capital_active_pnl(monkeypatch) -> None:
    prices = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    monkeypatch.setattr(wf_forensics, "_load_close_series", lambda ticker: prices)
    closed = pd.DataFrame([
        {
            "status": "closed",
            "ticker": "AAA",
            "entry_date": "2024-01-02",
            "exit_date": "2024-01-03",
            "shares": 10,
            "entry_price": 20.0,
            "gross_pnl": 30.0,
            "tax": 5.0,
            "net_pnl_after_tax": 25.0,
            "hold_days": 1,
            "entry_source_job": "JointPortfolioQPJob",
            "entry_regime": "BULL_CALM",
            "exit_regime": "CHOPPY",
            "exit_reason": "qp_close",
            "cut": "cut1",
        },
        {
            "status": "closed",
            "ticker": "SPY",
            "entry_date": "2024-01-02",
            "exit_date": "2024-01-03",
            "shares": 1,
            "entry_price": 100.0,
            "gross_pnl": 10.0,
            "tax": 1.0,
            "net_pnl_after_tax": 9.0,
            "hold_days": 1,
            "entry_source_job": "BenchmarkSleeveJob",
            "exit_reason": "benchmark_sleeve_rebalance",
            "cut": "cut1",
        },
    ])

    payload = wf_forensics._alpha_vs_benchmark(
        closed,
        benchmark_ticker="SPY",
        min_group_n=1,
    )

    overall = payload["overall"]
    assert overall["n"] == 1
    assert overall["net_pnl_after_tax"] == 25.0
    assert overall["benchmark_pnl_same_capital"] == pytest.approx(20.0)
    assert overall["active_net_after_tax"] == pytest.approx(5.0)
    assert overall["active_win_rate"] == 1.0
    transition = payload["by_entry_exit_regime"][0]
    assert transition["entry_exit_regime"] == "BULL_CALM->CHOPPY"
    assert transition["active_net_after_tax"] == pytest.approx(5.0)


def test_score_spearman_by_group_keeps_regime_contract() -> None:
    rows = []
    for i in range(12):
        rows.append({
            "entry_regime": "BULL_CALM",
            "entry_rank_score": float(i),
            "entry_mu": float(i),
            "entry_panel_score": float(i),
            "pnl_pct": float(i),
            "gross_pnl": float(i),
            "net_pnl_after_tax": float(i),
        })
    for i in range(12):
        rows.append({
            "entry_regime": "CHOPPY",
            "entry_rank_score": float(i),
            "entry_mu": float(i),
            "entry_panel_score": float(i),
            "pnl_pct": float(11 - i),
            "gross_pnl": float(11 - i),
            "net_pnl_after_tax": float(11 - i),
        })
    df = pd.DataFrame(rows)

    payload = wf_forensics._score_spearman_by_group(
        df,
        "entry_regime",
        min_n=10,
    )

    rank_rows = {
        row["entry_regime"]: row
        for row in payload
        if row["score_col"] == "entry_rank_score"
    }
    assert rank_rows["BULL_CALM"]["vs_pnl_pct"] == pytest.approx(1.0)
    assert rank_rows["CHOPPY"]["vs_pnl_pct"] == pytest.approx(-1.0)


def test_forward_return_alignment_separates_model_signal_from_trade_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ohlcv"
    dates = pd.bdate_range("2024-01-02", periods=80)
    spy = pd.DataFrame({"close": np.linspace(100.0, 110.0, len(dates))}, index=dates)
    (root / "SPY").mkdir(parents=True)
    spy.to_parquet(root / "SPY" / "1d.parquet")
    rows = []
    for i in range(12):
        ticker = f"T{i:02d}"
        # Higher entry score gets higher forward return.
        close = np.linspace(50.0, 50.0 + i * 2.0, len(dates))
        (root / ticker).mkdir(parents=True)
        pd.DataFrame({"close": close}, index=dates).to_parquet(
            root / ticker / "1d.parquet"
        )
        rows.append({
            "status": "closed",
            "cut": "cut1",
            "entry_event_id": i,
            "ticker": ticker,
            "entry_date": dates[0],
            "exit_reason": "qp_close",
            "shares": 1,
            "gross_pnl": 1.0,
            "tax": 0.0,
            "net_pnl_after_tax": 1.0,
            "pnl_pct": 0.01,
            "hold_days": 20,
            "entry_regime": "BULL_CALM",
            "entry_rank_score": float(i),
            "entry_panel_score": float(i),
            "entry_mu": float(i),
        })
    closed = pd.DataFrame(rows)

    payload = wf_forensics._forward_return_alignment(
        closed,
        ohlcv_root=root,
        benchmark_ticker="SPY",
        horizons=(20,),
        min_n=10,
    )

    assert payload["enabled"] is True
    assert payload["n_entry_events"] == 12
    row = next(
        r for r in payload["by_entry_regime"]
        if r["score_col"] == "entry_rank_score"
    )
    assert row["entry_regime"] == "BULL_CALM"
    assert row["spearman_vs_forward_excess"] == pytest.approx(1.0)


def test_cut_exposure_summary_separates_alpha_and_benchmark(monkeypatch) -> None:
    prices = {
        "SPY": pd.Series(
            [100.0, 100.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
        "AAA": pd.Series(
            [20.0, 30.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    }
    monkeypatch.setattr(
        wf_forensics,
        "_load_close_series",
        lambda ticker: prices[ticker.upper()],
    )
    equity = {"2024-01-02": 1000.0, "2024-01-03": 1000.0}
    trades = [
        {"action": "buy", "ticker": "SPY", "date": "2024-01-02", "shares": 5},
        {"action": "buy", "ticker": "AAA", "date": "2024-01-02", "shares": 10},
    ]

    row = wf_forensics._cut_exposure_summary(
        cut="cut1",
        equity=equity,
        trades=trades,
        benchmark_ticker="SPY",
    )

    assert row["avg_benchmark_weight"] == 0.5
    assert row["avg_alpha_weight"] == 0.25
    assert row["avg_gross_weight"] == 0.75
    assert row["avg_cash_weight"] == 0.25
    assert row["avg_alpha_positions"] == 1.0
    assert row["max_alpha_weight"] == 0.3
