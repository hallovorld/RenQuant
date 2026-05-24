from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

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
