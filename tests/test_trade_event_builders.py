"""Shared trade-event builder parity guards."""
from __future__ import annotations

import sys
from pathlib import Path


STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.trade_events import build_buy_trade_event  # noqa: E402


def test_build_buy_trade_event_computes_invest_and_audit_payload():
    row = build_buy_trade_event(
        {
            "ticker": "AAPL",
            "shares": 3,
            "price": 100.0,
            "target_pct": 0.12,
            "rank_score": 0.61,
            "panel_score": 0.04,
            "order_type": "QP_BUY",
            "source_job": "JointPortfolioQPJob",
            "source_task": "EmitOrdersFromQPSolutionTask",
        },
        date="2026-05-24",
        default_regime="BULL_CALM",
        default_confidence=0.8,
        attribution_version="unit_buy_v1",
    )

    assert row["action"] == "buy"
    assert row["invest"] == 300.0
    assert row["regime"] == "BULL_CALM"
    assert row["confidence"] == 0.8
    assert row["attribution_version"] == "unit_buy_v1"
    assert row["score_snapshot"]["rank_score"] == 0.61
    assert row["score_snapshot"]["panel_score"] == 0.04
    assert row["decision_inputs"]["acceptance_reason"] == "QP_BUY"
    assert row["decision_inputs"]["source_job"] == "JointPortfolioQPJob"


def test_adapters_use_shared_buy_trade_event_builder():
    for rel in ("adapters/sim.py", "adapters/runner.py", "adapters/lean.py"):
        src = (STRATEGY_DIR / rel).read_text()
        assert "build_buy_trade_event(" in src, (
            f"{rel} must build BUY trade rows through the shared helper"
        )
