"""AUDIT REGRESSION GUARD: live sell trades preserve source attribution.

Portfolio-level exits such as QP sells are emitted outside ``TickerSellJob``.
The live adapter must write the real source metadata into SQLite so the
decision-tree audit explains which pipeline component caused the order.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path


STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from adapters.runner import build_sell_trade_event_for_db  # noqa: E402
from kernel.exits import ExitSignal, HoldingState  # noqa: E402


def test_live_sell_trade_preserves_exit_signal_source_metadata():
    today = datetime.date(2026, 5, 22)
    holding = HoldingState(
        entry_price=100.0,
        entry_date=today - datetime.timedelta(days=60),
        high_watermark=125.0,
        shares=10.0,
    )
    holding.rank_score = 0.71
    holding.panel_score = 0.68

    sig = ExitSignal(
        should_exit=True,
        reason="QP reduced target weight",
        exit_type="qp_sell",
        quantity=3.0,
    )
    sig.source_job = "JointPortfolioQPJob"
    sig.source_task = "EmitOrdersFromQPSolutionTask"

    row = build_sell_trade_event_for_db(
        ticker="AAPL",
        sig=sig,
        holding=holding,
        price=110.0,
        today=today,
        regime="BULL_CALM",
        confidence=0.8,
        regime_params={
            "stop_loss_pct": 0.15,
            "take_profit_pct": 0.30,
            "stop_decay_days": 60,
            "stop_decay_floor": 0.08,
            "sdl_skip_if_unrealized_above": 0.02,
        },
    )

    assert row["source_job"] == "JointPortfolioQPJob"
    assert row["source_task"] == "EmitOrdersFromQPSolutionTask"
    assert row["order_source"] == "JointPortfolioQPJob.EmitOrdersFromQPSolutionTask"
    assert row["source"] == "ExitPipeline"
    assert row["order_type"] == "SELL_qp_sell"
    assert row["decision_inputs"]["quantity"] == 3.0
    assert row["decision_inputs"]["take_profit_pct"] == 0.30
    assert row["decision_inputs"]["sdl_skip_if_unrealized_above"] == 0.02
