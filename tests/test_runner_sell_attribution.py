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

from adapters.runner import (  # noqa: E402
    build_sell_trade_event_for_db,
    model_type_from_artifact,
    sell_event_price,
)
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
        config={
            "tax": {"cash_debit_mode": "reporting_only"},
            "rotation": {"joint_actions": {"qp_tax_lot_method": "hifo"}},
        },
    )

    assert row["source_job"] == "JointPortfolioQPJob"
    assert row["source_task"] == "EmitOrdersFromQPSolutionTask"
    assert row["order_source"] == "JointPortfolioQPJob.EmitOrdersFromQPSolutionTask"
    assert row["source"] == "ExitPipeline"
    assert row["order_type"] == "SELL_qp_sell"
    assert row["shares"] == 3.0
    assert row["gross_pnl"] == 30.0
    assert row["tax"] == 15.0
    assert row["net_pnl_after_tax"] == 15.0
    assert row["tax_cash_debited"] == 0.0
    assert row["tax_cash_debit_mode"] == "reporting_only"
    assert row["rank_score"] == 0.71
    assert row["decision_inputs"]["quantity"] == 3.0
    assert row["decision_inputs"]["shares"] == 3.0
    assert row["decision_inputs"]["gross_pnl"] == 30.0
    assert row["decision_inputs"]["tax_cash_debited"] == 0.0
    assert row["decision_inputs"]["tax_cash_debit_mode"] == "reporting_only"
    assert row["decision_inputs"]["tax_lot_method"] == "hifo"
    assert row["decision_inputs"]["take_profit_pct"] == 0.30
    assert row["decision_inputs"]["sdl_skip_if_unrealized_above"] == 0.02


def test_live_sql_sell_uses_broker_fill_price_when_available():
    sig = ExitSignal(
        should_exit=True,
        reason="broker filled below snapshot",
        exit_type="qp_sell",
    )
    sig.sell_price = 95.0

    assert sell_event_price(sig, 100.0) == 95.0


def test_live_sell_trade_persists_applied_exit_params_from_signal():
    today = datetime.date(2026, 5, 22)
    holding = HoldingState(
        entry_price=100.0,
        entry_date=today - datetime.timedelta(days=60),
        high_watermark=125.0,
        shares=10.0,
        entry_regime="BULL_CALM",
    )
    sig = ExitSignal(
        should_exit=True,
        reason="stop",
        exit_type="stop_loss",
        exit_params={
            "stop_loss_pct": 0.15,
            "stop_loss_anchor_policy": "max_entry_current",
            "stop_loss_anchor_regime": "BULL_CALM",
            "stop_loss_current_pct": 0.08,
            "stop_loss_entry_pct": 0.15,
        },
    )

    row = build_sell_trade_event_for_db(
        ticker="AAPL",
        sig=sig,
        holding=holding,
        price=84.0,
        today=today,
        regime="CHOPPY",
        confidence=0.8,
        regime_params={"stop_loss_pct": 0.08},
        config={
            "risk": {"stop_loss_anchor_policy": {"mode": "current_regime"}},
            "regime_params": {
                "BULL_CALM": {"stop_loss_pct": 0.15},
                "CHOPPY": {"stop_loss_pct": 0.08},
            },
        },
    )

    assert row["decision_inputs"]["stop_loss_pct"] == 0.15
    assert row["decision_inputs"]["stop_loss_anchor_policy"] == "max_entry_current"
    assert row["decision_inputs"]["stop_loss_anchor_regime"] == "BULL_CALM"
    assert row["decision_inputs"]["stop_loss_current_pct"] == 0.08
    assert row["decision_inputs"]["stop_loss_entry_pct"] == 0.15


def test_model_type_from_dict_artifact_metadata():
    assert model_type_from_artifact({"_metadata": {"best_approach": "XGBoost"}}) == "XGBoost"
    assert model_type_from_artifact({"_metadata": {"model_type": "Manual"}}) == "Manual"
    assert model_type_from_artifact({"kind": "hf_patchtst"}) == "hf_patchtst"
