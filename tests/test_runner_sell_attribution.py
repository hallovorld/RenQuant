"""AUDIT REGRESSION GUARD: live sell trades preserve source attribution.

Portfolio-level exits such as QP sells are emitted outside ``TickerSellJob``.
The live adapter must write the real source metadata into SQLite so the
decision-tree audit explains which pipeline component caused the order.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest


STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from adapters.runner import (  # noqa: E402
    apply_live_sell_lot_accounting,
    build_sell_trade_event_for_db,
    live_execution_attempt_events,
    live_trace_selection_maps,
    live_post_execution_snapshot,
    model_type_from_artifact,
    reconstruct_live_tax_lots_from_fills,
    sell_event_realized_kwargs,
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


def test_live_sell_trade_uses_explicit_broker_realized_economics():
    today = datetime.date(2026, 5, 22)
    holding = HoldingState(
        entry_price=90.0,  # fallback would produce the wrong P/L
        entry_date=today - datetime.timedelta(days=20),
        high_watermark=150.0,
        shares=10.0,
    )
    sig = ExitSignal(
        should_exit=True,
        reason="partial QP trim",
        exit_type="qp_sell",
        quantity=4.0,
    )
    sig.sell_price = 120.0
    sig.shares_sold = 4.0
    sig.cost_basis = 100.0
    sig.realized_pnl_dollar = 80.0
    sig.realized_pnl_pct = 20.0

    row = build_sell_trade_event_for_db(
        ticker="AAPL",
        sig=sig,
        holding=holding,
        price=120.0,
        today=today,
        regime="BULL_CALM",
        confidence=0.8,
        regime_params={
            "tax": {"short_term_rate": 0.50, "long_term_rate": 0.32},
        },
        config={"tax": {"cash_debit_mode": "reporting_only"}},
        **sell_event_realized_kwargs(sig, holding, today=today),
    )

    assert row["shares"] == 4.0
    assert row["gross_pnl"] == 80.0
    assert row["proceeds_basis"] == 400.0
    assert row["tax"] == 40.0
    assert row["net_pnl_after_tax"] == 40.0
    assert row["pnl_pct"] == 0.20
    assert row["hold_days"] == 20
    assert row["decision_inputs"]["gross_pnl"] == 80.0


def test_live_partial_sell_uses_reconstructed_hifo_tax_lots():
    today = datetime.date(2026, 1, 10)
    fills = [
        {
            "symbol": "AAPL", "action": "BUY", "qty": 10,
            "avg_price": 100.0, "filled_at": "2026-01-02T14:30:00+00:00",
        },
        {
            "symbol": "AAPL", "action": "BUY", "qty": 10,
            "avg_price": 150.0, "filled_at": "2026-01-03T14:30:00+00:00",
        },
    ]
    cfg = {
        "tax": {"short_term_rate": 0.50, "long_term_rate": 0.20},
        "rotation": {"joint_actions": {"qp_tax_lot_method": "hifo"}},
    }
    lots = reconstruct_live_tax_lots_from_fills(fills, config=cfg)
    holding = HoldingState(
        entry_price=125.0,
        entry_date=datetime.date(2026, 1, 2),
        high_watermark=160.0,
        shares=20.0,
    )
    holding.lots = lots["AAPL"]
    sig = ExitSignal(
        should_exit=True,
        reason="partial QP trim",
        exit_type="qp_sell",
        quantity=10.0,
    )

    assert apply_live_sell_lot_accounting(
        sig,
        holding,
        shares=10.0,
        price=160.0,
        today=today,
        config=cfg,
    ) is True

    row = build_sell_trade_event_for_db(
        ticker="AAPL",
        sig=sig,
        holding=holding,
        price=160.0,
        today=today,
        regime="BULL_CALM",
        confidence=0.8,
        regime_params={"tax": cfg["tax"]},
        config={**cfg, "tax": {**cfg["tax"], "cash_debit_mode": "reporting_only"}},
        **sell_event_realized_kwargs(sig, holding, today=today),
    )

    assert row["shares"] == 10.0
    assert row["proceeds_basis"] == 1500.0
    assert row["gross_pnl"] == 100.0
    assert row["tax"] == 50.0
    assert row["net_pnl_after_tax"] == 50.0
    assert row["pnl_pct"] == pytest.approx(100.0 / 1500.0)
    assert row["hold_days"] == 7
    assert holding.shares == 10.0
    assert holding.entry_price == 100.0


def test_live_post_execution_snapshot_uses_broker_state_and_confirmed_holdings():
    class _Broker:
        def get_account_value(self):
            return 101_000.0

        def get_cash(self):
            return 77_000.0

    ctx = type("Ctx", (), {"portfolio_value": 100_000.0, "cash": 90_000.0})()

    out = live_post_execution_snapshot(
        ctx,
        _Broker(),
        currently_held={"AAA", "BBB"},
    )

    assert out == {
        "portfolio_value": 101_000.0,
        "cash": 77_000.0,
        "n_holdings": 2,
    }


def test_live_trace_marks_pending_buy_as_blocked_not_selected():
    selected, blocked, pending = live_trace_selection_maps(
        trade_events=[],
        pending_orders=[{"ticker": "AAPL", "status": "accepted"}],
        blocked_map={},
    )

    assert selected == set()
    assert pending == {"AAPL"}
    assert blocked["AAPL"] == "broker_pending_submitted"


def test_live_execution_attempt_events_persist_pending_and_skipped_buys():
    ctx = type("Ctx", (), {
        "today": datetime.date(2026, 5, 22),
        "regime": "BULL_CALM",
        "confidence": 0.8,
        "orders_pending": [{
            "ticker": "AAPL", "shares": 3, "price": 100.0,
            "status": "accepted", "order_id": "ord-1",
            "source_job": "SelectionJob",
            "source_task": "SizeAndEmitTask",
        }],
        "orders_skipped": [{
            "ticker": "MSFT", "shares": 2, "price": 200.0,
            "skip_reason": "cash_budget_exhausted",
            "source_job": "SelectionJob",
            "source_task": "SizeAndEmitTask",
        }],
        "exits_pending": [],
        "exits_failed": [],
    })()

    rows = live_execution_attempt_events(ctx)

    assert [r["action"] for r in rows] == ["buy_pending", "buy_skipped"]
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["decision_inputs"]["order_id"] == "ord-1"
    assert rows[0]["decision_inputs"]["status"] == "accepted"
    assert rows[0]["score_snapshot"]["attempt_status"] == "buy_pending"
    assert rows[1]["blocked_by"] == "broker_skip:cash_budget_exhausted"
    assert rows[1]["decision_inputs"]["skip_reason"] == "cash_budget_exhausted"


def test_live_execution_attempt_events_persist_rejected_sell_context():
    holding = HoldingState(
        entry_price=100.0,
        entry_date=datetime.date(2026, 5, 1),
        high_watermark=120.0,
        shares=5.0,
        rank_score=0.44,
        panel_score=0.22,
        expected_return=0.03,
        expected_return_horizon_days=60,
    )
    holding.model_type = "xgb"
    holding.sector = "tech"
    ctx = type("Ctx", (), {
        "today": datetime.date(2026, 5, 22),
        "regime": "BULL_CALM",
        "confidence": 0.8,
        "holdings": {"AAPL": holding},
        "prices": {"AAPL": 110.0},
        "orders_pending": [],
        "orders_skipped": [],
        "exits_pending": [],
        "exits_failed": [{
            "ticker": "AAPL",
            "exit_type": "stop_loss",
            "reason": "stop",
            "qty": 5,
            "status": "rejected",
            "order_id": "ord-2",
            "error": "broker_status:rejected",
        }],
    })()

    rows = live_execution_attempt_events(ctx)

    assert len(rows) == 1
    row = rows[0]
    assert row["action"] == "sell_rejected"
    assert row["shares"] == 5
    assert row["price"] == 110.0
    assert row["blocked_by"] == "broker_status:rejected"
    assert row["score_snapshot"]["rank_score"] == 0.44
    assert row["score_snapshot"]["expected_return_horizon_days"] == 60
    assert row["model_type"] == "xgb"
    assert row["sector"] == "tech"
    assert row["decision_inputs"]["order_id"] == "ord-2"
    assert row["decision_inputs"]["error"] == "broker_status:rejected"


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
