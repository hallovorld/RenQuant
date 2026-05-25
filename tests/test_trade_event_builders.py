"""Shared trade-event builder parity guards."""
from __future__ import annotations

import datetime
import sys
from pathlib import Path


STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.exits import ExitSignal, HoldingState  # noqa: E402
from kernel.decision_trace import selected_buy_tickers  # noqa: E402
from kernel.trade_events import build_buy_trade_event, build_sell_trade_event  # noqa: E402


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


def test_build_buy_trade_event_promotes_snapshot_expected_return():
    """AUDIT REGRESSION GUARD: round-trip ledgers read top-level ER fields."""
    row = build_buy_trade_event(
        {
            "ticker": "AAPL",
            "shares": 3,
            "price": 100.0,
            "rank_score": 0.61,
            "score_snapshot": {
                "rank_score": 0.61,
                "panel_score": 0.04,
                "expected_return": 0.025,
                "expected_return_horizon_days": 60,
                "mu": 0.025,
                "mu_horizon_days": 60,
            },
        },
        date="2026-05-24",
    )

    assert row["expected_return"] == 0.025
    assert row["expected_return_horizon_days"] == 60
    assert row["mu_horizon_days"] == 60


def test_selected_buy_tickers_uses_normalized_trade_events():
    raw_order = {"ticker": "AAPL", "shares": 3, "price": 100.0}
    assert selected_buy_tickers([raw_order]) == set()

    row = build_buy_trade_event(raw_order, date="2026-05-24")
    assert selected_buy_tickers([row]) == {"AAPL"}


def test_adapters_use_shared_buy_trade_event_builder():
    for rel in ("adapters/sim.py", "adapters/runner.py", "adapters/lean.py"):
        src = (STRATEGY_DIR / rel).read_text()
        assert "build_buy_trade_event(" in src, (
            f"{rel} must build BUY trade rows through the shared helper"
        )


def test_build_sell_trade_event_preserves_exit_source_and_tax_payload():
    today = datetime.date(2026, 5, 24)
    holding = HoldingState(
        entry_price=100.0,
        entry_date=today - datetime.timedelta(days=45),
        high_watermark=132.0,
        shares=10.0,
        entry_regime="BULL_CALM",
        rank_score=0.72,
        panel_score=0.55,
        expected_return=0.022,
        expected_return_horizon_days=60,
        mu=0.014,
        mu_horizon_days=60,
        sigma=0.03,
        kelly_target_pct=0.11,
    )
    sig = ExitSignal(
        should_exit=True,
        reason="QP reduced target weight",
        exit_type="qp_sell",
        quantity=4.0,
    )
    sig.source_job = "JointPortfolioQPJob"
    sig.source_task = "EmitOrdersFromQPSolutionTask"
    sig.decision_inputs = {"qp_delta": -0.04}

    row = build_sell_trade_event(
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
            "tax": {"short_term_rate": 0.40, "long_term_rate": 0.20},
        },
        attribution_version="unit_sell_v1",
    )

    assert row["action"] == "sell"
    assert row["source_job"] == "JointPortfolioQPJob"
    assert row["source_task"] == "EmitOrdersFromQPSolutionTask"
    assert row["order_source"] == "JointPortfolioQPJob.EmitOrdersFromQPSolutionTask"
    assert row["shares"] == 4.0
    assert row["gross_pnl"] == 40.0
    assert row["proceeds_basis"] == 400.0
    assert row["tax"] == 16.0
    assert row["net_pnl_after_tax"] == 24.0
    assert row["tax_cash_debited"] == 16.0
    assert row["tax_cash_debit_mode"] == "event_level"
    assert row["tax_lot_method"] == "fifo"
    assert row["pnl_pct"] == 0.10
    assert row["hold_days"] == 45
    assert row["rank_score"] == 0.72
    assert row["panel_score"] == 0.55
    assert row["expected_return"] == 0.022
    assert row["kelly_target_pct"] == 0.11
    assert row["regime"] == "BULL_CALM"
    assert row["confidence"] == 0.8
    assert row["score_snapshot"]["expected_return_horizon_days"] == 60
    assert row["score_snapshot"]["mu_horizon_days"] == 60
    assert row["score_snapshot"]["kelly_target_pct"] == 0.11
    assert row["decision_inputs"]["take_profit_pct"] == 0.30
    assert row["decision_inputs"]["qp_delta"] == -0.04
    assert row["attribution_version"] == "unit_sell_v1"


def test_build_sell_trade_event_uses_applied_exit_params_when_present():
    today = datetime.date(2026, 5, 24)
    holding = HoldingState(
        entry_price=100.0,
        entry_date=today - datetime.timedelta(days=20),
        high_watermark=115.0,
        shares=5.0,
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

    row = build_sell_trade_event(
        ticker="AAPL",
        sig=sig,
        holding=holding,
        price=84.0,
        today=today,
        regime="CHOPPY",
        confidence=0.5,
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


def test_adapters_use_shared_sell_trade_event_builder():
    runner_src = (STRATEGY_DIR / "adapters/runner.py").read_text()
    lean_src = (STRATEGY_DIR / "adapters/lean.py").read_text()

    assert (
        "build_sell_trade_event as build_sell_trade_event_for_db" in runner_src
    )
    assert "build_sell_trade_event(" in lean_src
