"""Regression guard for sell-source attribution in SimAdapter.

Portfolio-level exits such as QP sells are emitted outside ``TickerSellJob``.
The simulator must preserve their source metadata in the trade log so the
post-run decision tree can explain which pipeline component caused the trade.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from adapters.sim import SimAdapter  # noqa: E402
from kernel.exits import ExitSignal, HoldingState  # noqa: E402


def test_apply_sell_preserves_exit_signal_source_metadata():
    adapter = SimAdapter.__new__(SimAdapter)
    today = datetime.date(2026, 5, 20)
    adapter._holdings = {
        "AAPL": HoldingState(
            entry_price=100.0,
            entry_date=today - datetime.timedelta(days=60),
            high_watermark=125.0,
            shares=10.0,
        )
    }
    adapter._pos_shares = {"AAPL": 10.0}
    adapter._cash = 0.0
    adapter._last_sell_date = {}
    adapter._last_sell_pls = {}
    adapter._trade_log = []
    adapter._ohlcv = {}

    ctx = SimpleNamespace(
        prices={"AAPL": 110.0},
        config={
            "tax": {
                "short_term_rate": 0.37,
                "long_term_rate": 0.20,
                "long_term_threshold_days": 365,
            },
            "regime_params": {
                "BULL_CALM": {
                    "take_profit_pct": 0.30,
                    "stop_decay_days": 60,
                    "stop_decay_floor": 0.08,
                    "sdl_skip_if_unrealized_above": 0.02,
                },
            },
        },
        regime="BULL_CALM",
        confidence=0.8,
    )
    sig = ExitSignal(
        should_exit=True,
        reason="QP reduced target weight",
        exit_type="qp_sell",
        quantity=3.0,
    )
    sig.source_job = "JointPortfolioQPJob"
    sig.source_task = "EmitOrdersFromQPSolutionTask"

    adapter._apply_sell("AAPL", sig, pd.Timestamp(today), ctx)

    row = adapter._trade_log[0]
    assert row["source_job"] == "JointPortfolioQPJob"
    assert row["source_task"] == "EmitOrdersFromQPSolutionTask"
    assert row["order_source"] == "JointPortfolioQPJob.EmitOrdersFromQPSolutionTask"
    assert row["source"] == "ExitPipeline"
    assert row["exit_take_profit_pct"] == 0.30
    assert row["exit_sdl_skip_if_unrealized_above"] == 0.02
    assert row["decision_inputs"]["take_profit_pct"] == 0.30
    assert row["decision_inputs"]["stop_decay_days"] == 60
    assert row["decision_inputs"]["stop_decay_floor"] == 0.08
    assert row["decision_inputs"]["sdl_skip_if_unrealized_above"] == 0.02
