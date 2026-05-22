from __future__ import annotations

import datetime
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
KERNEL = REPO / "backtesting" / "renquant_104" / "kernel"
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.pipeline.order_attribution import (  # noqa: E402
    ATTRIBUTION_VERSION,
    stamp_order_attribution,
    validate_order_attribution,
)
from kernel.pipeline.task_selection import SizeAndEmitTask  # noqa: E402
from kernel.selection import CandidateResult  # noqa: E402


BUY_EMITTERS = [
    KERNEL / "pipeline" / "task_selection.py",
    KERNEL / "pipeline" / "task_topup.py",
    KERNEL / "pipeline" / "task_rotation.py",
    KERNEL / "pipeline" / "task_joint_actions.py",
    KERNEL / "portfolio_qp" / "tasks.py",
]


def test_helper_rejects_unattributed_order() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        validate_order_attribution({"ticker": "AAPL", "order_type": "NEW_BUY"})


def test_helper_stamps_source_and_score_snapshot() -> None:
    cand = SimpleNamespace(
        rank_score=0.71,
        panel_score=0.25,
        rs_score=0.03,
        mu=0.02,
        sigma=0.10,
        kelly_target_pct=0.12,
        expected_return=0.04,
    )
    ctx = SimpleNamespace(regime="HIGH_CALM", confidence=0.8)

    order = stamp_order_attribution(
        {"ticker": "AAPL", "order_type": "NEW_BUY"},
        ctx=ctx,
        source_job="SelectionJob",
        source_task="SizeAndEmitTask",
        acceptance_reason="unit_test",
        source_obj=cand,
        decision_inputs={"floor": 0.20},
    )

    assert order["attribution_version"] == ATTRIBUTION_VERSION
    assert order["order_source"] == "SelectionJob.SizeAndEmitTask"
    assert order["decision_inputs"]["acceptance_reason"] == "unit_test"
    assert order["decision_inputs"]["floor"] == 0.20
    assert order["score_snapshot"]["rank_score"] == pytest.approx(0.71)
    assert order["score_snapshot"]["expected_return"] == pytest.approx(0.04)
    assert order["score_snapshot"]["regime"] == "HIGH_CALM"


def test_size_and_emit_task_stamps_runtime_order_attribution() -> None:
    cand = CandidateResult(
        ticker="AAPL",
        raw_score=0.10,
        rank_score=0.72,
        rs_score=0.01,
        detail="unit candidate",
        expected_return=0.05,
        panel_score=0.20,
        mu=0.02,
        sigma=0.10,
    )
    cand.kelly_target_pct = 0.12
    ctx = SimpleNamespace(
        config={
            "regime_params": {
                "BULL_CALM": {
                    "max_position_pct": 0.10,
                    "cash_reserve_pct": 0.0,
                }
            },
            "ranking": {"panel_scoring": {}, "kelly_sizing": {}},
        },
        today=datetime.date(2026, 5, 22),
        regime="BULL_CALM",
        confidence=1.0,
        buy_blocked=False,
        skip_buys=False,
        _selected=["AAPL"],
        ranked=[cand],
        orders=[],
        counters={},
        prices={"AAPL": 100.0},
        portfolio_value=10_000.0,
        cash=10_000.0,
        regime_state=None,
        bear_only=False,
    )

    SizeAndEmitTask().run(ctx)

    assert len(ctx.orders) == 1
    order = ctx.orders[0]
    validate_order_attribution(order)
    assert order["order_type"] == "NEW_BUY"
    assert order["order_source"] == "SelectionJob.SizeAndEmitTask"
    assert order["source_task"] == "SizeAndEmitTask"
    assert order["score_snapshot"]["rank_score"] == pytest.approx(0.72)
    assert order["decision_inputs"]["acceptance_reason"] == "selected_by_greedy_loop"


def test_all_buy_emitters_call_attribution_helper() -> None:
    offenders = []
    for path in BUY_EMITTERS:
        src = path.read_text()
        for line_no, line in enumerate(src.splitlines(), start=1):
            if "ctx.orders.append(" in line and "stamp_order_attribution" not in line:
                offenders.append(f"{path.relative_to(REPO)}:{line_no}")
    assert offenders == []
