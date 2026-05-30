"""ManualExecutionPipeline — T/J/P shape + safety-gate tests.

§5.13.5 mandate: prod-touching scripts get tests. Wash-sale + earnings
gates MUST refuse BUYs that would violate; the pipeline order is pinned.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "renquant-common" / "src"))

from execute_shadow_orders import (
    ExecuteJob,
    ExecutionContext,
    LoadJob,
    LoadOrderListTask,
    LoadStateTask,
    OrderItem,
    SubmitOrdersTask,
    ValidateBrokerLiveTask,
    ValidateEarningsTask,
    ValidateJob,
    ValidateMarketHoursTask,
    ValidateWashSaleTask,
    build_pipeline,
)


@pytest.fixture
def state_path(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({
        "last_sell_dates": {
            "DUK": "2026-05-18",   # 12 days ago vs 2026-05-30
            "GE": "2026-05-14",    # 16 days ago
            "ORCL": "2026-05-04",  # 26 days ago
            "OLDIE": "2026-01-01", # >30d ago
        },
    }))
    return p


@pytest.fixture
def ctx(tmp_path, state_path):
    return ExecutionContext(
        raw_orders=[],
        execute=False,
        state_path=state_path,
        today=datetime.date(2026, 5, 30),
    )


def test_pipeline_has_three_ordered_jobs():
    p = build_pipeline()
    assert p.name == "ManualExecution"
    assert [type(j).__name__ for j in p.jobs] == ["LoadJob", "ValidateJob", "ExecuteJob"]


def test_load_job_tasks():
    assert [type(t).__name__ for t in LoadJob().tasks] == [
        "LoadOrderListTask", "LoadStateTask",
    ]


def test_validate_job_tasks_in_order():
    """Order matters: broker first, wash-sale before earnings, market last."""
    assert [type(t).__name__ for t in ValidateJob().tasks] == [
        "ValidateBrokerLiveTask",
        "ValidateWashSaleTask",
        "ValidateEarningsTask",
        "ValidateMarketHoursTask",
    ]


def test_execute_job_has_submit_orders():
    assert [type(t).__name__ for t in ExecuteJob().tasks] == ["SubmitOrdersTask"]


def test_load_order_list_parses_canonical_form(ctx):
    ctx.raw_orders = ["MU:SELL:1", "GILD:BUY:4", "GE:SELL:3"]
    LoadOrderListTask().run(ctx)
    assert len(ctx.orders) == 3
    o0, o1, o2 = ctx.orders
    assert (o0.ticker, o0.side, o0.qty) == ("MU", "SELL", 1)
    assert (o1.ticker, o1.side, o1.qty) == ("GILD", "BUY", 4)
    assert (o2.ticker, o2.side, o2.qty) == ("GE", "SELL", 3)
    assert all(o.decision == "pending" for o in ctx.orders)


def test_load_order_list_rejects_bad_format(ctx):
    ctx.raw_orders = ["MU:BUY"]
    with pytest.raises(ValueError, match="T:SIDE:QTY"):
        LoadOrderListTask().run(ctx)


def test_load_order_list_rejects_bad_side(ctx):
    ctx.raw_orders = ["MU:SHORT:1"]
    with pytest.raises(ValueError, match="BUY"):
        LoadOrderListTask().run(ctx)


def test_load_state_task_reads_file(ctx, state_path):
    LoadStateTask().run(ctx)
    assert ctx.state is not None
    assert "last_sell_dates" in ctx.state


def test_load_state_task_raises_on_missing(ctx, tmp_path):
    ctx.state_path = tmp_path / "ghost.json"
    with pytest.raises(FileNotFoundError):
        LoadStateTask().run(ctx)


def test_wash_sale_refuses_buy_within_30d(ctx, state_path):
    ctx.raw_orders = ["DUK:BUY:6", "GILD:BUY:4", "ORCL:BUY:1"]
    LoadOrderListTask().run(ctx)
    LoadStateTask().run(ctx)
    ValidateWashSaleTask().run(ctx)
    by_t = {o.ticker: o for o in ctx.orders}
    # DUK sold 12d ago → refused
    assert by_t["DUK"].decision == "refused"
    assert "wash_sale" in by_t["DUK"].reason
    # GILD never sold → pending
    assert by_t["GILD"].decision == "pending"
    # ORCL sold 26d ago → refused (<30d)
    assert by_t["ORCL"].decision == "refused"


def test_wash_sale_does_not_affect_sells(ctx):
    """SELL of a recently-sold ticker is fine; wash-sale only blocks BUYs."""
    ctx.raw_orders = ["DUK:SELL:1"]
    LoadOrderListTask().run(ctx)
    LoadStateTask().run(ctx)
    ValidateWashSaleTask().run(ctx)
    assert ctx.orders[0].decision == "pending"


def test_wash_sale_allows_buy_after_30d(ctx):
    """30-day rule: BUYs after 30+ days are fine."""
    ctx.raw_orders = ["OLDIE:BUY:1"]
    LoadOrderListTask().run(ctx)
    LoadStateTask().run(ctx)
    ValidateWashSaleTask().run(ctx)
    assert ctx.orders[0].decision == "pending"


def test_submit_orders_dry_run_does_not_call_broker(ctx, monkeypatch):
    ctx.raw_orders = ["MU:SELL:1"]
    ctx.execute = False
    LoadOrderListTask().run(ctx)
    LoadStateTask().run(ctx)
    # Fake broker so SubmitOrdersTask can run without a real connection
    class FakeBroker:
        def submit_order(self, order_data):
            raise AssertionError("dry-run must NOT call submit_order")
    ctx.broker = FakeBroker()
    SubmitOrdersTask().run(ctx)
    assert ctx.orders[0].decision == "skipped"
    assert "dry-run" in ctx.orders[0].reason


def test_submit_orders_execute_calls_broker(ctx, monkeypatch):
    ctx.raw_orders = ["MU:SELL:1"]
    ctx.execute = True
    LoadOrderListTask().run(ctx)
    LoadStateTask().run(ctx)
    calls = []
    class FakeResp:
        id = "fake-order-id-xyz"
    class FakeBroker:
        def submit_order(self, order_data):
            calls.append(order_data)
            return FakeResp()
    ctx.broker = FakeBroker()
    SubmitOrdersTask().run(ctx)
    assert len(calls) == 1
    assert ctx.orders[0].decision == "submitted"
    assert ctx.orders[0].order_id == "fake-order-id-xyz"


def test_submit_orders_skips_refused(ctx, monkeypatch):
    """Refused orders MUST NOT be sent even in --execute mode."""
    ctx.raw_orders = ["DUK:BUY:6", "MU:SELL:1"]
    ctx.execute = True
    LoadOrderListTask().run(ctx)
    LoadStateTask().run(ctx)
    ValidateWashSaleTask().run(ctx)  # DUK becomes refused
    calls = []
    class FakeResp: id = "x"
    class FakeBroker:
        def submit_order(self, order_data):
            calls.append(order_data); return FakeResp()
    ctx.broker = FakeBroker()
    SubmitOrdersTask().run(ctx)
    # only MU sell submitted; DUK was already refused
    assert len(calls) == 1
    assert calls[0].symbol == "MU"
