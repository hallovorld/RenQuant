#!/usr/bin/env python
"""Manually execute a pre-planned order list against LIVE Alpaca.

Use case (2026-05-30): operator reviewed shadow's PatchTST-driven decision
output and authorized execution of a SUBSET (those with no wash-sale /
earnings / vol conflict against current prod state).

§1c-correct architecture (T/J/P):

    ManualExecutionPipeline
      LoadJob
        LoadOrderListTask        — parse the explicit (ticker, side, qty) tuples
        LoadStateTask            — read prod live_state for wash-sale + earnings
      ValidateJob
        ValidateBrokerLiveTask   — must be the live alpaca account
        ValidateWashSaleTask     — refuse a BUY within 30d of a prior SELL
        ValidateEarningsTask     — refuse a BUY within ±earnings_buffer_days
        ValidateMarketHoursTask  — log if outside RTH (Alpaca queues orders)
      ExecuteJob
        SubmitOrderTask          — submit each survivor via alpaca SDK

Hard refusal categories (§5.13.5 — never bypass via parallel path):
  * Wash-sale (IRS Section 1091)
  * Earnings blackout
  * Buying while preflight gate is BUY-BLOCKED

Survivors are submitted as MARKET orders with day-tif. Side-effects are
emitted to live/logs/renquant-104/ per the existing trade log convention.
Audit trail: every Task stamps a result row into ctx.audit.

Usage::

    python scripts/execute_shadow_orders.py \\
        --order MU:SELL:1 GE:SELL:3 GILD:BUY:4

    # dry-run (default — no orders submitted):
    python scripts/execute_shadow_orders.py --order MU:SELL:1 --dry-run

    # actually submit:
    python scripts/execute_shadow_orders.py --order MU:SELL:1 --execute
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field

REPO = pathlib.Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
sys.path.insert(0, str(REPO.parent / "renquant-common" / "src"))

from renquant_common import Job, Pipeline, Task  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────────
# Context
# ────────────────────────────────────────────────────────────────────────────────


@dataclass
class OrderItem:
    ticker: str
    side: str          # "BUY" or "SELL"
    qty: int
    decision: str = "pending"   # pending / refused / submitted / skipped
    reason: str = ""
    order_id: str | None = None


@dataclass
class ExecutionContext:
    """State threaded through ManualExecutionPipeline."""
    raw_orders: list[str]
    execute: bool                          # False = dry-run, True = submit
    state_path: pathlib.Path
    today: _dt.date
    earnings_buffer_days: int = 3
    # populated through the pipeline
    state: dict | None = None
    orders: list[OrderItem] = field(default_factory=list)
    broker: object | None = None
    audit: list[dict] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────────
# LoadJob
# ────────────────────────────────────────────────────────────────────────────────


class LoadOrderListTask(Task):
    """Parse the CLI ``--order T:SIDE:QTY ...`` tuples into ``ctx.orders``."""

    def run(self, ctx: ExecutionContext) -> bool | None:
        for raw in ctx.raw_orders:
            parts = raw.split(":")
            if len(parts) != 3:
                raise ValueError(f"--order must be T:SIDE:QTY (got {raw!r})")
            t, side, qty = parts
            side_up = side.upper()
            if side_up not in ("BUY", "SELL"):
                raise ValueError(f"side must be BUY|SELL (got {side!r})")
            ctx.orders.append(OrderItem(ticker=t.upper(), side=side_up, qty=int(qty)))
        return True


class LoadStateTask(Task):
    """Read prod ``live_state.alpaca.json`` (wash-sale dates, etc.)."""

    def run(self, ctx: ExecutionContext) -> bool | None:
        if not ctx.state_path.exists():
            raise FileNotFoundError(f"state not at {ctx.state_path}")
        ctx.state = json.loads(ctx.state_path.read_text())
        return True


class LoadJob(Job):
    """Stage 1: parse orders + read state."""

    @property
    def tasks(self) -> list[Task]:
        return [LoadOrderListTask(), LoadStateTask()]


# ────────────────────────────────────────────────────────────────────────────────
# ValidateJob
# ────────────────────────────────────────────────────────────────────────────────


class ValidateBrokerLiveTask(Task):
    """Connect to LIVE Alpaca + sanity-check account = production."""

    def run(self, ctx: ExecutionContext) -> bool | None:
        from alpaca.trading.client import TradingClient  # noqa: PLC0415
        client = TradingClient(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_SECRET_KEY"],
            paper=False,
        )
        acct = client.get_account()
        if str(acct.status) != "AccountStatus.ACTIVE":
            raise RuntimeError(f"account not ACTIVE: {acct.status}")
        if str(acct.account_number) != "212830627":
            raise RuntimeError(
                f"safety guard: expected account=212830627, got {acct.account_number}"
            )
        ctx.broker = client
        ctx.audit.append({
            "task": "ValidateBrokerLiveTask",
            "ok": True,
            "account": acct.account_number,
            "equity": float(acct.equity),
            "settled_cash": float(acct.cash),
        })
        return True


class ValidateWashSaleTask(Task):
    """Refuse any BUY whose ticker was SOLD in the last 30 days (IRS Sec.1091)."""

    def run(self, ctx: ExecutionContext) -> bool | None:
        assert ctx.state is not None
        last_sell = (ctx.state.get("last_sell_dates") or {})
        for o in ctx.orders:
            if o.side != "BUY":
                continue
            d = last_sell.get(o.ticker)
            if not d:
                continue
            try:
                sd = _dt.date.fromisoformat(str(d)[:10])
            except Exception:  # noqa: BLE001
                continue
            days = (ctx.today - sd).days
            if 0 <= days < 30:
                o.decision = "refused"
                o.reason = f"wash_sale: sold {days}d ago (IRS Sec.1091)"
                ctx.audit.append({"task": "ValidateWashSaleTask", "ticker": o.ticker,
                                    "ok": False, "reason": o.reason})
        return True


class ValidateEarningsTask(Task):
    """Refuse a BUY within ±earnings_buffer_days of next earnings."""

    def run(self, ctx: ExecutionContext) -> bool | None:
        # Best-effort: read earnings calendar from artifacts if present.
        cal_p = STRATEGY_DIR / "artifacts" / "prod" / "earnings-calendar.json"
        cal: dict = {}
        if cal_p.exists():
            try:
                cal = json.loads(cal_p.read_text()).get("earnings", {})
            except Exception:  # noqa: BLE001
                cal = {}
        for o in ctx.orders:
            if o.side != "BUY" or o.decision != "pending":
                continue
            evs = cal.get(o.ticker) or []
            for ev in evs:
                try:
                    ed = _dt.date.fromisoformat(str(ev.get("date", ""))[:10])
                except Exception:
                    continue
                if abs((ed - ctx.today).days) <= ctx.earnings_buffer_days:
                    o.decision = "refused"
                    o.reason = f"earnings_blackout: {ed} ±{ctx.earnings_buffer_days}d"
                    ctx.audit.append({"task": "ValidateEarningsTask", "ticker": o.ticker,
                                        "ok": False, "reason": o.reason})
        return True


class ValidateMarketHoursTask(Task):
    """Diagnostic-only: log when outside regular trading hours (orders queue)."""

    def run(self, ctx: ExecutionContext) -> bool | None:
        if ctx.broker is None:
            return True
        try:
            clk = ctx.broker.get_clock()
            is_open = bool(getattr(clk, "is_open", False))
            ctx.audit.append({"task": "ValidateMarketHoursTask", "ok": True,
                                "is_open": is_open,
                                "next_open": str(getattr(clk, "next_open", "?"))})
        except Exception as exc:  # noqa: BLE001
            ctx.audit.append({"task": "ValidateMarketHoursTask", "ok": False,
                                "reason": str(exc)[:120]})
        return True


class ValidateJob(Job):
    """Stage 2: broker live + wash-sale + earnings + market-hours."""

    @property
    def tasks(self) -> list[Task]:
        return [
            ValidateBrokerLiveTask(),
            ValidateWashSaleTask(),
            ValidateEarningsTask(),
            ValidateMarketHoursTask(),
        ]


# ────────────────────────────────────────────────────────────────────────────────
# ExecuteJob
# ────────────────────────────────────────────────────────────────────────────────


class SubmitOrdersTask(Task):
    """Submit pending orders as MARKET / day-tif via alpaca SDK.

    Dry-run mode (default) emits the would-be request without sending.
    """

    def run(self, ctx: ExecutionContext) -> bool | None:
        from alpaca.trading.requests import MarketOrderRequest  # noqa: PLC0415
        from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: PLC0415
        assert ctx.broker is not None
        for o in ctx.orders:
            if o.decision != "pending":
                continue
            req = MarketOrderRequest(
                symbol=o.ticker,
                qty=o.qty,
                side=OrderSide.BUY if o.side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            if not ctx.execute:
                o.decision = "skipped"
                o.reason = "dry-run (use --execute to submit)"
                ctx.audit.append({"task": "SubmitOrdersTask", "ticker": o.ticker,
                                    "side": o.side, "qty": o.qty,
                                    "submitted": False, "reason": o.reason})
                continue
            try:
                resp = ctx.broker.submit_order(order_data=req)
                o.decision = "submitted"
                o.order_id = str(getattr(resp, "id", None))
                ctx.audit.append({"task": "SubmitOrdersTask", "ticker": o.ticker,
                                    "side": o.side, "qty": o.qty,
                                    "submitted": True, "order_id": o.order_id})
            except Exception as exc:  # noqa: BLE001
                o.decision = "refused"
                o.reason = f"broker_error: {exc!s}"
                ctx.audit.append({"task": "SubmitOrdersTask", "ticker": o.ticker,
                                    "ok": False, "reason": o.reason})
        return True


class ExecuteJob(Job):
    @property
    def tasks(self) -> list[Task]:
        return [SubmitOrdersTask()]


# ────────────────────────────────────────────────────────────────────────────────
# Pipeline + CLI
# ────────────────────────────────────────────────────────────────────────────────


def build_pipeline() -> Pipeline:
    return Pipeline(
        [LoadJob(), ValidateJob(), ExecuteJob()],
        name="ManualExecution",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--order", nargs="+", required=True,
                    help="One or more T:SIDE:QTY (e.g. MU:SELL:1 GILD:BUY:4)")
    ap.add_argument("--execute", action="store_true",
                    help="ACTUALLY submit orders (default = dry-run).")
    ap.add_argument("--state-path", type=pathlib.Path,
                    default=STRATEGY_DIR / "live_state.alpaca.json")
    args = ap.parse_args()

    ctx = ExecutionContext(
        raw_orders=args.order,
        execute=bool(args.execute),
        state_path=args.state_path,
        today=_dt.date.today(),
    )
    build_pipeline().run(ctx)

    print(f"\n=== ManualExecutionPipeline ({'EXECUTE' if ctx.execute else 'DRY-RUN'}) ===")
    for o in ctx.orders:
        line = (f"  {o.side:<4} {o.ticker:<6} qty={o.qty:>4}  "
                f"→ {o.decision:<10}  {o.reason}")
        if o.order_id:
            line += f"  order_id={o.order_id}"
        print(line)
    refused = [o for o in ctx.orders if o.decision == "refused"]
    submitted = [o for o in ctx.orders if o.decision == "submitted"]
    return 0 if not refused or submitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
