#!/usr/bin/env python3
"""P0.2a protective-order census — every held name protected broker-side.

Design: renquant-orchestrator
doc/research/2026-06-12-intraday-trading-roadmap.md §4 P0.2, whose
acceptance gate is exactly this check: "order-book census shows every
held name protected broker-side." The G1 incident class (Z9 fully
implemented but never enabled → broker order book EMPTY → every
position naked) becomes a daily-verified invariant instead of a
post-mortem discovery.

Read-only: lists positions and open orders; never places, cancels, or
modifies anything. A long position counts as protected when a live
broker-resident SELL order of a protective type (stop / stop_limit /
trailing_stop) covers it; shorts respectively need a protective BUY
(cover) order.

Exit codes: 0 all protected / 1 naked positions found (+ ntfy) /
2 broker unreachable (+ ntfy — fail LOUD, an unreachable book is not a
clean census).

The bracket/trailing upgrade itself (replacing the polled trailing leg)
is deliberately sequenced AFTER the broker-reconciliation SM reaches
production — broker-side fills must be adopted cleanly before more
order types live broker-side.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

log = logging.getLogger("protective_census")

PROTECTIVE_TYPES = {"stop", "stop_limit", "trailing_stop"}


def census(positions: list[dict], open_orders: list[dict]) -> dict:
    """Pure: positions × orders → {protected, naked, orphan_orders}.

    positions: [{symbol, qty}]            (qty signed; + long, - short)
    open_orders: [{symbol, side, type, qty}]
    """
    protective: dict[str, float] = {}
    order_symbols = set()
    for o in open_orders:
        order_symbols.add(o["symbol"])
        if str(o.get("type", "")).lower() not in PROTECTIVE_TYPES:
            continue
        side = str(o.get("side", "")).lower()
        protective.setdefault(o["symbol"] + ":" + side, 0.0)
        protective[o["symbol"] + ":" + side] += abs(float(o.get("qty", 0) or 0))

    protected, naked = [], []
    held = set()
    for p in positions:
        sym, qty = p["symbol"], float(p["qty"])
        if qty == 0:
            continue
        held.add(sym)
        need_side = "sell" if qty > 0 else "buy"
        covered = protective.get(f"{sym}:{need_side}", 0.0)
        entry = {"symbol": sym, "qty": qty, "covered_qty": covered}
        if covered >= abs(qty):
            protected.append(entry)
        else:
            naked.append(entry)
    orphan = sorted(s.split(":")[0] for s in protective
                    if s.split(":")[0] not in held)
    return {"protected": protected, "naked": naked, "orphan_orders": orphan}


def _broker_snapshot():
    from live.alpaca_broker import AlpacaBroker  # noqa: PLC0415
    from alpaca.trading.enums import QueryOrderStatus  # noqa: PLC0415
    from alpaca.trading.requests import GetOrdersRequest  # noqa: PLC0415

    broker = AlpacaBroker(paper=False)
    broker.connect()
    positions = [{"symbol": p["symbol"], "qty": p["qty"]}
                 for p in broker.get_all_positions()]
    raw = broker._trading_client.get_orders(  # noqa: SLF001 — read-only census
        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
    open_orders = [{
        "symbol": str(o.symbol),
        "side": str(getattr(o.side, "value", o.side)),
        "type": str(getattr(o.type, "value", o.type)),
        "qty": float(o.qty or 0),
    } for o in raw]
    return positions, open_orders


def _alert(title: str, body: str, key_parts: tuple, priority: str = "high") -> None:
    try:
        from live.alerts import AlertEvent, post_ntfy_alert, stable_alert_key  # noqa: PLC0415

        topic = os.environ.get("RENQUANT_NTFY_TOPIC", "renquant")
        post_ntfy_alert(
            f"https://ntfy.sh/{topic}",
            AlertEvent(
                taxonomy="census.protective_orders",
                title=title,
                body=body,
                key=stable_alert_key("pcensus", *key_parts),
                priority=priority,
            ),
            logger=log,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("ntfy failed: %s", exc)


def main() -> int:
    try:
        positions, open_orders = _broker_snapshot()
    except Exception as exc:  # noqa: BLE001
        log.error("broker unreachable: %s", exc)
        _alert("PROTECTIVE CENSUS FAILED — broker unreachable",
               f"Cannot verify broker-side protection: {exc}",
               ("unreachable", dt.date.today()), priority="urgent")
        return 2
    result = census(positions, open_orders)
    log.info("census: %d protected, %d naked, %d orphan order symbol(s)",
             len(result["protected"]), len(result["naked"]),
             len(result["orphan_orders"]))
    if result["orphan_orders"]:
        log.warning("orphan protective orders (no position): %s",
                    result["orphan_orders"])
    if result["naked"]:
        names = ", ".join(f"{e['symbol']}({e['qty']:g}, covered "
                          f"{e['covered_qty']:g})" for e in result["naked"])
        _alert("NAKED POSITIONS — no broker-side protection",
               f"{len(result['naked'])} held name(s) lack a live protective "
               f"order: {names}. G1/Z9 should have placed GTC stops — "
               f"investigate before next session.",
               ("naked", dt.date.today()), priority="urgent")
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(main())
