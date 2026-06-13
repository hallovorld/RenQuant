"""LEAN order-ticket execution helpers — lean.py decomposition slice 3.

EXTRACTED 2026-06-13 from adapters/lean.py (eng plan S2 item 5, god-file
decomposition; the LEAN counterpart to runner_execmath.broker_order_execution).
Pure functions that read a LEAN QCAlgorithm OrderTicket (or a list of tickets)
into the uniform (filled, qty, avg_price, status) execution tuple — fail-closed
so an unconfirmed order can never mutate LEAN state/tax as if it filled. No
LeanAdapter state. Includes the _positive_finite_price fill-price validator
the execution path depends on. Re-exported from lean for back-compat.
"""
from __future__ import annotations

import math
from typing import Any


def _positive_finite_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(price) and price > 0.0:
        return price
    return None


def _lean_ticket_status_text(ticket: Any) -> str:
    status = getattr(ticket, "Status", None)
    return str(status or "").lower()


def _lean_ticket_float(ticket: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(ticket, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return None


def _lean_order_execution(
    ticket: Any,
    *,
    requested_qty: float,
    fallback_price: float,
) -> tuple[bool, float, float, str]:
    """Return (filled, qty, avg_price, status) for a LEAN order ticket.

    Real tickets must either expose a filled status or a positive filled
    quantity. A missing ticket is not execution evidence; fail closed so LEAN
    cannot mutate state/tax as if an unconfirmed order filled.
    """
    requested_abs = abs(float(requested_qty))
    fallback = _positive_finite_price(fallback_price) or 0.0
    if ticket is None:
        return False, 0.0, fallback, "missing_order_ticket"
    if isinstance(ticket, (list, tuple)):
        total_qty = 0.0
        total_value = 0.0
        statuses: list[str] = []
        for item in ticket:
            ok, qty, px, status = _lean_order_execution(
                item,
                requested_qty=requested_abs,
                fallback_price=fallback,
            )
            statuses.append(status)
            if ok and qty > 0:
                total_qty += qty
                total_value += qty * (px if px > 0 else fallback)
        if total_qty > 0:
            return True, total_qty, total_value / total_qty, ",".join(statuses)
        return False, 0.0, fallback, ",".join(statuses)

    status = _lean_ticket_status_text(ticket)
    if any(token in status for token in ("reject", "cancel", "invalid", "error")):
        return False, 0.0, fallback, status
    qty = _lean_ticket_float(
        ticket,
        "QuantityFilled",
        "AbsoluteQuantityFilled",
        "FilledQuantity",
    )
    price = _lean_ticket_float(
        ticket,
        "AverageFillPrice",
        "AvgFillPrice",
        "FillPrice",
        "Price",
    )
    if qty is not None and abs(qty) > 0:
        return True, abs(qty), price or fallback, status
    if "filled" in status:
        return True, requested_abs, price or fallback, status
    return False, 0.0, price or fallback, status or "unknown"
