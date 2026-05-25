"""Shared order-intent de-duplication helpers.

Adapters should not each invent their own same-bar buy semantics. The
pipeline contract is first-write-wins: once one task emits a buy for a ticker
on a bar, later duplicate buy intents for that ticker are audit-skipped.
"""
from __future__ import annotations

from typing import Any


def order_ticker(order: Any) -> str | None:
    if isinstance(order, dict):
        raw = order.get("ticker")
    else:
        raw = getattr(order, "ticker", None)
    if raw is None:
        return None
    ticker = str(raw).strip()
    return ticker or None


def dedupe_buy_orders_first_wins(
    orders: list[Any] | tuple[Any, ...] | None,
) -> tuple[list[Any], list[Any]]:
    """Return ``(kept, skipped_duplicates)`` for same-ticker buy intents."""
    seen: set[str] = set()
    kept: list[Any] = []
    skipped: list[Any] = []
    for order in orders or []:
        ticker = order_ticker(order)
        if ticker is None:
            kept.append(order)
            continue
        if ticker in seen:
            skipped.append(order)
            continue
        seen.add(ticker)
        kept.append(order)
    return kept, skipped
