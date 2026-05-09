"""Realized P/L computation from broker fill history (FIFO matching).

Used by the cost-aware wash-sale filter (`kernel.selection.is_wash_sale_blocked_with_cost`)
to determine whether a recent sale was a GAIN (no §1091 cost) or a LOSS
(deferred-tax NPV cost).

API:
    compute_recent_realized_pnl(broker, days=35) -> dict[ticker, $ P/L]

Returns: per-ticker total realized $ P/L over the window, FIFO-matched
against earlier buys. Tickers with no closed sells in the window are
absent from the dict (caller treats absence as "no recent sale").

Implementation:
  - Calls broker.get_filled_orders(after=...) when available (Alpaca)
  - Returns {} for paper-sim brokers without an order history API —
    caller falls back to binary wash-sale (conservative)
  - FIFO matches sells against buys; ignores any unmatched (cost basis
    pre-window)

Reference: IRC §1091 (the rule itself); §1012 + §1091(d) (basis tracking).
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger("kernel.realized_pnl")


def compute_recent_realized_pnl(
    broker: Any,
    *,
    days: int = 35,
) -> dict[str, float]:
    """Return ticker → realized $ P/L over the last `days` days.

    `days` should be slightly larger than the wash-sale window (default
    30+5=35) so we capture all sales that could trigger §1091 cost
    consideration.

    Sales matched FIFO against earlier buys. Returns total $ P/L per
    ticker (positive = realized gain → no wash-sale cost).
    """
    if not hasattr(broker, "get_filled_orders"):
        return {}

    after_dt = datetime.now(timezone.utc) - timedelta(days=days)
    after_str = after_dt.date().isoformat()
    try:
        fills = broker.get_filled_orders(after=after_str) or []
    except Exception as exc:
        log.warning("compute_recent_realized_pnl: broker.get_filled_orders failed: %s", exc)
        return {}

    # Sort by filled_at ascending so FIFO matching is correct
    def _ts(f):
        return f.get("filled_at") or ""
    fills_sorted = sorted(fills, key=_ts)

    # FIFO buy queue per symbol
    buys: dict[str, deque] = defaultdict(deque)   # symbol → deque of (qty, price)
    pl: dict[str, float] = defaultdict(float)
    for f in fills_sorted:
        sym = str(f.get("symbol") or "").upper()
        action = str(f.get("action") or "").upper()
        try:
            qty = float(f.get("qty") or 0)
            price = float(f.get("avg_price") or 0)
        except (TypeError, ValueError):
            continue
        if qty <= 0 or price <= 0 or not sym:
            continue
        if action == "BUY":
            buys[sym].append([qty, price])
        elif action == "SELL":
            remaining = qty
            while remaining > 0 and buys[sym]:
                buy_qty, buy_px = buys[sym][0]
                consumed = min(remaining, buy_qty)
                pl[sym] += (price - buy_px) * consumed
                buy_qty -= consumed
                remaining -= consumed
                if buy_qty <= 0:
                    buys[sym].popleft()
                else:
                    buys[sym][0][0] = buy_qty
            # Any unmatched sell quantity = sold pre-window inventory;
            # cannot compute that lot's P/L without older history.
            # Skip it — wash-sale logic will fall back to binary.

    # Filter out zero-pl entries (only buys, no sells)
    return {k: v for k, v in pl.items() if abs(v) > 1e-9}
