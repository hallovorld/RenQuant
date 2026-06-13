"""Execution math — runner.py decomposition slice 5 (order_emit, cash/exec).

EXTRACTED 2026-06-13 from adapters/runner.py (eng plan S2 item 5). Pure
functions for the cash/execution arithmetic around order submission:
cash-cap a buy, same-bar sell credit, normalize broker status, summarize
a broker execution attempt, project holdings after orders, and snapshot
post-execution. No broker calls of their own (broker_order_execution
takes the already-returned result dict). Moved verbatim; re-exported
from runner for back-compat.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("adapters.runner")  # same logger — log contract unchanged


def cap_buy_order_to_cash(order: dict, remaining_cash: float) -> tuple[dict | None, str | None]:
    """Resize or reject one buy intent against the runner's live cash ledger."""
    import math
    try:
        cash = float(remaining_cash)
        shares = float(order.get("shares", 0.0))
        price = float(order.get("price", 0.0))
    except (TypeError, ValueError, AttributeError):
        return None, "bad_order"
    if not (math.isfinite(cash) and math.isfinite(shares)
            and math.isfinite(price) and price > 0 and shares > 0):
        return None, "bad_order"
    invest = shares * price
    if invest <= cash + 1e-6:
        capped = dict(order)
        capped["invest"] = invest
        return capped, None
    affordable = int(cash // price)
    if affordable < 1:
        return None, "cash_budget_exhausted"
    capped = dict(order)
    capped["shares"] = affordable
    capped["invest"] = affordable * price
    capped["budget_adjustment"] = "cash_budget_resized"
    capped["original_shares"] = order.get("shares")
    return capped, "cash_budget_resized"


def same_bar_sell_credit(ctx: Any) -> float:
    """Estimated cash made available by broker-confirmed same-bar sells."""
    import math
    credit = 0.0
    for ticker, sig in getattr(ctx, "exits_placed", []) or []:
        try:
            shares = float(getattr(sig, "shares_sold", 0.0) or 0.0)
            price = float(getattr(sig, "sell_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(shares) and math.isfinite(price) and shares > 0 and price > 0:
            credit += shares * price
        else:
            log.warning(
                "LIVE-SAME-BAR-SELL-CREDIT: skip non-finite sell credit "
                "%s shares=%s price=%s",
                ticker, shares, price,
            )
    return credit


def normalize_order_status(status: Any) -> str:
    """Normalize broker enum/string order status to a lower-case token."""
    return str(status or "").split(".")[-1].strip().lower()


def broker_order_execution(
    result: dict | None,
    requested_qty: float,
    fallback_price: float,
) -> dict[str, Any]:
    """Classify a broker order response as filled, pending, or rejected.

    Live Alpaca can accept an after-close market DAY order without executing it
    until the next session. Only filled quantity is allowed to mutate live
    state, trade DB rows, same-bar cash credit, or realized P/L.
    """
    import math

    result = dict(result or {})
    status = normalize_order_status(result.get("status"))
    terminal_rejects = {
        "rejected", "canceled", "cancelled", "expired",
        "stopped", "suspended", "done_for_day",
    }
    def _finite_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return out if math.isfinite(out) else default

    requested = _finite_float(requested_qty)
    filled_qty = _finite_float(result.get("filled_qty"))
    if filled_qty <= 0 and status in {"filled", "partially_filled"}:
        filled_qty = _finite_float(result.get("quantity"), requested)
    avg_price = _finite_float(result.get("filled_avg_price"))
    if avg_price <= 0:
        avg_price = _finite_float(result.get("price"), fallback_price)

    is_filled = filled_qty > 0 or status == "filled"
    is_partial = (
        status == "partially_filled"
        or (is_filled and requested > 0 and filled_qty < requested - 1e-9)
    )
    is_rejected = status in terminal_rejects
    is_pending = not is_filled and not is_rejected
    return {
        **result,
        "status": status,
        "filled": bool(is_filled),
        "pending": bool(is_pending),
        "rejected": bool(is_rejected),
        "partial": bool(is_partial),
        "filled_qty": float(filled_qty if is_filled else 0.0),
        "filled_avg_price": float(avg_price if avg_price > 0 else fallback_price),
    }


def effective_live_holdings_after_orders(
    starting_holding_tickers: Any,
    full_exit_tickers: set[str],
    orders_placed: Any,
) -> set[str]:
    """Return live holdings after confirmed full exits and filled buys.

    ``ctx.holdings`` is a start-of-bar snapshot. RunnerAdapter must subtract
    broker-confirmed full exits before state GC, otherwise it can resurrect
    sell streak / HWM state for positions that were just liquidated.
    """
    current = {str(t) for t in (starting_holding_tickers or []) if t}
    current.difference_update({str(t) for t in (full_exit_tickers or set()) if t})
    for order in orders_placed or []:
        ticker = order.get("ticker") if isinstance(order, dict) else None
        if ticker:
            current.add(str(ticker))
    return current


def live_post_execution_snapshot(
    ctx: Any,
    broker: Any,
    currently_held: set[str],
) -> dict[str, Any]:
    """Best-effort post-order account snapshot for persistence metrics."""
    import math

    def _finite(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    pv = None
    if hasattr(broker, "get_account_value"):
        try:
            pv = _finite(broker.get_account_value())
        except Exception:
            pv = None
    if pv is None:
        pv = _finite(getattr(ctx, "portfolio_value", None))

    cash = None
    if hasattr(broker, "get_cash"):
        try:
            cash = _finite(broker.get_cash())
        except Exception:
            cash = None
    if cash is None:
        cash = _finite(getattr(ctx, "cash", None))

    return {
        "portfolio_value": pv,
        "cash": cash,
        "n_holdings": len(currently_held),
    }
