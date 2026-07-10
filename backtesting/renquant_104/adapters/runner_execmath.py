"""Execution math — runner.py decomposition slice 5 (order_emit, cash/exec).

EXTRACTED 2026-06-13 from adapters/runner.py (eng plan S2 item 5). Pure
functions for the cash/execution arithmetic around order submission:
cash-cap a buy, same-bar sell credit, normalize broker status, summarize
a broker execution attempt, project holdings after orders, and snapshot
post-execution. No broker calls of their own (broker_order_execution
takes the already-returned result dict). Moved verbatim; re-exported
from runner for back-compat.

Cash-cap sizing math is NOT implemented here: it is owned by
renquant-execution ``order_math.cap_affordable_qty`` (execution#25) and
``cap_buy_order_to_cash`` is a time-bounded compatibility call-site
(see its docstring for the fail-closed fallback contract).
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("adapters.runner")  # same logger — log contract unchanged

# Delegate OWNER of the cash-cap sizing math: renquant-execution
# ``order_math.cap_affordable_qty`` (execution#25 — ownership moved there
# per the RenQuant#454 review; the deprecated umbrella must not own new
# order math). The live daily run puts the pinned renquant-execution
# checkout on PYTHONPATH; an OLDER pin that predates ``order_math`` must
# degrade FAIL-CLOSED to the legacy whole-share truncation below — never
# crash the commit path, never re-implement fractional math here. This
# whole call-site is time-bounded compatibility surface: it is deleted
# when RunnerAdapter order math migrates into renquant-execution
# (adapter-migration program; renquant-execution owns the cutover).
try:
    from renquant_execution.order_math import (
        cap_affordable_qty as _cap_affordable_qty,
    )
except ImportError:  # older pinned checkout / renquant_execution not on path
    _cap_affordable_qty = None


def cap_buy_order_to_cash(
    order: dict,
    remaining_cash: float,
    *,
    fractional: bool = False,
) -> tuple[dict | None, str | None]:
    """Resize or reject one buy intent against the runner's live cash ledger.

    COMPATIBILITY CALL-SITE ONLY (time-bounded): the sizing math is owned
    by renquant-execution ``order_math.cap_affordable_qty`` and BOTH modes
    delegate there — this wrapper keeps only the runner's order-intent
    envelope (the afford-check epsilon, reason strings, and the resized
    dict shape) plus one fail-closed fallback for an older pinned
    renquant-execution that predates ``order_math``.

    ``fractional=False`` (the live default) is the unchanged legacy
    whole-share behavior: a cash-capped resize truncates to
    ``int(cash // price)`` shares and rejects when < 1 share is affordable
    — byte-identical whether delegated or on the inline fallback, pinned
    by the 4000-case grid in tests/test_runner_execmath_invariants.py.

    ``fractional=True`` (S-FRAC v2 stage 2, ``execution.fractional_shares``)
    floors the affordable quantity to the 6dp sizing grid and rejects below
    the ~$1 broker fractional min-notional (D7 gap inventory #1 semantics —
    see the owner module for the full contract). If the delegate is missing,
    a fractional request degrades to the legacy whole-share cap with a
    logged warning: a too-small whole-share resize (or a reject) is
    conservative; umbrella-local fractional math is not an option. The $25
    anti-churn dust floor (pipeline ``fractional_dust_floor_usd``) is a
    sizing-time ENTRY convention and deliberately does NOT re-apply to a
    budget resize of an already-admitted intent.
    """
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
    if fractional and _cap_affordable_qty is None:
        # FAIL CLOSED: the delegate owner is unavailable (pinned
        # renquant-execution predates the order-cash-cap ownership move).
        # Degrade to the legacy whole-share cap instead of crashing the
        # commit path; the warning makes the degradation observable.
        log.warning(
            "EXECMATH-CASHCAP-FALLBACK: renquant_execution.order_math "
            "unavailable (pinned checkout predates execution#25); "
            "fractional cash cap for %s fails closed to whole-share "
            "truncation",
            order.get("ticker"),
        )
        fractional = False
    if fractional:
        affordable = _cap_affordable_qty(price, cash, fractional=True)
        if affordable <= 0.0:
            return None, "cash_budget_exhausted"
    else:
        if _cap_affordable_qty is not None:
            affordable = _cap_affordable_qty(price, cash)
        else:
            # Inline legacy fallback — the ONE sanctioned int() truncation
            # (flag-off arm; see scripts/check_commit_path_no_int_truncation).
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
