"""Execution math — runner.py decomposition slice 5 (order_emit, cash/exec).

EXTRACTED 2026-06-13 from adapters/runner.py (eng plan S2 item 5). Pure
functions for the cash/execution arithmetic around order submission:
cash-cap a buy, same-bar sell credit, normalize broker status, summarize
a broker execution attempt, project holdings after orders, and snapshot
post-execution. No broker calls of their own (broker_order_execution
takes the already-returned result dict). Moved verbatim; re-exported
from runner for back-compat.

TIME-BOUNDED MIGRATION EXCEPTION (Codex review, renquant-orchestrator PR
#444): this module is umbrella-resident legacy, not the target architecture.
The owning repo for execution math is ``renquant-execution``; the removal
plan is the adapter-migration program (moving RunnerAdapter order math,
including this module, into that repo). Until that migration lands, changes
here must carry this same label and must not add umbrella-owned capability
beyond closing a specific, named contract gap.

Accordingly, cash-cap sizing math is NOT implemented here: it is owned by
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


# ── Buy-sizing cash (fix/size-on-settled-cash, 2026-08-30) ───────────────
#
# Closes ONE named contract gap (per the migration exception above): the
# strategy config declared ``execution.buying_power_mode`` for sim/live
# parity, the sim read it, and the live path never did — ``broker.get_cash()``
# was hard-wired to ``non_marginable_buying_power``. The vocabulary and the
# resolver are OWNED by the broker layer (``live/broker.py``); this helper
# only reads the key, asks the broker for its snapshot and logs the numbers.

_UNSETTLED_SPENDABLE_MODES = frozenset({
    "non_marginable_buying_power",
    "buying_power",
})


def unsettled_proceeds_spendable(mode: Any) -> bool:
    """Whether same-bar (T+1-unsettled) sell proceeds may fund buys.

    True ONLY for the modes that count unsettled proceeds by definition.
    ``settled_cash``, ``None`` and anything unrecognised → False: the
    conservative default, because crediting proceeds the account has not
    received is exactly the margin exposure this fix removes.
    """
    return str(mode or "").strip().lower() in _UNSETTLED_SPENDABLE_MODES


def resolve_buy_sizing_cash(broker: Any, config: dict | None) -> dict[str, Any]:
    """Read ``execution.buying_power_mode``, snapshot the broker, log both numbers.

    Returns the ``live.broker.resolve_sizing_cash`` dict plus
    ``configured_mode`` (the raw config value, ``None`` when absent) and
    ``broker_api`` (which broker call produced it). Raises on an
    unrecognised mode or a failed broker read — the caller's existing
    fail-SAFE (``cash = 0.0``) handles both.

    Brokers without ``get_buying_power_snapshot`` (test doubles, an older
    external broker) are read through ``get_cash()`` and that number is
    treated as SETTLED cash — never as buying power.
    """
    from live.broker import (  # noqa: PLC0415 — repo root is on sys.path in every runner entrypoint
        normalize_buying_power_mode,
        resolve_sizing_cash,
    )

    exec_cfg = (config or {}).get("execution") or {}
    configured = exec_cfg.get("buying_power_mode") if isinstance(exec_cfg, dict) else None
    mode = normalize_buying_power_mode(configured)

    snapshot_fn = getattr(broker, "get_buying_power_snapshot", None)
    if callable(snapshot_fn):
        snapshot = dict(snapshot_fn(mode))
        broker_api = "get_buying_power_snapshot"
    else:
        snapshot = resolve_sizing_cash(mode, settled_cash=broker.get_cash())
        broker_api = "get_cash (no snapshot API; treated as settled cash)"
    snapshot["configured_mode"] = configured
    snapshot["broker_api"] = broker_api

    def _money(value: Any) -> str:
        return "None" if value is None else f"{float(value):.2f}"

    log.info(
        "runner: buy-sizing cash=%s nmbp=%s buying_power=%s mode=%s -> "
        "sizing_cash=$%.2f (source=%s; execution.buying_power_mode=%r; via %s)",
        _money(snapshot.get("settled_cash")),
        _money(snapshot.get("non_marginable_buying_power")),
        _money(snapshot.get("buying_power")),
        snapshot.get("mode"),
        float(snapshot.get("sizing_cash") or 0.0),
        snapshot.get("sizing_source"),
        configured,
        broker_api,
    )
    if snapshot.get("sizing_reason"):
        log.warning(
            "runner: BUY budget is $0 this bar — %s (mode=%s cash=%s nmbp=%s). "
            "No new buys; exits are unaffected.",
            snapshot["sizing_reason"], snapshot.get("mode"),
            _money(snapshot.get("settled_cash")),
            _money(snapshot.get("non_marginable_buying_power")),
        )
    return snapshot
