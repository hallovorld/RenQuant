"""Live execution-trace builders — runner.py decomposition slice 4 (reporting).

EXTRACTED 2026-06-13 from adapters/runner.py (eng plan S2 item 5:
state_store / broker_sync / order_emit / reporting). Pure functions that
turn broker attempts (filled / pending / skipped / rejected) into
decision-trace audit events. No I/O, no broker calls. Moved verbatim;
re-exported from runner for back-compat.
"""
from __future__ import annotations

from typing import Any

from kernel.decision_trace import selected_buy_tickers
from kernel.trade_events import build_buy_trade_event


def live_trace_selection_maps(
    trade_events: list[dict[str, Any]] | None,
    pending_orders: list[dict[str, Any]] | None,
    blocked_map: dict[str, str] | None = None,
) -> tuple[set[str], dict[str, str], set[str]]:
    """Trace filled buys as selected and pending submissions as blocked."""
    pending_tickers = {
        str(o.get("ticker"))
        for o in (pending_orders or [])
        if isinstance(o, dict) and o.get("ticker")
    }
    out_blocked = dict(blocked_map or {})
    for ticker in pending_tickers:
        out_blocked.setdefault(ticker, "broker_pending_submitted")
    return selected_buy_tickers(trade_events), out_blocked, pending_tickers


def live_execution_attempt_events(ctx: Any) -> list[dict[str, Any]]:
    """Persist non-filled broker attempts as audit events.

    Filled orders remain `buy`/`sell` trade rows. Pending/skipped/rejected
    attempts are decision-tree evidence, not fills, so they use distinct
    actions and do not count as selected buys or realized exits.
    """
    events: list[dict[str, Any]] = []
    today = getattr(ctx, "today", None)
    regime = getattr(ctx, "regime", None)
    confidence = getattr(ctx, "confidence", None)
    for order in getattr(ctx, "orders_pending", []) or []:
        if isinstance(order, dict):
            events.append(_buy_attempt_event(
                order, "buy_pending", ctx, "broker_pending_submitted",
            ))
    for order in getattr(ctx, "orders_skipped", []) or []:
        if isinstance(order, dict):
            reason = f"broker_skip:{order.get('skip_reason', 'skipped')}"
            events.append(_buy_attempt_event(order, "buy_skipped", ctx, reason))
    for item in getattr(ctx, "exits_pending", []) or []:
        if isinstance(item, dict):
            events.append(_sell_attempt_event(item, "sell_pending", ctx))
    for item in getattr(ctx, "exits_failed", []) or []:
        if isinstance(item, dict):
            events.append(_sell_attempt_event(item, "sell_rejected", ctx))
    for event in events:
        event.setdefault("date", today)
        event.setdefault("regime", regime)
        event.setdefault("confidence", confidence)
    return events


def _buy_attempt_event(
    order: dict[str, Any],
    action: str,
    ctx: Any,
    blocked_by: str,
) -> dict[str, Any]:
    row = build_buy_trade_event(
        order,
        date=getattr(ctx, "today", None),
        default_regime=getattr(ctx, "regime", None),
        default_confidence=getattr(ctx, "confidence", None),
        default_acceptance_reason=blocked_by,
    )
    row["action"] = action
    row["blocked_by"] = blocked_by
    row["exit_reason"] = order.get("skip_reason") or order.get("status") or blocked_by
    row["status"] = order.get("status")
    inputs = dict(row.get("decision_inputs") or {})
    inputs.update({
        "attempt_status": action,
        "skip_reason": order.get("skip_reason"),
        "status": order.get("status"),
        "order_id": order.get("order_id"),
        "blocked_by": blocked_by,
    })
    inputs.setdefault("acceptance_reason", blocked_by)
    row["decision_inputs"] = inputs
    snap = dict(row.get("score_snapshot") or {})
    snap.update({
        "attempt_status": action,
        "blocked_by": blocked_by,
        "status": order.get("status"),
    })
    row["score_snapshot"] = snap
    return row


def _sell_attempt_event(
    item: dict[str, Any],
    action: str,
    ctx: Any,
) -> dict[str, Any]:
    ticker = item.get("ticker")
    hs = (getattr(ctx, "holdings", None) or {}).get(ticker)
    blocked_by = item.get("error") or item.get("status") or action
    price = (getattr(ctx, "prices", None) or {}).get(ticker)
    source_job = str(item.get("source_job") or getattr(item.get("sig", None), "source_job", None) or "LiveBroker")
    source_task = str(item.get("source_task") or getattr(item.get("sig", None), "source_task", None) or action)
    order_source = str(item.get("order_source") or f"{source_job}.{source_task}")
    snap = {
        "rank_score": getattr(hs, "rank_score", None),
        "panel_score": getattr(hs, "panel_score", None),
        "expected_return": getattr(hs, "expected_return", None),
        "expected_return_horizon_days": getattr(hs, "expected_return_horizon_days", None),
        "mu": getattr(hs, "mu", None),
        "mu_horizon_days": getattr(hs, "mu_horizon_days", None),
        "sigma": getattr(hs, "sigma", None),
        "confidence": getattr(ctx, "confidence", None),
        "regime": getattr(ctx, "regime", None),
        "model_type": getattr(hs, "model_type", None),
        "sector": getattr(hs, "sector", None),
        "blocked_by": blocked_by,
        "attempt_status": action,
        "status": item.get("status"),
    }
    inputs = {
        "acceptance_reason": blocked_by,
        "attempt_status": action,
        "exit_type": item.get("exit_type"),
        "signal_reason": item.get("reason"),
        "shares": item.get("qty"),
        "price": price,
        "is_partial": item.get("is_partial"),
        "status": item.get("status"),
        "order_id": item.get("order_id"),
        "error": item.get("error"),
        "source_job": source_job,
        "source_task": source_task,
        "order_source": order_source,
    }
    return {
        "ticker": ticker,
        "action": action,
        "date": getattr(ctx, "today", None),
        "shares": item.get("qty"),
        "price": price,
        "exit_reason": item.get("exit_type"),
        "blocked_by": blocked_by,
        "order_type": f"SELL_ATTEMPT_{item.get('exit_type') or action}",
        "source": "LiveBroker",
        "source_job": source_job,
        "source_task": source_task,
        "order_source": order_source,
        "attribution_version": "live_execution_attempt_v1",
        "score_snapshot": snap,
        "decision_inputs": inputs,
        "confidence": getattr(ctx, "confidence", None),
        "regime": getattr(ctx, "regime", None),
        "model_type": getattr(hs, "model_type", None),
        "sector": getattr(hs, "sector", None),
    }
