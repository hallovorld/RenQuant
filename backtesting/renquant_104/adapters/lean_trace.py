"""LEAN execution-trace builders — lean.py decomposition slice 1.

EXTRACTED 2026-06-13 from adapters/lean.py (eng plan S2 item 5,
god-file decomposition; lean.py is 1370 lines). Pure functions that turn
LEAN order attempts (filled / pending / skipped / rejected) into
decision-trace audit events — the LEAN counterpart to runner_trace.
No LeanAdapter state. Re-exported from lean for back-compat.
"""
from __future__ import annotations

from typing import Any

from kernel.pipeline.context import InferenceContext
from kernel.trade_events import build_buy_trade_event


def _lean_attempt_action(side: str, status: Any) -> str:
    status_l = str(status or "").lower()
    terminal_bad = ("reject", "cancel", "invalid", "error", "missing")
    suffix = "rejected" if any(t in status_l for t in terminal_bad) else "pending"
    return f"{side}_{suffix}"


def _lean_buy_attempt_event(
    order: dict[str, Any],
    *,
    ctx: InferenceContext,
    status: str,
    blocked_by: str,
    action: str | None = None,
) -> dict[str, Any]:
    action = action or _lean_attempt_action("buy", status)
    row = build_buy_trade_event(
        order,
        date=ctx.today,
        default_regime=ctx.regime,
        default_confidence=ctx.confidence,
        attribution_version="lean_execution_attempt_v1",
        default_acceptance_reason=blocked_by,
    )
    row["action"] = action
    row["blocked_by"] = blocked_by
    row["exit_reason"] = status or blocked_by
    inputs = dict(row.get("decision_inputs") or {})
    inputs.update({
        "attempt_status": action,
        "status": status,
        "blocked_by": blocked_by,
    })
    inputs.setdefault("acceptance_reason", blocked_by)
    row["decision_inputs"] = inputs
    snap = dict(row.get("score_snapshot") or {})
    snap.update({
        "attempt_status": action,
        "status": status,
        "blocked_by": blocked_by,
    })
    row["score_snapshot"] = snap
    return row


def _lean_sell_attempt_event(
    *,
    ticker: str,
    sig: Any,
    holding: Any,
    ctx: InferenceContext,
    requested_shares: float,
    price: float,
    status: str,
    blocked_by: str | None = None,
    action: str | None = None,
) -> dict[str, Any]:
    action = action or _lean_attempt_action("sell", status)
    blocked = blocked_by or status or action
    source_job = str(getattr(sig, "source_job", None) or "LeanOrderTicket")
    source_task = str(getattr(sig, "source_task", None) or action)
    order_source = str(getattr(sig, "order_source", None) or f"{source_job}.{source_task}")
    snap = {
        "rank_score": getattr(holding, "rank_score", None),
        "panel_score": getattr(holding, "panel_score", None),
        "expected_return": getattr(holding, "expected_return", None),
        "expected_return_horizon_days": getattr(holding, "expected_return_horizon_days", None),
        "mu": getattr(holding, "mu", None),
        "mu_horizon_days": getattr(holding, "mu_horizon_days", None),
        "sigma": getattr(holding, "sigma", None),
        "confidence": getattr(ctx, "confidence", None),
        "regime": getattr(ctx, "regime", None),
        "model_type": getattr(holding, "model_type", None),
        "sector": getattr(holding, "sector", None),
        "blocked_by": blocked,
        "attempt_status": action,
        "status": status,
    }
    inputs = {
        "acceptance_reason": blocked,
        "attempt_status": action,
        "exit_type": getattr(sig, "exit_type", None),
        "signal_reason": getattr(sig, "reason", None),
        "shares": requested_shares,
        "price": price,
        "status": status,
        "blocked_by": blocked,
        "source_job": source_job,
        "source_task": source_task,
        "order_source": order_source,
    }
    return {
        "ticker": ticker,
        "action": action,
        "date": ctx.today,
        "shares": requested_shares,
        "price": price,
        "exit_reason": getattr(sig, "exit_type", None),
        "blocked_by": blocked,
        "order_type": f"SELL_ATTEMPT_{getattr(sig, 'exit_type', None) or action}",
        "source": "LeanOrderTicket",
        "source_job": source_job,
        "source_task": source_task,
        "order_source": order_source,
        "attribution_version": "lean_execution_attempt_v1",
        "score_snapshot": snap,
        "decision_inputs": inputs,
        "confidence": getattr(ctx, "confidence", None),
        "regime": getattr(ctx, "regime", None),
        "model_type": getattr(holding, "model_type", None),
        "sector": getattr(holding, "sector", None),
    }
