"""Order-attribution contract for buy emitters.

Every buy order emitted into ``ctx.orders`` must answer:

  * which Job/Task owned the decision?
  * which score state did it see at emit time?
  * why did it pass the final hurdle?

This is intentionally lightweight and dict-based because adapters already
consume order dicts. The invariant is enforced at emission time by
``validate_order_attribution`` and by source-level tests.
"""
from __future__ import annotations

import math
from typing import Any


ATTRIBUTION_VERSION = "order_attribution_v1"


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pick(order: dict, obj: Any, key: str) -> Any:
    if key in order:
        return order.get(key)
    return getattr(obj, key, None) if obj is not None else None


def score_snapshot(order: dict, *, source_obj: Any = None, ctx: Any = None) -> dict[str, Any]:
    """Capture the model/risk score state visible when an order is emitted."""
    return {
        "rank_score": _finite_or_none(_pick(order, source_obj, "rank_score")),
        "panel_score": _finite_or_none(_pick(order, source_obj, "panel_score")),
        "rs_score": _finite_or_none(_pick(order, source_obj, "rs_score")),
        "mu": _finite_or_none(_pick(order, source_obj, "mu")),
        "sigma": _finite_or_none(_pick(order, source_obj, "sigma")),
        "kelly_target_pct": _finite_or_none(
            _pick(order, source_obj, "kelly_target_pct")
        ),
        "expected_return": _finite_or_none(getattr(source_obj, "expected_return", None)),
        "confidence": _finite_or_none(order.get("confidence", getattr(ctx, "confidence", None))),
        "regime": order.get("regime", getattr(ctx, "regime", None)),
    }


def stamp_order_attribution(
    order: dict,
    *,
    ctx: Any,
    source_job: str,
    source_task: str,
    acceptance_reason: str,
    source_obj: Any = None,
    decision_inputs: dict[str, Any] | None = None,
) -> dict:
    """Stamp required attribution fields and validate the order contract."""
    if not acceptance_reason:
        raise ValueError("order attribution requires non-empty acceptance_reason")
    order_type = str(order.get("order_type") or "")
    if not order_type:
        raise ValueError("order attribution requires order_type")
    order_source = f"{source_job}.{source_task}"
    merged_inputs = dict(decision_inputs or {})
    merged_inputs.setdefault("acceptance_reason", acceptance_reason)
    merged_inputs.setdefault("order_type", order_type)
    merged_inputs.setdefault("source_job", source_job)
    merged_inputs.setdefault("source_task", source_task)
    order.update({
        "attribution_version": ATTRIBUTION_VERSION,
        "source_job": source_job,
        "source_task": source_task,
        "order_source": order_source,
        "source": order.get("source") or order_source,
        "score_snapshot": score_snapshot(order, source_obj=source_obj, ctx=ctx),
        "decision_inputs": merged_inputs,
    })
    validate_order_attribution(order)
    return order


def validate_order_attribution(order: dict) -> None:
    required = [
        "ticker",
        "order_type",
        "attribution_version",
        "source_job",
        "source_task",
        "order_source",
        "score_snapshot",
        "decision_inputs",
    ]
    missing = [key for key in required if key not in order]
    if missing:
        raise ValueError(f"order attribution missing fields: {missing}")
    if order["attribution_version"] != ATTRIBUTION_VERSION:
        raise ValueError("unknown order attribution version")
    if not isinstance(order["score_snapshot"], dict):
        raise ValueError("order score_snapshot must be a dict")
    if not isinstance(order["decision_inputs"], dict):
        raise ValueError("order decision_inputs must be a dict")
    if not order["decision_inputs"].get("acceptance_reason"):
        raise ValueError("order decision_inputs.acceptance_reason is required")


__all__ = [
    "ATTRIBUTION_VERSION",
    "score_snapshot",
    "stamp_order_attribution",
    "validate_order_attribution",
]
