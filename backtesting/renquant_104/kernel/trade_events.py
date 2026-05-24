"""Shared trade-event row builders for sim/live/LEAN persistence."""
from __future__ import annotations

from typing import Any


def _none_or_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _computed_invest(order: dict[str, Any]) -> float | None:
    invest = _none_or_float(order.get("invest"))
    if invest is not None:
        return invest
    shares = _none_or_float(order.get("shares"))
    price = _none_or_float(order.get("price"))
    if shares is None or price is None:
        return None
    return shares * price


def _score_snapshot(order: dict[str, Any], regime: str | None,
                    confidence: float | None) -> dict[str, Any]:
    snap = order.get("score_snapshot")
    if isinstance(snap, dict) and snap:
        return snap
    return {
        "rank_score": order.get("rank_score"),
        "panel_score": order.get("panel_score"),
        "rs_score": order.get("rs_score"),
        "mu": order.get("mu"),
        "sigma": order.get("sigma"),
        "kelly_target_pct": order.get("kelly_target_pct"),
        "expected_return": order.get("expected_return"),
        "confidence": order.get("confidence", confidence),
        "regime": order.get("regime", regime),
    }


def _decision_inputs(
    order: dict[str, Any],
    *,
    invest: float | None,
    default_acceptance_reason: str,
) -> dict[str, Any]:
    raw = order.get("decision_inputs")
    if isinstance(raw, dict) and raw:
        return raw
    return {
        "acceptance_reason": (
            order.get("detail")
            or order.get("order_source")
            or order.get("order_type")
            or default_acceptance_reason
        ),
        "target_pct": order.get("target_pct"),
        "shares": order.get("shares"),
        "price": order.get("price"),
        "invest": invest,
        "order_source": order.get("order_source"),
        "source_job": order.get("source_job"),
        "source_task": order.get("source_task"),
    }


def build_buy_trade_event(
    order: dict[str, Any],
    *,
    date: Any,
    default_regime: str | None = None,
    default_confidence: float | None = None,
    attribution_version: str | None = None,
    default_acceptance_reason: str = "buy",
) -> dict[str, Any]:
    """Normalize an executed BUY event for sim/live/LEAN DB writers."""
    regime = order.get("regime", default_regime)
    confidence = order.get("confidence", default_confidence)
    invest = _computed_invest(order)
    return {
        "ticker": order.get("ticker"),
        "action": "buy",
        "date": date,
        "shares": order.get("shares"),
        "price": order.get("price"),
        "invest": invest,
        "target_pct": order.get("target_pct"),
        "rank_score": order.get("rank_score"),
        "conviction": order.get("conviction"),
        "sigma_mult": order.get("sigma_mult"),
        "mu": order.get("mu"),
        "sigma": order.get("sigma"),
        "order_type": order.get("order_type"),
        "source": order.get("source"),
        "source_job": order.get("source_job"),
        "source_task": order.get("source_task"),
        "order_source": order.get("order_source"),
        "attribution_version": (
            order.get("attribution_version") or attribution_version
        ),
        "score_snapshot": _score_snapshot(order, regime, confidence),
        "decision_inputs": _decision_inputs(
            order,
            invest=invest,
            default_acceptance_reason=default_acceptance_reason,
        ),
        "panel_score": order.get("panel_score"),
        "rs_score": order.get("rs_score"),
        "kelly_target_pct": order.get("kelly_target_pct"),
        "expected_return": order.get("expected_return"),
        "confidence": confidence,
        "regime": regime,
    }


__all__ = ["build_buy_trade_event"]
