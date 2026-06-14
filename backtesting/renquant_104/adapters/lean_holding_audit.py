"""LEAN holding audit-field stamping — lean.py decomposition slice 5.

EXTRACTED 2026-06-14 from adapters/lean.py (eng plan S2 item 5, god-file
decomposition). Pure functions that pull the decision-trace fields out of an
order (top-level, else from its score_snapshot / decision_inputs payloads) and
stamp them onto the live holding so LEAN's audit rows carry the same
attribution sim/live emit. No LeanAdapter state. Re-exported from lean for
back-compat.
"""
from __future__ import annotations

from typing import Any


def _order_payload(order: dict, key: str) -> Any:
    value = order.get(key)
    if value is not None:
        return value
    for field in ("score_snapshot", "decision_inputs"):
        payload = order.get(field)
        if isinstance(payload, dict) and payload.get(key) is not None:
            return payload.get(key)
    return None


def _stamp_holding_audit_fields(holding: Any, order: dict) -> None:
    if holding is None or not isinstance(order, dict):
        return
    for key in (
        "model_type",
        "sector",
        "blocked_by",
        "expected_return",
        "expected_return_horizon_days",
        "mu",
        "mu_horizon_days",
        "sigma",
        "panel_score",
        "rank_score",
        "kelly_target_pct",
    ):
        value = _order_payload(order, key)
        if value is not None:
            setattr(holding, key, value)
