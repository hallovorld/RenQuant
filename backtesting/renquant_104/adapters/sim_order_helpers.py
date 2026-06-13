"""Sim order/buying-power helpers — sim.py decomposition slice 3.

EXTRACTED 2026-06-13 from adapters/sim.py (eng plan S2 item 5). The
buying-power mode normalization (settled vs non-marginable, with alias
table), order-payload field access, and holding-audit-field stamping.
No SimAdapter state. Re-exported from sim for back-compat.
"""
from __future__ import annotations

from typing import Any


_BUYING_POWER_SETTLED = "settled_cash"
_BUYING_POWER_NMBP = "non_marginable_buying_power"
_BUYING_POWER_ALIASES = {
    _BUYING_POWER_SETTLED: _BUYING_POWER_SETTLED,
    "settled": _BUYING_POWER_SETTLED,
    "cash": _BUYING_POWER_SETTLED,
    _BUYING_POWER_NMBP: _BUYING_POWER_NMBP,
    "cash_plus_unsettled": _BUYING_POWER_NMBP,
    "unsettled": _BUYING_POWER_NMBP,
}


def _normalize_buying_power_mode(raw: Any) -> str:
    mode = str(raw or _BUYING_POWER_NMBP).strip().lower()
    if mode not in _BUYING_POWER_ALIASES:
        raise ValueError(
            "execution.buying_power_mode must be one of "
            f"{sorted(_BUYING_POWER_ALIASES)}; got {raw!r}"
        )
    return _BUYING_POWER_ALIASES[mode]


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
