"""Exit-parameter helpers shared by inference and adapter audit logs."""
from __future__ import annotations

import math
from typing import Any


def _positive_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0 else None


def _regime_allowed(regime: str | None, allowed: Any) -> bool:
    if not allowed:
        return True
    if isinstance(allowed, str):
        allowed_set = {allowed}
    else:
        allowed_set = {str(x) for x in allowed}
    return regime in allowed_set


def apply_stop_loss_anchor_policy(
    exit_params: dict[str, Any],
    *,
    config: dict[str, Any],
    current_regime: str | None,
    entry_regime: str | None,
    entry_regime_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Optionally keep cumulative stops no tighter than the entry-regime stop.

    Default ``current_regime`` preserves legacy behavior. ``max_entry_current``
    is an explicit A/B hook for the BULL_CALM thesis: if the entry thesis was a
    calmer momentum regime, do not silently tighten the cumulative stop after a
    later regime re-label unless the current regime's stop is already wider.
    """
    policy = ((config.get("risk") or {}).get("stop_loss_anchor_policy") or {})
    mode = str(policy.get("mode", "current_regime") or "current_regime")
    mode = mode.lower()
    if mode in {"current", "current_regime", "disabled", "off"}:
        return exit_params
    if mode != "max_entry_current":
        raise ValueError(f"unknown risk.stop_loss_anchor_policy.mode={mode!r}")

    if not entry_regime:
        return exit_params
    if not _regime_allowed(entry_regime, policy.get("entry_regimes")):
        return exit_params
    if not _regime_allowed(current_regime, policy.get("current_regimes")):
        return exit_params

    current_stop = _positive_float(exit_params.get("stop_loss_pct"))
    entry_stop = _positive_float((entry_regime_params or {}).get("stop_loss_pct"))
    if current_stop is None:
        raise ValueError("stop_loss_anchor_policy requires current stop_loss_pct")
    if entry_stop is None:
        raise ValueError(
            f"stop_loss_anchor_policy requires stop_loss_pct for entry regime {entry_regime}"
        )

    anchored_stop = max(current_stop, entry_stop)
    anchor_regime = entry_regime if entry_stop >= current_stop else current_regime
    exit_params["stop_loss_pct"] = anchored_stop
    exit_params["stop_loss_anchor_policy"] = mode
    exit_params["stop_loss_anchor_regime"] = anchor_regime
    exit_params["stop_loss_current_regime"] = current_regime
    exit_params["stop_loss_current_pct"] = current_stop
    exit_params["stop_loss_entry_regime"] = entry_regime
    exit_params["stop_loss_entry_pct"] = entry_stop
    return exit_params
