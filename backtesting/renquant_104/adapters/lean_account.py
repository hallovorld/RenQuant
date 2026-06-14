"""LEAN account / buying-power snapshot helpers — lean.py decomposition slice 2.

EXTRACTED 2026-06-13 from adapters/lean.py (eng plan S2 item 5, god-file
decomposition; lean.py counterpart to runner_execmath's snapshot helpers).
Pure functions that read a LEAN QCAlgorithm's Portfolio/cash state into the
uniform buying-power + post-execution snapshot dicts the adapter persists.
No LeanAdapter state. Re-exported from lean for back-compat.
"""
from __future__ import annotations

import math
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


def _finite_attr_float(obj: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(obj, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return None


def _lean_buying_power_snapshot(algo: Any, config: dict) -> dict[str, Any]:
    portfolio = getattr(algo, "Portfolio", None)
    settled_cash = _finite_attr_float(portfolio, "Cash") or 0.0
    exec_cfg = (config or {}).get("execution", {}) or {}
    mode = _normalize_buying_power_mode(
        exec_cfg.get("buying_power_mode", _BUYING_POWER_NMBP)
    )
    if mode == _BUYING_POWER_SETTLED:
        return {
            "cash": settled_cash,
            "settled_cash": settled_cash,
            "pending_settle_cash": 0.0,
            "buying_power_mode": mode,
            "buying_power_source": "portfolio_cash",
        }

    nmbp = _finite_attr_float(
        portfolio,
        "NonMarginableBuyingPower",
        "non_marginable_buying_power",
        "NonMarginableBuyingPowerAmount",
    )
    if nmbp is not None and nmbp >= 0.0:
        pending = max(0.0, nmbp - settled_cash)
        return {
            "cash": nmbp,
            "settled_cash": settled_cash,
            "pending_settle_cash": pending,
            "buying_power_mode": mode,
            "buying_power_source": "portfolio_non_marginable_buying_power",
        }

    pending = _finite_attr_float(algo, "_pending_settle_cash") or 0.0
    pending = max(0.0, pending)
    return {
        "cash": settled_cash + pending,
        "settled_cash": settled_cash,
        "pending_settle_cash": pending,
        "buying_power_mode": mode,
        "buying_power_source": (
            "algo_pending_settle_cash" if pending > 0.0
            else "portfolio_cash_fallback"
        ),
    }


def _lean_post_execution_snapshot(
    algo: Any,
    config: dict,
    ctx: Any,
) -> dict[str, Any]:
    portfolio = getattr(algo, "Portfolio", None)
    pv = _finite_attr_float(portfolio, "TotalPortfolioValue")
    if pv is None:
        try:
            pv = float(getattr(ctx, "portfolio_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            pv = None
    bp = _lean_buying_power_snapshot(algo, config)
    if _finite_attr_float(portfolio, "Cash") is None:
        try:
            fallback_cash = float(getattr(ctx, "cash", 0.0) or 0.0)
        except (TypeError, ValueError):
            fallback_cash = 0.0
        if math.isfinite(fallback_cash):
            bp = {
                **bp,
                "cash": fallback_cash,
                "settled_cash": fallback_cash,
                "pending_settle_cash": 0.0,
            }
    holdings = getattr(algo, "_holdings", {}) or {}
    n_holdings = sum(
        1 for hs in holdings.values()
        if _finite_attr_float(hs, "shares") and _finite_attr_float(hs, "shares") > 0
    )
    return {
        "portfolio_value": pv,
        "cash": float(bp["cash"]),
        "settled_cash": float(bp["settled_cash"]),
        "pending_settle_cash": float(bp["pending_settle_cash"]),
        "n_holdings": n_holdings,
    }
