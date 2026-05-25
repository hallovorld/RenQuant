"""Shared guardrails for model-driven soft exits.

These helpers are intentionally small and pure-ish: they are used by both the
per-ticker legacy panel exit and the pipeline-level cross-sectional panel exit.
Hard path exits (stop loss, trailing stop, single-day loss, max hold) do not
call these helpers.
"""
from __future__ import annotations

import datetime
import math
from typing import Any

from kernel.exits import nyse_trading_days_between


def holding_days(today: Any, holding: Any) -> int | None:
    entry_date = getattr(holding, "entry_date", None)
    if not isinstance(today, datetime.date) or not isinstance(entry_date, datetime.date):
        return None
    return max(0, (today - entry_date).days)


def trading_holding_days(today: Any, holding: Any) -> int | None:
    entry_date = getattr(holding, "entry_date", None)
    if not isinstance(today, datetime.date) or not isinstance(entry_date, datetime.date):
        return None
    return nyse_trading_days_between(entry_date, today)


def _configured_min_days(panel_cfg: dict[str, Any], regime: str | None) -> int:
    by_regime = panel_cfg.get("min_holding_days_by_regime") or {}
    if regime is not None and isinstance(by_regime, dict) and regime in by_regime:
        try:
            return max(0, int(by_regime[regime]))
        except (TypeError, ValueError):
            return 0
    try:
        return max(0, int(panel_cfg.get("min_holding_days", 0)))
    except (TypeError, ValueError):
        return 0


def configured_soft_exit_min_days(panel_cfg: dict[str, Any], regime: str | None) -> int:
    """Public wrapper for model-driven soft-exit thesis-age guards."""
    return _configured_min_days(panel_cfg, regime)


def soft_exit_thesis_regime(holding: Any, current_regime: str | None) -> str | None:
    """Use the entry thesis regime for soft-exit horizon gates when known."""
    entry_regime = getattr(holding, "entry_regime", None)
    if entry_regime:
        return str(entry_regime)
    return current_regime


def soft_exit_horizon_suppression(
    *,
    panel_cfg: dict[str, Any],
    regime: str | None,
    today: Any,
    holding: Any,
) -> tuple[bool, str]:
    """Return True when a model-driven soft exit should wait for thesis age."""
    min_days = _configured_min_days(panel_cfg, regime)
    if min_days <= 0:
        return False, ""
    days = trading_holding_days(today, holding)
    if days is None:
        return False, ""
    if days < min_days:
        return True, (
            f"horizon_min_days trading_days={days} < {min_days} "
            f"regime={regime}"
        )
    return False, ""


def resolve_current_price(source: Any, holding: Any, ticker: str | None = None) -> float | None:
    price = None
    if ticker and hasattr(source, "prices"):
        try:
            price = (getattr(source, "prices") or {}).get(ticker)
        except AttributeError:
            price = None
    if price is None:
        price = getattr(source, "today_close", None)
    if price is None:
        price = getattr(source, "price", None)
    if price is None:
        price = getattr(holding, "current_price", None)
    try:
        price_f = float(price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price_f) or price_f <= 0.0:
        return None
    return price_f


def lt_gate_suppression(
    *,
    config: dict[str, Any],
    today: Any,
    holding: Any,
    current_price: float | None,
) -> tuple[bool, str]:
    """Root-aware LT capital-gain gate for soft exits.

    Production stores ``lt_hold_gate_days`` and ``lt_hold_min_gain`` at the
    config root, while older task code looked under ``risk`` and silently fell
    back to 30 days. Prefer the root single source, then risk-level legacy.
    """
    risk_cfg = config.get("risk") or {}
    tax_cfg = config.get("tax") or {}
    try:
        lt_gate = int(config.get("lt_hold_gate_days", risk_cfg.get("lt_hold_gate_days", 30)))
        lt_thresh = int(
            risk_cfg.get(
                "lt_hold_threshold_days",
                tax_cfg.get("long_term_threshold_days", 365),
            )
        )
        lt_min_gain = float(config.get("lt_hold_min_gain", risk_cfg.get("lt_hold_min_gain", 0.10)))
    except (TypeError, ValueError):
        return False, ""
    if lt_gate <= 0:
        return False, ""
    days = holding_days(today, holding)
    entry_price = getattr(holding, "entry_price", None)
    try:
        entry_f = float(entry_price)
    except (TypeError, ValueError):
        return False, ""
    if days is None or current_price is None or not math.isfinite(entry_f) or entry_f <= 0:
        return False, ""
    unrealized_gain = (current_price - entry_f) / entry_f
    if lt_gate <= days < lt_thresh and unrealized_gain >= lt_min_gain:
        return (
            True,
            f"lt_tax_gate days={days} gain={unrealized_gain:+.1%} "
            f"window=[{lt_gate},{lt_thresh})",
        )
    return False, ""


def tax_adjusted_soft_exit_suppression(
    *,
    panel_cfg: dict[str, Any],
    tax_cfg: dict[str, Any],
    today: Any,
    holding: Any,
    current_price: float | None,
    mu: Any,
) -> tuple[bool, str]:
    """Return True when expected avoided loss does not cover tax drag.

    Units are fraction-of-position. This follows the existing project
    convention in ``kernel.rotation.tax_drag``: unrealized gain times tax rate
    is the immediate tax drag of selling now. The gate only applies to model
    driven soft exits and only when the position is at an unrealized gain.
    """
    cfg = panel_cfg.get("tax_adjusted_soft_exit") or {}
    if not bool(cfg.get("enabled", False)):
        return False, ""
    if current_price is None:
        return False, ""
    try:
        mu_f = float(mu)
        entry_f = float(getattr(holding, "entry_price", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False, ""
    if not (math.isfinite(mu_f) and math.isfinite(entry_f)) or entry_f <= 0:
        return False, ""

    unrealized_gain = (current_price - entry_f) / entry_f
    min_gain = float(cfg.get("min_unrealized_gain", 0.0) or 0.0)
    if unrealized_gain <= min_gain:
        return False, ""

    days = holding_days(today, holding)
    try:
        lt_threshold = int(tax_cfg.get("long_term_threshold_days", 365))
        st_rate = float(tax_cfg.get("short_term_rate", 0.50))
        lt_rate = float(tax_cfg.get("long_term_rate", 0.32))
    except (TypeError, ValueError):
        return False, ""

    short_only = bool(cfg.get("short_term_only", True))
    if short_only and days is not None and days >= lt_threshold:
        return False, ""

    rate = lt_rate if days is not None and days >= lt_threshold else st_rate
    drag_fraction = float(cfg.get("tax_drag_fraction", 1.0) or 0.0)
    transaction_cost = float(cfg.get("transaction_cost_pct", 0.0) or 0.0)
    min_edge = float(cfg.get("min_exit_edge", 0.0) or 0.0)
    tax_drag = max(0.0, unrealized_gain) * max(0.0, rate) * max(0.0, drag_fraction)
    hurdle = tax_drag + max(0.0, transaction_cost) + max(0.0, min_edge)
    expected_loss_avoided = max(0.0, -mu_f)
    if expected_loss_avoided < hurdle:
        return (
            True,
            f"tax_adjusted_exit expected_loss={expected_loss_avoided:.4f} "
            f"< hurdle={hurdle:.4f} tax_drag={tax_drag:.4f}",
        )
    return False, ""
