"""Shared trade-event row builders for sim/live/LEAN persistence."""
from __future__ import annotations

import datetime
from typing import Any

from kernel.pipeline.exit_params import apply_stop_loss_anchor_policy


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
        "expected_return_horizon_days": order.get("expected_return_horizon_days"),
        "mu_horizon_days": order.get("mu_horizon_days"),
        "confidence": order.get("confidence", confidence),
        "regime": order.get("regime", regime),
        "model_type": order.get("model_type"),
        "sector": order.get("sector"),
        "blocked_by": order.get("blocked_by"),
    }


def _score_field(order: dict[str, Any], snap: dict[str, Any], key: str) -> Any:
    value = order.get(key)
    if value is not None:
        return value
    return snap.get(key)


def _decision_field(inputs: dict[str, Any], key: str) -> Any:
    value = inputs.get(key)
    return value if value is not None else inputs.get(f"qp_{key}")


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
    snap = _score_snapshot(order, regime, confidence)
    inputs = _decision_inputs(
        order,
        invest=invest,
        default_acceptance_reason=default_acceptance_reason,
    )
    return {
        "ticker": order.get("ticker"),
        "action": "buy",
        "date": date,
        "shares": order.get("shares"),
        "price": order.get("price"),
        "invest": invest,
        "target_pct": order.get("target_pct"),
        "rank_score": _score_field(order, snap, "rank_score"),
        "conviction": order.get("conviction"),
        "sigma_mult": order.get("sigma_mult"),
        "mu": _score_field(order, snap, "mu"),
        "mu_horizon_days": _score_field(order, snap, "mu_horizon_days"),
        "sigma": _score_field(order, snap, "sigma"),
        "order_type": order.get("order_type"),
        "source": order.get("source"),
        "source_job": order.get("source_job"),
        "source_task": order.get("source_task"),
        "order_source": order.get("order_source"),
        "attribution_version": (
            order.get("attribution_version") or attribution_version
        ),
        "score_snapshot": snap,
        "decision_inputs": inputs,
        "panel_score": _score_field(order, snap, "panel_score"),
        "rs_score": _score_field(order, snap, "rs_score"),
        "kelly_target_pct": _score_field(order, snap, "kelly_target_pct"),
        "expected_return": _score_field(order, snap, "expected_return"),
        "expected_return_horizon_days": _score_field(
            order, snap, "expected_return_horizon_days",
        ),
        "confidence": confidence,
        "regime": regime,
        "model_type": _score_field(order, snap, "model_type"),
        "sector": _score_field(order, snap, "sector"),
        "blocked_by": _score_field(order, snap, "blocked_by"),
        "qp_delta_w": _decision_field(inputs, "delta_w"),
        "qp_target_w": _decision_field(inputs, "target_w"),
        "qp_status": _decision_field(inputs, "solver_status"),
    }


def _fill_payload_from_source(payload: dict[str, Any], source_obj: Any) -> None:
    if source_obj is None:
        return
    for key in (
        "rank_score", "panel_score", "rs_score", "mu", "mu_horizon_days",
        "sigma", "expected_return", "expected_return_horizon_days",
        "kelly_target_pct", "model_type", "sector", "blocked_by",
    ):
        if payload.get(key) is None:
            value = getattr(source_obj, key, None)
            if value is not None:
                payload[key] = value


def build_short_open_trade_event(
    *,
    ticker: str,
    sig: Any,
    price: float,
    shares: float,
    proceeds: float,
    today: Any,
    regime: str | None,
    confidence: float | None,
    source_obj: Any = None,
    attribution_version: str = "short_open_decision_v1",
) -> dict[str, Any]:
    """Normalize an executed short-open event for DB/audit writers."""
    raw_inputs = getattr(sig, "decision_inputs", None)
    payload = dict(raw_inputs) if isinstance(raw_inputs, dict) else {}
    _fill_payload_from_source(payload, source_obj)
    payload.update({
        "ticker": ticker,
        "action": "short_open",
        "shares": shares,
        "price": price,
        "proceeds": proceeds,
        "side": "sell_to_open",
        "exit_reason": getattr(sig, "exit_type", None) or "short_open",
        "signal_reason": getattr(sig, "reason", None),
        "source_job": getattr(sig, "source_job", None) or payload.get("source_job"),
        "source_task": getattr(sig, "source_task", None) or payload.get("source_task"),
    })
    payload.setdefault(
        "order_source",
        getattr(sig, "order_source", None)
        or (
            f"{payload.get('source_job')}.{payload.get('source_task')}"
            if payload.get("source_job") and payload.get("source_task")
            else None
        ),
    )
    payload.setdefault(
        "acceptance_reason",
        getattr(sig, "exit_type", None)
        or payload.get("order_source")
        or "short_open",
    )
    snap = _score_snapshot(payload, regime, confidence)
    source_job = str(payload.get("source_job") or "JointPortfolioQPJob")
    source_task = str(payload.get("source_task") or "short_open")
    order_source = str(payload.get("order_source") or f"{source_job}.{source_task}")
    exit_type = getattr(sig, "exit_type", None) or "short_open"
    return {
        "ticker": ticker,
        "action": "short_open",
        "date": today,
        "shares": shares,
        "price": price,
        "invest": -abs(proceeds),
        "target_pct": payload.get("target_w"),
        "exit_reason": exit_type,
        "pnl_pct": 0.0,
        "hold_days": 0,
        "tax": 0.0,
        "rank_score": _score_field(payload, snap, "rank_score"),
        "conviction": payload.get("conviction"),
        "sigma_mult": payload.get("sigma_mult"),
        "mu": _score_field(payload, snap, "mu"),
        "mu_horizon_days": _score_field(payload, snap, "mu_horizon_days"),
        "sigma": _score_field(payload, snap, "sigma"),
        "panel_score": _score_field(payload, snap, "panel_score"),
        "rs_score": _score_field(payload, snap, "rs_score"),
        "expected_return": _score_field(payload, snap, "expected_return"),
        "expected_return_horizon_days": _score_field(
            payload, snap, "expected_return_horizon_days",
        ),
        "kelly_target_pct": _score_field(payload, snap, "kelly_target_pct"),
        "model_type": _score_field(payload, snap, "model_type"),
        "sector": _score_field(payload, snap, "sector"),
        "blocked_by": _score_field(payload, snap, "blocked_by"),
        "order_type": f"SHORT_OPEN_{exit_type}",
        "source": "qp",
        "source_job": source_job,
        "source_task": source_task,
        "order_source": order_source,
        "attribution_version": attribution_version,
        "score_snapshot": snap,
        "decision_inputs": payload,
        "qp_delta_w": _decision_field(payload, "delta_w"),
        "qp_target_w": _decision_field(payload, "target_w"),
        "qp_status": _decision_field(payload, "solver_status"),
        "confidence": confidence,
        "regime": regime,
    }


def _date_obj(value: Any) -> datetime.date | None:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if hasattr(value, "date"):
        try:
            return value.date()
        except Exception:
            return None
    return None


def build_sell_trade_event(
    *,
    ticker: str,
    sig: Any,
    holding: Any,
    price: float,
    today: Any,
    regime: str | None,
    confidence: float | None,
    regime_params: dict,
    config: dict | None = None,
    shares: float | None = None,
    gross_pnl: float | None = None,
    proceeds_basis: float | None = None,
    tax: float | None = None,
    net_pnl_after_tax: float | None = None,
    pnl_pct: float | None = None,
    hold_days: int | None = None,
    attribution_version: str = "exit_decision_v1",
) -> dict[str, Any]:
    """Normalize an executed SELL event for DB/audit writers."""
    entry_p = float(getattr(holding, "entry_price", 0.0) or 0.0)
    entry_date = _date_obj(getattr(holding, "entry_date", None))
    today_date = _date_obj(today)
    if hold_days is None:
        hold_days = (
            (today_date - entry_date).days
            if holding and today_date is not None and entry_date is not None else 0
        )
    if shares is None:
        raw_qty = getattr(sig, "shares_sold", None)
        if raw_qty is None:
            raw_qty = getattr(sig, "quantity", None)
        if raw_qty is None:
            raw_qty = getattr(holding, "shares", None)
        shares = _none_or_float(raw_qty)
    if pnl_pct is None:
        pnl_pct = (price - entry_p) / entry_p if entry_p > 0 else 0.0
    if (
        gross_pnl is None
        and shares is not None and shares > 0 and entry_p > 0 and price > 0
    ):
        gross_pnl = (price - entry_p) * shares
    if (
        proceeds_basis is None
        and shares is not None and shares > 0 and entry_p > 0
    ):
        proceeds_basis = entry_p * shares
    if tax is None and gross_pnl is not None:
        tax_cfg = (regime_params or {}).get("tax", {})
        st_rate = float(tax_cfg.get("short_term_rate", 0.50))
        lt_rate = float(tax_cfg.get("long_term_rate", 0.32))
        lt_days = int(tax_cfg.get("long_term_threshold_days", 365))
        rate = lt_rate if hold_days >= lt_days else st_rate
        tax = max(gross_pnl, 0.0) * rate
    if net_pnl_after_tax is None and gross_pnl is not None and tax is not None:
        net_pnl_after_tax = gross_pnl - tax
    tax_cash_mode = _tax_cash_debit_mode(config or {})
    tax_cash_debited = _tax_cash_debit_amount(config or {}, tax or 0.0)
    tax_lot_method = _tax_lot_method(config or {})
    exit_type = getattr(sig, "exit_type", "") or ""
    reason = getattr(sig, "reason", None)
    source_job = str(getattr(sig, "source_job", None) or "TickerSellJob")
    source_task = str(getattr(sig, "source_task", None) or exit_type or "sell")
    order_source = str(
        getattr(sig, "order_source", None) or f"{source_job}.{source_task}"
    )
    exit_p = _applied_exit_params(
        sig=sig,
        holding=holding,
        regime=regime,
        regime_params=regime_params,
        config=config or {},
    )
    sig_inputs = getattr(sig, "decision_inputs", None) or {}
    return {
        "ticker": ticker,
        "action": "sell",
        "date": today,
        "shares": shares,
        "price": price,
        "gross_pnl": gross_pnl,
        "proceeds_basis": proceeds_basis,
        "tax": tax,
        "net_pnl_after_tax": net_pnl_after_tax,
        "tax_cash_debited": tax_cash_debited,
        "tax_cash_debit_mode": tax_cash_mode,
        "tax_lot_method": tax_lot_method,
        "exit_reason": exit_type,
        "pnl_pct": pnl_pct,
        "hold_days": hold_days,
        "rank_score": getattr(holding, "rank_score", None),
        "panel_score": getattr(holding, "panel_score", None),
        "kelly_target_pct": getattr(holding, "kelly_target_pct", None),
        "expected_return": getattr(holding, "expected_return", None),
        "expected_return_horizon_days": getattr(
            holding, "expected_return_horizon_days", None,
        ),
        "mu": getattr(holding, "mu", None),
        "mu_horizon_days": getattr(holding, "mu_horizon_days", None),
        "sigma": getattr(holding, "sigma", None),
        "order_type": f"SELL_{exit_type}" if exit_type else "SELL",
        "source": str(getattr(sig, "source", None) or "ExitPipeline"),
        "source_job": source_job,
        "source_task": source_task,
        "order_source": order_source,
        "attribution_version": attribution_version,
        "confidence": confidence,
        "regime": regime,
        "model_type": getattr(holding, "model_type", None),
        "sector": getattr(holding, "sector", None),
        "blocked_by": getattr(sig, "blocked_by", None) or getattr(
            holding, "blocked_by", None,
        ),
        "qp_delta_w": sig_inputs.get("delta_w"),
        "qp_target_w": sig_inputs.get("target_w"),
        "qp_status": sig_inputs.get("solver_status"),
        "score_snapshot": {
            "rank_score": getattr(holding, "rank_score", None),
            "panel_score": getattr(holding, "panel_score", None),
            "expected_return": getattr(holding, "expected_return", None),
            "expected_return_horizon_days": getattr(
                holding, "expected_return_horizon_days", None,
            ),
            "mu": getattr(holding, "mu", None),
            "mu_horizon_days": getattr(holding, "mu_horizon_days", None),
            "sigma": getattr(holding, "sigma", None),
            "kelly_target_pct": getattr(holding, "kelly_target_pct", None),
            "confidence": confidence,
            "regime": regime,
            "model_type": getattr(holding, "model_type", None),
            "sector": getattr(holding, "sector", None),
        },
        "decision_inputs": {
            "acceptance_reason": exit_type or reason,
            "exit_reason": exit_type,
            "signal_reason": reason,
            "quantity": getattr(sig, "quantity", None),
            "shares": shares,
            "gross_pnl": gross_pnl,
            "tax": tax,
            "tax_cash_debited": tax_cash_debited,
            "tax_cash_debit_mode": tax_cash_mode,
            "tax_lot_method": tax_lot_method,
            "net_pnl_after_tax": net_pnl_after_tax,
            "hold_days": hold_days,
            "pnl_pct": pnl_pct,
            "stop_loss_pct": exit_p.get("stop_loss_pct"),
            "stop_loss_anchor_policy": exit_p.get("stop_loss_anchor_policy"),
            "stop_loss_anchor_regime": exit_p.get("stop_loss_anchor_regime"),
            "stop_loss_current_regime": exit_p.get("stop_loss_current_regime"),
            "stop_loss_current_pct": exit_p.get("stop_loss_current_pct"),
            "stop_loss_entry_regime": exit_p.get("stop_loss_entry_regime"),
            "stop_loss_entry_pct": exit_p.get("stop_loss_entry_pct"),
            "stop_n_sigma": exit_p.get("stop_n_sigma"),
            "take_profit_pct": exit_p.get("take_profit_pct"),
            "stop_decay_days": exit_p.get("stop_decay_days"),
            "stop_decay_floor": exit_p.get("stop_decay_floor"),
            "max_single_day_loss_pct": exit_p.get("max_single_day_loss_pct"),
            "sdl_n_sigma": exit_p.get("sdl_n_sigma"),
            "sdl_skip_if_unrealized_above": exit_p.get(
                "sdl_skip_if_unrealized_above"
            ),
            "trailing_stop_trigger_pct": exit_p.get(
                "trailing_stop_trigger_pct"
            ),
            "trailing_stop_trail_pct": exit_p.get(
                "trailing_stop_trail_pct"
            ),
            "atr_n_multiplier": exit_p.get("atr_n_multiplier"),
            "max_hold_days": exit_p.get("max_hold_days"),
            "max_hold_anchor_regime": exit_p.get("max_hold_anchor_regime"),
            **sig_inputs,
        },
    }


def _applied_exit_params(
    *,
    sig: Any,
    holding: Any,
    regime: str | None,
    regime_params: dict,
    config: dict,
) -> dict[str, Any]:
    applied = getattr(sig, "exit_params", None)
    if isinstance(applied, dict) and applied:
        return dict(applied)
    exit_p = dict(regime_params or {})
    entry_regime = getattr(holding, "entry_regime", None)
    entry_regime_p = (
        (config.get("regime_params", {}) or {}).get(entry_regime, {})
        if entry_regime is not None else {}
    )
    if isinstance(entry_regime_p, dict) and "max_hold_days" in entry_regime_p:
        exit_p["max_hold_days"] = entry_regime_p["max_hold_days"]
        exit_p["max_hold_anchor_regime"] = entry_regime
    apply_stop_loss_anchor_policy(
        exit_p,
        config=config,
        current_regime=regime,
        entry_regime=entry_regime,
        entry_regime_params=entry_regime_p,
    )
    return exit_p


def _tax_cash_debit_mode(config: dict) -> str:
    tax_cfg = (config.get("tax") or {}) if isinstance(config, dict) else {}
    raw = str(tax_cfg.get("cash_debit_mode", "event_level") or "event_level").lower()
    aliases = {
        "event": "event_level",
        "immediate": "event_level",
        "stress": "event_level",
        "none": "reporting_only",
        "off": "reporting_only",
        "reporting": "reporting_only",
        "reporting-only": "reporting_only",
        "reporting_only": "reporting_only",
        "annual_net": "reporting_only",
        "event_level": "event_level",
        "event_cash_debit": "event_level",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"event_level", "reporting_only"}:
        raise ValueError(
            "invalid tax.cash_debit_mode "
            f"{raw!r}; expected event_level or reporting_only"
        )
    return mode


def _tax_cash_debit_amount(config: dict, tax: float) -> float:
    try:
        tax_f = float(tax)
    except (TypeError, ValueError):
        return 0.0
    if tax_f <= 0.0:
        return 0.0
    if _tax_cash_debit_mode(config) == "reporting_only":
        return 0.0
    return tax_f


def _tax_lot_method(config: dict) -> str:
    return str(
        ((config.get("rotation", {}) or {}).get("joint_actions", {}) or {})
        .get("qp_tax_lot_method", (config.get("tax", {}) or {}).get("lot_method", "fifo"))
    ).lower()


__all__ = [
    "build_buy_trade_event",
    "build_sell_trade_event",
    "build_short_open_trade_event",
]
