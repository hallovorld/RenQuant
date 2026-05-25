#!/usr/bin/env python
"""Trade-ledger utilities for renquant_104 simulations.

The sim engine already returns ``SimResult.trade_log`` in memory. This module
turns that volatile list into durable audit artifacts: raw trade events,
lot-matched round trips, and a compact forensic report.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


REGIME_THESES = {
    "BULL_CALM": (
        "Trend/momentum continuation: cross-sectional rank should select "
        "relative winners while market volatility is benign."
    ),
    "BULL_VOLATILE": (
        "Risk-managed upside: score must compensate for higher market "
        "volatility and faster drawdown risk."
    ),
    "CHOPPY": (
        "Relative-strength/divergence: entry needs stock-specific strength "
        "because broad beta is less reliable."
    ),
    "BEAR": (
        "Capital preservation: offensive buys are suspect unless the ticker "
        "is explicitly defensive."
    ),
}

EXIT_PARAM_FIELDS = (
    "exit_stop_loss_pct",
    "exit_stop_n_sigma",
    "exit_take_profit_pct",
    "exit_stop_decay_days",
    "exit_stop_decay_floor",
    "exit_max_single_day_loss_pct",
    "exit_sdl_n_sigma",
    "exit_sdl_skip_if_unrealized_above",
    "exit_trailing_stop_trigger_pct",
    "exit_trailing_stop_trail_pct",
    "exit_atr_n_multiplier",
    "exit_max_hold_days",
)

ENTRY_ATTRIBUTION_FIELDS = (
    "entry_source",
    "entry_source_job",
    "entry_source_task",
    "entry_order_source",
    "entry_order_type",
    "entry_attribution_version",
    "entry_score_snapshot",
    "entry_decision_inputs",
)

EXIT_ATTRIBUTION_FIELDS = (
    "exit_source",
    "exit_source_job",
    "exit_source_task",
    "exit_order_source",
    "exit_order_type",
    "exit_attribution_version",
    "exit_score_snapshot",
    "exit_decision_inputs",
)


def annual_net_tax_summary(
    round_trips: pd.DataFrame,
    tax_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate calendar-year capital-gains tax after annual netting.

    The simulator debits tax on every winning sale immediately and gives no
    same-year credit for losses. That is intentionally conservative for cash
    stress, but it is not how Schedule D economics are summarized: gains and
    losses are netted by short/long bucket, then cross-netted. This helper
    gives reports a second, IRS-aligned stress lens without changing sim cash.

    Simplifications: wash-sale basis deferrals and loss carryforwards are not
    modeled here; this is a same-calendar-year closed-trade estimate.
    """
    if round_trips is None or round_trips.empty:
        return {"total_estimated_tax": 0.0, "years": []}
    cfg = tax_config or {}
    st_rate = float(cfg.get("short_term_rate", 0.50))
    lt_rate = float(cfg.get("long_term_rate", 0.32))
    lt_days = int(cfg.get("long_term_threshold_days", 365))
    df = round_trips.copy()
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "closed"]
    if df.empty or "exit_date" not in df.columns or "gross_pnl" not in df.columns:
        return {"total_estimated_tax": 0.0, "years": []}
    df["gross_pnl"] = pd.to_numeric(df["gross_pnl"], errors="coerce")
    if "hold_days" in df.columns:
        df["hold_days"] = pd.to_numeric(df["hold_days"], errors="coerce")
    else:
        df["hold_days"] = 0
    df["exit_year"] = pd.to_datetime(df["exit_date"], errors="coerce").dt.year
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["gross_pnl", "exit_year"]
    )
    rows = []
    total_tax = 0.0
    for year, g in df.groupby("exit_year"):
        st = g[g["hold_days"].fillna(0) < lt_days]["gross_pnl"].sum()
        lt = g[g["hold_days"].fillna(0) >= lt_days]["gross_pnl"].sum()
        tax = _tax_on_netted_capital_gains(float(st), float(lt), st_rate, lt_rate)
        total_tax += tax
        rows.append({
            "year": int(year),
            "short_term_net": float(st),
            "long_term_net": float(lt),
            "estimated_tax": float(tax),
        })
    return {"total_estimated_tax": float(total_tax), "years": rows}


def _tax_on_netted_capital_gains(
    st_net: float,
    lt_net: float,
    st_rate: float,
    lt_rate: float,
) -> float:
    """Tax positive net capital gains after same-bucket and cross-netting."""
    if st_net >= 0 and lt_net >= 0:
        return st_net * st_rate + lt_net * lt_rate
    if st_net <= 0 and lt_net <= 0:
        return 0.0
    if st_net > 0 and lt_net < 0:
        return max(0.0, st_net + lt_net) * st_rate
    if lt_net > 0 and st_net < 0:
        return max(0.0, lt_net + st_net) * lt_rate
    return 0.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_date(value: Any) -> str:
    if value is None:
        return ""
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:  # noqa: BLE001
        return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _copy_fields(event: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: event.get(field) for field in fields}


def _event_score(event: dict[str, Any], key: str) -> Any:
    value = event.get(key)
    if value is not None:
        return value
    snap = _dict_payload(event.get("score_snapshot"))
    if isinstance(snap, dict):
        return snap.get(key)
    return None


def _dict_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _event_field(event: dict[str, Any], *keys: str) -> Any:
    """Read audit fields from top-level, score snapshot, or decision inputs."""
    containers = (
        event,
        _dict_payload(event.get("score_snapshot")) or {},
        _dict_payload(event.get("decision_inputs")) or {},
    )
    for key in keys:
        for src in containers:
            value = src.get(key)
            if value is not None:
                return value
    return None


def _event_float(event: dict[str, Any], *keys: str) -> float | None:
    value = _event_field(event, *keys)
    if value is None:
        return None
    out = _as_float(value, default=float("nan"))
    return out if math.isfinite(out) else None


def _empty_fields(fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: None for field in fields}


def _entry_lot_from_event(
    *,
    event_id: int,
    event: dict[str, Any],
    ticker: str,
    shares: float,
    side: str,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "ticker": ticker,
        "side": side,
        "entry_date": _as_date(event.get("date")),
        "entry_price": _as_float(event.get("price")),
        "remaining_shares": shares,
        "entry_invest": _as_float(event.get("invest")),
        "entry_regime": event.get("regime"),
        "entry_rank_score": _event_score(event, "rank_score"),
        "entry_rs_score": _event_score(event, "rs_score"),
        "entry_panel_score": _event_score(event, "panel_score"),
        "entry_mu": _event_score(event, "mu"),
        "entry_mu_horizon_days": _event_score(event, "mu_horizon_days"),
        "entry_sigma": _event_score(event, "sigma"),
        "entry_sigma_mult": event.get("sigma_mult"),
        "entry_kelly_target_pct": _event_score(event, "kelly_target_pct"),
        "entry_expected_return": _event_score(event, "expected_return"),
        "entry_expected_return_horizon_days": _event_score(
            event, "expected_return_horizon_days"
        ),
        "entry_model_type": _event_field(event, "model_type"),
        "entry_sector": _event_field(event, "sector"),
        "entry_blocked_by": _event_field(event, "blocked_by"),
        "entry_qp_delta_w": _event_float(event, "qp_delta_w", "delta_w"),
        "entry_qp_target_w": _event_float(event, "qp_target_w", "target_w"),
        "entry_qp_status": _event_field(event, "qp_status", "solver_status"),
        "entry_qp_mu_used": _event_float(event, "qp_mu_used"),
        "entry_qp_sigma_used": _event_float(event, "qp_sigma_used"),
        "entry_qp_mu_source": _event_field(event, "qp_mu_source"),
        "entry_alpha_to_mu_applied": _event_field(event, "alpha_to_mu_applied"),
        "entry_source": event.get("source"),
        "entry_source_job": event.get("source_job"),
        "entry_source_task": event.get("source_task"),
        "entry_order_source": event.get("order_source"),
        "entry_order_type": event.get("order_type"),
        "entry_attribution_version": event.get("attribution_version"),
        "entry_score_snapshot": _json_safe(event.get("score_snapshot")),
        "entry_decision_inputs": _json_safe(event.get("decision_inputs")),
    }


def trade_log_frame(trade_log: list[dict]) -> pd.DataFrame:
    """Return a stable raw-event DataFrame sorted by event date."""
    rows = []
    for idx, row in enumerate(trade_log or []):
        r = dict(row)
        r["event_id"] = idx
        r["date"] = _as_date(r.get("date"))
        rows.append(r)
    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df = df.sort_values(["date", "event_id"]).reset_index(drop=True)
    return df


def _equity_regime_map(result: Any) -> dict[str, str]:
    equity = getattr(result, "equity_df", None)
    if equity is None or getattr(equity, "empty", True) or "regime" not in equity.columns:
        return {}
    out: dict[str, str] = {}
    for idx, regime in equity["regime"].items():
        if regime is not None and regime == regime:
            out[_as_date(idx)] = str(regime)
    return out


def _enrich_trade_log_from_result(result: Any) -> list[dict]:
    """Fill audit-only fields that older order emitters may omit."""
    regime_by_date = _equity_regime_map(result)
    enriched: list[dict] = []
    for event in list(getattr(result, "trade_log", []) or []):
        row = dict(event)
        action = str(row.get("action") or "").lower()
        if action in {"buy", "sell"} and not row.get("regime"):
            row["regime"] = regime_by_date.get(_as_date(row.get("date")))
        enriched.append(row)
    return enriched


def round_trips_from_trade_log(
    trade_log: list[dict],
    *,
    end_prices: dict[str, float] | None = None,
    lot_method: str = "fifo",
) -> pd.DataFrame:
    """Match long buys to long sells with the simulator's tax-lot method.

    The simulator can top up and partially trim positions. Per-trade sell rows
    contain event-level realized P&L, but root-cause analysis needs entry-side
    fields (regime, rank_score, mu/sigma) joined to each realized exit.

    The matching method must mirror the simulator's configured disposal rule.
    Production 104 uses HIFO to minimize realized gain on partial trims; a
    forensic FIFO replay can then disagree with the sell event's tax basis and
    fabricate rows where allocated tax is larger than the row's gross profit.
    Event-level tax is allocated only across profitable matched lots in
    proportion to their positive gross P&L. This preserves the simulator's
    event-level tax estimate while avoiding losing lots carrying positive tax.
    """
    method = (lot_method or "fifo").lower()
    if method not in {"fifo", "hifo", "avg"}:
        method = "fifo"
    lots: dict[str, list[dict]] = defaultdict(list)
    short_lots: dict[str, list[dict]] = defaultdict(list)
    rows: list[dict] = []

    for event_id, event in enumerate(trade_log or []):
        action = str(event.get("action") or "").lower()
        ticker = str(event.get("ticker") or "")
        if not ticker:
            continue
        if action == "buy":
            shares = _as_float(event.get("shares"))
            if shares <= 0:
                continue
            lots[ticker].append(_entry_lot_from_event(
                event_id=event_id,
                event=event,
                ticker=ticker,
                shares=shares,
                side="long",
            ))
            continue

        if action == "short_open":
            shares = _as_float(event.get("shares"))
            if shares <= 0:
                continue
            short_lots[ticker].append(_entry_lot_from_event(
                event_id=event_id,
                event=event,
                ticker=ticker,
                shares=shares,
                side="short",
            ))
            continue

        if action == "short_cover":
            cover_shares = _as_float(event.get("shares"))
            if cover_shares <= 0:
                continue
            cover_price = _as_float(event.get("price"))
            event_tax = _as_float(event.get("tax"))
            event_tax_cash = _as_float(event.get("tax_cash_debited"), default=event_tax)
            matched_rows: list[dict[str, Any]] = []
            lot_takes = _lot_takes(short_lots[ticker], cover_shares, method)
            for lot_idx, take in lot_takes:
                lot = short_lots[ticker][lot_idx]
                if take <= 1e-9:
                    continue
                entry_price = _as_float(lot.get("entry_price"))
                gross_pnl = (entry_price - cover_price) * take
                entry_value = entry_price * take
                pnl_pct = gross_pnl / entry_value if entry_value > 0 else 0.0
                entry_date = lot.get("entry_date", "")
                exit_date = _as_date(event.get("date"))
                matched_rows.append({
                    "status": "closed",
                    "direction": "short",
                    "ticker": ticker,
                    "entry_event_id": lot.get("event_id"),
                    "exit_event_id": event_id,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "hold_days": (
                        pd.Timestamp(exit_date) - pd.Timestamp(entry_date)
                    ).days if entry_date and exit_date else event.get("hold_days"),
                    "shares": take,
                    "entry_price": entry_price,
                    "exit_price": cover_price,
                    "gross_pnl": gross_pnl,
                    "tax": 0.0,
                    "tax_cash_debited": 0.0,
                    "tax_cash_debit_mode": event.get("tax_cash_debit_mode", "event_level"),
                    "net_pnl_after_tax": gross_pnl,
                    "tax_allocation_method": "pending",
                    "pnl_pct": pnl_pct,
                    "sim_sell_pnl_pct": event.get("pnl_pct"),
                    "exit_reason": event.get("exit_reason", "short_cover"),
                    "partial_exit": bool(event.get("partial", False)),
                    "exit_regime": event.get("regime"),
                    "exit_confidence": event.get("confidence"),
                    "exit_signal_reason": event.get("exit_signal_reason"),
                    "exit_rank_score": _event_score(event, "rank_score"),
                    "exit_rs_score": _event_score(event, "rs_score"),
                    "exit_panel_score": _event_score(event, "panel_score"),
                    "exit_mu": _event_score(event, "mu"),
                    "exit_mu_horizon_days": _event_score(event, "mu_horizon_days"),
                    "exit_sigma": _event_score(event, "sigma"),
                    "exit_kelly_target_pct": _event_score(event, "kelly_target_pct"),
                    "exit_expected_return": _event_score(event, "expected_return"),
                    "exit_expected_return_horizon_days": _event_score(
                        event, "expected_return_horizon_days",
                    ),
                    "exit_model_type": _event_field(event, "model_type"),
                    "exit_sector": _event_field(event, "sector"),
                    "exit_blocked_by": _event_field(event, "blocked_by"),
                    "exit_qp_delta_w": _event_float(event, "qp_delta_w", "delta_w"),
                    "exit_qp_target_w": _event_float(event, "qp_target_w", "target_w"),
                    "exit_qp_status": _event_field(event, "qp_status", "solver_status"),
                    "exit_qp_mu_used": _event_float(event, "qp_mu_used"),
                    "exit_qp_sigma_used": _event_float(event, "qp_sigma_used"),
                    "exit_qp_mu_source": _event_field(event, "qp_mu_source"),
                    "exit_alpha_to_mu_applied": _event_field(
                        event, "alpha_to_mu_applied",
                    ),
                    **_copy_fields(event, EXIT_PARAM_FIELDS),
                    "exit_source": event.get("source"),
                    "exit_source_job": event.get("source_job"),
                    "exit_source_task": event.get("source_task"),
                    "exit_order_source": event.get("order_source"),
                    "exit_order_type": event.get("order_type"),
                    "exit_attribution_version": event.get("attribution_version"),
                    "exit_score_snapshot": _json_safe(event.get("score_snapshot")),
                    "exit_decision_inputs": _json_safe(event.get("decision_inputs")),
                    "entry_regime": lot.get("entry_regime"),
                    "entry_rank_score": lot.get("entry_rank_score"),
                    "entry_rs_score": lot.get("entry_rs_score"),
                    "entry_panel_score": lot.get("entry_panel_score"),
                    "entry_mu": lot.get("entry_mu"),
                    "entry_mu_horizon_days": lot.get("entry_mu_horizon_days"),
                    "entry_sigma": lot.get("entry_sigma"),
                    "entry_sigma_mult": lot.get("entry_sigma_mult"),
                    "entry_kelly_target_pct": lot.get("entry_kelly_target_pct"),
                    "entry_expected_return": lot.get("entry_expected_return"),
                    "entry_expected_return_horizon_days": lot.get(
                        "entry_expected_return_horizon_days"
                    ),
                    "entry_model_type": lot.get("entry_model_type"),
                    "entry_sector": lot.get("entry_sector"),
                    "entry_blocked_by": lot.get("entry_blocked_by"),
                    "entry_qp_delta_w": lot.get("entry_qp_delta_w"),
                    "entry_qp_target_w": lot.get("entry_qp_target_w"),
                    "entry_qp_status": lot.get("entry_qp_status"),
                    "entry_qp_mu_used": lot.get("entry_qp_mu_used"),
                    "entry_qp_sigma_used": lot.get("entry_qp_sigma_used"),
                    "entry_qp_mu_source": lot.get("entry_qp_mu_source"),
                    "entry_alpha_to_mu_applied": lot.get(
                        "entry_alpha_to_mu_applied"
                    ),
                    **{field: lot.get(field) for field in ENTRY_ATTRIBUTION_FIELDS},
                })

            for lot_idx, take in sorted(lot_takes, key=lambda x: x[0], reverse=True):
                if lot_idx >= len(short_lots[ticker]):
                    continue
                short_lots[ticker][lot_idx]["remaining_shares"] -= take
                if short_lots[ticker][lot_idx]["remaining_shares"] <= 1e-9:
                    short_lots[ticker].pop(lot_idx)

            remaining = cover_shares - sum(take for _, take in lot_takes)
            if matched_rows:
                positive_gross = sum(
                    max(0.0, _as_float(r.get("gross_pnl")))
                    for r in matched_rows
                )
                if event_tax > 0 and positive_gross > 0:
                    for r in matched_rows:
                        gross = _as_float(r.get("gross_pnl"))
                        if gross <= 0:
                            r["tax_allocation_method"] = "loss_no_tax"
                            continue
                        tax_alloc = event_tax * (gross / positive_gross)
                        r["tax"] = tax_alloc
                        r["net_pnl_after_tax"] = gross - tax_alloc
                        r["tax_allocation_method"] = "positive_gross_prorata"
                        if event_tax_cash > 0:
                            r["tax_cash_debited"] = event_tax_cash * (
                                gross / positive_gross
                            )
                else:
                    for r in matched_rows:
                        gross = _as_float(r.get("gross_pnl"))
                        r["tax_allocation_method"] = (
                            "loss_no_tax" if gross <= 0 else "event_tax_zero"
                        )
                rows.extend(matched_rows)
            if remaining > 1e-9:
                rows.append({
                    "status": "unmatched_short_cover",
                    "direction": "short",
                    "ticker": ticker,
                    "entry_event_id": None,
                    "exit_event_id": event_id,
                    "entry_date": "",
                    "exit_date": _as_date(event.get("date")),
                    "hold_days": event.get("hold_days"),
                    "shares": remaining,
                    "entry_price": None,
                    "exit_price": cover_price,
                    "gross_pnl": None,
                    "tax": 0.0,
                    "tax_cash_debited": 0.0,
                    "tax_cash_debit_mode": event.get("tax_cash_debit_mode", "event_level"),
                    "net_pnl_after_tax": None,
                    "tax_allocation_method": "unmatched_short_cover_unallocated",
                    "pnl_pct": event.get("pnl_pct"),
                    "sim_sell_pnl_pct": event.get("pnl_pct"),
                    "exit_reason": event.get("exit_reason", "short_cover"),
                    "partial_exit": bool(event.get("partial", False)),
                    "exit_regime": event.get("regime"),
                    "exit_confidence": event.get("confidence"),
                    "exit_signal_reason": event.get("exit_signal_reason"),
                    **_empty_fields(EXIT_PARAM_FIELDS),
                    **_empty_fields(EXIT_ATTRIBUTION_FIELDS),
                    **_empty_fields(ENTRY_ATTRIBUTION_FIELDS),
                })
            continue

        if action != "sell":
            continue
        sell_shares = _as_float(event.get("shares"))
        if sell_shares <= 0:
            continue
        exit_price = _as_float(event.get("price"))
        event_tax = _as_float(event.get("tax"))
        event_tax_cash = _as_float(event.get("tax_cash_debited"), default=event_tax)
        matched_rows: list[dict[str, Any]] = []
        lot_takes = _lot_takes(lots[ticker], sell_shares, method)
        for lot_idx, take in lot_takes:
            lot = lots[ticker][lot_idx]
            if take <= 1e-9:
                continue
            entry_price = _as_float(lot.get("entry_price"))
            gross_pnl = (exit_price - entry_price) * take
            entry_value = entry_price * take
            pnl_pct = gross_pnl / entry_value if entry_value > 0 else 0.0
            entry_date = lot.get("entry_date", "")
            exit_date = _as_date(event.get("date"))
            matched_rows.append({
                "status": "closed",
                "ticker": ticker,
                "entry_event_id": lot.get("event_id"),
                "exit_event_id": event_id,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "hold_days": (
                    pd.Timestamp(exit_date) - pd.Timestamp(entry_date)
                ).days if entry_date and exit_date else event.get("hold_days"),
                "shares": take,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "tax": 0.0,
                "tax_cash_debited": 0.0,
                "tax_cash_debit_mode": event.get("tax_cash_debit_mode", "event_level"),
                "net_pnl_after_tax": gross_pnl,
                "tax_allocation_method": "pending",
                "pnl_pct": pnl_pct,
                "sim_sell_pnl_pct": event.get("pnl_pct"),
                "exit_reason": event.get("exit_reason"),
                "partial_exit": bool(event.get("partial", False)),
                "exit_regime": event.get("regime"),
                "exit_confidence": event.get("confidence"),
                "exit_signal_reason": event.get("exit_signal_reason"),
                "exit_rank_score": _event_score(event, "rank_score"),
                "exit_rs_score": _event_score(event, "rs_score"),
                "exit_panel_score": _event_score(event, "panel_score"),
                "exit_mu": _event_score(event, "mu"),
                "exit_mu_horizon_days": _event_score(event, "mu_horizon_days"),
                "exit_sigma": _event_score(event, "sigma"),
                "exit_kelly_target_pct": _event_score(event, "kelly_target_pct"),
                "exit_expected_return": _event_score(event, "expected_return"),
                "exit_expected_return_horizon_days": _event_score(
                    event, "expected_return_horizon_days",
                ),
                "exit_model_type": _event_field(event, "model_type"),
                "exit_sector": _event_field(event, "sector"),
                "exit_blocked_by": _event_field(event, "blocked_by"),
                "exit_qp_delta_w": _event_float(event, "qp_delta_w", "delta_w"),
                "exit_qp_target_w": _event_float(event, "qp_target_w", "target_w"),
                "exit_qp_status": _event_field(event, "qp_status", "solver_status"),
                "exit_qp_mu_used": _event_float(event, "qp_mu_used"),
                "exit_qp_sigma_used": _event_float(event, "qp_sigma_used"),
                "exit_qp_mu_source": _event_field(event, "qp_mu_source"),
                "exit_alpha_to_mu_applied": _event_field(
                    event, "alpha_to_mu_applied",
                ),
                **_copy_fields(event, EXIT_PARAM_FIELDS),
                "exit_source": event.get("source"),
                "exit_source_job": event.get("source_job"),
                "exit_source_task": event.get("source_task"),
                "exit_order_source": event.get("order_source"),
                "exit_order_type": event.get("order_type"),
                "exit_attribution_version": event.get("attribution_version"),
                "exit_score_snapshot": _json_safe(event.get("score_snapshot")),
                "exit_decision_inputs": _json_safe(event.get("decision_inputs")),
                "entry_regime": lot.get("entry_regime"),
                "entry_rank_score": lot.get("entry_rank_score"),
                "entry_rs_score": lot.get("entry_rs_score"),
                "entry_panel_score": lot.get("entry_panel_score"),
                "entry_mu": lot.get("entry_mu"),
                "entry_mu_horizon_days": lot.get("entry_mu_horizon_days"),
                "entry_sigma": lot.get("entry_sigma"),
                "entry_sigma_mult": lot.get("entry_sigma_mult"),
                "entry_kelly_target_pct": lot.get("entry_kelly_target_pct"),
                "entry_expected_return": lot.get("entry_expected_return"),
                "entry_expected_return_horizon_days": lot.get(
                    "entry_expected_return_horizon_days"
                ),
                "entry_model_type": lot.get("entry_model_type"),
                "entry_sector": lot.get("entry_sector"),
                "entry_blocked_by": lot.get("entry_blocked_by"),
                "entry_qp_delta_w": lot.get("entry_qp_delta_w"),
                "entry_qp_target_w": lot.get("entry_qp_target_w"),
                "entry_qp_status": lot.get("entry_qp_status"),
                "entry_qp_mu_used": lot.get("entry_qp_mu_used"),
                "entry_qp_sigma_used": lot.get("entry_qp_sigma_used"),
                "entry_qp_mu_source": lot.get("entry_qp_mu_source"),
                "entry_alpha_to_mu_applied": lot.get(
                    "entry_alpha_to_mu_applied"
                ),
                **{field: lot.get(field) for field in ENTRY_ATTRIBUTION_FIELDS},
            })

        for lot_idx, take in sorted(lot_takes, key=lambda x: x[0], reverse=True):
            if lot_idx >= len(lots[ticker]):
                continue
            lots[ticker][lot_idx]["remaining_shares"] -= take
            if lots[ticker][lot_idx]["remaining_shares"] <= 1e-9:
                lots[ticker].pop(lot_idx)

        remaining = sell_shares - sum(take for _, take in lot_takes)

        if matched_rows:
            positive_gross = sum(
                max(0.0, _as_float(r.get("gross_pnl")))
                for r in matched_rows
            )
            if event_tax > 0 and positive_gross > 0:
                for r in matched_rows:
                    gross = _as_float(r.get("gross_pnl"))
                    if gross <= 0:
                        r["tax_allocation_method"] = "loss_no_tax"
                        continue
                    tax_alloc = event_tax * (gross / positive_gross)
                    r["tax"] = tax_alloc
                    r["net_pnl_after_tax"] = gross - tax_alloc
                    r["tax_allocation_method"] = "positive_gross_prorata"
                    if event_tax_cash > 0:
                        r["tax_cash_debited"] = event_tax_cash * (gross / positive_gross)
            else:
                for r in matched_rows:
                    gross = _as_float(r.get("gross_pnl"))
                    r["tax_allocation_method"] = (
                        "loss_no_tax" if gross <= 0 else "event_tax_zero"
                    )
            rows.extend(matched_rows)

        if remaining > 1e-9:
            rows.append({
                "status": "unmatched_sell",
                "ticker": ticker,
                "entry_event_id": None,
                "exit_event_id": event_id,
                "entry_date": "",
                "exit_date": _as_date(event.get("date")),
                "hold_days": event.get("hold_days"),
                "shares": remaining,
                "entry_price": None,
                "exit_price": exit_price,
                "gross_pnl": None,
                "tax": 0.0,
                "tax_cash_debited": 0.0,
                "tax_cash_debit_mode": event.get("tax_cash_debit_mode", "event_level"),
                "net_pnl_after_tax": None,
                "tax_allocation_method": "unmatched_sell_unallocated",
                "pnl_pct": event.get("pnl_pct"),
                "sim_sell_pnl_pct": event.get("pnl_pct"),
                "exit_reason": event.get("exit_reason"),
                "partial_exit": bool(event.get("partial", False)),
                "exit_regime": event.get("regime"),
                "exit_confidence": event.get("confidence"),
                "exit_signal_reason": event.get("exit_signal_reason"),
                "exit_rank_score": _event_score(event, "rank_score"),
                "exit_rs_score": _event_score(event, "rs_score"),
                "exit_panel_score": _event_score(event, "panel_score"),
                "exit_mu": _event_score(event, "mu"),
                "exit_mu_horizon_days": _event_score(event, "mu_horizon_days"),
                "exit_sigma": _event_score(event, "sigma"),
                "exit_kelly_target_pct": _event_score(event, "kelly_target_pct"),
                "exit_expected_return": _event_score(event, "expected_return"),
                "exit_expected_return_horizon_days": _event_score(
                    event, "expected_return_horizon_days",
                ),
                "exit_model_type": _event_field(event, "model_type"),
                "exit_sector": _event_field(event, "sector"),
                "exit_blocked_by": _event_field(event, "blocked_by"),
                "exit_qp_delta_w": _event_float(event, "qp_delta_w", "delta_w"),
                "exit_qp_target_w": _event_float(event, "qp_target_w", "target_w"),
                "exit_qp_status": _event_field(event, "qp_status", "solver_status"),
                "exit_qp_mu_used": _event_float(event, "qp_mu_used"),
                "exit_qp_sigma_used": _event_float(event, "qp_sigma_used"),
                "exit_qp_mu_source": _event_field(event, "qp_mu_source"),
                "exit_alpha_to_mu_applied": _event_field(
                    event, "alpha_to_mu_applied",
                ),
                **_copy_fields(event, EXIT_PARAM_FIELDS),
                "exit_source": event.get("source"),
                "exit_source_job": event.get("source_job"),
                "exit_source_task": event.get("source_task"),
                "exit_order_source": event.get("order_source"),
                "exit_order_type": event.get("order_type"),
                "exit_attribution_version": event.get("attribution_version"),
                "exit_score_snapshot": _json_safe(event.get("score_snapshot")),
                "exit_decision_inputs": _json_safe(event.get("decision_inputs")),
            })

    end_prices = end_prices or {}
    for ticker, q in lots.items():
        mark = _as_float(end_prices.get(ticker), default=float("nan"))
        for lot in q:
            shares = _as_float(lot.get("remaining_shares"))
            entry_price = _as_float(lot.get("entry_price"))
            gross_pnl = (
                (mark - entry_price) * shares
                if math.isfinite(mark) and entry_price > 0 else None
            )
            rows.append({
                "status": "open",
                "ticker": ticker,
                "entry_event_id": lot.get("event_id"),
                "exit_event_id": None,
                "entry_date": lot.get("entry_date"),
                "exit_date": "",
                "hold_days": None,
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": mark if math.isfinite(mark) else None,
                "gross_pnl": gross_pnl,
                "tax": 0.0,
                "tax_cash_debited": 0.0,
                "tax_cash_debit_mode": None,
                "net_pnl_after_tax": gross_pnl,
                "pnl_pct": (
                    gross_pnl / (entry_price * shares)
                    if gross_pnl is not None and entry_price > 0 and shares > 0
                    else None
                ),
                "sim_sell_pnl_pct": None,
                "exit_reason": "open",
                "partial_exit": False,
                "exit_regime": None,
                "exit_confidence": None,
                "exit_signal_reason": None,
                "exit_rank_score": None,
                "exit_rs_score": None,
                "exit_panel_score": None,
                "exit_mu": None,
                "exit_mu_horizon_days": None,
                "exit_sigma": None,
                "exit_kelly_target_pct": None,
                "exit_expected_return": None,
                "exit_expected_return_horizon_days": None,
                "exit_model_type": None,
                "exit_sector": None,
                "exit_blocked_by": None,
                "exit_qp_delta_w": None,
                "exit_qp_target_w": None,
                "exit_qp_status": None,
                "exit_qp_mu_used": None,
                "exit_qp_sigma_used": None,
                "exit_qp_mu_source": None,
                "exit_alpha_to_mu_applied": None,
                **_empty_fields(EXIT_PARAM_FIELDS),
                **_empty_fields(EXIT_ATTRIBUTION_FIELDS),
                "entry_regime": lot.get("entry_regime"),
                "entry_rank_score": lot.get("entry_rank_score"),
                "entry_rs_score": lot.get("entry_rs_score"),
                "entry_panel_score": lot.get("entry_panel_score"),
                "entry_mu": lot.get("entry_mu"),
                "entry_mu_horizon_days": lot.get("entry_mu_horizon_days"),
                "entry_sigma": lot.get("entry_sigma"),
                "entry_sigma_mult": lot.get("entry_sigma_mult"),
                "entry_kelly_target_pct": lot.get("entry_kelly_target_pct"),
                "entry_expected_return": lot.get("entry_expected_return"),
                "entry_expected_return_horizon_days": lot.get(
                    "entry_expected_return_horizon_days"
                ),
                "entry_model_type": lot.get("entry_model_type"),
                "entry_sector": lot.get("entry_sector"),
                "entry_blocked_by": lot.get("entry_blocked_by"),
                "entry_qp_delta_w": lot.get("entry_qp_delta_w"),
                "entry_qp_target_w": lot.get("entry_qp_target_w"),
                "entry_qp_status": lot.get("entry_qp_status"),
                "entry_qp_mu_used": lot.get("entry_qp_mu_used"),
                "entry_qp_sigma_used": lot.get("entry_qp_sigma_used"),
                "entry_qp_mu_source": lot.get("entry_qp_mu_source"),
                "entry_alpha_to_mu_applied": lot.get(
                    "entry_alpha_to_mu_applied"
                ),
                **{field: lot.get(field) for field in ENTRY_ATTRIBUTION_FIELDS},
            })

    for ticker, q in short_lots.items():
        mark = _as_float(end_prices.get(ticker), default=float("nan"))
        for lot in q:
            shares = _as_float(lot.get("remaining_shares"))
            entry_price = _as_float(lot.get("entry_price"))
            gross_pnl = (
                (entry_price - mark) * shares
                if math.isfinite(mark) and entry_price > 0 else None
            )
            rows.append({
                "status": "open",
                "direction": "short",
                "ticker": ticker,
                "entry_event_id": lot.get("event_id"),
                "exit_event_id": None,
                "entry_date": lot.get("entry_date"),
                "exit_date": "",
                "hold_days": None,
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": mark if math.isfinite(mark) else None,
                "gross_pnl": gross_pnl,
                "tax": 0.0,
                "tax_cash_debited": 0.0,
                "tax_cash_debit_mode": None,
                "net_pnl_after_tax": gross_pnl,
                "pnl_pct": (
                    gross_pnl / (entry_price * shares)
                    if gross_pnl is not None and entry_price > 0 and shares > 0
                    else None
                ),
                "sim_sell_pnl_pct": None,
                "exit_reason": "open",
                "partial_exit": False,
                "exit_regime": None,
                "exit_confidence": None,
                "exit_signal_reason": None,
                "exit_rank_score": None,
                "exit_rs_score": None,
                "exit_panel_score": None,
                "exit_mu": None,
                "exit_mu_horizon_days": None,
                "exit_sigma": None,
                "exit_kelly_target_pct": None,
                "exit_expected_return": None,
                "exit_expected_return_horizon_days": None,
                "exit_model_type": None,
                "exit_sector": None,
                "exit_blocked_by": None,
                "exit_qp_delta_w": None,
                "exit_qp_target_w": None,
                "exit_qp_status": None,
                "exit_qp_mu_used": None,
                "exit_qp_sigma_used": None,
                "exit_qp_mu_source": None,
                "exit_alpha_to_mu_applied": None,
                **_empty_fields(EXIT_PARAM_FIELDS),
                **_empty_fields(EXIT_ATTRIBUTION_FIELDS),
                "entry_regime": lot.get("entry_regime"),
                "entry_rank_score": lot.get("entry_rank_score"),
                "entry_rs_score": lot.get("entry_rs_score"),
                "entry_panel_score": lot.get("entry_panel_score"),
                "entry_mu": lot.get("entry_mu"),
                "entry_mu_horizon_days": lot.get("entry_mu_horizon_days"),
                "entry_sigma": lot.get("entry_sigma"),
                "entry_sigma_mult": lot.get("entry_sigma_mult"),
                "entry_kelly_target_pct": lot.get("entry_kelly_target_pct"),
                "entry_expected_return": lot.get("entry_expected_return"),
                "entry_expected_return_horizon_days": lot.get(
                    "entry_expected_return_horizon_days"
                ),
                "entry_model_type": lot.get("entry_model_type"),
                "entry_sector": lot.get("entry_sector"),
                "entry_blocked_by": lot.get("entry_blocked_by"),
                "entry_qp_delta_w": lot.get("entry_qp_delta_w"),
                "entry_qp_target_w": lot.get("entry_qp_target_w"),
                "entry_qp_status": lot.get("entry_qp_status"),
                "entry_qp_mu_used": lot.get("entry_qp_mu_used"),
                "entry_qp_sigma_used": lot.get("entry_qp_sigma_used"),
                "entry_qp_mu_source": lot.get("entry_qp_mu_source"),
                "entry_alpha_to_mu_applied": lot.get(
                    "entry_alpha_to_mu_applied"
                ),
                **{field: lot.get(field) for field in ENTRY_ATTRIBUTION_FIELDS},
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["entry_thesis"] = df["entry_regime"].map(REGIME_THESES).fillna(
        "No regime thesis recorded on the buy event."
    )
    df["outcome"] = df["gross_pnl"].apply(
        lambda x: "win" if _as_float(x) > 0 else ("loss" if _as_float(x) < 0 else "flat")
    )
    return df.sort_values(["entry_date", "exit_date", "ticker"]).reset_index(drop=True)


def _lot_takes(
    open_lots: list[dict],
    shares_to_sell: float,
    method: str,
) -> list[tuple[int, float]]:
    """Return ``(lot_index, shares)`` disposals matching sim lot semantics."""
    if not open_lots:
        return []
    remaining = float(shares_to_sell)
    if remaining <= 0:
        return []

    if method == "avg":
        total = sum(_as_float(l.get("remaining_shares")) for l in open_lots)
        if total <= 0:
            return []
        take_frac = min(1.0, remaining / total)
        return [
            (i, _as_float(lot.get("remaining_shares")) * take_frac)
            for i, lot in enumerate(open_lots)
            if _as_float(lot.get("remaining_shares")) > 0
        ]

    if method == "hifo":
        order = sorted(
            range(len(open_lots)),
            key=lambda i: -_as_float(open_lots[i].get("entry_price")),
        )
    else:
        order = list(range(len(open_lots)))

    out: list[tuple[int, float]] = []
    for idx in order:
        if remaining <= 1e-9:
            break
        take = min(remaining, _as_float(open_lots[idx].get("remaining_shares")))
        if take <= 0:
            continue
        out.append((idx, take))
        remaining -= take
    return out


def _tax_lot_method_from_config(config: dict[str, Any] | None) -> str:
    ja_cfg = (
        ((config or {}).get("rotation") or {})
        .get("joint_actions", {})
        or {}
    )
    method = str(ja_cfg.get("qp_tax_lot_method", "fifo")).lower()
    return method if method in {"fifo", "hifo", "avg"} else "fifo"


def build_forensic_report(
    *,
    raw_trades: pd.DataFrame,
    round_trips: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any] | None = None,
    title: str = "Sim Trade Forensics",
) -> str:
    """Build a Markdown report summarizing how the sim made or lost money."""
    lines: list[str] = [f"# {title}", ""]
    lines.append("## Run Metrics")
    for key in ("config", "start", "end", "final_value", "total_return",
                "apy", "sharpe", "max_dd", "win_rate", "n_buys", "n_sells",
                "tax_lot_method"):
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    if config:
        ranking = (config.get("ranking") or {}).get("panel_scoring") or {}
        lines.append(f"- panel_buy_floor: {ranking.get('buy_floor')}")
        wf = config.get("walkforward") or {}
        if wf.get("enabled"):
            lines.append("- scoring_mode: walkforward_manifest_per_bar")
            lines.append(
                f"- panel_artifact_path_config_seed: {ranking.get('artifact_path')}"
            )
            lines.append(f"- walkforward_manifest: {wf.get('manifest_path')}")
        else:
            lines.append(f"- panel_artifact_path: {ranking.get('artifact_path')}")
    lines.append("")

    lines.append("## Theoretical Frame")
    lines.append(
        "A long-only cross-sectional rank strategy should earn money only if "
        "entry scores have positive realized cross-sectional information "
        "coefficient, the regime label matches the regime-specific thesis, and "
        "sizing/exits do not convert alpha into tax or drawdown drag."
    )
    lines.append(
        "For each round trip below, the entry thesis is derived from the buy "
        "event's regime. A losing closed trade means the entry thesis failed, "
        "the exit came too late, sizing was too large, or the regime label was "
        "not economically correct for that bar."
    )
    lines.append("")

    if raw_trades.empty:
        lines.append("No trade events recorded.")
        return "\n".join(lines) + "\n"

    closed = round_trips[round_trips.get("status", "") == "closed"].copy() if not round_trips.empty else pd.DataFrame()
    open_rows = round_trips[round_trips.get("status", "") == "open"].copy() if not round_trips.empty else pd.DataFrame()

    lines.append("## Attribution")
    if closed.empty:
        lines.append("No closed round trips.")
    else:
        total_gross = float(closed["gross_pnl"].fillna(0).sum())
        total_tax = float(closed["tax"].fillna(0).sum())
        total_tax_cash = (
            float(closed["tax_cash_debited"].fillna(0).sum())
            if "tax_cash_debited" in closed.columns else total_tax
        )
        total_net = float(closed["net_pnl_after_tax"].fillna(0).sum())
        lines.append(f"- closed_round_trips: {len(closed)}")
        lines.append(f"- gross_pnl: {total_gross:+.2f}")
        lines.append(f"- tax_estimate: {total_tax:+.2f}")
        lines.append(f"- tax_cash_debited: {total_tax_cash:+.2f}")
        lines.append(f"- net_pnl_after_tax: {total_net:+.2f}")
        lines.append(
            f"- win_rate_closed: {closed['gross_pnl'].gt(0).mean():.2%}"
        )
        lines.append("")

        tax_summary = annual_net_tax_summary(
            closed, (config or {}).get("tax") if config else None,
        )
        lines.append("### Tax Stress")
        lines.append(
            f"- event_level_tax_estimate: {total_tax:+.2f}"
        )
        lines.append(
            f"- event_level_tax_debited: {total_tax_cash:+.2f}"
        )
        lines.append(
            f"- annual_net_tax_estimate: "
            f"{tax_summary['total_estimated_tax']:+.2f}"
        )
        annual_net_tax = float(tax_summary["total_estimated_tax"])
        lines.append(
            f"- tax_overstatement_vs_annual_net: "
            f"{total_tax - annual_net_tax:+.2f}"
        )
        lines.append(
            f"- annual_net_pnl_estimate: "
            f"{total_gross - annual_net_tax:+.2f}"
        )
        if tax_summary["years"]:
            lines.append(pd.DataFrame(tax_summary["years"]).to_markdown(index=False, floatfmt=".2f"))
        lines.append("")

        for group_col in ("entry_regime", "exit_regime", "exit_reason", "ticker"):
            if group_col not in closed.columns:
                continue
            grp = (
                closed.groupby(group_col, dropna=False)
                .agg(
                    n=("ticker", "size"),
                    gross_pnl=("gross_pnl", "sum"),
                    net_pnl_after_tax=("net_pnl_after_tax", "sum"),
                    mean_pnl_pct=("pnl_pct", "mean"),
                    win_rate=("gross_pnl", lambda s: float((s > 0).mean())),
                    median_hold_days=("hold_days", "median"),
                )
                .sort_values("net_pnl_after_tax")
            )
            lines.append(f"### By {group_col}")
            lines.append(grp.to_markdown(floatfmt=".4f"))
            lines.append("")

        worst_cols = [
            "ticker", "entry_date", "exit_date", "entry_regime", "exit_regime", "exit_reason",
            "shares", "entry_price", "exit_price", "gross_pnl",
            "net_pnl_after_tax", "pnl_pct", "hold_days", "entry_rank_score",
            "entry_mu", "entry_sigma", "exit_stop_loss_pct",
            "exit_take_profit_pct", "exit_stop_decay_days",
            "exit_stop_decay_floor", "exit_sdl_n_sigma",
            "exit_sdl_skip_if_unrealized_above",
            "exit_trailing_stop_trigger_pct", "exit_source_job",
            "exit_source_task",
        ]
        worst_cols = [c for c in worst_cols if c in closed.columns]
        lines.append("### Worst 25 Closed Round Trips")
        lines.append(closed.sort_values("net_pnl_after_tax").head(25)[worst_cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        lines.append("### Best 15 Closed Round Trips")
        lines.append(closed.sort_values("net_pnl_after_tax", ascending=False).head(15)[worst_cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    if not open_rows.empty:
        cols = [
            "ticker", "entry_date", "entry_regime", "shares", "entry_price",
            "exit_price", "gross_pnl", "pnl_pct", "entry_rank_score",
            "entry_mu", "entry_sigma",
        ]
        lines.append("## Open Lots At End")
        lines.append(open_rows[cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("## Full Ledger Location")
    lines.append(
        "The CSV sidecars contain every raw event and every FIFO-matched "
        "round trip; this report only shows the most diagnostic slices."
    )
    return "\n".join(lines) + "\n"


def write_trade_outputs(
    *,
    result: Any,
    config: dict[str, Any] | None = None,
    trade_json: str | Path | None = None,
    trade_csv: str | Path | None = None,
    round_trips_csv: str | Path | None = None,
    report_md: str | Path | None = None,
    end_prices: dict[str, float] | None = None,
    title: str = "Sim Trade Forensics",
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write requested trade-ledger artifacts and return written paths."""
    trade_log = _enrich_trade_log_from_result(result)
    raw = trade_log_frame(trade_log)
    lot_method = _tax_lot_method_from_config(config)
    trips = round_trips_from_trade_log(
        trade_log,
        end_prices=end_prices,
        lot_method=lot_method,
    )
    metrics = {
        "final_value": float(getattr(result, "final_value", 0.0)),
        "total_return": float(getattr(result, "total_return", 0.0)),
        "apy": float(getattr(result, "apy", 0.0)),
        "sharpe": float(getattr(result, "sharpe", float("nan"))),
        "max_dd": float(getattr(result, "max_dd", float("nan"))),
        "win_rate": float(getattr(result, "win_rate", 0.0)),
        "n_buys": len(getattr(result, "buys", []) or []),
        "n_sells": len(getattr(result, "sells", []) or []),
        "tax_lot_method": lot_method,
    }
    metrics.update(extra_metrics or {})

    written: dict[str, str] = {}
    if trade_json:
        p = Path(trade_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_json_safe(raw.to_dict(orient="records")), indent=2))
        written["trade_json"] = str(p)
    if trade_csv:
        p = Path(trade_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(p, index=False)
        written["trade_csv"] = str(p)
    if round_trips_csv:
        p = Path(round_trips_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        trips.to_csv(p, index=False)
        written["round_trips_csv"] = str(p)
    if report_md:
        p = Path(report_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            build_forensic_report(
                raw_trades=raw,
                round_trips=trips,
                metrics=metrics,
                config=config,
                title=title,
            )
        )
        written["report_md"] = str(p)
    return written
