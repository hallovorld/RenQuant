"""Shared decision-trace helpers for sim, live runner, and LEAN.

Adapters own execution side effects, but decision rows should not be
hand-built three different ways. This module centralizes the audit surface
used by `candidate_scores` and `ticker_daily_state`.
"""
from __future__ import annotations

from typing import Any


def model_type_from_artifact(model: Any) -> str | None:
    """Extract a readable model type from dict/object artifacts."""
    if model is None:
        return None
    if isinstance(model, dict):
        meta = model.get("_metadata") or model.get("metadata") or {}
        for src in (meta, model):
            if not isinstance(src, dict):
                continue
            for key in ("best_approach", "model_type", "policy_type", "type", "kind"):
                val = src.get(key)
                if isinstance(val, str) and val:
                    return val
        return None
    meta = getattr(model, "metadata", None)
    if isinstance(meta, dict):
        for key in ("best_approach", "model_type", "policy_type", "type", "kind"):
            val = meta.get(key)
            if isinstance(val, str) and val:
                return val
    val = getattr(model, "model_type", None)
    return val if isinstance(val, str) and val else None


def model_types_from_models(models: dict[str, Any] | None) -> dict[str, str | None]:
    return {
        tk: model_type_from_artifact(model)
        for tk, model in (models or {}).items()
    }


def selected_buy_tickers(trade_events: list[dict[str, Any]] | None) -> set[str]:
    """Return tickers with buy trade/order events."""
    return {
        str(event.get("ticker"))
        for event in (trade_events or [])
        if str(event.get("action") or "").lower() == "buy" and event.get("ticker")
    }


def candidate_trace_pool(ctx: Any) -> list[Any]:
    """Full candidate pool for trace persistence, including filtered candidates."""
    return list(
        getattr(ctx, "_full_candidate_snapshot", None)
        or getattr(ctx, "candidates", None)
        or []
    )


def qp_trace_maps(ctx: Any) -> tuple[dict[str, float], dict[str, float], str | None]:
    """Extract per-ticker QP delta/target/status from the shared QP solution."""
    qp_delta_by_ticker: dict[str, float] = {}
    qp_target_by_ticker: dict[str, float] = {}
    qp_status = None
    qp_sol = getattr(ctx, "_qp_solution", None)
    qp_tickers = list(getattr(ctx, "_qp_tickers", None) or [])
    if qp_sol is not None and qp_tickers:
        qp_status = getattr(qp_sol, "status", None)
        for idx, tk in enumerate(qp_tickers):
            try:
                qp_delta_by_ticker[tk] = float(qp_sol.delta_w[idx])
            except Exception:
                pass
            try:
                qp_target_by_ticker[tk] = float(qp_sol.target_w[idx])
            except Exception:
                pass
    return qp_delta_by_ticker, qp_target_by_ticker, qp_status


def _score_value(src: Any, snap: dict[str, Any], name: str) -> Any:
    value = getattr(src, name, None) if src is not None else None
    return value if value is not None else snap.get(name)


def build_ticker_daily_state_rows(
    *,
    config: dict,
    ctx: Any,
    selected_tickers: set[str],
    blocked_map: dict[str, str] | None,
    model_types: dict[str, str | None],
    universe_rejections: dict[str, str] | None = None,
    model_keys: set[str] | None = None,
    pending_broker_tickers: set[str] | None = None,
    portfolio_value: float | None = None,
    sector_map: dict[str, str] | None = None,
    qp_delta_by_ticker: dict[str, float] | None = None,
    qp_target_by_ticker: dict[str, float] | None = None,
    qp_status: str | None = None,
) -> list[dict[str, Any]]:
    """Build one `ticker_daily_state` row per decision-trace ticker.

    This is shared by sim/live/LEAN so blocked reasons, score snapshots, QP
    fields, and pending-broker semantics cannot drift adapter-by-adapter.
    """
    from kernel.pipeline.task_benchmark_sleeve import decision_trace_tickers  # noqa: PLC0415

    blocked_map = blocked_map or {}
    sector_map = sector_map if sector_map is not None else (config.get("sector_map", {}) or {})
    universe_rejections = universe_rejections or {}
    pending_broker_tickers = pending_broker_tickers or set()
    qp_delta_by_ticker = qp_delta_by_ticker or {}
    qp_target_by_ticker = qp_target_by_ticker or {}
    model_keys = model_keys if model_keys is not None else set(model_types)

    cand_pool = candidate_trace_pool(ctx)
    cand_by_t = {c.ticker: c for c in cand_pool}
    score_snapshots = getattr(ctx, "_ticker_score_snapshot", {}) or {}
    prices = getattr(ctx, "prices", {}) or {}
    holdings = getattr(ctx, "holdings", {}) or {}
    pf_value = (
        float(portfolio_value)
        if portfolio_value is not None
        else float(getattr(ctx, "portfolio_value", 0.0) or 0.0)
    )

    rows: list[dict[str, Any]] = []
    for tk in decision_trace_tickers(config):
        hs = holdings.get(tk)
        cand = cand_by_t.get(tk)
        src = cand if cand is not None else hs
        snap = score_snapshots.get(tk, {}) or {}
        has_pos = 1 if hs is not None else 0
        pos_qty = float(getattr(hs, "shares", 0.0)) if hs else None
        px = prices.get(tk, 0.0)
        pos_pct = None
        if hs and pf_value > 0 and px:
            pos_pct = (pos_qty * px) / pf_value

        blocked_str = blocked_map.get(tk)
        if blocked_str is None and tk not in model_keys:
            reason = universe_rejections.get(tk, "not_loaded")
            blocked_str = f"universe:{reason}"
        if blocked_str is None and tk in pending_broker_tickers:
            blocked_str = "broker_pending"
        if blocked_str is None and cand is None and hs is not None:
            blocked_str = "held_no_new_buy"
        if blocked_str is None and cand is None:
            blocked_str = "no_model_signal"
        if blocked_str is None and tk not in selected_tickers:
            blocked_str = "not_selected"

        if cand is not None:
            model_action = "buy"
        elif hs is not None and getattr(hs, "sell_streak", 0) > 0:
            model_action = "sell"
        else:
            model_action = snap.get("model_action", "hold")

        rows.append({
            "ticker": tk,
            "regime": getattr(ctx, "regime", None),
            "confidence": getattr(ctx, "confidence", None),
            "in_watchlist": 1,
            "in_universe": 1 if tk in model_keys else 0,
            "pending_at_broker": 1 if tk in pending_broker_tickers else 0,
            "has_position": has_pos,
            "position_qty": pos_qty,
            "position_pct": pos_pct,
            "model_type": model_types.get(tk),
            "model_action": model_action,
            "sell_streak": int(getattr(hs, "sell_streak", 0)) if hs else None,
            "panel_score": _score_value(src, snap, "panel_score"),
            "rank_score": _score_value(src, snap, "rank_score"),
            "expected_return": _score_value(src, snap, "expected_return"),
            "kelly_target_pct": _score_value(src, snap, "kelly_target_pct"),
            "mu": _score_value(src, snap, "mu"),
            "sigma": _score_value(src, snap, "sigma"),
            "in_candidates": 1 if cand is not None else 0,
            "selected": 1 if tk in selected_tickers else 0,
            "blocked_by": blocked_str,
            "sector": sector_map.get(tk),
            "qp_delta_w": qp_delta_by_ticker.get(tk),
            "qp_target_w": qp_target_by_ticker.get(tk),
            "qp_status": qp_status,
        })
    return rows


__all__ = [
    "build_ticker_daily_state_rows",
    "candidate_trace_pool",
    "model_type_from_artifact",
    "model_types_from_models",
    "qp_trace_maps",
    "selected_buy_tickers",
]
