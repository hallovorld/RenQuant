"""RunnerAdapter — bridges live broker state → InferenceContext → order execution.

Can import kernel/ and common/ (runs on host, not in LEAN Docker).
"""
from __future__ import annotations

import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from adapters.panel_runtime import (
    attach_panel_runtime_frames,
    build_runtime_feature_cache,
    describe_panel_frame_bundle,
    prepare_panel_runtime_frames,
)
from kernel.decision_trace import (
    build_ticker_daily_state_rows,
    candidate_trace_pool,
    model_type_from_artifact as _shared_model_type_from_artifact,
    model_types_from_models,
    qp_trace_maps,
    selected_buy_tickers,
    trade_event_blocked_map,
    trade_event_tickers,
)
from kernel.pipeline.task_execution import (
    dedupe_exit_signals,
    is_full_liquidate_signal,
)
from kernel.trade_events import (
    build_buy_trade_event,
    build_sell_trade_event as build_sell_trade_event_for_db,
)

log = logging.getLogger("adapters.runner")


# ── Helpers ────────────────────────────────────────────────────────────────────

# Ratio above which a stored high_water_mark is treated as "stale" relative to
# current account value and snapped down. Chosen so that a real 33% drawdown
# (hwm/equity ratio = 1.49) is preserved but the typical stale-seed case
# (hwm=$100k, equity=$10k → ratio 10×) trips the snap.
_HWM_STALE_RATIO = 1.5


def _parse_iso_dt(s: Any) -> "datetime.datetime | None":
    """Parse an ISO-formatted datetime string; return None on any failure.

    Used to restore RegimeState.cooldown_start from live_state.json across
    invocations (CUSUM-v2 Design C).
    """
    if s is None or s == "":
        return None
    try:
        return datetime.datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def resolve_hwm(stored_hwm: float, account_value: float,
                stale_ratio: float = _HWM_STALE_RATIO) -> tuple[float, bool]:
    """Resolve the effective high_water_mark for the current bar.

    The live-trading DrawdownCircuitTask divides `(hwm - equity) / hwm` and
    compares to `halt_pct`. When `hwm` is stale (e.g. initial-seed $100k
    from a fresh install, actual Alpaca equity a fraction of that), this
    ratio blows up and the drawdown halt latches on every bar — exactly
    the 2026-04-23 "zero orders despite healthy models" bug.

    Rule: if stored_hwm > stale_ratio * account_value, snap to account_value.
    Otherwise ratchet up to max(stored_hwm, account_value) as before.

    Returns (resolved_hwm, was_snapped).

    Audit fix RU-1 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
    account_value (broker outage / Alpaca returns NaN equity) slipped
    past `account_value > 0` (NaN comparisons False), then `max(hwm,
    NaN) = NaN` → resolved_hwm = NaN → DrawdownCircuitTask's
    `(NaN - equity) / NaN` = NaN → `NaN >= halt_pct` False → drawdown
    gate silently disabled in LIVE TRADING for the rest of the run.
    Post-fix: explicit isfinite check; on bad account_value, fall back
    to stored_hwm unchanged (fail-SAFE behaviour — keeps the gate
    armed against the LAST-known good HWM).
    """
    import math
    if not math.isfinite(account_value):
        # Bad broker data → preserve stored HWM intact, no snap.
        if math.isfinite(stored_hwm):
            return float(stored_hwm), False
        return 0.0, False
    if not math.isfinite(stored_hwm):
        # Stored HWM is corrupted but account_value is good → reset to
        # account_value so future drawdown calc is meaningful.
        return float(account_value), True
    if account_value > 0 and stored_hwm > stale_ratio * account_value:
        return float(account_value), True
    return float(max(stored_hwm, account_value)), False


def persisted_skip_buys(state: dict | None) -> bool:
    """Read persisted drawdown-halt state with legacy-safe coercion.

    SimAdapter carries ``_skip_buys`` across bars in-process. RunnerAdapter is
    relaunched by scheduled jobs, so the same hysteresis state must round-trip
    through live_state; otherwise live exits the drawdown recovery band earlier
    than sim.
    """
    if not isinstance(state, dict):
        return False
    value = state.get("skip_buys", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def sell_event_price(sig: Any, fallback_price: Any) -> float:
    """Use broker-confirmed sell fill price when present, else fallback."""
    import math
    for value in (getattr(sig, "sell_price", None), fallback_price):
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and price > 0.0:
            return price
    return 0.0


def _finite_number(value: Any, default: float = 0.0) -> float:
    import math
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def sell_event_realized_kwargs(
    sig: Any,
    holding: Any,
    *,
    today: Any,
) -> dict[str, Any]:
    """Explicit economics for live sell rows after broker confirmation."""
    out: dict[str, Any] = {}
    shares = _finite_number(getattr(sig, "shares_sold", None))
    if shares > 0:
        out["shares"] = shares
    gross_attr = getattr(sig, "realized_pnl_dollar", None)
    gross = _finite_number(gross_attr)
    if gross_attr is not None:
        out["gross_pnl"] = gross
    cost_basis = _finite_number(getattr(sig, "cost_basis", None))
    proceeds_basis_attr = getattr(sig, "proceeds_basis", None)
    proceeds_basis = _finite_number(proceeds_basis_attr)
    if proceeds_basis_attr is not None and proceeds_basis > 0:
        out["proceeds_basis"] = proceeds_basis
    elif cost_basis > 0 and shares > 0:
        out["proceeds_basis"] = cost_basis * shares
    tax_attr = getattr(sig, "realized_tax", None)
    if tax_attr is not None:
        out["tax"] = _finite_number(tax_attr)
    net_attr = getattr(sig, "net_pnl_after_tax", None)
    if net_attr is not None:
        out["net_pnl_after_tax"] = _finite_number(net_attr)
    pnl_pct_attr = getattr(sig, "realized_pnl_pct", None)
    if pnl_pct_attr is not None:
        out["pnl_pct"] = _finite_number(pnl_pct_attr) / 100.0

    hold_days_attr = getattr(sig, "hold_days", None)
    if hold_days_attr is not None:
        out["hold_days"] = max(int(_finite_number(hold_days_attr)), 0)
        return out
    entry_date = getattr(holding, "entry_date", None)
    today_date = today.date() if isinstance(today, datetime.datetime) else today
    if isinstance(entry_date, datetime.datetime):
        entry_date = entry_date.date()
    if isinstance(today_date, datetime.date) and isinstance(entry_date, datetime.date):
        out["hold_days"] = max((today_date - entry_date).days, 0)
    return out


def _fill_date(fill: dict[str, Any]) -> datetime.date | None:
    raw = fill.get("filled_at")
    if not raw:
        return None
    try:
        return datetime.datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        return None


def reconstruct_live_tax_lots_from_fills(
    fills: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, list[Any]]:
    """Rebuild current long tax lots from broker fill history.

    This keeps live partial-sell accounting on the same FIFO/HIFO contract as
    sim/LEAN. Alpaca exposes only average entry on positions, which is not
    enough to audit a partial trim's realized basis.
    """
    from kernel.exits import HoldingState, TaxLot, apply_buy_lot, apply_sell_lots_detailed

    lot_method = str(
        (((config or {}).get("rotation", {}) or {}).get("joint_actions", {}) or {})
        .get("qp_tax_lot_method", ((config or {}).get("tax", {}) or {}).get("lot_method", "fifo"))
    ).lower()
    states: dict[str, HoldingState] = {}
    ordered = sorted(
        [f for f in (fills or []) if isinstance(f, dict)],
        key=lambda f: str(f.get("filled_at") or ""),
    )
    for fill in ordered:
        ticker = str(fill.get("symbol") or "").strip()
        action = str(fill.get("action") or "").upper()
        try:
            qty = float(fill.get("qty") or 0.0)
            price = float(fill.get("avg_price") or fill.get("filled_avg_price") or 0.0)
        except (TypeError, ValueError):
            continue
        fill_date = _fill_date(fill)
        if not ticker or qty <= 0 or price <= 0 or fill_date is None:
            continue
        hs = states.get(ticker)
        if action == "BUY":
            if hs is None:
                hs = HoldingState(
                    entry_price=price,
                    entry_date=fill_date,
                    high_watermark=price,
                    shares=0.0,
                )
                states[ticker] = hs
            apply_buy_lot(hs, qty, price, fill_date)
            hs.shares = hs.total_shares()
            hs.entry_price = hs.weighted_avg_entry_price()
            hs.high_watermark = max(float(hs.high_watermark or price), price)
        elif action == "SELL" and hs is not None:
            apply_sell_lots_detailed(hs, qty, lot_method)
            hs.shares = hs.total_shares()
            hs.entry_price = hs.weighted_avg_entry_price()
            if hs.shares <= 1e-9:
                states.pop(ticker, None)
    return {
        ticker: [TaxLot(shares=L.shares, price=L.price, date=L.date) for L in hs.lots]
        for ticker, hs in states.items()
        if hs.lots
    }


def apply_live_sell_lot_accounting(
    sig: Any,
    holding: Any,
    *,
    shares: float,
    price: float,
    today: Any,
    config: dict[str, Any] | None = None,
) -> bool:
    """Stamp live sell economics from reconstructed tax lots when available."""
    import math
    from kernel.exits import apply_sell_lots_detailed
    from kernel.portfolio import compute_disposed_lot_tax

    if holding is None or not getattr(holding, "lots", None):
        return False
    if not (math.isfinite(float(shares)) and shares > 0
            and math.isfinite(float(price)) and price > 0):
        return False
    lot_method = str(
        (((config or {}).get("rotation", {}) or {}).get("joint_actions", {}) or {})
        .get("qp_tax_lot_method", ((config or {}).get("tax", {}) or {}).get("lot_method", "fifo"))
    ).lower()
    proceeds_basis, _, disposed_lots = apply_sell_lots_detailed(
        holding, float(shares), lot_method,
    )
    if not (math.isfinite(float(proceeds_basis)) and proceeds_basis > 0):
        return False
    gross_pnl = float(shares) * float(price) - float(proceeds_basis)
    today_date = today.date() if isinstance(today, datetime.datetime) else today
    tax_cfg = (config or {}).get("tax", {}) or {}
    lot_tax = compute_disposed_lot_tax(
        float(price),
        today_date,
        disposed_lots,
        float(tax_cfg.get("short_term_rate", 0.50)),
        float(tax_cfg.get("long_term_rate", 0.32)),
        int(tax_cfg.get("long_term_threshold_days", 365)),
    )
    tax = float(lot_tax.get("tax", 0.0))
    cost_basis = float(proceeds_basis) / float(shares)
    sig.cost_basis = cost_basis
    sig.proceeds_basis = float(proceeds_basis)
    sig.realized_pnl_dollar = float(gross_pnl)
    sig.realized_tax = tax
    sig.net_pnl_after_tax = float(gross_pnl - tax)
    sig.realized_pnl_pct = (
        float(gross_pnl) / float(proceeds_basis) * 100.0
        if proceeds_basis > 0 else 0.0
    )
    sig.hold_days = int(round(float(lot_tax.get("weighted_hold_days", 0.0))))
    try:
        holding.shares = max(0.0, float(getattr(holding, "shares", 0.0) or 0.0) - float(shares))
        if getattr(holding, "lots", None):
            holding.entry_price = holding.weighted_avg_entry_price()
    except Exception:
        pass
    return True


def live_trace_selection_maps(
    trade_events: list[dict[str, Any]] | None,
    pending_orders: list[dict[str, Any]] | None,
    blocked_map: dict[str, str] | None = None,
) -> tuple[set[str], dict[str, str], set[str]]:
    """Trace filled buys as selected and pending submissions as blocked."""
    pending_tickers = {
        str(o.get("ticker"))
        for o in (pending_orders or [])
        if isinstance(o, dict) and o.get("ticker")
    }
    out_blocked = dict(blocked_map or {})
    for ticker in pending_tickers:
        out_blocked.setdefault(ticker, "broker_pending_submitted")
    return selected_buy_tickers(trade_events), out_blocked, pending_tickers


def live_execution_attempt_events(ctx: Any) -> list[dict[str, Any]]:
    """Persist non-filled broker attempts as audit events.

    Filled orders remain `buy`/`sell` trade rows. Pending/skipped/rejected
    attempts are decision-tree evidence, not fills, so they use distinct
    actions and do not count as selected buys or realized exits.
    """
    events: list[dict[str, Any]] = []
    today = getattr(ctx, "today", None)
    regime = getattr(ctx, "regime", None)
    confidence = getattr(ctx, "confidence", None)
    for order in getattr(ctx, "orders_pending", []) or []:
        if isinstance(order, dict):
            events.append(_buy_attempt_event(
                order, "buy_pending", ctx, "broker_pending_submitted",
            ))
    for order in getattr(ctx, "orders_skipped", []) or []:
        if isinstance(order, dict):
            reason = f"broker_skip:{order.get('skip_reason', 'skipped')}"
            events.append(_buy_attempt_event(order, "buy_skipped", ctx, reason))
    for item in getattr(ctx, "exits_pending", []) or []:
        if isinstance(item, dict):
            events.append(_sell_attempt_event(item, "sell_pending", ctx))
    for item in getattr(ctx, "exits_failed", []) or []:
        if isinstance(item, dict):
            events.append(_sell_attempt_event(item, "sell_rejected", ctx))
    for event in events:
        event.setdefault("date", today)
        event.setdefault("regime", regime)
        event.setdefault("confidence", confidence)
    return events


def _buy_attempt_event(
    order: dict[str, Any],
    action: str,
    ctx: Any,
    blocked_by: str,
) -> dict[str, Any]:
    row = build_buy_trade_event(
        order,
        date=getattr(ctx, "today", None),
        default_regime=getattr(ctx, "regime", None),
        default_confidence=getattr(ctx, "confidence", None),
        default_acceptance_reason=blocked_by,
    )
    row["action"] = action
    row["blocked_by"] = blocked_by
    row["exit_reason"] = order.get("skip_reason") or order.get("status") or blocked_by
    row["status"] = order.get("status")
    inputs = dict(row.get("decision_inputs") or {})
    inputs.update({
        "attempt_status": action,
        "skip_reason": order.get("skip_reason"),
        "status": order.get("status"),
        "order_id": order.get("order_id"),
        "blocked_by": blocked_by,
    })
    inputs.setdefault("acceptance_reason", blocked_by)
    row["decision_inputs"] = inputs
    snap = dict(row.get("score_snapshot") or {})
    snap.update({
        "attempt_status": action,
        "blocked_by": blocked_by,
        "status": order.get("status"),
    })
    row["score_snapshot"] = snap
    return row


def _sell_attempt_event(
    item: dict[str, Any],
    action: str,
    ctx: Any,
) -> dict[str, Any]:
    ticker = item.get("ticker")
    hs = (getattr(ctx, "holdings", None) or {}).get(ticker)
    blocked_by = item.get("error") or item.get("status") or action
    price = (getattr(ctx, "prices", None) or {}).get(ticker)
    source_job = str(item.get("source_job") or getattr(item.get("sig", None), "source_job", None) or "LiveBroker")
    source_task = str(item.get("source_task") or getattr(item.get("sig", None), "source_task", None) or action)
    order_source = str(item.get("order_source") or f"{source_job}.{source_task}")
    snap = {
        "rank_score": getattr(hs, "rank_score", None),
        "panel_score": getattr(hs, "panel_score", None),
        "expected_return": getattr(hs, "expected_return", None),
        "expected_return_horizon_days": getattr(hs, "expected_return_horizon_days", None),
        "mu": getattr(hs, "mu", None),
        "mu_horizon_days": getattr(hs, "mu_horizon_days", None),
        "sigma": getattr(hs, "sigma", None),
        "confidence": getattr(ctx, "confidence", None),
        "regime": getattr(ctx, "regime", None),
        "model_type": getattr(hs, "model_type", None),
        "sector": getattr(hs, "sector", None),
        "blocked_by": blocked_by,
        "attempt_status": action,
        "status": item.get("status"),
    }
    inputs = {
        "acceptance_reason": blocked_by,
        "attempt_status": action,
        "exit_type": item.get("exit_type"),
        "signal_reason": item.get("reason"),
        "shares": item.get("qty"),
        "price": price,
        "is_partial": item.get("is_partial"),
        "status": item.get("status"),
        "order_id": item.get("order_id"),
        "error": item.get("error"),
        "source_job": source_job,
        "source_task": source_task,
        "order_source": order_source,
    }
    return {
        "ticker": ticker,
        "action": action,
        "date": getattr(ctx, "today", None),
        "shares": item.get("qty"),
        "price": price,
        "exit_reason": item.get("exit_type"),
        "blocked_by": blocked_by,
        "order_type": f"SELL_ATTEMPT_{item.get('exit_type') or action}",
        "source": "LiveBroker",
        "source_job": source_job,
        "source_task": source_task,
        "order_source": order_source,
        "attribution_version": "live_execution_attempt_v1",
        "score_snapshot": snap,
        "decision_inputs": inputs,
        "confidence": getattr(ctx, "confidence", None),
        "regime": getattr(ctx, "regime", None),
        "model_type": getattr(hs, "model_type", None),
        "sector": getattr(hs, "sector", None),
    }


def cap_buy_order_to_cash(order: dict, remaining_cash: float) -> tuple[dict | None, str | None]:
    """Resize or reject one buy intent against the runner's live cash ledger."""
    import math
    try:
        cash = float(remaining_cash)
        shares = float(order.get("shares", 0.0))
        price = float(order.get("price", 0.0))
    except (TypeError, ValueError, AttributeError):
        return None, "bad_order"
    if not (math.isfinite(cash) and math.isfinite(shares)
            and math.isfinite(price) and price > 0 and shares > 0):
        return None, "bad_order"
    invest = shares * price
    if invest <= cash + 1e-6:
        capped = dict(order)
        capped["invest"] = invest
        return capped, None
    affordable = int(cash // price)
    if affordable < 1:
        return None, "cash_budget_exhausted"
    capped = dict(order)
    capped["shares"] = affordable
    capped["invest"] = affordable * price
    capped["budget_adjustment"] = "cash_budget_resized"
    capped["original_shares"] = order.get("shares")
    return capped, "cash_budget_resized"


def same_bar_sell_credit(ctx: Any) -> float:
    """Estimated cash made available by broker-confirmed same-bar sells."""
    import math
    credit = 0.0
    for ticker, sig in getattr(ctx, "exits_placed", []) or []:
        try:
            shares = float(getattr(sig, "shares_sold", 0.0) or 0.0)
            price = float(getattr(sig, "sell_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(shares) and math.isfinite(price) and shares > 0 and price > 0:
            credit += shares * price
        else:
            log.warning(
                "LIVE-SAME-BAR-SELL-CREDIT: skip non-finite sell credit "
                "%s shares=%s price=%s",
                ticker, shares, price,
            )
    return credit


def normalize_order_status(status: Any) -> str:
    """Normalize broker enum/string order status to a lower-case token."""
    return str(status or "").split(".")[-1].strip().lower()


def broker_order_execution(
    result: dict | None,
    requested_qty: float,
    fallback_price: float,
) -> dict[str, Any]:
    """Classify a broker order response as filled, pending, or rejected.

    Live Alpaca can accept an after-close market DAY order without executing it
    until the next session. Only filled quantity is allowed to mutate live
    state, trade DB rows, same-bar cash credit, or realized P/L.
    """
    import math

    result = dict(result or {})
    status = normalize_order_status(result.get("status"))
    terminal_rejects = {
        "rejected", "canceled", "cancelled", "expired",
        "stopped", "suspended", "done_for_day",
    }
    def _finite_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return out if math.isfinite(out) else default

    requested = _finite_float(requested_qty)
    filled_qty = _finite_float(result.get("filled_qty"))
    if filled_qty <= 0 and status in {"filled", "partially_filled"}:
        filled_qty = _finite_float(result.get("quantity"), requested)
    avg_price = _finite_float(result.get("filled_avg_price"))
    if avg_price <= 0:
        avg_price = _finite_float(result.get("price"), fallback_price)

    is_filled = filled_qty > 0 or status == "filled"
    is_partial = (
        status == "partially_filled"
        or (is_filled and requested > 0 and filled_qty < requested - 1e-9)
    )
    is_rejected = status in terminal_rejects
    is_pending = not is_filled and not is_rejected
    return {
        **result,
        "status": status,
        "filled": bool(is_filled),
        "pending": bool(is_pending),
        "rejected": bool(is_rejected),
        "partial": bool(is_partial),
        "filled_qty": float(filled_qty if is_filled else 0.0),
        "filled_avg_price": float(avg_price if avg_price > 0 else fallback_price),
    }


def effective_live_holdings_after_orders(
    starting_holding_tickers: Any,
    full_exit_tickers: set[str],
    orders_placed: Any,
) -> set[str]:
    """Return live holdings after confirmed full exits and filled buys.

    ``ctx.holdings`` is a start-of-bar snapshot. RunnerAdapter must subtract
    broker-confirmed full exits before state GC, otherwise it can resurrect
    sell streak / HWM state for positions that were just liquidated.
    """
    current = {str(t) for t in (starting_holding_tickers or []) if t}
    current.difference_update({str(t) for t in (full_exit_tickers or set()) if t})
    for order in orders_placed or []:
        ticker = order.get("ticker") if isinstance(order, dict) else None
        if ticker:
            current.add(str(ticker))
    return current


def live_post_execution_snapshot(
    ctx: Any,
    broker: Any,
    currently_held: set[str],
) -> dict[str, Any]:
    """Best-effort post-order account snapshot for persistence metrics."""
    import math

    def _finite(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    pv = None
    if hasattr(broker, "get_account_value"):
        try:
            pv = _finite(broker.get_account_value())
        except Exception:
            pv = None
    if pv is None:
        pv = _finite(getattr(ctx, "portfolio_value", None))

    cash = None
    if hasattr(broker, "get_cash"):
        try:
            cash = _finite(broker.get_cash())
        except Exception:
            cash = None
    if cash is None:
        cash = _finite(getattr(ctx, "cash", None))

    return {
        "portfolio_value": pv,
        "cash": cash,
        "n_holdings": len(currently_held),
    }


def _preopen_cancel_symbols(strategy_dir: Path, broker_name: str | None, today_str: str) -> set[str]:
    """Symbols whose queued orders were cancelled by the pre-open gate today."""
    if broker_name != "alpaca":
        return set()
    ledger = strategy_dir.parent.parent / "logs" / "alerts" / "preopen_cancel_ledger.jsonl"
    if not ledger.exists():
        return set()
    out: set[str] = set()
    try:
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("date") == today_str and row.get("broker") == broker_name:
                sym = str(row.get("symbol") or "").strip().upper()
                if sym:
                    out.add(sym)
    except Exception as exc:  # noqa: BLE001
        log.warning("preopen cancel ledger read failed: %s", exc)
    return out


def model_type_from_artifact(model: Any) -> str | None:
    """Extract the human model type from dict/object artifacts for DB audit rows."""
    return _shared_model_type_from_artifact(model)


class RunnerAdapter:
    """Translate between the live broker/state and InferenceContext.

    Usage::

        adapter = RunnerAdapter(config, models, broker, strategy_dir, sell_only)
        ctx = adapter.make_context()
        InferencePipeline().run(ctx)
        adapter.commit(ctx)
    """

    def __init__(
        self,
        config: dict,
        models: dict,
        broker: Any,
        strategy_dir: Path,
        sell_only: bool = False,
        use_intraday_prices: bool = False,
    ) -> None:
        self._config              = config
        self._models              = models
        self._broker              = broker
        self._strategy_dir        = strategy_dir
        self._sell_only           = sell_only
        self._use_intraday_prices = use_intraday_prices
        self._universe_rejections = dict(
            config.get("_universe_rejections") or {}
        )

        # 2026-04-27: broker-isolated state. paper / alpaca-paper / alpaca
        # each get their own live_state.{broker}.json + runs.{broker}.db so
        # a paper smoke can never contaminate alpaca-live state. See
        # kernel/state_paths.py for the path convention.
        # 2026-04-28 self-audit (TEST-2 follow-up): require str to avoid
        # Mock objects in tests (or any non-str caller) tripping the
        # allowlist check inside state_paths._safe_broker.
        _bn = getattr(broker, "broker_name", None)
        self._broker_name: str | None = _bn if isinstance(_bn, str) else None

        # Mutate config.persistence.db_path to broker-specific BEFORE
        # constructing the DB connection (kernel.persistence reads it).
        if self._broker_name:
            from kernel.state_paths import runs_db_path  # noqa: PLC0415
            persist_cfg = config.setdefault("persistence", {})
            base_db = persist_cfg.get("db_path", "data/runs.db")
            persist_cfg["db_path"] = str(runs_db_path(base_db, self._broker_name))

        from kernel.persistence import get_connection  # noqa: PLC0415
        self._db = get_connection(config, strategy_dir=strategy_dir)

        # ── Meta-label snapshot logger (P5, 2026-05-11) ────────────────────
        # Mirror of SimAdapter wiring. Owned by adapter so it persists
        # across bars. Attached to ctx in make_context(); MetaLabelLoggingJob
        # writes one row per held ticker per bar; dumped at runner
        # teardown. Disabled (None) when meta_label_training.enabled is
        # false — i.e. always disabled in prod, only ON during a
        # dedicated training data-capture run (intraday cron or research
        # batch).
        ml_train_cfg = config.get("meta_label_training", {}) or {}
        if ml_train_cfg.get("enabled", False):
            from kernel.meta_label import SnapshotLogger  # noqa: PLC0415
            self._meta_label_logger = SnapshotLogger()
            self._meta_label_output_path = str(
                ml_train_cfg.get("output_path", "data/position_day_snapshots.parquet")
            )
        else:
            self._meta_label_logger = None
            self._meta_label_output_path = None

        # ── Meta-label veto predictor (P5, 2026-05-11) ────────────────────
        # Loads the XGBoost classifier trained by scripts/_meta_label_train.py
        # and exposes a `predictor(feats: dict) -> P(profitable_exit)`
        # callable that MetaLabelVetoTask queries to drop false-positive
        # path-rule exits. This is the PROD deployment surface for the
        # meta-label mechanism — same artifact format / fallback contract
        # as SimAdapter so models trained in sim research deploy cleanly
        # to live without code change.
        veto_cfg = (config.get("ranking") or {}).get("meta_label") or {}
        if veto_cfg.get("enabled", False):
            from kernel.meta_label.predictor import load_meta_label_predictor  # noqa: PLC0415
            art_path = veto_cfg.get(
                "artifact_path",
                "backtesting/renquant_104/artifacts/meta-label-exit.json",
            )
            art_resolved = Path(art_path)
            if not art_resolved.is_absolute():
                art_resolved = Path(strategy_dir).parent.parent / art_resolved
            self._meta_label_predictor = load_meta_label_predictor(art_resolved)
        else:
            self._meta_label_predictor = None

    # ── make_context ───────────────────────────────────────────────────────────

    def make_context(self):  # noqa: ANN201
        """Build InferenceContext from broker, parquet cache, and live_state.json."""
        from kernel.pipeline.context import InferenceContext  # noqa: PLC0415
        from kernel.regime import RegimeState                  # noqa: PLC0415
        from kernel.config import REGIMES                      # noqa: PLC0415

        config  = self._config
        today   = datetime.date.today()
        broker  = self._broker

        # ── Load persisted live state ────────────────────────────────────────
        # Plan #144 (2026-04-26 round-7): db is canonical, JSON is cache.
        # Read JSON first (fast). On JSON missing/corrupt, fall back to
        # the latest live_state_snapshots row (per-bar mirror).
        # Per user spec: "live state json应该至少备份在db里" — db wins
        # on conflict between JSON and db.
        from kernel.state_paths import resolve_live_state_read  # noqa: PLC0415
        state_file, used_legacy = resolve_live_state_read(
            self._strategy_dir, self._broker_name,
        )
        if used_legacy:
            log.warning(
                "BROKER-ISOLATION: live_state.{broker}.json missing for "
                "broker=%s — reading legacy live_state.json (one-time "
                "migration fallback). Future writes go to broker-specific "
                "path. Verify this state belongs to broker '%s'.",
                self._broker_name, self._broker_name,
            )
        state: dict = {}
        json_loaded = False
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text()) or {}
                json_loaded = True
            except (json.JSONDecodeError, OSError) as exc:
                log.warning(
                    "live_state read failed (%s) — falling back to db",
                    exc,
                )
        if not json_loaded:
            try:
                from kernel.persistence import (  # noqa: PLC0415
                    get_connection, load_latest_live_state,
                )
                conn = get_connection(config, strategy_dir=self._strategy_dir)
                strategy_name = config.get("_strategy_name", "renquant_104")
                # max_age_days=14 — defensive: don't resurrect ancient state
                # (e.g. from a 6-month-old test db). 14d aligns with the
                # max plausible gap before a sim/restore is needed.
                db_state = load_latest_live_state(
                    conn, strategy=strategy_name, max_age_days=14,
                )
                if db_state:
                    log.warning(
                        "RESTORE-FROM-DB (#144): live_state.json missing/"
                        "corrupt — restored from live_state_snapshots "
                        "(strategy=%s). Writing JSON cache now.",
                        strategy_name,
                    )
                    state = db_state
                    # Write the recovered state back to JSON so subsequent
                    # bars see a hot cache (no need to re-query db).
                    try:
                        state_file.write_text(json.dumps(state, default=str))
                    except OSError as exc:
                        log.warning(
                            "RESTORE-FROM-DB: JSON write-back failed (%s) "
                            "— state recovered in-memory only", exc,
                        )
            except Exception as exc:
                log.warning(
                    "RESTORE-FROM-DB: db load failed (%s) — proceeding "
                    "with empty state", exc,
                )

        entry_dates     = state.get("entry_dates",     {})
        sell_streaks    = state.get("sell_streaks",    {})
        last_sell_dates = state.get("last_sell_dates", {})
        # G8 (2026-05-04): per-ticker date when a path-rule exit
        # (trailing_stop / stop_loss / single_day_loss / max_hold / gap_down)
        # last fired. Persisted across runs so restart-after-stop honours
        # the cooldown. Read by PostStopCooldownFilterTask.
        last_stop_exit_dates = state.get("last_stop_exit_dates", {})
        position_hwm    = state.get("position_hwm",    {})
        # Thesis-degradation baselines (Approach A) — per-ticker
        # {rank_score, panel_score, kelly_target_pct} stamped at buy.
        entry_signals   = state.get("entry_signals",   {})
        # Z9 (2026-04-28): broker-side stop orders. Per-ticker
        # {order_id, stop_price, qty, stamped_at}. GTC at the broker
        # so they survive cron restarts. Local cache; broker.get_open_orders()
        # is the source of truth (we reconcile on every commit).
        stop_orders     = state.get("stop_orders",     {})
        # Runner-submitted SELL order ids, persisted across invocations so a
        # later reconciliation pass can tell its OWN exit fills apart from a
        # genuinely external/manual disposition. Maps
        # ``order_id -> {ticker, exit_type, qty, submitted_at}``. GC'd to a
        # 5-day window in commit() (matches the STATE-EXT-SELL fill lookback).
        # 2026-06-03 HON incident: a runner single_day_loss sell filled, then
        # the next tick's reconciler mislabeled it source=external_or_manual
        # because only Z9 stop order_ids were tracked.
        recent_sell_orders = state.get("recent_sell_orders", {}) or {}
        hwm             = float(state.get("high_water_mark", 0.0))
        # Persisted RegimeState across live runs. Without this, each fresh
        # `daily_104.sh` invocation starts countdown=0 → CUSUM re-trips every
        # bar → `transition_window` stays True forever → buys perpetually
        # blocked. (Sim doesn't hit this because state lives in-process.)
        regime_persist  = state.get("regime_state", {}) or {}

        # ── Broker account ───────────────────────────────────────────────────
        account_value = broker.get_account_value()
        # 2026-05-04 audit Issue 36 fix: silent broker.get_cash() failure
        # used to default cash := account_value (= total NAV including
        # held positions). Downstream sizing then thought ALL of NAV was
        # liquid and could over-allocate. Fail-SAFE: fall back to ZERO
        # cash (the safest assumption — no fresh buys this bar) and log
        # loud so operator knows broker is partially down.
        try:
            cash = broker.get_cash()
        except Exception as _cash_exc:
            log.error(
                "runner: broker.get_cash() failed (%s: %s) — "
                "fail-SAFE setting cash=0 for this bar to prevent "
                "over-allocation. Pre-fix this defaulted to account_value "
                "(= total NAV) which silently allowed Kelly oversizing.",
                type(_cash_exc).__name__, _cash_exc,
            )
            cash = 0.0

        # Stale-HWM guard (see `resolve_hwm` docstring above). Snaps when
        # stored HWM is wildly above current equity, preserves normal
        # drawdowns otherwise.
        hwm, snapped = resolve_hwm(hwm, account_value)
        if snapped:
            log.warning("Stale HWM: snapped to current equity=%.2f "
                        "(stored HWM was stale; see adapters.runner.resolve_hwm)",
                        account_value)

        try:
            all_pos = broker.get_all_positions()
        except Exception:
            all_pos = []
        positions_cache = {p["symbol"]: p for p in all_pos}

        # Audit fix BROKER-PRECHECK (2026-04-26): pre-fetch broker's
        # currently open / pending orders ONCE per bar. Pre-fix, the
        # adapter called broker.get_open_orders() per-order at submit
        # time — N API calls per bar, AND the pipeline didn't know
        # which tickers were going to be rejected as duplicates BEFORE
        # sizing, so the cash budget assumed all orders fillable. The
        # e2e on 2026-04-26 03:20 showed 4 buys queued, 2 rejected as
        # duplicates → cash spent ≠ cash planned. Now: snapshot once,
        # inject into ctx, let upstream tasks (joint mode, selection,
        # rotation) skip these tickers BEFORE sizing.
        pending_broker_tickers: set[str] = set()
        try:
            pending_broker_tickers = set(broker.get_open_orders() or [])
            if pending_broker_tickers:
                log.info(
                    "BROKER-PRECHECK: %d pending order(s) at broker → "
                    "excluded from buy/rotate menus this bar: %s",
                    len(pending_broker_tickers),
                    sorted(pending_broker_tickers),
                )
        except Exception as exc:
            log.warning(
                "BROKER-PRECHECK: get_open_orders failed (%s) — "
                "duplicate-order guard falls back to per-order check at submit",
                exc,
            )

        # Audit fix ENTRY-DATE-FROM-FILLS (Round 4 deep audit, 2026-04-25):
        # Pre-fix, an inherited position with no entry_date in state got
        # stamped to TODAY (line ~191). Result: a position bought 60 days
        # ago was treated as fresh → min_hold_days=30 lockout started NOW
        # → user's old position couldn't be sold by the model for another
        # 30 days. Now: query broker fill history once per cycle, build a
        # ticker → earliest-BUY-fill-date map; use it as the seed for
        # missing entry_dates so hold tenure reflects actual cost-basis
        # tenure, not "first time the runner saw this position".
        first_fill_map: dict[str, datetime.date] = {}
        broker_fills: list[dict[str, Any]] = []
        try:
            fills = broker.get_filled_orders()
            broker_fills = list(fills or [])
            for f in fills or []:
                sym = f.get("symbol")
                if not sym or f.get("action") != "BUY":
                    continue
                fa = f.get("filled_at")
                if not fa:
                    continue
                try:
                    d = datetime.date.fromisoformat(str(fa)[:10])
                except (ValueError, TypeError):
                    continue
                # Earliest BUY for this symbol — only updated when no
                # SELL has happened in between (we don't currently track
                # the trip-lifecycle here; conservative: take the OLDEST
                # buy date so min_hold gives the position max benefit of
                # the doubt).
                if sym not in first_fill_map or d < first_fill_map[sym]:
                    first_fill_map[sym] = d
        except (AttributeError, NotImplementedError, Exception) as exc:
            log.info("ENTRY-DATE-FROM-FILLS: broker.get_filled_orders unavailable "
                     "(%s) — will fall back to sentinel for missing entry dates",
                     type(exc).__name__)
        live_tax_lots = reconstruct_live_tax_lots_from_fills(
            broker_fills,
            config=config,
        )

        # ── Holdings from live state + broker positions ─────────────────────
        from kernel.exits import HoldingState  # noqa: PLC0415

        from kernel.pipeline.task_benchmark_sleeve import (  # noqa: PLC0415
            benchmark_sleeve_ticker,
            decision_trace_tickers,
            is_benchmark_sleeve_enabled,
        )
        managed_symbols = list(config["watchlist"])
        sleeve_ticker = benchmark_sleeve_ticker(config)
        if is_benchmark_sleeve_enabled(config) and sleeve_ticker not in managed_symbols:
            managed_symbols.append(sleeve_ticker)
        held_set  = set(s for s in managed_symbols
                        if float(positions_cache.get(s, {}).get("qty", 0)) > 0)
        # Audit #59: log positions held outside the watchlist so the operator
        # knows they exist (the runner won't manage them — exits/buys only
        # apply to watchlist symbols — but silent invisibility is worse).
        non_wl_holds = [
            s for s, pos in positions_cache.items()
            if s not in held_set
            and s not in managed_symbols
            and float(pos.get("qty", 0)) > 0
        ]
        if non_wl_holds:
            log.warning("RunnerAdapter: %d position(s) held outside watchlist "
                        "(unmanaged): %s",
                        len(non_wl_holds), ", ".join(sorted(non_wl_holds)))
        # Audit fix UNMANAGED-NTFY (Round 4 deep audit, 2026-04-25): surface
        # non_wl_holds via ctx so live/runner.py::_notify_decision can include
        # an UNMANAGED line on the operator's phone — pre-fix, this was a
        # log-only warning that the user only saw if they opened the log file.
        # Real positions could sit in the broker for weeks with stop-loss /
        # trailing-stop never firing because the strategy doesn't know they
        # exist.
        self._non_wl_holds = list(sorted(non_wl_holds))
        holdings: dict[str, HoldingState] = {}
        for ticker in held_set:
            pos     = positions_cache.get(ticker, {})
            avg_cost = float(pos.get("avg_entry_price", 0.0))
            hwm_pos  = float(position_hwm.get(ticker, avg_cost))
            # entry_dates lookup with **persistent fallback**: if a position
            # is held but missing from entry_dates (e.g. inherited from
            # renquant_103 or manually added), stamp today and persist so
            # hold_days is measured from first-sighting, not from
            # today-minus-today (which made hold_days=0 forever → locked all
            # min_hold_days / rotation gates). Ideally seed from Alpaca's
            # fill timestamp on migration; today is the least-bad fallback.
            # Audit fix ENTRY-DATE-FROM-FILLS / ENTRY-DATE-BACKFILL
            # (Bug C extended, 2026-04-25): the broker's first BUY-fill
            # date is the AUTHORITATIVE entry date (cost-basis tenure
            # from Alpaca). Three cases:
            #   1. State has no entry_date → seed from broker fill if
            #      available, else use sentinel (31d ago).
            #   2. State has entry_date but broker shows OLDER first
            #      BUY → broker is correct (state was wrongly stamped
            #      "today" by a prior runner that didn't have ENTRY-DATE-
            #      FROM-FILLS). Override state with broker's earlier date.
            #      This unlocks min_hold_days / rotation tenure that the
            #      stale state was artificially extending.
            #   3. State has entry_date and it matches/predates broker →
            #      preserve (handles top-ups + cost-basis-fifo cases).
            broker_first = first_fill_map.get(ticker)
            if ticker not in entry_dates:
                if broker_first is not None:
                    entry_dates[ticker] = broker_first.isoformat()
                    log.info("ENTRY-DATE-SEED %s ← %s (broker fill history)",
                             ticker, broker_first.isoformat())
                else:
                    sentinel = today - datetime.timedelta(days=31)
                    entry_dates[ticker] = sentinel.isoformat()
                    log.warning("ENTRY-DATE-SEED %s ← %s (sentinel — broker had no "
                                "fill history; manual fix recommended)",
                                ticker, sentinel.isoformat())
            else:
                # Backfill: broker authority overrides stale state when older.
                if broker_first is not None:
                    try:
                        cur_entry = datetime.date.fromisoformat(entry_dates[ticker])
                    except (ValueError, TypeError):
                        cur_entry = today
                    if broker_first < cur_entry:
                        log.info("ENTRY-DATE-BACKFILL %s: state=%s → broker=%s "
                                 "(broker fill is older — stale state corrected)",
                                 ticker, entry_dates[ticker], broker_first.isoformat())
                        entry_dates[ticker] = broker_first.isoformat()
            entry_str = entry_dates[ticker]
            try:
                entry_dt = datetime.date.fromisoformat(entry_str)
            except ValueError:
                entry_dt = today
                entry_dates[ticker] = today.isoformat()
            # Audit fix QTY-NaN-HYDRATE (Round 2 deep audit, 2026-04-25):
            # broker NaN qty during a snapshot race would make
            # HoldingState.shares = NaN, then propagate into Kelly
            # current_pct calc. Downstream TopUp/Trim now have isfinite
            # guards (TU/TR-NaN) but cleaner to sanitize at hydration.
            import math as _math
            _qty_raw = float(pos.get("qty", 0))
            qty_held = _qty_raw if _math.isfinite(_qty_raw) else 0.0
            # Thesis-degradation baselines (Approach A) — hydrate from
            # persisted entry_signals. Missing keys → None, which the
            # rotation criterion treats as "no baseline, fall back to
            # legacy rule".
            es = entry_signals.get(ticker, {}) if isinstance(entry_signals.get(ticker, {}), dict) else {}
            holdings[ticker] = HoldingState(
                entry_price    = avg_cost,
                entry_date     = entry_dt,
                high_watermark = hwm_pos,
                sell_streak    = int(sell_streaks.get(ticker, 0)),
                shares         = qty_held,   # broker qty for Kelly top-up sizing
                entry_rank_score       = es.get("rank_score"),
                entry_panel_score      = es.get("panel_score"),
                entry_kelly_target_pct = es.get("kelly_target_pct"),
                entry_regime           = es.get("regime"),
            )
            lots = live_tax_lots.get(ticker)
            if lots:
                lot_qty = sum(float(getattr(L, "shares", 0.0) or 0.0) for L in lots)
                if abs(lot_qty - qty_held) <= max(0.01, abs(qty_held) * 1e-4):
                    holdings[ticker].lots = lots
                    holdings[ticker].entry_price = (
                        holdings[ticker].weighted_avg_entry_price()
                    )
                else:
                    log.warning(
                        "LIVE-TAX-LOTS: %s reconstructed lot qty %.4f != broker "
                        "qty %.4f; using broker avg_entry_price fallback",
                        ticker, lot_qty, qty_held,
                    )
            holdings[ticker].model_type = model_type_from_artifact(
                self._models.get(ticker)
            )
            sector = config.get("sector_map", {}).get(ticker)
            if isinstance(sector, str) and sector:
                holdings[ticker].sector = sector

        # ── Current prices from broker positions ────────────────────────────
        # 2026-05-09 audit fix (RU-PRICE-1): pre-fix `if qty > 0 and mkt > 0`
        # passed for micro-qty (e.g. 1e-7 fractional shares from a botched
        # broker fill) → `mkt / qty` produced an inflated price (e.g.
        # market_value=$100, qty=1e-6 → price=$100M/share). Guard with
        # isfinite + a 1-share floor so we treat sub-share dust as "no
        # trustworthy price" and fall back to OHLCV close below.
        #
        # Full daily runs must not mix real-time broker marks for held symbols
        # with daily OHLCV closes for candidates. Keep broker marks only for
        # sell-only/intraday risk checks or as an OHLCV-missing fallback.
        import math as _math_p  # noqa: PLC0415
        prices: dict[str, float] = {}
        broker_mark_prices: dict[str, float] = {}
        for ticker, pos in positions_cache.items():
            qty = float(pos.get("qty", 0))
            mkt = float(pos.get("market_value", 0))
            if (_math_p.isfinite(qty) and _math_p.isfinite(mkt)
                    and qty >= 0.5 and mkt > 0):
                px = mkt / qty
                if _math_p.isfinite(px) and 0 < px < 1e6:
                    broker_mark_prices[ticker] = px
                    if self._sell_only or self._use_intraday_prices:
                        prices[ticker] = px

        # ── OHLCV from parquet cache ─────────────────────────────────────────
        from kernel.data import fetch_ohlcv  # noqa: PLC0415

        watchlist   = config["watchlist"]
        benchmark   = config.get("benchmark", "SPY")
        sector_etfs = set(config.get("sector_etf_map", {}).values())
        extra_symbols = []
        if is_benchmark_sleeve_enabled(config) and sleeve_ticker:
            extra_symbols.append(sleeve_ticker)
        all_symbols = list(dict.fromkeys(
            watchlist
            + [benchmark]
            + sorted(sector_etfs)
            + sorted(held_set)
            + extra_symbols
        ))

        ohlcv: dict[str, Any] = {}
        for sym in all_symbols:
            try:
                df = fetch_ohlcv(sym)
                if not df.empty:
                    ohlcv[sym] = df
                    # Fill prices from OHLCV last close if broker didn't supply.
                    # 2026-05-09 audit fix (RU-PRICE-2): isfinite guard on
                    # the close value. Pre-fix, a NaN close in the last bar
                    # (data-feed glitch on suspended/halted ticker) silently
                    # propagated into ctx.prices → Kelly/HWM/QP all received
                    # NaN → cascade of silent failures.
                    if (not self._sell_only and not self._use_intraday_prices) or sym not in prices:
                        close_val = float(df["close"].iloc[-1])
                        if _math_p.isfinite(close_val) and close_val > 0:
                            prices[sym] = close_val
                        else:
                            log.warning(
                                "OHLCV close for %s is non-finite (%s) — "
                                "skipping price entry; downstream tasks will "
                                "see ticker as 'no price' (fail-safe).",
                                sym, close_val,
                            )
            except Exception as exc:
                log.warning("OHLCV fetch failed for %s: %s", sym, exc)

        if not (self._sell_only or self._use_intraday_prices):
            for sym, px in broker_mark_prices.items():
                prices.setdefault(sym, px)

        # ── Intraday price overlay (for intraday sell-only checks) ──────────
        if self._use_intraday_prices:
            try:
                from kernel.data import fetch_intraday_bars  # noqa: PLC0415
                ibars = fetch_intraday_bars(
                    list(all_symbols),
                    timeframe="5Min",
                    start=datetime.datetime.combine(
                        today, datetime.datetime.min.time(),
                    ),
                )
                overlaid = 0
                for sym, idf in ibars.items():
                    if idf is None or idf.empty:
                        continue
                    latest_close = float(idf["close"].iloc[-1])
                    # 2026-05-09 audit fix (RU-INTRADAY-NaN): same guard as
                    # OHLCV daily path — non-finite intraday close (5Min bar
                    # gap on halted ticker) silently corrupted ctx.prices.
                    if not _math_p.isfinite(latest_close) or latest_close <= 0:
                        log.warning("Intraday close for %s non-finite (%s) — skipping",
                                     sym, latest_close)
                        continue
                    prices[sym] = latest_close
                    # Overwrite today's daily bar's close so kernel.exits sees the intraday level.
                    # Audit #58: copy the frame before mutating — fetch_ohlcv may
                    # return a cached reference that other downstream calls (sim,
                    # training, panel features) would see leak via the in-place
                    # write. The sliced copy in `ohlcv` is what the pipeline reads.
                    if sym in ohlcv and not ohlcv[sym].empty:
                        df = ohlcv[sym].copy()
                        last_day = df.index.max()
                        if last_day.date() == today:
                            df.at[last_day, "close"] = latest_close
                            ohlcv[sym] = df
                    overlaid += 1
                log.info("Intraday overlay: %d/%d symbols had fresh minute bars",
                         overlaid, len(all_symbols))
            except Exception as exc:
                log.warning("Intraday overlay failed — falling back to daily closes: %s", exc)

        spy_df = ohlcv.get(benchmark)
        spy_returns: list[float] = []
        if spy_df is not None:
            spy_close   = spy_df["close"].astype(float)
            spy_returns = list(spy_close.pct_change().dropna().values[-100:])

        # ── Load artifacts ───────────────────────────────────────────────────
        from kernel.regime import load_gmm_artifact  # noqa: PLC0415
        from kernel.config import artifact_path       # noqa: PLC0415

        regime_cfg = config.get("regime", {})
        artifacts_dir = self._strategy_dir / "artifacts"
        if not artifacts_dir.exists():
            artifacts_dir = self._strategy_dir

        # 2026-05-11 sim/prod isolation: defaults relocated to prod/.
        # Sim configs override these keys to sim/<file>.
        gmm_path  = artifacts_dir / regime_cfg.get("gmm_artifact", "prod/spy-gmm-regime.json")
        gmm       = load_gmm_artifact(gmm_path)

        corr_path = artifacts_dir / regime_cfg.get("correlation_artifact", "prod/watchlist-correlation.json")
        # 2026-05-09 audit fix (RU-JSON-MALFORMED): pre-fix, malformed JSON
        # in corr/earnings artifacts raised JSONDecodeError straight up
        # → adapter __init__ crashed → live trade aborted with cryptic
        # traceback. Now: malformed file logged + treated as missing
        # (downstream tasks already handle None gracefully).
        #
        # 2026-05-10 audit §5.13.5: also unwrap v2-schema correlation
        # artifact (matrix + as_of_date). Live runner is in live mode,
        # so the leakage guard is a no-op — but we still parse the v2
        # schema correctly so downstream `corr.get(t, {})` keeps working.
        from kernel.walk_forward import parse_correlation_artifact  # noqa: PLC0415
        try:
            corr_raw = json.loads(corr_path.read_text()) if corr_path.exists() else None
            corr, _corr_as_of = parse_correlation_artifact(corr_raw)
            if not corr:
                corr = None
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("corr artifact %s malformed (%s) — treating as missing", corr_path, exc)
            corr = None

        # 2026-05-11 sim/prod isolation + audit fix (was hardcoded, now config-driven).
        earn_path = artifacts_dir / regime_cfg.get("earnings_artifact", "prod/earnings-calendar.json")
        try:
            earnings = json.loads(earn_path.read_text()) if earn_path.exists() else None
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("earnings artifact %s malformed (%s) — treating as missing", earn_path, exc)
            earnings = None

        # Convert last_sell_dates strings to date objects for kernel.selection guards
        last_sells_d: dict[str, datetime.date | None] = {}
        for sym, d_str in last_sell_dates.items():
            try:
                last_sells_d[sym] = datetime.date.fromisoformat(d_str)
            except (ValueError, TypeError):
                last_sells_d[sym] = None

        # 2026-05-09 cost-aware wash-sale: compute realized $ P/L per
        # ticker for the most-recent full liquidation in the last 30d.
        # Used by WashSaleFilterTask to skip block on GAIN sales (§1091
        # rule does not apply) and to compute NPV cost on loss sales.
        last_sells_pl: dict[str, float | None] = {}
        try:
            from kernel.realized_pnl import compute_recent_realized_pnl  # noqa: PLC0415
            last_sells_pl = compute_recent_realized_pnl(
                self._broker, days=int(config.get("wash_sale_days", 30)) + 5,
            )
            n_gains = sum(1 for v in last_sells_pl.values() if v is not None and v >= 0)
            n_losses = sum(1 for v in last_sells_pl.values() if v is not None and v < 0)
            log.info("realized_pnl: %d gains + %d losses in last %dd  (gains skip wash-sale block)",
                     n_gains, n_losses, int(config.get("wash_sale_days", 30)) + 5)
        except Exception as exc:
            log.warning("realized_pnl compute failed: %s — wash-sale falls back to binary", exc)
            last_sells_pl = {}
        # G8: same coercion for stop-exit dates
        last_stops_d: dict[str, datetime.date | None] = {}
        for sym, d_str in (last_stop_exit_dates or {}).items():
            try:
                last_stops_d[sym] = datetime.date.fromisoformat(d_str)
            except (ValueError, TypeError):
                last_stops_d[sym] = None

        # ── Persisted live state on context for commit() ─────────────────────
        self._state          = state
        self._entry_dates    = entry_dates
        self._entry_signals  = entry_signals   # Approach A — persisted per-ticker
        self._sell_streaks   = sell_streaks
        self._last_sell_dates_str = last_sell_dates
        self._last_stop_exit_dates_str = dict(last_stop_exit_dates or {})
        self._position_hwm   = position_hwm
        self._stop_orders    = stop_orders     # Z9: per-ticker stop_order metadata
        self._recent_sell_orders = recent_sell_orders  # runner-submitted SELL order_ids
        self._positions_cache = positions_cache
        self._account_value  = account_value

        ctx = InferenceContext(
            config            = config,
            today             = today,
            run_timestamp     = datetime.datetime.now().astimezone(),
            broker_name       = self._broker_name,
            ohlcv             = ohlcv,
            spy_returns       = spy_returns,
            models            = self._models,
            gmm               = gmm,
            corr_matrix       = corr,
            earnings_calendar = earnings,
            holdings          = holdings,
            last_sell_dates   = last_sells_d,
            last_sell_pls     = last_sells_pl,
            last_stop_exit_dates = last_stops_d,
            portfolio_value   = account_value,
            cash              = cash,
            prices            = prices,
            hwm               = hwm,
            skip_buys         = persisted_skip_buys(state),
            regime_state      = RegimeState(
                regime        = regime_persist.get("regime",        "BULL_CALM"),
                confidence    = float(regime_persist.get("confidence",     0.5)),
                in_transition = bool(regime_persist.get("in_transition", False)),
                countdown     = int(regime_persist.get("countdown",          0)),
                cusum_pos     = float(regime_persist.get("cusum_pos",      0.0)),
                cusum_neg     = float(regime_persist.get("cusum_neg",      0.0)),
                # CUSUM-v2 Design C — restore wall-clock cooldown start.
                # Parse ISO string; None / missing → no cooldown active.
                cooldown_start = _parse_iso_dt(regime_persist.get("cooldown_start")),
            ),
            regime_counts     = {r: 0 for r in REGIMES},
            monitor_state     = dict(state.get("monitor_state", {}) or {}),
        )
        ctx.run_id = f"{today.isoformat()}-live-{uuid.uuid4().hex[:8]}"
        ctx.supports_short_open = False
        ctx._run_type = "live"  # noqa: SLF001

        # Bug 11 fix (2026-04-24): Rotation V4 (thesis_symmetric scoring
        # mode) needs ctx._db to look up candidate scores on each held's
        # entry date via lookup_candidate_scores_on_date. Previously this
        # was wired only on SimAdapter; without it, V4 silently no-ops on
        # the live runner path. RunnerAdapter writes to runs.db (live);
        # rotation V4 reads from there for entry-day score lookup.
        if self._db is not None:
            ctx._db = self._db   # noqa: SLF001

        # UNMANAGED-NTFY: pass through to ntfy decision-summary path.
        ctx.non_wl_holds = list(self._non_wl_holds)

        # BROKER-PRECHECK (2026-04-26): expose pending broker orders so
        # JointActionTask + SelectionJob + RotationJob can pre-filter
        # candidates BEFORE sizing.
        ctx.pending_broker_tickers = pending_broker_tickers

        # Rotation V1 persistence gate — live runner has no per-bar
        # state file pinned to rotation_proposals (yet); seed with empty
        # so the gate fails-closed when persistence_bars > 0. (When the
        # user enables persistence in production we'll wire it through
        # live_state.json — same pattern as monitor_state.)
        ctx.prior_rotation_proposals = list(
            state.get("rotation_proposals", []) or []
        )

        # Run-local feature cache (2026-05-25): live/shadow should use the
        # same causal feature surface as sim. The OHLCV has already passed the
        # live freshness guard; caching it for this one run avoids rebuilding
        # indicators separately in sell and candidate jobs.
        if config.get("live", {}).get("feature_cache_enabled", True):
            ctx.feature_cache = build_runtime_feature_cache(
                config=config,
                ohlcv=ohlcv,
            )
            log.info("Feature cache attached to live context: %d tickers",
                     len(ctx.feature_cache))

        # ── Panel scoring prep (optional) ────────────────────────────────────
        panel_cfg = config.get("ranking", {}).get("panel_scoring", {})
        if panel_cfg.get("enabled", False) and not self._sell_only:
            try:
                bundle = prepare_panel_runtime_frames(
                    config=config,
                    ohlcv=ohlcv,
                )
                attach_panel_runtime_frames(ctx, bundle)
                n_ff, n_fac, macro_desc, n_emb = describe_panel_frame_bundle(bundle)
                log.info("Panel frames prepared: feat=%d  factor=%d  macro=%s  emb=%d",
                         n_ff, n_fac, macro_desc, n_emb)
            except Exception as exc:
                msg = (
                    "Panel frame prep failed while panel_scoring.enabled=true; "
                    "aborting live inference instead of silently trading without "
                    f"panel scores: {exc}"
                )
                log.error(msg)
                raise RuntimeError(msg) from exc

        # ── P5 (2026-05-11) attach meta-label hooks ──────────────────────
        # snapshot_logger: None unless meta_label_training.enabled
        #   (training-data capture mode — typically OFF in prod)
        # _meta_label_predictor: None unless ranking.meta_label.enabled +
        #   artifact loaded. The MetaLabelVetoTask in pp_inference.py
        #   reads this to drop false-positive path-rule exits at live time.
        ctx.snapshot_logger = self._meta_label_logger
        ctx._meta_label_predictor = self._meta_label_predictor  # noqa: SLF001

        return ctx

    # ── Z9: broker-side stop helpers ────────────────────────────────────────
    # Invariants:
    #   • stops live broker-side (GTC), not in our polling loop
    #   • single stop per ticker; stamped at BUY, replaced on TOPUP, cancelled
    #     on full SELL or external disposition (Z2 STATE-EXT-SELL)
    #   • new_stop_price ≤ existing_stop_price (never loosen on TOPUP)
    #   • disabled by default; enable via live.broker_side_stops.enabled=true
    #   • broker must support — silently skipped if broker.supports_broker_side_stops()=false

    def _z9_enabled(self, ctx) -> bool:  # noqa: ANN001
        cfg = ctx.config.get("live", {}).get("broker_side_stops", {})
        if not cfg.get("enabled", False):
            return False
        broker = self._broker
        if not getattr(broker, "supports_broker_side_stops", lambda: False)():
            log.debug("Z9: broker %s does not support broker-side stops — skip",
                      type(broker).__name__)
            return False
        return True

    @staticmethod
    def _z9_stop_pct(ctx) -> float:  # noqa: ANN001
        """Per-regime intraday loss cap. Default 6% in BULL_*/CHOPPY (set
        2026-04-28 after NVTS post-mortem). BEAR=0 means no buys, so no
        stops needed."""
        regime_p = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        return float(regime_p.get("max_single_day_loss_pct", 0.06))

    def _override_no_trade_streak_from_broker(self, ctx) -> None:  # noqa: ANN001
        """Replace stateful no_trade_streak counter with broker-derived truth.

        Stateful counter has bug surface: per-invocation vs per-day inflation,
        SIGKILL mid-write corrupting live_state.json, race between intraday
        SellOnly and daily full pipeline writing the same field. Real source
        of truth = the broker's order book. Logs both for cross-validation.

        2026-05-20: introduced after the per-trading-day fix exposed how
        much state-file divergence had accumulated (counter=32 while LIVE
        Alpaca had fills on 16 of last 25 trading days).
        """
        if not hasattr(self._broker, "get_filled_orders"):
            return
        from datetime import date, timedelta  # noqa: PLC0415
        import datetime as _dt  # noqa: PLC0415
        from kernel.exits import _is_nyse_trading_day  # noqa: PLC0415

        # Look back ~120 calendar days = enough cushion to detect very long
        # idle periods without paging more than necessary.
        today = ctx.today if isinstance(ctx.today, date) else _dt.date.today()
        after = (today - timedelta(days=120)).isoformat()
        try:
            fills = self._broker.get_filled_orders(after=after)
        except Exception as exc:
            log.warning("broker.get_filled_orders failed: %s", exc)
            return

        fill_dates: set[date] = set()
        for f in fills:
            iso = f.get("filled_at")
            if not iso:
                continue
            try:
                # Tolerate Z, +HH:MM, naive ISO
                if iso.endswith("Z"):
                    iso = iso[:-1] + "+00:00"
                fill_dates.add(_dt.datetime.fromisoformat(iso).date())
            except Exception:
                continue

        if not fill_dates:
            broker_streak = 120  # capped at lookback
            most_recent: date | None = None
        else:
            most_recent = max(fill_dates)
            # Count NYSE trading days strictly between most_recent and today.
            broker_streak = 0
            d = most_recent + timedelta(days=1)
            while d <= today:
                if _is_nyse_trading_day(d):
                    broker_streak += 1
                d += timedelta(days=1)

        mon = self._state.setdefault("monitor_state", {}) or {}
        counter_streak = int(mon.get("no_trade_streak", 0))
        log.info(
            "no_trade_streak: broker-derived=%d  stateful-counter=%d  "
            "most_recent_fill=%s  fill_dates_in_window=%d",
            broker_streak, counter_streak,
            most_recent.isoformat() if most_recent else "none",
            len(fill_dates),
        )
        if broker_streak != counter_streak:
            # 2026-06-01: divergence is the EXPECTED outcome of this
            # architecture, not a bug. The stateful counter is
            # per-bar-incremented by MonitorIdleStreakTask using
            # `bool(ctx.orders) or bool(ctx.exits)` as the activity signal,
            # which misses externally-driven activity (broker-side stop
            # fills, manual closes, corporate actions surfaced via
            # STATE-EXT-SELL). Broker fill history is the §7.5 single
            # source of truth, and this method is the override that
            # enforces it. The log used to be WARNING with "Counter bug
            # or state corruption" wording — that mislabelled normal
            # operation as an incident. Downgraded to INFO with neutral
            # wording so the actual signal (broker truth = N) is visible
            # without ntfy alert noise.
            log.info(
                "no_trade_streak override: stateful-counter=%d  broker-truth=%d  "
                "(stateful counter misses ext-fills; broker history is canonical).",
                counter_streak, broker_streak,
            )
        mon["no_trade_streak"] = broker_streak
        mon["no_trade_streak_source"] = "broker_filled_orders"
        mon["last_fill_date"] = most_recent.isoformat() if most_recent else None
        # codex PR #84 review: this override used to clobber
        # ``last_activity_date`` and ``first_trade_date`` with broker truth
        # (any-source fills), wiping the runner-emission semantic that
        # MonitorIdleStreakTask had just written from ctx.orders/ctx.exits.
        # That made it impossible for a downstream consumer (e.g. the
        # P-BROKER-FILL-FRESHNESS preflight) to distinguish a manual /
        # Z9-only fill from a genuine runner-driven decision.
        #
        # Fix: the broker-truth fields stay on ``last_fill_date`` /
        # ``no_trade_streak`` (their canonical homes); the runner-emission
        # fields stay on ``last_activity_date`` / ``first_trade_date`` and
        # are NOT touched here. Consumers wanting "runner alpha" semantic
        # read the activity field; consumers wanting "any broker activity"
        # read ``last_fill_date`` / ``no_trade_streak`` (broker source).
        self._state["monitor_state"] = mon

    def _z9_place_or_replace_stop(
        self, ticker: str, qty: float, reference_price: float, today_str: str,
    ) -> None:
        """Place a stop at reference × (1 - pct). If a stop already exists for
        this ticker, cancel it first; the new stop_price is the MIN of
        (existing, new) so we never loosen.
        """
        # 2026-05-09 audit fix (Z9-NaN): pre-fix, NaN qty / reference_price
        # slipped past `<= 0` (NaN comparisons return False) → target=NaN
        # → broker.place_stop_order crashed inside int(qty). Same QTY-NaN
        # pattern as the exit-side audit fix. Now: explicit isfinite guard.
        import math as _math_z9  # noqa: PLC0415
        if (not _math_z9.isfinite(qty) or qty <= 0
                or not _math_z9.isfinite(reference_price) or reference_price <= 0):
            log.warning(
                "Z9: skipping stop for %s — non-finite or non-positive "
                "qty=%s reference_price=%s", ticker, qty, reference_price,
            )
            return
        broker = self._broker
        ctx_pct = getattr(self, "_last_ctx_stop_pct", 0.06)
        if not _math_z9.isfinite(ctx_pct) or ctx_pct <= 0 or ctx_pct >= 1:
            ctx_pct = 0.06
        target = reference_price * (1.0 - ctx_pct)
        if not _math_z9.isfinite(target) or target <= 0:
            log.warning("Z9: derived target=%s non-finite — skipping", target)
            return

        existing = self._stop_orders.get(ticker)
        if existing is not None:
            # Never loosen — pick the tighter of current vs proposed.
            target = min(target, float(existing.get("stop_price", target)))
            try:
                broker.cancel_order(existing.get("order_id", ""))
            except Exception as exc:
                log.warning("Z9: cancel existing stop %s for %s failed: %s",
                            existing.get("order_id"), ticker, exc)
            self._stop_orders.pop(ticker, None)

        try:
            result = broker.place_stop_order(ticker, qty, target)
        except Exception as exc:
            log.warning("Z9: place_stop_order(%s, qty=%s, stop=%.2f) failed: %s",
                        ticker, qty, target, exc)
            return
        self._stop_orders[ticker] = {
            "order_id":   result.get("order_id"),
            "stop_price": float(target),
            "qty":        float(qty),
            "stamped_at": today_str,
        }
        log.info("Z9: %s stop placed @ $%.2f × %s shares (order=%s)",
                 ticker, target, int(qty), result.get("order_id"))

    def _z9_cancel_stop(self, ticker: str, reason: str = "") -> None:
        """Cancel and forget the stop for a ticker. No-op if none exists."""
        existing = self._stop_orders.pop(ticker, None)
        if existing is None:
            return
        try:
            self._broker.cancel_order(existing.get("order_id", ""))
            log.info("Z9: cancelled stop %s for %s (%s)",
                     existing.get("order_id"), ticker, reason or "no reason")
        except Exception as exc:
            log.warning("Z9: cancel stop %s for %s failed: %s",
                        existing.get("order_id"), ticker, exc)

    # ── STATE-EXT-SELL fill attribution (issue #71 / audit #5) ────────────────

    # Codex #76: the two in-repo broker implementations of get_filled_orders
    # return DIFFERENT keys. Normalize through this schema map so the lookup
    # works on both AND any future broker that mostly follows one convention.
    #
    # umbrella live/alpaca_broker.py returns:
    #   symbol, action ("BUY"/"SELL"), qty, filled_at, avg_price, partial
    #   + order_id (added in this PR)
    #
    # renquant-execution/alpaca_broker.py returns:
    #   order_id, status, symbol, filled_qty, filled_avg_price,
    #   created_at, submitted_at, filled_at
    #   (no side/action — but status=="filled" means we don't know direction)
    _FILL_SIDE_KEYS  = ("side", "action")
    _FILL_PRICE_KEYS = ("avg_price", "fill_price", "filled_avg_price")
    _FILL_QTY_KEYS   = ("qty", "filled_qty", "fill_qty")
    _FILL_ID_KEYS    = ("order_id", "id")

    @staticmethod
    def _normalize_fill_record(f: dict) -> dict:
        """Project a broker-specific fill dict onto a uniform schema:

            {order_id, side ("sell"/"buy"/""), price, qty, filled_at}

        Returns ``side=""`` only when NO direction field is present at all
        — caller must then fail-closed and skip the row to avoid mistaking
        a buy for a sell."""
        side_raw = ""
        for key in RunnerAdapter._FILL_SIDE_KEYS:
            v = f.get(key)
            if v:
                side_raw = str(v).lower()
                break
        side = "sell" if "sell" in side_raw else ("buy" if "buy" in side_raw else "")
        price = None
        for key in RunnerAdapter._FILL_PRICE_KEYS:
            v = f.get(key)
            if v is not None:
                try:
                    pf = float(v)
                except (TypeError, ValueError):
                    continue
                if pf > 0:
                    price = pf
                    break
        qty = None
        for key in RunnerAdapter._FILL_QTY_KEYS:
            v = f.get(key)
            if v is not None:
                try:
                    qf = float(v)
                except (TypeError, ValueError):
                    continue
                if qf > 0:
                    qty = qf
                    break
        order_id = None
        for key in RunnerAdapter._FILL_ID_KEYS:
            v = f.get(key)
            if v:
                order_id = str(v)
                break
        return {
            "order_id":  order_id,
            "side":      side,
            "price":     price,
            "qty":       qty,
            "filled_at": str(f.get("filled_at") or ""),
        }

    def _lookup_ext_sell_fills(
        self,
        ctx,  # noqa: ANN001
        disappeared: list[str],
    ) -> dict[str, dict]:
        """Fetch the most recent broker SELL fill per disappeared ticker.

        Issue #71: STATE-EXT-SELL used to log only the ticker name. Now we
        correlate against ``broker.get_filled_orders`` so the operator sees
        WHICH sell fill emptied the position — Z9 stop, manual close, or
        corporate action?

        Codex #76: both in-repo brokers return DIFFERENT keys (umbrella uses
        ``action``+``avg_price``, execution subrepo uses no side field +
        ``filled_avg_price``). Normalize through ``_normalize_fill_record``
        so attribution works against either schema.

        Returns ``{ticker: {order_id, price, qty, filled_at, side}}``.
        Empty dict if the broker can't surface fills (e.g., sim path).
        """
        if not disappeared:
            return {}
        if not hasattr(self._broker, "get_filled_orders"):
            return {}
        import datetime as _dt  # noqa: PLC0415
        today = ctx.today if isinstance(ctx.today, _dt.date) else _dt.date.today()
        after = (today - _dt.timedelta(days=5)).isoformat()
        try:
            fills = self._broker.get_filled_orders(after=after) or []
        except Exception as exc:
            log.info(
                "STATE-EXT-SELL attribution: broker.get_filled_orders failed (%s); "
                "logging without fill record",
                exc,
            )
            return {}
        wanted = set(disappeared)
        latest: dict[str, dict] = {}
        for f in fills:
            sym = str(f.get("symbol") or f.get("ticker") or "")
            if sym not in wanted:
                continue
            normalized = self._normalize_fill_record(f)
            # Fail-closed on direction: if the broker DID surface a side
            # field and it isn't "sell", skip. If NO side field exists
            # (execution subrepo schema) we accept the row — caller wants
            # the most-recent fill regardless because absence of side is
            # not the same as "this is a buy".
            side = normalized["side"]
            side_present = any(f.get(k) for k in self._FILL_SIDE_KEYS)
            if side_present and side != "sell":
                continue
            filled_at = normalized["filled_at"]
            existing = latest.get(sym)
            if existing is None or str(existing.get("filled_at") or "") < filled_at:
                latest[sym] = normalized
        return latest

    def _gc_recent_sell_orders(self, ctx) -> dict:
        """Drop runner-submitted SELL order_ids older than the fill lookback.

        ``_collect_disappeared_fills`` only queries broker fills from the last
        5 days, so order_ids older than that can never match a disappeared
        position — keeping them would grow the state file unbounded. Prune to
        a 6-day window (one day of slack over the 5-day fill lookback). Entries
        with an unparseable ``submitted_at`` are kept (fail-open: never lose an
        order_id we might still need to attribute).
        """
        import datetime as _dt  # noqa: PLC0415
        today = ctx.today if isinstance(ctx.today, _dt.date) else _dt.date.today()
        cutoff = today - _dt.timedelta(days=6)
        kept: dict = {}
        for oid, meta in (self._recent_sell_orders or {}).items():
            stamp = str((meta or {}).get("submitted_at") or "")
            try:
                submitted = _dt.date.fromisoformat(stamp)
            except ValueError:
                kept[oid] = meta   # unparseable → keep (fail-open)
                continue
            if submitted >= cutoff:
                kept[oid] = meta
        self._recent_sell_orders = kept
        return kept

    def _attribute_ext_sell(
        self,
        ticker: str,
        fills: dict[str, dict],
    ) -> str:
        """Produce a human-readable attribution string for a STATE-EXT-SELL.

        Decision order:
          1. If the matching fill's ``order_id`` equals a Z9 stop we tracked
             for this ticker, attribute to ``z9_stop``.
          2. If the fill's ``order_id`` is one the runner submitted this
             session (single_day_loss / trailing_stop / model_sell / rotation
             / etc.), attribute to ``runner_<exit_type>`` — it is NOT external.
          3. Otherwise the fill is external — manual close, corporate action,
             or out-of-band liquidation. Surface as ``external_or_manual``.

        Returns a short string suitable for inclusion in the WARNING log.
        Falls back to ``"no_broker_fill_record"`` when the broker didn't
        surface a fill we can match.

        2026-06-03 (HON incident): step 2 added. Previously a runner
        single_day_loss sell that filled would be mislabeled
        ``external_or_manual`` on the next tick's reconciliation because only
        Z9 stop order_ids were matched — polluting the decision-trace audit
        surface with false "external" dispositions.
        """
        fill = fills.get(ticker)
        if not fill:
            return "no_broker_fill_record"
        fill_oid = fill.get("order_id")
        z9_meta = self._stop_orders.get(ticker) or {}
        z9_order_id = z9_meta.get("order_id")
        runner_meta = (
            self._recent_sell_orders.get(str(fill_oid)) if fill_oid else None
        ) or {}
        if z9_order_id and fill_oid and z9_order_id == fill_oid:
            source = "z9_stop"
        elif runner_meta:
            _et = str(runner_meta.get("exit_type") or "").strip()
            source = f"runner_{_et}" if _et else "runner_sell"
        else:
            source = "external_or_manual"
        # Codex #76: fill dict now carries the normalized keys produced by
        # ``_normalize_fill_record`` — ``price`` (not ``fill_price``),
        # ``qty`` (not ``fill_qty``). Compact single-line rendering.
        return (
            f"source={source} "
            f"order_id={fill.get('order_id') or '?'} "
            f"price={fill.get('price') if fill.get('price') is not None else '?'} "
            f"qty={fill.get('qty') if fill.get('qty') is not None else '?'} "
            f"filled_at={fill.get('filled_at') or '?'}"
        )

    # ── commit ─────────────────────────────────────────────────────────────────

    def commit(self, ctx) -> None:  # noqa: ANN001
        """Apply pipeline outputs: execute broker orders, update live_state.json."""
        broker        = self._broker
        today_str     = ctx.today.isoformat()
        pos_cache     = self._positions_cache

        # ── Apply exits ──────────────────────────────────────────────────────
        # Honours optional sig.quantity for partial sells (Kelly trim path).
        # When quantity is None or ≥ current qty → full liquidation (old
        # behaviour). When quantity is a positive float < current qty →
        # partial sell, position stays open with reduced shares; we keep
        # entry_dates / position_hwm / sell_streaks intact.
        #
        # Audit fix EXITS-FAIL (Round 2 deep audit, 2026-04-25): pre-fix,
        # broker.place_order failures inside the SELL branch logged with
        # log.error and `continue`d, but ctx.exits was the only list the
        # ntfy code read for "EXIT ticker (reason)" messages. Result: a
        # failed sell appeared on the operator's phone as if it had
        # succeeded — they thought the position closed, but it was still
        # held at the broker. Now: split ctx.exits → ctx.exits_placed
        # (broker-confirmed) and ctx.exits_failed (broker error). The
        # ntfy path will need to read exits_placed (analogous to the
        # orders_placed/orders_skipped split that already exists for
        # buys); falls back to ctx.exits when those fields aren't set.
        if not hasattr(ctx, "exits_placed"):
            ctx.exits_placed = []
        if not hasattr(ctx, "exits_pending"):
            ctx.exits_pending = []
        if not hasattr(ctx, "exits_failed"):
            ctx.exits_failed = []
        full_exit_tickers: set[str] = set()
        # Audit fix QTY-NaN (Round 2 deep audit, 2026-04-25): same NaN-
        # slip pattern as SE-1/TR-NaN/ROT-NaN-PRICE. Pre-fix, a broker
        # response with NaN qty (rare but possible during account
        # snapshot races) slipped past `qty <= 0` (NaN<=0 False), then
        # `sell_qty = abs(NaN) = NaN` was passed to broker.place_order
        # which crashed inside Alpaca's int(quantity). Now: skip with
        # a clear log on non-finite qty.
        import math as _math

        def _held_qty(t: str) -> float:
            pos = pos_cache.get(t, {})
            try:
                qty_f = float(pos.get("qty", 0))
            except (TypeError, ValueError):
                return 0.0
            return qty_f if _math.isfinite(qty_f) and qty_f > 0 else 0.0

        for ticker, sig in dedupe_exit_signals(ctx.exits, held_qty_for=_held_qty):
            pos = pos_cache.get(ticker, {})
            qty = float(pos.get("qty", 0))
            if not _math.isfinite(qty) or qty <= 0:
                if not _math.isfinite(qty):
                    log.warning("EXIT %s: broker qty=%s non-finite, skipping", ticker, qty)
                continue

            # Audit fix PLTR-AVAILABLE-QTY (2026-04-26 round-3, e2e finding):
            # use qty_available (= qty - held_for_orders) so we don't ask
            # the broker to sell shares locked in pending orders. Alpaca
            # rejects with available=0 in that case. Pre-fix, e2e round 3
            # saw PLTR sell fail this way. Falls back to qty when broker
            # doesn't expose qty_available.
            # Audit fix PLTR-AVAILABLE-QTY-V2 (2026-04-26 round-4):
            # `pos.get("qty_available", qty) or qty` collapses 0-available
            # back to qty (0 is falsy). Pre-fix bug — saw PLTR sell still
            # fail in e2e round 4 with available=0 because we fell back
            # to qty=5. Fix: use explicit None check.
            _qa_raw = pos.get("qty_available", None)
            qty_avail = qty if _qa_raw is None else float(_qa_raw)
            if not _math.isfinite(qty_avail) or qty_avail <= 0:
                # Audit fix LOG-FORMAT (2026-04-26 round-5): arg order
                # was swapped — log output showed "qty=PLTR" instead of
                # "qty=5". Now: ticker first, then qty_avail, then qty.
                log.warning(
                    "EXIT %s: qty_available=%s, qty=%s (likely held in "
                    "pending orders) — skipping. Cancel pending order first.",
                    ticker, qty_avail, qty,
                )
                if not hasattr(ctx, "exits_failed"):
                    ctx.exits_failed = []
                ctx.exits_failed.append({
                    "ticker": ticker, "qty": qty,
                    "exit_type": getattr(sig, "exit_type", ""),
                    "reason": getattr(sig, "reason", ""),
                    "error": f"qty_available={qty_avail}, all locked in pending orders",
                })
                continue

            req_qty = getattr(sig, "quantity", None)
            if (
                not is_full_liquidate_signal(sig, qty)
                and req_qty is not None
                and _math.isfinite(req_qty)
                and 0 < req_qty < qty_avail
            ):
                sell_qty   = float(req_qty)
                is_partial = True
            else:
                # Cap at qty_available (not qty) to avoid broker rejection.
                sell_qty   = abs(qty_avail)
                is_partial = (qty_avail < qty)

            try:
                result = broker.place_order(ticker, "SELL", sell_qty)
            except Exception as exc:
                log.error("SELL failed for %s: %s", ticker, exc)
                ctx.exits_failed.append({
                    "ticker":     ticker,
                    "exit_type":  getattr(sig, "exit_type", ""),
                    "reason":     getattr(sig, "reason", ""),
                    "qty":        sell_qty,
                    "is_partial": is_partial,
                    "error":      str(exc),
                })
                continue
            execution = broker_order_execution(
                result, requested_qty=sell_qty,
                fallback_price=ctx.prices.get(ticker, 0.0),
            )
            if execution["rejected"]:
                log.error(
                    "SELL rejected for %s: status=%s order_id=%s",
                    ticker, execution["status"], execution.get("order_id"),
                )
                ctx.exits_failed.append({
                    "ticker":     ticker,
                    "exit_type":  getattr(sig, "exit_type", ""),
                    "reason":     getattr(sig, "reason", ""),
                    "qty":        sell_qty,
                    "is_partial": is_partial,
                    "order_id":   execution.get("order_id"),
                    "status":     execution["status"],
                    "error":      f"broker_status:{execution['status']}",
                })
                continue
            # Record the runner-submitted SELL order_id (pending OR filled) so a
            # later reconciliation pass attributes the fill to the runner, not
            # external_or_manual (2026-06-03 HON single_day_loss incident).
            _submitted_oid = execution.get("order_id")
            if _submitted_oid:
                import datetime as _dt2  # noqa: PLC0415
                _now_iso = (
                    ctx.today.isoformat()
                    if isinstance(ctx.today, _dt2.date)
                    else _dt2.date.today().isoformat()
                )
                self._recent_sell_orders[str(_submitted_oid)] = {
                    "ticker":       ticker,
                    "exit_type":    getattr(sig, "exit_type", "") or "",
                    "qty":          float(sell_qty),
                    "submitted_at": _now_iso,
                }
            if execution["pending"]:
                pending = {
                    "ticker":     ticker,
                    "exit_type":  getattr(sig, "exit_type", ""),
                    "reason":     getattr(sig, "reason", ""),
                    "qty":        sell_qty,
                    "is_partial": is_partial,
                    "order_id":   execution.get("order_id"),
                    "status":     execution["status"],
                }
                ctx.exits_pending.append(pending)
                log.warning(
                    "SELL pending at broker for %s: %.0f shares status=%s "
                    "order_id=%s; live_state/DB not mutated until fill.",
                    ticker, sell_qty, execution["status"], execution.get("order_id"),
                )
                continue

            sell_qty = float(execution["filled_qty"] or sell_qty)
            price = float(execution["filled_avg_price"] or ctx.prices.get(ticker, 0.0))
            is_partial = bool(execution["partial"] or sell_qty < qty - 1e-9)

            # Use HoldingState.entry_price as the running avg-cost fallback.
            hs = (ctx.holdings or {}).get(ticker)
            lot_accounted = apply_live_sell_lot_accounting(
                sig,
                hs,
                shares=float(sell_qty),
                price=float(price),
                today=ctx.today,
                config=self._config,
            )
            if not lot_accounted:
                # 2026-05-18: stamp P/L on the ExitSignal so live/runner.py's
                # _notify_decision can render explicit $ realized P/L in ntfy.
                # Fallback cost basis is broker avg_entry_price when fill
                # history cannot reconstruct tax lots.
                cost_basis = float(pos_cache.get(ticker, {}).get(
                    "avg_entry_price", 0.0
                ))
                if hs is not None and cost_basis <= 0:
                    cost_basis = float(getattr(hs, "entry_price", 0.0) or 0.0)
                if cost_basis > 0 and price > 0:
                    gain_per_share = price - cost_basis
                    gain_dollar = gain_per_share * sell_qty
                    gain_pct = (price / cost_basis - 1.0) * 100.0
                    try:
                        sig.realized_pnl_dollar = float(gain_dollar)
                        sig.realized_pnl_pct = float(gain_pct)
                        sig.cost_basis = float(cost_basis)
                        sig.sell_price = float(price)
                        sig.shares_sold = float(sell_qty)
                    except Exception:
                        pass
                else:
                    try:
                        sig.sell_price = float(price)
                        sig.shares_sold = float(sell_qty)
                    except Exception:
                        pass
            else:
                try:
                    sig.sell_price = float(price)
                    sig.shares_sold = float(sell_qty)
                except Exception:
                    pass

            ctx.exits_placed.append((ticker, sig))

            tag   = "TRIM" if is_partial else "SELL"
            pl_str = ""
            if getattr(sig, "realized_pnl_dollar", None) is not None:
                pl_str = (f"  P/L=${sig.realized_pnl_dollar:+.2f} "
                          f"({sig.realized_pnl_pct:+.2f}%)")
            log.info("%s  %s  [%s]  %.0f shares @ %.2f%s  %s",
                     tag, ticker, sig.exit_type, sell_qty, price, pl_str,
                     sig.reason)

            # Wash-sale clock: stamp ONLY on full liquidation. Partial
            # trims (Kelly rebalance) intentionally don't block subsequent
            # top-ups — that would prevent the position from ever growing
            # back toward the Kelly target after an over-weight trim.
            if not is_partial:
                full_exit_tickers.add(ticker)
                self._last_sell_dates_str[ticker] = today_str
                self._entry_dates.pop(ticker, None)
                self._entry_signals.pop(ticker, None)   # Approach A cleanup
                self._sell_streaks.pop(ticker, None)
                self._position_hwm.pop(ticker, None)
                # Z9: cancel broker-side stop on full liquidation.
                if self._z9_enabled(ctx):
                    self._z9_cancel_stop(ticker, reason="full liquidation")
            # G8 (2026-05-04): stamp post-stop blackout on path-rule
            # exits regardless of partial/full. Distinct from wash-sale —
            # this fires even on small partial trims because the timing
            # signal (a stop tripped) invalidates re-entry.
            from kernel.pipeline.task_post_stop_cooldown import (  # noqa: PLC0415
                DEFAULT_STOP_EXIT_TYPES,
            )
            if str(getattr(sig, "exit_type", "")) in DEFAULT_STOP_EXIT_TYPES:
                self._last_stop_exit_dates_str[ticker] = today_str
            else:
                # TRIM (partial): replace stop with reduced qty at the same
                # stop_price (never loosens; see _z9_place_or_replace_stop).
                if self._z9_enabled(ctx):
                    held_now = (
                        broker.get_position(ticker)
                        if hasattr(broker, "get_position") else 0.0
                    )
                    if held_now > 0:
                        # Use the current price as reference; the helper
                        # min's against existing stop_price so the stop
                        # never moves up after a trim.
                        self._last_ctx_stop_pct = self._z9_stop_pct(ctx)
                        self._z9_place_or_replace_stop(
                            ticker, float(held_now), float(price), today_str,
                        )
                    else:
                        self._z9_cancel_stop(ticker, reason="trim → flat")
            regime_p = (self._config.get("regime_params", {}) or {}).get(
                ctx.regime, {},
            ) or {}
            sell_log_record = build_sell_trade_event_for_db(
                ticker=ticker,
                sig=sig,
                holding=hs,
                price=price,
                today=ctx.today,
                regime=getattr(ctx, "regime", None),
                confidence=getattr(ctx, "confidence", None),
                regime_params={
                    **regime_p,
                    "tax": self._config.get("tax", {}) or {},
                },
                config=self._config,
                **sell_event_realized_kwargs(sig, hs, today=ctx.today),
            )
            sell_log_record.update({
                "action":    "SELL",
                "symbol":    ticker,
                "exit_type": sig.exit_type,
                "reason":    sig.reason,
                "qty":       sell_qty,
                "partial":   is_partial,
            })
            self._log_trade(ctx, sell_log_record)

        # ── Apply buys ───────────────────────────────────────────────────────
        # Track BUYS as they actually execute vs what the pipeline merely
        # intended. `ctx.orders_placed` = filled/partially-filled at broker,
        # `ctx.orders_pending` = submitted but not filled yet, and
        # `ctx.orders_skipped` = blocked locally or rejected.
        if not hasattr(ctx, "orders_placed"):
            ctx.orders_placed = []
        if not hasattr(ctx, "orders_pending"):
            ctx.orders_pending = []
        if not hasattr(ctx, "orders_skipped"):
            ctx.orders_skipped = []
        import math
        try:
            buy_cash_remaining = float(ctx.cash)
        except (TypeError, ValueError):
            buy_cash_remaining = 0.0
        if not math.isfinite(buy_cash_remaining):
            buy_cash_remaining = 0.0
        sell_credit = same_bar_sell_credit(ctx)
        if sell_credit > 0:
            buy_cash_remaining += sell_credit
            log.info(
                "LIVE-SAME-BAR-SELL-CREDIT: buy budget credited by "
                "$%.2f from broker-confirmed exits",
                sell_credit,
            )
        if not self._sell_only:
            from kernel.pipeline.order_dedupe import (  # noqa: PLC0415
                dedupe_buy_orders_first_wins,
            )
            deduped_orders, skipped_duplicate_buys = (
                dedupe_buy_orders_first_wins(ctx.orders)
            )
            for order_intent in skipped_duplicate_buys:
                ticker = (
                    order_intent.get("ticker")
                    if isinstance(order_intent, dict) else
                    getattr(order_intent, "ticker", "?")
                )
                log.info("BUY skipped: duplicate same-bar buy intent for %s", ticker)
                if isinstance(order_intent, dict):
                    ctx.orders_skipped.append({
                        **order_intent,
                        "skip_reason": "duplicate_buy_intent",
                    })
            for order_intent in deduped_orders:
                order, budget_reason = cap_buy_order_to_cash(
                    order_intent, buy_cash_remaining,
                )
                if order is None:
                    log.info(
                        "BUY skipped: live cash budget rejected %s (%s)",
                        order_intent.get("ticker") if isinstance(order_intent, dict) else "?",
                        budget_reason,
                    )
                    if isinstance(order_intent, dict):
                        ctx.orders_skipped.append({
                            **order_intent,
                            "skip_reason": budget_reason or "cash_budget_rejected",
                        })
                    continue
                if budget_reason == "cash_budget_resized":
                    log.info(
                        "BUY resized by live cash budget: %s shares %s → %s",
                        order["ticker"], order.get("original_shares"), order["shares"],
                    )
                ticker = order["ticker"]
                shares = order["shares"]
                price  = order["price"]
                # Duplicate-order guard
                try:
                    pending = broker.get_open_orders()
                    if ticker in pending:
                        log.info("BUY skipped: pending order exists for %s", ticker)
                        ctx.orders_skipped.append({
                            **order, "skip_reason": "pending_order_exists",
                        })
                        continue
                except Exception as exc:
                    log.error(
                        "BUY skipped: could not verify open orders for %s: %s",
                        ticker, exc,
                    )
                    ctx.orders_skipped.append({
                        **order,
                        "skip_reason": f"open_orders_check_failed:{type(exc).__name__}",
                    })
                    continue

                try:
                    result = broker.place_order(ticker, "BUY", shares)
                except Exception as exc:
                    log.error("BUY failed for %s: %s", ticker, exc)
                    ctx.orders_skipped.append({
                        **order, "skip_reason": f"broker_error:{type(exc).__name__}",
                    })
                    continue
                execution = broker_order_execution(
                    result, requested_qty=shares, fallback_price=price,
                )
                if execution["rejected"]:
                    log.error(
                        "BUY rejected for %s: status=%s order_id=%s",
                        ticker, execution["status"], execution.get("order_id"),
                    )
                    ctx.orders_skipped.append({
                        **order,
                        "skip_reason": f"broker_status:{execution['status']}",
                        "order_id": execution.get("order_id"),
                        "status": execution["status"],
                    })
                    continue

                submitted_notional = shares * price
                if execution["pending"]:
                    ctx.orders_pending.append({
                        **order,
                        "order_id": execution.get("order_id"),
                        "status": execution["status"],
                    })
                    buy_cash_remaining = max(buy_cash_remaining - submitted_notional, 0.0)
                    log.warning(
                        "BUY pending at broker for %s: %d shares status=%s "
                        "order_id=%s; entry state/DB not mutated until fill.",
                        ticker, shares, execution["status"], execution.get("order_id"),
                    )
                    continue

                shares = int(execution["filled_qty"] or shares)
                price = float(execution["filled_avg_price"] or price)
                order = {**order, "shares": shares, "price": price}
                if execution.get("order_id") is not None:
                    order["order_id"] = execution.get("order_id")
                order["status"] = execution["status"]
                order["filled_qty"] = shares
                order["filled_avg_price"] = price
                ctx.orders_placed.append(order)

                invest = shares * price
                buy_cash_remaining = max(buy_cash_remaining - invest, 0.0)
                # Top-up detection: a buy on a ticker we already track is
                # an add-to-existing, not a fresh entry. Preserve entry_date,
                # entry_signals, sell_streaks, and last_sell_dates so the
                # original cost-basis tenure / wash-sale state stays intact.
                # HWM ratchets with current price (whichever is higher).
                is_topup = ticker in self._entry_dates
                action_tag = "TOPUP" if is_topup else "BUY"
                log.info("%s  %s  %d shares @ %.2f  invest=$%.0f",
                         action_tag, ticker, shares, price, invest)

                if not is_topup:
                    self._entry_dates[ticker]       = today_str
                    self._sell_streaks.pop(ticker, None)
                    self._last_sell_dates_str.pop(ticker, None)
                    self._position_hwm[ticker]      = price
                    # Thesis-degradation baseline (Approach A) — stamp entry
                    # scores ONLY on a fresh buy (not a top-up to an already-
                    # held position). Persist in live_state.json so rotation
                    # checks on future bars see a fixed baseline.
                    self._entry_signals[ticker] = {
                        "rank_score":       order.get("rank_score"),
                        "panel_score":      order.get("panel_score"),
                        "kelly_target_pct": order.get("kelly_target_pct"),
                        "regime":           order.get("regime"),
                    }
                else:
                    # Top-up: only HWM may need to ratchet up.
                    self._position_hwm[ticker] = max(
                        float(self._position_hwm.get(ticker, 0.0)), price,
                    )
                # Z9 (2026-04-28): place / replace broker-side stop. Default
                # OFF; honors `live.broker_side_stops.enabled` config flag and
                # the broker's supports_broker_side_stops() capability.
                # On TOPUP: invariant is "never loosen" — handled by
                # _z9_place_or_replace_stop (it min's against existing stop).
                if self._z9_enabled(ctx):
                    self._last_ctx_stop_pct = self._z9_stop_pct(ctx)
                    # Total post-trade qty = previous + new shares.
                    held_now = (
                        broker.get_position(ticker)
                        if hasattr(broker, "get_position") else float(shares)
                    )
                    self._z9_place_or_replace_stop(
                        ticker, float(held_now), float(price), today_str,
                    )
                # Bug #22 fix (2026-04-26 round-7): defensive .get() on
                # order keys. The QP solver path (task_joint_qp.py) emits
                # order dicts WITHOUT rs_score / regime — they're produced
                # by SizeAndEmitTask but not by JointPortfolioQPTask.
                # Pre-fix the bare order["rs_score"] raised KeyError →
                # commit() crashed AFTER orders were submitted to Alpaca,
                # leaving live state inconsistent (orders filled but trade
                # log not written). Now: defensive get with safe defaults
                # so all order producers (selection / rotation / topup /
                # qp / future) are tolerated. rs_score is retired from
                # ranking math anyway (CLAUDE.md), so 0.0 is correct.
                # Trade-log distinguishes order provenance (2026-05-01 audit):
                # `order_type` carries through whatever the producer set
                # (TopUpHeldTask → "TOP_UP", SizeAndEmitTask → "NEW_BUY",
                # rotation/QP → respective tags). When absent, fall back to
                # the runner's own is_topup detection (handles legacy producers
                # that don't tag).
                fallback_type = "TOP_UP" if is_topup else "NEW_BUY"
                order_type    = order.get("order_type", fallback_type)
                buy_log_record = build_buy_trade_event(
                    order,
                    date=ctx.today,
                    default_regime=ctx.regime,
                    default_confidence=ctx.confidence,
                    default_acceptance_reason="live_buy",
                )
                buy_log_record.update({
                    "action":     "BUY",
                    "symbol":     ticker,
                    "shares":     shares,
                    "price":      price,
                    "invest":     invest,
                    "order_type": order_type,
                    "rank_score": order.get("rank_score", 0.0) or 0.0,
                    "rs_score":   order.get("rs_score",   0.0) or 0.0,
                    "regime":     order.get("regime",     ctx.regime),
                })
                self._log_trade(ctx, buy_log_record)

        # ── Persist updated sell streaks from SellJob ─────────────────────
        # Audit fix LS-HWM-1 (Round 2 deep audit, 2026-04-25): pre-fix,
        # this loop recomputed position_hwm from `ctx.prices[ticker]`
        # directly with `max(stored, price)`. That bypassed the EX-HWM
        # safety net living on hs.high_watermark — if ctx.prices[ticker]
        # was NaN/inf (one bad OHLCV bar), `max(stored, NaN) = NaN` and
        # the NaN got SERIALISED into live_state.json, surviving across
        # process restarts until the next compute_exits could recover it.
        # Now: prefer hs.high_watermark (already validated by compute_exits
        # via EX-HWM), fall back to a finite-checked max if hs is missing.
        import math
        for ticker, hs in ctx.holdings.items():
            if ticker in full_exit_tickers:
                continue
            self._sell_streaks[ticker] = hs.sell_streak
            # Prefer the validated HWM that compute_exits computed for
            # this bar; only fall back to a price-based max if hs is
            # somehow missing or non-finite.
            hs_hwm = getattr(hs, "high_watermark", None)
            if hs_hwm is not None and math.isfinite(hs_hwm):
                self._position_hwm[ticker] = float(hs_hwm)
            elif ticker in ctx.prices and math.isfinite(ctx.prices[ticker]):
                stored = float(self._position_hwm.get(ticker, 0.0))
                if not math.isfinite(stored):
                    stored = 0.0
                self._position_hwm[ticker] = max(stored, ctx.prices[ticker])

        # ── State garbage-collection (Bug A — stale entries) ──────────────
        # Audit fix STATE-GC (Round 4 deep audit, 2026-04-25): pre-fix,
        # `commit()` only added to live_state.json on buys/sells; it never
        # removed entries for tickers no longer held. Result: tickers like
        # XLU that were once held but later sold (manually or by a previous
        # version) remained in live_state forever — stale entry_date,
        # phantom position_hwm, ghost sell_streak — confusing the operator
        # and bloating state. Now: drop entries for tickers not in current
        # held_set, EXCEPT keep last_sell_dates entries inside the 30-day
        # wash-sale window (those are still load-bearing for future buys).
        #
        # Audit fix STATE-GC-NEWBUYS (Bug K2, 2026-04-25): pre-fix, ctx.holdings
        # was captured at start-of-bar (broker positions BEFORE today's buys
        # executed). New buys added entries to entry_dates via the buy loop,
        # then GC immediately dropped them because they weren't in
        # ctx.holdings. The state was self-correcting next iter (broker fills
        # would re-seed) but the immediate write was wrong. Fix: extend
        # currently_held with tickers from ctx.orders_placed (broker-confirmed
        # buys) so GC preserves them.
        currently_held = effective_live_holdings_after_orders(
            ctx.holdings.keys(),
            full_exit_tickers,
            getattr(ctx, "orders_placed", []) or [],
        )
        post_snapshot = live_post_execution_snapshot(ctx, broker, currently_held)

        # ── Manual / external disposition detection (Z2, 2026-04-28) ──────
        # Invariant: ANY position that disappears between bars must stamp
        # last_sell_dates, regardless of who sold it. Pre-fix, only sells
        # the runner itself executed got the wash-sale clock; manual sells
        # via the Alpaca app, broker-side liquidations (margin calls,
        # end-of-day flats), and IBKR-side closes were INVISIBLE to the
        # wash-sale guard — the bot could re-buy a hand-sold ticker the
        # next bar. NVTS post-mortem (2026-04-28): user manually exited
        # NVTS after the bot bought a parabolic top; with the old logic
        # NVTS could have been re-bought before the 30-day clock ran.
        # Heuristic: ticker was in entry_dates at start-of-bar AND is not
        # currently held AND wasn't stamped as a runner-sell today
        # → treat as external disposition, stamp today.
        #
        # 2026-05-17 Bug fix: EXCLUDE tickers with a pending broker order.
        # Pre-fix, a Sunday-afternoon BUY whose Alpaca order is still
        # `status=accepted` (not yet filled) shows the position as missing
        # at end-of-bar → STATE-EXT-SELL stamped wash-sale → next day's
        # fill couldn't re-enter even though it was the runner's own buy.
        # Today's HON (and 5/15's META) were both blocked this way.
        # Invariant: pending-at-broker ≠ externally-sold.
        pending_broker = set(getattr(ctx, "pending_broker_tickers", set()) or set())
        preopen_canceled = _preopen_cancel_symbols(
            self._strategy_dir, self._broker_name, today_str,
        )
        stale_canceled = [t for t in self._entry_dates
                          if t not in currently_held
                          and t not in pending_broker
                          and t in preopen_canceled
                          and self._last_sell_dates_str.get(t) != today_str]
        if stale_canceled:
            for t in stale_canceled:
                self._entry_dates.pop(t, None)
                self._entry_signals.pop(t, None)
                self._sell_streaks.pop(t, None)
                self._position_hwm.pop(t, None)
            log.warning(
                "STALE_STATE: %d ticker(s) missing from positions after "
                "pre-open cancelled order — clearing local entry state without "
                "wash-sale stamp: %s",
                len(stale_canceled), sorted(stale_canceled),
            )
        disappeared = [t for t in self._entry_dates
                       if t not in currently_held
                       and t not in pending_broker
                       and t not in preopen_canceled
                       and self._last_sell_dates_str.get(t) != today_str]
        skipped_pending = [t for t in self._entry_dates
                           if t not in currently_held
                           and t in pending_broker
                           and self._last_sell_dates_str.get(t) != today_str]
        if skipped_pending:
            log.info(
                "STATE-EXT-SELL: %d ticker(s) missing from positions but have "
                "pending broker orders — skipping wash-sale stamp (in-flight buy, "
                "not external sell): %s",
                len(skipped_pending), sorted(skipped_pending),
            )
        # Issue #71 / audit #5: STATE-EXT-SELL used to log only the ticker
        # name, leaving the operator unable to distinguish Z9 broker-side
        # stops from manual closes or corporate actions. Pre-fetch the recent
        # SELL-side broker fill history once so each disappeared ticker can
        # be attributed to a specific fill record (order_id, price, qty,
        # filled_at) and a source guess (z9_stop / external).
        ext_sell_fills = self._lookup_ext_sell_fills(ctx, disappeared)
        for t in disappeared:
            self._last_sell_dates_str[t] = today_str
            attribution = self._attribute_ext_sell(t, ext_sell_fills)
            log.warning(
                "STATE-EXT-SELL: %s disappeared from broker without runner sell — "
                "stamping wash-sale clock today (%s) to prevent re-entry within 30d "
                "(attribution: %s)",
                t, today_str, attribution,
            )
            # Z9: cancel any orphan broker-side stop for this ticker.
            # The position is already gone; the stop on broker side is now
            # for 0 shares — Alpaca would auto-cancel, but be explicit.
            if self._z9_enabled(ctx):
                self._z9_cancel_stop(t, reason="external disposition")

        wash_sale_window_days = 30
        cutoff = ctx.today - datetime.timedelta(days=wash_sale_window_days)
        # 2026-05-17: preserve state for tickers with pending broker orders.
        # Same root as the STATE-EXT-SELL fix above: an in-flight buy is not
        # yet a position but its entry_date / entry_signal / position_hwm
        # are load-bearing for when it eventually fills (Monday open for
        # weekend-queued orders). Pre-fix, GC dropped them as "stale".
        held_or_pending = currently_held | pending_broker
        for store_name, store in (
            ("entry_dates",   self._entry_dates),
            ("entry_signals", self._entry_signals),
            ("sell_streaks",  self._sell_streaks),
            ("position_hwm",  self._position_hwm),
        ):
            stale = [t for t in store if t not in held_or_pending]
            for t in stale:
                store.pop(t, None)
            if stale:
                log.info("STATE-GC: dropped %d stale entries from %s: %s",
                         len(stale), store_name, ", ".join(sorted(stale)))
        # Z9: stop_orders GC. Orphan stops (no longer held) get cancelled
        # at the broker too — the position is gone so the stop is for 0
        # shares; Alpaca would no-op but be explicit.
        z9_stale = [t for t in self._stop_orders if t not in currently_held]
        for t in z9_stale:
            if self._z9_enabled(ctx):
                self._z9_cancel_stop(t, reason="stop_orders GC")
            else:
                self._stop_orders.pop(t, None)
        if z9_stale:
            log.info("STATE-GC: dropped %d stale stop_orders entries: %s",
                     len(z9_stale), ", ".join(sorted(z9_stale)))
        # last_sell_dates: keep if within wash-sale window OR ticker still held.
        wash_stale = []
        for t, d_str in list(self._last_sell_dates_str.items()):
            if t in currently_held:
                continue
            try:
                d_obj = datetime.date.fromisoformat(d_str)
            except (ValueError, TypeError):
                wash_stale.append(t)
                continue
            if d_obj < cutoff:
                wash_stale.append(t)
        for t in wash_stale:
            self._last_sell_dates_str.pop(t, None)
        if wash_stale:
            log.info("STATE-GC: dropped %d expired wash-sale entries: %s",
                     len(wash_stale), ", ".join(sorted(wash_stale)))

        # ── Save live_state.json ──────────────────────────────────────────
        # Snapshot RegimeState (countdown / cusum / in_transition) so the
        # next live invocation resumes mid-cooldown instead of re-tripping
        # CUSUM from scratch. Without this, transition_window=True stays
        # stuck whenever SPY's 20-day window still differs from the 20-day
        # reference — which can last 20+ bars after a genuine regime shift.
        rs = getattr(ctx, "regime_state", None)
        regime_state_out = {
            "regime":        ctx.regime,
            "confidence":    round(ctx.confidence, 4),
            "in_transition": bool(getattr(rs, "in_transition", False)),
            "countdown":     int(getattr(rs, "countdown", 0)),
            "cusum_pos":     float(getattr(rs, "cusum_pos", 0.0)),
            "cusum_neg":     float(getattr(rs, "cusum_neg", 0.0)),
            # CUSUM-v2 Design C wall-clock cooldown start (ISO string or null).
            # Let intraday runs read elapsed time instead of ticking bar-count.
            "cooldown_start": (getattr(rs, "cooldown_start", None).isoformat()
                                if getattr(rs, "cooldown_start", None) is not None
                                else None),
        } if rs is not None else {}
        self._state.update({
            "regime":            ctx.regime,
            "regime_confidence": round(ctx.confidence, 4),
            "high_water_mark":   ctx.hwm,
            "skip_buys":         bool(ctx.skip_buys),
            "entry_dates":       self._entry_dates,
            "entry_signals":     self._entry_signals,   # Approach A
            "sell_streaks":      self._sell_streaks,
            "last_sell_dates":   self._last_sell_dates_str,
            "last_stop_exit_dates": self._last_stop_exit_dates_str,
            "position_hwm":      self._position_hwm,
            "stop_orders":       self._stop_orders,    # Z9
            "recent_sell_orders": self._gc_recent_sell_orders(ctx),
            "regime_state":      regime_state_out,
            # MonitorIdleStreakTask counters — persisted across scheduled runs
            "monitor_state":     dict(getattr(ctx, "monitor_state", {}) or {}),
        })
        # 2026-05-20 fix: prefer broker-driven no_trade_streak when the broker
        # exposes get_filled_orders. Stateful counter has been bug-prone
        # (per-invocation vs per-day inflation, state-file corruption from
        # SIGKILL mid-write, …). Real source of truth = Alpaca's order book.
        try:
            self._override_no_trade_streak_from_broker(ctx)
        except Exception as exc:
            log.warning(
                "broker-driven no_trade_streak query failed (%s) — keeping "
                "stateful counter (value=%d). Counter is best-effort but may "
                "drift; investigate if persists.",
                exc,
                int(self._state.get("monitor_state", {}).get("no_trade_streak", 0)),
            )
        # Audit fix LS-ATOM (Round 2 deep audit, 2026-04-25): same atomic
        # write pattern as the parquet stores (DC-2-CACHE / FU-1 /
        # INT-ATOM / etc). Pre-fix, `write_text` opened the file in
        # truncate mode + wrote in-place. A SIGKILL or kernel panic
        # mid-write left a truncated/empty live_state.json on disk —
        # next live run loaded `{}` (default), losing all entry_dates,
        # position_hwm, sell_streaks, regime cooldown state. Wash-sale
        # guards then misfired, regime cooldowns reset, holding tenure
        # reset to today (corrupting tax classification + min_hold).
        # Now: write to .tmp + atomic rename, so a crash can leave the
        # .tmp half-written but the canonical file is still the prior
        # complete snapshot.
        from kernel.state_paths import live_state_path  # noqa: PLC0415
        # Always write to broker-specific path. Legacy live_state.json is
        # never overwritten — it stays as a frozen pre-isolation snapshot
        # for forensics until the operator manually retires it.
        state_file = live_state_path(self._strategy_dir, self._broker_name)
        tmp_path   = state_file.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self._state, indent=2))
        tmp_path.replace(state_file)
        log.info("State saved → %s (atomic, broker=%s)",
                 state_file, self._broker_name)

        # ── Optional SQLite decision trace ────────────────────────────────
        if self._db is not None:
            from kernel.persistence import (  # noqa: PLC0415
                record_pipeline_run, record_candidate_scores, record_trades,
                record_live_state_snapshot, record_rotations,
                record_ticker_daily_state, validate_decision_trace_integrity,
            )
            from kernel.artifact_contract import build_run_bundle  # noqa: PLC0415
            # Reconstruct trade events from ctx (live path doesn't keep an
            # in-memory trade list — we synthesise from exits + orders).
            #
            # Audit fix EXITS-FAIL-DB (Round 4 deep audit, 2026-04-25):
            # pre-fix, this used `ctx.exits` (pipeline intent) instead of
            # `ctx.exits_placed` (broker-confirmed). Failed sells (caught
            # into `ctx.exits_failed` when broker rejected) silently were
            # written to `trades` table as successful — distorting PnL
            # analytics + n_exits count. Match the ntfy logic at
            # live/runner.py which already prefers exits_placed.
            # `*_placed` may legitimately be an empty list when every broker
            # attempt failed or was skipped. Do not fall back to pipeline
            # intent in that case; `trades` is an executed-trade table.
            if hasattr(ctx, "exits_placed"):
                exits_for_db = list(getattr(ctx, "exits_placed", []) or [])
            else:
                exits_for_db = list(ctx.exits or [])
            if hasattr(ctx, "orders_placed"):
                orders_for_db = list(getattr(ctx, "orders_placed", []) or [])
            else:
                orders_for_db = list(ctx.orders or [])
            pending_orders_for_trace = list(getattr(ctx, "orders_pending", []) or [])
            _, _, pending_tickers_for_trace = (
                live_trace_selection_maps(
                    [],
                    pending_orders_for_trace,
                    getattr(ctx, "_blocked_by_ticker", None) or {},
                )
            )
            if pending_tickers_for_trace:
                ctx.counters["broker_pending_submitted"] = (
                    ctx.counters.get("broker_pending_submitted", 0)
                    + len(pending_tickers_for_trace)
                )
            trade_events: list[dict] = []
            regime_p = (self._config.get("regime_params", {}) or {}).get(
                ctx.regime, {},
            ) or {}
            for t, sig in exits_for_db:
                hs    = ctx.holdings.get(t)
                price = sell_event_price(sig, ctx.prices.get(t, 0.0))
                trade_events.append(build_sell_trade_event_for_db(
                    ticker=t,
                    sig=sig,
                    holding=hs,
                    price=price,
                    today=ctx.today,
                    regime=getattr(ctx, "regime", None),
                    confidence=getattr(ctx, "confidence", None),
                    regime_params={**regime_p, "tax": self._config.get("tax", {}) or {}},
                    config=self._config,
                    **sell_event_realized_kwargs(sig, hs, today=ctx.today),
                ))
            for o in orders_for_db:
                trade_events.append(build_buy_trade_event(
                    o,
                    date=ctx.today,
                    default_regime=ctx.regime,
                    default_confidence=ctx.confidence,
                    default_acceptance_reason="live_buy",
                ))
            trade_events.extend(live_execution_attempt_events(ctx))
            run_bundle = build_run_bundle(
                self._config,
                self._strategy_dir,
                run_id=str(getattr(ctx, "run_id", "")),
                run_type="live",
                ctx=ctx,
                broker_mode=self._broker_name,
            )
            run_id = record_pipeline_run(
                self._db,
                run_type        = "live",
                run_date        = ctx.today,
                strategy        = str(self._config.get("model_name", "")),
                regime          = ctx.regime,
                confidence      = float(ctx.confidence) if ctx.confidence is not None else None,
                portfolio_value = post_snapshot["portfolio_value"],
                cash            = post_snapshot["cash"],
                n_candidates    = len(ctx.candidates),
                n_exits         = len(exits_for_db),
                # Audit fix ROT-COUNTER (Bug L, 2026-04-25): use EMITTED
                # rotations count (from EmitRotationsTask via counters dict),
                # not the considered count (len(ctx.rotations) before
                # Kelly/cash filters). Stops SQLite analytics from
                # double-counting rotations that were never executed.
                n_rotations     = int(ctx.counters.get("rotations", 0)),
                n_buys          = len(orders_for_db),
                buy_blocked     = bool(getattr(ctx, "buy_blocked", False)),
                skip_buys        = bool(getattr(ctx, "skip_buys", False)),
                bear_only        = bool(getattr(ctx, "bear_only", False)),
                counters         = getattr(ctx, "counters", {}) or {},
                run_bundle       = run_bundle,
                run_id          = getattr(ctx, "run_id", None),
            )
            selected_tickers, blocked_map, _pending_trace_tickers = (
                live_trace_selection_maps(
                    trade_events,
                    pending_orders_for_trace,
                    getattr(ctx, "_blocked_by_ticker", None) or {},
                )
            )
            for o in getattr(ctx, "orders_skipped", []) or []:
                if isinstance(o, dict) and o.get("ticker"):
                    blocked_map.setdefault(
                        o["ticker"], f"broker_skip:{o.get('skip_reason', 'skipped')}",
                    )
            blocked_map.update(trade_event_blocked_map(trade_events))
            # Audit fix DB-DECISION-FACTORS (2026-04-26 round-5): include
            # sector_map + model_types + panel_artifact path so post-hoc
            # analysis has the FULL decision context per (date, ticker).
            sector_map  = self._config.get("sector_map", {}) or {}
            model_types = model_types_from_models(self._models)
            panel_artifact = (
                self._config.get("ranking", {})
                            .get("panel_scoring", {})
                            .get("artifact_path")
            )
            qp_delta_by_ticker, qp_target_by_ticker, qp_status = qp_trace_maps(ctx)
            # 2026-05-04 user mandate ("rank_score need to be collected
            # properly for future fine tune"): persist the FULL pre-veto
            # candidate list so candidate_scores captures the complete
            # rank_score distribution per bar, not just survivors. The
            # snapshot is set by VetoWeakBuysTask before it filters
            # ctx.candidates. Vetoed rows are tagged via blocked_map
            # (veto:rank_score_below_floor / veto:rank_score_nan).
            cand_pool = candidate_trace_pool(ctx)
            from kernel.decision_trace import candidate_score_excluded_holding_tickers  # noqa: PLC0415
            record_candidate_scores(
                self._db, run_id, cand_pool, ctx.holdings,
                selected_tickers=selected_tickers,
                blocked_map=blocked_map,
                sector_map=sector_map,
                model_types=model_types,
                panel_artifact=panel_artifact,
                qp_delta_by_ticker=qp_delta_by_ticker,
                qp_target_by_ticker=qp_target_by_ticker,
                qp_status=qp_status,
                excluded_holding_tickers=candidate_score_excluded_holding_tickers(self._config),
            )
            record_trades(self._db, run_id, trade_events)
            record_rotations(self._db, run_id, ctx)

            # ── ticker_daily_state — every watchlist ticker, every bar ──
            # Per user spec round-5 (2026-04-26): write a row for EVERY
            # watchlist ticker at decision time, including those filtered
            # at universe / broker / no-model gates. Lets post-hoc
            # analysis answer "what did we KNOW about XYZ on this date
            # and WHY didn't we trade it?" — instead of just the cands.
            try:
                pf_value = float(ctx.portfolio_value) if ctx.portfolio_value else 0.0
                # Bug #20 fix (2026-04-26): pending_broker_tickers is a local
                # of make_context() (line 170), not visible in commit()'s
                # scope. It IS persisted onto ctx at line 478 — read from
                # there. Pre-fix, the bare-name reference raised NameError
                # → swallowed by the outer try/except → ticker_daily_state
                # silently dropped EVERY bar. Defensive default to set()
                # so a sell-only path that didn't run BROKER-PRECHECK still
                # writes the row (with pending_at_broker=0).
                pending_broker_tickers: set = set(
                    getattr(ctx, "pending_broker_tickers", None) or set()
                )
                tds_rows = build_ticker_daily_state_rows(
                    config=self._config,
                    ctx=ctx,
                    selected_tickers=selected_tickers,
                    blocked_map=blocked_map,
                    model_types=model_types,
                    universe_rejections=self._universe_rejections,
                    model_keys=set(self._models or {}),
                    pending_broker_tickers=pending_broker_tickers,
                    portfolio_value=pf_value,
                    sector_map=sector_map,
                    qp_delta_by_ticker=qp_delta_by_ticker,
                    qp_target_by_ticker=qp_target_by_ticker,
                    qp_status=qp_status,
                    extra_tickers=trade_event_tickers(trade_events),
                )
                n_tds = record_ticker_daily_state(
                    self._db, run_date=ctx.today, rows=tds_rows,
                    run_id=run_id,
                )
                log.info("ticker_daily_state: wrote %d row(s) for %s",
                         n_tds, ctx.today.isoformat())
            except Exception as exc:
                # Diagnostic table — never block the bar on a write error.
                log.warning("ticker_daily_state write failed: %s", exc)
                if bool((self._config.get("persistence", {}) or {})
                        .get("strict_ticker_daily_state", True)):
                    raise

            # Plan S — append live_state snapshot. The JSON file is still
            # the source of truth (fast bootstrap + human edits); this row
            # is an append-only audit trail for "what was state X on date Y?"
            record_live_state_snapshot(
                self._db, run_id,
                run_date        = ctx.today,
                strategy        = str(self._config.get("model_name", "")),
                state           = self._state,
                cash            = post_snapshot["cash"],
                portfolio_value = post_snapshot["portfolio_value"],
                n_holdings      = int(post_snapshot["n_holdings"]),
            )
            validate_decision_trace_integrity(
                self._db,
                run_id,
                self._config,
                context="RunnerAdapter.commit",
            )

    # ── Trade log ─────────────────────────────────────────────────────────────

    def _log_trade(self, ctx, record: dict) -> None:
        import datetime as _dt
        strategy_name = self._config.get("model_name", "renquant_104")
        repo_root     = self._strategy_dir.parent.parent
        log_dir       = repo_root / "live" / "logs" / strategy_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{_dt.datetime.now().strftime('%Y-%m-%d')}.json"
        entries = json.loads(log_file.read_text()) if log_file.exists() else []
        record["timestamp"] = _dt.datetime.now().isoformat()
        entries.append(record)
        log_file.write_text(json.dumps(entries, indent=2, default=str))
