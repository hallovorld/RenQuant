"""Live sell tax-lot accounting — runner.py decomposition slice 6.

EXTRACTED 2026-06-13 from adapters/runner.py (eng plan S2 item 5). Pure
functions that reconstruct long tax lots from broker fill history and
stamp realized-PnL / disposed-lot-tax economics onto live sell events.
The kernel.exits / kernel.portfolio tax-lot primitives are imported
lazily inside each function (verbatim), so this module has no import-time
kernel dependency. Re-exported from runner for back-compat.
"""
from __future__ import annotations

import datetime
from typing import Any


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
