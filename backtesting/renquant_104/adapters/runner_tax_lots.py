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
import logging
import math
from typing import Any

# Same logger the runner uses so LIVE-TAX-LOTS lines land in the daily log.
log = logging.getLogger("adapters.runner")

# RQ#618 class C (2026-08-29): the replay used to `continue` past any fill
# whose ``filled_avg_price`` was missing/0 — a price-less SELL then never
# decremented lots, so the reconstructed qty exceeded the broker qty every
# run (VLO 7 vs 5, PANW 6 vs 3, APH 14 vs 8) and the lots fell back to the
# broker average price. Now a price-less fill is APPLIED at its qty with a
# stand-in basis, flagged ``price_missing=True`` on the lot / the degraded
# record, warned ONCE per fill, and counted in ``stats`` so the hydration
# site can explain a mismatch instead of silently falling back.
_DEGRADED_KEYS = (
    "price_missing_sell",
    "price_missing_buy",
    "dropped_sell_without_lots",
    "oversell_clamped",
)


def new_replay_stats() -> dict[str, Any]:
    """Fresh bookkeeping for one ``reconstruct_live_tax_lots_from_fills`` call."""
    return {
        "fills_total": 0,               # dict fills seen (after the type filter)
        "fills_applied": 0,             # BUY/SELL applied with a real fill price
        "price_missing_sell": 0,        # SELL qty>0, no price: lots reduced at basis
        "price_missing_buy": 0,         # BUY qty>0, no price: lot appended, basis stand-in
        "dropped_unparseable": 0,       # no symbol / qty<=0 or non-numeric / no date
        "dropped_sell_without_lots": 0, # SELL precedes any BUY in the window
        "dropped_unknown_action": 0,    # neither BUY nor SELL
        "oversell_clamped": 0,          # SELL qty > lots held: excess clamped
        "fills_by_ticker": {},          # ticker -> fills that reached the replay
        "degraded_by_ticker": {},       # ticker -> {key: count} over _DEGRADED_KEYS
        "degraded_fills": [],           # one record per degraded fill (tax reports)
    }


class LiveTaxLotReconstruction(dict):
    """``{ticker: [TaxLot, ...]}`` plus the replay bookkeeping (RQ#618 class C).

    A plain ``dict`` subclass so every existing consumer (``.get``, ``==``,
    iteration) is unchanged; ``.stats`` (see ``new_replay_stats``) carries
    what the replay did with each fill so ``adopt_live_tax_lots`` can log a
    diagnosable invariant when the lots disagree with the broker qty.
    """

    def __init__(self, *args: Any, stats: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.stats: dict[str, Any] = stats if stats is not None else new_replay_stats()


def degraded_counts_for(stats: dict[str, Any] | None, ticker: str) -> dict[str, int]:
    """Per-ticker degraded-fill counts, zero-filled over ``_DEGRADED_KEYS``."""
    per = ((stats or {}).get("degraded_by_ticker") or {}).get(ticker) or {}
    return {k: int(per.get(k, 0) or 0) for k in _DEGRADED_KEYS}


def _record_degraded(
    stats: dict[str, Any],
    key: str,
    ticker: str,
    fill: dict[str, Any],
    *,
    qty: float,
    stand_in_price: float,
    applied: bool,
) -> None:
    stats[key] = int(stats.get(key, 0)) + 1
    per = stats["degraded_by_ticker"].setdefault(ticker, {})
    per[key] = int(per.get(key, 0)) + 1
    stats["degraded_fills"].append({
        "symbol": ticker,
        "action": str(fill.get("action") or "").upper(),
        "qty": float(qty),
        "filled_at": fill.get("filled_at"),
        "order_id": str(fill.get("order_id") or ""),
        "kind": key,
        "stand_in_price": float(stand_in_price),
        "price_missing": key.startswith("price_missing"),
        "applied": bool(applied),
    })


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


def _parse_qty(fill: dict[str, Any]) -> float | None:
    """Fill quantity; ``None`` when absent or non-numeric (unparseable)."""
    raw = fill.get("qty")
    if raw is None or raw == "":
        return None
    try:
        qty = float(raw)
    except (TypeError, ValueError):
        return None
    return qty if math.isfinite(qty) else None


def _parse_price(fill: dict[str, Any]) -> float:
    """Fill price; ``0.0`` when absent, non-numeric, non-finite or <= 0.

    A missing price is DEGRADATION, not a reason to drop the fill (RQ#618
    class C) — the caller applies the qty and flags the basis.
    """
    for key in ("avg_price", "filled_avg_price"):
        raw = fill.get(key)
        if raw is None or raw == "":
            continue
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and price > 0.0:
            return price
    return 0.0


def _lot_shares(hs: Any) -> float:
    """Shares held per the LOTS — never the legacy ``shares`` field.

    ``HoldingState.total_shares()`` falls back to ``self.shares`` when the
    lot list is empty, so after a FULL sell the replay state still reported
    the pre-sell qty, was never popped, and the next BUY's ``ensure_lots``
    re-synthesised the already-sold lot from those legacy fields. That is
    the exact arithmetic of the observed mismatches (VLO 2+5=7 vs 5,
    NVDA 7+7=14 vs 7, PANW 3+3=6 vs 3, APH 6+8=14 vs 8): every full exit
    followed by a re-entry resurrected the disposed lot.
    """
    return float(sum(float(getattr(L, "shares", 0.0) or 0.0) for L in (hs.lots or [])))


def _copy_lot(lot: Any) -> Any:
    from kernel.exits import TaxLot

    out = TaxLot(shares=lot.shares, price=lot.price, date=lot.date)
    if getattr(lot, "price_missing", False):
        out.price_missing = True  # type: ignore[attr-defined]
    return out


def reconstruct_live_tax_lots_from_fills(
    fills: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    config: dict[str, Any] | None = None,
) -> LiveTaxLotReconstruction:
    """Rebuild current long tax lots from broker fill history.

    This keeps live partial-sell accounting on the same FIFO/HIFO contract as
    sim/LEAN. Alpaca exposes only average entry on positions, which is not
    enough to audit a partial trim's realized basis.

    Fill handling (RQ#618 class C, 2026-08-29):

    * a fill with no symbol, a non-numeric/non-positive qty, or no parseable
      ``filled_at`` is dropped and counted (``dropped_unparseable``);
    * a SELL with qty>0 but no fill price STILL reduces lots — the disposed
      lots' cost basis is the stand-in price for the realized-P&L record,
      tagged ``price_missing=True`` in ``stats["degraded_fills"]``, one
      warning per fill;
    * a BUY with qty>0 but no fill price is appended at qty with a
      NaN/None-safe stand-in basis (the ticker's weighted-average basis so
      far, else 0.0), the lot carries ``price_missing=True``; the hydration
      site (``adopt_live_tax_lots``) back-fills the basis from the broker
      average and refuses to attach lots whose basis is still unknown;
    * a SELL that precedes any BUY in the window is not applied (there is no
      lot to consume) and counted (``dropped_sell_without_lots``); a SELL
      that exceeds the lots held is clamped and counted (``oversell_clamped``);
    * a FULL sell flattens the ticker from the LOT sum (see ``_lot_shares``):
      the legacy ``total_shares()`` fallback resurrected the disposed lot on
      the next BUY, which is the arithmetic behind every observed mismatch.

    Returns a ``LiveTaxLotReconstruction`` — a ``dict`` with a ``.stats``
    attribute; the summary is logged once per call on the runner logger.
    """
    from kernel.exits import HoldingState, TaxLot, apply_buy_lot, apply_sell_lots_detailed

    lot_method = str(
        (((config or {}).get("rotation", {}) or {}).get("joint_actions", {}) or {})
        .get("qp_tax_lot_method", ((config or {}).get("tax", {}) or {}).get("lot_method", "fifo"))
    ).lower()
    stats = new_replay_stats()
    states: dict[str, HoldingState] = {}
    ordered = sorted(
        [f for f in (fills or []) if isinstance(f, dict)],
        key=lambda f: str(f.get("filled_at") or ""),
    )
    for fill in ordered:
        stats["fills_total"] += 1
        ticker = str(fill.get("symbol") or "").strip()
        action = str(fill.get("action") or "").upper()
        qty = _parse_qty(fill)
        price = _parse_price(fill)
        fill_date = _fill_date(fill)
        if not ticker or qty is None or qty <= 0 or fill_date is None:
            stats["dropped_unparseable"] += 1
            continue
        if action not in ("BUY", "SELL"):
            stats["dropped_unknown_action"] += 1
            continue
        stats["fills_by_ticker"][ticker] = int(stats["fills_by_ticker"].get(ticker, 0)) + 1
        hs = states.get(ticker)
        if action == "BUY":
            if price > 0.0:
                if hs is None:
                    hs = HoldingState(
                        entry_price=price,
                        entry_date=fill_date,
                        high_watermark=price,
                        shares=0.0,
                    )
                    states[ticker] = hs
                apply_buy_lot(hs, qty, price, fill_date)
                hs.shares = _lot_shares(hs)
                hs.entry_price = hs.weighted_avg_entry_price()
                hs.high_watermark = max(float(hs.high_watermark or price), price)
                stats["fills_applied"] += 1
                continue
            # Price-less BUY: apply at qty with a stand-in basis, never drop.
            stand_in = 0.0
            if hs is not None and hs.lots:
                stand_in = _finite_number(hs.weighted_avg_entry_price())
                stand_in = stand_in if stand_in > 0.0 else 0.0
            if hs is None:
                hs = HoldingState(
                    entry_price=stand_in,
                    entry_date=fill_date,
                    high_watermark=stand_in,
                    shares=0.0,
                )
                states[ticker] = hs
            lot = TaxLot(shares=float(qty), price=float(stand_in), date=fill_date)
            lot.price_missing = True  # type: ignore[attr-defined]
            hs.lots.append(lot)
            hs.shares = _lot_shares(hs)
            hs.entry_price = hs.weighted_avg_entry_price()
            _record_degraded(
                stats, "price_missing_buy", ticker, fill,
                qty=qty, stand_in_price=stand_in, applied=True,
            )
            log.warning(
                "LIVE-TAX-LOTS: %s BUY qty=%.4f filled_at=%s order_id=%s has no "
                "fill price; lot appended at stand-in basis %.4f "
                "(price_missing=True) — applied, not dropped",
                ticker, qty, fill.get("filled_at"), fill.get("order_id") or "",
                stand_in,
            )
            continue
        # SELL
        if hs is None or not hs.lots:
            _record_degraded(
                stats, "dropped_sell_without_lots", ticker, fill,
                qty=qty, stand_in_price=price, applied=False,
            )
            log.info(
                "LIVE-TAX-LOTS: %s SELL qty=%.4f filled_at=%s order_id=%s precedes "
                "any BUY in the replay window; no lot to consume — not applied",
                ticker, qty, fill.get("filled_at"), fill.get("order_id") or "",
            )
            continue
        held_before = _lot_shares(hs)
        basis, _, disposed = apply_sell_lots_detailed(hs, qty, lot_method)
        disposed_sh = float(sum(float(d.shares) for d in disposed))
        if price > 0.0:
            stats["fills_applied"] += 1
        else:
            stand_in = basis / disposed_sh if disposed_sh > 0.0 else 0.0
            _record_degraded(
                stats, "price_missing_sell", ticker, fill,
                qty=qty, stand_in_price=stand_in, applied=True,
            )
            log.warning(
                "LIVE-TAX-LOTS: %s SELL qty=%.4f filled_at=%s order_id=%s has no "
                "fill price; lots reduced at cost basis %.4f as the stand-in "
                "price (price_missing=True) — applied, not dropped",
                ticker, qty, fill.get("filled_at"), fill.get("order_id") or "",
                stand_in,
            )
        if qty - disposed_sh > 1e-9:
            _record_degraded(
                stats, "oversell_clamped", ticker, fill,
                qty=qty, stand_in_price=price, applied=True,
            )
            log.info(
                "LIVE-TAX-LOTS: %s SELL qty=%.4f filled_at=%s exceeds lots held "
                "%.4f; excess clamped (a BUY is missing from the replay window)",
                ticker, qty, fill.get("filled_at"), held_before,
            )
        hs.shares = _lot_shares(hs)
        hs.entry_price = hs.weighted_avg_entry_price()
        if hs.shares <= 1e-9:
            states.pop(ticker, None)
    out = LiveTaxLotReconstruction(stats=stats)
    for ticker, hs in states.items():
        if hs.lots:
            out[ticker] = [_copy_lot(L) for L in hs.lots]
    if stats["fills_total"] > 0:
        degraded = sum(int(stats[k]) for k in _DEGRADED_KEYS)
        emit = log.warning if degraded > 0 else log.info
        emit(
            "LIVE-TAX-LOTS replay summary: fills=%d applied=%d "
            "price_missing_sell=%d price_missing_buy=%d "
            "dropped_unparseable=%d dropped_sell_without_lots=%d "
            "dropped_unknown_action=%d oversell_clamped=%d tickers_with_lots=%d",
            stats["fills_total"], stats["fills_applied"],
            stats["price_missing_sell"], stats["price_missing_buy"],
            stats["dropped_unparseable"], stats["dropped_sell_without_lots"],
            stats["dropped_unknown_action"], stats["oversell_clamped"],
            len(out),
        )
    return out


def adopt_live_tax_lots(
    holding: Any,
    ticker: str,
    lots: list[Any] | None,
    broker_qty: float,
    broker_avg_price: float,
    *,
    stats: dict[str, Any] | None = None,
) -> bool:
    """Attach reconstructed lots to a hydrated live holding when they
    reconcile with the broker qty; otherwise log a diagnosable invariant.

    Hydration invariant (RQ#618 class C): when Σ lot shares != broker qty the
    ``LIVE-TAX-LOTS`` warning now carries the signed delta, the number of
    fills the replay saw for the ticker, and the per-ticker degraded-fill
    counts, so the next mismatch names its cause instead of being a bare
    "using broker avg_entry_price fallback".

    Price-missing lots (a BUY with no fill price) are back-filled here from
    the broker average: with the other lots' basis known and the qty
    reconciled, the missing lots' basis is the residual
    ``(avg_entry_price × qty − Σ known cost) / missing shares``; when that
    is not a positive finite number the broker average itself is used. The
    ``price_missing`` flag stays on the lot so tax reporting can flag it.
    If any lot's basis is STILL unknown (no usable broker average) the lots
    are NOT attached — a 0-basis lot would understate the weighted entry
    price and overstate realized gains downstream.

    Returns ``True`` iff ``holding.lots`` was set from ``lots``.
    """
    counts = degraded_counts_for(stats, ticker)
    degraded_total = sum(counts.values())
    seen = int(((stats or {}).get("fills_by_ticker") or {}).get(ticker, 0) or 0)
    try:
        broker_qty_f = float(broker_qty)
    except (TypeError, ValueError):
        broker_qty_f = 0.0
    if not math.isfinite(broker_qty_f):
        broker_qty_f = 0.0
    if not lots:
        if degraded_total > 0:
            log.warning(
                "LIVE-TAX-LOTS: %s no reconstructed lots (broker qty %.4f, "
                "delta=%+.4f); replay saw %d fill(s) for it, degraded: "
                "price_missing_sell=%d price_missing_buy=%d "
                "sell_without_lots=%d oversell_clamped=%d; using broker "
                "avg_entry_price fallback",
                ticker, broker_qty_f, -broker_qty_f, seen,
                counts["price_missing_sell"], counts["price_missing_buy"],
                counts["dropped_sell_without_lots"], counts["oversell_clamped"],
            )
        return False
    lot_qty = sum(_finite_number(getattr(L, "shares", 0.0)) for L in lots)
    delta = lot_qty - broker_qty_f
    if abs(delta) > max(0.01, abs(broker_qty_f) * 1e-4):
        log.warning(
            "LIVE-TAX-LOTS: %s reconstructed lot qty %.4f != broker qty %.4f; "
            "using broker avg_entry_price fallback (delta=%+.4f; replay saw "
            "%d fill(s) for it, degraded: price_missing_sell=%d "
            "price_missing_buy=%d sell_without_lots=%d oversell_clamped=%d)",
            ticker, lot_qty, broker_qty_f, delta, seen,
            counts["price_missing_sell"], counts["price_missing_buy"],
            counts["dropped_sell_without_lots"], counts["oversell_clamped"],
        )
        return False
    missing = [
        L for L in lots
        if getattr(L, "price_missing", False) or _finite_number(getattr(L, "price", 0.0)) <= 0.0
    ]
    if missing:
        avg = _finite_number(broker_avg_price)
        missing_sh = sum(_finite_number(L.shares) for L in missing)
        stand_in = 0.0
        source = "none"
        if avg > 0.0 and missing_sh > 0.0:
            known_cost = sum(
                _finite_number(L.shares) * _finite_number(L.price)
                for L in lots if L not in missing
            )
            residual = (avg * lot_qty - known_cost) / missing_sh
            if math.isfinite(residual) and residual > 0.0:
                stand_in, source = residual, "broker_avg_residual"
            else:
                stand_in, source = avg, "broker_avg_entry_price"
        if stand_in > 0.0:
            for L in missing:
                L.price = float(stand_in)
                L.price_missing = True  # type: ignore[attr-defined]
            log.warning(
                "LIVE-TAX-LOTS: %s %d lot(s) / %.4f sh have no fill price; basis "
                "back-filled at %.4f from %s (price_missing=True — tax reports "
                "must flag these lots)",
                ticker, len(missing), missing_sh, stand_in, source,
            )
        else:
            log.warning(
                "LIVE-TAX-LOTS: %s %d lot(s) / %.4f sh have no fill price and the "
                "broker avg_entry_price (%s) cannot back-fill them; lots NOT "
                "attached, using broker avg_entry_price fallback",
                ticker, len(missing), missing_sh, broker_avg_price,
            )
            return False
    holding.lots = lots
    try:
        holding.entry_price = holding.weighted_avg_entry_price()
    except Exception:
        pass
    return True


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
