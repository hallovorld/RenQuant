"""STATE-EXT-SELL fill attribution — runner.py decomposition slice 8.

EXTRACTED 2026-06-13 from adapters/runner.py (eng plan S2 item 5). The
broker-fill correlation + attribution path for external dispositions
(Z9 stop / runner sell / manual / corporate action), the umbrella-side
counterpart to kernel.broker_reconciliation's EXT_SELL action. Self-deps
parameterized (broker, the stop_orders + recent_sell_orders caches);
re-exported from runner. Same logger.
"""
from __future__ import annotations

import logging

log = logging.getLogger("adapters.runner")


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

def normalize_fill_record(f: dict) -> dict:
    """Project a broker-specific fill dict onto a uniform schema:

        {order_id, side ("sell"/"buy"/""), price, qty, filled_at}

    Returns ``side=""`` only when NO direction field is present at all
    — caller must then fail-closed and skip the row to avoid mistaking
    a buy for a sell."""
    side_raw = ""
    for key in _FILL_SIDE_KEYS:
        v = f.get(key)
        if v:
            side_raw = str(v).lower()
            break
    side = "sell" if "sell" in side_raw else ("buy" if "buy" in side_raw else "")
    price = None
    for key in _FILL_PRICE_KEYS:
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
    for key in _FILL_QTY_KEYS:
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
    for key in _FILL_ID_KEYS:
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

def lookup_ext_sell_fills(broker, ctx, disappeared: list[str]) -> dict[str, dict]:  # noqa: ANN001
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
    if not hasattr(broker, "get_filled_orders"):
        return {}
    import datetime as _dt  # noqa: PLC0415
    today = ctx.today if isinstance(ctx.today, _dt.date) else _dt.date.today()
    after = (today - _dt.timedelta(days=5)).isoformat()
    try:
        fills = broker.get_filled_orders(after=after) or []
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
        normalized = normalize_fill_record(f)
        # Fail-closed on direction: if the broker DID surface a side
        # field and it isn't "sell", skip. If NO side field exists
        # (execution subrepo schema) we accept the row — caller wants
        # the most-recent fill regardless because absence of side is
        # not the same as "this is a buy".
        side = normalized["side"]
        side_present = any(f.get(k) for k in _FILL_SIDE_KEYS)
        if side_present and side != "sell":
            continue
        filled_at = normalized["filled_at"]
        existing = latest.get(sym)
        if existing is None or str(existing.get("filled_at") or "") < filled_at:
            latest[sym] = normalized
    return latest

def bar_date(ctx) -> "datetime.date":
    """Return the bar date as a pure ``date``.

    ``ctx.today`` is a ``date`` in live runs but a ``datetime`` in
    several tests / crash-recovery paths (and ``datetime`` is a
    subclass of ``date``, so a naive ``isinstance(x, date)`` does NOT
    distinguish them). Normalize with ``.date()`` so downstream
    date-granularity arithmetic and ``date.fromisoformat`` parsing
    never see a stray time component (codex #199 review, finding 2).
    """
    import datetime as _dt  # noqa: PLC0415
    t = getattr(ctx, "today", None)
    if isinstance(t, _dt.datetime):
        return t.date()
    if isinstance(t, _dt.date):
        return t
    return _dt.date.today()



def attribute_ext_sell(stop_orders: dict, recent_sell_orders: dict, ticker: str, fills: dict[str, dict]) -> str:
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
    z9_meta = stop_orders.get(ticker) or {}
    z9_order_id = z9_meta.get("order_id")
    runner_meta = (
        recent_sell_orders.get(str(fill_oid)) if fill_oid else None
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
