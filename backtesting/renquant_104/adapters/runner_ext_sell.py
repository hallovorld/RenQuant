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


# Codex #428 review (finding 1): the fill lookback used by
# ``lookup_ext_sell_fills`` must cover the MAXIMUM plausible reconciliation
# gap, not an arbitrary short window. The real production incident this
# module's ``ext_sell_fill_date`` fix cites had the broker SELL fill on
# 2026-06-02 discovered by reconciliation on 2026-06-26 — a 24-day gap. The
# prior 5-day window could never have found that fill (it would still have
# fallen back to `today_str`, the exact bug being fixed). Widen to the
# 30-day wash-sale window itself PLUS an operational buffer, so any fill
# that could still be wash-sale-relevant is always discoverable. The
# umbrella `live/alpaca_broker.py::get_filled_orders` already paginates
# properly (up to 5000 orders / ~1y) — this constant only bounds the
# `after=` query filter, not broker-side pagination capacity.
EXT_SELL_LOOKBACK_DAYS = 45


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

    Queries ``EXT_SELL_LOOKBACK_DAYS`` (45d — 30d wash-sale window plus an
    operational buffer, codex #428 review) of broker fill history so a
    reconciliation delay of multiple weeks still finds the real fill
    instead of silently falling back to "today" (see module-level comment
    on ``EXT_SELL_LOOKBACK_DAYS``).

    Returns ``{ticker: {order_id, price, qty, filled_at, side}}``. The
    ``side`` key here may be ``""`` (ambiguous — no side/action field
    surfaced by the broker) or ``"sell"``/``"buy"``; this dict is used
    for BOTH log-attribution (``attribute_ext_sell``, tolerant of
    ambiguity) and wash-sale stamping (``ext_sell_fill_date``, which
    requires a CONFIRMED ``side == "sell"`` — see that function).
    Empty dict if the broker can't surface fills (e.g., sim path).
    """
    if not disappeared:
        return {}
    if not hasattr(broker, "get_filled_orders"):
        return {}
    import datetime as _dt  # noqa: PLC0415
    today = ctx.today if isinstance(ctx.today, _dt.date) else _dt.date.today()
    after = (today - _dt.timedelta(days=EXT_SELL_LOOKBACK_DAYS)).isoformat()
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

def _ny_trade_date_from_aware_timestamp(raw: str) -> "datetime.date | None":
    """Parse ``raw`` as an AWARE ISO-8601 timestamp and return its trade
    DATE in the account's trade-date timezone convention —
    America/New_York, the same zone used by ``live/clock.py``'s
    ``NY = ZoneInfo("America/New_York")`` / ``trading_date()`` and by
    ``kernel/data.py``'s NYSE freshness checks. Not imported from those
    modules directly: this file is deliberately dependency-free (module
    docstring — "Self-contained"), and ``live/clock.py`` isn't even on
    this package's import path (``backtesting/renquant_104`` never
    depends on ``live/``). Same zone name, independently applied.

    Codex #428 review, finding 3: the prior implementation sliced the
    first 10 characters of the raw string (``YYYY-MM-DD``) — a fill at
    ``00:30 UTC`` was read as if that UTC calendar date were already the
    NY trade date, when it actually belongs to the PRIOR NY trading date
    (UTC-4/UTC-5 puts 00:30 UTC at 19:30/20:30 the previous NY day). That
    slicing approach could shift the wash-sale clock by one day in either
    direction near the UTC midnight boundary. Fixed here by properly
    parsing an aware datetime and converting it.

    Fails CLOSED (returns ``None``) rather than guessing when:
      * ``raw`` isn't parseable ISO-8601 at all, or
      * it parses but carries NO timezone/offset (naive) — a naive
        timestamp cannot be safely mapped to a trade date near a
        day-boundary; silently assuming a timezone is exactly the class
        of bug this function exists to eliminate.
    """
    import datetime as _dt  # noqa: PLC0415
    text = str(raw).strip()
    # datetime.fromisoformat() before Python 3.11 does not accept the
    # 'Z' (Zulu/UTC) suffix common in broker/REST API timestamps —
    # normalize to an explicit +00:00 offset first.
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = _dt.datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None  # naive — fail closed, never guess
    try:
        from zoneinfo import ZoneInfo  # noqa: PLC0415
        ny_local = parsed.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        return None
    return ny_local.date()


def ext_sell_fill_date(fill: dict | None) -> "datetime.date | None":
    """Extract the broker fill DATE from a normalized ext-sell fill record.

    ``fill`` is the per-ticker value out of ``lookup_ext_sell_fills``'s
    return dict (already run through ``normalize_fill_record``), so
    ``filled_at`` is the broker-reported timestamp for the most recent
    qualifying fill. Returns ``None`` — callers must then fall back to a
    "no confirmed fill" path rather than inventing a date — when:

      * there is no fill record at all;
      * the fill's ``side`` is not a CONFIRMED ``"sell"`` (codex #428
        review, finding 2). ``lookup_ext_sell_fills`` deliberately keeps
        rows with an AMBIGUOUS side (``""`` — broker surfaced no
        side/action field at all, e.g. the execution-subrepo schema) so
        the log-attribution string in ``attribute_ext_sell`` can still
        name a candidate fill. That tolerance is fine for a human-
        readable log line but NOT authoritative enough to actually set
        ``last_sell_dates`` (the wash-sale clock): an ambiguous or
        actual-BUY fill must never be mistaken for a sell here. If a
        broker's schema needs enrichment to surface a side (e.g. an
        ``order_id`` follow-up lookup), that is a broker-adapter change,
        not a guess made in this reconciliation path;
      * ``filled_at`` is missing or cannot be parsed as an AWARE
        timestamp (see ``_ny_trade_date_from_aware_timestamp`` — also
        fails closed on naive timestamps rather than guessing a
        timezone).

    Authority principle (2026-07-01 META incident): the broker's fill
    timestamp is authoritative over "today, because that's when this
    reconciliation code happened to run" — mirrors the ENTRY-DATE-FROM-
    FILLS principle already established for ``entry_dates``. Live
    incident: ``last_sell_dates`` was wrongly stamped with the
    reconciliation run date (2026-06-26) instead of the real broker SELL
    fill date (2026-06-02), extending the wash-sale block by 24 days.
    """
    if not fill:
        return None
    if fill.get("side") != "sell":
        return None
    fa = fill.get("filled_at")
    if not fa:
        return None
    return _ny_trade_date_from_aware_timestamp(fa)


def ext_sell_stamp_decision(
    fill_date: "datetime.date | None",
    prior_stamp: str | None,
    today_str: str,
) -> "tuple[str, str]":
    """Decide the ``last_sell_dates`` value to stamp for a ticker that
    disappeared from the broker book, and which log path the caller
    (``adapters/runner.py``'s STATE-EXT-SELL loop) should take.

    Returns ``(stamp_value, path)`` where ``path`` is one of:

      * ``"actual_fill"`` — a confirmed broker SELL fill date was found
        (via ``ext_sell_fill_date``); authoritative, always wins.
      * ``"unresolved_preserve"`` — no confirmed fill within the lookback
        window, but an OLDER ``last_sell_dates`` value already exists for
        this ticker. Codex #428 review ("ALSO reconsider"): overwriting a
        known older date with "today" DESTROYS real evidence and
        recreates the over-extension bug in a different form — if the
        real fill is older than even the widened lookback window,
        "today" is a WORSE guess than the value already on file.
        Preserve it; the caller must flag this loudly as an UNRESOLVED
        reconciliation needing operator/broker-history recovery, not
        silently present "today" as a known sale date.
      * ``"no_fill_fallback"`` — no confirmed fill AND no prior value to
        fall back on; the ticker's disposition is genuinely unknown to
        this state file. Stamping ``today_str`` here is the conservative
        (block re-entry starting now) choice, not a claim that the sale
        actually happened today.
    """
    if fill_date is not None:
        return fill_date.isoformat(), "actual_fill"
    if prior_stamp:
        return prior_stamp, "unresolved_preserve"
    return today_str, "no_fill_fallback"


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
