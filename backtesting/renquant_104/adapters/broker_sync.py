"""Broker-sync helpers — runner.py decomposition slice 2 (broker_sync).

EXTRACTED 2026-06-13 from adapters/runner.py (eng plan S2 item 5).
Line-faithful, test-gated (the sim replay does not cover the live
adapter): bodies moved verbatim with self-dependencies parameterized
(broker, state dict, bar date); same logger; the runner delegates.
"""
from __future__ import annotations

import logging

log = logging.getLogger("adapters.runner")


def override_no_trade_streak_from_broker(broker, state, ctx) -> None:  # noqa: ANN001
    """Replace stateful no_trade_streak counter with broker-derived truth.

    Stateful counter has bug surface: per-invocation vs per-day inflation,
    SIGKILL mid-write corrupting live_state.json, race between intraday
    SellOnly and daily full pipeline writing the same field. Real source
    of truth = the broker's order book. Logs both for cross-validation.

    2026-05-20: introduced after the per-trading-day fix exposed how
    much state-file divergence had accumulated (counter=32 while LIVE
    Alpaca had fills on 16 of last 25 trading days).
    """
    if not hasattr(broker, "get_filled_orders"):
        return
    from datetime import date, timedelta  # noqa: PLC0415
    import datetime as _dt  # noqa: PLC0415
    from kernel.exits import _is_nyse_trading_day  # noqa: PLC0415

    # Look back ~120 calendar days = enough cushion to detect very long
    # idle periods without paging more than necessary.
    today = ctx.today if isinstance(ctx.today, date) else _dt.date.today()
    after = (today - timedelta(days=120)).isoformat()
    try:
        fills = broker.get_filled_orders(after=after)
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

    mon = state.setdefault("monitor_state", {}) or {}
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
    state["monitor_state"] = mon


def gc_recent_sell_orders(recent_sell_orders: dict, bar_date) -> dict:
    """Drop runner-submitted SELL order_ids older than the fill lookback.

    ``_collect_disappeared_fills`` only queries broker fills from the last
    5 days, so order_ids older than that can never match a disappeared
    position — keeping them would grow the state file unbounded. Prune to
    a 6-day window (one day of slack over the 5-day fill lookback). Entries
    with an unparseable ``submitted_at`` are kept (fail-open: never lose an
    order_id we might still need to attribute).

    Date granularity throughout: ``ctx.today`` is normalized to a pure
    date via ``_bar_date`` (handles the datetime-subclass-of-date case),
    and stamps are parsed from their leading ``YYYY-MM-DD`` so a stored
    timestamp that happens to carry a time component still compares
    cleanly (codex #199 review, findings 1+2).
    """
    import datetime as _dt  # noqa: PLC0415
    cutoff = bar_date - _dt.timedelta(days=6)
    kept: dict = {}
    for oid, meta in (recent_sell_orders or {}).items():
        stamp = str((meta or {}).get("submitted_at") or "")
        try:
            # Parse only the date portion — robust to both 'YYYY-MM-DD'
            # and full-ISO 'YYYY-MM-DDTHH:MM:SS[+tz]' stamps.
            submitted = _dt.date.fromisoformat(stamp[:10])
        except ValueError:
            kept[oid] = meta   # unparseable → keep (fail-open)
            continue
        if submitted >= cutoff:
            kept[oid] = meta
    return kept
