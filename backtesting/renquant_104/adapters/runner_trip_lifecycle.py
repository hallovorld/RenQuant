"""Trip-lifecycle replay of broker fill history — RenQuant#618 class B.

A *trip* is one round-trip of a position: it opens on the first BUY fill
after the position was flat and closes on the SELL fill that takes the
running quantity back to zero. The runner's ``entry_dates`` must carry the
start of the CURRENT trip, never the oldest BUY ever seen for the symbol.

Incident (RenQuant#618, 2026-08-24..28): the seed map was built from the
OLDEST BUY fill per symbol ("we don't currently track the trip-lifecycle"),
so after a full exit + re-entry the re-entered name inherited the previous
trip's date — ``ENTRY-DATE-SEED NVDA ← 2026-04-17`` on 2026-08-25, hold=130d
one session after the buy — and ``min_hold_days`` plus every other
hold-days-keyed guard was bypassed. NVDA was ``model_sell``-ed one session
after entry, VLO rotated out one session after re-entry.

This module is deliberately dependency-free (no kernel import) so it can be
unit-tested without the pipeline and re-used by both the seed map and the
re-entry cooldown in ``adapters/runner.py``.

Replay rules (qty-only — price is irrelevant to the lifecycle, so a
price-less SELL is NOT dropped here; contrast ``runner_tax_lots`` which
needs a price to build lots):

* fills are normalized (symbol / side / qty / fill date / order_id),
  de-duplicated on ``(symbol, order_id)`` — the umbrella broker's page walk
  re-fetches the boundary order (RenQuant#618 class C) — and replayed in
  ``filled_at`` order;
* BUY adds qty; when the running qty was flat the BUY starts a trip;
* SELL subtracts qty; a SELL that brings the running qty to ``<= 0`` closes
  the trip (qty is clamped to 0, the date becomes ``last_exit``);
* rows lacking a symbol, a side, a positive qty or a parseable fill date
  cannot be placed on the timeline and are counted in ``dropped`` — never
  silently.

Broker-anchored correction: when the caller passes the broker's CURRENT
quantities and the forward replay does not land on them, the fill history
is known to be inconsistent for that symbol (dropped or duplicated rows
outside our control). The trip start is then recovered by walking the
fills BACKWARD from the broker quantity (BUY subtracts, SELL adds) until
the running quantity reaches zero — the fills of the current trip are the
only ones that matter and they are the most recent, so this is robust to
corruption in older history. If the walk never reaches zero the history
does not cover the trip start and the result is ``None`` (unknown) rather
than a guess.
"""
from __future__ import annotations

import dataclasses
import datetime
from typing import Any, Iterable

_SIDE_KEYS = ("side", "action")
_QTY_KEYS = ("qty", "filled_qty", "fill_qty")
_ID_KEYS = ("order_id", "id")
_EPS = 1e-9


@dataclasses.dataclass
class TripState:
    """Per-symbol lifecycle summary after replay."""

    trip_start: datetime.date | None = None
    last_exit: datetime.date | None = None
    replay_qty: float = 0.0
    broker_qty: float | None = None
    anchored: bool = False        # broker-anchored correction was applied
    consistent: bool = True       # forward replay landed on the broker qty
    n_fills: int = 0


@dataclasses.dataclass(frozen=True)
class _Fill:
    symbol: str
    side: str            # "BUY" | "SELL"
    qty: float
    date: datetime.date
    sort_key: str
    order_id: str | None


def fill_trade_date(raw: Any) -> datetime.date | None:
    """Trade date of a broker ``filled_at`` timestamp.

    Aware timestamps are converted to America/New_York (the account's
    trade-date convention, same zone ``live/clock.py`` uses); a naive or
    date-only string falls back to its leading ``YYYY-MM-DD``. Returns
    ``None`` only when nothing date-like can be read.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        parsed = None
    if parsed is not None and parsed.tzinfo is not None and parsed.utcoffset() is not None:
        try:
            from zoneinfo import ZoneInfo  # noqa: PLC0415
            return parsed.astimezone(ZoneInfo("America/New_York")).date()
        except Exception:  # pragma: no cover — zoneinfo missing
            return parsed.date()
    if parsed is not None:
        return parsed.date()
    try:
        return datetime.date.fromisoformat(text[:10])
    except (ValueError, TypeError):
        return None


def _normalize(fill: Any) -> _Fill | None:
    if not isinstance(fill, dict):
        return None
    symbol = str(fill.get("symbol") or fill.get("ticker") or "").strip().upper()
    if not symbol:
        return None
    side_raw = ""
    for key in _SIDE_KEYS:
        v = fill.get(key)
        if v:
            side_raw = str(v).lower()
            break
    side = "SELL" if "sell" in side_raw else ("BUY" if "buy" in side_raw else "")
    if not side:
        return None
    qty: float | None = None
    for key in _QTY_KEYS:
        v = fill.get(key)
        if v is None:
            continue
        try:
            q = float(v)
        except (TypeError, ValueError):
            continue
        if q == q and q > 0:   # finite, positive (NaN != NaN)
            qty = q
            break
    if qty is None:
        return None
    date = fill_trade_date(fill.get("filled_at"))
    if date is None:
        return None
    order_id = None
    for key in _ID_KEYS:
        v = fill.get(key)
        if v:
            order_id = str(v)
            break
    return _Fill(
        symbol=symbol, side=side, qty=qty, date=date,
        sort_key=str(fill.get("filled_at") or ""), order_id=order_id,
    )


def normalize_fills(fills: Iterable[Any] | None) -> tuple[dict[str, list[_Fill]], int]:
    """Group usable fills by symbol in chronological order.

    Returns ``(by_symbol, dropped)``; ``dropped`` counts rows that could
    not be placed on the timeline (no symbol / side / qty / date) PLUS
    duplicate ``(symbol, order_id)`` rows.
    """
    by_symbol: dict[str, list[_Fill]] = {}
    seen: set[tuple[str, str]] = set()
    dropped = 0
    for raw in fills or []:
        f = _normalize(raw)
        if f is None:
            dropped += 1
            continue
        if f.order_id:
            key = (f.symbol, f.order_id)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
        by_symbol.setdefault(f.symbol, []).append(f)
    for sym in by_symbol:
        # date first (NY trade date), then the raw timestamp string for a
        # stable intra-day order; Alpaca emits uniform +00:00 offsets.
        by_symbol[sym].sort(key=lambda f: (f.date, f.sort_key))
    return by_symbol, dropped


def _forward(fills: list[_Fill]) -> TripState:
    st = TripState(n_fills=len(fills))
    qty = 0.0
    for f in fills:
        if f.side == "BUY":
            if qty <= _EPS:
                st.trip_start = f.date
                qty = 0.0
            qty += f.qty
        else:
            qty -= f.qty
            if qty <= _EPS:
                qty = 0.0
                st.trip_start = None
                st.last_exit = f.date
    st.replay_qty = qty
    return st


def _anchored_trip_start(fills: list[_Fill], broker_qty: float) -> datetime.date | None:
    """Walk backward from the broker quantity to the BUY that opened the
    current trip. ``None`` when the history does not reach a flat point."""
    running = float(broker_qty)
    for f in reversed(fills):
        if f.side == "BUY":
            running -= f.qty
            if running <= _EPS:
                return f.date
        else:
            running += f.qty
    return None


def replay_trip_lifecycle(
    fills: Iterable[Any] | None,
    *,
    current_qty: dict[str, float] | None = None,
    qty_tol: float = 0.01,
) -> tuple[dict[str, TripState], int]:
    """Replay broker fills into per-symbol trip state.

    ``current_qty`` (symbol → broker quantity, 0 for flat) enables the
    broker-anchored correction described in the module docstring. Symbols
    absent from ``current_qty`` are treated as "unknown" and keep the
    forward result. Returns ``(states, dropped_rows)``.
    """
    by_symbol, dropped = normalize_fills(fills)
    out: dict[str, TripState] = {}
    for sym, rows in by_symbol.items():
        st = _forward(rows)
        if current_qty is not None and sym in current_qty:
            try:
                bq = float(current_qty[sym])
            except (TypeError, ValueError):
                bq = float("nan")
            if bq == bq:
                st.broker_qty = bq
                tol = max(qty_tol, abs(bq) * 1e-4)
                if abs(st.replay_qty - bq) > tol:
                    st.consistent = False
                    if bq > _EPS:
                        st.anchored = True
                        st.trip_start = _anchored_trip_start(rows, bq)
                    else:
                        # Broker is flat: the trip is closed whatever the
                        # replay thinks; the latest SELL is the exit.
                        st.anchored = True
                        st.trip_start = None
                        last = rows[-1]
                        st.last_exit = last.date if last.side == "SELL" else st.last_exit
        out[sym] = st
    return out, dropped


def trip_start_map(states: dict[str, TripState]) -> dict[str, datetime.date]:
    """``symbol → current trip start`` for symbols whose trip is open."""
    return {s: st.trip_start for s, st in states.items() if st.trip_start is not None}


def last_exit_map(states: dict[str, TripState]) -> dict[str, datetime.date]:
    """``symbol → date of the SELL that last flattened the position``."""
    return {s: st.last_exit for s, st in states.items() if st.last_exit is not None}


def resolve_entry_date(
    state_entry: str | None,
    trip_start: datetime.date | None,
    today: datetime.date,
    *,
    sentinel_days: int = 31,
) -> tuple[str, str]:
    """Decide the ``entry_dates`` value for a HELD ticker.

    Returns ``(iso_date, action)`` with ``action`` one of:

    * ``"seed"``     — no state; the current trip start is used;
    * ``"sentinel"`` — no state and no trip start (broker history absent):
      ``today - sentinel_days`` (the pre-existing fallback);
    * ``"keep"``     — state present; equals the trip start, or the trip
      start is unknown, or the state is unparseable-but-present with no
      trip start to correct it;
    * ``"backfill"`` — state is INSIDE the trip but later than its first
      fill (stamped late, e.g. "today" by an older runner): moved back to
      the trip start (the pre-existing ENTRY-DATE-BACKFILL, now bounded by
      the trip);
    * ``"reseed"``   — state is OLDER than the trip start, i.e. it belongs
      to a PREVIOUS trip (full exit + re-entry): replaced by the trip
      start (RenQuant#618 class B). The old "state predates broker →
      preserve" rule survives ONLY inside the current trip.
    """
    if not state_entry:
        if trip_start is not None:
            return trip_start.isoformat(), "seed"
        return (today - datetime.timedelta(days=sentinel_days)).isoformat(), "sentinel"
    if trip_start is None:
        return str(state_entry), "keep"
    try:
        cur = datetime.date.fromisoformat(str(state_entry)[:10])
    except (ValueError, TypeError):
        cur = today
    if trip_start < cur:
        return trip_start.isoformat(), "backfill"
    if cur < trip_start:
        return trip_start.isoformat(), "reseed"
    return str(state_entry), "keep"


def days_since_last_exit(
    ticker: str,
    today: datetime.date,
    *,
    state_last_sell: str | datetime.date | None,
    replay_last_exit: datetime.date | None,
) -> tuple[int | None, datetime.date | None]:
    """Days since the most recent KNOWN full exit of ``ticker``.

    Two ledgers are consulted and the LATER date wins: the persisted
    ``last_sell_dates`` entry (stamped by the runner's own full sells and
    by STATE-EXT-SELL from the broker fill date — key verified in
    ``renquant_pipeline.kernel.live_state_v2``) and the fill replay's
    ``last_exit``. Returns ``(days, date)``; ``(None, None)`` when neither
    ledger has an exit.
    """
    candidates: list[datetime.date] = []
    if isinstance(state_last_sell, datetime.datetime):
        candidates.append(state_last_sell.date())
    elif isinstance(state_last_sell, datetime.date):
        candidates.append(state_last_sell)
    elif state_last_sell:
        try:
            candidates.append(datetime.date.fromisoformat(str(state_last_sell)[:10]))
        except (ValueError, TypeError):
            pass
    if replay_last_exit is not None:
        candidates.append(replay_last_exit)
    if not candidates:
        return None, None
    last = max(candidates)
    return (today - last).days, last


def reentry_blocked(
    ticker: str,
    today: datetime.date,
    *,
    min_reentry_days: int,
    state_last_sell: str | datetime.date | None,
    replay_last_exit: datetime.date | None,
) -> tuple[bool, int | None, datetime.date | None]:
    """Same rule as the QP path's churn leg
    (``kernel/portfolio_qp/tasks.py::_compute_qp_wash_mask``):
    block when ``0 <= days_since < min_reentry_days``.

    Returns ``(blocked, days_since, last_exit_date)``.
    """
    if int(min_reentry_days or 0) <= 0:
        return False, None, None
    days, last = days_since_last_exit(
        ticker, today,
        state_last_sell=state_last_sell, replay_last_exit=replay_last_exit,
    )
    if days is None:
        return False, None, None
    return (0 <= days < int(min_reentry_days)), days, last
