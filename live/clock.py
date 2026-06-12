"""Session clock authority — trading-date semantics live here, nowhere else.

Design: intraday roadmap §4 P0.3 (TZ debt burn-down for session paths);
the 2026-06 audit found 217 naive time sources repo-wide. The harmful
class is naive LOCAL dates used for TRADING-day semantics (day rolls,
trade-log naming, staleness day-math): on this PT machine they roll at
midnight Pacific, not with the exchange. Next DST transition 2026-11-01.

Epoch timestamps (time.time()) are timezone-agnostic and fine; explicit
"%H:%M PT" operator labels are intentional. Neither is migrated.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def ny_now() -> dt.datetime:
    """Aware current time in exchange wall clock."""
    return dt.datetime.now(tz=NY)


def trading_date(now: dt.datetime | None = None) -> dt.date:
    """The exchange-calendar date for trading-day semantics (G2 cap
    rolls, trade-log naming, day-age math). DST-proof: wall clock in
    America/New_York, never the machine's local zone."""
    now = now.astimezone(NY) if now is not None else ny_now()
    return now.date()
