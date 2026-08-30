"""Earnings-calendar freshness rail + recent-print selection (stdlib only).

Defect this closes (2026-08-30 data audit): the live pre/post-earnings
buffer (`is_earnings_blocked`, ±3d buy / -2d..+5d sell) reads
`artifacts/prod/earnings-calendar.json`, whose producer was never
scheduled — the artifact froze at 2026-04-24 (last date 2026-07-24), so
from late July every Aug/Sep print was invisible and the buffer silently
never fired (HPE bought 2026-08-27 into an early-Sep print). A stale
calendar and a missing calendar are indistinguishable to the consumer:
`is_earnings_blocked` just returns False. This module makes staleness a
LOUD, checkable verdict without ever aborting a run (fail soft).

Single source of truth for:
  * assess_earnings_calendar_freshness — the staleness rail verdict
  * select_recent_prints — the incremental-refresh ticker selection
    (tickers with a print in the last N days) used by the daily
    earnings-surprise refresh job.

Consumers:
  * adapters/runner_artifacts.load_context_artifacts (live daily runner)
  * scripts/earnings_calendar_rail.py (CLI for shell preflights/jobs)
  * tests/test_earnings_calendar_rail.py
"""
from __future__ import annotations

import datetime

#: The buffer needs to see at least this many days ahead: a print at
#: today+5 must already be inside the sell-side pre-buffer horizon when
#: the calendar was last written. Below this the buffer is effectively
#: disabled for upcoming prints.
DEFAULT_MIN_HORIZON_DAYS = 5


def _parse_dates(values) -> list[datetime.date]:
    out: list[datetime.date] = []
    if not isinstance(values, (list, tuple)):
        return out
    for v in values:
        try:
            out.append(datetime.date.fromisoformat(str(v)[:10]))
        except (ValueError, TypeError):
            continue
    return out


def earnings_calendar_horizon(calendar) -> datetime.date | None:
    """Max parseable date across the whole calendar, or None."""
    if not isinstance(calendar, dict):
        return None
    best: datetime.date | None = None
    for dates in calendar.values():
        for d in _parse_dates(dates):
            if best is None or d > best:
                best = d
    return best


def assess_earnings_calendar_freshness(
    calendar,
    today: datetime.date | None = None,
    min_horizon_days: int = DEFAULT_MIN_HORIZON_DAYS,
) -> dict:
    """Staleness verdict for the earnings calendar.

    Returns {"status": "ok"|"stale"|"missing", "last_date": date|None,
    "message": str}. "stale" means last_date < today + min_horizon_days:
    the pre/post-earnings buffer cannot see upcoming prints and is
    effectively DISABLED. Pure + fail-soft: never raises on garbage.
    """
    today = today or datetime.date.today()
    if not isinstance(calendar, dict) or not calendar:
        return {
            "status": "missing", "last_date": None,
            "message": "earnings calendar missing or empty — "
                       "earnings buffer is DISABLED",
        }
    last = earnings_calendar_horizon(calendar)
    if last is None:
        return {
            "status": "missing", "last_date": None,
            "message": "earnings calendar has no parseable dates — "
                       "earnings buffer is DISABLED",
        }
    required = today + datetime.timedelta(days=min_horizon_days)
    if last < required:
        return {
            "status": "stale", "last_date": last,
            "message": (
                f"earnings calendar STALE: last date {last.isoformat()} < "
                f"today+{min_horizon_days}d ({required.isoformat()}) — the "
                f"pre/post-earnings buffer cannot fire for upcoming prints"
            ),
        }
    return {
        "status": "ok", "last_date": last,
        "message": f"earnings calendar fresh through {last.isoformat()}",
    }


def select_recent_prints(
    calendar,
    today: datetime.date | None = None,
    lookback_days: int = 7,
) -> list[str]:
    """Tickers with an earnings print in [today-lookback_days, today].

    The incremental selection for the daily earnings-surprise refresh:
    only names that just printed need a PEAD/SUE refetch — everyone
    else's cache is already current. Future-dated prints are NOT
    selected (no actual EPS exists yet). Sorted, deduped, uppercased.
    """
    today = today or datetime.date.today()
    if not isinstance(calendar, dict):
        return []
    lo = today - datetime.timedelta(days=lookback_days)
    picked: set[str] = set()
    for ticker, dates in calendar.items():
        for d in _parse_dates(dates):
            if lo <= d <= today:
                picked.add(str(ticker).upper())
                break
    return sorted(picked)
