"""Campaign B5 lockstep tests: the umbrella ops scripts' NYSE session
helpers are now composed over the canonical
``renquant_common.market_calendar`` (orchestrator audit #296 §4.1). Only
umbrella *scripts* are re-pointed — the kernel mirror is untouched."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from renquant_common.market_calendar import (  # noqa: E402
    is_session,
    session_bounds,
)

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc


def test_preopen_gate_is_nyse_session_date_lockstep():
    from scripts import preopen_cancel_gate as gate

    for day, expected in [
        (dt.date(2026, 6, 30), True),   # regular Tuesday
        (dt.date(2026, 6, 28), False),  # Sunday
        (dt.date(2026, 7, 3), False),   # Independence Day observed
        (dt.date(2026, 11, 27), True),  # half day IS a session
    ]:
        assert gate._is_nyse_session_date(day) is expected
        assert gate._is_nyse_session_date(day) is is_session(day)


def test_preopen_gate_previous_nyse_close_lockstep():
    from scripts import preopen_cancel_gate as gate

    vectors = [
        # Monday 2026-07-06 08:00 ET pre-open -> Thursday 2026-07-02 16:00 close
        (dt.datetime(2026, 7, 6, 8, 0, tzinfo=ET), dt.date(2026, 7, 2)),
        # Half day: Friday 2026-11-27 14:00 ET -> that day's 13:00 EARLY close
        (dt.datetime(2026, 11, 27, 14, 0, tzinfo=ET), dt.date(2026, 11, 27)),
        # Exactly at the close instant: strictly-before => the PRIOR close
        (dt.datetime(2026, 6, 30, 16, 0, tzinfo=ET), dt.date(2026, 6, 29)),
    ]
    for now, expected_session in vectors:
        got = gate._previous_nyse_close(pd.Timestamp(now).tz_convert("UTC"))
        expected_close = pd.Timestamp(
            session_bounds(expected_session).close
        ).tz_convert("UTC")
        assert got == expected_close


def test_preopen_gate_previous_nyse_close_fail_closed(monkeypatch):
    from scripts import preopen_cancel_gate as gate
    from renquant_common import market_calendar as mc

    # Empty session window (unreachable on a real 14-day NYSE window, kept
    # fail-closed): the pre-B5 hand copy raised the same ValueError.
    monkeypatch.setattr(
        mc, "sessions_between", lambda *a, **k: pd.DatetimeIndex([])
    )
    with pytest.raises(ValueError, match="no recent sessions"):
        gate._previous_nyse_close(pd.Timestamp("2026-06-30 12:00", tz="UTC"))


def test_stops_liveness_market_session_open_lockstep():
    from scripts import check_software_stops_liveness as stops

    vectors = [
        (dt.datetime(2026, 6, 30, 12, 0, tzinfo=ET), True),    # mid-session
        (dt.datetime(2026, 6, 30, 8, 0, tzinfo=ET), False),    # pre-open
        (dt.datetime(2026, 6, 30, 16, 0, tzinfo=ET), True),    # close inclusive
        (dt.datetime(2026, 6, 30, 16, 1, tzinfo=ET), False),   # post-close
        (dt.datetime(2026, 6, 28, 12, 0, tzinfo=ET), False),   # Sunday
        (dt.datetime(2026, 11, 27, 12, 0, tzinfo=ET), True),   # half day, open
        (dt.datetime(2026, 11, 27, 14, 0, tzinfo=ET), False),  # half day, closed
    ]
    for now, expected in vectors:
        assert stops.market_session_open(now.astimezone(UTC)) is expected
        bounds = session_bounds(now.date())
        canonical = bounds is not None and bounds.open <= now <= bounds.close
        assert stops.market_session_open(now.astimezone(UTC)) is canonical


def test_no_direct_mcal_import_remains_in_scripts():
    """Umbrella scripts must consume renquant_common.market_calendar. The
    kernel mirror (pipeline-owned) is exempt by design — untouched by B5."""
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "scripts").glob("*.py")
        if "import pandas_market_calendars" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], (
        f"hand-rolled market-calendar use crept back into scripts/: "
        f"{offenders}; import renquant_common.market_calendar (campaign B5)"
    )
