"""`sample_end` must not be able to become a wall again (orch#1015).

WHAT WENT WRONG. `sample_end` was a literal `"2026-06-30"` in the served
strategy config, set once at bootstrap on 2026-05-25 — 36 days in the future at
the time — and never touched again (the only later commits to that line are
whitespace reflows). It carried no `_reason` note in a config that documents
every deliberate choice, so it was headroom, not policy.

The calendar overran it and the wall stopped moving. On the 2026-08-16
tournament run, `DataFetchJob` fetched `2016-01-01 -> 2026-06-30` and loaded
2637 rows per ticker while the on-disk store held 2672 — **35 fresh trading
days per ticker fetched away, every week**. With the tournament's 5-day label
lookahead the feature frame ended 2026-06-23, so `today - frame_end` grew 7 days
a week until it crossed the acceptance gate's 45-day cap on 2026-08-09, after
which all 142 per-ticker candidates were rejected every week and the incumbents
kept. Nothing was broken: the gate was right, the data was fine, a hand-set
bound had become a wall.

These tests pin the two properties that keep both halves true — an explicit
date still pins the window, and the absence of one follows the calendar.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_103"))

from kernel.data import resolve_sample_end  # noqa: E402


class TestAnExplicitDateStillPins:
    """Reproducible backtests must keep working byte-identically."""

    def test_explicit_date_is_returned_unchanged(self):
        assert resolve_sample_end({"sample_end": "2026-06-30"},
                                  today="2026-08-20") == "2026-06-30"

    def test_an_explicit_date_wins_over_the_clock(self):
        """A pinned window is pinned even when it is far in the past — that is
        the whole point of pinning one."""
        assert resolve_sample_end({"sample_end": "2020-01-01"}) == "2020-01-01"


class TestAbsenceFollowsTheCalendar:
    """The regression: a config that does not pin must not silently freeze."""

    def test_null_resolves_to_today(self):
        assert resolve_sample_end({"sample_end": None}, today="2026-08-20") == "2026-08-20"

    def test_absent_key_resolves_to_today(self):
        assert resolve_sample_end({}, today="2026-08-20") == "2026-08-20"

    def test_empty_string_is_not_a_date(self):
        """An empty string is a config typo, not a bound. Treating it as one
        would reproduce the wall with an invisible cause."""
        assert resolve_sample_end({"sample_end": ""}, today="2026-08-20") == "2026-08-20"

    def test_the_real_clock_path_returns_an_iso_date(self):
        """`today=` exists for tests; the production path reads the clock, and
        it must produce something `fetch_ohlcv` and `has_range` can parse."""
        got = resolve_sample_end({})
        assert dt.date.fromisoformat(got) == dt.date.today()


class TestItNeverReturnsNone:
    """Returning None would 'fix' the wall by DISABLING a different guard.

    `ParquetStore.has_range` skips its staleness check entirely when `end` is
    falsy (`if end and df.index.max() < ...`), so an unpinned window expressed
    as None would silently stop verifying that the cache is fresh — trading one
    silent failure for another. A concrete date keeps that check alive.
    """

    def test_every_unpinned_form_yields_a_concrete_date(self):
        for cfg in ({}, {"sample_end": None}, {"sample_end": ""}):
            got = resolve_sample_end(cfg, today="2026-08-20")
            assert got, cfg
            assert dt.date.fromisoformat(got)
