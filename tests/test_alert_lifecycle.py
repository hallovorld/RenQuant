"""Alert escalation lifecycle tests (L6 sidecar §12.3)."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_STRATEGY_DIR = _Path(__file__).resolve().parent.parent / 'backtesting' / 'renquant_104'
if str(_STRATEGY_DIR) not in _sys.path:
    _sys.path.insert(0, str(_STRATEGY_DIR))

import datetime as dt

from kernel.alert_lifecycle import (
    CRITICAL,
    RESOLVED,
    WARN,
    AlertBook,
)

D0 = dt.date(2026, 2, 10)


def _day(n):
    return D0 + dt.timedelta(days=n)


class TestEscalation:
    def test_121_identical_obs_is_two_notifications(self):
        # The motivating incident: a 121-day-old condition is ONE incident
        # with exactly NEW + escalation = 2 ntfys, not 121.
        book = AlertBook(escalate_after_days=5)
        a = None
        for i in range(121):
            a = book.observe("staleness", "fund_daily", "max=2026-02-10", _day(i))
        assert a.notifications == 2
        assert a.state == CRITICAL
        assert "fund_daily" in book.blocking_scopes()

    def test_warn_before_escalation_window(self):
        book = AlertBook(escalate_after_days=5)
        a = None
        for i in range(4):  # day 0..3, < 5 days
            a = book.observe("x", "s", "h", _day(i))
        assert a.state == WARN and a.notifications == 1

    def test_acked_never_escalates(self):
        book = AlertBook(escalate_after_days=5)
        for i in range(3):
            book.observe("x", "s", "h", _day(i))
        book.ack("x", "s", "h")
        a = None
        for i in range(3, 30):
            a = book.observe("x", "s", "h", _day(i))
        assert a.state == WARN and a.notifications == 1
        assert book.blocking_scopes() == set()


class TestResolution:
    def test_absent_condition_resolves(self):
        book = AlertBook(escalate_after_days=5)
        for i in range(10):
            book.observe("x", "s", "h", _day(i))
        assert book.blocking_scopes() == {"s"}
        # next day the condition is gone (not in seen set)
        resolved = book.resolve_if_absent(seen_today=set(), today=_day(11))
        assert len(resolved) == 1 and resolved[0].state == RESOLVED
        assert book.blocking_scopes() == set()

    def test_resolved_then_recurs_is_new_incident(self):
        book = AlertBook(escalate_after_days=5)
        book.observe("x", "s", "h", _day(0))
        book.resolve_if_absent(seen_today=set(), today=_day(1))
        a = book.observe("x", "s", "h", _day(2))
        assert a.state == WARN and a.notifications == 1  # fresh incident
        assert a.first_seen == _day(2)

    def test_still_present_not_resolved(self):
        book = AlertBook(escalate_after_days=5)
        book.observe("x", "s", "h", _day(0))
        resolved = book.resolve_if_absent(seen_today={("x", "s", "h")},
                                          today=_day(1))
        assert resolved == []


class TestDedup:
    def test_distinct_causes_are_distinct_incidents(self):
        book = AlertBook()
        book.observe("staleness", "fund", "max=A", _day(0))
        book.observe("staleness", "fund", "max=B", _day(0))   # cause changed
        assert len(book.open_incidents()) == 2

    def test_distinct_scopes_separate(self):
        book = AlertBook(escalate_after_days=2)
        for i in range(5):
            book.observe("drift", "MU", "psi_hi", _day(i))
            book.observe("drift", "GE", "psi_hi", _day(i))
        assert book.blocking_scopes() == {"MU", "GE"}
