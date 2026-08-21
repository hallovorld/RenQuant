"""`sample_end` must not be able to become a wall again (orch#1015).

WHAT WENT WRONG. `sample_end` in the tournament's config
(`backtesting/renquant_104/strategy_config.json`) was the literal
`"2026-06-30"`, set once at bootstrap on 2026-05-25 — 36 days in the FUTURE at
the time — and never changed; the only later commits to that line are
whitespace reflows. It carries no `_reason` note in a config that documents
every deliberate choice, and `kernel/tournament_acceptance.py` says in passing
that "sample_end is bumped manually". It was headroom, not policy.

The calendar overran it and the wall stopped moving. On the 2026-08-16
tournament run `DataFetchJob` fetched `2016-01-01 -> 2026-06-30` and loaded
2637 rows per ticker while the on-disk store held 2672 — **35 fresh trading
days per ticker fetched away, every week**. With the 5-day label lookahead the
feature frame ended 2026-06-23, so `today - frame_end` grew 7 days a week until
it crossed the acceptance gate's 45-day cap on 2026-08-09, after which all 142
per-ticker candidates were rejected weekly and the incumbents kept. The gate
was right, the data was fine, a hand-set bound had become a wall.

WHICH COPY. These import from `backtesting/renquant_104/`, which is the copy
the tournament actually runs — verified by resolving `kernel.data` under
`weekly_tournament_retrain.sh`'s own PYTHONPATH, not by reading the repo
layout. An earlier attempt at this fix patched `renquant_103/`, which nothing
imports.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.data import resolve_sample_end  # noqa: E402


class TestAnExplicitDateStillPins:
    """Reproducible backtests must keep working byte-identically."""

    def test_explicit_date_is_returned_unchanged(self):
        assert resolve_sample_end({"sample_end": "2026-06-30"},
                                  today="2026-08-21") == "2026-06-30"

    def test_an_explicit_date_wins_over_the_clock_even_when_ancient(self):
        """A pinned window is pinned even far in the past — that is the point
        of pinning one, and it is what keeps historical backtests stable."""
        assert resolve_sample_end({"sample_end": "2020-01-01"}) == "2020-01-01"


class TestAbsenceFollowsTheCalendar:
    """The regression: a config that does not pin must not silently freeze."""

    def test_null_resolves_to_today(self):
        assert resolve_sample_end({"sample_end": None}, today="2026-08-21") == "2026-08-21"

    def test_absent_key_resolves_to_today(self):
        assert resolve_sample_end({}, today="2026-08-21") == "2026-08-21"

    def test_empty_string_is_not_a_date(self):
        """An empty string is a config typo, not a bound. Honouring it would
        reproduce the wall with an invisible cause."""
        assert resolve_sample_end({"sample_end": ""}, today="2026-08-21") == "2026-08-21"

    def test_the_real_clock_path_returns_a_parseable_iso_date(self):
        """`today=` exists for tests; production reads the clock and must
        produce something `fetch_ohlcv` and `has_range` can parse."""
        got = resolve_sample_end({})
        assert dt.date.fromisoformat(got) == dt.date.today()


class TestItNeverReturnsNone:
    """Not for the reason an earlier draft gave.

    That draft claimed `has_range` skips its staleness check when `end` is
    falsy, so returning None would fix the wall by disabling a freshness guard.
    **That is false in this copy** — the 2026-05-03 P0 fix removed the
    `end=None` short-circuit, and `has_range` now derives
    `ref = _market_timestamp(end)` and enforces NYSE-aware staleness against
    the wall clock when `end` is None. It was true of `renquant_103/`, which
    the tournament does not run.

    The real reasons are smaller: `pp_training.py` reads `cfg["sample_end"]` as
    a hard subscript, so an absent key raises KeyError; and a concrete date
    keeps the fetch window legible in the run log instead of printing `None`.
    """

    def test_every_unpinned_form_yields_a_concrete_parseable_date(self):
        for cfg in ({}, {"sample_end": None}, {"sample_end": ""}):
            got = resolve_sample_end(cfg, today="2026-08-21")
            assert got, cfg
            assert dt.date.fromisoformat(got)


class TestBothCallSitesUseTheSharedDefinition:
    """One resolution, two readers — they must not drift apart again.

    Source-level on purpose and narrowly: the assertion is that neither reader
    still takes `sample_end` straight off the config, which is a property of
    the wiring, not of any single run.
    """

    def _src(self, rel: str) -> str:
        return (REPO_ROOT / "backtesting" / "renquant_104" / rel).read_text(encoding="utf-8")

    def test_tournament_path_resolves(self):
        s = self._src("kernel/pipeline/pp_training.py")
        assert "resolve_sample_end(cfg)" in s
        assert 'end   = cfg["sample_end"]' not in s

    def test_panel_path_resolves(self):
        s = self._src("training_panel/pp_panel_training.py")
        assert "resolve_sample_end(cfg)" in s
        assert 'end   = cfg.get("sample_end")' not in s
