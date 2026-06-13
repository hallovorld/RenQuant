"""Property/invariant tests for the sim point-in-time leakage predicates.

Eng plan S2 item 6 (test-ladder rebalance). `adapters/sim_leakage_guards.py`
(sim.py decomposition slice 4) had NO direct test. These predicates decide
whether a correlation/GMM artifact's stamped as_of is strictly after the
backtest start — i.e. whether the run is about to use future information. A
false NEGATIVE here is silent lookahead bias (an invalid backtest); a false
POSITIVE needlessly discards a legitimate artifact. So the leakage boundary
deserves invariants, not just examples.

No `hypothesis` dependency (hermetic requirements.lock.txt lacks it): date
pairs are swept over a deterministic seeded grid.

Invariants pinned:
- leakage IFF as_of strictly AFTER start; same-day is NOT leakage (the guard
  admits as_of == start).
- fail-SAFE on None / unparseable / unstamped inputs → False (never substitute
  on missing provenance; let the hard guard decide).
- timezone invariance: the SAME instant expressed in different offsets yields
  the same verdict (everything is normalized to UTC before comparison).
- strict order: not both (a after b) and (b after a); equal ⇒ both False.
- gmm_leakage_present(art, start, extractor) ==
  corr_leakage_present(extractor(art), start) — the two predicates share one
  semantics, differing only in how as_of is sourced.
"""
from __future__ import annotations

import datetime
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.sim_leakage_guards import (  # noqa: E402
    corr_leakage_present,
    gmm_leakage_present,
)

SEED = 0xC0FFEE
N = 3000


def _rand_date(rng):
    base = datetime.date(2024, 1, 1)
    return base + datetime.timedelta(days=rng.randint(0, 1500))


class TestCorrLeakageCore:

    def test_leakage_iff_strictly_after(self):
        rng = random.Random(SEED)
        for _ in range(N):
            start = _rand_date(rng)
            delta = rng.choice([-30, -1, 0, 1, 30, rng.randint(-800, 800)])
            as_of = start + datetime.timedelta(days=delta)
            got = corr_leakage_present(str(as_of), str(start))
            assert got == (as_of > start), (as_of, start, got)

    def test_same_day_is_not_leakage(self):
        rng = random.Random(SEED + 1)
        for _ in range(500):
            d = _rand_date(rng)
            assert corr_leakage_present(str(d), str(d)) is False
            # also exercise timestamp-with-time on the same calendar day at
            # the same instant: still not leakage.
            ts = pd.Timestamp(d) + pd.Timedelta(hours=rng.randint(0, 23))
            assert corr_leakage_present(ts, ts) is False

    def test_none_and_unparseable_are_failsafe_false(self):
        rng = random.Random(SEED + 2)
        good = str(_rand_date(rng))
        for bad in (None, "not-a-date", "", "2026-13-99", object()):
            assert corr_leakage_present(bad, good) is False, bad
            assert corr_leakage_present(good, bad) is False, bad
        assert corr_leakage_present(None, None) is False

    def test_strict_order_antisymmetry(self):
        rng = random.Random(SEED + 3)
        for _ in range(N):
            a, b = _rand_date(rng), _rand_date(rng)
            ab = corr_leakage_present(str(a), str(b))
            ba = corr_leakage_present(str(b), str(a))
            assert not (ab and ba), (a, b)
            if a == b:
                assert not ab and not ba


class TestTimezoneInvariance:

    def test_same_instant_different_offsets_same_verdict(self):
        """An as_of timestamp expressed in two different UTC offsets is the
        same instant and must give the same leakage verdict against a fixed
        start. Guards against a tz-handling regression silently shifting the
        boundary by hours."""
        rng = random.Random(SEED + 4)
        for _ in range(N):
            # an aware UTC instant near the start boundary
            start = pd.Timestamp("2026-01-01T00:00:00+00:00")
            offset_hours = rng.choice([-12, -5, 0, 1, 5, 9, 14])
            instant = start + pd.Timedelta(hours=rng.randint(-48, 48))
            # express the SAME instant in two different zones
            tzA = instant.tz_convert("UTC")
            tzB = instant.tz_convert(f"Etc/GMT{'+' if offset_hours<=0 else '-'}{abs(offset_hours)}")
            a = corr_leakage_present(tzA, start)
            b = corr_leakage_present(tzB, start)
            assert a == b, (tzA, tzB, a, b)

    def test_aware_vs_equivalent_utc_naive(self):
        # A tz-aware instant and its UTC-naive equivalent compare identically
        # because the guard normalizes aware stamps to naive-UTC.
        aware = pd.Timestamp("2026-06-13T18:30:00+00:00")
        naive_utc = pd.Timestamp("2026-06-13T18:30:00")
        start = pd.Timestamp("2026-06-13T12:00:00")
        assert (corr_leakage_present(aware, start)
                == corr_leakage_present(naive_utc, start))


class TestGmmMatchesCorr:

    def test_gmm_equivalent_to_corr_via_extractor(self):
        rng = random.Random(SEED + 5)
        for _ in range(N):
            start = _rand_date(rng)
            as_of = start + datetime.timedelta(days=rng.randint(-400, 400))
            art = {"as_of_date": str(as_of)}
            ext = lambda a: a["as_of_date"]
            assert (gmm_leakage_present(art, str(start), ext)
                    == corr_leakage_present(str(as_of), str(start))), (as_of, start)

    def test_gmm_failsafe_paths(self):
        start = "2026-01-01"
        # None artifact, None start, extractor → None all fail safe to False
        assert gmm_leakage_present(None, start, lambda a: a) is False
        assert gmm_leakage_present({"x": 1}, None, lambda a: a["x"]) is False
        assert gmm_leakage_present({"as_of": None}, start,
                                   lambda a: a["as_of"]) is False
        # unparseable extracted value → False
        assert gmm_leakage_present({"as_of": "garbage"}, start,
                                   lambda a: a["as_of"]) is False
        # a clearly-after stamp → True
        assert gmm_leakage_present({"as_of": "2026-02-01"}, start,
                                   lambda a: a["as_of"]) is True
