"""Invariant tests for _held_mark_ohlcv_frame (runner held-position risk frame).

Eng plan S2 item 6 (test-ladder rebalance). adapters/runner_prep.py's
_held_mark_ohlcv_frame was untested. make_context calls it to synthesize a
one-bar OHLCV frame from a broker mark when a held position has no fresh
OHLCV, so its risk/sell checks still have a price. A wrong frame here mis-runs
a held position's stop / sell logic on a live book.

No `hypothesis` dependency (hermetic requirements.lock.txt lacks it): prices
and frame shapes are swept over a deterministic seeded grid.

Invariants pinned:
- a bad mark (non-finite, <= 0, unparseable) yields None — no synthetic bar
  from junk.
- a valid mark with no base frame yields exactly one bar at `today` whose
  O=H=L=C == the mark and volume == 0.
- with a base frame: today's stale bar is replaced by the mark bar, the index
  stays datetime + sorted, and duplicate timestamps are de-duped (keep last).
- any failure massaging the base frame degrades to the lone mark bar (never
  raises).
"""
from __future__ import annotations

import datetime
import math
import random
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.runner_prep import _held_mark_ohlcv_frame  # noqa: E402

TODAY = datetime.date(2026, 6, 14)
SEED = 0xB0A7


class TestBadPrice:
    def test_non_finite_non_positive_unparseable_yield_none(self):
        for bad in (0.0, -1.0, -1e6, float("nan"), float("inf"),
                    float("-inf"), "x", None):
            assert _held_mark_ohlcv_frame("AAPL", TODAY, bad) is None, bad


class TestLoneMarkBar:
    def test_single_bar_ohlc_equals_mark_volume_zero(self):
        rng = random.Random(SEED)
        for _ in range(2000):
            px = rng.uniform(0.01, 5000)
            out = _held_mark_ohlcv_frame("AAPL", TODAY, px)
            assert list(out.columns) == ["open", "high", "low", "close", "volume"]
            assert len(out) == 1
            row = out.iloc[0]
            assert row["open"] == px == row["high"] == row["low"] == row["close"]
            assert row["volume"] == 0.0
            assert out.index[0] == pd.Timestamp(TODAY)

    def test_empty_base_frame_treated_as_none(self):
        out = _held_mark_ohlcv_frame("AAPL", TODAY, 100.0,
                                     base_df=pd.DataFrame())
        assert len(out) == 1 and out["close"].iloc[0] == 100.0


class TestMergeWithBaseFrame:
    def _hist(self, days, today=TODAY):
        idx = pd.to_datetime([today - datetime.timedelta(days=d)
                              for d in range(days, 0, -1)])
        return pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 9.0},
            index=idx)

    def test_appends_mark_and_keeps_sorted(self):
        base = self._hist(5)
        out = _held_mark_ohlcv_frame("AAPL", TODAY, 250.0, base_df=base)
        assert list(out.index) == sorted(out.index)
        assert out.index[-1] == pd.Timestamp(TODAY)
        assert out["close"].iloc[-1] == 250.0  # the mark bar is last

    def test_replaces_today_stale_bar(self):
        # base already has a (stale) bar dated today → it must be dropped in
        # favor of the fresh mark, not duplicated.
        idx = pd.to_datetime([TODAY - datetime.timedelta(days=1), TODAY])
        base = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 99.0, "volume": 9.0},
            index=idx)
        out = _held_mark_ohlcv_frame("AAPL", TODAY, 250.0, base_df=base)
        today_rows = out[out.index == pd.Timestamp(TODAY)]
        assert len(today_rows) == 1
        assert today_rows["close"].iloc[0] == 250.0  # mark won, stale 99 gone

    def test_dedups_duplicate_timestamps_keep_last(self):
        d = TODAY - datetime.timedelta(days=1)
        idx = pd.to_datetime([d, d])  # duplicate prior-day stamps
        base = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 2.0],
             "volume": 9.0}, index=idx)
        out = _held_mark_ohlcv_frame("AAPL", TODAY, 250.0, base_df=base)
        prior = out[out.index == pd.Timestamp(d)]
        assert len(prior) == 1 and prior["close"].iloc[0] == 2.0  # keep last

    def test_string_indexed_base_is_coerced(self):
        # base_df with a non-datetime (string) index must still merge.
        base = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 9.0},
            index=["2026-06-12", "2026-06-13"])
        out = _held_mark_ohlcv_frame("AAPL", TODAY, 250.0, base_df=base)
        assert out.index[-1] == pd.Timestamp(TODAY)
        assert out["close"].iloc[-1] == 250.0

    def test_unmassageable_base_degrades_to_mark_bar(self):
        # a base_df that explodes when copied/indexed → fall back to mark bar,
        # never raise.
        class _Exploding(pd.DataFrame):
            @property
            def _constructor(self):
                return _Exploding

            def copy(self, *a, **k):
                raise RuntimeError("cannot copy")

        base = _Exploding({"close": [1.0]})
        out = _held_mark_ohlcv_frame("AAPL", TODAY, 250.0, base_df=base)
        assert len(out) == 1 and out["close"].iloc[0] == 250.0


class TestMarkBarValuesAcrossPrices:
    def test_ohlc_invariant_holds_under_sweep(self):
        rng = random.Random(SEED + 1)
        base = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 9.0},
            index=pd.to_datetime([TODAY - datetime.timedelta(days=2),
                                  TODAY - datetime.timedelta(days=1)]))
        for _ in range(1000):
            px = rng.uniform(0.01, 9000)
            out = _held_mark_ohlcv_frame("AAPL", TODAY, px, base_df=base)
            mark = out.iloc[-1]
            assert mark["open"] == mark["high"] == mark["low"] == mark["close"] == px
            assert mark["volume"] == 0.0
            assert math.isfinite(out["close"].iloc[-1])
