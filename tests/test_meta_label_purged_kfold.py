"""TDD — PurgedKFold cross-validator.

Faithful port of López de Prado AFML 2018 ch.7 Snippet 7.3 (pp. 105-108).

Contract:
  * For each test fold spanning [test_start, test_end]:
      - Training set EXCLUDES samples whose label-period overlaps test
        (purging: events with date in [test_start - label_lookahead,
        test_end] are dropped from train)
      - An embargo of pct_embargo × N samples is removed after each
        test fold from the training set (prevents post-test contamination)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.meta_label.purged_kfold import PurgedKFold  # noqa: E402


def _make_event_times(n: int, start="2024-01-01") -> pd.Series:
    """Business-day spaced event times for n samples."""
    return pd.Series(pd.bdate_range(start=start, periods=n))


class TestPurgedKFoldShape:

    def test_n_splits_yields_n_split_iterations(self):
        n = 100
        times = _make_event_times(n)
        cv = PurgedKFold(n_splits=5, event_times=times, label_horizon_days=20)
        splits = list(cv.split(np.arange(n)))
        assert len(splits) == 5

    def test_train_and_test_disjoint(self):
        n = 100
        times = _make_event_times(n)
        cv = PurgedKFold(n_splits=4, event_times=times, label_horizon_days=20)
        for train_idx, test_idx in cv.split(np.arange(n)):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_every_sample_used_for_test_exactly_once(self):
        n = 60
        times = _make_event_times(n)
        cv = PurgedKFold(n_splits=3, event_times=times, label_horizon_days=10)
        all_test = []
        for _, test_idx in cv.split(np.arange(n)):
            all_test.extend(test_idx.tolist())
        # Each sample should appear exactly once across all test folds.
        from collections import Counter
        c = Counter(all_test)
        for cnt in c.values():
            assert cnt == 1


class TestPurgedKFoldPurging:

    def test_purges_events_overlapping_test_label_window(self):
        # Setup: 100 daily events; 5-day label horizon; 5 splits
        # Test fold 0 = samples [0..19]
        # Label period for an event at idx i runs through i + horizon
        # Train must exclude any event whose label_period intersects
        # [test_start_date, test_end_date]
        n = 100
        times = _make_event_times(n)
        horizon = 10
        cv = PurgedKFold(n_splits=5, event_times=times,
                         label_horizon_days=horizon)
        splits = list(cv.split(np.arange(n)))

        # Pick a MIDDLE fold so we can check purging on both sides
        train_idx, test_idx = splits[2]  # samples 40..59
        test_start_date = times.iloc[test_idx[0]]
        test_end_date   = times.iloc[test_idx[-1]]

        # Any event whose label_end (event + horizon days) >= test_start_date
        # AND whose event_date <= test_end_date must NOT be in train
        for ti in train_idx:
            ev_date = times.iloc[ti]
            label_end = ev_date + pd.offsets.BDay(horizon)
            overlap = (label_end >= test_start_date) and (ev_date <= test_end_date)
            assert not overlap, (
                f"train idx {ti} (ev={ev_date.date()} label_end={label_end.date()}) "
                f"overlaps test [{test_start_date.date()}, {test_end_date.date()}]"
            )

    def test_zero_horizon_equals_plain_kfold(self):
        # With horizon=0 + pct_embargo=0, every sample not in test is in train
        n = 30
        times = _make_event_times(n)
        cv = PurgedKFold(n_splits=3, event_times=times,
                         label_horizon_days=0, pct_embargo=0.0)
        for train_idx, test_idx in cv.split(np.arange(n)):
            covered = set(train_idx) | set(test_idx)
            assert covered == set(range(n))


class TestPurgedKFoldEmbargo:

    def test_embargo_removes_samples_after_test_fold(self):
        # 100 events, 5 splits → 20 per fold
        # embargo=0.05 of 100 = 5 samples
        n = 100
        times = _make_event_times(n)
        cv = PurgedKFold(n_splits=5, event_times=times,
                         label_horizon_days=0,  # disable purging
                         pct_embargo=0.05)
        splits = list(cv.split(np.arange(n)))

        # Fold 0 covers samples [0..19]; embargo removes [20..24] from train
        train_idx, test_idx = splits[0]
        assert 25 in train_idx
        assert 20 not in train_idx
        assert 21 not in train_idx
        assert 24 not in train_idx
