"""PurgedKFold — time-series cross-validator with purging + embargo.

Faithful port of López de Prado AFML 2018 ch.7 Snippet 7.3
(``PurgedKFold``, pp. 105-108).

Why standard KFold fails in finance: labels generated from forward
returns (triple-barrier, fwd-N-day-ret) span multiple training
samples. Random shuffling leaks future information into past training
data and yields wildly optimistic CV scores. PurgedKFold fixes this by:

1. PURGING: training samples whose label-period (event_date,
   event_date + horizon_days) overlaps with the test fold's date
   range are dropped from training.
2. EMBARGO: an additional buffer of ``pct_embargo × N`` consecutive
   samples after each test fold is dropped from training to absorb
   serial-correlation effects beyond the label horizon.

mlfinlab's wrapper exposed this as
``mlfinlab.cross_validation.PurgedKFold`` but moved behind the Hudson &
Thames commercial tier in 2023. This module re-ports from the textbook
pseudocode + holds the same API surface (n_splits, event_times,
label_horizon_days, pct_embargo).

References
----------
* López de Prado 2018 *Advances in Financial Machine Learning* (Wiley)
    ch.7.3 "Solution 1: Purging" pp. 103-105
    ch.7.4 "Solution 2: Embargo"  pp. 107-108
    Snippet 7.3 — PurgedKFold reference implementation
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


class PurgedKFold:
    """Time-series K-fold with purging and embargo.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds. Last fold may be slightly smaller if N not
        divisible.
    event_times : pd.Series
        Time of each event (one per training sample), in original
        sample order. Used for purging decisions.
    label_horizon_days : int, default 0
        How many BUSINESS days the label lookahead covers. 0 disables
        purging (label is computed at the event time itself).
    pct_embargo : float, default 0.0
        Fraction of total samples to embargo after each test fold.
        Common choices: 0.01-0.05.

    Example
    -------
    >>> times = pd.Series(pd.bdate_range("2024-01-01", periods=100))
    >>> cv = PurgedKFold(n_splits=5, event_times=times, label_horizon_days=20)
    >>> for train, test in cv.split(np.arange(100)):
    ...     model.fit(X[train], y[train])
    ...     scores.append(model.score(X[test], y[test]))
    """

    def __init__(
        self,
        n_splits:               int,
        event_times:            pd.Series,
        label_horizon_days:     int = 0,
        pct_embargo:            float = 0.0,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be ≥ 2, got {n_splits}")
        if label_horizon_days < 0:
            raise ValueError(f"label_horizon_days must be ≥ 0, got {label_horizon_days}")
        if pct_embargo < 0 or pct_embargo > 0.5:
            raise ValueError(f"pct_embargo must be in [0, 0.5], got {pct_embargo}")
        self.n_splits           = n_splits
        self.event_times        = pd.Series(event_times).reset_index(drop=True)
        self.label_horizon_days = int(label_horizon_days)
        self.pct_embargo        = float(pct_embargo)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) pairs per AFML Snippet 7.3."""
        n = len(X)
        if n != len(self.event_times):
            raise ValueError(
                f"len(X)={n} does not match len(event_times)="
                f"{len(self.event_times)}"
            )
        indices = np.arange(n)
        # Contiguous test folds — preserves time order.
        # Last fold absorbs the remainder.
        fold_size = n // self.n_splits
        embargo_n = int(round(self.pct_embargo * n))

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end   = (k + 1) * fold_size if k < self.n_splits - 1 else n
            test_idx   = indices[test_start:test_end]
            test_start_date = self.event_times.iloc[test_start]
            test_end_date   = self.event_times.iloc[test_end - 1]

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start:test_end] = False

            # Purging: drop any sample whose label_period overlaps test
            # window. For event at idx i:
            #   label_period = [event_times[i], event_times[i] + horizon]
            #   overlaps test if event_times[i] <= test_end_date
            #                  AND label_end_date >= test_start_date
            if self.label_horizon_days > 0:
                horizon_off = pd.offsets.BDay(self.label_horizon_days)
                # Drop earlier samples whose forward label spills into test
                event_dates = self.event_times
                label_end   = event_dates + horizon_off
                overlap = (label_end >= test_start_date) & (event_dates <= test_end_date)
                train_mask &= ~overlap.values

            # Embargo: drop embargo_n samples immediately after test fold
            if embargo_n > 0 and test_end < n:
                embargo_stop = min(test_end + embargo_n, n)
                train_mask[test_end:embargo_stop] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx
