"""Purged K-fold cross-validation with embargo (AFML ch.7).

Standard K-fold is biased for time-series panels because label windows
overlap — the train and test labels share information. We fix that by:

  1. Splitting the set of *unique dates* into K contiguous folds.
     Fold k's test set = rows whose date falls in dates[k].
  2. **Purge**: remove training rows whose label window overlaps the test
     dates — i.e., rows dated in [test_start − lookahead_days + 1, test_end].
  3. **Embargo**: additionally drop training rows in
     (test_end, test_end + embargo_days] to guard against post-fold leakage
     when features include slow-moving trailing stats.

This yields clean out-of-fold predictions that we can pool into an IC
(information coefficient) estimate.

Public API::

    PurgedKFold(n_splits, embargo_days, lookahead_days).split(panel)
        ─► iterator of (train_idx, test_idx)
    evaluate_fold_ic(model, panel, feature_cols, label_col, test_idx)
        ─► Series of per-date Spearman IC on the test slice
    cross_validated_ic(model_factory, panel, feature_cols, label_col, cv)
        ─► {'mean_ic', 'std_ic', 'per_fold_ic', 'per_fold_ic_series'}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class PurgedKFold:
    n_splits: int = 5
    embargo_days: int = 5
    lookahead_days: int = 5

    def split(
        self, panel: pd.DataFrame, date_col: str = "date",
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield `(train_idx, test_idx)` arrays of *positional* row indices.

        Splits on unique sorted dates; within each fold, every row whose
        date falls in the fold's date range is a test row.
        """
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        dates = pd.to_datetime(panel[date_col]).values
        unique_dates = np.array(sorted(set(dates)))
        n_dates = len(unique_dates)
        if n_dates < self.n_splits:
            raise ValueError(
                f"Not enough unique dates ({n_dates}) for {self.n_splits}-fold CV",
            )

        # Contiguous date folds
        fold_edges = np.linspace(0, n_dates, self.n_splits + 1, dtype=int)

        all_idx = np.arange(len(panel), dtype=np.int64)

        for k in range(self.n_splits):
            lo, hi = fold_edges[k], fold_edges[k + 1]
            test_dates = unique_dates[lo:hi]
            test_start = test_dates[0]
            test_end   = test_dates[-1]

            test_mask = np.isin(dates, test_dates)
            test_idx = all_idx[test_mask]

            # Purge window: [test_start − L + 1, test_end]
            purge_start = pd.Timestamp(test_start) - pd.Timedelta(days=int(self.lookahead_days) - 1)
            # Embargo window: (test_end, test_end + embargo_days]
            embargo_end = pd.Timestamp(test_end) + pd.Timedelta(days=int(self.embargo_days))

            train_mask = ~test_mask
            # Drop training rows inside the purge window (before test start,
            # but whose label window leaks into test)
            leak_mask = (dates >= np.datetime64(purge_start)) & (dates < np.datetime64(test_start))
            train_mask &= ~leak_mask
            # Drop embargo rows (after test end, too close to it)
            emb_mask = (dates > np.datetime64(test_end)) & (dates <= np.datetime64(embargo_end))
            train_mask &= ~emb_mask

            train_idx = all_idx[train_mask]
            yield train_idx, test_idx


def evaluate_fold_ic(
    model,
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    test_idx: np.ndarray,
    *,
    date_col: str = "date",
) -> pd.Series:
    """Per-date Spearman IC on the test slice.

    `model` must implement `.predict(X)` where X is a DataFrame of
    feature_cols.
    """
    sub = panel.iloc[test_idx]
    X = sub[feature_cols]
    y = sub[label_col].values
    preds = model.predict(X)
    dates = sub[date_col].values

    df = pd.DataFrame({"date": dates, "pred": preds, "y": y})
    out: dict = {}
    for d, g in df.groupby("date", sort=True):
        y_g = g["y"].values
        p_g = g["pred"].values
        if len(y_g) < 2 or np.all(y_g == y_g[0]) or np.all(p_g == p_g[0]):
            continue
        rho, _ = spearmanr(p_g, y_g)
        if np.isnan(rho):
            continue
        out[d] = rho
    return pd.Series(out).sort_index()


def cross_validated_ic(
    model_factory: Callable,
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cv: PurgedKFold,
    weight_col: str | None = "weight",
) -> dict:
    """Fit `model_factory()` on each CV train fold, score on test.

    Returns dict with:
      - mean_ic: grand-average of per-date ICs across folds
      - std_ic:  std across folds of the fold-level mean IC
      - per_fold_ic: list of per-fold mean ICs
      - per_fold_ic_series: list of per-date IC Series per fold
    """
    per_fold_mean: list[float] = []
    per_fold_series: list[pd.Series] = []

    for train_idx, test_idx in cv.split(panel):
        model = model_factory()
        tr = panel.iloc[train_idx]
        X_tr = tr[feature_cols]
        y_tr = tr[label_col].values
        w_tr = tr[weight_col].values if weight_col and weight_col in tr.columns else None

        try:
            if w_tr is not None:
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                model.fit(X_tr, y_tr)
        except TypeError:
            # Model doesn't accept sample_weight
            model.fit(X_tr, y_tr)

        ic_s = evaluate_fold_ic(
            model, panel, feature_cols, label_col, test_idx, date_col="date",
        )
        per_fold_series.append(ic_s)
        if len(ic_s) > 0:
            per_fold_mean.append(float(ic_s.mean()))

    per_fold_mean_arr = np.asarray(per_fold_mean)
    return {
        "mean_ic": float(per_fold_mean_arr.mean()) if len(per_fold_mean_arr) else float("nan"),
        "std_ic":  float(per_fold_mean_arr.std(ddof=1))
                   if len(per_fold_mean_arr) > 1 else float("nan"),
        "per_fold_ic": per_fold_mean,
        "per_fold_ic_series": per_fold_series,
    }
