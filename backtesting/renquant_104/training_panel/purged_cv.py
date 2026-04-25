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

            # Purge window: a row dated d carries label ret(d → d+L). It
            # leaks into test when d + L >= test_start, i.e. d >= test_start - L.
            # Round-2 audit (#R2-30): prior code used `lookahead - 1` which
            # purged 4 days for L=5, leaving the d = test_start - L row's
            # label looking forward INTO the test window. Now uses the full L.
            purge_start = pd.Timestamp(test_start) - pd.Timedelta(days=int(self.lookahead_days))
            # Embargo window: (test_end, test_end + embargo_days]
            embargo_end = pd.Timestamp(test_end) + pd.Timedelta(days=int(self.embargo_days))

            train_mask = ~test_mask
            # Drop training rows inside the purge window (before test start,
            # but whose label window leaks into test) — INCLUSIVE on both ends
            # so test_start - L is also dropped.
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


# ─────────────────────────────────────────────────────────────────────────────
# Combinatorial Purged CV (López de Prado AFML §12)
# Instead of 1 train/test split per fold, we enumerate C(N, k) combinations of
# k test groups out of N. Result: a richer distribution of OOS IC estimates.
# ─────────────────────────────────────────────────────────────────────────────

from itertools import combinations


@dataclass
class CombinatorialPurgedCV:
    """Purged K-fold where each split tests on `n_test_groups` groups.

    With `n_splits=6, n_test_groups=2`, yields C(6, 2) = 15 distinct
    train/test splits — each a valid OOS estimate. Aggregate mean/std/quantiles
    give a distribution of IC rather than a single noisy mean.

    When `n_test_groups == 1` this reduces to standard PurgedKFold.
    """
    n_splits: int = 6
    n_test_groups: int = 2
    embargo_days: int = 5
    lookahead_days: int = 5

    def split(
        self, panel: pd.DataFrame, date_col: str = "date",
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.n_test_groups < 1 or self.n_test_groups >= self.n_splits:
            raise ValueError(
                f"n_test_groups must be in [1, n_splits-1] "
                f"(got {self.n_test_groups}, n_splits={self.n_splits})",
            )

        dates = pd.to_datetime(panel[date_col]).values
        unique_dates = np.array(sorted(set(dates)))
        n_dates = len(unique_dates)
        if n_dates < self.n_splits:
            raise ValueError(
                f"Not enough unique dates ({n_dates}) for {self.n_splits}-fold CV",
            )

        fold_edges = np.linspace(0, n_dates, self.n_splits + 1, dtype=int)
        groups = [unique_dates[fold_edges[k]:fold_edges[k + 1]]
                  for k in range(self.n_splits)]

        all_idx = np.arange(len(panel), dtype=np.int64)

        for combo in combinations(range(self.n_splits), self.n_test_groups):
            test_dates = np.concatenate([groups[k] for k in combo])
            test_mask  = np.isin(dates, test_dates)
            test_idx   = all_idx[test_mask]

            # Build purge + embargo windows for each contiguous test block.
            # Since combo might be non-contiguous (e.g. folds 0 and 3),
            # we apply purge/embargo to each selected group separately.
            train_mask = ~test_mask
            for k in combo:
                block = groups[k]
                block_start = block[0]
                block_end   = block[-1]
                purge_start = pd.Timestamp(block_start) - pd.Timedelta(
                    days=int(self.lookahead_days) - 1,
                )
                embargo_end = pd.Timestamp(block_end) + pd.Timedelta(
                    days=int(self.embargo_days),
                )
                leak_mask = (
                    (dates >= np.datetime64(purge_start))
                    & (dates < np.datetime64(block_start))
                )
                train_mask &= ~leak_mask
                emb_mask = (
                    (dates > np.datetime64(block_end))
                    & (dates <= np.datetime64(embargo_end))
                )
                train_mask &= ~emb_mask

            train_idx = all_idx[train_mask]
            yield train_idx, test_idx


def cross_validated_ic_cpcv(
    model_factory: Callable,
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cv: CombinatorialPurgedCV,
    weight_col: str | None = "weight",
) -> dict:
    """Same as `cross_validated_ic` but returns a distribution of IC.

    Returns dict with:
      - mean_ic, std_ic   — across splits
      - quantiles         — dict of {0.05, 0.25, 0.5, 0.75, 0.95}
      - per_fold_ic       — list of per-split mean ICs
      - per_fold_ic_series — list of per-date IC Series per split
    """
    result = cross_validated_ic(
        model_factory, panel, feature_cols, label_col, cv, weight_col,
    )
    fold_ics = np.asarray(result["per_fold_ic"], dtype=float)
    qs = np.quantile(fold_ics, [0.05, 0.25, 0.5, 0.75, 0.95]) if len(fold_ics) else np.full(5, np.nan)
    result["quantiles"] = {
        "q05": float(qs[0]), "q25": float(qs[1]), "q50": float(qs[2]),
        "q75": float(qs[3]), "q95": float(qs[4]),
    }
    return result
