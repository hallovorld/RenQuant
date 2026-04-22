"""Tests for training_panel/purged_cv.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_panel(n_dates: int = 60, n_tickers: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    for d in dates:
        for i in range(n_tickers):
            rows.append({
                "date":   d,
                "ticker": f"T{i}",
                "x1":     rng.normal(),
                "x2":     rng.normal(),
                "label":  rng.normal(),
                "weight": 1.0,
            })
    df = pd.DataFrame(rows)
    return df


class TestPurgedKFoldSplit:
    def test_each_row_in_exactly_one_test_fold(self):
        from training_panel.purged_cv import PurgedKFold
        panel = _make_panel(n_dates=50, n_tickers=4, seed=1)
        cv = PurgedKFold(n_splits=5, embargo_days=0, lookahead_days=1)
        counts = np.zeros(len(panel), dtype=int)
        for _, test_idx in cv.split(panel):
            counts[test_idx] += 1
        assert (counts == 1).all()

    def test_purge_removes_rows_with_overlap_into_test(self):
        from training_panel.purged_cv import PurgedKFold
        panel = _make_panel(n_dates=40, n_tickers=3, seed=2)
        cv = PurgedKFold(n_splits=4, embargo_days=0, lookahead_days=5)
        for train_idx, test_idx in cv.split(panel):
            test_dates = pd.to_datetime(panel.iloc[test_idx]["date"]).values
            train_dates = pd.to_datetime(panel.iloc[train_idx]["date"]).values
            test_start = test_dates.min()
            purge_lo = test_start - np.timedelta64(5 - 1, "D")
            # No train date in [purge_lo, test_start)
            leak = (train_dates >= purge_lo) & (train_dates < test_start)
            assert not leak.any(), f"purge failed for test_start={test_start}"

    def test_embargo_removes_post_fold_bars(self):
        from training_panel.purged_cv import PurgedKFold
        panel = _make_panel(n_dates=40, n_tickers=3, seed=3)
        cv = PurgedKFold(n_splits=4, embargo_days=5, lookahead_days=0)
        for train_idx, test_idx in cv.split(panel):
            test_dates = pd.to_datetime(panel.iloc[test_idx]["date"]).values
            train_dates = pd.to_datetime(panel.iloc[train_idx]["date"]).values
            test_end = test_dates.max()
            emb_hi = test_end + np.timedelta64(5, "D")
            # No train date in (test_end, emb_hi]
            leak = (train_dates > test_end) & (train_dates <= emb_hi)
            assert not leak.any(), f"embargo failed for test_end={test_end}"

    def test_raises_on_too_few_dates(self):
        from training_panel.purged_cv import PurgedKFold
        panel = _make_panel(n_dates=3, n_tickers=2, seed=4)
        cv = PurgedKFold(n_splits=5)
        with pytest.raises(ValueError):
            list(cv.split(panel))

    def test_raises_on_n_splits_one(self):
        from training_panel.purged_cv import PurgedKFold
        panel = _make_panel(n_dates=20, n_tickers=2, seed=5)
        cv = PurgedKFold(n_splits=1)
        with pytest.raises(ValueError):
            list(cv.split(panel))


class _PerfectModel:
    """Model that predicts y exactly — for testing IC=1."""
    def fit(self, X, y, sample_weight=None): pass
    def predict(self, X): return X["_y_"].values


class _RandomModel:
    """Model that predicts random — for testing IC≈0."""
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def fit(self, X, y, sample_weight=None): pass
    def predict(self, X): return self.rng.normal(0, 1, len(X))


class TestEvaluateFoldIc:
    def test_fold_ic_equals_1_on_perfect_signal(self):
        from training_panel.purged_cv import evaluate_fold_ic
        panel = _make_panel(n_dates=10, n_tickers=6, seed=10)
        # Inject a perfect feature = label
        panel["_y_"] = panel["label"].values
        test_idx = np.arange(len(panel), dtype=np.int64)
        ic = evaluate_fold_ic(
            _PerfectModel(), panel,
            feature_cols=["_y_"], label_col="label",
            test_idx=test_idx,
        )
        assert np.allclose(ic.values, 1.0)

    def test_fold_ic_near_zero_on_random_signal(self):
        from training_panel.purged_cv import evaluate_fold_ic
        panel = _make_panel(n_dates=50, n_tickers=20, seed=11)
        panel["_y_"] = panel["label"].values
        test_idx = np.arange(len(panel), dtype=np.int64)
        ic = evaluate_fold_ic(
            _RandomModel(seed=123), panel,
            feature_cols=["_y_"], label_col="label",
            test_idx=test_idx,
        )
        # With 20 tickers per date and random preds, IC concentrates near 0
        assert abs(ic.mean()) < 0.15


class TestCrossValidatedIc:
    def test_cv_reports_mean_and_std(self):
        from training_panel.purged_cv import PurgedKFold, cross_validated_ic
        panel = _make_panel(n_dates=50, n_tickers=8, seed=20)
        panel["_y_"] = panel["label"].values

        out = cross_validated_ic(
            model_factory=_PerfectModel,
            panel=panel, feature_cols=["_y_"], label_col="label",
            cv=PurgedKFold(n_splits=4, embargo_days=2, lookahead_days=2),
            weight_col="weight",
        )
        assert "mean_ic" in out
        assert "std_ic" in out
        assert "per_fold_ic" in out
        assert "per_fold_ic_series" in out
        # Perfect model should have IC ≈ 1 per fold
        assert out["mean_ic"] > 0.99

    def test_per_fold_series_align_to_test_dates(self):
        from training_panel.purged_cv import PurgedKFold, cross_validated_ic
        panel = _make_panel(n_dates=30, n_tickers=5, seed=21)
        panel["_y_"] = panel["label"].values
        out = cross_validated_ic(
            _PerfectModel, panel, ["_y_"], "label",
            PurgedKFold(n_splits=3, embargo_days=1, lookahead_days=1),
        )
        # Sum of series lengths should equal number of unique dates
        total_test_dates = sum(len(s) for s in out["per_fold_ic_series"])
        assert total_test_dates == panel["date"].nunique()
