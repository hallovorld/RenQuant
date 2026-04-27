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


class TestAuditHigh1PurgeInBars:
    """Audit HIGH-1 (2026-04-27): pre-fix used pd.Timedelta(days=L) which
    purges CALENDAR days. Labels are constructed by `c.shift(-lookahead)`
    which is BAR shift. With L=10, calendar-day purge of 10d = ~7 trading
    days but the label spans ~14 calendar days → ~3 trading days of
    training rows reach into the test window. Pin the new BAR-based
    semantics."""

    def test_purge_count_matches_lookahead_in_bars(self):
        """For lookahead_days=L on a business-day panel, exactly L bars
        of training rows immediately before the test fold must be purged
        — regardless of what calendar-day delta those L bars span."""
        from training_panel.purged_cv import PurgedKFold
        # 60 business days
        panel = _make_panel(n_dates=60, n_tickers=3, seed=99)
        L = 10  # PROD value
        cv = PurgedKFold(n_splits=5, embargo_days=0, lookahead_days=L)

        unique_dates = sorted(set(pd.to_datetime(panel["date"]).values))
        for train_idx, test_idx in cv.split(panel):
            test_dates = sorted(set(pd.to_datetime(panel.iloc[test_idx]["date"]).values))
            train_dates = set(pd.to_datetime(panel.iloc[train_idx]["date"]).values)
            test_start_pos = unique_dates.index(test_dates[0])
            # The L bars immediately before test_start MUST be purged
            for k in range(1, L + 1):
                pos = test_start_pos - k
                if pos < 0:
                    continue
                purged_date = unique_dates[pos]
                assert purged_date not in train_dates, (
                    f"BAR-{k} before test_start ({purged_date}) leaked into train; "
                    f"this is the HIGH-1 leak — purge must count BARS not calendar days"
                )

    def test_combinatorial_purges_l_bars_before_each_block(self):
        from training_panel.purged_cv import CombinatorialPurgedCV
        panel = _make_panel(n_dates=60, n_tickers=3, seed=100)
        L = 10
        cv = CombinatorialPurgedCV(
            n_splits=6, n_test_groups=2,
            embargo_days=0, lookahead_days=L,
        )
        unique_dates = sorted(set(pd.to_datetime(panel["date"]).values))
        seen = 0
        for train_idx, test_idx in cv.split(panel):
            test_dates = sorted(set(pd.to_datetime(panel.iloc[test_idx]["date"]).values))
            train_dates = set(pd.to_datetime(panel.iloc[train_idx]["date"]).values)
            # Find each contiguous block in the test set
            test_dates_arr = np.array(test_dates, dtype="datetime64[ns]")
            block_starts = [test_dates[0]]
            for i in range(1, len(test_dates)):
                pos_curr = unique_dates.index(test_dates[i])
                pos_prev = unique_dates.index(test_dates[i - 1])
                if pos_curr - pos_prev > 1:
                    block_starts.append(test_dates[i])
            for bs in block_starts:
                bs_pos = unique_dates.index(bs)
                for k in range(1, L + 1):
                    pos = bs_pos - k
                    if pos < 0:
                        continue
                    purged_date = unique_dates[pos]
                    assert purged_date not in train_dates, (
                        f"BAR-{k} before block_start ({purged_date}) "
                        f"leaked into train (CPCV HIGH-1 regression)"
                    )
            seen += 1
        assert seen > 0, "CPCV produced no splits"

    def test_no_calendar_day_undercount_for_l10(self):
        """The bug scenario: lookahead=10 with business-day panel.
        Pre-fix: `Timedelta(days=10)` only covered the most recent 6-7
        business days. Post-fix: covers exactly 10 business days. Verify
        the 8th-most-recent business day (which leaked pre-fix) is now
        purged."""
        from training_panel.purged_cv import PurgedKFold
        panel = _make_panel(n_dates=80, n_tickers=2, seed=101)
        cv = PurgedKFold(n_splits=4, embargo_days=0, lookahead_days=10)
        unique_dates = sorted(set(pd.to_datetime(panel["date"]).values))
        for train_idx, test_idx in cv.split(panel):
            test_dates = sorted(set(pd.to_datetime(panel.iloc[test_idx]["date"]).values))
            train_dates = set(pd.to_datetime(panel.iloc[train_idx]["date"]).values)
            test_start_pos = unique_dates.index(test_dates[0])
            # The 8th business day before test_start was the "leaker"
            # in the pre-fix code (calendar-day purge of 10d ≈ 6-7
            # business days, so 8th was just outside the purge window
            # despite its label reaching into the test window).
            leaker_pos = test_start_pos - 8
            if leaker_pos >= 0:
                leaker_date = unique_dates[leaker_pos]
                assert leaker_date not in train_dates, (
                    f"Pre-fix leaker date {leaker_date} (8 BARS before "
                    f"test_start) is in train — HIGH-1 NOT FIXED"
                )


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


# ── Combinatorial Purged CV ──────────────────────────────────────────────────

from math import comb
from training_panel.purged_cv import CombinatorialPurgedCV, cross_validated_ic_cpcv


class TestCombinatorialPurgedCV:
    def _panel(self, n_dates=60, n_tickers=8, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for d in pd.bdate_range("2024-01-01", periods=n_dates):
            for i in range(n_tickers):
                x1 = rng.normal()
                rows.append({"date": d, "ticker": f"T{i}",
                             "x1": x1, "x2": rng.normal(),
                             "label": x1 + 0.2 * rng.normal(),
                             "weight": 1.0})
        return pd.DataFrame(rows)

    def test_yields_n_choose_k_splits(self):
        panel = self._panel(n_dates=60)
        cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2,
                                     embargo_days=2, lookahead_days=5)
        assert len(list(cv.split(panel))) == comb(6, 2)   # 15

    def test_reduces_to_k_fold_when_n_test_groups_1(self):
        panel = self._panel(n_dates=40)
        cv = CombinatorialPurgedCV(n_splits=5, n_test_groups=1,
                                     embargo_days=2, lookahead_days=5)
        assert len(list(cv.split(panel))) == 5

    def test_every_row_in_at_least_one_test_fold(self):
        panel = self._panel(n_dates=30)
        cv = CombinatorialPurgedCV(n_splits=5, n_test_groups=2,
                                     embargo_days=1, lookahead_days=3)
        seen: set[int] = set()
        for _, test_idx in cv.split(panel):
            seen.update(int(i) for i in test_idx)
        assert seen == set(range(len(panel)))

    def test_train_test_disjoint_per_split(self):
        panel = self._panel(n_dates=30)
        cv = CombinatorialPurgedCV(n_splits=5, n_test_groups=2,
                                     embargo_days=1, lookahead_days=3)
        for train_idx, test_idx in cv.split(panel):
            assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0

    def test_rejects_invalid_n_test_groups(self):
        panel = self._panel(n_dates=30)
        with pytest.raises(ValueError, match="n_test_groups"):
            list(CombinatorialPurgedCV(n_splits=5, n_test_groups=0).split(panel))
        with pytest.raises(ValueError, match="n_test_groups"):
            list(CombinatorialPurgedCV(n_splits=5, n_test_groups=5).split(panel))

    def test_cpcv_produces_ic_distribution(self):
        panel = self._panel(n_dates=60, n_tickers=10, seed=1)
        cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2,
                                     embargo_days=1, lookahead_days=3)

        class _Stub:
            def fit(self, X, y, sample_weight=None):
                pass
            def predict(self, X):
                return X["x1"].values

        result = cross_validated_ic_cpcv(
            _Stub, panel, feature_cols=["x1", "x2"], label_col="label", cv=cv,
        )
        assert len(result["per_fold_ic"]) == comb(6, 2)
        assert result["mean_ic"] > 0.2     # x1 is truly predictive
        assert "quantiles" in result
        q = result["quantiles"]
        assert q["q05"] <= q["q50"] <= q["q95"]
