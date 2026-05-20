"""Regression tests for kernel/walk_forward_splits.py.

Pin the regime-coverage requirement so future cut edits can't silently
re-introduce the 2026-05-18 bug (val period containing 0 SPIKED days →
PRIME DIRECTIVE violation in any objective computed over it).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kernel.walk_forward_splits import (build_default_cuts, WalkForwardCut,
                                         assign_split_column,
                                         verify_regime_coverage)


class TestDefaultCuts:
    def test_five_cuts(self):
        cuts = build_default_cuts()
        assert len(cuts) == 5

    def test_cut_names_distinct(self):
        cuts = build_default_cuts()
        names = [c.name for c in cuts]
        assert len(set(names)) == 5

    def test_val_periods_nonoverlapping(self):
        cuts = build_default_cuts()
        # Sort by val_start
        sorted_cuts = sorted(cuts, key=lambda c: c.val_start)
        for a, b in zip(sorted_cuts, sorted_cuts[1:]):
            assert a.val_end <= b.val_start, (
                f"val overlap: {a.name} ends {a.val_end} > {b.name} starts {b.val_start}")

    def test_train_strictly_before_val(self):
        for c in build_default_cuts():
            assert c.train_end == c.val_start  # train ends where val begins
            assert c.train_start < c.train_end


class TestAssignSplit:
    def test_three_buckets_no_embargo(self):
        """With embargo_days=0 (legacy behavior), pure 3-way split."""
        cut = WalkForwardCut(
            name="t", train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2022-01-01"),
            val_start=pd.Timestamp("2022-01-01"),
            val_end=pd.Timestamp("2022-04-01"))
        panel = pd.DataFrame({
            "date": pd.date_range("2020-06-01", "2023-01-01", freq="MS")
        })
        split = assign_split_column(panel, cut, embargo_days=0)
        assert (split == "train").sum() == 19
        assert (split == "val").sum() == 3
        assert (split == "test").sum() == 10
        assert (split == "embargo").sum() == 0

    def test_val_inclusive_start_exclusive_end(self):
        cut = WalkForwardCut(
            name="t", train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2022-01-01"),
            val_start=pd.Timestamp("2022-01-01"),
            val_end=pd.Timestamp("2022-04-01"))
        panel = pd.DataFrame({
            "date": [pd.Timestamp("2022-01-01"),  # val (start inclusive)
                     pd.Timestamp("2022-04-01")]  # test (end exclusive)
        })
        split = assign_split_column(panel, cut, embargo_days=0)
        assert split.iloc[0] == "val"
        assert split.iloc[1] == "test"


class TestEmbargo:
    """2026-05-20 audit P0-1 regression guard: embargo MUST gap train→val
    by at least `lookahead_days` of label to prevent leakage.

    `fwd_60d_excess` (default label) means train rows with date in the
    60 trading days before val_start have labels that PEEK into val.
    Those rows must go to the "embargo" bucket, not "train".
    """

    def test_default_60d_embargo_excludes_train_rows_near_val(self):
        cut = WalkForwardCut(
            name="t", train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2022-01-01"),
            val_start=pd.Timestamp("2022-01-01"),
            val_end=pd.Timestamp("2022-04-01"))
        # Mix of dates: 2021-11-01 should be in 60-bday embargo before 2022-01-01
        # (60 business days back from 2022-01-01 ≈ 2021-10-08)
        panel = pd.DataFrame({
            "date": [
                pd.Timestamp("2020-06-01"),  # train (far from val)
                pd.Timestamp("2021-09-01"),  # train (outside 60 bday embargo)
                pd.Timestamp("2021-11-01"),  # embargo (within 60 bday of val_start)
                pd.Timestamp("2021-12-31"),  # embargo (1 day before val)
                pd.Timestamp("2022-01-01"),  # val (boundary inclusive)
                pd.Timestamp("2022-02-15"),  # val
                pd.Timestamp("2022-04-15"),  # test
            ]
        })
        split = assign_split_column(panel, cut)  # default embargo_days=60
        assert split.iloc[0] == "train", "far-past row should be train"
        assert split.iloc[1] == "train", "outside 60 bday embargo should be train"
        assert split.iloc[2] == "embargo", "within 60 bday of val MUST be embargo"
        assert split.iloc[3] == "embargo", "1 day before val MUST be embargo"
        assert split.iloc[4] == "val", "val boundary inclusive"
        assert split.iloc[5] == "val"
        assert split.iloc[6] == "test"

    def test_no_train_row_has_label_window_overlapping_val(self):
        """Pin the invariant: with embargo_days=60 and label fwd_60d_excess,
        no train row's label window reaches val_start."""
        cut = WalkForwardCut(
            name="t", train_start=pd.Timestamp("2018-01-01"),
            train_end=pd.Timestamp("2022-01-01"),
            val_start=pd.Timestamp("2022-01-01"),
            val_end=pd.Timestamp("2022-04-01"))
        panel = pd.DataFrame({"date": pd.bdate_range("2020-01-01", "2022-04-01")})
        split = assign_split_column(panel, cut, embargo_days=60)
        train_dates = pd.to_datetime(panel.loc[split == "train", "date"])
        # Max train date + 60 bdays MUST still be < val_start
        max_train = train_dates.max()
        last_label_target = max_train + pd.offsets.BDay(60)
        assert last_label_target < cut.val_start, (
            f"Train date {max_train.date()} + 60 bday = {last_label_target.date()} "
            f"reaches into val (starts {cut.val_start.date()}) — LEAKAGE")

    def test_embargo_zero_falls_back_to_legacy_behavior(self):
        cut = WalkForwardCut(
            name="t", train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2022-01-01"),
            val_start=pd.Timestamp("2022-01-01"),
            val_end=pd.Timestamp("2022-04-01"))
        panel = pd.DataFrame({
            "date": pd.bdate_range("2021-10-01", "2022-04-01")
        })
        split = assign_split_column(panel, cut, embargo_days=0)
        # With embargo=0, no embargo bucket
        assert (split == "embargo").sum() == 0
        # Everything pre-val is train
        train_dates = pd.to_datetime(panel.loc[split == "train", "date"])
        assert (train_dates < cut.val_start).all()

    def test_embargo_bucket_is_not_train_and_not_val(self):
        cut = WalkForwardCut(
            name="t", train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2022-01-01"),
            val_start=pd.Timestamp("2022-01-01"),
            val_end=pd.Timestamp("2022-04-01"))
        panel = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", "2022-06-01")
        })
        split = assign_split_column(panel, cut, embargo_days=60)
        # Embargo rows must be excluded from BOTH train and val
        embargo_count = (split == "embargo").sum()
        assert embargo_count > 0, "60-day embargo should produce some rows"
        # No date can be both in train and within 60 bdays of val
        n_total = len(panel)
        assert n_total == (split == "train").sum() + (split == "embargo").sum() \
            + (split == "val").sum() + (split == "test").sum()


@pytest.mark.skipif(not (REPO / "data/ohlcv/SPY/1d.parquet").exists(),
                     reason="SPY parquet not available")
class TestRegimeCoveragePrimeDirective:
    """PRIME DIRECTIVE: every cut's val MUST have either SPIKED days OR
    ≥4 regimes. A cut with 1 regime is useless for min-across-regime."""

    def test_every_cut_has_either_spiked_or_diverse(self):
        spy_path = REPO / "data/ohlcv/SPY/1d.parquet"
        for c in build_default_cuts():
            counts = verify_regime_coverage(c, spy_path, require_spiked=False)
            spiked = sum(v for k, v in counts.items() if "SPIKED" in str(k))
            n_regimes = len(counts)
            assert spiked > 0 or n_regimes >= 4, (
                f"Cut {c.name} has 0 SPIKED days AND only {n_regimes} regimes "
                f"— PRIME DIRECTIVE violation. Counts: {counts}"
            )

    def test_total_spiked_coverage_meets_threshold(self):
        """Aggregate across all cuts: ≥30% of val days should be SPIKED.
        2026-05-18 baseline (single 2023 val): 0% — failed.
        """
        spy_path = REPO / "data/ohlcv/SPY/1d.parquet"
        total_days = 0
        spiked_days = 0
        for c in build_default_cuts():
            counts = verify_regime_coverage(c, spy_path, require_spiked=False)
            total_days += sum(counts.values())
            spiked_days += sum(v for k, v in counts.items() if "SPIKED" in str(k))
        pct = spiked_days / max(total_days, 1)
        assert pct >= 0.30, f"SPIKED coverage {pct:.1%} < 30% threshold"

    def test_at_least_three_cuts_have_spiked(self):
        spy_path = REPO / "data/ohlcv/SPY/1d.parquet"
        spiked_cuts = 0
        for c in build_default_cuts():
            counts = verify_regime_coverage(c, spy_path, require_spiked=False)
            if any("SPIKED" in str(k) for k in counts):
                spiked_cuts += 1
        assert spiked_cuts >= 3, (
            f"Only {spiked_cuts}/5 cuts contain SPIKED days. PRIME DIRECTIVE "
            f"requires majority of cuts cover SPIKED for robust verdicts."
        )

    def test_verify_regime_coverage_raises_when_no_spiked_required(self):
        spy_path = REPO / "data/ohlcv/SPY/1d.parquet"
        # cut4_svb has 0 SPIKED days — should raise with require_spiked=True
        cut4 = next(c for c in build_default_cuts() if c.name == "cut4_svb")
        with pytest.raises(ValueError, match="0 SPIKED"):
            verify_regime_coverage(cut4, spy_path, require_spiked=True)
