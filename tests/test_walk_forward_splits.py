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
    def test_three_buckets(self):
        cut = WalkForwardCut(
            name="t", train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2022-01-01"),
            val_start=pd.Timestamp("2022-01-01"),
            val_end=pd.Timestamp("2022-04-01"))
        panel = pd.DataFrame({
            "date": pd.date_range("2020-06-01", "2023-01-01", freq="MS")
        })
        split = assign_split_column(panel, cut)
        # 2020-06 → 2021-12 = train (19 rows)
        # 2022-01, 02, 03 = val (3 rows)
        # 2022-04 → 2023-01 = test
        assert (split == "train").sum() == 19
        assert (split == "val").sum() == 3
        assert (split == "test").sum() == 10

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
        split = assign_split_column(panel, cut)
        assert split.iloc[0] == "val"
        assert split.iloc[1] == "test"


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
