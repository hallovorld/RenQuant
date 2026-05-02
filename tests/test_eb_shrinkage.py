"""Tests for eb_shrink_percentile / eb_shrink_per_ticker — Layer 5 of
the sector-aware ranking architecture.

Invariant under test
--------------------
For any inputs (p_sector ∈ [0,1], p_global ∈ [0,1], n ≥ 1, k > 0):

  p_shrunk = (n/(n+k)) * p_sector + (k/(n+k)) * p_global

  ∈ [min(p_sector, p_global), max(p_sector, p_global)]   (convexity)
  → p_sector as n → ∞                                    (large-n limit)
  → p_global as n → 0                                    (small-n limit)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.eb_shrinkage import (   # noqa: E402
    compute_sector_size_per_date,
    eb_shrink_percentile,
    eb_shrink_per_ticker,
)


# ── Single-value scalar contract ──────────────────────────────────────────────

class TestScalar:
    def test_large_n_approaches_sector_pct(self):
        """n >> k → result ≈ p_sector."""
        assert eb_shrink_percentile(0.80, 0.50, n_sector=10_000, k=10) == \
            pytest.approx(0.80, abs=1e-3)

    def test_small_n_approaches_global_pct(self):
        """n << k → result ≈ p_global."""
        assert eb_shrink_percentile(0.80, 0.50, n_sector=1, k=100) == \
            pytest.approx(0.50, abs=0.01)

    def test_n_equals_k_exact_midpoint(self):
        """n = k → equal weighting → exact midpoint."""
        assert eb_shrink_percentile(0.80, 0.50, n_sector=10, k=10) == \
            pytest.approx(0.65)

    def test_zero_n_returns_global(self):
        """Empty sector → exact global."""
        assert eb_shrink_percentile(0.80, 0.50, n_sector=0, k=10) == \
            pytest.approx(0.50)

    def test_convexity_invariant(self):
        """Result always between p_sector and p_global, never outside."""
        for ps, pg, n in [(0.95, 0.40, 5), (0.05, 0.60, 50),
                           (0.50, 0.50, 1), (0.30, 0.70, 100)]:
            lo, hi = sorted([ps, pg])
            r = eb_shrink_percentile(ps, pg, n, k=10)
            assert lo - 1e-9 <= r <= hi + 1e-9, (
                f"convex hull violated: {ps=} {pg=} {n=} → {r=} "
                f"not in [{lo}, {hi}]"
            )

    def test_nan_propagates(self):
        assert np.isnan(eb_shrink_percentile(np.nan, 0.50, n_sector=10))
        assert np.isnan(eb_shrink_percentile(0.50, np.nan, n_sector=10))
        assert np.isnan(eb_shrink_percentile(0.50, 0.50, n_sector=float("nan")))

    def test_invalid_k_returns_nan(self):
        assert np.isnan(eb_shrink_percentile(0.50, 0.50, n_sector=10, k=0))
        assert np.isnan(eb_shrink_percentile(0.50, 0.50, n_sector=10, k=-1))

    def test_negative_n_returns_nan(self):
        assert np.isnan(eb_shrink_percentile(0.50, 0.50, n_sector=-5, k=10))


# ── Vectorized per-ticker contract ────────────────────────────────────────────

def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq="B")


class TestPerTicker:
    def test_shape_preserved(self):
        idx = _idx(5)
        sec = {"A": pd.Series([0.9]*5, index=idx),
               "B": pd.Series([0.3]*5, index=idx)}
        glob = {"A": pd.Series([0.5]*5, index=idx),
                "B": pd.Series([0.5]*5, index=idx)}
        sizes = {"A": 50, "B": 50}
        out = eb_shrink_per_ticker(sec, glob, sizes, k=10)
        assert set(out.keys()) == {"A", "B"}
        assert len(out["A"]) == 5
        assert len(out["B"]) == 5

    def test_per_date_size_dict(self):
        """Per-date size dict — sector grew over time."""
        idx = _idx(3)
        sec = {"A": pd.Series([0.9, 0.9, 0.9], index=idx)}
        glob = {"A": pd.Series([0.5, 0.5, 0.5], index=idx)}
        # Day 0: small sector (n=1); day 1: medium (n=10); day 2: large (n=100)
        sizes = {"A": {idx[0]: 1, idx[1]: 10, idx[2]: 100}}
        out = eb_shrink_per_ticker(sec, glob, sizes, k=10)
        # Day 0: weight 1/(1+10) = 0.091 → 0.091*0.9 + 0.909*0.5 ≈ 0.536
        # Day 1: weight 10/20 = 0.5 → 0.5*0.9 + 0.5*0.5 = 0.7
        # Day 2: weight 100/110 ≈ 0.909 → 0.909*0.9 + 0.091*0.5 ≈ 0.864
        vals = out["A"].values
        assert vals[0] == pytest.approx(0.536, abs=0.01)
        assert vals[1] == pytest.approx(0.7,   abs=0.01)
        assert vals[2] == pytest.approx(0.864, abs=0.01)

    def test_missing_global_yields_nan(self):
        idx = _idx(3)
        sec = {"A": pd.Series([0.9]*3, index=idx),
               "MISSING_GLOBAL": pd.Series([0.7]*3, index=idx)}
        glob = {"A": pd.Series([0.5]*3, index=idx)}   # MISSING_GLOBAL absent
        sizes = {"A": 50, "MISSING_GLOBAL": 50}
        out = eb_shrink_per_ticker(sec, glob, sizes, k=10)
        # The ticker without global pct gets all-NaN
        assert out["MISSING_GLOBAL"].isna().all()
        # The other one gets the normal shrinkage result
        assert not out["A"].isna().any()

    def test_index_alignment_on_disjoint_dates(self):
        """sector_pct has dates [d0, d1]; global_pct has [d1, d2].
        Result should be on union [d0, d1, d2] with NaN where one source
        is missing.
        """
        d = _idx(3)
        sec = {"A": pd.Series([0.9, 0.8], index=d[:2])}
        glob = {"A": pd.Series([0.5, 0.4], index=d[1:])}
        sizes = {"A": 50}
        out = eb_shrink_per_ticker(sec, glob, sizes, k=10)
        # Union index: 3 dates
        assert len(out["A"]) == 3
        # d0: only sector → NaN (we don't fall back; missing-data signal preserved)
        assert pd.isna(out["A"].loc[d[0]])
        # d1: both present → normal shrinkage
        assert not pd.isna(out["A"].loc[d[1]])
        # d2: only global → NaN
        assert pd.isna(out["A"].loc[d[2]])


# ── compute_sector_size_per_date helper ───────────────────────────────────────

class TestSectorSizeHelper:
    def test_counts_per_date_per_sector(self):
        idx = _idx(2)
        sec_pct = {
            f"T{i}": pd.Series([float(i)/10, float(i)/20], index=idx)
            for i in range(1, 6)
        }
        # T1, T2, T3 → tech (3); T4, T5 → fin (2)
        ticker_sectors = {"T1": "tech", "T2": "tech", "T3": "tech",
                          "T4": "fin",  "T5": "fin"}
        out = compute_sector_size_per_date(sec_pct, ticker_sectors)
        # T1's sector count on each date = 3; T4's = 2
        assert out["T1"][idx[0]] == 3
        assert out["T1"][idx[1]] == 3
        assert out["T4"][idx[0]] == 2
        assert out["T4"][idx[1]] == 2

    def test_excludes_nan_from_counts(self):
        """A NaN percentile means the ticker had no valid value on that
        date — it shouldn't INFLATE the sector size for OTHER tickers'
        shrinkage on that date.

        Correct semantics: out[ticker][date] = count of tickers IN THE
        SAME SECTOR with valid data on that date. So T2 on day 1 sees
        n=2 (T2 + T3, NOT T1 which is NaN). Whether T1 itself has a
        valid entry for day 1 in `out` is implementation detail —
        shrinkage will yield NaN for T1 on day 1 regardless because
        T1's p_sector is NaN.
        """
        idx = _idx(2)
        sec_pct = {
            "T1": pd.Series([0.1, np.nan], index=idx),  # day 1 missing
            "T2": pd.Series([0.2, 0.4],     index=idx),
            "T3": pd.Series([0.3, 0.6],     index=idx),
        }
        ticker_sectors = {"T1": "tech", "T2": "tech", "T3": "tech"}
        out = compute_sector_size_per_date(sec_pct, ticker_sectors)
        # Day 0: T2, T3 see n=3 (all tech tickers valid)
        # Day 1: T2, T3 see n=2 (T1 dropped — NaN)
        assert out["T2"][idx[0]] == 3
        assert out["T2"][idx[1]] == 2
        assert out["T3"][idx[0]] == 3
        assert out["T3"][idx[1]] == 2
        # T1 on day 0 → 3 (it's a valid contributor)
        assert out["T1"][idx[0]] == 3
        # T1 on day 1: helper still records the sector's size (which is 2,
        # NaN-excluded). The shrinkage step gets a NaN p_sector and
        # produces NaN regardless of n, so this entry is harmless.
        assert out["T1"][idx[1]] == 2


# ── Sanity check: A/A test (CLAUDE.md §5.2 requirement) ──────────────────────

class TestSanityChecks:
    def test_aa_test_identical_inputs_identical_output(self):
        """Same input twice → bit-for-bit identical output.
        Determinism is the floor for any new metric."""
        idx = _idx(5)
        sec = {f"T{i}": pd.Series([0.5]*5, index=idx) for i in range(3)}
        glob = {f"T{i}": pd.Series([0.5]*5, index=idx) for i in range(3)}
        sizes = {f"T{i}": 50 for i in range(3)}
        out1 = eb_shrink_per_ticker(sec, glob, sizes, k=10)
        out2 = eb_shrink_per_ticker(sec, glob, sizes, k=10)
        for t in out1:
            pd.testing.assert_series_equal(out1[t], out2[t])

    def test_shuffle_label_has_no_effect_on_arithmetic(self):
        """Shrinkage is pure arithmetic — relabeling tickers only relabels
        outputs, never alters the per-ticker arithmetic. This catches any
        accidental dependency on ticker order or hash."""
        idx = _idx(3)
        sec = {"A": pd.Series([0.9]*3, index=idx),
               "B": pd.Series([0.4]*3, index=idx)}
        glob = {"A": pd.Series([0.5]*3, index=idx),
                "B": pd.Series([0.5]*3, index=idx)}
        sizes = {"A": 50, "B": 50}
        out_AB = eb_shrink_per_ticker(sec, glob, sizes, k=10)
        # Shuffle: swap A and B's data
        sec_swap = {"A": sec["B"], "B": sec["A"]}
        glob_swap = {"A": glob["B"], "B": glob["A"]}
        out_BA = eb_shrink_per_ticker(sec_swap, glob_swap, sizes, k=10)
        # A's output in the swapped run should equal B's output in the
        # original run (because A now has B's data).
        np.testing.assert_array_almost_equal(
            out_AB["B"].values, out_BA["A"].values,
        )


# ── Boundary: k = effective prior sample size ────────────────────────────────

class TestKHyperparameter:
    def test_increasing_k_increases_shrinkage_to_global(self):
        """Bigger k = more shrinkage = closer to global, monotonically."""
        ps, pg, n = 0.90, 0.50, 20
        ks = [1, 5, 10, 20, 50, 100]
        results = [eb_shrink_percentile(ps, pg, n, k=k) for k in ks]
        # Should monotonically decrease toward 0.50 as k grows
        for i in range(len(ks) - 1):
            assert results[i] >= results[i + 1] - 1e-9, (
                f"k={ks[i]} → {results[i]} should be ≥ k={ks[i+1]} → {results[i+1]}"
            )
        # And never cross below the target
        assert all(r >= pg - 1e-9 for r in results)
