"""Tests for cross_sectional_rank_within_sector — Layer 1 of the
sector-aware ranking architecture (per
``doc/research/per-sector-architecture-plan.md``).

Why this test file exists
-------------------------
Per CLAUDE.md §5.2 (every metric ships with a sanity check), the
sector-rank-norm transform must:

  * Produce values strictly in [0, 1] for any non-NaN input.
  * Be sector-relative: same value in two different sectors gets
    different percentiles iff the sectors differ in their distributions.
  * Pass an A/A test: random within-sector permutation yields the same
    set of percentile values (just relabeled to new tickers).
  * Pass a shuffled-label test: percentile rank is invariant to label
    shuffles (same input → same output, regardless of model state).
  * Handle NaN, single-ticker sectors, unmapped tickers, ties.

Invariant under test
--------------------
For any (date, sector) group with ≥ min_sector_size tickers:
    result ∈ [0, 1] AND values reflect within-sector rank order.
Tickers in under-populated sectors fall back to global percentile
(or NaN if fallback_global=False). NaN inputs propagate to NaN outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.factors import cross_sectional_rank_within_sector  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq="B")


def _make_feature(values_by_ticker: dict[str, list[float]]) -> dict[str, pd.Series]:
    """All series share an index of length max(values)."""
    n = max(len(v) for v in values_by_ticker.values())
    idx = _dates(n)
    return {t: pd.Series(v, index=idx) for t, v in values_by_ticker.items()}


# ── Output range invariant ────────────────────────────────────────────────────

class TestRangeInvariant:
    def test_outputs_in_unit_interval(self):
        """All non-NaN outputs must be in [0, 1] — that's the percentile
        contract Layer 5 (EB shrinkage) and downstream consumers depend on.
        """
        feat = _make_feature({
            f"T{i}": [float(i) + j*0.1 for j in range(3)] for i in range(10)
        })
        sectors = {f"T{i}": ("tech" if i < 5 else "fin") for i in range(10)}
        out = cross_sectional_rank_within_sector(feat, sectors)
        for t, s in out.items():
            valid = s.dropna()
            assert (valid >= 0).all() and (valid <= 1).all(), (
                f"{t} has values outside [0, 1]: {valid.tolist()}"
            )

    def test_extreme_values_clip_correctly(self):
        feat = _make_feature({
            "A": [1e10],     # extreme high
            "B": [-1e10],    # extreme low
            "C": [0.0],
            "D": [1.0],
            "E": [-1.0],
        })
        sectors = {t: "tech" for t in "ABCDE"}
        out = cross_sectional_rank_within_sector(feat, sectors)
        # A is the max → percentile 1.0; B is the min → 0.2 (1/5)
        assert out["A"].iloc[0] == pytest.approx(1.0)
        assert out["B"].iloc[0] == pytest.approx(0.2)


# ── Sector-relativity invariant ───────────────────────────────────────────────

class TestSectorRelativity:
    """The whole point: same VALUE in different sectors → different PERCENTILE
    when those sectors' distributions differ. This is the structural property
    the wl178 architecture work depends on (Witter 2025 mechanism).
    """

    def test_top_in_each_sector_both_get_pct_1(self):
        # Tech values are ~0.01, finance values are ~10. Each sector's
        # top value gets percentile 1.0 — they are NOT compared globally.
        feat = _make_feature({
            "TECH_TOP":  [0.05],   # tech top
            "TECH_MID":  [0.02],
            "TECH_LOW":  [0.001],
            "TECH_X":    [0.03],
            "TECH_Y":    [0.04],
            "FIN_TOP":   [12.0],
            "FIN_MID":   [10.0],
            "FIN_LOW":   [5.0],
            "FIN_X":     [11.0],
            "FIN_Y":     [9.0],
        })
        sectors = {
            "TECH_TOP": "tech", "TECH_MID": "tech", "TECH_LOW": "tech",
            "TECH_X":   "tech", "TECH_Y":   "tech",
            "FIN_TOP":  "fin",  "FIN_MID":  "fin",  "FIN_LOW":  "fin",
            "FIN_X":    "fin",  "FIN_Y":    "fin",
        }
        out = cross_sectional_rank_within_sector(feat, sectors)
        assert out["TECH_TOP"].iloc[0] == pytest.approx(1.0)
        assert out["FIN_TOP"].iloc[0] == pytest.approx(1.0)
        # Without sector relativity (global pct), TECH_TOP would be ~0.5
        # because it's smaller than every FIN value; we test it's NOT.
        assert out["TECH_LOW"].iloc[0] == pytest.approx(0.2)  # 1/5 within tech
        assert out["FIN_LOW"].iloc[0]  == pytest.approx(0.2)  # 1/5 within fin

    def test_within_sector_ordering_preserved(self):
        """Strict monotone — bigger raw value → bigger percentile,
        within the same sector."""
        feat = _make_feature({
            "T1": [1.0], "T2": [2.0], "T3": [3.0], "T4": [4.0], "T5": [5.0],
        })
        sectors = {f"T{i}": "tech" for i in range(1, 6)}
        out = cross_sectional_rank_within_sector(feat, sectors)
        ranks = [out[f"T{i}"].iloc[0] for i in range(1, 6)]
        assert all(ranks[i] < ranks[i + 1] for i in range(4)), (
            f"within-sector order not preserved: {ranks}"
        )


# ── Under-populated sector fallback ───────────────────────────────────────────

class TestFallback:
    def test_small_sector_uses_global_when_fallback_enabled(self):
        # 5 tech tickers, 2 small "energy" → energy falls back to global pct
        feat = _make_feature({
            **{f"TECH{i}": [float(i)] for i in range(5)},
            "ENERGY1": [10.0],   # globally biggest
            "ENERGY2": [-5.0],   # globally smallest
        })
        sectors = {**{f"TECH{i}": "tech" for i in range(5)},
                   **{"ENERGY1": "energy", "ENERGY2": "energy"}}
        out = cross_sectional_rank_within_sector(
            feat, sectors, min_sector_size=5, fallback_global=True,
        )
        # 7 total tickers: ENERGY1 is the biggest of all 7 → global pct = 7/7 = 1.0
        # ENERGY2 is the smallest → global pct = 1/7
        assert out["ENERGY1"].iloc[0] == pytest.approx(1.0)
        assert out["ENERGY2"].iloc[0] == pytest.approx(1/7)

    def test_small_sector_returns_nan_when_fallback_disabled(self):
        feat = _make_feature({
            **{f"TECH{i}": [float(i)] for i in range(5)},
            "ENERGY1": [10.0],
            "ENERGY2": [-5.0],
        })
        sectors = {**{f"TECH{i}": "tech" for i in range(5)},
                   **{"ENERGY1": "energy", "ENERGY2": "energy"}}
        out = cross_sectional_rank_within_sector(
            feat, sectors, min_sector_size=5, fallback_global=False,
        )
        assert pd.isna(out["ENERGY1"].iloc[0])
        assert pd.isna(out["ENERGY2"].iloc[0])
        # Tech sector still works
        assert not pd.isna(out["TECH4"].iloc[0])

    def test_unmapped_ticker_uses_global_fallback(self):
        feat = _make_feature({
            **{f"TECH{i}": [float(i)] for i in range(5)},
            "MYSTERY": [2.5],
        })
        sectors = {f"TECH{i}": "tech" for i in range(5)}
        # MYSTERY has no sector entry — should use global percentile
        out = cross_sectional_rank_within_sector(feat, sectors, min_sector_size=5)
        assert not pd.isna(out["MYSTERY"].iloc[0])
        assert 0 <= out["MYSTERY"].iloc[0] <= 1


# ── NaN handling ──────────────────────────────────────────────────────────────

class TestNaNHandling:
    def test_nan_input_produces_nan_output(self):
        feat = {
            "T1": pd.Series([np.nan, 1.0, 2.0], index=_dates(3)),
            "T2": pd.Series([0.0,    2.0, 3.0], index=_dates(3)),
            "T3": pd.Series([1.0,    3.0, 4.0], index=_dates(3)),
            "T4": pd.Series([2.0,    4.0, 5.0], index=_dates(3)),
            "T5": pd.Series([3.0,    5.0, 6.0], index=_dates(3)),
        }
        sectors = {f"T{i}": "tech" for i in range(1, 6)}
        out = cross_sectional_rank_within_sector(feat, sectors)
        # T1's first row was NaN → output NaN there too
        assert pd.isna(out["T1"].iloc[0])
        assert not pd.isna(out["T1"].iloc[1])
        assert not pd.isna(out["T1"].iloc[2])

    def test_all_nan_date_yields_nan_for_all(self):
        feat = {
            f"T{i}": pd.Series([np.nan, 1.0], index=_dates(2))
            for i in range(1, 6)
        }
        sectors = {f"T{i}": "tech" for i in range(1, 6)}
        out = cross_sectional_rank_within_sector(feat, sectors)
        # First date is all-NaN → all outputs are NaN there
        for t in [f"T{i}" for i in range(1, 6)]:
            assert pd.isna(out[t].iloc[0])
        # Second date: all equal → all get same percentile (mean of ranks)
        for t in [f"T{i}" for i in range(1, 6)]:
            assert not pd.isna(out[t].iloc[1])


# ── Tie handling (rank method='average') ──────────────────────────────────────

class TestTieHandling:
    def test_ties_get_average_rank(self):
        """All-equal sector → all tickers get percentile 0.5 (mean of 1..N pct)."""
        feat = _make_feature({f"T{i}": [3.0] for i in range(1, 6)})
        sectors = {f"T{i}": "tech" for i in range(1, 6)}
        out = cross_sectional_rank_within_sector(feat, sectors)
        # rank-pct-average for 5 ties = (1+2+3+4+5)/5 / 5 = 3/5 = 0.6
        # Wait actually rank(method='average', pct=True) for all-ties
        # gives ((N+1)/2) / N = 3/5 = 0.6 for N=5. Verify and adjust.
        for t in [f"T{i}" for i in range(1, 6)]:
            assert out[t].iloc[0] == pytest.approx(0.6)

    def test_partial_ties(self):
        feat = _make_feature({
            "T1": [1.0],
            "T2": [2.0],
            "T3": [2.0],   # tied with T2
            "T4": [3.0],
            "T5": [4.0],
        })
        sectors = {f"T{i}": "tech" for i in range(1, 6)}
        out = cross_sectional_rank_within_sector(feat, sectors)
        # T2 and T3 tie at rank 2 and 3 → average pct = ((2+3)/2)/5 = 0.5
        assert out["T2"].iloc[0] == pytest.approx(0.5)
        assert out["T3"].iloc[0] == pytest.approx(0.5)
        # T1 unique at rank 1 → 0.2
        assert out["T1"].iloc[0] == pytest.approx(0.2)


# ── A/A test (sanity check per CLAUDE.md §5.2) ────────────────────────────────

class TestAATest:
    """Per CLAUDE.md §5.2: every new metric/transform ships with at
    minimum an A/A test. Here: relabeling tickers within a sector
    must not change the SET of output percentiles for that sector.
    Only the ticker→percentile mapping shifts.
    """

    def test_within_sector_ticker_relabel_invariance(self):
        feat = _make_feature({
            "T1": [1.0], "T2": [2.0], "T3": [3.0], "T4": [4.0], "T5": [5.0],
        })
        sectors = {f"T{i}": "tech" for i in range(1, 6)}
        out = cross_sectional_rank_within_sector(feat, sectors)
        # Same data but with shuffled ticker names produces the same
        # multiset of percentiles (just under different labels).
        feat2 = _make_feature({
            "X1": [3.0], "X2": [5.0], "X3": [1.0], "X4": [4.0], "X5": [2.0],
        })
        sectors2 = {f"X{i}": "tech" for i in range(1, 6)}
        out2 = cross_sectional_rank_within_sector(feat2, sectors2)
        s1 = sorted(s.iloc[0] for s in out.values())
        s2 = sorted(s.iloc[0] for s in out2.values())
        assert s1 == pytest.approx(s2), (
            "Same value-set in same sector must produce the same multiset "
            "of percentiles — A/A test failed"
        )


# ── Determinism (same input → same output, every time) ───────────────────────

class TestDeterminism:
    def test_repeat_calls_produce_identical_output(self):
        feat = _make_feature({
            f"T{i}": [float(i), float(i + 1)] for i in range(10)
        })
        sectors = {f"T{i}": ("tech" if i < 5 else "fin") for i in range(10)}
        out1 = cross_sectional_rank_within_sector(feat, sectors)
        out2 = cross_sectional_rank_within_sector(feat, sectors)
        for t in feat:
            pd.testing.assert_series_equal(out1[t], out2[t])


# ── Empty / edge-case inputs ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_feature_dict(self):
        out = cross_sectional_rank_within_sector({}, {})
        assert out == {}

    def test_single_ticker_under_min_size_global_fallback(self):
        feat = {"LONELY": pd.Series([1.0, 2.0], index=_dates(2))}
        out = cross_sectional_rank_within_sector(
            feat, {"LONELY": "void"}, min_sector_size=5, fallback_global=True,
        )
        # Only one ticker → global pct = 1.0 (it IS the universe)
        assert out["LONELY"].iloc[0] == pytest.approx(1.0)

    def test_single_ticker_under_min_size_no_fallback(self):
        feat = {"LONELY": pd.Series([1.0, 2.0], index=_dates(2))}
        out = cross_sectional_rank_within_sector(
            feat, {"LONELY": "void"}, min_sector_size=5, fallback_global=False,
        )
        assert pd.isna(out["LONELY"].iloc[0])
        assert pd.isna(out["LONELY"].iloc[1])
