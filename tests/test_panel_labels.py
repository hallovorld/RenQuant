"""Tests for training_panel/labels.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _dates(n: int = 300) -> pd.DatetimeIndex:
    return pd.bdate_range("2022-01-03", periods=n)


def _make_fwd_returns(n: int = 300, seed: int = 0, beta: float = 1.0,
                     sector_beta: float = 0.5):
    """Return (ticker_fwd, spy_fwd, sector_fwd) with known beta structure."""
    rng = np.random.default_rng(seed)
    idx = _dates(n)
    spy = pd.Series(rng.normal(0, 0.01, n), index=idx)
    sec = pd.Series(rng.normal(0, 0.012, n), index=idx)
    noise = rng.normal(0, 0.015, n)
    tkr = beta * spy + sector_beta * sec + pd.Series(noise, index=idx)
    return tkr, spy, sec


class TestComputeResidualReturns:
    def test_beta_regression_uses_only_prior_data(self):
        """If we flip the sign of fwd_return at the last bar, β should not change."""
        from training_panel.labels import compute_residual_returns
        tkr, spy, sec = _make_fwd_returns(n=300, seed=7)

        fwd1 = {"AAA": tkr}
        sec_map = {"AAA": sec}
        res1 = compute_residual_returns(fwd1, spy, sec_map, beta_window=60, lookahead_days=5)

        tkr2 = tkr.copy()
        tkr2.iloc[-1] = -tkr2.iloc[-1] * 10  # wildly perturb last bar
        fwd2 = {"AAA": tkr2}
        res2 = compute_residual_returns(fwd2, spy, sec_map, beta_window=60, lookahead_days=5)

        # Every residual before the last 5 bars (purge window) must equal.
        # Because beta uses data shifted by `purge`, changing the last bar
        # can only affect residuals at indices where the window sees it,
        # i.e. the last `purge` residuals. So residuals up to index -6 match.
        a = res1["AAA"].iloc[:-5]
        b = res2["AAA"].iloc[:-5]
        mask = a.notna() & b.notna()
        assert np.allclose(a[mask].values, b[mask].values)

    def test_residuals_uncorrelated_with_spy_returns(self):
        from training_panel.labels import compute_residual_returns
        tkr, spy, sec = _make_fwd_returns(n=500, seed=1, beta=1.2, sector_beta=0.6)
        res = compute_residual_returns(
            {"AAA": tkr}, spy, {"AAA": sec},
            beta_window=60, lookahead_days=5,
        )
        r = res["AAA"]
        df = pd.DataFrame({"r": r, "spy": spy}).dropna()
        # |corr| should be small — residuals should not be loaded on SPY
        corr = df["r"].corr(df["spy"])
        assert abs(corr) < 0.15, f"residual still loaded on SPY: corr={corr:.3f}"

    def test_residuals_uncorrelated_with_sector_returns(self):
        from training_panel.labels import compute_residual_returns
        tkr, spy, sec = _make_fwd_returns(n=500, seed=2, beta=1.0, sector_beta=0.8)
        res = compute_residual_returns(
            {"AAA": tkr}, spy, {"AAA": sec},
            beta_window=60, lookahead_days=5,
        )
        r = res["AAA"]
        df = pd.DataFrame({"r": r, "sec": sec}).dropna()
        corr = df["r"].corr(df["sec"])
        assert abs(corr) < 0.2, f"residual still loaded on sector: corr={corr:.3f}"

    def test_missing_sector_falls_back_to_spy_only(self):
        from training_panel.labels import compute_residual_returns
        tkr, spy, _ = _make_fwd_returns(n=200, seed=3)
        # sector_returns_by_ticker empty → should not raise
        res = compute_residual_returns(
            {"AAA": tkr}, spy, sector_returns_by_ticker={},
            beta_window=60, lookahead_days=5,
        )
        # All we assert: returns a Series with same index
        assert "AAA" in res
        assert res["AAA"].index.equals(tkr.index)


class TestGaussianizeCrossSection:
    def test_gaussianize_preserves_ranking(self):
        from training_panel.labels import gaussianize_cross_section
        idx = _dates(5)
        # 4 tickers, on each date known ranking AAA<BBB<CCC<DDD
        vals = {
            "AAA": pd.Series(np.arange(5, dtype=float) * 1.0 + 0.0, index=idx),
            "BBB": pd.Series(np.arange(5, dtype=float) * 1.0 + 1.0, index=idx),
            "CCC": pd.Series(np.arange(5, dtype=float) * 1.0 + 2.0, index=idx),
            "DDD": pd.Series(np.arange(5, dtype=float) * 1.0 + 3.0, index=idx),
        }
        out = gaussianize_cross_section(vals)
        for d in idx:
            row = {t: out[t].loc[d] for t in ("AAA", "BBB", "CCC", "DDD")}
            assert row["AAA"] < row["BBB"] < row["CCC"] < row["DDD"]

    def test_gaussianize_output_approx_unit_normal_per_date(self):
        from training_panel.labels import gaussianize_cross_section
        rng = np.random.default_rng(0)
        # 50 tickers × 10 dates ⇒ strong cross-section
        idx = _dates(10)
        tickers = [f"T{i:02d}" for i in range(50)]
        residuals = {
            t: pd.Series(rng.normal(0, 5, len(idx)), index=idx)
            for t in tickers
        }
        out = gaussianize_cross_section(residuals)

        # Collect per-date distributions
        per_date = {d: [] for d in idx}
        for t, s in out.items():
            for d, v in s.items():
                per_date[d].append(v)
        for d, xs in per_date.items():
            arr = np.asarray(xs)
            # N=50 samples from ~N(0,1): expect mean≈0, std≈1
            assert abs(arr.mean()) < 0.3
            assert 0.7 < arr.std() < 1.3

    def test_constant_input_maps_to_zero_on_ties(self):
        """All tickers equal on a date → average rank → mid-uniform → 0."""
        from training_panel.labels import gaussianize_cross_section
        idx = _dates(3)
        residuals = {
            "AAA": pd.Series([1.0, 1.0, 1.0], index=idx),
            "BBB": pd.Series([1.0, 1.0, 1.0], index=idx),
            "CCC": pd.Series([1.0, 1.0, 1.0], index=idx),
        }
        out = gaussianize_cross_section(residuals)
        for t in residuals:
            for d in idx:
                assert abs(out[t].loc[d]) < 1e-9

    def test_single_ticker_day_edge_case(self):
        """One ticker on a date — map to NaN (audit fix #9, 2026-04-29).

        Cross-sectional rank with N=1 is undefined. Pre-fix the function
        returned 0.0, silently injecting a "neutral" label into training
        and biasing weights for any date the universe shrinks to one
        ticker — common at watchlist-start dates where younger names
        enter progressively. NaN is the correct verdict: drop the row.
        """
        from training_panel.labels import gaussianize_cross_section
        idx = _dates(3)
        residuals = {"AAA": pd.Series([0.1, 0.2, 0.3], index=idx)}
        out = gaussianize_cross_section(residuals)
        for d in idx:
            assert np.isnan(out["AAA"].loc[d]), (
                "single-ticker day must yield NaN (not 0.0) so the row is "
                "dropped rather than silently labeled neutral"
            )

    def test_nans_pass_through(self):
        from training_panel.labels import gaussianize_cross_section
        idx = _dates(3)
        residuals = {
            "AAA": pd.Series([np.nan, 0.5, 0.1], index=idx),
            "BBB": pd.Series([0.2, np.nan, 0.2], index=idx),
        }
        out = gaussianize_cross_section(residuals)
        assert np.isnan(out["AAA"].loc[idx[0]])
        assert np.isnan(out["BBB"].loc[idx[1]])


class TestBuildLabels:
    def test_end_to_end_shape(self):
        from training_panel.labels import build_labels
        # Two tickers with realistic sector/SPY structure
        tkr_a, spy, sec = _make_fwd_returns(n=300, seed=4, beta=1.1)
        tkr_b, _, _     = _make_fwd_returns(n=300, seed=5, beta=0.9)
        labels = build_labels(
            {"AAA": tkr_a, "BBB": tkr_b}, spy,
            {"AAA": sec, "BBB": sec},
            beta_window=60, lookahead_days=5,
        )
        assert set(labels.keys()) == {"AAA", "BBB"}
        assert len(labels["AAA"]) == 300
        # Majority of rows should be finite after warmup. With LBL-1 fix
        # (FWL: sec orthogonalized against SPY first), warmup roughly
        # doubles — two sequential rolling betas of window=60+purge=5
        # stack to ~125 NaN bars. Pre-LBL-1 was ~65 NaN bars. Lowered
        # threshold from 200→150 to match the corrected (joint OLS) path.
        finite = labels["AAA"].dropna()
        assert len(finite) > 150
