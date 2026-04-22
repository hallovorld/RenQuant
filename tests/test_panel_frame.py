"""Tests for training_panel/panel_frame.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_feature_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({
        "rsi":       rng.uniform(20, 80, n),
        "macd_hist": rng.normal(0, 0.5, n),
        "trend":     rng.normal(1, 0.05, n),
    }, index=idx)


def _make_labels(ff: pd.DataFrame, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, len(ff)), index=ff.index)


def _make_inputs(tickers=("AAA", "BBB", "CCC"), n: int = 300):
    ffs = {t: _make_feature_frame(n, seed=i) for i, t in enumerate(tickers)}
    lbs = {t: _make_labels(ff, seed=i + 100) for i, (t, ff) in enumerate(ffs.items())}
    sectors = {t: "SEC" + t[0] for t in tickers}
    return ffs, lbs, sectors


class TestBuildPanelFrame:
    def test_build_shape_and_columns(self):
        from training_panel.panel_frame import build_panel_frame
        ffs, lbs, sectors = _make_inputs(n=300)
        panel, groups, meta = build_panel_frame(
            ffs, lbs, sectors, min_history_days=10, lookahead_days=5,
        )
        for col in ("date", "ticker", "sector", "label",
                    "weight_concurrency", "weight_age", "weight",
                    "rsi", "macd_hist", "trend"):
            assert col in panel.columns, f"missing {col}"
        assert meta["n_rows"] == len(panel)
        assert meta["n_tickers"] == 3

    def test_sorted_by_date_then_ticker(self):
        from training_panel.panel_frame import build_panel_frame
        ffs, lbs, sectors = _make_inputs()
        panel, _, _ = build_panel_frame(ffs, lbs, sectors, min_history_days=5)
        # strictly non-decreasing dates
        assert (panel["date"].values[1:] >= panel["date"].values[:-1]).all()
        # within each date, tickers are alphabetically sorted
        for _, grp in panel.groupby("date", sort=False):
            tkrs = grp["ticker"].tolist()
            assert tkrs == sorted(tkrs)

    def test_group_sizes_sum_equals_row_count(self):
        from training_panel.panel_frame import build_panel_frame
        ffs, lbs, sectors = _make_inputs()
        panel, groups, _ = build_panel_frame(ffs, lbs, sectors, min_history_days=5)
        assert groups.sum() == len(panel)

    def test_group_sizes_align_with_date_counts(self):
        from training_panel.panel_frame import build_panel_frame
        ffs, lbs, sectors = _make_inputs()
        panel, groups, _ = build_panel_frame(ffs, lbs, sectors, min_history_days=5)
        per_date = panel.groupby("date", sort=True).size().values
        assert np.array_equal(groups, per_date)

    def test_min_history_gate_drops_young_tickers(self):
        from training_panel.panel_frame import build_panel_frame
        # ticker ZZZ has only 50 bars → dropped when min_history_days=60
        ffs, lbs, sectors = _make_inputs(tickers=("AAA", "BBB"))
        short = _make_feature_frame(50, seed=9)
        ffs["ZZZ"] = short
        lbs["ZZZ"] = _make_labels(short, seed=99)
        sectors["ZZZ"] = "SECZ"
        panel, _, meta = build_panel_frame(
            ffs, lbs, sectors, min_history_days=60,
        )
        assert "ZZZ" not in panel["ticker"].unique()
        assert "ZZZ" not in meta["per_ticker"]

    def test_missingness_indicators_binary_and_match_nan(self):
        from training_panel.panel_frame import build_panel_frame
        ffs, lbs, sectors = _make_inputs()
        # Inject NaNs into one column of one ticker
        ffs["AAA"].iloc[50:60, ffs["AAA"].columns.get_loc("rsi")] = np.nan
        panel, _, meta = build_panel_frame(
            ffs, lbs, sectors, min_history_days=5,
            nan_prone_cols=["rsi"],
        )
        assert "rsi_is_missing" in panel.columns
        assert set(panel["rsi_is_missing"].unique()).issubset({0, 1})
        # Count of is_missing==1 matches nan count among panel rows
        nan_mask = panel["rsi"].isna()
        assert (panel["rsi_is_missing"] == nan_mask.astype(int)).all()
        assert "rsi_is_missing" in meta["missing_cols"]

    def test_concurrency_weight_inversely_related_to_active_overlap(self):
        from training_panel.panel_frame import compute_concurrency_weight
        # Dates with 5 tickers per bar over 10 bars
        dates = pd.bdate_range("2024-01-01", periods=10).repeat(5)
        w5 = compute_concurrency_weight(dates, lookahead_days=5)
        # Dates with 1 ticker per bar over 50 bars
        dates1 = pd.bdate_range("2024-01-01", periods=50)
        w1 = compute_concurrency_weight(dates1, lookahead_days=5)
        # More concurrent labels ⇒ smaller weights
        assert w5.mean() < w1.mean()

    def test_age_weight_linear_ramp_and_cap(self):
        from training_panel.panel_frame import compute_age_weight
        listing = {"AAA": pd.Timestamp("2024-01-01")}
        dates = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=d) for d in (0, 100, 252, 504, 800)]
        tickers = ["AAA"] * 5
        w = compute_age_weight(dates, tickers, listing_dates=listing, warmup_days=504)
        # Day 0 → ~0 (we set 0 for non-positive age), Day 100 ≈ 100/504, Day 504+ → 1
        assert w.iloc[0] == 0.0
        assert abs(w.iloc[1] - 100 / 504) < 1e-9
        assert abs(w.iloc[2] - 252 / 504) < 1e-9
        assert w.iloc[3] == 1.0
        assert w.iloc[4] == 1.0

    def test_metadata_ticker_stats_correct(self):
        from training_panel.panel_frame import build_panel_frame
        ffs, lbs, sectors = _make_inputs(tickers=("AAA", "BBB"), n=200)
        panel, _, meta = build_panel_frame(ffs, lbs, sectors, min_history_days=20)
        for t in ("AAA", "BBB"):
            stats = meta["per_ticker"][t]
            assert stats["n_rows"] == 200 - 20
            assert stats["first_bar"] <= stats["last_bar"]
        assert meta["n_tickers"] == 2
        assert meta["n_rows"] == (200 - 20) * 2

    def test_raises_when_no_rows_produced(self):
        from training_panel.panel_frame import build_panel_frame
        ffs, lbs, sectors = _make_inputs(n=30)
        with pytest.raises(ValueError):
            build_panel_frame(ffs, lbs, sectors, min_history_days=100)
