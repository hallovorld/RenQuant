"""Tests for training_panel/imputation.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_ff(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({
        "rsi": rng.uniform(20, 80, n),
        "macd_hist": rng.normal(0, 0.5, n),
    }, index=idx)


class TestApplyMinHistoryGate:
    def test_gate_drops_expected_row_count(self):
        from training_panel.imputation import apply_min_history_gate
        ffs = {"A": _make_ff(300, 0), "B": _make_ff(300, 1)}
        out = apply_min_history_gate(ffs, min_history_days=100)
        assert len(out["A"]) == 200
        assert len(out["B"]) == 200

    def test_short_tickers_dropped(self):
        from training_panel.imputation import apply_min_history_gate
        ffs = {"A": _make_ff(300, 0), "SHORT": _make_ff(50, 1)}
        out = apply_min_history_gate(ffs, min_history_days=100)
        assert "A" in out
        assert "SHORT" not in out

    def test_does_not_mutate_input(self):
        from training_panel.imputation import apply_min_history_gate
        ffs = {"A": _make_ff(300, 0)}
        orig_len = len(ffs["A"])
        _ = apply_min_history_gate(ffs, min_history_days=100)
        assert len(ffs["A"]) == orig_len


class TestAddMissingnessIndicators:
    def test_indicator_equals_1_iff_nan(self):
        from training_panel.imputation import add_missingness_indicators
        panel = pd.DataFrame({
            "rsi": [10.0, np.nan, 30.0, np.nan, 50.0],
            "adx": [1.0, 2.0, np.nan, 4.0, 5.0],
        })
        out = add_missingness_indicators(panel, cols=["rsi", "adx"])
        assert out["rsi_is_missing"].tolist() == [0, 1, 0, 1, 0]
        assert out["adx_is_missing"].tolist() == [0, 0, 1, 0, 0]

    def test_dtype_int8(self):
        from training_panel.imputation import add_missingness_indicators
        panel = pd.DataFrame({"rsi": [np.nan, 1.0]})
        out = add_missingness_indicators(panel, cols=["rsi"])
        assert out["rsi_is_missing"].dtype == np.int8

    def test_skips_unknown_columns(self):
        from training_panel.imputation import add_missingness_indicators
        panel = pd.DataFrame({"rsi": [1.0, 2.0]})
        out = add_missingness_indicators(panel, cols=["rsi", "nonexistent"])
        assert "rsi_is_missing" in out.columns
        assert "nonexistent_is_missing" not in out.columns


class TestSectorMedianFill:
    def test_sector_median_respects_date_bucket(self):
        from training_panel.imputation import sector_median_fill
        d = pd.Timestamp("2024-01-03")
        panel = pd.DataFrame({
            "date": [d, d, d, d],
            "sector": ["TECH", "TECH", "TECH", "FIN"],
            "ticker": ["A", "B", "C", "D"],
            "x": [1.0, 3.0, np.nan, 100.0],
        })
        out = sector_median_fill(panel, cols=["x"])
        # TECH median (excluding NaN) = median([1,3]) = 2 → fills C's NaN
        assert out.loc[2, "x"] == 2.0
        # FIN D unaffected
        assert out.loc[3, "x"] == 100.0

    def test_sector_median_unaffected_by_other_sector_values(self):
        from training_panel.imputation import sector_median_fill
        d = pd.Timestamp("2024-01-03")
        panel = pd.DataFrame({
            "date":   [d, d, d],
            "sector": ["TECH", "FIN", "TECH"],
            "ticker": ["A", "B", "C"],
            "x":      [2.0, 1000.0, np.nan],
        })
        out = sector_median_fill(panel, cols=["x"])
        # C's fill should be 2.0 (the TECH median), NOT influenced by FIN's 1000
        assert out.loc[2, "x"] == 2.0

    def test_falls_back_to_date_median_when_sector_all_nan(self):
        from training_panel.imputation import sector_median_fill
        d = pd.Timestamp("2024-01-03")
        panel = pd.DataFrame({
            "date":   [d, d, d, d],
            "sector": ["TECH", "FIN", "FIN", "ENE"],
            "ticker": ["A", "B", "C", "D"],
            "x":      [10.0, np.nan, np.nan, 20.0],
        })
        out = sector_median_fill(panel, cols=["x"])
        # FIN has all NaN, so falls back to date median of [10, 20] = 15
        assert out.loc[1, "x"] == 15.0
        assert out.loc[2, "x"] == 15.0

    def test_different_dates_not_cross_contaminated(self):
        from training_panel.imputation import sector_median_fill
        d1 = pd.Timestamp("2024-01-03")
        d2 = pd.Timestamp("2024-01-04")
        panel = pd.DataFrame({
            "date":   [d1, d1, d2, d2],
            "sector": ["TECH", "TECH", "TECH", "TECH"],
            "ticker": ["A", "B", "A", "B"],
            "x":      [1.0, np.nan, 100.0, np.nan],
        })
        out = sector_median_fill(panel, cols=["x"])
        # B on day 1 filled with day-1 TECH median = 1.0
        assert out.loc[1, "x"] == 1.0
        # B on day 2 filled with day-2 TECH median = 100.0
        assert out.loc[3, "x"] == 100.0


class TestComputeAgeWeight:
    def test_age_weight_linear_ramp(self):
        from training_panel.imputation import compute_age_weight
        dates = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
                 for d in (0, 100, 252, 504, 800)]
        panel = pd.DataFrame({
            "date":   dates,
            "ticker": ["A"] * 5,
        })
        listing = {"A": pd.Timestamp("2024-01-01")}
        w = compute_age_weight(panel, listing, warmup_days=504)
        assert w.iloc[0] == 0.0
        assert abs(w.iloc[1] - 100 / 504) < 1e-9
        assert abs(w.iloc[2] - 252 / 504) < 1e-9
        assert w.iloc[3] == 1.0
        assert w.iloc[4] == 1.0

    def test_age_weight_caps_at_1(self):
        from training_panel.imputation import compute_age_weight
        panel = pd.DataFrame({
            "date":   [pd.Timestamp("2024-12-31")],
            "ticker": ["A"],
        })
        listing = {"A": pd.Timestamp("2000-01-01")}
        w = compute_age_weight(panel, listing, warmup_days=504)
        assert w.iloc[0] == 1.0

    def test_unknown_ticker_weight_one(self):
        from training_panel.imputation import compute_age_weight
        panel = pd.DataFrame({
            "date":   [pd.Timestamp("2024-01-01")],
            "ticker": ["UNKNOWN"],
        })
        w = compute_age_weight(panel, {"A": pd.Timestamp("2020-01-01")}, warmup_days=504)
        assert w.iloc[0] == 1.0

    def test_empty_listing_all_ones(self):
        from training_panel.imputation import compute_age_weight
        panel = pd.DataFrame({
            "date":   [pd.Timestamp("2024-01-01")] * 3,
            "ticker": ["A", "B", "C"],
        })
        w = compute_age_weight(panel, {}, warmup_days=504)
        assert (w == 1.0).all()
