"""Tests for Phase 1C: panel_frame.build_panel_frame macro_frame merge.

Per macro design doc Phase 1C — adds optional `macro_frame` parameter
to build_panel_frame; broadcasts macro features to every panel row by
date with forward-fill within ticker (weekend/holiday alignment).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.panel_frame import build_panel_frame   # noqa: E402


def _per_ticker_features(ticker: str, n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + abs(hash(ticker)) % 1000)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "rsi":  rng.normal(0, 1, n_days),
        "macd": rng.normal(0, 1, n_days),
    }, index=dates)


def _label(ticker: str, n_days: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed + abs(hash(ticker)) % 1000 + 7)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.Series(rng.normal(0, 0.01, n_days), index=dates)


def _build_simple_inputs(tickers: list[str], n_days: int = 300):
    feature_frames = {t: _per_ticker_features(t, n_days, seed=i)
                       for i, t in enumerate(tickers)}
    labels = {t: _label(t, n_days, seed=i) for i, t in enumerate(tickers)}
    sectors = {t: "tech" for t in tickers}
    return feature_frames, labels, sectors


def _macro_frame(n_days: int = 300, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or ["vix_level_z", "hyg_level_z", "uup_chg_5d_z"]
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {c: rng.normal(0, 1, n_days) for c in cols},
        index=dates,
    )


# ── Backwards compatibility — no macro_frame is no-op ──────────────────────────

class TestBackwardsCompat:
    def test_no_macro_frame_is_identical_to_before(self):
        """When macro_frame=None, panel must look exactly like the
        pre-Phase-1C path."""
        ff, lab, sec = _build_simple_inputs(["A", "B", "C"], n_days=400)
        panel, _, meta = build_panel_frame(
            ff, lab, sec, min_history_days=252,
        )
        # No vix_/hyg_/etc. cols
        for col in panel.columns:
            assert not col.startswith("vix_"), f"unexpected macro col {col}"
        assert meta.get("macro_cols", []) == []

    def test_empty_macro_frame_is_no_op(self):
        ff, lab, sec = _build_simple_inputs(["A", "B"], n_days=400)
        panel, _, meta = build_panel_frame(
            ff, lab, sec, macro_frame=pd.DataFrame(),
            min_history_days=252,
        )
        assert meta.get("macro_cols", []) == []


# ── Macro broadcast — happy path ──────────────────────────────────────────────

class TestMacroBroadcast:
    def test_macro_columns_added_to_panel(self):
        ff, lab, sec = _build_simple_inputs(["A", "B"], n_days=400)
        macro = _macro_frame(n_days=400)
        panel, _, meta = build_panel_frame(
            ff, lab, sec, macro_frame=macro, min_history_days=252,
        )
        for col in macro.columns:
            assert col in panel.columns, f"macro col {col} missing from panel"
        assert set(meta["macro_cols"]) == set(macro.columns)

    def test_macro_value_broadcast_per_date(self):
        """Same date, same macro value across all tickers."""
        ff, lab, sec = _build_simple_inputs(["A", "B", "C"], n_days=400)
        macro = _macro_frame(n_days=400)
        panel, _, _ = build_panel_frame(
            ff, lab, sec, macro_frame=macro, min_history_days=252,
        )
        # Pick any date; assert all 3 tickers have same vix_level_z
        sample_date = panel["date"].iloc[100]
        rows = panel[panel["date"] == sample_date]
        assert len(rows) == 3   # 3 tickers
        assert rows["vix_level_z"].nunique() == 1, \
            "macro must be broadcast (same value across tickers per date)"

    def test_panel_row_count_unchanged(self):
        """Macro merge must not add or drop panel rows."""
        ff, lab, sec = _build_simple_inputs(["A", "B"], n_days=400)
        macro = _macro_frame(n_days=400)
        no_macro_panel, _, _ = build_panel_frame(
            ff, lab, sec, min_history_days=252,
        )
        macro_panel, _, _ = build_panel_frame(
            ff, lab, sec, macro_frame=macro, min_history_days=252,
        )
        assert len(no_macro_panel) == len(macro_panel)


# ── Forward-fill (weekend / holiday alignment) ────────────────────────────────

class TestForwardFill:
    def test_macro_with_weekday_only_dates_ffilled(self):
        """Macro frame indexed by weekdays only; panel may have
        Saturdays (synthetic). Forward-fill should populate missing."""
        ff, lab, sec = _build_simple_inputs(["A"], n_days=400)
        # Macro has only every-other day — simulates weekend gaps
        sparse_dates = pd.date_range("2024-01-01", periods=200, freq="2D")
        macro = pd.DataFrame(
            {"vix_level_z": np.linspace(-1, 1, 200)},
            index=sparse_dates,
        )
        panel, _, _ = build_panel_frame(
            ff, lab, sec, macro_frame=macro, min_history_days=252,
        )
        # No NaN in macro col — all forward-filled
        # (trailing NaN goes to 0.0, not missing)
        assert not panel["vix_level_z"].isna().any()


# ── Trailing NaN → 0.0 (warmup safety) ────────────────────────────────────────

class TestTrailingNaN:
    def test_warmup_nan_filled_with_zero(self):
        """Macro frame's first N rows are NaN (rolling-z warmup);
        panel rows in that window must show 0.0, not NaN."""
        ff, lab, sec = _build_simple_inputs(["A"], n_days=400)
        macro = _macro_frame(n_days=400)
        # Inject NaN into first 100 rows of vix_level_z (simulates warmup)
        macro.iloc[:100, macro.columns.get_loc("vix_level_z")] = np.nan
        panel, _, _ = build_panel_frame(
            ff, lab, sec, macro_frame=macro, min_history_days=252,
        )
        # NO NaN should remain
        assert not panel["vix_level_z"].isna().any(), \
            "trailing NaN must be filled with 0.0"


# ── Column collision defense ──────────────────────────────────────────────────

class TestColumnCollision:
    def test_collision_with_existing_column_renamed(self):
        """If a macro col name collides with a feature col, suffix '_macro'."""
        ff, lab, sec = _build_simple_inputs(["A"], n_days=400)
        # Use 'rsi' as macro col — collides with existing feature
        macro = pd.DataFrame(
            {"rsi": np.linspace(-1, 1, 400)},
            index=pd.date_range("2024-01-01", periods=400, freq="D"),
        )
        panel, _, meta = build_panel_frame(
            ff, lab, sec, macro_frame=macro, min_history_days=252,
        )
        # Original 'rsi' should still be the per-ticker feature (not overwritten)
        # Renamed 'rsi_macro' should be the broadcast macro
        assert "rsi" in panel.columns
        assert "rsi_macro" in panel.columns
        assert "rsi_macro" in meta["macro_cols"]


# ── Datetime index normalization ──────────────────────────────────────────────

class TestIndexNormalization:
    def test_string_indexed_macro_normalized(self):
        """If macro_frame.index is not DatetimeIndex (operator built it
        wrong), build_panel_frame coerces."""
        ff, lab, sec = _build_simple_inputs(["A"], n_days=400)
        # Build with string index
        macro = pd.DataFrame(
            {"vix_level_z": np.zeros(400)},
            index=[f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}"
                   for i in range(400)],
        )
        # Trim to valid dates only (some made-up dates won't parse)
        # Use a cleaner builder
        valid_dates = pd.date_range("2024-01-01", periods=400, freq="D")
        macro = pd.DataFrame(
            {"vix_level_z": np.zeros(400)},
            index=[d.strftime("%Y-%m-%d") for d in valid_dates],
        )
        # Should not raise
        panel, _, _ = build_panel_frame(
            ff, lab, sec, macro_frame=macro, min_history_days=252,
        )
        assert "vix_level_z" in panel.columns
