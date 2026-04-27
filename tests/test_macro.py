"""Tests for kernel/macro.py — MacroFactorStore + macro feature builder.

Per macro design doc (doc/components/macro-factor-frame-design.md) §11
safety harness — these tests pin failure modes F1-F5 + F9.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.macro import (   # noqa: E402
    DEFAULT_MACRO_SYMBOLS,
    DEFAULT_TRANSFORMS,
    DEFAULT_ROLLING_WINDOW,
    MacroFactorStore,
    build_macro_frame,
    compute_macro_features,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synth_ohlcv(n_days: int = 500, start: str = "2024-01-01",
                  drift: float = 0.0001, vol: float = 0.01,
                  seed: int = 0) -> pd.DataFrame:
    """Build synthetic OHLCV with controllable returns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    rets = rng.normal(drift, vol, size=n_days)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "open":   close * (1 + rng.normal(0, 0.001, n_days)),
        "high":   close * (1 + np.abs(rng.normal(0, 0.002, n_days))),
        "low":    close * (1 - np.abs(rng.normal(0, 0.002, n_days))),
        "close":  close,
        "volume": rng.integers(1_000_000, 10_000_000, n_days).astype(float),
    }, index=dates)


# ── MacroFactorStore — store roundtrip + corruption defense ───────────────────

class TestStoreRoundTrip:
    def test_save_then_load_returns_same(self, tmp_path):
        store = MacroFactorStore(data_dir=tmp_path)
        df = _synth_ohlcv(100)
        store.save(df, "VXX")
        loaded = store.load("VXX")
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, df.sort_index(), check_freq=False)

    def test_load_missing_returns_none(self, tmp_path):
        store = MacroFactorStore(data_dir=tmp_path)
        assert store.load("NONEXISTENT") is None

    def test_save_atomic_via_tmp(self, tmp_path):
        """Save uses .tmp + replace to avoid partial-write corruption."""
        store = MacroFactorStore(data_dir=tmp_path)
        df = _synth_ohlcv(50)
        store.save(df, "VXX")
        # No .tmp file should remain after successful save
        tmps = list(tmp_path.glob("*.tmp"))
        assert tmps == []

    def test_save_dedupes_on_index(self, tmp_path):
        store = MacroFactorStore(data_dir=tmp_path)
        df1 = _synth_ohlcv(50, start="2024-01-01")
        df2 = _synth_ohlcv(50, start="2024-01-15")  # overlaps
        store.save(df1, "VXX")
        store.save(df2, "VXX")
        loaded = store.load("VXX")
        # Should have unique dates (no duplicates)
        assert loaded.index.is_unique

    def test_F9_corrupt_parquet_treated_as_cache_miss(self, tmp_path):
        """Safety F9: corrupt parquet → log warn + return None."""
        store = MacroFactorStore(data_dir=tmp_path)
        path = store._path("BAD")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not a valid parquet")
        result = store.load("BAD")
        assert result is None


# ── compute_macro_features — z-score correctness ──────────────────────────────

class TestComputeMacroFeatures:
    def test_default_transforms_produce_3_columns(self):
        df = _synth_ohlcv(500)
        feats = compute_macro_features(df, symbol="VXX")
        # 3 transforms × 1 symbol = 3 cols
        assert set(feats.keys()) == {"vxx_level_z", "vxx_chg_5d_z", "vxx_chg_20d_z"}

    def test_each_series_has_same_length_as_input(self):
        df = _synth_ohlcv(500)
        feats = compute_macro_features(df, symbol="HYG")
        for col, s in feats.items():
            assert len(s) == len(df)

    def test_z_scores_mostly_in_5_sigma_band(self):
        """Healthy z-score: ≥99% in [-5, 5] band (no inf, no extreme outliers)."""
        df = _synth_ohlcv(1000, vol=0.01, seed=42)
        feats = compute_macro_features(df, symbol="HYG", rolling_window=200)
        for col, s in feats.items():
            non_nan = s.dropna()
            if len(non_nan) == 0:
                continue
            in_band_pct = float(((non_nan >= -5) & (non_nan <= 5)).mean())
            assert in_band_pct >= 0.99, f"{col}: only {in_band_pct:.3f} in [-5,5]"

    def test_F5_zero_variance_clamped_to_zero(self):
        """Constant series → z-score = 0 (not inf or NaN)."""
        # Constant close prices = zero variance everywhere
        df = pd.DataFrame({
            "close":  [100.0] * 500,
            "open":   [100.0] * 500,
            "high":   [100.0] * 500,
            "low":    [100.0] * 500,
            "volume": [1e6] * 500,
        }, index=pd.date_range("2024-01-01", periods=500))
        feats = compute_macro_features(df, symbol="FLAT")
        for col, s in feats.items():
            assert np.all(np.isfinite(s) | s.isna()), \
                f"{col} contains inf — F5 clamp failed"
            # All values should be 0 or NaN, never inf
            non_nan = s.dropna()
            assert (non_nan == 0.0).all() or len(non_nan) == 0

    def test_unknown_transform_logged_and_skipped(self):
        df = _synth_ohlcv(100)
        feats = compute_macro_features(df, symbol="VXX",
                                        transforms=["level_z", "made_up_xform"])
        assert "vxx_level_z" in feats
        assert "vxx_made_up_xform" not in feats

    def test_empty_input_returns_empty(self):
        feats = compute_macro_features(pd.DataFrame(), symbol="VXX")
        assert feats == {}

    def test_missing_close_column_returns_empty(self):
        df = pd.DataFrame({"high": [1.0]}, index=[pd.Timestamp("2024-01-01")])
        feats = compute_macro_features(df, symbol="VXX")
        assert feats == {}


# ── build_macro_frame — full assembly + safety F1/F2/F4 ───────────────────────

class TestBuildMacroFrame:
    def test_assembles_3_cols_per_symbol(self, tmp_path):
        store = MacroFactorStore(data_dir=tmp_path)
        for sym in ["VXX", "HYG"]:
            store.save(_synth_ohlcv(500, seed=hash(sym) % 1000), sym)
        frame, meta = build_macro_frame(store, symbols=["VXX", "HYG"])
        # 2 symbols × 3 transforms = 6 cols
        assert len(frame.columns) == 6
        assert meta["symbols_used"] == ["VXX", "HYG"]
        assert meta["n_features"] == 6
        assert meta["symbols_skipped"] == []

    def test_F2_missing_symbol_skipped(self, tmp_path):
        """One symbol cached, one not → frame has 3 cols, other is in skipped."""
        store = MacroFactorStore(data_dir=tmp_path)
        store.save(_synth_ohlcv(500), "VXX")
        # HYG never saved
        frame, meta = build_macro_frame(store, symbols=["VXX", "HYG"])
        assert "vxx_level_z" in frame.columns
        assert not any(c.startswith("hyg_") for c in frame.columns)
        assert meta["symbols_used"] == ["VXX"]
        skipped_syms = {s[0] for s in meta["symbols_skipped"]}
        assert "HYG" in skipped_syms

    def test_F1_load_exception_doesnt_kill_others(self, tmp_path, monkeypatch):
        """If store.load() raises, that symbol is skipped, others proceed."""
        store = MacroFactorStore(data_dir=tmp_path)
        store.save(_synth_ohlcv(500), "VXX")

        original_load = store.load
        def flaky_load(sym: str):
            if sym == "POISON":
                raise RuntimeError("simulated load explosion")
            return original_load(sym)
        monkeypatch.setattr(store, "load", flaky_load)

        frame, meta = build_macro_frame(store, symbols=["VXX", "POISON"])
        assert "vxx_level_z" in frame.columns
        skipped_syms = {s[0] for s in meta["symbols_skipped"]}
        assert "POISON" in skipped_syms

    def test_F4_short_history_drops_macro(self, tmp_path):
        """If macro has insufficient warmup coverage, drop it."""
        store = MacroFactorStore(data_dir=tmp_path)
        # Symbol with only 30 days of history; rolling window=252
        store.save(_synth_ohlcv(30), "BABY_ETF")
        # Plenty of training window
        training_end = pd.Timestamp("2024-12-31")
        frame, meta = build_macro_frame(
            store, symbols=["BABY_ETF"],
            rolling_window=252,
            min_window_overlap_pct=0.95,
            training_end=training_end,
        )
        # Should be dropped
        assert "baby_etf_level_z" not in frame.columns
        skipped_syms = {s[0] for s in meta["symbols_skipped"]}
        assert "BABY_ETF" in skipped_syms

    def test_F4_long_history_keeps_macro(self, tmp_path):
        """Plenty of history → kept."""
        store = MacroFactorStore(data_dir=tmp_path)
        store.save(_synth_ohlcv(2000, start="2020-01-01"), "OLD_ETF")
        frame, meta = build_macro_frame(
            store, symbols=["OLD_ETF"],
            rolling_window=252,
            training_end=pd.Timestamp("2024-12-31"),
        )
        assert "old_etf_level_z" in frame.columns

    def test_no_symbols_cached_returns_empty_frame(self, tmp_path):
        store = MacroFactorStore(data_dir=tmp_path)
        frame, meta = build_macro_frame(store, symbols=["VXX"])
        assert frame.empty
        assert meta["n_features"] == 0

    def test_frame_is_date_indexed_and_sorted(self, tmp_path):
        store = MacroFactorStore(data_dir=tmp_path)
        store.save(_synth_ohlcv(500), "VXX")
        frame, _ = build_macro_frame(store, symbols=["VXX"])
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert frame.index.is_monotonic_increasing


# ── Defaults sanity ───────────────────────────────────────────────────────────

class TestDefaults:
    def test_default_symbols_match_design(self):
        """The 11 symbols in design doc §2 must match DEFAULT_MACRO_SYMBOLS."""
        expected = ["VXX", "HYG", "UUP", "DBC", "GLD", "TLT", "XLV", "XLU",
                    "KRE", "MTUM", "USMV"]
        assert DEFAULT_MACRO_SYMBOLS == expected

    def test_default_transforms_match_design(self):
        assert DEFAULT_TRANSFORMS == ["level_z", "chg_5d_z", "chg_20d_z"]

    def test_default_rolling_window_252(self):
        """252 trading days = ~1 year. Per design, prevents look-ahead bias."""
        assert DEFAULT_ROLLING_WINDOW == 252
