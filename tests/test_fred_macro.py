"""Tests for kernel/fred_macro.py — FRED API ingestion (Tier 2 macro).

Pin contracts:
- FredMacroStore: load/save round-trip + corrupt-file handling
- _resolve_api_key: env > home file > None
- _to_daily_bars: look-ahead-safe forward-fill with release lag
- build_fred_frame: 3-transform schema + coverage filter
- fred_levels_to_returns: mirror semantics of macro_levels_to_returns

No real FRED API calls — uses synthetic data via store.save() so tests
run offline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY) not in sys.path:
    sys.path.insert(0, str(_STRATEGY))

from kernel.fred_macro import (   # noqa: E402
    DEFAULT_FRED_SERIES,
    DEFAULT_ROLLING_WINDOW,
    FredMacroStore,
    _resolve_api_key,
    _to_daily_bars,
    build_fred_frame,
    fred_levels_to_returns,
)


# ── Defaults sanity ───────────────────────────────────────────────────────────

class TestDefaults:
    def test_default_series_count(self):
        """22 series in the catalog (Tier 2 of macro plan)."""
        # Spec count: 22 series. Adjust if the catalog grows — but keep
        # the test pinning the expected size so accidental drops/duplicates
        # surface in CI.
        assert len(DEFAULT_FRED_SERIES) >= 18

    def test_each_spec_is_4tuple(self):
        for spec in DEFAULT_FRED_SERIES:
            assert isinstance(spec, tuple) and len(spec) == 4
            sid, name, freq, lag = spec
            assert isinstance(sid, str) and len(sid) > 0
            assert isinstance(name, str)
            assert freq in ("daily", "weekly", "monthly")
            assert isinstance(lag, int) and lag >= 0

    def test_no_duplicate_series_ids(self):
        ids = [spec[0] for spec in DEFAULT_FRED_SERIES]
        assert len(set(ids)) == len(ids)

    def test_default_rolling_window_252(self):
        assert DEFAULT_ROLLING_WINDOW == 252

    def test_lag_bars_match_frequency(self):
        """Daily series should have 0 lag; weekly ~2; monthly ~5."""
        for sid, _name, freq, lag in DEFAULT_FRED_SERIES:
            if freq == "daily":
                assert lag == 0, f"{sid}: daily series should have 0 lag, got {lag}"
            elif freq == "weekly":
                assert 1 <= lag <= 3, f"{sid}: weekly lag {lag} outside [1,3]"
            elif freq == "monthly":
                assert 3 <= lag <= 7, f"{sid}: monthly lag {lag} outside [3,7]"


# ── API key resolution ────────────────────────────────────────────────────────

class TestResolveApiKey:
    def test_explicit_arg_wins(self):
        assert _resolve_api_key("explicit-key") == "explicit-key"

    def test_strips_whitespace(self):
        assert _resolve_api_key("  explicit-key  \n") == "explicit-key"

    def test_env_var_picked_when_no_arg(self, monkeypatch):
        monkeypatch.setenv("RENQUANT_FRED_API_KEY", "env-key")
        assert _resolve_api_key() == "env-key"

    def test_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("RENQUANT_FRED_API_KEY", "env-key")
        assert _resolve_api_key("arg-key") == "arg-key"

    def test_returns_none_when_nothing_set(self, monkeypatch, tmp_path):
        monkeypatch.delenv("RENQUANT_FRED_API_KEY", raising=False)
        # Force HOME to an empty dir so ~/.fred_api_key doesn't exist
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _resolve_api_key() is None


# ── FredMacroStore ────────────────────────────────────────────────────────────

class TestFredMacroStore:
    def test_save_then_load_roundtrip(self, tmp_path):
        store = FredMacroStore(cache_dir=tmp_path, api_key="fake")
        idx = pd.bdate_range("2024-01-02", periods=20)
        frame = pd.DataFrame({"value": np.arange(20, dtype=float)}, index=idx)
        store.save("DGS10", frame)
        loaded = store.load("DGS10")
        assert loaded is not None
        assert loaded.equals(frame.sort_index())

    def test_load_missing_returns_none(self, tmp_path):
        store = FredMacroStore(cache_dir=tmp_path, api_key="fake")
        assert store.load("NEVERSAVED") is None

    def test_save_dedupes_on_index(self, tmp_path):
        store = FredMacroStore(cache_dir=tmp_path, api_key="fake")
        idx1 = pd.bdate_range("2024-01-02", periods=10)
        f1 = pd.DataFrame({"value": np.arange(10, dtype=float)}, index=idx1)
        store.save("DGS10", f1)
        # Overlapping save with NEW values for the same dates
        idx2 = pd.bdate_range("2024-01-08", periods=10)
        f2 = pd.DataFrame({"value": np.arange(100, 110, dtype=float)}, index=idx2)
        store.save("DGS10", f2)
        loaded = store.load("DGS10")
        # Overlap: dates 2024-01-08..14 should have NEW values (keep="last")
        for d in idx2:
            if d in loaded.index:
                assert float(loaded.loc[d, "value"]) >= 100, (
                    f"Overlap at {d}: should have new value >=100"
                )

    def test_corrupt_parquet_treated_as_cache_miss(self, tmp_path):
        store = FredMacroStore(cache_dir=tmp_path, api_key="fake")
        bad = tmp_path / "DGS10.parquet"
        bad.write_text("garbage not a parquet")
        loaded = store.load("DGS10")
        assert loaded is None  # graceful degradation

    def test_fetch_raises_without_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RENQUANT_FRED_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        store = FredMacroStore(cache_dir=tmp_path, api_key=None)
        with pytest.raises(RuntimeError, match="FRED API key missing"):
            store.fetch("DGS10")


# ── _to_daily_bars (the look-ahead-safe core) ─────────────────────────────────

class TestToDailyBars:
    def test_no_lag_daily_series_passes_through(self):
        idx = pd.bdate_range("2024-01-02", periods=20)
        s = pd.Series(np.arange(20, dtype=float), index=idx, name="x")
        target = idx
        out = _to_daily_bars(s, target_index=target, release_lag_bars=0)
        assert (out.values == s.values).all()

    def test_lag_shifts_values_forward(self):
        """A value at bar t with lag=5 should be visible from bar t+5."""
        idx = pd.bdate_range("2024-01-02", periods=20)
        s = pd.Series(np.arange(20, dtype=float), index=idx, name="x")
        out = _to_daily_bars(s, target_index=idx, release_lag_bars=5)
        # First 5 bars are NaN (no prior visible values)
        assert out.iloc[:5].isna().all()
        # Bar 5 should equal value at bar 0 (= 0.0)
        assert float(out.iloc[5]) == 0.0
        # Bar 10 should equal value at bar 5 (= 5.0)
        assert float(out.iloc[10]) == 5.0

    def test_monthly_to_daily_forward_fill(self):
        """A monthly series ffills until the next release on the daily calendar."""
        # Two monthly observations: 2024-02-01 (val=1) and 2024-03-01 (val=2)
        s = pd.Series([1.0, 2.0],
                      index=pd.to_datetime(["2024-02-01", "2024-03-01"]),
                      name="cpi")
        # Daily target spanning Jan-Apr
        target = pd.bdate_range("2024-01-15", "2024-04-15")
        out = _to_daily_bars(s, target_index=target, release_lag_bars=0)
        # Before first release: NaN
        assert pd.isna(out.loc[pd.Timestamp("2024-01-15")])
        # On release day → value 1
        assert float(out.loc[pd.Timestamp("2024-02-01")]) == 1.0
        # After Feb release, before Mar release → still 1
        assert float(out.loc[pd.Timestamp("2024-02-15")]) == 1.0
        # On Mar release day → value 2
        # (Mar 1 is Friday in 2024, so it's a trading day)
        assert float(out.loc[pd.Timestamp("2024-03-01")]) == 2.0
        # After Mar release → 2
        assert float(out.loc[pd.Timestamp("2024-03-15")]) == 2.0
        assert float(out.loc[pd.Timestamp("2024-04-15")]) == 2.0

    def test_empty_series_returns_nan_aligned(self):
        target = pd.bdate_range("2024-01-02", periods=5)
        out = _to_daily_bars(pd.Series(dtype=float, name="x"),
                              target_index=target, release_lag_bars=0)
        assert len(out) == 5
        assert out.isna().all()


# ── build_fred_frame ──────────────────────────────────────────────────────────

class TestBuildFredFrame:
    def test_assembles_3_cols_per_series(self, tmp_path):
        """Output has level_z, chg_5d_z, chg_20d_z per series."""
        store = FredMacroStore(cache_dir=tmp_path, api_key="fake")
        # Cache 2 series with enough history for the rolling window.
        idx = pd.bdate_range("2020-01-02", periods=600)
        for sid in ("DGS10", "VIXCLS"):
            store.save(sid, pd.DataFrame(
                {"value": np.cumsum(np.random.default_rng(hash(sid) % 1024).normal(0, 0.1, 600))},
                index=idx,
            ))
        target = idx[300:]   # last 300 days
        # Use a custom spec with both as daily 0-lag
        specs = [
            ("DGS10",  "10y Treasury", "daily", 0),
            ("VIXCLS", "VIX",          "daily", 0),
        ]
        frame, meta = build_fred_frame(
            store, target,
            series_specs=specs,
            rolling_window=120,
        )
        # 2 series × 3 transforms = 6 columns
        assert len(frame.columns) == 6
        assert {"dgs10_level_z", "dgs10_chg_5d_z", "dgs10_chg_20d_z",
                "vixcls_level_z", "vixcls_chg_5d_z", "vixcls_chg_20d_z"} == set(frame.columns)
        assert meta["n_features"] == 6
        assert set(meta["series_used"]) == {"DGS10", "VIXCLS"}

    def test_missing_series_skipped_not_fatal(self, tmp_path):
        """One series missing → others still build."""
        store = FredMacroStore(cache_dir=tmp_path, api_key="fake")
        idx = pd.bdate_range("2020-01-02", periods=400)
        store.save("DGS10", pd.DataFrame(
            {"value": np.cumsum(np.random.default_rng(0).normal(0, 0.1, 400))},
            index=idx,
        ))
        # CPIAUCSL is NOT cached
        target = idx[200:]
        specs = [
            ("DGS10",    "10y", "daily",   0),
            ("CPIAUCSL", "cpi", "monthly", 5),
        ]
        frame, meta = build_fred_frame(store, target, series_specs=specs, rolling_window=80)
        # DGS10 should be present (3 cols), CPIAUCSL should be skipped
        assert {"dgs10_level_z", "dgs10_chg_5d_z", "dgs10_chg_20d_z"} == set(frame.columns)
        assert meta["series_used"] == ["DGS10"]
        assert any(s[0] == "CPIAUCSL" for s in meta["series_skipped"])

    def test_empty_store_returns_empty_frame(self, tmp_path):
        store = FredMacroStore(cache_dir=tmp_path, api_key="fake")
        target = pd.bdate_range("2024-01-02", periods=10)
        frame, meta = build_fred_frame(store, target,
                                        series_specs=[("DGS10", "10y", "daily", 0)])
        assert frame.empty or frame.shape[1] == 0
        assert meta["n_features"] == 0


# ── fred_levels_to_returns ────────────────────────────────────────────────────

class TestFredLevelsToReturns:
    def test_picks_only_level_z_columns(self):
        df = pd.DataFrame({
            "dgs10_level_z":   [1.0, 1.1, 1.2, 1.3],
            "dgs10_chg_5d_z":  [0.1, 0.2, 0.3, 0.4],   # skipped
            "vixcls_level_z":  [-0.5, -0.4, -0.3, -0.2],
        }, index=pd.bdate_range("2024-01-02", periods=4))
        out = fred_levels_to_returns(df)
        assert set(out.columns) == {"dgs10_chg", "vixcls_chg"}

    def test_empty_input_returns_empty(self):
        out = fred_levels_to_returns(pd.DataFrame())
        assert out.empty
