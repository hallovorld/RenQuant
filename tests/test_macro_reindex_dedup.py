"""AUDIT REGRESSION GUARD — pins the 2026-05-10 C9 incident.

Walkforward training cutoff=2024-05-06 failed for ticker APP with:
  cannot reindex on an axis with duplicate labels

Root cause: kernel/macro.py:178 `close.sort_index()` preserves duplicate
index rows (yfinance returns dups around splits/dividends). Downstream
kernel/macro_per_ticker.py:122 `reindex(ticker_returns.index)` raises.

Fixes (per §5.13.5 single source of truth + §5.13.11 NaN/dup guards):
- Upstream: macro.py:178 drops duplicate indices BEFORE sort_index
  (mirrors the pattern at macro.py:135 already in place for other paths).
- Defense in depth: macro_per_ticker.py:122 checks `index.has_duplicates`
  and dedups defensively before reindex.

Per CLAUDE.md §5.13.3, this regression guard pins both fixes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = (
    Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
)
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.macro import compute_macro_features  # noqa: E402
from kernel.macro_per_ticker import compute_per_ticker_macro_betas  # noqa: E402


def _ohlcv_with_duplicate_dates(n_rows: int = 100,
                                 n_dups: int = 3) -> pd.DataFrame:
    """Synthetic OHLCV frame with n_dups duplicate index rows."""
    dates = pd.date_range("2024-01-02", periods=n_rows, freq="B")
    base = pd.DataFrame({
        "close": np.linspace(100.0, 130.0, n_rows),
        "volume": np.full(n_rows, 1_000_000.0),
    }, index=dates)
    # Inject n_dups duplicate rows (yfinance-style around splits)
    dup_idx = base.index[::n_rows // (n_dups + 1)][:n_dups]
    dup_rows = base.loc[dup_idx].copy()
    out = pd.concat([base, dup_rows]).sort_index()
    assert out.index.has_duplicates, "fixture must contain duplicates"
    return out


class TestMacroFeaturesDedupsAtSource:
    """Pin: compute_macro_features outputs MUST have unique index even
    when input OHLCV has duplicate dates."""

    def test_output_index_unique_despite_duplicate_input(self):
        ohlcv = _ohlcv_with_duplicate_dates(n_rows=120, n_dups=5)
        out = compute_macro_features(
            symbol="^GSPC", ohlcv=ohlcv,
            transforms=["level_z", "chg_5d_z"],
            rolling_window=20,
        )
        assert out, "no transforms emitted"
        for col_name, series in out.items():
            assert not series.index.has_duplicates, (
                f"{col_name} has duplicate index after compute_macro_features"
            )

    def test_output_index_sorted_ascending(self):
        ohlcv = _ohlcv_with_duplicate_dates(n_rows=80, n_dups=3)
        out = compute_macro_features(
            symbol="^VIX", ohlcv=ohlcv,
            transforms=["level_z"], rolling_window=20,
        )
        for series in out.values():
            assert series.index.is_monotonic_increasing

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=["close", "volume"])
        out = compute_macro_features(
            symbol="^GSPC", ohlcv=empty,
            transforms=["level_z"], rolling_window=20,
        )
        assert out == {}


class TestPerTickerBetaDefenseInDepth:
    """Pin: compute_per_ticker_macro_betas does NOT raise on duplicate
    macro_returns indices, even if upstream dedup is somehow bypassed."""

    def _synthetic_macro_returns(self, n: int = 100,
                                  inject_dups: int = 3) -> pd.DataFrame:
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        df = pd.DataFrame({
            "spy_level_z": np.random.RandomState(0).randn(n),
            "vix_level_z": np.random.RandomState(1).randn(n),
        }, index=dates)
        if inject_dups > 0:
            dup_idx = df.index[::n // (inject_dups + 1)][:inject_dups]
            df = pd.concat([df, df.loc[dup_idx]]).sort_index()
        return df

    def _synthetic_ticker_returns(self, n: int = 100) -> dict:
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        ohlcv = pd.DataFrame({
            "close": np.cumprod(
                1.0 + np.random.RandomState(42).randn(n) * 0.01
            ) * 100.0,
            "volume": np.full(n, 1_000_000.0),
        }, index=dates)
        return {"AAPL": ohlcv}

    def test_does_not_raise_on_duplicate_macro_index(self):
        macro = self._synthetic_macro_returns(n=120, inject_dups=5)
        ohlcv_by_ticker = self._synthetic_ticker_returns(n=120)
        # Must not raise "cannot reindex on an axis with duplicate labels"
        result = compute_per_ticker_macro_betas(
            ohlcv=ohlcv_by_ticker,
            macro_returns=macro,
            rolling_window=20,
            min_window=20,
        )
        assert "AAPL" in result
        assert not result["AAPL"].empty

    def test_clean_macro_index_still_works(self):
        # Baseline: no duplicates → unchanged behavior
        macro = self._synthetic_macro_returns(n=120, inject_dups=0)
        ohlcv_by_ticker = self._synthetic_ticker_returns(n=120)
        result = compute_per_ticker_macro_betas(
            ohlcv=ohlcv_by_ticker,
            macro_returns=macro,
            rolling_window=20,
            min_window=20,
        )
        assert "AAPL" in result


class TestAudit20260510C9Regression:
    """AUDIT REGRESSION GUARD per CLAUDE.md §5.13.3.

    Pins the exact pattern from walkforward chunk_A_v2 log:
        APP: TickerPanelFeatureJob failed — cannot reindex on an axis
             with duplicate labels

    This test combines both fix paths (source dedup + defensive guard)
    by running the full chain on duplicate-containing OHLCV input.
    """

    def test_full_chain_no_reindex_error(self):
        # Simulate APP ticker scenario: OHLCV with duplicate dates
        spy_ohlcv = _ohlcv_with_duplicate_dates(n_rows=120, n_dups=4)
        macro_features = compute_macro_features(
            symbol="^GSPC", ohlcv=spy_ohlcv,
            transforms=["level_z"], rolling_window=20,
        )
        macro_frame = pd.DataFrame(macro_features).sort_index()
        # Manually inject duplicates back into macro_frame (worst case)
        dup_idx = macro_frame.index[::40][:2]
        macro_frame = pd.concat([macro_frame, macro_frame.loc[dup_idx]])
        macro_frame = macro_frame.sort_index()

        # Ticker-side OHLCV must be clean (per fix at macro.py:178 it
        # would be deduped upstream too).
        dates = pd.date_range("2024-01-02", periods=120, freq="B")
        ticker_ohlcv = pd.DataFrame({
            "close": np.linspace(150, 180, 120),
            "volume": np.full(120, 1_000_000.0),
        }, index=dates)

        # The full chain must not raise.
        result = compute_per_ticker_macro_betas(
            ohlcv={"APP": ticker_ohlcv},
            macro_returns=macro_frame,
            rolling_window=20,
            min_window=20,
        )
        assert "APP" in result
        assert isinstance(result["APP"], pd.DataFrame)
