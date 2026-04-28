"""Phase 1F — end-to-end smoke test for macro_factor_frame.

Exercises the full path (LoadMacroFactorsTask → ctx.macro_factor_frame
→ BuildPanelTask → build_panel_frame merge → ctx.panel includes macro
columns) on a tiny synthetic panel.

This catches integration bugs that unit tests miss (e.g. config wiring,
config-key typos, ctx field naming drift, merge-into-actual-panel).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.macro import MacroFactorStore  # noqa: E402
from training_panel.context import PanelTrainingContext  # noqa: E402
from training_panel.pp_panel_training import LoadMacroFactorsTask   # noqa: E402
from training_panel.panel_frame import build_panel_frame  # noqa: E402


def _synth_ohlcv(n: int = 1000, start: str = "2022-01-01",
                  drift: float = 0.0001, vol: float = 0.01,
                  seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV with controllable returns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    rets = rng.normal(drift, vol, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": 1e6,
    }, index=dates)


# ── End-to-end: macro DISABLED → no macro cols in panel (regression) ─────────

class TestE2EDisabled:
    def test_disabled_path_panel_has_no_macro_cols(self, tmp_path):
        """When panel_ltr.macro.enabled=false, the entire pipeline is
        a no-op — panel doesn't gain any macro columns. This is the
        critical safety claim: feature OFF == zero behavioral change."""
        # Cache populated but flag off — must NOT show up in panel.
        cache = tmp_path / "macro"
        store = MacroFactorStore(data_dir=cache)
        store.save(_synth_ohlcv(1000, seed=1), "VXX")

        ctx = PanelTrainingContext(
            config={
                "panel_ltr": {
                    "macro": {
                        "enabled":  False,    # ← OFF
                        "cache_dir": str(cache),
                        "symbols": ["VXX"],
                    },
                },
                "benchmark": "SPY",
            },
            ohlcv={"SPY": _synth_ohlcv(1000)},
        )
        # Run the load task
        LoadMacroFactorsTask().run(ctx)
        assert ctx.macro_factor_frame is None, \
            "disabled flag must leave macro_factor_frame as None"

        # Now build a panel without macro
        n = 500
        ff = {"AAA": pd.DataFrame({
            "rsi":  np.random.normal(0, 1, n),
            "macd": np.random.normal(0, 1, n),
        }, index=pd.date_range("2024-01-01", periods=n))}
        labels = {"AAA": pd.Series(np.random.normal(0, 0.01, n),
                                     index=pd.date_range("2024-01-01", periods=n))}
        sectors = {"AAA": "tech"}
        panel, _, meta = build_panel_frame(
            ff, labels, sectors,
            macro_frame=ctx.macro_factor_frame,   # None
            min_history_days=200,
        )
        assert meta["macro_cols"] == []
        # No vix_/hyg_/etc cols anywhere
        assert not any(c.startswith("vxx_") for c in panel.columns)


# ── End-to-end: macro ENABLED → macro cols broadcast across panel ────────────

class TestE2EEnabled:
    def test_enabled_path_panel_has_macro_cols(self, tmp_path):
        """When flag on + cache populated, panel has macro features
        broadcast to every (date, ticker) row."""
        cache = tmp_path / "macro"
        store = MacroFactorStore(data_dir=cache)
        # 2 symbols → 6 macro cols (3 transforms each)
        store.save(_synth_ohlcv(1500, seed=1), "VXX")
        store.save(_synth_ohlcv(1500, seed=2), "HYG")

        ctx = PanelTrainingContext(
            config={
                "panel_ltr": {
                    "macro": {
                        "enabled":  True,
                        "cache_dir": str(cache),
                        "symbols": ["VXX", "HYG"],
                        "rolling_window": 252,
                    },
                },
                "benchmark": "SPY",
            },
            ohlcv={"SPY": _synth_ohlcv(1500)},
        )

        # Step 1: load task populates ctx.macro_factor_frame
        LoadMacroFactorsTask().run(ctx)
        assert ctx.macro_factor_frame is not None
        assert not ctx.macro_factor_frame.empty
        assert len(ctx.macro_factor_frame.columns) == 6   # 2 sym × 3 transforms

        # Step 2: build per-ticker panel
        n = 500
        dates = pd.date_range("2024-01-01", periods=n)
        rng = np.random.default_rng(0)
        ff = {
            t: pd.DataFrame({
                "rsi":  rng.normal(0, 1, n),
                "macd": rng.normal(0, 1, n),
            }, index=dates)
            for t in ["AAA", "BBB", "CCC"]
        }
        labels = {t: pd.Series(rng.normal(0, 0.01, n), index=dates)
                  for t in ff.keys()}
        sectors = {t: "tech" for t in ff.keys()}

        # Step 3: build_panel_frame WITH macro.
        # Bug-1 fix (2026-04-27) made v1 broadcast a no-op by default
        # (within-date variance = 0 → zero gradient + dilutes feature
        # set). force_broadcast=True opts back in to exercise the merge
        # path that this E2E smoke is verifying.
        panel, _, meta = build_panel_frame(
            ff, labels, sectors,
            macro_frame=ctx.macro_factor_frame,
            min_history_days=200,
            force_broadcast=True,
        )

        # Verify macro cols are in panel
        for col in ["vxx_level_z", "vxx_chg_5d_z", "vxx_chg_20d_z",
                     "hyg_level_z", "hyg_chg_5d_z", "hyg_chg_20d_z"]:
            assert col in panel.columns, f"macro col {col} missing"
        assert len(meta["macro_cols"]) == 6

        # Per-date broadcast: same date → same macro value across tickers
        date0 = panel["date"].iloc[100]
        rows = panel[panel["date"] == date0]
        assert len(rows) == 3   # 3 tickers
        assert rows["vxx_level_z"].nunique() == 1, \
            "macro must be same across all tickers for a given date"

    def test_enabled_no_cache_falls_back_safely(self, tmp_path):
        """Flag on but cache empty → ctx.macro_factor_frame is empty
        DataFrame; build_panel_frame skips merge; panel has no macro cols.
        Nothing breaks."""
        cache = tmp_path / "macro"
        cache.mkdir()
        # Empty cache

        ctx = PanelTrainingContext(
            config={
                "panel_ltr": {
                    "macro": {
                        "enabled":  True,
                        "cache_dir": str(cache),
                        "symbols": ["VXX"],
                    },
                },
                "benchmark": "SPY",
            },
            ohlcv={"SPY": _synth_ohlcv(500)},
        )
        LoadMacroFactorsTask().run(ctx)
        # Empty cache → empty frame
        assert ctx.macro_factor_frame is not None
        assert ctx.macro_factor_frame.empty

        # build_panel_frame with empty frame → no-op
        n = 400
        dates = pd.date_range("2024-01-01", periods=n)
        ff = {"AAA": pd.DataFrame({"rsi": np.zeros(n)}, index=dates)}
        labels = {"AAA": pd.Series(np.zeros(n), index=dates)}
        sectors = {"AAA": "tech"}
        panel, _, meta = build_panel_frame(
            ff, labels, sectors,
            macro_frame=ctx.macro_factor_frame,
            min_history_days=200,
        )
        assert meta["macro_cols"] == []
        # No macro cols in panel
        assert not any(c.startswith("vxx_") for c in panel.columns)
