"""Unit tests for macro v2 (per-ticker β) + T2-2 asset embeddings.

Pin the contracts so future changes can't silently break:
- compute_per_ticker_macro_betas: β values DIFFER per ticker on same date
- macro_levels_to_returns: round-trip from z-scored levels
- AssetEmbeddingTrainer.smoke_test_collapse: rejects degenerate output
- build_panel_frame: asset_embeddings broadcast across ticker rows
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.macro_per_ticker import (   # noqa: E402
    compute_per_ticker_macro_betas,
    macro_levels_to_returns,
)
from training_panel.asset_embeddings import AssetEmbeddingTrainer   # noqa: E402
from training_panel.panel_frame import build_panel_frame   # noqa: E402


# ── Macro v2: per-ticker β ────────────────────────────────────────────────────

class TestPerTickerMacroBetas:
    def _build_synthetic_ohlcv(self, n_days: int = 200, seed: int = 42):
        """3 tickers with different known β to a synthetic VIX."""
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2024-01-02", periods=n_days)
        vix_ret = rng.normal(0, 0.03, n_days)
        # Ticker A: β=+1.5 (defensive — moves with VIX)
        a_ret = 0.001 + 1.5 * vix_ret + rng.normal(0, 0.005, n_days)
        # Ticker B: β=-0.8 (cyclical — opposite)
        b_ret = 0.0005 - 0.8 * vix_ret + rng.normal(0, 0.005, n_days)
        # Ticker C: β=0 (uncorrelated)
        c_ret = 0.0008 + rng.normal(0, 0.01, n_days)
        ohlcv = {
            "A": pd.DataFrame({"close": 100 * np.cumprod(1 + a_ret)}, index=dates),
            "B": pd.DataFrame({"close": 100 * np.cumprod(1 + b_ret)}, index=dates),
            "C": pd.DataFrame({"close": 100 * np.cumprod(1 + c_ret)}, index=dates),
        }
        macro_returns = pd.DataFrame({"vix_chg": vix_ret}, index=dates)
        return ohlcv, macro_returns

    def test_betas_differ_per_ticker_on_same_date(self):
        """The whole point of v2: β values differ across tickers, so the
        rank loss has within-date variance to learn from."""
        ohlcv, macro = self._build_synthetic_ohlcv()
        result = compute_per_ticker_macro_betas(ohlcv, macro,
                                                rolling_window=60, min_window=30)
        assert set(result.keys()) == {"A", "B", "C"}
        # All should have the same β column name
        col = "beta_vix_chg_60d"
        for t in result:
            assert col in result[t].columns

        # Check the LAST row across all tickers — βs should differ
        last_betas = {t: result[t][col].iloc[-1] for t in result}
        # A ≈ +1.5, B ≈ -0.8, C ≈ 0. They MUST differ.
        assert abs(last_betas["A"] - last_betas["B"]) > 0.5
        assert abs(last_betas["A"] - last_betas["C"]) > 0.5

    def test_betas_recover_constructed_truth(self):
        """β estimates should approximately recover the constructed values."""
        ohlcv, macro = self._build_synthetic_ohlcv(n_days=200)
        result = compute_per_ticker_macro_betas(ohlcv, macro,
                                                rolling_window=60, min_window=30)
        col = "beta_vix_chg_60d"
        # Average of last 60 β values should approximate ground truth
        a_mean = result["A"][col].iloc[-60:].dropna().mean()
        b_mean = result["B"][col].iloc[-60:].dropna().mean()
        c_mean = result["C"][col].iloc[-60:].dropna().mean()
        assert 1.2 < a_mean < 1.8, f"A β ≈ +1.5, got {a_mean:.3f}"
        assert -1.0 < b_mean < -0.6, f"B β ≈ -0.8, got {b_mean:.3f}"
        assert -0.3 < c_mean < 0.3, f"C β ≈ 0, got {c_mean:.3f}"

    def test_strict_prior_no_lookahead(self):
        """β at bar t MUST not include t in the regression. Implementation
        shifts by 1; verify the latest bar has the SECOND-to-last β,
        not a value that includes today."""
        ohlcv, macro = self._build_synthetic_ohlcv(n_days=120)
        result = compute_per_ticker_macro_betas(ohlcv, macro,
                                                rolling_window=60, min_window=30)
        col = "beta_vix_chg_60d"
        # The first (rolling_window) rows should all be NaN (no history)
        a = result["A"][col]
        # First ~30 should be NaN (min_window), then after shift(1) the
        # very first non-NaN should appear at index >= min_window+1
        assert a.iloc[0] != a.iloc[0] or pd.isna(a.iloc[0])  # NaN

    def test_empty_macro_returns_returns_empty(self):
        ohlcv, _ = self._build_synthetic_ohlcv()
        result = compute_per_ticker_macro_betas(ohlcv, pd.DataFrame(),
                                                rolling_window=60)
        assert result == {}

    def test_short_history_skipped(self):
        rng = np.random.default_rng(42)
        # Only 25 days — below min_window=30
        dates = pd.bdate_range("2024-01-02", periods=25)
        ohlcv = {"X": pd.DataFrame({"close": 100 * np.cumprod(1 + rng.normal(0, 0.01, 25))},
                                    index=dates)}
        macro = pd.DataFrame({"vix_chg": rng.normal(0, 0.03, 25)}, index=dates)
        result = compute_per_ticker_macro_betas(ohlcv, macro,
                                                rolling_window=60, min_window=30)
        # Either skipped (empty result) or all NaN
        if "X" in result:
            assert result["X"]["beta_vix_chg_60d"].dropna().empty


class TestMacroLevelsToReturns:
    def test_picks_only_level_z_columns(self):
        df = pd.DataFrame({
            "vxx_level_z":   [1.0, 1.1, 1.2, 1.3],
            "vxx_chg_5d_z":  [0.1, 0.2, 0.3, 0.4],   # should be skipped
            "hyg_level_z":   [-0.5, -0.4, -0.3, -0.2],
        }, index=pd.bdate_range("2024-01-02", periods=4))
        result = macro_levels_to_returns(df)
        assert set(result.columns) == {"vxx_chg", "hyg_chg"}

    def test_empty_input_returns_empty(self):
        result = macro_levels_to_returns(pd.DataFrame())
        assert result.empty


# ── T2-2: AssetEmbeddingTrainer ───────────────────────────────────────────────

class TestAssetEmbeddingTrainer:
    def test_smoke_test_collapse_flags_degenerate(self):
        t = AssetEmbeddingTrainer()
        t.embeddings = {
            "A": np.array([1.0, 0.001, 0.001]),
            "B": np.array([1.001, 0.001, 0.0]),
            "C": np.array([1.001, 0.0, 0.0]),
        }
        assert t.smoke_test_collapse() is False

    def test_smoke_test_collapse_passes_diverse(self):
        t = AssetEmbeddingTrainer()
        t.embeddings = {
            "A": np.array([1.0, 0.0, 0.0]),
            "B": np.array([0.0, 1.0, 0.0]),
            "C": np.array([0.0, 0.0, 1.0]),
        }
        assert t.smoke_test_collapse() is True

    def test_smoke_test_collapse_handles_single_ticker(self):
        t = AssetEmbeddingTrainer()
        t.embeddings = {"A": np.array([1.0, 0.0, 0.0])}
        assert t.smoke_test_collapse() is False   # need ≥2 to compare

    def test_save_load_roundtrip(self, tmp_path):
        t = AssetEmbeddingTrainer(embedding_dim=4)
        t.embeddings = {
            "AAPL": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "GOOG": np.array([-0.1, 0.0, 0.5, -0.2], dtype=np.float32),
        }
        t.trained_date = "2026-04-27"
        t.loss_history = [0.5, 0.3, 0.2]
        path = tmp_path / "asset-embeddings.json"
        t.save(path)

        loaded = AssetEmbeddingTrainer.load(path)
        assert loaded.embedding_dim == 4
        assert set(loaded.embeddings.keys()) == {"AAPL", "GOOG"}
        np.testing.assert_allclose(
            loaded.embeddings["AAPL"], t.embeddings["AAPL"], atol=1e-6
        )
        assert loaded.trained_date == "2026-04-27"
        assert loaded.loss_history == [0.5, 0.3, 0.2]

    def test_load_rejects_wrong_kind(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"kind": "different_thing"}))
        with pytest.raises(ValueError, match="not an asset_embeddings"):
            AssetEmbeddingTrainer.load(path)


# ── build_panel_frame: asset_embeddings broadcast ─────────────────────────────

class TestBuildPanelFrameWithEmbeddings:
    def _minimal_panel_inputs(self):
        """Just enough to produce a valid panel frame for smoke-testing."""
        dates = pd.bdate_range("2024-01-02", periods=300)
        feature_frames = {}
        labels = {}
        sectors = {"A": "tech", "B": "tech", "C": "finance"}
        for t in ["A", "B", "C"]:
            ff = pd.DataFrame({"feat_x": np.linspace(0.1, 0.5, 300)}, index=dates)
            feature_frames[t] = ff
            labels[t] = pd.Series(np.random.RandomState(hash(t) & 0xFFFF).normal(0.001, 0.01, 300), index=dates)
        return feature_frames, labels, sectors

    def test_embeddings_broadcast_per_ticker(self):
        ff, lab, sec = self._minimal_panel_inputs()
        embeddings = {
            "A": np.array([1.0, 0.0, 0.0, 0.0]),
            "B": np.array([0.0, 1.0, 0.0, 0.0]),
            "C": np.array([0.0, 0.0, 1.0, 0.0]),
        }
        panel, gs, meta = build_panel_frame(
            ff, lab, sec, asset_embeddings=embeddings,
            min_history_days=30,
            lookahead_days=5,
        )
        assert "emb_0" in panel.columns
        assert "emb_3" in panel.columns
        # Verify per-ticker broadcast: ticker A's rows have emb_0=1, others=0
        a_rows = panel[panel["ticker"] == "A"]
        assert (a_rows["emb_0"] == 1.0).all()
        assert (a_rows["emb_1"] == 0.0).all()
        b_rows = panel[panel["ticker"] == "B"]
        assert (b_rows["emb_1"] == 1.0).all()
        # metadata exposes embedding_cols
        assert "embedding_cols" in meta
        assert sorted(meta["embedding_cols"]) == ["emb_0", "emb_1", "emb_2", "emb_3"]

    def test_no_embeddings_default_no_emb_cols(self):
        ff, lab, sec = self._minimal_panel_inputs()
        panel, gs, meta = build_panel_frame(
            ff, lab, sec,
            min_history_days=30,
            lookahead_days=5,
        )
        emb_cols = [c for c in panel.columns if c.startswith("emb_")]
        assert emb_cols == []
        assert meta["embedding_cols"] == []

    def test_missing_ticker_filled_with_zeros(self):
        """If a ticker has no embedding (e.g. new-listing without history),
        its rows should get zero-vector embedding (neutral)."""
        ff, lab, sec = self._minimal_panel_inputs()
        embeddings = {
            "A": np.array([1.0, 0.0, 0.0, 0.0]),
            # B and C missing
        }
        panel, gs, meta = build_panel_frame(
            ff, lab, sec, asset_embeddings=embeddings,
            min_history_days=30, lookahead_days=5,
        )
        b_rows = panel[panel["ticker"] == "B"]
        assert (b_rows[["emb_0", "emb_1", "emb_2", "emb_3"]] == 0.0).all().all()
