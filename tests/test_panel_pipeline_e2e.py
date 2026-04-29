"""End-to-end tests for training_panel/pipeline.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _make_ohlcv(n: int = 400, seed: int = 0, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    rets = rng.normal(drift, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": close}, index=idx)


def _make_feature_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Simulate the per-ticker feature frame shape produced upstream."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({
        "rsi":         rng.uniform(20, 80, n),
        "macd_hist":   rng.normal(0, 0.5, n),
        "cci":         rng.normal(0, 100, n),
        "bbp":         rng.uniform(0, 1, n),
        "williams_r":  rng.uniform(-100, 0, n),
        "rel_mom_20d": rng.normal(0, 0.05, n),
        "rel_mom_60d": rng.normal(0, 0.08, n),
        "trend":       1 + rng.normal(0, 0.02, n),
        "trend_long":  1 + rng.normal(0, 0.03, n),
    }, index=idx)


def _make_synthetic_cohort(n_tickers: int = 6, n_bars: int = 500):
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    sectors = ["XLK", "XLF", "XLE"]
    ticker_sectors = {t: sectors[i % len(sectors)] for i, t in enumerate(tickers)}

    spy = _make_ohlcv(n_bars, seed=9999, drift=0.0003)
    sector_etf_ohlcv = {s: _make_ohlcv(n_bars, seed=8000 + i, drift=0.0004)
                        for i, s in enumerate(sectors)}

    ohlcv = {t: _make_ohlcv(n_bars, seed=100 + i, drift=0.0005)
             for i, t in enumerate(tickers)}
    feature_frames = {t: _make_feature_frame(n_bars, seed=200 + i)
                      for i, t in enumerate(tickers)}
    listing_dates = {
        t: pd.Timestamp("2010-01-01") for t in tickers  # all seasoned
    }

    config = {
        "lookahead_days": 5,
        "beta_window": 60,
        "min_history_days": 60,    # gentle for small synthetic panels
        "age_warmup_days": 504,
        "cv_n_splits": 3,
        "cv_embargo_days": 5,
        "num_boost_round": 30,
        "neutralize_features": True,
        "nan_prone_cols": ["cci", "williams_r"],
        "training_notes": "e2e test",
        # BUG-CV-2 (2026-04-28): synthetic panels are too small to reach
        # best_iter ≥ 20. Disable the guard for fixtures.
        "panel_ltr": {"min_best_iter": 0},
        "min_best_iter": 0,   # also at top level in case flat-config path is used
    }
    return {
        "watchlist": tickers,
        "feature_frames": feature_frames,
        "ohlcv": ohlcv,
        "spy_ohlcv": spy,
        "sector_etf_ohlcv": sector_etf_ohlcv,
        "ticker_sectors": ticker_sectors,
        "listing_dates": listing_dates,
        "config": config,
    }


class TestEndToEnd:
    def test_e2e_returns_valid_artifact(self, tmp_path):
        from training_panel.pipeline import train_panel_model
        cohort = _make_synthetic_cohort()
        out = tmp_path / "panel_model_v1.json"
        summary = train_panel_model(
            cohort["watchlist"], cohort["feature_frames"], cohort["ohlcv"],
            cohort["spy_ohlcv"], cohort["sector_etf_ohlcv"],
            cohort["ticker_sectors"], cohort["listing_dates"],
            cohort["config"], out,
        )

        # Artifact exists with required keys
        assert out.exists()
        payload = json.loads(out.read_text())
        for k in ("version", "feature_cols", "params", "booster_raw_json"):
            assert k in payload

        # Summary shape
        assert "mean_ic" in summary
        assert "per_fold_ic" in summary
        assert "feature_cols" in summary
        assert Path(summary["artifact_path"]) == out
        # Mean IC is a finite number (may be low on random-ish data)
        assert np.isfinite(summary["mean_ic"])

    def test_artifact_disk_roundtrip(self, tmp_path):
        """Save → load via PanelLTRModel.load → predict again → finite output."""
        from training_panel.pipeline import train_panel_model
        from training_panel.ltr_model import PanelLTRModel

        cohort = _make_synthetic_cohort()
        out = tmp_path / "panel_model_v1.json"
        _ = train_panel_model(
            cohort["watchlist"], cohort["feature_frames"], cohort["ohlcv"],
            cohort["spy_ohlcv"], cohort["sector_etf_ohlcv"],
            cohort["ticker_sectors"], cohort["listing_dates"],
            cohort["config"], out,
        )
        loaded = PanelLTRModel.load(out)
        assert loaded.booster is not None
        assert len(loaded.feature_cols) > 0

    def test_reported_mean_ic_matches_cv_output(self, tmp_path):
        from training_panel.pipeline import train_panel_model

        cohort = _make_synthetic_cohort()
        out = tmp_path / "panel_model_v1.json"
        summary = train_panel_model(
            cohort["watchlist"], cohort["feature_frames"], cohort["ohlcv"],
            cohort["spy_ohlcv"], cohort["sector_etf_ohlcv"],
            cohort["ticker_sectors"], cohort["listing_dates"],
            cohort["config"], out,
        )
        payload = json.loads(out.read_text())
        # Artifact's oos_mean_ic should match summary's mean_ic
        assert abs(payload["oos_mean_ic"] - summary["mean_ic"]) < 1e-9
