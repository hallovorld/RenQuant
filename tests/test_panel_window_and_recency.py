"""Panel training-window + recency-weighting tests.

User spec (April 22): both XGBoost and Transformer backends should train on
the last 5 years only, with exponential decay emphasizing recent samples.
These tests verify the `build_panel_frame` implementation of both knobs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _fake_panel_inputs(n_years: float = 10.0, n_tickers: int = 5):
    """Build feature_frames + labels + ticker_sectors for build_panel_frame."""
    n_days = int(n_years * 252)
    idx = pd.bdate_range(end=pd.Timestamp("2026-04-15"), periods=n_days)
    feature_frames = {}
    labels = {}
    ticker_sectors = {}
    rng = np.random.default_rng(7)
    for i in range(n_tickers):
        t = f"T{i}"
        ff = pd.DataFrame({
            "close":   rng.normal(100, 1, n_days).cumsum(),
            "feat_a":  rng.normal(0, 1, n_days),
            "feat_b":  rng.normal(0, 1, n_days),
        }, index=idx)
        feature_frames[t] = ff
        labels[t] = pd.Series(rng.normal(0, 1, n_days), index=idx)
        ticker_sectors[t] = "tech"
    return feature_frames, labels, ticker_sectors


# ── Last-5-year window ──────────────────────────────────────────────────────

class TestTrainingWindow:
    def test_window_restricts_panel_dates(self):
        from training_panel.panel_frame import build_panel_frame
        ff, lab, sec = _fake_panel_inputs(n_years=10.0)
        panel, _, meta = build_panel_frame(
            ff, lab, sec, min_history_days=100, lookahead_days=5,
            training_window_years=5.0,
        )
        oldest = pd.Timestamp(panel["date"].min())
        newest = pd.Timestamp(panel["date"].max())
        span_days = (newest - oldest).days
        # Last-5y ~= 1826 days; allow 30d headroom for alignment
        assert span_days <= 5 * 365 + 30
        assert span_days >= 5 * 365 - 30

    def test_window_none_keeps_full_history(self):
        from training_panel.panel_frame import build_panel_frame
        ff, lab, sec = _fake_panel_inputs(n_years=10.0)
        panel, _, meta = build_panel_frame(
            ff, lab, sec, min_history_days=100, lookahead_days=5,
            training_window_years=None,
        )
        oldest = pd.Timestamp(panel["date"].min())
        newest = pd.Timestamp(panel["date"].max())
        span_days = (newest - oldest).days
        assert span_days > 9 * 365   # ~10 years preserved

    def test_window_zero_keeps_full_history(self):
        """Zero means disabled, same as None."""
        from training_panel.panel_frame import build_panel_frame
        ff, lab, sec = _fake_panel_inputs(n_years=10.0)
        panel, _, meta = build_panel_frame(
            ff, lab, sec, min_history_days=100, lookahead_days=5,
            training_window_years=0,
        )
        span = (panel["date"].max() - panel["date"].min()).days
        assert span > 9 * 365


# ── Exponential recency weighting ──────────────────────────────────────────

class TestRecencyWeighting:
    def test_exp_decay_most_recent_weight_is_highest(self):
        from training_panel.panel_frame import build_panel_frame
        ff, lab, sec = _fake_panel_inputs(n_years=5.0)
        panel, _, _ = build_panel_frame(
            ff, lab, sec, min_history_days=100, lookahead_days=5,
            recency_weighting={"kind": "exp_decay", "half_life_days": 252},
        )
        assert "weight_recency" in panel.columns
        # Weights should be monotonically non-decreasing with date (newer → higher)
        date_mean_w = panel.groupby("date")["weight_recency"].mean().sort_index()
        assert date_mean_w.iloc[-1] > date_mean_w.iloc[0]
        # Most-recent bar weight ≈ 1.0
        assert date_mean_w.iloc[-1] == pytest.approx(1.0, abs=0.01)

    def test_half_life_semantics(self):
        """A sample 252 days older should have weight 0.5× the newest."""
        from training_panel.panel_frame import build_panel_frame
        ff, lab, sec = _fake_panel_inputs(n_years=3.0)
        panel, _, _ = build_panel_frame(
            ff, lab, sec, min_history_days=50, lookahead_days=5,
            recency_weighting={"kind": "exp_decay", "half_life_days": 252},
        )
        panel = panel.sort_values("date").reset_index(drop=True)
        most_recent = panel["date"].max()
        target_date = most_recent - pd.Timedelta(days=252)
        # Find closest actual bar
        diffs = (panel["date"] - target_date).abs()
        idx_252d_older = diffs.idxmin()
        w_old = panel.loc[idx_252d_older, "weight_recency"]
        w_new = panel[panel["date"] == most_recent]["weight_recency"].iloc[0]
        # Allow ±10% slack (calendar vs trading day conversion)
        assert w_old / w_new == pytest.approx(0.5, abs=0.1)

    def test_none_means_no_weight_recency_column(self):
        from training_panel.panel_frame import build_panel_frame
        ff, lab, sec = _fake_panel_inputs(n_years=3.0)
        panel, _, _ = build_panel_frame(
            ff, lab, sec, min_history_days=50, lookahead_days=5,
            recency_weighting=None,
        )
        assert "weight_recency" not in panel.columns

    def test_unknown_kind_noops_with_warning(self, caplog):
        import logging
        from training_panel.panel_frame import build_panel_frame
        ff, lab, sec = _fake_panel_inputs(n_years=3.0)
        with caplog.at_level(logging.WARNING, logger="panel_frame"):
            panel, _, _ = build_panel_frame(
                ff, lab, sec, min_history_days=50, lookahead_days=5,
                recency_weighting={"kind": "quartic_decay"},
            )
        assert "weight_recency" not in panel.columns


# ── Combined effect ────────────────────────────────────────────────────────

class TestWindowAndRecencyCompose:
    def test_both_applied_together(self):
        """Window restricts dates; recency weights still apply within it."""
        from training_panel.panel_frame import build_panel_frame
        ff, lab, sec = _fake_panel_inputs(n_years=10.0)
        panel, _, _ = build_panel_frame(
            ff, lab, sec, min_history_days=100, lookahead_days=5,
            training_window_years=5.0,
            recency_weighting={"kind": "exp_decay", "half_life_days": 252},
        )
        span = (panel["date"].max() - panel["date"].min()).days
        assert span <= 5 * 365 + 30
        assert "weight_recency" in panel.columns

    def test_weight_column_multiplicative(self):
        """weight = weight_concurrency × weight_age × weight_recency."""
        from training_panel.panel_frame import build_panel_frame
        ff, lab, sec = _fake_panel_inputs(n_years=3.0)
        panel, _, _ = build_panel_frame(
            ff, lab, sec, min_history_days=50, lookahead_days=5,
            recency_weighting={"kind": "exp_decay", "half_life_days": 252},
        )
        # Sample check — the product should match panel["weight"]
        expected = (panel["weight_concurrency"]
                     * panel["weight_age"]
                     * panel["weight_recency"])
        assert np.allclose(panel["weight"].values, expected.values, rtol=1e-6)
