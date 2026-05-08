"""Tests for training_panel/tasks_data_cleaning.py — Qlib-standard processors."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from training_panel.tasks_data_cleaning import (
    CSZScoreNormFeaturesTask,
    DataCleaningJob,
    FillnaTask,
    ProcessInfTask,
    RobustZScoreNormTask,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_ctx(n_dates=10, n_tickers=20, n_feats=5, seed=0):
    """Minimal PanelTrainingContext stub with panel DataFrame."""
    from unittest.mock import MagicMock
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rows = []
    for d in dates:
        for t in [f"T{i:02d}" for i in range(n_tickers)]:
            row = {"date": d, "ticker": t, "split_label": "train"}
            for f in range(n_feats):
                row[f"f{f}"] = rng.normal()
            rows.append(row)
    panel = pd.DataFrame(rows)
    ctx = MagicMock()
    ctx.panel = panel
    ctx.feature_cols = [f"f{i}" for i in range(n_feats)]
    ctx._clean_train_mask = panel["split_label"] == "train"
    return ctx


# ── ProcessInfTask ────────────────────────────────────────────────────────

class TestProcessInfTask:
    def test_replaces_pos_inf(self):
        ctx = _make_ctx()
        ctx.panel.loc[0, "f0"] = np.inf
        ProcessInfTask().run(ctx)
        assert not np.isinf(ctx.panel["f0"]).any()
        assert ctx.panel.loc[0, "f0"] != ctx.panel.loc[0, "f0"]  # is NaN

    def test_replaces_neg_inf(self):
        ctx = _make_ctx()
        ctx.panel.loc[1, "f1"] = -np.inf
        ProcessInfTask().run(ctx)
        assert not np.isinf(ctx.panel["f1"]).any()

    def test_noop_when_no_inf(self):
        ctx = _make_ctx()
        before = ctx.panel[ctx.feature_cols].copy()
        ProcessInfTask().run(ctx)
        pd.testing.assert_frame_equal(ctx.panel[ctx.feature_cols], before)

    def test_skips_on_none_panel(self):
        ctx = _make_ctx()
        ctx.panel = None
        assert ProcessInfTask().run(ctx) is None


# ── RobustZScoreNormTask ──────────────────────────────────────────────────

class TestRobustZScoreNormTask:
    def test_output_clipped_at_3_sigma(self):
        ctx = _make_ctx()
        # Insert extreme outlier
        ctx.panel.loc[0, "f0"] = 1000.0
        RobustZScoreNormTask().run(ctx)
        assert ctx.panel["f0"].max() <= 3.0 + 1e-9
        assert ctx.panel["f0"].min() >= -3.0 - 1e-9

    def test_stores_feature_stats(self):
        ctx = _make_ctx()
        RobustZScoreNormTask().run(ctx)
        assert hasattr(ctx, "_clean_feature_stats")
        stats = ctx._clean_feature_stats
        assert "f0" in stats
        assert "median" in stats["f0"]
        assert "mad_scale" in stats["f0"]

    def test_uses_train_only_stats(self):
        ctx = _make_ctx(n_dates=20)
        # Mark last 5 dates as test — give them extreme values
        dates = sorted(ctx.panel["date"].unique())
        test_dates = dates[-5:]
        ctx.panel.loc[ctx.panel["date"].isin(test_dates), "split_label"] = "test"
        ctx._clean_train_mask = ctx.panel["split_label"] == "train"
        ctx.panel.loc[ctx.panel["split_label"] == "test", "f0"] = 500.0
        RobustZScoreNormTask().run(ctx)
        # Median computed on train should not be 500; test rows clipped at ±3
        assert ctx.panel.loc[ctx.panel["split_label"] == "test", "f0"].max() <= 3.0 + 1e-9


# ── CSZScoreNormFeaturesTask ──────────────────────────────────────────────

class TestCSZScoreNormFeaturesTask:
    def test_cross_section_mean_near_zero(self):
        ctx = _make_ctx(n_dates=10, n_tickers=50)
        CSZScoreNormFeaturesTask().run(ctx)
        daily_means = ctx.panel.groupby("date")["f0"].mean()
        assert (daily_means.abs() < 1e-6).all(), f"Daily means not ≈0: {daily_means.describe()}"

    def test_cross_section_std_near_one(self):
        ctx = _make_ctx(n_dates=10, n_tickers=50)
        CSZScoreNormFeaturesTask().run(ctx)
        daily_stds = ctx.panel.groupby("date")["f0"].std()
        assert ((daily_stds - 1.0).abs() < 0.05).all()

    def test_skips_missing_date_col(self):
        ctx = _make_ctx()
        ctx.panel = ctx.panel.drop(columns=["date"])
        result = CSZScoreNormFeaturesTask().run(ctx)
        assert result is None


# ── FillnaTask ────────────────────────────────────────────────────────────

class TestFillnaTask:
    def test_fills_nan_with_zero(self):
        ctx = _make_ctx()
        ctx.panel.loc[0, "f0"] = np.nan
        ctx.panel.loc[5, "f2"] = np.nan
        FillnaTask().run(ctx)
        assert not ctx.panel[ctx.feature_cols].isna().any().any()
        assert ctx.panel.loc[0, "f0"] == 0.0

    def test_noop_when_no_nan(self):
        ctx = _make_ctx()
        ctx.panel[ctx.feature_cols] = ctx.panel[ctx.feature_cols].fillna(0.0)
        before = ctx.panel[ctx.feature_cols].copy()
        FillnaTask().run(ctx)
        pd.testing.assert_frame_equal(ctx.panel[ctx.feature_cols], before)


# ── DataCleaningJob ───────────────────────────────────────────────────────

class TestDataCleaningJob:
    def test_full_pipeline_no_nan_no_inf(self):
        ctx = _make_ctx(n_tickers=30)
        ctx.panel.loc[0, "f0"] = np.inf
        ctx.panel.loc[1, "f1"] = np.nan
        job = DataCleaningJob()
        for task in job.tasks:
            task.run(ctx)
        assert not ctx.panel[ctx.feature_cols].isna().any().any()
        assert not np.isinf(ctx.panel[ctx.feature_cols].values).any()

    def test_should_skip_when_no_panel(self):
        ctx = _make_ctx()
        ctx.panel = None
        assert DataCleaningJob().should_skip(ctx) is True

    def test_four_tasks_in_order(self):
        job = DataCleaningJob()
        names = [t.name for t in job.tasks]
        assert names == [
            "ProcessInfTask",
            "RobustZScoreNormTask",
            "CSZScoreNormFeaturesTask",
            "FillnaTask",
        ]
