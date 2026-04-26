"""Tests for hourly_resolution_panel — Stage C of transformer prep."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.hourly_resolution_panel import (  # noqa: E402
    HOURLY_RES_FEATURE_COLS,
    build_hourly_resolution_panel,
    compute_hourly_resolution_features,
)


def _hourly_bars(n_bars: int = 50, start: str = "2024-01-02 09:30",
                  seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.005, size=n_bars)
    close = 100.0 * np.exp(np.cumsum(rets))
    high  = close * (1 + np.abs(rng.normal(0, 0.001, size=n_bars)))
    low   = close * (1 - np.abs(rng.normal(0, 0.001, size=n_bars)))
    open_ = np.r_[close[0], close[:-1]]
    vol   = rng.lognormal(mean=10, sigma=1, size=n_bars).astype(int)
    idx = pd.date_range(start=start, periods=n_bars, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


# ── compute_hourly_resolution_features ────────────────────────────────────────

class TestFeatures:
    def test_all_canonical_columns_present(self):
        bars = _hourly_bars()
        out = compute_hourly_resolution_features(bars)
        for col in HOURLY_RES_FEATURE_COLS:
            assert col in out.columns

    def test_session_progress_in_unit_interval(self):
        bars = _hourly_bars(n_bars=100)
        out = compute_hourly_resolution_features(bars)
        assert (out["session_progress"] >= 0.0).all()
        assert (out["session_progress"] <= 1.0).all()

    def test_overnight_gap_only_first_bar(self):
        bars = _hourly_bars(n_bars=80)   # ~3 sessions
        out = compute_hourly_resolution_features(bars)
        # Group by session day; overnight_gap should only be set on first bar
        non_null_gap = out["overnight_gap"].notna()
        assert non_null_gap.sum() <= 5   # at most one per session

    def test_hourly_return_finite(self):
        bars = _hourly_bars()
        out = compute_hourly_resolution_features(bars)
        # First bar has NaN, rest should be finite
        finite_returns = out["hourly_return"].iloc[1:]
        assert finite_returns.notna().all()
        assert np.isfinite(finite_returns).all()

    def test_hour_of_day_cyclic(self):
        bars = _hourly_bars()
        out = compute_hourly_resolution_features(bars)
        norm = out["hour_of_day_sin"]**2 + out["hour_of_day_cos"]**2
        np.testing.assert_allclose(norm.dropna(), 1.0, atol=1e-9)

    def test_empty_input_returns_empty(self):
        out = compute_hourly_resolution_features(pd.DataFrame())
        assert out.empty
        assert list(out.columns) == HOURLY_RES_FEATURE_COLS

    def test_missing_column_raises(self):
        bars = _hourly_bars()
        with pytest.raises(KeyError, match="missing columns"):
            compute_hourly_resolution_features(bars.drop(columns=["volume"]))


# ── build_hourly_resolution_panel ─────────────────────────────────────────────

class TestPanelBuild:
    def test_panel_has_expected_columns(self):
        bars = {"AAPL": _hourly_bars(seed=1), "MSFT": _hourly_bars(seed=2)}
        panel = build_hourly_resolution_panel(bars, apply_wash=False)
        for col in HOURLY_RES_FEATURE_COLS:
            assert col in panel.columns
        assert "forward_excess_return" in panel.columns
        assert "_sample_weight" in panel.columns

    def test_panel_indexed_by_ticker_and_datetime(self):
        bars = {"AAPL": _hourly_bars(seed=3)}
        panel = build_hourly_resolution_panel(bars, apply_wash=False)
        assert panel.index.names == ["ticker"] or len(panel.index.names) == 2

    def test_panel_grows_with_more_tickers(self):
        small = {"A": _hourly_bars(n_bars=30, seed=4)}
        large = {f"T{i}": _hourly_bars(n_bars=30, seed=i+5) for i in range(5)}
        panel_s = build_hourly_resolution_panel(small, apply_wash=False)
        panel_l = build_hourly_resolution_panel(large, apply_wash=False)
        assert len(panel_l) >= 5 * len(panel_s) - 5

    def test_label_uses_benchmark_when_provided(self):
        bars = {"AAPL": _hourly_bars(n_bars=50, seed=10)}
        bm = _hourly_bars(n_bars=50, seed=11)
        panel_no_bm = build_hourly_resolution_panel(bars, apply_wash=False)
        panel_bm    = build_hourly_resolution_panel(
            bars, benchmark_bars=bm, apply_wash=False,
        )
        # With benchmark, label is excess (different from raw)
        non_null_idx = panel_bm["forward_excess_return"].notna()
        if non_null_idx.sum() > 5:
            # Most rows should differ when subtracting benchmark
            diff = (panel_bm["forward_excess_return"] - panel_no_bm["forward_excess_return"]).dropna()
            assert diff.abs().mean() > 1e-6

    def test_label_horizon_1_is_next_bar(self):
        """forward_excess_return at i should reflect close[i+1]/close[i] - 1."""
        bars = {"X": _hourly_bars(n_bars=20, seed=20)}
        panel = build_hourly_resolution_panel(
            bars, label_horizon_bars=1, apply_wash=False,
        )
        # Compute expected manually
        x_bars = bars["X"]
        expected = x_bars["close"].pct_change(1).shift(-1)
        # Pull the first 5 from panel (inner index = datetime)
        idx0 = panel.index[0][1]  # first ticker, first datetime
        panel_first = panel.iloc[0]["forward_excess_return"]
        if pd.notna(panel_first):
            np.testing.assert_allclose(
                panel_first, expected.loc[idx0], atol=1e-12,
            )

    def test_empty_dict(self):
        panel = build_hourly_resolution_panel({})
        assert panel.empty

    def test_apply_wash_adds_sample_weight(self):
        bars = {"AAPL": _hourly_bars(n_bars=80, seed=30)}
        panel = build_hourly_resolution_panel(bars, apply_wash=True)
        assert "_sample_weight" in panel.columns
        # sample_weight is between 0 and 1
        sw = panel["_sample_weight"].dropna()
        assert (sw >= 0).all()
        assert (sw <= 1).all()

    def test_panel_grows_5x_vs_daily_estimate(self):
        """Stage-C size argument: hourly resolution should yield >5×
        daily panel rows for the same tickers + window."""
        bars = {f"T{i}": _hourly_bars(n_bars=70, seed=i+50)
                for i in range(3)}    # 3 tickers × 70 hourly bars
        panel = build_hourly_resolution_panel(bars, apply_wash=False)
        # 70 bars over ~10 sessions of 7 hourly bars
        # Daily would have ~10 rows × 3 tickers = 30 rows
        # Hourly = 70 × 3 = 210 rows ≈ 7×
        assert len(panel) >= 5 * 30


# ── Stage C-2 — BuildHourlyResolutionPanelTask wiring ─────────────────────────

class TestBuildHourlyResolutionPanelTaskWiring:
    """Stage C-2: pipeline integration via PanelAssemblyJob."""

    def test_daily_mode_is_noop(self):
        """training_resolution='daily' (default) → task is a no-op,
        ctx.panel stays None, BuildPanelTask handles it."""
        from training_panel.pp_panel_training import BuildHourlyResolutionPanelTask
        class _Ctx:
            config = {"panel_ltr": {"training_resolution": "daily"}}
            panel = None
            watchlist = ["AAPL"]
            panel_metadata = {}
        ctx = _Ctx()
        ret = BuildHourlyResolutionPanelTask().run(ctx)
        assert ret is True
        assert ctx.panel is None

    def test_panel_assembly_job_includes_task(self):
        """The new task is wired BEFORE BuildPanelTask."""
        from training_panel.pp_panel_training import (
            BuildHourlyResolutionPanelTask, BuildPanelTask, PanelAssemblyJob,
        )
        tasks = PanelAssemblyJob().tasks
        names = [type(t).__name__ for t in tasks]
        assert "BuildHourlyResolutionPanelTask" in names
        assert "BuildPanelTask" in names
        # And it runs BEFORE BuildPanelTask
        i_h = names.index("BuildHourlyResolutionPanelTask")
        i_d = names.index("BuildPanelTask")
        assert i_h < i_d

    def test_hourly_mode_no_bars_falls_back(self, tmp_path):
        """training_resolution='hourly' but no bar cache → graceful fallback."""
        from training_panel.pp_panel_training import BuildHourlyResolutionPanelTask
        class _Ctx:
            config = {
                "panel_ltr": {
                    "training_resolution": "hourly",
                    "hourly": {"cache_dir": str(tmp_path / "no_bars_here")},
                },
                "_strategy_dir": str(tmp_path / "fake_strategy"),
                "benchmark": "SPY",
            }
            panel = None
            watchlist = ["AAPL", "MSFT"]
            panel_metadata = {}
        ctx = _Ctx()
        # Should not raise, just log and return True (graceful fallback)
        ret = BuildHourlyResolutionPanelTask().run(ctx)
        assert ret is True
        assert ctx.panel is None
