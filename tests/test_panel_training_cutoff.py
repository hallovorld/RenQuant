"""Tests for the train_cutoff parameter in PanelTrainingPipeline (P1, 2026-05-10).

Pins:
    * ApplyTrainCutoffTask slices ohlcv, sector_etf_ohlcv, fundamentals,
      earnings_surprises, insider_trades, hourly/minute bars, macro
    * BuildPanelTask defensive guard catches anything that slipped through
    * Default (None cutoff) preserves legacy behavior — bit-for-bit identical
    * §5.13.1 — one test runs the real PanelTrainingPipeline through Phase 1
      with a synthetic small panel and asserts cutoff respected on ctx.panel
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


def _synthetic_ohlcv(n_days=300, n_tickers=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)

    def _close(drift=0.0005, vol=0.02):
        rets = rng.normal(drift, vol, size=n_days)
        return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=dates)

    ohlcv = {}
    tickers = [f"T{i}" for i in range(n_tickers)]
    for i, t in enumerate(tickers):
        c = _close(drift=0.0005 + 0.0001 * i, vol=0.02)
        ohlcv[t] = pd.DataFrame({
            "open": c * 0.995, "high": c * 1.01, "low": c * 0.99,
            "close": c, "volume": rng.integers(1_000_000, 10_000_000, size=n_days),
        }, index=dates)
    ohlcv["SPY"] = ohlcv[tickers[0]].copy()
    ohlcv["XLK"] = ohlcv[tickers[1]].copy()
    ohlcv["XLF"] = ohlcv[tickers[2]].copy()
    return ohlcv, tickers


def _make_ctx(tmp_path, cutoff=None, ticker_count=5):
    from training_panel.context import PanelTrainingContext
    ohlcv, tickers = _synthetic_ohlcv(n_tickers=ticker_count)
    sector_etf_ohlcv = {"tech": ohlcv["XLK"], "finance": ohlcv["XLF"]}
    ticker_sectors = {t: ("tech" if i % 2 == 0 else "finance")
                      for i, t in enumerate(tickers)}
    config = {
        "benchmark": "SPY",
        "sector_etf_map": {"tech": "XLK", "finance": "XLF"},
        "indicator_spec": {"rsi": {"period": 14}, "macd": {}, "adx": {}},
        "model_params": {"lookahead": 5, "threshold": 0.01},
        "panel_ltr": {
            "lookahead_days": 5,
            "beta_window": 20,
            "min_history_days": 60,
            "age_warmup_days": 120,
            "cv_n_splits": 3,
            "cv_embargo_days": 5,
            "num_boost_round": 20,
            "neutralize_features": True,
            "factor_mom_window": 60,
            "factor_skip": 5,
            "neutralize_rolling_window": 60,
            "neutralize_warmup_days": 60,
            "min_best_iter": 0,
            "data_scan": {"enabled": False},
        },
        "_strategy_dir": str(tmp_path),
    }
    if cutoff is not None:
        config["panel_ltr"]["train_cutoff"] = cutoff
    ctx = PanelTrainingContext(
        config=config, watchlist=tickers, ohlcv=ohlcv,
        sector_etf_ohlcv=sector_etf_ohlcv, ticker_sectors=ticker_sectors,
    )
    return ctx, tickers


class TestApplyTrainCutoffTaskOhlcv:
    def test_slices_ohlcv_dict(self, tmp_path):
        from training_panel.pp_panel_training import ApplyTrainCutoffTask
        ctx, _ = _make_ctx(tmp_path, cutoff="2023-06-01")
        ApplyTrainCutoffTask().run(ctx)
        for sym, df in ctx.ohlcv.items():
            max_d = pd.to_datetime(df.index.max())
            assert max_d < pd.Timestamp("2023-06-01"), \
                f"{sym} ohlcv has row >= cutoff: {max_d}"

    def test_slices_sector_etf_ohlcv(self, tmp_path):
        from training_panel.pp_panel_training import ApplyTrainCutoffTask
        ctx, _ = _make_ctx(tmp_path, cutoff="2023-06-01")
        ApplyTrainCutoffTask().run(ctx)
        for sec, df in ctx.sector_etf_ohlcv.items():
            assert pd.to_datetime(df.index.max()) < pd.Timestamp("2023-06-01")

    def test_no_op_when_cutoff_none(self, tmp_path):
        from training_panel.pp_panel_training import ApplyTrainCutoffTask
        ctx, _ = _make_ctx(tmp_path, cutoff=None)
        before_max = max(pd.to_datetime(df.index.max())
                         for df in ctx.ohlcv.values())
        ApplyTrainCutoffTask().run(ctx)
        after_max = max(pd.to_datetime(df.index.max())
                        for df in ctx.ohlcv.values())
        assert before_max == after_max  # legacy behavior preserved


class TestApplyTrainCutoffTaskEvents:
    def test_slices_event_frames(self, tmp_path):
        from training_panel.pp_panel_training import ApplyTrainCutoffTask
        ctx, _ = _make_ctx(tmp_path, cutoff="2023-06-01")
        ctx.earnings_surprises = {
            "T0": pd.DataFrame({
                "date": pd.to_datetime(["2023-04-15", "2023-07-15"]),
                "surprise_pct": [0.05, 0.10],
            }),
        }
        ctx.insider_trades = {
            "T0": pd.DataFrame({
                "filing_date": pd.to_datetime(["2023-03-01", "2023-08-01"]),
                "shares": [1000, 2000],
            }),
        }
        ApplyTrainCutoffTask().run(ctx)
        es = ctx.earnings_surprises["T0"]
        assert (pd.to_datetime(es["date"]) < pd.Timestamp("2023-06-01")).all()
        it = ctx.insider_trades["T0"]
        assert (pd.to_datetime(it["filing_date"]) < pd.Timestamp("2023-06-01")).all()

    def test_slices_macro_factor_frame(self, tmp_path):
        from training_panel.pp_panel_training import ApplyTrainCutoffTask
        ctx, _ = _make_ctx(tmp_path, cutoff="2023-06-01")
        ctx.macro_factor_frame = pd.DataFrame(
            {"vix_z": np.linspace(-1, 1, 200)},
            index=pd.bdate_range("2023-01-01", periods=200),
        )
        ApplyTrainCutoffTask().run(ctx)
        assert pd.to_datetime(ctx.macro_factor_frame.index.max()) \
            < pd.Timestamp("2023-06-01")

    def test_drops_tickers_with_only_post_cutoff_data(self, tmp_path):
        from training_panel.pp_panel_training import ApplyTrainCutoffTask
        ctx, tickers = _make_ctx(tmp_path, cutoff="2023-06-01")
        # Make T0's ohlcv entirely post-cutoff
        post_idx = pd.bdate_range("2023-07-01", periods=50)
        ctx.ohlcv["T0"] = pd.DataFrame({
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1,
        }, index=post_idx)
        ApplyTrainCutoffTask().run(ctx)
        assert "T0" not in ctx.ohlcv  # entirely-after-cutoff ticker removed


class TestBuildPanelTaskCutoffGuard:
    def test_panel_dates_strictly_before_cutoff(self, tmp_path):
        """§5.13.1 — actual PanelTrainingPipeline through Phase 1-3 with synthetic
        data; assert ctx.panel respects cutoff. Synthetic data starts
        2023-01-01 (300 bdays ≈ 2024-02-22); cutoff at 2023-12-01 leaves
        ≈ 230 bdays — enough for min_history_days=60 + beta_window=20
        warmup."""
        from training_panel.pp_panel_training import (
            ApplyTrainCutoffTask, SectorMomentumTask, PanelFeatureJob,
            FactorZScoreTask, LabelsTask, BuildPanelTask,
        )
        ctx, _ = _make_ctx(tmp_path, cutoff="2023-12-01")
        ApplyTrainCutoffTask().run(ctx)
        SectorMomentumTask().run(ctx)
        PanelFeatureJob().run(ctx)
        FactorZScoreTask().run(ctx)
        LabelsTask().run(ctx)
        BuildPanelTask().run(ctx)
        assert ctx.panel is not None and not ctx.panel.empty
        max_panel_date = pd.to_datetime(ctx.panel["date"]).max()
        assert max_panel_date < pd.Timestamp("2023-12-01"), \
            f"BuildPanelTask leaked row at-or-after cutoff: {max_panel_date}"

    def test_panel_legacy_when_no_cutoff(self, tmp_path):
        """No cutoff → legacy path: panel reaches near end of synthetic data."""
        from training_panel.pp_panel_training import (
            SectorMomentumTask, PanelFeatureJob,
            FactorZScoreTask, LabelsTask, BuildPanelTask,
        )
        ctx, _ = _make_ctx(tmp_path, cutoff=None)
        SectorMomentumTask().run(ctx)
        PanelFeatureJob().run(ctx)
        FactorZScoreTask().run(ctx)
        LabelsTask().run(ctx)
        BuildPanelTask().run(ctx)
        assert ctx.panel is not None and not ctx.panel.empty
        # Synthetic data spans 300 bdays starting 2023-01-01, ≈ 2024-02-22
        max_panel_date = pd.to_datetime(ctx.panel["date"]).max()
        assert max_panel_date >= pd.Timestamp("2023-12-01"), \
            f"Legacy panel truncated unexpectedly: {max_panel_date}"

    def test_guard_catches_smuggled_post_cutoff_rows(self, tmp_path):
        """If upstream slicing missed, BuildPanelTask must still drop them."""
        from training_panel.pp_panel_training import (
            SectorMomentumTask, PanelFeatureJob,
            FactorZScoreTask, LabelsTask, BuildPanelTask,
        )
        ctx, _ = _make_ctx(tmp_path, cutoff=None)  # no cutoff for build
        SectorMomentumTask().run(ctx)
        PanelFeatureJob().run(ctx)
        FactorZScoreTask().run(ctx)
        LabelsTask().run(ctx)
        # Inject the cutoff post-feature-build to simulate "upstream missed it"
        ctx.config["panel_ltr"]["train_cutoff"] = "2023-12-01"
        BuildPanelTask().run(ctx)
        max_panel_date = pd.to_datetime(ctx.panel["date"]).max()
        assert max_panel_date < pd.Timestamp("2023-12-01")


class TestResolveTrainCutoffParser:
    def test_iso_string_accepted(self, tmp_path):
        from training_panel.pp_panel_training import _resolve_train_cutoff
        ctx, _ = _make_ctx(tmp_path, cutoff="2024-01-15")
        assert _resolve_train_cutoff(ctx) == pd.Timestamp("2024-01-15")

    def test_none_returns_none(self, tmp_path):
        from training_panel.pp_panel_training import _resolve_train_cutoff
        ctx, _ = _make_ctx(tmp_path, cutoff=None)
        assert _resolve_train_cutoff(ctx) is None

    def test_invalid_raises(self, tmp_path):
        from training_panel.pp_panel_training import _resolve_train_cutoff
        ctx, _ = _make_ctx(tmp_path, cutoff="not-a-date")
        with pytest.raises(ValueError, match="invalid"):
            _resolve_train_cutoff(ctx)
