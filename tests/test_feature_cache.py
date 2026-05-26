"""Feature cache — SimAdapter pre-builds per-ticker full-range feature frames.

Critical: must produce IDENTICAL feature output per bar as the
non-cached path. If cached and live paths diverge, sim and live would
make different decisions — hard bug to catch.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _synthetic_ohlcv(n: int = 200):
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range(start="2025-01-02", periods=n)
    return pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": np.ones(n) * 1e6,
    }, index=idx)


class TestEquivalence:
    """Cached slice vs uncached rebuild — correctness of the optimization.

    FIXED 2026-04-24: `build_spy_context_series` replaces the scalar-
    broadcast `build_spy_context`. SPY regime features now compute
    strictly-causally per bar, so `build_feature_frame(full_history)`
    sliced at bar t equals `build_feature_frame(truncated_to_t)`.

    Unlocks `sim.feature_cache_enabled` for 5-8x sim speedup.
    """

    def test_last_row_identical(self):
        """For any bar t, cache.loc[:t].iloc[-1] must equal build_feature_frame
        called on OHLCV truncated to t."""
        from kernel.indicators import build_feature_frame

        spy = _synthetic_ohlcv()
        stock = _synthetic_ohlcv()

        spec = {}
        vol_win = 20

        # Full-range (cache approach)
        full_cache = build_feature_frame(stock, spy, spec, vol_win)
        assert full_cache is not None and not full_cache.empty

        # Pick a mid-date
        mid_date = stock.index[100]

        # Slice cache
        cached_row = full_cache.loc[:mid_date].iloc[-1]

        # Rebuild from truncated OHLCV
        stock_truncated = stock.loc[:mid_date]
        spy_truncated = spy.loc[:mid_date]
        uncached = build_feature_frame(stock_truncated, spy_truncated, spec, vol_win)
        uncached_row = uncached.iloc[-1]

        # All columns must match (within floating-point tolerance)
        import pandas as pd
        pd.testing.assert_series_equal(
            cached_row, uncached_row, rtol=1e-9, atol=1e-12,
            check_names=False,
        )

    def test_non_spy_features_match(self):
        """The NON-SPY-derived features (rsi/macd/cci/bbp/williams_r/
        obv_slope) must match exactly between cached and uncached."""
        from kernel.indicators import build_feature_frame
        import pandas as pd

        spy = _synthetic_ohlcv()
        stock = _synthetic_ohlcv()
        full_cache = build_feature_frame(stock, spy, {}, 20)
        mid_date = stock.index[100]
        cached = full_cache.loc[:mid_date].iloc[-1]
        uncached = build_feature_frame(stock.loc[:mid_date],
                                        spy.loc[:mid_date], {}, 20).iloc[-1]
        # Only these features are trusted to be strictly causal today:
        causal_cols = ["rsi", "macd_hist", "cci", "bbp", "williams_r",
                       "obv_slope", "adx"]
        for col in causal_cols:
            if col in cached.index and col in uncached.index:
                assert abs(cached[col] - uncached[col]) < 1e-9, \
                    f"{col} diverges: cached={cached[col]} uncached={uncached[col]}"

    def test_alpha158_full_frame_matches_single_bar_path(self):
        """Alpha158 sim cache must be byte-equivalent to live single-bar inference."""
        from kernel.panel_pipeline.alpha158_features import (
            alpha158_feature_names,
            compute_alpha158_at,
            compute_alpha158_frame,
        )
        import pandas as pd

        stock = _synthetic_ohlcv(180)
        cache = compute_alpha158_frame(stock)
        assert not cache.empty

        for idx in (80, 120, 160):
            day = stock.index[idx]
            cached = cache.loc[:day].iloc[-1].reindex(alpha158_feature_names())
            uncached = pd.Series(compute_alpha158_at(stock.loc[:day])).reindex(
                alpha158_feature_names()
            )
            pd.testing.assert_series_equal(
                cached,
                uncached,
                rtol=1e-9,
                atol=1e-12,
                check_names=False,
                check_dtype=False,
            )

    def test_precomputed_feature_assembly_matches_public_builder(self):
        """Sim cache assembly must match the public live-style feature builder."""
        from kernel.indicators import (
            assemble_feature_frame_from_indicators,
            build_feature_frame,
            build_spy_context_series,
            compute_all,
        )
        import pandas as pd

        spy = _synthetic_ohlcv()
        stock = _synthetic_ohlcv()
        spec = {}
        vol_win = 20

        public_frame = build_feature_frame(stock, spy, spec, vol_win)
        assert public_frame is not None and not public_frame.empty

        stock_ind = compute_all(stock, spec)
        spy_ind = compute_all(spy, spec)
        spy_context = build_spy_context_series(spy, vol_window=vol_win)
        cached_frame = assemble_feature_frame_from_indicators(
            stock_ind, spy_ind, spy_context,
        )

        pd.testing.assert_frame_equal(
            cached_frame,
            public_frame,
            rtol=1e-9,
            atol=1e-12,
            check_dtype=False,
        )


class TestContextPlumbing:
    def test_context_has_feature_cache_field(self):
        from kernel.pipeline.context import InferenceContext
        ctx = InferenceContext(config={}, today=datetime.date(2026, 4, 24))
        assert ctx.feature_cache == {}  # default empty dict

    def test_ticker_context_has_cache_field(self):
        from kernel.pipeline.context import TickerInferenceContext
        tc = TickerInferenceContext(
            ticker="NVDA", ohlcv={}, model=None, config={},
            today=datetime.date(2026, 4, 24), regime="BULL_CALM",
            regime_params={}, exit_params={},
        )
        assert tc.feature_cache_frame is None


class TestBuildFeaturesTaskUsesCache:
    def test_cache_slice_path(self):
        """If feature_cache_frame set on tc, BuildFeaturesTask slices it
        and skips the expensive build_feature_frame rebuild."""
        from kernel.pipeline.task_candidates import BuildFeaturesTask
        import pandas as pd

        # Fake a 3-day feature frame
        idx = pd.bdate_range("2026-04-22", periods=3)
        frame = pd.DataFrame({"rsi": [50.0, 55.0, 60.0]}, index=idx)

        tc = SimpleNamespace(
            ticker="NVDA",
            today=datetime.date(2026, 4, 23),   # mid date
            feature_cache_frame=frame,
            ohlcv={},
            model=SimpleNamespace(),   # non-None to pass guard
            config={},
        )
        BuildFeaturesTask().run(tc)

        # Expect slice up to today only
        assert tc.features is not None
        assert len(tc.features) == 2   # 2 rows up to day 2 of 3

    def test_no_cache_falls_back_to_build(self):
        """No cached frame → task calls build_feature_frame. If that fails
        (OHLCV missing), returns False."""
        from kernel.pipeline.task_candidates import BuildFeaturesTask

        tc = SimpleNamespace(
            ticker="NVDA",
            today=datetime.date(2026, 4, 24),
            feature_cache_frame=None,
            ohlcv={},                  # empty → build will fail
            model=SimpleNamespace(),
            config={},
        )
        result = BuildFeaturesTask().run(tc)
        assert result is False


class TestSimAdapterCacheBounds:
    def test_shared_runtime_feature_cache_reuses_spy_context_once(self, monkeypatch):
        """The shared sim/live cache builder should preserve the sim invariant."""
        from adapters.panel_runtime import build_runtime_feature_cache
        import pandas as pd

        stock = _synthetic_ohlcv(140)
        spy = _synthetic_ohlcv(140)
        context_calls = 0

        def fake_compute_all(df, *_args, **_kwargs):
            return pd.DataFrame({"close": range(len(df))}, index=df.index)

        def fake_context(df, *_args, **_kwargs):
            nonlocal context_calls
            context_calls += 1
            return pd.DataFrame({"spy_trend": [1.0] * len(df)}, index=df.index)

        def fake_assemble(stock_ind, *_args, **_kwargs):
            return pd.DataFrame({"rsi": [50.0]}, index=[stock_ind.index.max()])

        monkeypatch.setattr("kernel.indicators.compute_all", fake_compute_all)
        monkeypatch.setattr("kernel.indicators.build_spy_context_series", fake_context)
        monkeypatch.setattr(
            "kernel.indicators.assemble_feature_frame_from_indicators",
            fake_assemble,
        )

        cache = build_runtime_feature_cache(
            config={},
            ohlcv={"SPY": spy, "AAPL": stock, "MSFT": stock},
        )

        assert context_calls == 1
        assert sorted(cache) == ["AAPL", "MSFT"]

    def test_runner_adapter_attaches_run_local_feature_cache(self):
        """Live/shadow must not drift from sim by rebuilding features per ticker."""
        from adapters.runner import RunnerAdapter

        broker = MagicMock()
        broker.get_account_value.return_value = 100_000.0
        broker.get_cash.return_value = 100_000.0
        broker.get_all_positions.return_value = []
        config = {
            "watchlist": ["AAA"],
            "benchmark": "SPY",
            "ranking": {"panel_scoring": {"enabled": False}},
        }
        sentinel_cache = {"AAA": SimpleNamespace(name="cached_features")}

        adapter = RunnerAdapter(
            config,
            models={},
            broker=broker,
            strategy_dir=_STRATEGY_DIR,
            sell_only=False,
        )

        with patch("kernel.data.fetch_ohlcv", return_value=_synthetic_ohlcv()), \
             patch("adapters.runner.build_runtime_feature_cache",
                   return_value=sentinel_cache) as cache_mock:
            ctx = adapter.make_context()

        assert cache_mock.called
        assert ctx.feature_cache is sentinel_cache

    def test_feature_caches_clip_source_ohlcv_to_backtest_end(self, monkeypatch):
        """Historical sim caches should not compute rows beyond the sim end."""
        from adapters.sim import SimAdapter
        import pandas as pd

        end = pd.Timestamp("2025-04-30")
        stock = _synthetic_ohlcv(140)
        spy = _synthetic_ohlcv(140)
        assert stock.index.max() > end

        seen_compute: list[pd.Timestamp] = []
        seen_context: list[pd.Timestamp] = []

        def fake_compute_all(df, *_args, **_kwargs):
            seen_compute.append(df.index.max())
            return pd.DataFrame({"close": range(len(df))}, index=df.index)

        def fake_context(df, *_args, **_kwargs):
            seen_context.append(df.index.max())
            return pd.DataFrame({"spy_trend": [1.0] * len(df)}, index=df.index)

        def fake_assemble(stock_ind, *_args, **_kwargs):
            return pd.DataFrame({"rsi": [50.0]}, index=[stock_ind.index.max()])

        monkeypatch.setattr("kernel.indicators.compute_all", fake_compute_all)
        monkeypatch.setattr("kernel.indicators.build_spy_context_series", fake_context)
        monkeypatch.setattr(
            "kernel.indicators.assemble_feature_frame_from_indicators",
            fake_assemble,
        )
        adapter = SimAdapter.__new__(SimAdapter)
        adapter._ohlcv = {"SPY": spy, "NVDA": stock}
        adapter._config = {}
        adapter._backtest_end = end
        adapter._feature_cache = {}

        adapter._build_feature_cache()

        assert seen_context == [end]
        assert seen_compute == [end, end]  # SPY once, stock once
        assert adapter._feature_cache["NVDA"].index.max() == end

    def test_feature_cache_reuses_spy_context_once(self, monkeypatch):
        """SPY regime context is shared across tickers during cache prebuild."""
        from adapters.sim import SimAdapter
        import pandas as pd

        stock = _synthetic_ohlcv(140)
        spy = _synthetic_ohlcv(140)
        context_calls = 0

        def fake_compute_all(df, *_args, **_kwargs):
            return pd.DataFrame({"close": range(len(df))}, index=df.index)

        def fake_context(df, *_args, **_kwargs):
            nonlocal context_calls
            context_calls += 1
            return pd.DataFrame({"spy_trend": [1.0] * len(df)}, index=df.index)

        def fake_assemble(stock_ind, *_args, **_kwargs):
            return pd.DataFrame({"rsi": [50.0]}, index=[stock_ind.index.max()])

        monkeypatch.setattr("kernel.indicators.compute_all", fake_compute_all)
        monkeypatch.setattr("kernel.indicators.build_spy_context_series", fake_context)
        monkeypatch.setattr(
            "kernel.indicators.assemble_feature_frame_from_indicators",
            fake_assemble,
        )
        adapter = SimAdapter.__new__(SimAdapter)
        adapter._ohlcv = {"SPY": spy, "AAPL": stock, "MSFT": stock}
        adapter._config = {}
        adapter._backtest_end = None
        adapter._feature_cache = {}

        adapter._build_feature_cache()

        assert context_calls == 1
        assert sorted(adapter._feature_cache) == ["AAPL", "MSFT"]

    def test_alpha158_cache_clips_source_ohlcv_to_backtest_end(self, monkeypatch):
        """The alpha158 cache should obey the same sim-window bound."""
        from adapters.sim import SimAdapter
        import pandas as pd

        end = pd.Timestamp("2025-04-30")
        stock = _synthetic_ohlcv(140)
        assert stock.index.max() > end

        seen: list[pd.Timestamp] = []

        def fake_alpha158_frame(df):
            seen.append(df.index.max())
            return pd.DataFrame({"KMID": [0.0]}, index=[df.index.max()])

        monkeypatch.setattr(
            "kernel.panel_pipeline.alpha158_features.compute_alpha158_frame",
            fake_alpha158_frame,
        )
        adapter = SimAdapter.__new__(SimAdapter)
        adapter._ohlcv = {"SPY": stock, "NVDA": stock}
        adapter._backtest_end = end
        adapter._alpha158_feature_cache = {}
        adapter._panel_scorer = SimpleNamespace(metadata={"kind": "panel_ltr_xgboost"})
        adapter._walkforward_loader = None

        adapter._build_alpha158_feature_cache()

        assert seen == [end]
        assert adapter._alpha158_feature_cache["NVDA"].index.max() == end


class TestAlpha158ScoringTaskUsesCache:
    def test_apply_scores_uses_alpha158_cache_before_single_bar_fallback(self, monkeypatch):
        """The alpha158 scorer hot path should read the sim cache when available."""
        from kernel.panel_pipeline.alpha158_features import (
            alpha158_feature_names,
            compute_alpha158_frame,
        )
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask
        import pandas as pd

        names = alpha158_feature_names()
        stock = _synthetic_ohlcv(120)
        day = stock.index[100]
        cache = {"NVDA": compute_alpha158_frame(stock)}

        def fail_uncached(*_args, **_kwargs):
            raise AssertionError("uncached alpha158 path should not run on cache hit")

        monkeypatch.setattr(
            "kernel.panel_pipeline.alpha158_features.compute_alpha158_at",
            fail_uncached,
        )

        class Scorer:
            metadata = {"kind": "panel_ltr_xgboost"}
            feature_cols = names

            def score(self, X):
                assert list(X.columns) == names
                return pd.Series({"NVDA": 0.73})

        cand = SimpleNamespace(ticker="NVDA", rank_score=0.0, panel_score=None)
        ctx = SimpleNamespace(
            _panel_scorer=Scorer(),
            _panel_matrix=pd.DataFrame({"__alpha158_target__": [1.0]}, index=["NVDA"]),
            _alpha158_feature_cache=cache,
            ohlcv={"NVDA": stock.loc[:day]},
            today=day.date(),
            candidates=[cand],
            holdings={},
        )

        ApplyScoresTask().run(ctx)

        assert cand.rank_score == pytest.approx(0.73)
        assert cand.panel_score == pytest.approx(0.73)
