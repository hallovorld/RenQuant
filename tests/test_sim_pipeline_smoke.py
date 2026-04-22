"""Smoke test — SimAdapter + InferencePipeline can run without crashing.

The legacy hand-written sim loop has been deleted; `run_backtest` now
always drives `InferencePipeline` via `SimAdapter`. These tests confirm
the import surface and adapter construction still work.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _tiny_ohlcv(days: int = 300, seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=days)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, days)))
    return pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": np.ones(days) * 1e6,
    }, index=idx)


class TestSimRunnerSurface:
    def test_adapter_importable(self):
        from adapters.sim import SimAdapter  # noqa: F401
        from sim.runner import run_backtest, run_backtest_via_pipeline  # noqa: F401
        from sim import runner as sim_runner
        # run_backtest is now the single entry point; old helpers are gone.
        assert not hasattr(sim_runner, "_run_backtest_legacy")
        assert not hasattr(sim_runner, "swap_in_panel_scores")
        assert not hasattr(sim_runner, "apply_ngboost_head")

    def test_backcompat_alias_points_to_pipeline_entry(self):
        from sim.runner import run_backtest, run_backtest_via_pipeline
        assert run_backtest is run_backtest_via_pipeline

    def test_run_backtest_errors_without_dates(self, tmp_path):
        from sim.runner import run_backtest
        cfg = {"watchlist": []}
        with pytest.raises(ValueError, match="backtest_start"):
            run_backtest(
                config=cfg, strategy_dir=tmp_path,
                ohlcv={}, spy_df=pd.DataFrame(), sector_etf_map={},
            )


class TestSimAdapterInit:
    """SimAdapter can be constructed from a minimal repo shape without crashing."""

    def test_init_with_real_strategy_dir(self):
        from adapters.sim import SimAdapter
        ohlcv = {"SPY": _tiny_ohlcv(seed=1)}
        cfg = {
            "watchlist": [],           # no models → no policy loads
            "sector_etf_map": {},
            "tax": {},
            "regime": {},
        }
        adapter = SimAdapter(
            config=cfg,
            strategy_dir=_STRATEGY_DIR,
            ohlcv=ohlcv,
            spy_df=ohlcv["SPY"],
            sector_etf_map={},
            initial_cash=100_000,
        )
        assert adapter._cash == 100_000.0  # noqa: SLF001
        assert adapter._hwm  == 100_000.0  # noqa: SLF001
        assert adapter._holdings == {}     # noqa: SLF001

    def test_make_context_returns_inference_context(self):
        from adapters.sim import SimAdapter
        from kernel.pipeline.context import InferenceContext

        ohlcv = {"SPY": _tiny_ohlcv(seed=2)}
        cfg = {"watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {}}
        adapter = SimAdapter(
            config=cfg, strategy_dir=_STRATEGY_DIR,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
            initial_cash=100_000,
        )
        today = ohlcv["SPY"].index[50]
        ctx = adapter.make_context(today)
        assert isinstance(ctx, InferenceContext)
        assert ctx.today == today.date()
        assert ctx.cash == 100_000.0
        assert ctx.portfolio_value == 100_000.0  # no positions yet
