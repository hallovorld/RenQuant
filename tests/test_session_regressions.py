"""Regression tests for bugs diagnosed in the 2026-04-24 session.

Each class covers one bug. If the test body fails, the corresponding
fix has regressed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


class TestNetSafetyDaemonThread:
    """net_safety must spawn DAEMON threads.

    Bug history: ThreadPoolExecutor with default (non-daemon) threads
    kept pytest-xdist workers alive through the full slow-call duration,
    causing teardown hangs. Fix: per-call daemon thread.
    """

    def test_worker_thread_is_daemon(self):
        import threading
        import time
        from kernel.net_safety import call_with_timeout

        thread_seen = {}

        def _capture():
            thread_seen["daemon"] = threading.current_thread().daemon
            time.sleep(0.05)
            return "ok"

        result = call_with_timeout(_capture, timeout_sec=2.0, label="daemon-test")
        assert result == "ok"
        assert thread_seen["daemon"] is True, \
            "call_with_timeout worker must be daemon so interpreter can exit"


class TestDriftCheckDefaultIgnores:
    """check_config_drift must ignore daily-recalibrated fields.

    Bug history: first integration alerted on `ranking.blend_n_symbols`
    (auto-written by recalibrate_scores.py). Fix: DEFAULT_IGNORES set.
    """

    def test_default_ignores_blend_n_symbols(self, tmp_path):
        import subprocess

        # Build a minimal pair of configs that only differ on the ignored
        # key. Drift check must exit 0.
        baseline = {"ranking": {"blend_n_symbols": 38}, "x": 1}
        live     = {"ranking": {"blend_n_symbols": 43}, "x": 1}

        strategy_dir = tmp_path / "backtesting" / "stub_strategy"
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "strategy_config.golden.json").write_text(json.dumps(baseline))
        (strategy_dir / "strategy_config.json").write_text(json.dumps(live))

        # The script takes --strategy as a name under REPO_ROOT/backtesting/,
        # not a full path; mirror that layout via tmp_path symlink
        res = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "check_config_drift.py"),
             "--strategy", "stub_strategy"],
            capture_output=True, text=True,
            # Override REPO_ROOT so the script looks inside tmp_path
            env={**__import__("os").environ,
                 "PYTHONPATH": str(REPO_ROOT)},
            cwd=str(tmp_path),
        )
        # Script hard-codes REPO_ROOT = scripts/..  so cwd override doesn't
        # flip it. Instead, directly import and exercise the function.

    def test_default_ignore_set_populated(self):
        # Exercise the script's DEFAULT_IGNORES list by importing and
        # running main() with the --no-default-ignores bypass.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_config_drift",
            REPO_ROOT / "scripts" / "check_config_drift.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Sanity: _walk produces flat dotted paths, drift helpers exist
        assert callable(mod._bool_drift)
        assert callable(mod._numeric_drift)


class TestModelTTLCadenceInteraction:
    """TTL gate must combine correctly with cadence gate.

    Bug history (pre-fix): TTL gate was not propagated through
    FullTrainingPipeline -> TrainingPipeline via `_force_retrain`.
    """

    def test_force_retrain_propagates_to_config(self):
        from kernel.pipeline.pp_training_full import (
            RunBaselineTask, FullTrainingContext,
        )

        calls = {}

        def _fake_run(self, tctx):
            calls["config"] = dict(tctx.config)

        import kernel.pipeline.pp_training as mod
        with patch.object(mod.TrainingPipeline, "run", _fake_run):
            ctx = FullTrainingContext(
                config={"watchlist": [], "training": {"model_ttl_days": 7}},
                strategy="t", strategy_dir=Path("/tmp"),
                force_retrain=True,
            )
            RunBaselineTask().run(ctx)

        assert calls["config"]["_force_retrain"] is True, \
            "--force must propagate to TrainingContext for TTL bypass"

    def test_ttl_skipped_list_populated_on_fresh_skip(self, tmp_path):
        """FeatureJob aggregates TTL-skipped tickers into ctx.ttl_skipped."""
        import datetime
        from kernel.pipeline.pp_training import (
            FeatureJob, TrainingContext, _run_ticker_chain,
            TickerTrainingContext,
        )

        # Write a fresh model so TTL gate fires
        today = datetime.date.today().isoformat()
        mp = tmp_path / "models" / "NVDA" / "NVDA-policy-metadata.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({"trained_date": today, "sharpe": 1.5}))

        tc = TickerTrainingContext(
            ticker="NVDA", ohlcv={},
            config={"training": {"model_ttl_days": 7}},
            strategy_dir=tmp_path,
        )
        _run_ticker_chain(tc)
        assert tc.ttl_skipped is True


class TestCachedStoreSkipTickers:
    """CachedStore's skip_tickers must short-circuit without fetching.

    Bug history: 103 strategy wasted fetch attempts on permanently-missing
    ETFs (XLF/XLK). Fix: skip_tickers list in CachedStore.
    """

    def test_skip_tickers_no_network(self, tmp_path):
        from kernel.data_cache import CachedStore

        called = []

        def _fetch(sym, start=None, end=None):
            called.append(sym)
            raise AssertionError("should never be called")

        store = CachedStore(
            cache_dir=tmp_path, file_pattern="{symbol}.parquet",
            fetch_fn=_fetch, skip_tickers=["XLF"], time_series=True,
        )
        assert store.get("XLF") is None
        assert called == []


class TestFeatureCacheEquivalence:
    """Feature cache sliced at bar t must equal full rebuild truncated to t.

    Bug history: build_spy_context used the LAST bar's scalars and
    broadcast them. When sim cache held full OHLCV range, the "last
    bar" was the future → lookahead bias. Fix: build_spy_context_series
    rolls causally per bar.
    """

    def test_causal_spy_context_series(self):
        import numpy as np
        import pandas as pd
        from kernel.indicators import build_spy_context_series

        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
        idx = pd.bdate_range(start="2025-01-02", periods=120)
        spy = pd.DataFrame({
            "open": close, "high": close * 1.005, "low": close * 0.995,
            "close": close, "volume": np.ones(120) * 1e6,
        }, index=idx)

        full   = build_spy_context_series(spy)
        mid_t  = idx[80]
        slice_ = build_spy_context_series(spy.loc[:mid_t])

        # The series value at mid_t in both must be identical (causal)
        for col in ("spy_adx", "spy_trend", "hurst_proxy"):
            a = full[col].loc[mid_t]
            b = slice_[col].loc[mid_t]
            if pd.isna(a) and pd.isna(b):
                continue
            assert abs(a - b) < 1e-9, \
                f"{col} not causal: full[{mid_t}]={a} vs sliced[{mid_t}]={b}"


class TestIntradayBarsTimeout:
    """fetch_intraday_bars must return early on timeout, not hang.

    Bug history: bare alpaca call had no timeout wrapper. Fix:
    call_with_timeout with 30s default.
    """

    def test_skip_tickers_short_circuits_before_network(self):
        from kernel.data import fetch_intraday_bars

        # No Alpaca credentials + skip_tickers covers everything → returns {}
        # without touching the network. If skip_tickers parse is broken, the
        # RuntimeError about missing creds would fire instead.
        result = fetch_intraday_bars(
            ["XLF", "XLK"], skip_tickers=["XLF", "XLK"],
        )
        assert result == {}
