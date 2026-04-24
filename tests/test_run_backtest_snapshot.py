"""run_backtest(snapshot=True) — notebook A/B isolation.

When the user runs N sim variants in a notebook AND starts a retrain
on a separate terminal, the shared strategy_dir (artifacts/, models/,
strategy_config.json) can mutate mid-sim. snapshot=True isolates each
sim call against that race by copying the relevant subdirs to a tmp
location and pointing the adapter at the snapshot for the duration of
the run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


class TestSnapshotFlag:
    def test_snapshot_param_accepted(self):
        """Just confirm the parameter is on the signature — feature is not
        silently dropped. The full-run exercise is covered by the existing
        sim-e2e tests."""
        import inspect
        from sim.runner import run_backtest
        sig = inspect.signature(run_backtest)
        assert "snapshot" in sig.parameters
        # Default flipped to True on 2026-04-24 — all sim calls are
        # isolated against concurrent retrains unless opted out.
        assert sig.parameters["snapshot"].default is True

    def test_snapshot_routes_through_context_manager(self, tmp_path, monkeypatch):
        """snapshot=True must call snapshot_artifacts_ctx at least once."""
        from sim import runner

        calls = {"enter": 0, "exit": 0}

        class _FakeCtx:
            def __init__(self, strategy_dir):
                calls["strategy_dir"] = strategy_dir
            def __enter__(self):
                calls["enter"] += 1
                return tmp_path
            def __exit__(self, *exc):
                calls["exit"] += 1

        def _fake_ctx(path):
            return _FakeCtx(path)

        monkeypatch.setattr(
            "kernel.artifact_snapshot.snapshot_artifacts_ctx", _fake_ctx,
        )

        # Stub out the actual backtest body: second (inner) call with
        # snapshot=False should early-return via a second monkey-patch.
        sentinel = object()
        def _stub_inner(**kwargs):
            assert kwargs["snapshot"] is False
            assert kwargs["strategy_dir"] == tmp_path
            return sentinel

        # We want the outer call (snapshot=True) to enter the context,
        # then the inner recursive call (snapshot=False) to hit our stub.
        original = runner.run_backtest
        call_count = {"n": 0}

        def _wrapped(**kwargs):
            call_count["n"] += 1
            if kwargs["snapshot"] is False:
                return _stub_inner(**kwargs)
            return original(**kwargs)

        monkeypatch.setattr(runner, "run_backtest", _wrapped)

        # Write a minimal strategy_config.json into tmp_path so the snapshot
        # path's re-load branch has something to read.
        (tmp_path / "strategy_config.json").write_text(
            json.dumps({"watchlist": ["NVDA"]}),
        )

        result = runner.run_backtest(
            config={"stub": True},
            strategy_dir=tmp_path,
            ohlcv={}, spy_df=None,  # type: ignore[arg-type]
            sector_etf_map={},
            backtest_start="2025-01-01", backtest_end="2025-02-01",
            snapshot=True,
        )

        assert calls["enter"] == 1
        assert calls["exit"] == 1
        assert result is sentinel
