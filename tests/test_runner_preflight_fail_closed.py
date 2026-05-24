"""Regression guards for live.runner preflight fail-closed behavior."""
from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from live import runner


class _Broker:
    broker_name = "paper"


def _install_fake_pipeline(monkeypatch, *, preflight_exc: Exception, seen: dict) -> None:
    kernel_pkg = types.ModuleType("kernel")
    preflight_mod = types.ModuleType("kernel.preflight")
    pipeline_mod = types.ModuleType("kernel.pipeline")
    adapters_pkg = types.ModuleType("adapters")
    runner_adapter_mod = types.ModuleType("adapters.runner")

    class PreflightFailed(Exception):
        pass

    def run_preflight(*_args, **_kwargs):
        raise preflight_exc

    class _Pipeline:
        def run(self, ctx):
            seen["pipeline_run"] = True
            ctx.orders_placed = []
            ctx.exits_placed = []
            ctx.exits_failed = []

    class _RunnerAdapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def make_context(self):
            return SimpleNamespace(
                orders_placed=[],
                exits=[],
                exits_failed=[],
                holdings={},
                counters={},
                candidates=[],
                ranked=[],
                portfolio_value=0.0,
                confidence=None,
                regime=None,
            )

        def commit(self, ctx):
            seen["commit"] = True

    preflight_mod.run_preflight = run_preflight
    preflight_mod.PreflightFailed = PreflightFailed
    pipeline_mod.InferencePipeline = _Pipeline
    pipeline_mod.SellOnlyPipeline = _Pipeline
    runner_adapter_mod.RunnerAdapter = _RunnerAdapter

    monkeypatch.setitem(__import__("sys").modules, "kernel", kernel_pkg)
    monkeypatch.setitem(__import__("sys").modules, "kernel.preflight", preflight_mod)
    monkeypatch.setitem(__import__("sys").modules, "kernel.pipeline", pipeline_mod)
    monkeypatch.setitem(__import__("sys").modules, "adapters", adapters_pkg)
    monkeypatch.setitem(__import__("sys").modules, "adapters.runner", runner_adapter_mod)
    monkeypatch.setattr(runner, "_load_kernel", lambda _strategy_dir: True)
    monkeypatch.setattr(runner, "_notify_decision", lambda *_args, **_kwargs: None)


def test_unexpected_preflight_exception_aborts_full_run(monkeypatch):
    seen: dict = {}
    _install_fake_pipeline(
        monkeypatch,
        preflight_exc=RuntimeError("broken preflight"),
        seen=seen,
    )

    with pytest.raises(SystemExit) as exc:
        runner._run_once_multi_pipeline(
            {"live": {"preflight": {"enabled": True}}},
            models={},
            broker=_Broker(),
            strategy_dir=Path("backtesting/renquant_104"),
            sell_only=False,
        )

    assert exc.value.code == 2
    assert seen == {}


def test_unexpected_preflight_exception_allows_sell_only_risk_exit(monkeypatch):
    seen: dict = {}
    _install_fake_pipeline(
        monkeypatch,
        preflight_exc=RuntimeError("broken preflight"),
        seen=seen,
    )

    runner._run_once_multi_pipeline(
        {"live": {"preflight": {"enabled": True}}},
        models={},
        broker=_Broker(),
        strategy_dir=Path("backtesting/renquant_104"),
        sell_only=True,
    )

    assert seen == {"pipeline_run": True, "commit": True}
