"""Regression guards for live.runner --dry-run (GOAL-5 AC5 preflight probe).

readonly-alpaca alone does not make a cycle side-effect-free: adapter.commit()
still writes live_state.json / run-bundle records and places orders through
whatever broker it is given. dry_run must skip commit() entirely, on every
broker, and must attest that it did so via a log line the dawn-preflight
analyzer can require.
"""
from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from live import runner


class _Broker:
    broker_name = "paper"


class _ReadonlyBroker:
    broker_name = "alpaca_shadow"


def _install_fake_pipeline(monkeypatch, *, seen: dict) -> None:
    kernel_pkg = types.ModuleType("kernel")
    preflight_mod = types.ModuleType("kernel.preflight")
    pipeline_mod = types.ModuleType("kernel.pipeline")
    adapters_pkg = types.ModuleType("adapters")
    runner_adapter_mod = types.ModuleType("adapters.runner")

    class PreflightFailed(Exception):
        pass

    def run_preflight(*_args, **_kwargs):
        seen["preflight_ran"] = True

    class _Pipeline:
        def run(self, ctx):
            seen["pipeline_run"] = True
            ctx.orders = [{"ticker": "AAPL", "action": "BUY"}]
            ctx.exits = []

    class _RunnerAdapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def make_context(self):
            return SimpleNamespace(
                orders=[],
                exits=[],
                exits_failed=[],
                holdings={},
                counters={},
                candidates=[],
                ranked=[],
                portfolio_value=0.0,
                confidence=None,
                regime="BULL_CALM",
            )

        def commit(self, ctx):
            # If dry_run is wired correctly this must never be called.
            seen["commit"] = True
            seen.setdefault("orders_placed_on_commit", []).append("AAPL")

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


def _install_ntfy_spy(monkeypatch, seen: dict):
    def fake_notify_dry_run_probe(label, run_mode):
        seen["dry_run_probe_notified"] = {"label": label, "run_mode": run_mode}

    def fake_notify_decision(*_args, **_kwargs):
        seen["real_decision_notified"] = True

    monkeypatch.setattr(runner, "_notify_dry_run_probe", fake_notify_dry_run_probe)
    monkeypatch.setattr(runner, "_notify_decision", fake_notify_decision)


@pytest.mark.parametrize("broker", [_Broker(), _ReadonlyBroker()])
def test_dry_run_never_calls_commit_on_any_broker(monkeypatch, broker):
    """readonly-alpaca is not sufficient by itself (codex #565 finding):
    dry_run must skip commit() regardless of which broker is passed."""
    seen: dict = {}
    _install_fake_pipeline(monkeypatch, seen=seen)
    _install_ntfy_spy(monkeypatch, seen)

    runner._run_once_multi_pipeline(
        {"live": {"preflight": {"enabled": True}}},
        models={},
        broker=broker,
        strategy_dir=Path("backtesting/renquant_104"),
        sell_only=False,
        dry_run=True,
    )

    assert seen["pipeline_run"] is True
    assert "commit" not in seen
    assert "orders_placed_on_commit" not in seen
    assert "real_decision_notified" not in seen


def test_dry_run_attests_via_log_and_dedicated_probe_notifier(monkeypatch, caplog):
    seen: dict = {}
    _install_fake_pipeline(monkeypatch, seen=seen)
    _install_ntfy_spy(monkeypatch, seen)

    with caplog.at_level("INFO", logger="live.runner"):
        runner._run_once_multi_pipeline(
            {"live": {"preflight": {"enabled": True}}},
            models={},
            broker=_ReadonlyBroker(),
            strategy_dir=Path("backtesting/renquant_104"),
            sell_only=False,
            dry_run=True,
        )

    assert "DRY_RUN_ATTESTATION" in caplog.text
    assert "commit=skipped" in caplog.text
    assert seen["dry_run_probe_notified"]["run_mode"].startswith("full")


def test_dry_run_probe_notifier_never_reads_commit_only_fields(monkeypatch):
    """_notify_dry_run_probe must not depend on ctx.orders_placed /
    ctx.exits_placed — those are only populated inside adapter.commit(),
    which dry-run mode never calls. A regression that reads them would
    silently render every dry-run cycle as a false "0 orders" decision."""
    posted: dict = {}

    def fake_post(url, *, title, body, priority, taxonomy, key, cooldown_seconds=0, force=False):
        posted.update(url=url, title=title, body=body, taxonomy=taxonomy)
        return True

    monkeypatch.setattr(runner, "_post_ntfy_with_retries", fake_post)

    runner._notify_dry_run_probe("[READONLY]RENQUANT-104", "full")

    assert posted["taxonomy"] == "DRY_RUN_PREFLIGHT"
    assert "[DRY-RUN PREFLIGHT]" in posted["title"]
    assert "No orders placed" in posted["body"]


def test_non_dry_run_cycle_still_commits_and_notifies_normally(monkeypatch):
    """Behavior-invariance: dry_run defaults False and must not change any
    existing commit/notify call for real cycles (fix-wave-protects-production)."""
    seen: dict = {}
    _install_fake_pipeline(monkeypatch, seen=seen)
    _install_ntfy_spy(monkeypatch, seen)

    runner._run_once_multi_pipeline(
        {"live": {"preflight": {"enabled": True}}},
        models={},
        broker=_Broker(),
        strategy_dir=Path("backtesting/renquant_104"),
        sell_only=False,
    )

    assert seen["commit"] is True
    assert seen["real_decision_notified"] is True
    assert "dry_run_probe_notified" not in seen
