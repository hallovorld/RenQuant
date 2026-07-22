"""Integration guards for live.runner --preflight (dry-run) mode.

GOAL-5 AC5 / PR #565 codex CR: the dawn preflight must be a TRUE read-only
probe. `--preflight` drives the funnel to a decision line but must guarantee
zero of: DB/state persistence, order placement, promotion, notification — and
must emit a machine-readable `preflight_attestation:` line the shell guard
verifies. These tests assert that contract against isolated state paths and
that the attestation flips (fails closed) when a side-effect boundary is hit.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from live import runner


# ── shell-guard contract mirror ────────────────────────────────────────────────
# The dawn shell guard treats an attestation as PASSING only when it reaches a
# decision with no persistence/notification. Keep this in lockstep with
# ops/renquant104/dawn_funnel_preflight.sh.
def _attestation_is_positive(payload: dict) -> bool:
    return (
        payload.get("persisted") is False
        and payload.get("notified") is False
        and payload.get("reached_decision") is True
    )


def _parse_attestation(captured_stdout: str) -> dict:
    for line in captured_stdout.splitlines():
        if line.startswith("preflight_attestation:"):
            return json.loads(line.split("preflight_attestation:", 1)[1].strip())
    raise AssertionError(
        f"no preflight_attestation line in output:\n{captured_stdout}"
    )


class _Broker:
    broker_name = "paper"


class _ShadowBroker:
    broker_name = "alpaca_shadow"


def _install_fake_pipeline(
    monkeypatch,
    *,
    preflight_exc: Exception | None = None,
    preflight_fail_message: str | None = None,
    pipeline_side_effect=None,
    commit_sentinel: Path | None = None,
    seen: dict,
) -> None:
    """Inject fake kernel.preflight / kernel.pipeline / adapters.runner.

    Mirrors tests/test_runner_preflight_fail_closed.py so the runner is driven
    without the heavy real funnel. The fake RunnerAdapter.commit writes a
    sentinel file if configured — the dry-run path must NEVER call it.
    """
    kernel_pkg = types.ModuleType("kernel")
    preflight_mod = types.ModuleType("kernel.preflight")
    pipeline_mod = types.ModuleType("kernel.pipeline")
    adapters_pkg = types.ModuleType("adapters")
    runner_adapter_mod = types.ModuleType("adapters.runner")

    class PreflightFailed(Exception):
        pass

    def run_preflight(*_args, **_kwargs):
        seen["preflight_kwargs"] = dict(_kwargs)
        # Raise the LOCAL PreflightFailed so the type matches what the runner
        # imports from this same fake module (avoids a stale-class mismatch).
        if preflight_fail_message is not None:
            raise PreflightFailed(preflight_fail_message)
        if preflight_exc is not None:
            raise preflight_exc

    class _Pipeline:
        def run(self, ctx):
            seen["pipeline_run"] = True
            ctx.orders_placed = []
            ctx.exits_placed = []
            ctx.exits_failed = []
            if pipeline_side_effect is not None:
                pipeline_side_effect(ctx)

    class _RunnerAdapter:
        def __init__(self, *_args, **kwargs):
            seen["adapter_kwargs"] = dict(kwargs)

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
            if commit_sentinel is not None:
                commit_sentinel.write_text("committed")

    preflight_mod.run_preflight = run_preflight
    preflight_mod.PreflightFailed = PreflightFailed
    pipeline_mod.InferencePipeline = _Pipeline
    pipeline_mod.SellOnlyPipeline = _Pipeline
    runner_adapter_mod.RunnerAdapter = _RunnerAdapter

    monkeypatch.setitem(sys.modules, "kernel", kernel_pkg)
    monkeypatch.setitem(sys.modules, "kernel.preflight", preflight_mod)
    monkeypatch.setitem(sys.modules, "kernel.pipeline", pipeline_mod)
    monkeypatch.setitem(sys.modules, "adapters", adapters_pkg)
    monkeypatch.setitem(sys.modules, "adapters.runner", runner_adapter_mod)
    monkeypatch.setattr(runner, "_load_kernel", lambda _strategy_dir: True)


# ── Positive: clean probe reaches a decision with no side effects ──────────────

def test_dry_run_reaches_decision_no_persist_no_notify(monkeypatch, tmp_path, capsys):
    seen: dict = {}
    sentinel = tmp_path / "runs_and_state_sentinel.txt"
    _install_fake_pipeline(monkeypatch, commit_sentinel=sentinel, seen=seen)

    # Any real ntfy send would go through here — assert it never fires.
    ntfy_calls: list = []
    monkeypatch.setattr(
        runner, "post_ntfy_alert",
        lambda *a, **k: ntfy_calls.append((a, k)) or True,
    )

    runner._run_once_multi_pipeline(
        {"live": {"preflight": {"enabled": True}}},
        models={},
        broker=_Broker(),
        strategy_dir=Path("backtesting/renquant_104"),
        sell_only=False,
        dry_run=True,
    )

    # Reached the decision line (pipeline ran) ...
    assert seen.get("pipeline_run") is True
    # ... but commit() (the persistence/order/promotion chokepoint) never ran,
    # so the isolated DB/state sentinel is untouched.
    assert "commit" not in seen
    assert not sentinel.exists()
    # ... and no notification left the process.
    assert ntfy_calls == []

    # Attestation is present and positive.
    payload = _parse_attestation(capsys.readouterr().out)
    assert payload == {
        "persisted": False, "notified": False, "promoted": False,
        "ordered": False, "reached_decision": True,
    }
    assert _attestation_is_positive(payload)
    # The adapter was told it is a preflight run.
    assert seen["adapter_kwargs"].get("preflight") is True
    assert seen["adapter_kwargs"].get("preflight_guard") is not None


def test_dry_run_shadow_broker_also_clean(monkeypatch, capsys):
    seen: dict = {}
    _install_fake_pipeline(monkeypatch, seen=seen)
    runner._run_once_multi_pipeline(
        {"live": {"preflight": {"enabled": True}}},
        models={},
        broker=_ShadowBroker(),
        strategy_dir=Path("backtesting/renquant_104"),
        sell_only=False,
        dry_run=True,
    )
    assert "commit" not in seen
    payload = _parse_attestation(capsys.readouterr().out)
    assert _attestation_is_positive(payload)


# ── Negative: a stray notification flips the attestation → fail closed ──────────

def test_dry_run_stray_notify_flips_attestation(monkeypatch, capsys):
    """If any code path attempts an ntfy during the probe, the send chokepoint
    records it and suppresses it — the attestation reports notified:true so the
    shell guard fails closed."""
    seen: dict = {}
    real_sends: list = []
    monkeypatch.setattr(
        runner, "post_ntfy_alert",
        lambda *a, **k: real_sends.append((a, k)) or True,
    )

    def _stray_notify(ctx):
        # A pipeline job (or miswiring) tries to notify mid-run.
        runner._post_ntfy_with_retries(
            "https://ntfy.sh/renquant",
            title="stray", body="should be suppressed", priority="default",
        )

    _install_fake_pipeline(
        monkeypatch, pipeline_side_effect=_stray_notify, seen=seen,
    )

    runner._run_once_multi_pipeline(
        {"live": {"preflight": {"enabled": True}}},
        models={},
        broker=_Broker(),
        strategy_dir=Path("backtesting/renquant_104"),
        sell_only=False,
        dry_run=True,
    )

    # The actual send was suppressed ...
    assert real_sends == []
    # ... but the attestation honestly reports the boundary was hit.
    payload = _parse_attestation(capsys.readouterr().out)
    assert payload["notified"] is True
    assert not _attestation_is_positive(payload)


# ── Preflight-check failure semantics under dry-run ─────────────────────────────

def test_dry_run_buy_side_block_fails_closed(monkeypatch, capsys):
    """A buy-side model-contract block is a daily-killer: the funnel never
    reached a normal decision line, so the attestation is reached_decision:false
    (fail closed), no ntfy is sent, and the runner exits non-zero."""
    seen: dict = {}
    real_sends: list = []
    monkeypatch.setattr(
        runner, "post_ntfy_alert",
        lambda *a, **k: real_sends.append((a, k)) or True,
    )
    _install_fake_pipeline(
        monkeypatch,
        preflight_fail_message="✗ P-WF-GATE stale wf gate metadata",
        seen=seen,
    )

    with pytest.raises(SystemExit) as exc:
        runner._run_once_multi_pipeline(
            {"live": {"preflight": {"enabled": True}}},
            models={},
            broker=_Broker(),
            strategy_dir=Path("backtesting/renquant_104"),
            sell_only=False,
            dry_run=True,
        )
    assert exc.value.code == 2
    assert "pipeline_run" not in seen  # never reached the pipeline
    assert real_sends == []            # no ntfy
    payload = _parse_attestation(capsys.readouterr().out)
    assert payload["reached_decision"] is False
    assert not _attestation_is_positive(payload)


def test_dry_run_hard_preflight_failure_not_reached(monkeypatch, capsys):
    """A hard/broker preflight failure means the probe is broken:
    reached_decision:false, no ntfy, SystemExit(2) → shell guard fails closed."""
    seen: dict = {}
    real_sends: list = []
    monkeypatch.setattr(
        runner, "post_ntfy_alert",
        lambda *a, **k: real_sends.append((a, k)) or True,
    )
    _install_fake_pipeline(
        monkeypatch,
        preflight_fail_message="✗ P-BROKER-CONN alpaca unreachable",
        seen=seen,
    )

    with pytest.raises(SystemExit) as exc:
        runner._run_once_multi_pipeline(
            {"live": {"preflight": {"enabled": True}}},
            models={},
            broker=_Broker(),
            strategy_dir=Path("backtesting/renquant_104"),
            sell_only=False,
            dry_run=True,
        )
    assert exc.value.code == 2
    assert real_sends == []
    payload = _parse_attestation(capsys.readouterr().out)
    assert payload["reached_decision"] is False
    assert not _attestation_is_positive(payload)


def test_active_guard_cleared_after_run(monkeypatch, capsys):
    """The module-level active guard must be reset so a later NON-preflight run
    still notifies normally."""
    seen: dict = {}
    _install_fake_pipeline(monkeypatch, seen=seen)
    runner._run_once_multi_pipeline(
        {"live": {"preflight": {"enabled": True}}},
        models={},
        broker=_Broker(),
        strategy_dir=Path("backtesting/renquant_104"),
        sell_only=False,
        dry_run=True,
    )
    capsys.readouterr()
    assert runner._ACTIVE_PREFLIGHT_GUARD is None
