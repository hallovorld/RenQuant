from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "refresh_subrepo_lock.py"
OLD_SHA = "a" * 40
NEW_SHA = "b" * 40


def _load_module():
    spec = importlib.util.spec_from_file_location("refresh_subrepo_lock_for_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _lock_payload(commit: str = "aaaaaaa") -> dict:
    return {
        "schema_version": 1,
        "source_repo": {"name": "RenQuant", "never_delete": True},
        "subrepos": [
            {
                "name": "renquant-backtesting",
                "remote": "https://github.com/hallovorld/renquant-backtesting",
                "branch": "main",
                "commit": commit,
                "test_command": "make test",
                "status": "bootstrapped",
            },
            {
                "name": "renquant-model",
                "remote": "https://github.com/hallovorld/renquant-model",
                "branch": "main",
                "commit": "ccccccc",
                "test_command": "make test",
                "status": "active",
            },
        ],
    }


def _write_lock(tmp_path: Path, payload: dict | None = None) -> Path:
    lock = tmp_path / "subrepos.lock.json"
    lock.write_text(json.dumps(payload or _lock_payload(), indent=2) + "\n", encoding="utf-8")
    return lock


def test_refresh_writes_only_after_ci_green_gate_passes(tmp_path: Path) -> None:
    mod = _load_module()
    lock = _write_lock(tmp_path)
    observed_original_during_gate = False

    def resolve(remote: str, branch: str) -> str:
        assert remote == "https://github.com/hallovorld/renquant-backtesting"
        assert branch == "main"
        return NEW_SHA

    def check(candidate_path: Path, *, only_subrepos: set[str]):
        nonlocal observed_original_during_gate
        observed_original_during_gate = json.loads(lock.read_text())["subrepos"][0]["commit"] == "aaaaaaa"
        candidate = json.loads(candidate_path.read_text())
        assert only_subrepos == {"renquant-backtesting"}
        assert candidate["subrepos"][0]["commit"] == "bbbbbbb"
        return {"ok": True, "pins": [{"ok": True}], "validated_subrepos": sorted(only_subrepos)}

    result = mod.refresh_lock(
        lock_file=lock,
        only_subrepos={"renquant-backtesting"},
        resolve_branch_head=resolve,
        check_lock_func=check,
    )

    assert result["ok"] is True
    assert result["wrote"] is True
    assert observed_original_during_gate is True
    assert json.loads(lock.read_text())["subrepos"][0]["commit"] == "bbbbbbb"


def test_ci_red_candidate_refuses_write_without_force(tmp_path: Path) -> None:
    mod = _load_module()
    lock = _write_lock(tmp_path)

    result = mod.refresh_lock(
        lock_file=lock,
        only_subrepos={"renquant-backtesting"},
        resolve_branch_head=lambda remote, branch: NEW_SHA,
        check_lock_func=lambda path, *, only_subrepos: {
            "ok": False,
            "pins": [{"name": "renquant-backtesting", "ok": False, "reason": "CI is red"}],
        },
    )

    assert result["ok"] is False
    assert result["wrote"] is False
    assert result["forced"] is False
    assert json.loads(lock.read_text())["subrepos"][0]["commit"] == "aaaaaaa"


def test_force_reason_writes_and_records_audit_event(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    lock = _write_lock(tmp_path)
    audit_log = tmp_path / "logs" / "force.jsonl"
    monkeypatch.setenv("GITHUB_ACTOR", "architect")

    result = mod.refresh_lock(
        lock_file=lock,
        only_subrepos={"renquant-backtesting"},
        force_reason="prod unblock approved in incident bridge",
        audit_log=audit_log,
        resolve_branch_head=lambda remote, branch: NEW_SHA,
        check_lock_func=lambda path, *, only_subrepos: {
            "ok": False,
            "pins": [{"name": "renquant-backtesting", "ok": False, "reason": "CI is red"}],
        },
    )

    assert result["ok"] is True
    assert result["forced"] is True
    assert json.loads(lock.read_text())["subrepos"][0]["commit"] == "bbbbbbb"
    events = [json.loads(line) for line in audit_log.read_text().splitlines()]
    assert events[0]["event"] == "subrepo_lock_ci_green_force_override"
    assert events[0]["actor"] == "architect"
    assert events[0]["reason"] == "prod unblock approved in incident bridge"
    assert events[0]["changes"][0]["old_commit"] == "aaaaaaa"
    assert events[0]["changes"][0]["new_commit"] == "bbbbbbb"


def test_no_pin_changes_skips_ci_gate_and_write(tmp_path: Path) -> None:
    mod = _load_module()
    lock = _write_lock(tmp_path, _lock_payload(commit=OLD_SHA[:7]))

    result = mod.refresh_lock(
        lock_file=lock,
        only_subrepos={"renquant-backtesting"},
        resolve_branch_head=lambda remote, branch: OLD_SHA,
        check_lock_func=lambda path, *, only_subrepos: (_ for _ in ()).throw(AssertionError("gate should not run")),
    )

    assert result["ok"] is True
    assert result["wrote"] is False
    assert result["changes"] == []


def test_dry_run_checks_gate_but_does_not_write(tmp_path: Path) -> None:
    mod = _load_module()
    lock = _write_lock(tmp_path)

    result = mod.refresh_lock(
        lock_file=lock,
        only_subrepos={"renquant-backtesting"},
        dry_run=True,
        resolve_branch_head=lambda remote, branch: NEW_SHA,
        check_lock_func=lambda path, *, only_subrepos: {
            "ok": True,
            "pins": [{"ok": True}],
            "validated_subrepos": sorted(only_subrepos),
        },
    )

    assert result["ok"] is True
    assert result["wrote"] is False
    assert result["dry_run"] is True
    assert result["changes"][0]["new_commit"] == "bbbbbbb"
    assert json.loads(lock.read_text())["subrepos"][0]["commit"] == "aaaaaaa"


def test_refresh_write_lock_blocks_concurrent_writer(tmp_path: Path) -> None:
    mod = _load_module()
    lock = _write_lock(tmp_path)
    lock_guard = lock.with_name(f"{lock.name}.lock")
    started = threading.Event()
    finished = threading.Event()
    result_box: dict[str, dict] = {}

    def refresh_in_thread() -> None:
        started.set()
        result_box["result"] = mod.refresh_lock(
            lock_file=lock,
            only_subrepos={"renquant-backtesting"},
            resolve_branch_head=lambda remote, branch: NEW_SHA,
            check_lock_func=lambda path, *, only_subrepos: {
                "ok": True,
                "pins": [{"ok": True}],
                "validated_subrepos": sorted(only_subrepos),
            },
        )
        finished.set()

    with lock_guard.open("w", encoding="utf-8") as fh:
        mod.fcntl.flock(fh.fileno(), mod.fcntl.LOCK_EX)
        worker = threading.Thread(target=refresh_in_thread)
        worker.start()
        assert started.wait(timeout=1.0)
        time.sleep(0.1)
        assert not finished.is_set()
        assert json.loads(lock.read_text())["subrepos"][0]["commit"] == "aaaaaaa"
        mod.fcntl.flock(fh.fileno(), mod.fcntl.LOCK_UN)

    worker.join(timeout=2.0)
    assert finished.is_set()
    assert result_box["result"]["ok"] is True
    assert json.loads(lock.read_text())["subrepos"][0]["commit"] == "bbbbbbb"
