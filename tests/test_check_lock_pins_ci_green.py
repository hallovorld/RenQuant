from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_lock_pins_ci_green.py"
FULL_SHA = "a" * 40


def _load_module():
    spec = importlib.util.spec_from_file_location("check_lock_pins_ci_green_for_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entry() -> dict[str, str]:
    return {
        "name": "renquant-backtesting",
        "remote": "https://github.com/hallovorld/renquant-backtesting",
        "branch": "main",
        "commit": "9a676e7",
    }


def _fake_github(*, workflow_runs=None, check_runs=None, status_state="pending", statuses=None, compare_status="ahead"):
    workflow_runs = [] if workflow_runs is None else workflow_runs
    check_runs = [] if check_runs is None else check_runs
    statuses = [] if statuses is None else statuses

    def fake(path: str):
        if path == "repos/hallovorld/renquant-backtesting/commits/9a676e7":
            return {"sha": FULL_SHA}
        if path == f"repos/hallovorld/renquant-backtesting/compare/{FULL_SHA}...main":
            return {"status": compare_status, "ahead_by": 1, "behind_by": 0}
        if path.startswith(f"repos/hallovorld/renquant-backtesting/actions/runs?head_sha={FULL_SHA}"):
            return {"workflow_runs": workflow_runs}
        if path == f"repos/hallovorld/renquant-backtesting/commits/{FULL_SHA}/check-runs":
            return {"check_runs": check_runs}
        if path == f"repos/hallovorld/renquant-backtesting/commits/{FULL_SHA}/status":
            return {"state": status_state, "statuses": statuses}
        raise AssertionError(path)

    return fake


def test_parse_github_remote_accepts_common_forms():
    mod = _load_module()

    assert mod.parse_github_remote("https://github.com/hallovorld/RenQuant.git") == "hallovorld/RenQuant"
    assert mod.parse_github_remote("git@github.com:hallovorld/RenQuant.git") == "hallovorld/RenQuant"
    assert mod.parse_github_remote("ssh://git@github.com/hallovorld/RenQuant.git") == "hallovorld/RenQuant"
    assert mod.parse_github_remote("https://example.com/hallovorld/RenQuant") is None


def test_completed_success_workflow_run_passes_even_if_legacy_status_is_empty_pending():
    mod = _load_module()
    result = mod.check_pin(
        _entry(),
        _fake_github(
            workflow_runs=[
                {"name": "CI", "event": "push", "status": "completed", "conclusion": "success"},
            ],
            status_state="pending",
            statuses=[],
        ),
    )

    assert result.ok is True
    assert result.evidence["checks"][0]["source"] == "actions/runs"


def test_pending_workflow_run_with_null_conclusion_fails():
    mod = _load_module()
    result = mod.check_pin(
        _entry(),
        _fake_github(
            workflow_runs=[
                {"name": "CI", "event": "push", "status": "in_progress", "conclusion": None},
            ],
        ),
    )

    assert result.ok is False
    assert result.reason == "CI is red, pending, or missing"


def test_failed_latest_workflow_run_fails():
    mod = _load_module()
    result = mod.check_pin(
        _entry(),
        _fake_github(
            workflow_runs=[
                {"name": "CI", "event": "push", "status": "completed", "conclusion": "failure"},
            ],
        ),
    )

    assert result.ok is False


def test_latest_rerun_by_workflow_name_wins():
    mod = _load_module()
    result = mod.check_pin(
        _entry(),
        _fake_github(
            workflow_runs=[
                {
                    "name": "CI",
                    "status": "completed",
                    "conclusion": "failure",
                    "updated_at": "2026-06-02T01:00:00Z",
                },
                {
                    "name": "CI",
                    "status": "completed",
                    "conclusion": "success",
                    "updated_at": "2026-06-02T01:10:00Z",
                },
            ],
        ),
    )

    assert result.ok is True


def test_check_run_fallback_passes_when_no_workflow_runs_exist():
    mod = _load_module()
    result = mod.check_pin(
        _entry(),
        _fake_github(
            workflow_runs=[],
            check_runs=[
                {"name": "test", "status": "completed", "conclusion": "success"},
            ],
        ),
    )

    assert result.ok is True
    assert result.evidence["checks"][1]["source"] == "commits/check-runs"


def test_no_checks_and_empty_pending_legacy_status_fails():
    mod = _load_module()
    result = mod.check_pin(
        _entry(),
        _fake_github(workflow_runs=[], check_runs=[], status_state="pending", statuses=[]),
    )

    assert result.ok is False
    assert "missing" in result.reason


def test_legacy_status_fallback_passes_when_no_check_runs_exist():
    mod = _load_module()
    result = mod.check_pin(
        _entry(),
        _fake_github(
            workflow_runs=[],
            check_runs=[],
            status_state="success",
            statuses=[{"context": "ci", "state": "success"}],
        ),
    )

    assert result.ok is True
    assert result.evidence["checks"][2]["source"] == "commits/status"


def test_commit_not_on_configured_branch_fails():
    mod = _load_module()
    result = mod.check_pin(_entry(), _fake_github(compare_status="diverged"))

    assert result.ok is False
    assert result.reason == "pinned commit is not on main"


def test_check_lock_reads_all_subrepos(tmp_path: Path):
    mod = _load_module()
    lock = tmp_path / "subrepos.lock.json"
    lock.write_text(json.dumps({"subrepos": [_entry()]}), encoding="utf-8")

    result = mod.check_lock(
        lock,
        _fake_github(
            workflow_runs=[
                {"name": "CI", "event": "push", "status": "completed", "conclusion": "success"},
            ],
        ),
    )

    assert result["ok"] is True
    assert result["pins"][0]["name"] == "renquant-backtesting"
