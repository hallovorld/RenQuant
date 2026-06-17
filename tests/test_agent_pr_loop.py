from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "agent_pr_loop.py"

ROADMAP_NEXT_STDOUT = (
    "Implement roadmap item `s3-fastore-wire` — Wire feature store into daily run\n"
    "Category: data. Target repo: renquant-orchestrator.\n\n"
    "Task:\nWire the PoC feature store into the daily run.\n\n"
    "Rules:\n- Branch off origin/main; write the code AND focused tests; run them.\n"
    "- Open a pull request. Do NOT merge anything — the merge gate owns merging.\n"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("agent_pr_loop_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ok(stdout: str = "", rc: int = 0) -> dict:
    return {"rc": rc, "stdout": stdout, "stderr": "", "cmd": [], "cwd": "", "elapsed_s": 0.0}


def _install_idle_main_stubs(mod, monkeypatch, *, recorder):
    """Stub everything main() needs so review/fix/merge are idle (no work),
    and route _orch + agent dispatch through recorders. Returns nothing; the
    caller inspects `recorder`."""
    monkeypatch.setattr(mod, "_require_local_clis", lambda: None)
    monkeypatch.setattr(mod, "_write_status", lambda payload: None)

    # Idle review/fix: empty queue -> no exec, no work.
    monkeypatch.setattr(mod, "_run_review_or_fix", lambda agent, wf: {"skipped": True})
    # Idle merge: nothing merged.
    monkeypatch.setattr(
        mod,
        "_run_merge",
        lambda agent: {"agent": agent, "workflow": "merge", "total_merged": 0, "plan": {}},
    )
    # No-op gh env for dispatch.
    monkeypatch.setattr(mod, "_agent_gh_env", lambda agent: {})
    monkeypatch.setattr(mod, "_llm_command", lambda agent: ["FAKE-AGENT", agent])

    def fake_run(cmd, *, cwd, env=None, stdin_text=None):
        recorder["dispatched"].append({"cmd": cmd, "stdin_text": stdin_text})
        return _ok("agent done")

    monkeypatch.setattr(mod, "_run", fake_run)


def test_build_agent_prompt_review_mentions_visible_review_marker() -> None:
    mod = _load_module()

    prompt = mod.build_agent_prompt("codex", "review")

    assert "reviewed by codex" in prompt
    assert "repos agent --as codex --workflow review --repo all" in prompt
    assert "Do not touch PRs outside that queue." in prompt


def test_build_agent_prompt_fix_mentions_visible_fix_marker() -> None:
    mod = _load_module()

    prompt = mod.build_agent_prompt("claude", "fix")

    assert "fixed by claude" in prompt
    assert "repos agent --as claude --workflow fix --repo all" in prompt


def test_queue_total_sums_cross_repo_plan_rows() -> None:
    mod = _load_module()

    total = mod._queue_total(
        {
            "repos": [
                {"repo": "RenQuant", "plan": {"queue": [{"number": 1}, {"number": 2}]}},
                {"repo": "renquant-pipeline", "plan": {"queue": [{"number": 3}]}},
                {"repo": "renquant-model", "plan": {"queue": []}},
            ]
        }
    )

    assert total == 3


# --- roadmap driver hook (OPT-IN, default OFF) ---------------------------------


def test_parse_roadmap_item_id_reads_first_line() -> None:
    mod = _load_module()

    assert mod._parse_roadmap_item_id(ROADMAP_NEXT_STDOUT) == "s3-fastore-wire"
    assert mod._parse_roadmap_item_id("no actionable roadmap item") is None
    assert mod._parse_roadmap_item_id("") is None


def test_roadmap_driver_step_dispatches_one_and_marks_in_progress(monkeypatch) -> None:
    mod = _load_module()
    orch_calls: list[list[str]] = []
    dispatched: list[dict] = []

    def fake_orch(args):
        orch_calls.append(args)
        if args[:2] == ["roadmap", "next"]:
            return _ok(ROADMAP_NEXT_STDOUT, rc=0)
        if args[:2] == ["roadmap", "mark"]:
            return _ok(f"marked {args[2]} -> {args[3]}", rc=0)
        raise AssertionError(f"unexpected orch call: {args}")

    def fake_run(cmd, *, cwd, env=None, stdin_text=None):
        dispatched.append({"cmd": cmd, "stdin_text": stdin_text})
        return _ok("agent done")

    monkeypatch.setattr(mod, "_orch", fake_orch)
    monkeypatch.setattr(mod, "_run", fake_run)
    monkeypatch.setattr(mod, "_agent_gh_env", lambda agent: {})
    monkeypatch.setattr(mod, "_llm_command", lambda agent: ["FAKE-AGENT", agent])

    step = mod._run_roadmap_driver("claude")

    assert step["dispatched"] is True
    assert step["item_id"] == "s3-fastore-wire"
    # Exactly one agent dispatched, fed the roadmap prompt verbatim.
    assert len(dispatched) == 1
    assert dispatched[0]["stdin_text"] == ROADMAP_NEXT_STDOUT
    # Item marked in_progress so it is not re-dispatched.
    assert ["roadmap", "mark", "s3-fastore-wire", "in_progress"] in orch_calls


def test_roadmap_driver_step_noop_when_nothing_actionable(monkeypatch) -> None:
    mod = _load_module()
    orch_calls: list[list[str]] = []
    dispatched: list[dict] = []

    def fake_orch(args):
        orch_calls.append(args)
        # rc=1 == nothing actionable.
        return _ok("no actionable roadmap item", rc=1)

    monkeypatch.setattr(mod, "_orch", fake_orch)
    monkeypatch.setattr(
        mod,
        "_run",
        lambda *a, **k: dispatched.append(k) or _ok("should not run"),
    )

    step = mod._run_roadmap_driver("claude")

    assert step["dispatched"] is False
    assert dispatched == []
    # Never marks anything when there is nothing to dispatch.
    assert all(c[:2] != ["roadmap", "mark"] for c in orch_calls)


def test_roadmap_driver_does_not_mark_when_agent_fails(monkeypatch) -> None:
    mod = _load_module()
    orch_calls: list[list[str]] = []

    def fake_orch(args):
        orch_calls.append(args)
        if args[:2] == ["roadmap", "next"]:
            return _ok(ROADMAP_NEXT_STDOUT, rc=0)
        if args[:2] == ["roadmap", "mark"]:
            raise AssertionError("must not mark a failed dispatch")
        raise AssertionError(f"unexpected orch call: {args}")

    monkeypatch.setattr(mod, "_orch", fake_orch)
    monkeypatch.setattr(mod, "_run", lambda *a, **k: _ok("agent failed", rc=7))
    monkeypatch.setattr(mod, "_agent_gh_env", lambda agent: {})
    monkeypatch.setattr(mod, "_llm_command", lambda agent: ["FAKE-AGENT", agent])

    step = mod._run_roadmap_driver("claude")

    assert step["dispatched"] is False
    assert step["exec"]["rc"] == 7
    assert all(c[:2] != ["roadmap", "mark"] for c in orch_calls)


def test_main_skips_roadmap_when_env_unset(monkeypatch) -> None:
    mod = _load_module()
    recorder: dict = {"dispatched": []}
    orch_calls: list[list[str]] = []

    _install_idle_main_stubs(mod, monkeypatch, recorder=recorder)

    def fake_orch(args):
        orch_calls.append(args)
        # identity / sync succeed; roadmap should NEVER be called here.
        return _ok("{}", rc=0)

    monkeypatch.setattr(mod, "_orch", fake_orch)
    monkeypatch.delenv("RQ_ROADMAP_DRIVER", raising=False)

    rc = mod.main()

    assert rc == 0
    # Default OFF: no roadmap call, no dispatch.
    assert all(c[:1] != ["roadmap"] for c in orch_calls)
    assert recorder["dispatched"] == []


def test_main_dispatches_one_roadmap_item_when_enabled_and_idle(monkeypatch) -> None:
    mod = _load_module()
    recorder: dict = {"dispatched": []}
    orch_calls: list[list[str]] = []

    _install_idle_main_stubs(mod, monkeypatch, recorder=recorder)

    def fake_orch(args):
        orch_calls.append(args)
        if args[:2] == ["roadmap", "next"]:
            return _ok(ROADMAP_NEXT_STDOUT, rc=0)
        if args[:2] == ["roadmap", "mark"]:
            return _ok("marked", rc=0)
        # identity / sync
        return _ok("{}", rc=0)

    monkeypatch.setattr(mod, "_orch", fake_orch)
    monkeypatch.setenv("RQ_ROADMAP_DRIVER", "1")

    rc = mod.main()

    assert rc == 0
    # Exactly one agent dispatched with the roadmap prompt.
    assert len(recorder["dispatched"]) == 1
    assert recorder["dispatched"][0]["stdin_text"] == ROADMAP_NEXT_STDOUT
    # Item marked in_progress.
    assert ["roadmap", "mark", "s3-fastore-wire", "in_progress"] in orch_calls


def test_main_enabled_but_nothing_actionable_dispatches_nothing(monkeypatch) -> None:
    mod = _load_module()
    recorder: dict = {"dispatched": []}
    orch_calls: list[list[str]] = []

    _install_idle_main_stubs(mod, monkeypatch, recorder=recorder)

    def fake_orch(args):
        orch_calls.append(args)
        if args[:2] == ["roadmap", "next"]:
            return _ok("no actionable roadmap item", rc=1)
        return _ok("{}", rc=0)

    monkeypatch.setattr(mod, "_orch", fake_orch)
    monkeypatch.setenv("RQ_ROADMAP_DRIVER", "1")

    rc = mod.main()

    assert rc == 0
    # roadmap next was consulted, but nothing dispatched or marked.
    assert ["roadmap", "next"] in orch_calls
    assert recorder["dispatched"] == []
    assert all(c[:2] != ["roadmap", "mark"] for c in orch_calls)


def test_main_fails_when_roadmap_mark_fails(monkeypatch) -> None:
    mod = _load_module()
    recorder: dict = {"dispatched": []}

    _install_idle_main_stubs(mod, monkeypatch, recorder=recorder)

    def fake_orch(args):
        if args[:2] == ["roadmap", "next"]:
            return _ok(ROADMAP_NEXT_STDOUT, rc=0)
        if args[:2] == ["roadmap", "mark"]:
            return _ok("mark failed", rc=2)
        return _ok("{}", rc=0)

    monkeypatch.setattr(mod, "_orch", fake_orch)
    monkeypatch.setenv("RQ_ROADMAP_DRIVER", "1")

    assert mod.main() == 1
    assert len(recorder["dispatched"]) == 1
