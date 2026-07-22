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
    assert "doc/progress/<date>-<slug>.md" in prompt
    assert "doc/AGENT-RETROSPECTIVE.md" in prompt


def test_build_agent_prompt_fix_mentions_visible_fix_marker() -> None:
    mod = _load_module()

    prompt = mod.build_agent_prompt("claude", "fix")

    assert "fixed by claude" in prompt
    assert "repos agent --as claude --workflow fix --repo all" in prompt
    assert "SHORT memory local/gitignored" in prompt or "SHORT memory" in prompt


def test_bootstrap_short_term_state_copies_template_when_missing(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    template = tmp_path / "doc" / "memory" / "short-term-state.template.md"
    target = tmp_path / "doc" / "memory" / "short-term-state.md"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("template body\n", encoding="utf-8")
    monkeypatch.setattr(mod, "SHORT_TERM_TEMPLATE", template)
    monkeypatch.setattr(mod, "SHORT_TERM_LOCAL", target)

    result = mod._bootstrap_short_term_state()

    assert result["bootstrapped"] is True
    assert target.read_text(encoding="utf-8") == "template body\n"


def test_bootstrap_short_term_state_skips_when_template_absent(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    template = tmp_path / "doc" / "memory" / "short-term-state.template.md"
    target = tmp_path / "doc" / "memory" / "short-term-state.md"
    monkeypatch.setattr(mod, "SHORT_TERM_TEMPLATE", template)
    monkeypatch.setattr(mod, "SHORT_TERM_LOCAL", target)

    result = mod._bootstrap_short_term_state()

    assert result["skipped"] is True
    assert result["bootstrapped"] is False


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


def test_roadmap_driver_records_hard_next_failure(monkeypatch) -> None:
    mod = _load_module()

    monkeypatch.setattr(
        mod,
        "_orch",
        lambda args: _ok("roadmap broken", rc=2)
        if args[:2] == ["roadmap", "next"]
        else (_ for _ in ()).throw(AssertionError(f"unexpected orch call: {args}")),
    )

    step = mod._run_roadmap_driver("claude")

    assert step["dispatched"] is False
    assert step["next"]["rc"] == 2
    assert step["next_error"] is True


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


def test_main_fails_when_roadmap_next_hard_fails(monkeypatch) -> None:
    mod = _load_module()
    recorder: dict = {"dispatched": []}

    _install_idle_main_stubs(mod, monkeypatch, recorder=recorder)

    def fake_orch(args):
        if args[:2] == ["roadmap", "next"]:
            return _ok("roadmap broken", rc=2)
        return _ok("{}", rc=0)

    monkeypatch.setattr(mod, "_orch", fake_orch)
    monkeypatch.setenv("RQ_ROADMAP_DRIVER", "1")

    assert mod.main() == 1
    assert recorder["dispatched"] == []


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


def _big_plan_json(n_repos: int = 12) -> str:
    """A control-plane plan bundle comfortably larger than the 8000-char log tail."""
    import json as _json

    filler = "x" * 900
    repos = [
        {
            "plan": {
                "repo": f"owner/repo-{i}",
                "instructions": filler,
                "queue": [{"number": 100 + i, "title": f"pr {i}"}],
            }
        }
        for i in range(n_repos)
    ]
    return _json.dumps({"action": "agent", "n_repos": n_repos, "repos": repos}, indent=2)


def test_run_keeps_untruncated_stdout_alongside_log_tail(tmp_path) -> None:
    """Regression: `stdout` is a log-sized tail, so parsers need `stdout_full`."""
    import json as _json

    mod = _load_module()
    payload = _big_plan_json()
    assert len(payload) > 8000

    result = mod._run(
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
        cwd=tmp_path,
        stdin_text=payload,
    )

    assert result["rc"] == 0
    assert len(result["stdout"]) == 8000
    assert result["stdout_full"] == payload
    assert _json.loads(result["stdout_full"])["n_repos"] == 12


def test_orch_json_parses_payload_larger_than_log_tail(monkeypatch) -> None:
    """Regression for the loop dying every cycle on a >8KB `repos agent` plan:
    json.loads must see the full stdout, not the truncated tail."""
    mod = _load_module()
    payload = _big_plan_json()

    def fake_orch(args):
        return {
            "rc": 0,
            "stdout": payload[-8000:],
            "stdout_full": payload,
            "stderr": "",
            "cmd": args,
            "cwd": "",
            "elapsed_s": 0.0,
        }

    monkeypatch.setattr(mod, "_orch", fake_orch)

    plan = mod._orch_json(["repos", "agent", "--as", "codex", "--workflow", "review"])

    assert plan["n_repos"] == 12
    assert mod._queue_total(plan) == 12


def test_orch_json_falls_back_to_stdout_when_full_absent(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_orch", lambda args: _ok('{"ok": true}'))

    assert mod._orch_json(["repos", "agent"]) == {"ok": True}


def test_orch_json_reports_non_json_stdout_with_head(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_orch", lambda args: _ok("not json at all"))

    try:
        mod._orch_json(["repos", "agent"])
    except RuntimeError as exc:
        assert "non-JSON stdout" in str(exc)
        assert "not json at all" in str(exc)
    else:  # pragma: no cover - guard
        raise AssertionError("expected RuntimeError")


def test_write_status_strips_full_stdout(tmp_path, monkeypatch) -> None:
    """status.json must stay bounded: the untruncated copy is dropped on write."""
    import json as _json

    mod = _load_module()
    monkeypatch.setattr(mod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "status.json")

    mod._write_status(
        {
            "ok": True,
            "steps": [{"name": "plan", "result": {"stdout": "tail", "stdout_full": "x" * 50000}}],
        }
    )

    written = (tmp_path / "status.json").read_text(encoding="utf-8")
    assert "stdout_full" not in written
    assert _json.loads(written)["steps"][0]["result"]["stdout"] == "tail"
