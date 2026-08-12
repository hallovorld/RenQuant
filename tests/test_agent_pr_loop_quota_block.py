"""Non-retryable agent failures: name the cause, and stop re-running into it.

Regression oracle for the 2026-08-11 incident in which
`com.renquant.agent-pr-loop` reported `ok:false, error:"claude fix failed"`
every 300s for hours. The cause -- the spawned `claude` CLI had hit its
monthly spend limit -- was present in the subprocess's stdout and in
status.json, and appeared ZERO times in the durable log, which carried 471
copies of the reasonless summary instead.

Two properties are pinned here:
  1. the wrapper's error text carries the subprocess's own first line;
  2. a failure a retry cannot clear suppresses the next spawn instead of
     reproducing itself every cycle.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "agent_pr_loop.py"

# Verbatim from logs/agent_pr_loop/status.json on 2026-08-12T01:23:27Z.
SPEND_LIMIT_STDOUT = (
    "You've hit your monthly spend limit. Run /usage-credits to manage your "
    "limit and keep using Fable 5 or switch models to continue this chat.\n"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("agent_pr_loop_quota_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def mod(tmp_path, monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "QUOTA_BLOCK_PATH", tmp_path / "agent_quota_block.json")
    return m


def test_cause_is_read_from_stdout_when_stderr_is_empty(mod):
    """The real shape: the CLI puts the actionable line on stdout, stderr empty."""
    cause = mod._exec_failure_cause(
        {"rc": 1, "stdout": SPEND_LIMIT_STDOUT, "stdout_full": SPEND_LIMIT_STDOUT, "stderr": ""}
    )
    assert "monthly spend limit" in cause
    # The old behaviour discarded this entirely; that is the defect.
    assert cause != ""


def test_spend_limit_is_non_retryable_and_a_crash_is_not(mod):
    assert mod._is_non_retryable(
        "You've hit your monthly spend limit. Run /usage-credits to manage your limit"
    )
    # Default stays RETRYABLE: an unrecognised failure must behave as it does
    # today, so this classifier can only ever reduce noise.
    assert not mod._is_non_retryable("Traceback (most recent call last):")
    assert not mod._is_non_retryable("connection reset by peer")
    assert not mod._is_non_retryable("")


def test_block_suppresses_the_next_spawn_then_expires(mod):
    mod._record_quota_block("claude", "monthly spend limit")

    active = mod._quota_block_active("claude")
    assert active is not None and "monthly spend limit" in active["cause"]
    assert mod._quota_block_active("codex") is None, "block must be per-agent"

    # Expiry is what makes this a suppression WINDOW, not a kill switch: a
    # lifted cap is picked up without anyone clearing state by hand.
    import time as _t

    later = _t.time() + mod.QUOTA_REPROBE_SECONDS + 1
    assert mod._quota_block_active("claude", now=later) is None


def test_observations_accumulate_and_first_seen_is_preserved(mod):
    mod._record_quota_block("claude", "monthly spend limit")
    first = json.loads(mod.QUOTA_BLOCK_PATH.read_text())["claude"]["first_seen_at"]
    mod._record_quota_block("claude", "monthly spend limit")
    rec = json.loads(mod.QUOTA_BLOCK_PATH.read_text())["claude"]
    assert rec["observations"] == 2
    assert rec["first_seen_at"] == first, "first_seen_at must survive re-observation"


def test_success_clears_the_block(mod):
    mod._record_quota_block("claude", "monthly spend limit")
    assert mod._quota_block_active("claude") is not None
    mod._clear_quota_block("claude")
    assert mod._quota_block_active("claude") is None


def test_corrupt_block_file_reads_as_no_block(mod):
    """Fail OPEN here on purpose: a damaged state file must not wedge the loop."""
    mod.QUOTA_BLOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    mod.QUOTA_BLOCK_PATH.write_text("{not json")
    assert mod._quota_block_active("claude") is None


def test_review_or_fix_skips_the_spawn_while_blocked(mod, monkeypatch):
    """The behavioural claim: no subprocess is launched into a known-capped CLI."""
    monkeypatch.setattr(
        mod, "_orch_json", lambda *a, **k: {"repos": [{"plan": {"queue": [1, 2]}}]}
    )
    spawned = []
    monkeypatch.setattr(mod, "_run", lambda *a, **k: spawned.append(a) or {"rc": 0})
    monkeypatch.setattr(mod, "_agent_gh_env", lambda agent: {})
    monkeypatch.setattr(mod, "build_agent_prompt", lambda agent, wf: "prompt")

    # Queue is non-empty, so the queue_total==0 short-circuit does NOT apply --
    # this is exactly the state the incident was in (q=2, skipped=None).
    step = mod._run_review_or_fix("claude", "fix")
    assert step["queue_total"] == 2 and step.get("skipped") is not True
    assert spawned, "sanity: an unblocked agent must still be spawned"

    spawned.clear()
    mod._record_quota_block("claude", "monthly spend limit")
    step = mod._run_review_or_fix("claude", "fix")
    assert step["skipped"] is True
    assert step["quota_blocked"]["cause"] == "monthly spend limit"
    assert not spawned, "a blocked agent must not be spawned again"


# ---------------------------------------------------------------------------
# main()-level containment. The focused tests above all passed while main()
# still aborted the cycle on the FIRST spend-cap failure -- the claim and the
# code disagreed and no test sat at the boundary where the claim is decided.
# These do.
# ---------------------------------------------------------------------------


def _drive_main(mod, monkeypatch, *, claude_cap: bool):
    """Run main() with every external call stubbed; report what really ran."""
    seen = {"orch": [], "spawned": [], "merges": [], "status": []}

    def _orch(args):
        seen["orch"].append(list(args))
        return {"cmd": args, "rc": 0, "stdout": "{}", "stdout_full": "{}", "stderr": ""}

    def _orch_json(args):
        # Non-empty queue for claude/fix only -- the incident's exact state.
        as_agent = args[args.index("--as") + 1] if "--as" in args else ""
        workflow = args[args.index("--workflow") + 1] if "--workflow" in args else ""
        n = 2 if (as_agent == "claude" and workflow == "fix") else 0
        return {"repos": [{"plan": {"queue": [{}] * n}}]}

    def _run(cmd, **kw):
        seen["spawned"].append(cmd[0])
        if cmd[0] == "claude" and claude_cap:
            return {
                "cmd": cmd, "rc": 1, "stderr": "",
                "stdout": SPEND_LIMIT_STDOUT, "stdout_full": SPEND_LIMIT_STDOUT,
            }
        return {"cmd": cmd, "rc": 0, "stdout": "", "stdout_full": "", "stderr": ""}

    def _run_merge(agent):
        seen["merges"].append(agent)
        return {"agent": agent, "total_merged": 0}

    monkeypatch.setattr(mod, "_require_local_clis", lambda: None)
    monkeypatch.setattr(mod, "_bootstrap_short_term_state", lambda: {"skipped": True})
    monkeypatch.setattr(mod, "_orch", _orch)
    monkeypatch.setattr(mod, "_orch_json", _orch_json)
    monkeypatch.setattr(mod, "_run", _run)
    monkeypatch.setattr(mod, "_run_merge", _run_merge)
    monkeypatch.setattr(mod, "_agent_gh_env", lambda agent: {})
    monkeypatch.setattr(mod, "build_agent_prompt", lambda agent, wf: "prompt")
    monkeypatch.setattr(mod, "_write_status", lambda p: seen["status"].append(dict(p)))
    monkeypatch.delenv("RQ_ROADMAP_DRIVER", raising=False)

    rc = mod.main()
    return rc, seen, (seen["status"][-1] if seen["status"] else {})


def test_spend_cap_does_not_cost_the_merges_or_the_audit(mod, monkeypatch):
    """THE regression: the cycle that DISCOVERS the cap must still finish.

    Before the fix main() recorded the block and then raised, so this very
    cycle lost both merge stages and `repos merge-audit --strict` -- the exact
    opposite of the containment the change claims.
    """
    rc, seen, status = _drive_main(mod, monkeypatch, claude_cap=True)

    assert seen["merges"] == ["codex", "claude"], (
        f"merge stages were skipped: {seen['merges']}"
    )
    audits = [a for a in seen["orch"] if "merge-audit" in a]
    assert audits, "strict merge audit never ran"
    assert "--strict" in audits[0]

    assert rc == 0, "a contained, degraded cycle must not report a hard failure"
    assert status.get("degraded") == ["claude"]
    assert "monthly spend limit" in status["quota_blocked"]["claude"]["cause"]
    assert status.get("ok") is True
    assert "error" not in status, "a contained cap is not a cycle error"


def test_second_cycle_does_not_respawn_the_capped_agent(mod, monkeypatch):
    _drive_main(mod, monkeypatch, claude_cap=True)
    _, seen2, status2 = _drive_main(mod, monkeypatch, claude_cap=True)

    assert "claude" not in seen2["spawned"], (
        f"capped agent was respawned: {seen2['spawned']}"
    )
    assert seen2["merges"] == ["codex", "claude"], "merges must keep running"
    assert status2.get("degraded") == ["claude"]


def test_unknown_failure_still_aborts_the_cycle(mod, monkeypatch):
    """The default must not drift: an unrecognised failure fails loudly."""
    def _run(cmd, **kw):
        return {"cmd": cmd, "rc": 1, "stdout": "Traceback (most recent call last):",
                "stdout_full": "Traceback (most recent call last):", "stderr": ""}

    monkeypatch.setattr(mod, "_require_local_clis", lambda: None)
    monkeypatch.setattr(mod, "_bootstrap_short_term_state", lambda: {"skipped": True})
    monkeypatch.setattr(mod, "_orch", lambda a: {"rc": 0, "stdout": "{}", "stdout_full": "{}", "stderr": ""})
    monkeypatch.setattr(mod, "_orch_json", lambda a: {"repos": [{"plan": {"queue": [{}, {}]}}]})
    monkeypatch.setattr(mod, "_run", _run)
    monkeypatch.setattr(mod, "_run_merge", lambda agent: {"total_merged": 0})
    monkeypatch.setattr(mod, "_agent_gh_env", lambda agent: {})
    monkeypatch.setattr(mod, "build_agent_prompt", lambda agent, wf: "prompt")
    captured = []
    monkeypatch.setattr(mod, "_write_status", lambda p: captured.append(dict(p)))
    monkeypatch.delenv("RQ_ROADMAP_DRIVER", raising=False)

    assert mod.main() == 1
    last = captured[-1]
    assert last["ok"] is False
    # The cause must still be lifted into the error -- that is the other half
    # of this PR and it applies to retryable failures too.
    assert "Traceback" in last["error"]


def test_expired_block_is_not_reported_as_active(mod, monkeypatch):
    """Secondary: an expired block must drop out of the terminal view.

    Otherwise a cap that lifted while the queue was empty (so no success ever
    cleared it) is reported as an active quota condition forever.
    """
    mod._record_quota_block("claude", "monthly spend limit")
    monkeypatch.setattr(mod, "QUOTA_REPROBE_SECONDS", 0)

    # The queue must be EMPTY: with work to do the exec succeeds and
    # _clear_quota_block() removes the record, which hides the defect. The
    # reported scenario is precisely the one where nothing can clear it.
    monkeypatch.setattr(mod, "_require_local_clis", lambda: None)
    monkeypatch.setattr(mod, "_bootstrap_short_term_state", lambda: {"skipped": True})
    monkeypatch.setattr(mod, "_orch", lambda a: {"rc": 0, "stdout": "{}", "stdout_full": "{}", "stderr": ""})
    monkeypatch.setattr(mod, "_orch_json", lambda a: {"repos": [{"plan": {"queue": []}}]})
    monkeypatch.setattr(mod, "_run", lambda *a, **k: pytest.fail("nothing should spawn"))
    monkeypatch.setattr(mod, "_run_merge", lambda agent: {"total_merged": 0})
    captured = []
    monkeypatch.setattr(mod, "_write_status", lambda p: captured.append(dict(p)))
    monkeypatch.delenv("RQ_ROADMAP_DRIVER", raising=False)

    assert mod.main() == 0
    status = captured[-1]
    assert "degraded" not in status, f"expired block still reported: {status.get('degraded')}"
    assert "quota_blocked" not in status
    # The record itself survives on disk -- history is not destroyed, only the
    # ACTIVE view is filtered.
    assert "claude" in json.loads(mod.QUOTA_BLOCK_PATH.read_text())
