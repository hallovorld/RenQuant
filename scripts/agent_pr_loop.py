#!/usr/bin/env python3
"""Launchd-safe local loop for cross-repo agent PR review/fix/merge.

This is the real scheduled replacement for the conversational "/loop" habit:
launchd fires a shell wrapper every 5 minutes, the wrapper loads agent tokens
from Keychain, and this script drives the deterministic control plane plus the
two local agent CLIs.

Fail-closed policy:
  * missing Claude/Codex actor tokens -> non-zero
  * missing local agent CLIs -> non-zero
  * orchestrator identity preflight failure -> non-zero
  * any review/fix/merge step failure -> non-zero
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB_ROOT = REPO_ROOT.parent
ORCH_ROOT = Path(
    os.environ.get("RENQUANT_ORCHESTRATOR_ROOT", str(GITHUB_ROOT / "renquant-orchestrator"))
).resolve()
PYTHON = Path(os.environ.get("RENQUANT_LOOP_PYTHON", str(REPO_ROOT / ".venv/bin/python")))
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

LOG_DIR = REPO_ROOT / "logs" / "agent_pr_loop"
STATUS_PATH = LOG_DIR / "status.json"
MAX_MERGES = 20


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(payload: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        input=stdin_text,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "rc": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
        "elapsed_s": round(time.time() - started, 3),
    }


def _orch_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ORCH_ROOT / "src")
    return env


def _agent_gh_env(agent: str) -> dict[str, str]:
    env = dict(os.environ)
    token_var = f"RENQUANT_{agent.upper()}_GH_TOKEN"
    token = env.get(token_var)
    if not token:
        raise RuntimeError(f"missing required token env: {token_var}")
    env["GH_TOKEN"] = token
    env["GITHUB_TOKEN"] = token
    return env


def _orch(args: list[str]) -> dict[str, Any]:
    return _run([str(PYTHON), "-m", "renquant_orchestrator", *args], cwd=ORCH_ROOT, env=_orch_env())


def _orch_json(args: list[str]) -> dict[str, Any]:
    result = _orch(args)
    if result["rc"] != 0:
        raise RuntimeError(
            f"orchestrator command failed rc={result['rc']}: {result['stderr'] or result['stdout']}"
        )
    text = str(result["stdout"]).strip()
    return json.loads(text) if text else {}


def _queue_total(plan_bundle: dict[str, Any]) -> int:
    total = 0
    for repo_row in plan_bundle.get("repos", []):
        total += len(((repo_row.get("plan") or {}).get("queue") or []))
    return total


def build_agent_prompt(agent: str, workflow: str) -> str:
    orch_cmd = (
        f"PYTHONPATH={ORCH_ROOT / 'src'} {PYTHON} -m renquant_orchestrator "
        f"repos agent --as {agent} --workflow {workflow} --repo all"
    )
    common = (
        f"You are running unattended on the operator machine.\n"
        f"Workspace root: {REPO_ROOT}\n"
        f"Sibling repo root: {GITHUB_ROOT}\n"
        f"Use only the PR queue from this command:\n{orch_cmd}\n\n"
        "Fail closed. Do not touch PRs outside that queue. If the queue is empty, stop.\n"
    )
    if workflow == "review":
        return common + (
            f"For each queued PR, inspect the diff, run focused validation as needed, and post one "
            f"consolidated GitHub review. The review text must include visible text `reviewed by {agent}`. "
            "Approve only when there is no blocking issue. Use request-changes only for BLOCKER/HIGH/MED findings. "
            "Do not merge anything in this step. Return a concise summary."
        )
    if workflow == "fix":
        return common + (
            f"For each queued PR authored by {agent}, make the smallest correct fix, run focused tests, "
            f"comment `fixed by {agent}`, commit, and push. Do not merge anything in this step. "
            "Return a concise summary with files changed and tests run."
        )
    raise ValueError(f"unsupported workflow {workflow!r}")


def _llm_command(agent: str) -> list[str]:
    if agent == "codex":
        return [
            "codex",
            "exec",
            "-c",
            "shell_environment_policy.inherit=all",
            "-C",
            str(REPO_ROOT),
            "--add-dir",
            str(GITHUB_ROOT),
            "--dangerously-bypass-approvals-and-sandbox",
            "-s",
            "danger-full-access",
            "-",
        ]
    if agent == "claude":
        return [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--add-dir",
            str(GITHUB_ROOT),
            "-",
        ]
    raise ValueError(f"unsupported agent {agent!r}")


def _require_local_clis() -> None:
    missing = [name for name in ("codex", "claude", "gh") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"missing required local CLIs: {', '.join(missing)}")


def _run_review_or_fix(agent: str, workflow: str) -> dict[str, Any]:
    plan = _orch_json(["repos", "agent", "--as", agent, "--workflow", workflow, "--repo", "all"])
    queue_total = _queue_total(plan)
    step: dict[str, Any] = {
        "agent": agent,
        "workflow": workflow,
        "queue_total": queue_total,
        "plan": plan,
    }
    if queue_total == 0:
        step["skipped"] = True
        return step
    step["exec"] = _run(
        _llm_command(agent),
        cwd=REPO_ROOT,
        env=_agent_gh_env(agent),
        stdin_text=build_agent_prompt(agent, workflow),
    )
    return step


def _run_merge(agent: str) -> dict[str, Any]:
    plan = _orch_json(
        [
            "repos",
            "agent",
            "--as",
            agent,
            "--workflow",
            "merge",
            "--repo",
            "all",
            "--execute",
            "--allow-all",
            "--max-merges",
            str(MAX_MERGES),
        ]
    )
    total_merged = int(plan.get("total_merged") or 0)
    return {
        "agent": agent,
        "workflow": "merge",
        "total_merged": total_merged,
        "plan": plan,
    }


def main() -> int:
    status: dict[str, Any] = {
        "started_at": _now(),
        "repo_root": str(REPO_ROOT),
        "orchestrator_root": str(ORCH_ROOT),
        "steps": [],
        "ok": False,
    }
    _write_status(status)
    try:
        _require_local_clis()

        identity = _orch(["agent-identity", "--strict"])
        status["steps"].append({"name": "agent-identity", "result": identity})
        if identity["rc"] != 0:
            raise RuntimeError("agent identity preflight failed")

        sync = _orch(["repos", "sync", "--repo", "all"])
        status["steps"].append({"name": "repos-sync", "result": sync})
        if sync["rc"] != 0:
            raise RuntimeError("repo sync failed")

        for agent, workflow in (
            ("codex", "review"),
            ("claude", "review"),
            ("codex", "fix"),
            ("claude", "fix"),
        ):
            step = _run_review_or_fix(agent, workflow)
            status["steps"].append({"name": f"{agent}-{workflow}", "result": step})
            exec_result = step.get("exec") or {}
            if exec_result and exec_result.get("rc", 0) != 0:
                raise RuntimeError(f"{agent} {workflow} failed")

        for agent in ("codex", "claude"):
            step = _run_merge(agent)
            status["steps"].append({"name": f"{agent}-merge", "result": step})

        status["ok"] = True
        status["finished_at"] = _now()
        _write_status(status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status["ok"] = False
        status["finished_at"] = _now()
        status["error"] = str(exc)
        _write_status(status)
        print(f"agent_pr_loop: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
