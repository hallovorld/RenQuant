#!/usr/bin/env python3
"""Fail lock-file pin bumps unless each pinned GitHub commit has green CI.

This is a pin-advance gate, not a runtime trading guard. It intentionally reads
remote GitHub state so ``subrepos.lock.json`` cannot advance to a commit whose
own repository has red, pending, or missing checks.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_FILE = ROOT / "subrepos.lock.json"
PASSING_CONCLUSIONS = {"success", "skipped", "neutral"}


GithubGet = Callable[[str], dict[str, Any]]
_TOKEN_SENTINEL = object()
_TOKEN_CACHE: str | None | object = _TOKEN_SENTINEL


@dataclass(frozen=True)
class PinResult:
    name: str
    remote: str
    commit: str
    owner_repo: str | None
    full_sha: str | None
    ok: bool
    reason: str
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "remote": self.remote,
            "commit": self.commit,
            "owner_repo": self.owner_repo,
            "full_sha": self.full_sha,
            "ok": self.ok,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def parse_github_remote(remote: str) -> str | None:
    """Return ``owner/repo`` for common GitHub remote URL forms."""
    remote = remote.strip().removesuffix(".git")
    if remote.startswith("git@github.com:"):
        return remote.split(":", 1)[1]
    if remote.startswith("ssh://git@github.com/"):
        return remote.removeprefix("ssh://git@github.com/")

    parsed = urllib.parse.urlparse(remote)
    if parsed.netloc.lower() != "github.com":
        return None
    path = parsed.path.strip("/")
    return path or None


def _github_token() -> str | None:
    """Resolve a GitHub token from env, then local gh auth for operator runs."""
    global _TOKEN_CACHE
    if _TOKEN_CACHE is not _TOKEN_SENTINEL:
        return _TOKEN_CACHE if isinstance(_TOKEN_CACHE, str) else None

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        try:
            token = subprocess.check_output(("gh", "auth", "token"), text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            token = ""
    _TOKEN_CACHE = token or None
    return token or None


def _github_get(path: str) -> dict[str, Any]:
    token = _github_token()
    req = urllib.request.Request(
        f"https://api.github.com/{path.lstrip('/')}",
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 - fixed GitHub API host
        return json.loads(resp.read().decode("utf-8"))


def _resolve_commit(owner_repo: str, commit: str, github_get: GithubGet) -> str:
    data = github_get(f"repos/{owner_repo}/commits/{commit}")
    sha = str(data.get("sha", ""))
    if not sha:
        raise ValueError("commit API response did not include sha")
    return sha


def _commit_on_branch(
    owner_repo: str,
    full_sha: str,
    branch: str,
    github_get: GithubGet,
) -> tuple[bool, dict[str, Any]]:
    encoded_branch = urllib.parse.quote(branch, safe="")
    compare = github_get(f"repos/{owner_repo}/compare/{full_sha}...{encoded_branch}")
    status = str(compare.get("status", ""))
    evidence = {
        "branch": branch,
        "compare_status": status,
        "ahead_by": compare.get("ahead_by"),
        "behind_by": compare.get("behind_by"),
    }
    return status in {"identical", "ahead"}, evidence


def _latest_by_name(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("name") or row.get("context") or "<unnamed>")
        stamp = str(row.get("updated_at") or row.get("completed_at") or row.get("started_at") or "")
        previous = latest.get(name)
        previous_stamp = (
            str(previous.get("updated_at") or previous.get("completed_at") or previous.get("started_at") or "")
            if previous
            else ""
        )
        if previous is None or stamp >= previous_stamp:
            latest[name] = row
    return list(latest.values())


def _workflow_runs_green(owner_repo: str, full_sha: str, github_get: GithubGet) -> tuple[bool | None, dict[str, Any]]:
    data = github_get(
        f"repos/{owner_repo}/actions/runs?"
        + urllib.parse.urlencode({"head_sha": full_sha, "per_page": "50"})
    )
    runs = _latest_by_name(list(data.get("workflow_runs", [])))
    evidence = {
        "source": "actions/runs",
        "runs": [
            {
                "name": run.get("name"),
                "event": run.get("event"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "html_url": run.get("html_url"),
            }
            for run in runs
        ],
    }
    if not runs:
        return None, evidence
    bad = [
        run for run in runs
        if run.get("status") != "completed" or run.get("conclusion") not in PASSING_CONCLUSIONS
    ]
    return not bad, evidence


def _check_runs_green(owner_repo: str, full_sha: str, github_get: GithubGet) -> tuple[bool | None, dict[str, Any]]:
    data = github_get(f"repos/{owner_repo}/commits/{full_sha}/check-runs")
    runs = _latest_by_name(list(data.get("check_runs", [])))
    evidence = {
        "source": "commits/check-runs",
        "runs": [
            {
                "name": run.get("name"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "html_url": run.get("html_url"),
            }
            for run in runs
        ],
    }
    if not runs:
        return None, evidence
    bad = [
        run for run in runs
        if run.get("status") != "completed" or run.get("conclusion") not in PASSING_CONCLUSIONS
    ]
    return not bad, evidence


def _legacy_status_green(owner_repo: str, full_sha: str, github_get: GithubGet) -> tuple[bool | None, dict[str, Any]]:
    data = github_get(f"repos/{owner_repo}/commits/{full_sha}/status")
    statuses = list(data.get("statuses", []))
    evidence = {
        "source": "commits/status",
        "state": data.get("state"),
        "statuses": [
            {
                "context": status.get("context"),
                "state": status.get("state"),
                "target_url": status.get("target_url"),
            }
            for status in statuses
        ],
    }
    if not statuses:
        return None, evidence
    return data.get("state") == "success", evidence


def _ci_green(owner_repo: str, full_sha: str, github_get: GithubGet) -> tuple[bool, dict[str, Any]]:
    """Return true only when a concrete completed-success signal exists."""
    checks: list[dict[str, Any]] = []
    for checker in (_workflow_runs_green, _check_runs_green, _legacy_status_green):
        ok, evidence = checker(owner_repo, full_sha, github_get)
        checks.append(evidence)
        if ok is not None:
            return ok, {"checks": checks}
    return False, {"checks": checks, "reason": "no workflow runs, check runs, or statuses found"}


def check_pin(entry: dict[str, Any], github_get: GithubGet = _github_get) -> PinResult:
    name = str(entry.get("name", "<missing>"))
    remote = str(entry.get("remote", ""))
    commit = str(entry.get("commit", ""))
    branch = str(entry.get("branch", "main") or "main")
    owner_repo = parse_github_remote(remote)
    if not owner_repo:
        return PinResult(name, remote, commit, None, None, False, "remote is not a GitHub repository", {})
    if not commit:
        return PinResult(name, remote, commit, owner_repo, None, False, "missing commit in lock entry", {})

    try:
        full_sha = _resolve_commit(owner_repo, commit, github_get)
        on_branch, branch_evidence = _commit_on_branch(owner_repo, full_sha, branch, github_get)
        if not on_branch:
            return PinResult(
                name,
                remote,
                commit,
                owner_repo,
                full_sha,
                False,
                f"pinned commit is not on {branch}",
                {"branch": branch_evidence},
            )
        ci_ok, ci_evidence = _ci_green(owner_repo, full_sha, github_get)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        return PinResult(name, remote, commit, owner_repo, None, False, f"GitHub lookup failed: {exc}", {})

    return PinResult(
        name,
        remote,
        commit,
        owner_repo,
        full_sha,
        ci_ok,
        "CI green" if ci_ok else "CI is red, pending, or missing",
        {"branch": branch_evidence, **ci_evidence},
    )


def check_lock(lock_file: Path = DEFAULT_LOCK_FILE, github_get: GithubGet = _github_get) -> dict[str, Any]:
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    results = [check_pin(entry, github_get) for entry in lock.get("subrepos", [])]
    return {
        "ok": all(result.ok for result in results),
        "lock_file": str(lock_file),
        "pins": [result.as_dict() for result in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--json", action="store_true", help="emit machine-readable result")
    args = parser.parse_args(argv)

    result = check_lock(args.lock_file)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for pin in result["pins"]:
            status = "PASS" if pin["ok"] else "FAIL"
            print(f"{status} {pin['name']}@{pin['commit']}: {pin['reason']}")
            if not pin["ok"]:
                print(json.dumps(pin["evidence"], indent=2, sort_keys=True), file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
