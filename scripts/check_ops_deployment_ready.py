#!/usr/bin/env python3
"""Preflight production ops deployment for the multirepo runtime.

This checker is intentionally read-only. It verifies that the canonical
RenQuant checkout is the code path launchd will run, that a pinned subrepo
runtime env exists, and that installed LaunchAgents match the repo sources.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CANONICAL_REPO = Path("/Users/renhao/git/github/RenQuant")

sys.path.insert(0, str(SCRIPTS))

from check_launchagents import inspect_launchagents  # noqa: E402
from subrepo_ops_contract import run_contract  # noqa: E402


@dataclass(frozen=True)
class Issue:
    severity: str
    check: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "check": self.check, "reason": self.reason}


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo_root), *args), text=True).strip()


def _read_exports(path: Path) -> dict[str, str]:
    exports: dict[str, str] = {}
    if not path.exists():
        return exports
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export ") :].split("=", 1)
        exports[key] = value.strip().strip('"').strip("'")
    return exports


def run_readiness(
    *,
    repo_root: Path = ROOT,
    canonical_repo: Path = CANONICAL_REPO,
    launchagents_dir: Path | None = None,
    allow_non_canonical: bool = False,
    allow_non_main: bool = False,
    skip_launchagents: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    canonical_repo = canonical_repo.expanduser().resolve()
    issues: list[Issue] = []
    details: dict[str, Any] = {
        "repo_root": str(repo_root),
        "canonical_repo": str(canonical_repo),
    }

    if repo_root != canonical_repo and not allow_non_canonical:
        issues.append(
            Issue(
                "error",
                "canonical_repo",
                f"run from canonical repo {canonical_repo}, not {repo_root}",
            )
        )

    try:
        branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        head = _git(repo_root, "rev-parse", "HEAD")
        origin_main = _git(repo_root, "rev-parse", "--verify", "origin/main")
        dirty = bool(_git(repo_root, "status", "--porcelain"))
    except subprocess.CalledProcessError as exc:
        issues.append(Issue("error", "git_state", f"git metadata failed: {exc}"))
        branch = head = origin_main = ""
        dirty = False

    details.update({"branch": branch, "head": head, "origin_main": origin_main, "dirty": dirty})
    if branch != "main" and not allow_non_main:
        issues.append(Issue("error", "git_branch", f"expected main, got {branch!r}"))
    if head and origin_main and head != origin_main:
        issues.append(Issue("error", "git_head", "HEAD does not match origin/main"))

    env_path = repo_root / ".subrepo_assembly" / "current.env"
    exports = _read_exports(env_path)
    details["subrepo_env"] = str(env_path)
    details["subrepo_root"] = exports.get("RENQUANT_SUBREPO_ROOT")
    if not exports.get("RENQUANT_SUBREPO_ROOT"):
        issues.append(
            Issue(
                "error",
                "runtime_root",
                "missing RENQUANT_SUBREPO_ROOT in .subrepo_assembly/current.env; run make subrepo-runtime-root",
            )
        )
    elif not Path(exports["RENQUANT_SUBREPO_ROOT"]).expanduser().exists():
        issues.append(
            Issue("error", "runtime_root", f"runtime root does not exist: {exports['RENQUANT_SUBREPO_ROOT']}")
        )

    contract = run_contract()
    details["subrepo_ops_contract_ok"] = bool(contract.get("ok"))
    if not contract.get("ok"):
        issues.append(Issue("error", "subrepo_ops_contract", json.dumps(contract.get("failures", []))))

    if skip_launchagents:
        launchagents: dict[str, Any] = {"ok": None, "skipped": True, "issues": [], "entries": []}
        details["launchagents_ok"] = None
    else:
        launchagents = inspect_launchagents(
            repo_root=repo_root,
            launchagents_dir=launchagents_dir or Path.home() / "Library" / "LaunchAgents",
        )
        details["launchagents_ok"] = bool(launchagents.get("ok"))
        if not launchagents.get("ok"):
            issues.append(
                Issue(
                    "error",
                    "launchagents",
                    "installed LaunchAgents drift from repo; run install after repo is ready",
                )
            )

    errors = [issue for issue in issues if issue.severity == "error"]
    return {
        "ok": not errors,
        "details": details,
        "issues": [issue.as_dict() for issue in issues],
        "launchagents": launchagents,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--canonical-repo", default=str(CANONICAL_REPO))
    parser.add_argument("--launchagents-dir", default=None)
    parser.add_argument("--allow-non-canonical", action="store_true")
    parser.add_argument("--allow-non-main", action="store_true")
    parser.add_argument(
        "--skip-launchagents",
        action="store_true",
        help="Skip installed LaunchAgents drift checks for pre-install gating.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run_readiness(
        repo_root=Path(args.repo_root),
        canonical_repo=Path(args.canonical_repo),
        launchagents_dir=Path(args.launchagents_dir).expanduser() if args.launchagents_dir else None,
        allow_non_canonical=args.allow_non_canonical,
        allow_non_main=args.allow_non_main,
        skip_launchagents=args.skip_launchagents,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = "PASS" if result["ok"] else "FAIL"
        print(f"RenQuant ops deployment readiness: {status}")
        for issue in result["issues"]:
            print(f"- {issue['severity']}: {issue['check']}: {issue['reason']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
