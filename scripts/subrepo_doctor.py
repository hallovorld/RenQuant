#!/usr/bin/env python3
"""Validate the RenQuant physical-subrepo assembly."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "subrepos.lock.json"
REQUIRED_FILES = (
    "CLAUDE.md",
    "README.md",
    "RENQUANT_REPOS.md",
    "renquant_repo.yml",
    "Makefile",
    ".github/workflows/ci.yml",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _run(repo: Path, command: str) -> int:
    return subprocess.run(command, cwd=repo, shell=True, check=False).returncode


def _norm_remote(remote: str) -> str:
    return remote.removesuffix(".git").rstrip("/")


def check_repo(entry: dict[str, Any], *, run_tests: bool) -> dict[str, Any]:
    path = Path(entry["local_path"])
    issues: list[str] = []
    if not path.exists():
        return {"name": entry["name"], "ok": False, "issues": [f"missing path: {path}"]}

    for rel in REQUIRED_FILES:
        if not (path / rel).exists():
            issues.append(f"missing required file: {rel}")

    try:
        branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
        commit = _git(path, "rev-parse", "HEAD")
        remote = _git(path, "remote", "get-url", "origin")
    except subprocess.CalledProcessError as exc:
        issues.append(f"git metadata failed: {exc}")
        branch = commit = remote = ""

    if branch and branch != entry.get("branch"):
        issues.append(f"branch mismatch: local={branch} lock={entry.get('branch')}")
    if commit and commit != entry.get("commit"):
        issues.append(f"commit mismatch: local={commit} lock={entry.get('commit')}")
    if remote and _norm_remote(remote) != _norm_remote(entry.get("remote", "")):
        issues.append(f"remote mismatch: local={remote} lock={entry.get('remote')}")

    test_rc = None
    if run_tests:
        test_rc = _run(path, entry["test_command"])
        if test_rc != 0:
            issues.append(f"test_command failed with rc={test_rc}: {entry['test_command']}")

    return {
        "name": entry["name"],
        "ok": not issues,
        "branch": branch,
        "commit": commit,
        "remote": remote,
        "test_rc": test_rc,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text())
    if lock["source_repo"].get("never_delete") is not True:
        raise SystemExit("source_repo.never_delete must be true")

    results = [check_repo(entry, run_tests=args.run_tests) for entry in lock["subrepos"]]
    print(json.dumps({"ok": all(r["ok"] for r in results), "repos": results}, indent=2))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
