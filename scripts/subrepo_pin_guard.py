#!/usr/bin/env python3
"""Runtime guard for RenQuant sibling subrepo pins."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PinIssue:
    repo: str
    path: str
    reason: str

    def format(self) -> str:
        return f"{self.repo} ({self.path}): {self.reason}"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _norm_remote(remote: str) -> str:
    return remote.removesuffix(".git").rstrip("/")


def _candidate_repo_paths(
    *,
    name: str,
    entry: dict,
    siblings: Path,
    root_override: str | None,
) -> list[Path]:
    candidates: list[Path] = []
    if root_override:
        candidates.append(Path(root_override) / name)
    if entry.get("local_path"):
        candidates.append(Path(str(entry["local_path"])))
    candidates.append(siblings / name)
    return candidates


def resolve_subrepo_src_roots(
    *,
    lock_file: Path,
    names: Iterable[str],
    siblings: Path,
    root_override: str | None = None,
) -> tuple[list[Path], list[PinIssue]]:
    """Resolve source roots and collect pin-drift issues.

    The guard checks the actual local checkout because runtime imports Python
    modules from local sibling repos, not directly from GitHub. A mismatch means
    the process would import code different from ``subrepos.lock.json``.
    """
    payload = json.loads(lock_file.read_text(encoding="utf-8"))
    lock_entries = {str(e["name"]): e for e in payload.get("subrepos", [])}
    roots: list[Path] = []
    issues: list[PinIssue] = []

    for name in names:
        entry = lock_entries.get(name)
        if entry is None:
            issues.append(PinIssue(name, str(lock_file), "missing from subrepos.lock.json"))
            continue

        repo_path = next(
            (candidate for candidate in _candidate_repo_paths(
                name=name,
                entry=entry,
                siblings=siblings,
                root_override=root_override,
            ) if (candidate / "src").is_dir()),
            None,
        )
        if repo_path is None:
            issues.append(PinIssue(name, "", "missing local src root"))
            continue

        roots.append(repo_path / "src")
        try:
            head = _git(repo_path, "rev-parse", "HEAD")
            remote = _git(repo_path, "remote", "get-url", "origin")
            dirty = bool(_git(repo_path, "status", "--porcelain"))
        except subprocess.CalledProcessError as exc:
            issues.append(PinIssue(name, str(repo_path), f"git metadata failed: {exc}"))
            continue

        expected = str(entry.get("commit", ""))
        if expected and not head.startswith(expected):
            issues.append(
                PinIssue(
                    name,
                    str(repo_path),
                    f"HEAD {head[:12]} does not match lock commit {expected}",
                )
            )
        expected_remote = str(entry.get("remote", ""))
        if expected_remote and _norm_remote(remote) != _norm_remote(expected_remote):
            issues.append(
                PinIssue(
                    name,
                    str(repo_path),
                    f"remote {remote} does not match lock remote {expected_remote}",
                )
            )
        if dirty:
            issues.append(PinIssue(name, str(repo_path), "working tree is dirty"))

    return roots, issues


def enforce_or_warn(issues: list[PinIssue], *, strict_env: str = "RENQUANT_STRICT_SUBREPO_PINS") -> None:
    if not issues:
        return
    message = "[multirepo] subrepo pin drift:\n" + "\n".join(
        f"  - {issue.format()}" for issue in issues
    )
    if os.environ.get(strict_env) == "1":
        print(message, file=sys.stderr)
        raise SystemExit(2)
    print(message + f"\n[multirepo] set {strict_env}=1 to fail closed.", file=sys.stderr)
