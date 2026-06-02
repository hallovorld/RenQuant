#!/usr/bin/env python3
"""Advance subrepos.lock.json pins, refusing CI-red candidate commits by default."""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_lock_pins_ci_green import check_lock  # noqa: E402


DEFAULT_LOCK_FILE = ROOT / "subrepos.lock.json"
DEFAULT_FORCE_AUDIT_LOG = ROOT / "logs" / "subrepo_lock_force_overrides.jsonl"

ResolveBranchHead = Callable[[str, str], str]
CheckLockFunc = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class PinChange:
    name: str
    remote: str
    branch: str
    old_commit: str
    new_commit: str
    full_sha: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "remote": self.remote,
            "branch": self.branch,
            "old_commit": self.old_commit,
            "new_commit": self.new_commit,
            "full_sha": self.full_sha,
        }


def _run_git_ls_remote(remote: str, branch: str) -> str:
    output = subprocess.check_output(
        ("git", "ls-remote", "--heads", remote, branch),
        text=True,
        stderr=subprocess.STDOUT,
    )
    ref = f"refs/heads/{branch}"
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    raise RuntimeError(f"could not resolve {remote} {branch!r}")


def _pin_width(entry: dict[str, Any], *, full_sha: bool) -> int:
    if full_sha:
        return 40
    current = str(entry.get("commit", ""))
    if 7 <= len(current) <= 40:
        return len(current)
    return 7


def _candidate_lock(
    lock: dict[str, Any],
    *,
    only_subrepos: set[str],
    full_sha: bool,
    resolve_branch_head: ResolveBranchHead,
) -> tuple[dict[str, Any], list[PinChange]]:
    candidate = copy.deepcopy(lock)
    changes: list[PinChange] = []
    for entry in candidate.get("subrepos", []):
        name = str(entry.get("name", ""))
        if only_subrepos and name not in only_subrepos:
            continue
        remote = str(entry.get("remote", ""))
        branch = str(entry.get("branch", "main") or "main")
        old_commit = str(entry.get("commit", ""))
        full = resolve_branch_head(remote, branch)
        new_commit = full[: _pin_width(entry, full_sha=full_sha)]
        if old_commit == new_commit:
            continue
        entry["commit"] = new_commit
        changes.append(
            PinChange(
                name=name,
                remote=remote,
                branch=branch,
                old_commit=old_commit,
                new_commit=new_commit,
                full_sha=full,
            )
        )
    return candidate, changes


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@contextlib.contextmanager
def _refresh_write_lock(lock_file: Path):
    """Serialize refreshes so concurrent agents cannot lose pin updates."""
    lock_path = lock_file.with_name(f"{lock_file.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _check_candidate_lock(
    candidate: dict[str, Any],
    *,
    changed_names: set[str],
    check_lock_func: CheckLockFunc,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="renquant-lock-refresh-") as tmp:
        candidate_path = Path(tmp) / "subrepos.lock.json"
        candidate_path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
        return check_lock_func(candidate_path, only_subrepos=changed_names)


def _write_force_audit(
    *,
    audit_log: Path,
    force_reason: str,
    changes: list[PinChange],
    gate_result: dict[str, Any],
) -> None:
    event = {
        "event": "subrepo_lock_ci_green_force_override",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "actor": os.environ.get("GITHUB_ACTOR") or os.environ.get("USER") or "unknown",
        "reason": force_reason,
        "changes": [change.as_dict() for change in changes],
        "gate_result": gate_result,
    }
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def refresh_lock(
    *,
    lock_file: Path = DEFAULT_LOCK_FILE,
    only_subrepos: set[str] | None = None,
    dry_run: bool = False,
    force_reason: str | None = None,
    audit_log: Path = DEFAULT_FORCE_AUDIT_LOG,
    full_sha: bool = False,
    resolve_branch_head: ResolveBranchHead = _run_git_ls_remote,
    check_lock_func: CheckLockFunc = check_lock,
) -> dict[str, Any]:
    if dry_run:
        return _refresh_lock_unlocked(
            lock_file=lock_file,
            only_subrepos=only_subrepos,
            dry_run=dry_run,
            force_reason=force_reason,
            audit_log=audit_log,
            full_sha=full_sha,
            resolve_branch_head=resolve_branch_head,
            check_lock_func=check_lock_func,
        )
    with _refresh_write_lock(lock_file):
        return _refresh_lock_unlocked(
            lock_file=lock_file,
            only_subrepos=only_subrepos,
            dry_run=dry_run,
            force_reason=force_reason,
            audit_log=audit_log,
            full_sha=full_sha,
            resolve_branch_head=resolve_branch_head,
            check_lock_func=check_lock_func,
        )


def _refresh_lock_unlocked(
    *,
    lock_file: Path = DEFAULT_LOCK_FILE,
    only_subrepos: set[str] | None = None,
    dry_run: bool = False,
    force_reason: str | None = None,
    audit_log: Path = DEFAULT_FORCE_AUDIT_LOG,
    full_sha: bool = False,
    resolve_branch_head: ResolveBranchHead = _run_git_ls_remote,
    check_lock_func: CheckLockFunc = check_lock,
) -> dict[str, Any]:
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    candidate, changes = _candidate_lock(
        lock,
        only_subrepos=set(only_subrepos or set()),
        full_sha=full_sha,
        resolve_branch_head=resolve_branch_head,
    )
    changed_names = {change.name for change in changes}
    gate_result: dict[str, Any] = {"ok": True, "pins": [], "validated_subrepos": []}
    forced = False

    if changes:
        gate_result = _check_candidate_lock(
            candidate,
            changed_names=changed_names,
            check_lock_func=check_lock_func,
        )
        if not gate_result.get("ok", False):
            if not force_reason:
                return {
                    "ok": False,
                    "wrote": False,
                    "forced": False,
                    "lock_file": str(lock_file),
                    "changes": [change.as_dict() for change in changes],
                    "gate_result": gate_result,
                    "reason": "candidate subrepo pin failed CI-green gate; pass --force REASON only for an approved emergency override",
                }
            forced = True
            if not dry_run:
                _write_force_audit(
                    audit_log=audit_log,
                    force_reason=force_reason,
                    changes=changes,
                    gate_result=gate_result,
                )

    if changes and not dry_run:
        _write_json_atomic(lock_file, candidate)

    return {
        "ok": True,
        "wrote": bool(changes and not dry_run),
        "forced": forced,
        "dry_run": dry_run,
        "lock_file": str(lock_file),
        "audit_log": str(audit_log) if forced else None,
        "changes": [change.as_dict() for change in changes],
        "gate_result": gate_result,
    }


def _print_human(result: dict[str, Any]) -> None:
    for change in result.get("changes", []):
        print(
            f"{change['name']}: {change['old_commit']} -> {change['new_commit']} "
            f"({change['branch']})"
        )
    if not result.get("changes"):
        print("subrepos.lock.json already matches configured branch heads")
    elif result.get("forced"):
        print(f"FORCED write after CI-green gate failure; audit={result.get('audit_log')}")
    elif result.get("wrote"):
        print("subrepos.lock.json updated after CI-green gate passed")
    elif result.get("dry_run"):
        print("dry-run: subrepos.lock.json would be updated")
    if not result.get("ok", False):
        print(result.get("reason", "subrepo lock refresh failed"), file=sys.stderr)
        print(json.dumps(result.get("gate_result", {}), indent=2, sort_keys=True), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--only-subrepo", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable result")
    parser.add_argument("--full-sha", action="store_true", help="write 40-character commits instead of preserving pin width")
    parser.add_argument(
        "--force",
        metavar="REASON",
        default=None,
        help="emergency override for CI-red/missing candidate pins; requires an explicit reason",
    )
    parser.add_argument("--audit-log", type=Path, default=DEFAULT_FORCE_AUDIT_LOG)
    args = parser.parse_args(argv)

    force_reason = args.force.strip() if args.force is not None else None
    if args.force is not None and not force_reason:
        parser.error("--force requires a non-empty reason")

    result = refresh_lock(
        lock_file=args.lock_file,
        only_subrepos=set(args.only_subrepo or []),
        dry_run=args.dry_run,
        force_reason=force_reason,
        audit_log=args.audit_log,
        full_sha=args.full_sha,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_human(result)
    return 0 if result.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
