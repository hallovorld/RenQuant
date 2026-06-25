#!/usr/bin/env python3
"""Atomic, verified, reversible subrepo-pin promotion.

Replaces the manual "hand-edit subrepos.lock.json + run subrepo_assemble + hope"
dance — the 6-step process that produced the 2026-06-23 deploy fragility (see
doc/.../model-fixes-cant-reach-production postmortem, the roadmap's #2 eng fix).
DRY-RUN by default; nothing is written without --apply.

    promote_pin.py bump --subrepo renquant-pipeline --commit <sha>          # preview
    promote_pin.py bump --subrepo renquant-pipeline --commit <sha> --apply  # do it
    promote_pin.py revert --apply                                           # undo last

--apply path: backup subrepos.lock.json (timestamped) -> atomically write the new
pin (temp + os.replace, never a half-written lock) -> materialize via
subrepo_assemble.py --sync -> optionally run a verify command (e.g. the bundle
self-consistency check) -> AUTO-REVERT (restore backup + re-sync) if sync or
verify fails. Always prints the one-command manual revert.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "subrepos.lock.json"
ASSEMBLE = REPO / "scripts" / "subrepo_assemble.py"
DEFAULT_RUNTIME_ROOT = REPO / ".subrepo_runtime" / "repos"
BACKUP_SUFFIX = ".promote-bak."


def load_lock(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_entry(lock: dict, name: str) -> dict:
    for e in lock.get("subrepos", []):
        if e.get("name") == name:
            return e
    raise KeyError(f"subrepo {name!r} not in {LOCK.name} "
                   f"(have: {[e.get('name') for e in lock.get('subrepos', [])]})")


def _is_sha(s: str) -> bool:
    s = s.strip()
    return len(s) >= 7 and all(c in "0123456789abcdef" for c in s.lower())


def bump_pin(lock: dict, name: str, commit: str) -> tuple[str, dict]:
    """Return (old_commit, new_lock_dict). Pure — no I/O."""
    if not _is_sha(commit):
        raise ValueError(f"{commit!r} does not look like a git sha")
    new = json.loads(json.dumps(lock))  # deep copy
    entry = find_entry(new, name)
    old = entry.get("commit", "")
    if old == commit:
        raise ValueError(f"{name} is already pinned at {commit} — no-op")
    entry["commit"] = commit
    return old, new


def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    json.loads(tmp.read_text(encoding="utf-8"))  # validate it parses
    os.replace(tmp, path)


def backup_lock(path: Path, stamp: str) -> Path:
    dst = path.with_name(path.name + BACKUP_SUFFIX + stamp)
    dst.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def latest_backup(path: Path) -> Path | None:
    cands = sorted(path.parent.glob(path.name + BACKUP_SUFFIX + "*"))
    return cands[-1] if cands else None


def _sync(runtime_root: Path, python: str) -> tuple[int, str]:
    cmd = [python, str(ASSEMBLE), "--sync", "--runtime-root", str(runtime_root)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-800:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bump", help="bump one subrepo pin")
    b.add_argument("--subrepo", required=True)
    b.add_argument("--commit", required=True)
    r = sub.add_parser("revert", help="restore the most recent backup")
    for p in (b, r):
        p.add_argument("--apply", action="store_true", help="actually write + sync (else DRY-RUN)")
        p.add_argument("--no-sync", action="store_true", help="skip subrepo_assemble --sync")
        p.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
        p.add_argument("--python", default=sys.executable)
        p.add_argument("--verify-cmd", default=None,
                       help="shell command run after sync; non-zero → auto-revert")
        p.add_argument("--lock", default=str(LOCK))
    args = ap.parse_args(argv)
    lock_path = Path(args.lock)
    runtime_root = Path(args.runtime_root)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    if args.cmd == "revert":
        bak = latest_backup(lock_path)
        if not bak:
            print("no backup found — nothing to revert"); return 1
        print(f"REVERT: restore {bak.name} -> {lock_path.name}"
              + ("" if args.apply else "   [DRY-RUN, pass --apply]"))
        if not args.apply:
            return 0
        atomic_write_json(lock_path, json.loads(bak.read_text(encoding="utf-8")))
        if not args.no_sync:
            rc, out = _sync(runtime_root, args.python)
            print(out)
            return rc
        return 0

    # bump
    lock = load_lock(lock_path)
    old, new_lock = bump_pin(lock, args.subrepo, args.commit)
    print(f"BUMP {args.subrepo}:  {old or '(none)'}  ->  {args.commit}"
          + ("" if args.apply else "   [DRY-RUN, pass --apply]"))
    if not args.apply:
        return 0
    bak = backup_lock(lock_path, stamp)
    print(f"  backed up -> {bak.name}")
    atomic_write_json(lock_path, new_lock)

    def rollback(reason: str) -> int:
        print(f"  FAILED ({reason}) — auto-reverting", file=sys.stderr)
        atomic_write_json(lock_path, json.loads(bak.read_text(encoding="utf-8")))
        if not args.no_sync:
            _sync(runtime_root, args.python)
        return 1

    if not args.no_sync:
        rc, out = _sync(runtime_root, args.python)
        print(out)
        if rc != 0:
            return rollback("subrepo_assemble --sync")
    if args.verify_cmd:
        v = subprocess.run(args.verify_cmd, shell=True, capture_output=True, text=True)
        print(v.stdout[-800:])
        if v.returncode != 0:
            return rollback(f"verify ({args.verify_cmd!r})")
    print(f"  OK. revert with:  promote_pin.py revert --apply   "
          f"(or restore {bak.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
