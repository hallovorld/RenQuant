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
import difflib
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
SNAPSHOT_RENDERER = REPO / "scripts" / "render_strategy_104_snapshot.py"
SNAPSHOT_OUTPUT = REPO / "doc" / "arch" / "strategy-104-snapshot.md"


def check_snapshot_freshness(python: str, repo: Path = REPO) -> tuple[bool, str]:
    """Regenerate-and-compare the strategy-104 production snapshot (M9/A6)
    against what a pin bump/rollback just produced. Renders to a SCRATCH path
    only — never auto-commits the regenerated content as the committed
    doc/arch/strategy-104-snapshot.md (per Codex review, PR #432 round 3);
    this only detects and reports drift so a human can review + commit the
    diff themselves. Returns (fresh, message)."""
    import tempfile

    renderer = repo / "scripts" / "render_strategy_104_snapshot.py"
    committed = repo / "doc" / "arch" / "strategy-104-snapshot.md"
    if not renderer.exists():
        return True, "snapshot renderer not present in this checkout — skipped"
    with tempfile.TemporaryDirectory(prefix="snapshot-freshness-") as td:
        scratch = Path(td) / "fresh.md"
        rendered = subprocess.run(
            [python, str(renderer), "--repo-root", str(repo), "--output", str(scratch)],
            capture_output=True, text=True,
        )
        if rendered.returncode != 0:
            # Pin drift (or another refusal) — surfaced by the renderer
            # itself, not something this backstop can resolve; report it.
            return False, (
                "ACTION REQUIRED: strategy-104 snapshot could not be regenerated:\n"
                + (rendered.stdout + rendered.stderr)[-1000:]
            )
        fresh_text = scratch.read_text(encoding="utf-8")
        committed_text = committed.read_text(encoding="utf-8") if committed.exists() else None
        if committed_text == fresh_text:
            return True, "strategy-104 snapshot is fresh"
        diff_preview = "".join(difflib.unified_diff(
            (committed_text or "").splitlines(keepends=True),
            fresh_text.splitlines(keepends=True),
            fromfile=str(committed), tofile="regenerated (not committed)", n=2,
        ))[:2000]
        return False, (
            "ACTION REQUIRED: doc/arch/strategy-104-snapshot.md is STALE relative "
            "to the sources this promote/rollback just changed. Regenerate and "
            "commit it yourself: `make snapshot`, then review + commit the diff. "
            "This tool does NOT auto-commit the regenerated snapshot.\n"
            + diff_preview
        )


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
        p.add_argument(
            "--skip-snapshot-check", action="store_true",
            help="skip the strategy-104 snapshot freshness backstop (M9/A6). "
                 "Default is ON: after a successful sync+verify, this tool "
                 "regenerates the snapshot to a scratch path and compares it "
                 "against the committed doc — if they differ, the command "
                 "exits non-zero with an actionable message. It never "
                 "auto-commits the regenerated snapshot and never reverts a "
                 "pin change for this reason alone (the pin change itself may "
                 "be entirely correct; only the doc needs a follow-up commit).",
        )
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
        rc = 0
        if not args.no_sync:
            rc, out = _sync(runtime_root, args.python)
            print(out)
        if rc == 0 and not args.skip_snapshot_check:
            fresh, msg = check_snapshot_freshness(args.python)
            print(f"  {msg}")
            if not fresh:
                rc = 1
        return rc

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
    # Default verify (no explicit --verify-cmd): the still-buys guard, so any
    # promote that would zero out admissions (the sell-only footgun) auto-reverts.
    verify_cmd = args.verify_cmd
    if verify_cmd is None:
        default_check = REPO / "scripts" / "check_conviction_admits.py"
        if default_check.exists():
            verify_cmd = f"{args.python} {default_check} --min-admits 1"
            print("  default verify: check_conviction_admits (still-buys guard)")
    if verify_cmd:
        v = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
        print(v.stdout[-800:])
        if v.returncode != 0:
            return rollback(f"verify ({verify_cmd!r})")
    print(f"  OK. revert with:  promote_pin.py revert --apply   "
          f"(or restore {bak.name})")
    if not args.skip_snapshot_check:
        # NOT gated on the pin being renquant-strategy-104 specifically: any
        # subrepo bump can indirectly change what the snapshot renders (e.g.
        # via an artifact/calibrator dependency), so the backstop always
        # runs. The pin change itself is NOT reverted for a stale-snapshot
        # finding alone — it may be entirely correct; only the doc needs a
        # follow-up commit.
        fresh, msg = check_snapshot_freshness(args.python)
        print(f"  {msg}")
        if not fresh:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
