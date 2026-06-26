#!/usr/bin/env python3
"""System health check — `make doctor`.

One command to verify the live system is deploy-consistent. Composes the checks
whose ABSENCE produced the 2026-06-23 deploy fragility (see the
model-fixes-cant-reach-production postmortem):

  1. PIN/RUNTIME DRIFT — every materialized .subrepo_runtime/repos/<name> is
     checked out at EXACTLY its subrepos.lock.json pin and is not dirty. Nothing
     else verifies the live runtime still matches the audited pins; a hand-edit or
     a half-applied promote silently drifts it.
  2. LOCK INTEGRITY — subrepos.lock.json parses, source_repo.never_delete is true,
     and every pin is a full 40-hex git sha (catches a truncated/garbage pin).
  3. BUNDLE CONSISTENCY (best-effort) — shells out to the orchestrator's
     pre-deploy model-bundle self-consistency check (#188) if it is present in the
     runtime; reports SKIP (not RED) when unavailable.
  4. PROMOTE-BACKUP HYGIENE — warns if stale subrepos.lock.json.promote-bak.*
     backups have piled up (a half-finished promote leaves one behind).

Exit 0 = all green, 1 = at least one RED. --json for automation. Read-only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "subrepos.lock.json"
RUNTIME_ROOT = REPO / ".subrepo_runtime" / "repos"
_SHA = set("0123456789abcdef")


def load_lock(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def check_lock_integrity(lock: dict) -> list[dict]:
    out: list[dict] = []
    sr = lock.get("source_repo", {})
    out.append({"check": "source_repo.never_delete", "ok": sr.get("never_delete") is True,
                "detail": str(sr.get("never_delete"))})
    for e in lock.get("subrepos", []):
        c = str(e.get("commit", ""))
        ok = len(c) == 40 and all(ch in _SHA for ch in c.lower())
        out.append({"check": f"pin_is_full_sha[{e.get('name')}]", "ok": ok, "detail": c[:12]})
    return out


def check_pin_runtime_drift(lock: dict, runtime_root: Path) -> list[dict]:
    out: list[dict] = []
    for e in lock.get("subrepos", []):
        name, pin = e.get("name"), str(e.get("commit", ""))
        rt = runtime_root / name
        if not (rt / ".git").exists():
            out.append({"check": f"runtime[{name}]", "ok": True,
                        "detail": "not materialized (uses sibling/PYTHONPATH fallback)", "skip": True})
            continue
        try:
            head = _git(rt, "rev-parse", "HEAD")
            dirty = bool(_git(rt, "status", "--porcelain"))
        except Exception as exc:  # noqa: BLE001
            out.append({"check": f"runtime[{name}]", "ok": False, "detail": f"git error: {exc}"})
            continue
        at_pin = head == pin
        out.append({"check": f"runtime_at_pin[{name}]", "ok": at_pin,
                    "detail": f"head={head[:12]} pin={pin[:12]}" + (" DRIFT" if not at_pin else "")})
        out.append({"check": f"runtime_clean[{name}]", "ok": not dirty,
                    "detail": "dirty" if dirty else "clean"})
    return out


def check_promote_backups(lock_path: Path, warn_above: int = 3) -> list[dict]:
    baks = sorted(lock_path.parent.glob(lock_path.name + ".promote-bak.*"))
    return [{"check": "promote_backups", "ok": len(baks) <= warn_above,
             "detail": f"{len(baks)} stale backup(s)" + (f" (>{warn_above}, prune)" if len(baks) > warn_above else "")}]


def check_live_checkout_branch(repo: Path = REPO, expected: str = "main") -> dict:
    """The live umbrella checkout must stay on ``main``. A stray git op / a sub-agent
    operating in this shared tree instead of an isolated worktree can leave it on a
    feature branch or detached HEAD — the 2026-06-25 incident class. A feature-branch
    checkout carries that branch's committed pins, which the next ``preflight_pin_align``
    would deploy, silently reverting the live model. Catch it before the next run."""
    try:
        branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    except Exception as exc:  # noqa: BLE001
        return {"check": "live_checkout_branch", "ok": False, "detail": f"git error: {exc}"}
    ok = branch == expected
    return {"check": "live_checkout_branch", "ok": ok,
            "detail": f"on {branch}" + ("" if ok else f" — EXPECTED {expected} (stray checkout?)")}


def check_bundle(python: str | None = None) -> dict:
    checker = REPO / ".subrepo_runtime" / "repos" / "renquant-orchestrator" / "scripts" / "check_model_bundle_consistency.py"
    if not checker.exists():
        return {"check": "bundle_consistency", "ok": True, "skip": True,
                "detail": "checker not in runtime (orchestrator pin pre-#188) — SKIP"}
    import sys
    py = python or sys.executable
    p = subprocess.run([py, str(checker), "--json"], capture_output=True, text=True)
    ok = p.returncode == 0
    return {"check": "bundle_consistency", "ok": ok,
            "detail": "deploy_ready" if ok else (p.stdout or p.stderr)[-200:]}


def run_all(lock_path: Path = LOCK, runtime_root: Path = RUNTIME_ROOT) -> dict:
    lock = load_lock(lock_path)
    checks: list[dict] = []
    checks += check_lock_integrity(lock)
    checks.append(check_live_checkout_branch())
    checks += check_pin_runtime_drift(lock, runtime_root)
    checks += check_promote_backups(lock_path)
    checks.append(check_bundle())
    red = [c for c in checks if not c["ok"]]
    return {"ok": not red, "red": [c["check"] for c in red], "checks": checks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    res = run_all()
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        for c in res["checks"]:
            mark = "SKIP" if c.get("skip") else ("OK  " if c["ok"] else "RED ")
            print(f"  [{mark}] {c['check']}: {c['detail']}")
        print(f"\n{'✓ system green' if res['ok'] else '✗ RED: ' + ', '.join(res['red'])}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
