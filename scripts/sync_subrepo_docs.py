#!/usr/bin/env python3
"""Generate each subrepo's RENQUANT_REPOS.md from subrepos.lock.json.

Single source of truth = the lock. The per-repo registry doc is a generated
snapshot (identical in every repo), so it can never drift from the roles/pins
the umbrella actually tracks. `subrepo_doctor.py` imports `render_repo_registry`
and fails any repo whose RENQUANT_REPOS.md differs from the generated content.

  python scripts/sync_subrepo_docs.py            # write to every repo + umbrella
  python scripts/sync_subrepo_docs.py --check    # exit 1 if any copy drifted
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "subrepos.lock.json"
REGISTRY_FILENAME = "RENQUANT_REPOS.md"

_STATIC = """## System flow — the model-factory pipeline

1. `renquant-base-data` publishes the training-data input + freshness/fingerprint contracts.
2. `renquant-common` + `renquant-pipeline` provide the shared code (Task/Job/Pipeline,
   purged_cv, walk_forward_splits, hmm_regime_labels, config_consistency).
3. `renquant-model` is the **model factory**: the `renquant_model_gbdt` and
   `renquant_model_patchtst` packages research + train models from base-data input and
   shared code, then publish artifact manifests to `renquant-artifacts`.
4. `renquant-artifacts` is the model registry (contracts + promotion status).
5. `renquant-strategy-104`, `renquant-pipeline` (runtime), `renquant-backtesting`, and
   `renquant-orchestrator` CONSUME models by `artifact_path` — never by importing the factory.
6. `renquant-execution` consumes order intents and performs broker actions with audit records.
7. `RenQuant` (umbrella) pins the whole assembly in `subrepos.lock.json` and stays the
   permanent integration harness + rollback source.

## Model lifecycle (build → validate → publish → consume)

See `RenQuant/doc/arch/multirepo-sop.md` §3 for the full SOP. In brief: the factory
BUILDs a candidate through the canonical engine, VALIDATEs it placebo-clean (WF IC +
shuffle/time-shift placebos, DSR/PBO, 3-tier gate), PUBLISHes the fingerprinted model to
`renquant-artifacts`, then the consuming side is PINned (`artifact_path` + lock pin). No
live flip without Tier 3. The factory never writes into a consumer.

## Cross-repo rules

- New code goes in the repo that OWNS the subject; never duplicate across repos; never add
  code to the umbrella `RenQuant` (integration/rollback only).
- Use `renquant-common` pipeline primitives for every workflow.
- Cross-repo docs (architecture/SOP/roles) live ONCE in `RenQuant/doc/arch/` and are
  referenced, never copied — replication is what causes stale-doc drift.
- Workflow: PR-based for ALL repos per umbrella `CLAUDE.md` §3.1 (2026-05-30 mandate,
  reverses the deleted 2026-05-27 verbal-merge convention). Feature branch →
  `make test` green → `git push -u origin <branch>` → `gh pr create --base main`
  → after verbal approval, `gh pr merge --merge --delete-branch`. NEVER
  `git push origin main` from a branch. Per umbrella `CLAUDE.md` §3.2, also
  `git fetch origin && git rebase origin/main` before opening any PR and before
  declaring merge-ready. After a subrepo PR merges, advance its pin in
  `subrepos.lock.json`.
- Large data, checkpoints, DBs, experiment dumps are referenced by manifest + fingerprint,
  not committed. A subrepo commit is not production-active until the umbrella pins it.
"""


def render_repo_registry(lock: dict) -> str:
    """Render the canonical RENQUANT_REPOS.md content from the lock (single source)."""
    src = lock["source_repo"]
    rows = [(src["name"], src["remote"], src.get("role", "permanent umbrella / rollback source"))]
    for r in lock["subrepos"]:
        rows.append((r["name"], r["remote"], " ".join(r["role"].split())))
    table = "\n".join(f"| `{n}` | `{u}` | {role} |" for n, u, role in rows)
    return (
        "# RenQuant Repository Map\n\n"
        "> AUTO-GENERATED from `RenQuant/subrepos.lock.json` by "
        "`scripts/sync_subrepo_docs.py`.\n"
        "> **Do not edit by hand** — edit the lock and re-run the sync. "
        "`subrepo_doctor.py` fails if this drifts.\n\n"
        "Gives an agent local big-picture context when starting inside any RenQuant repo.\n"
        "Canonical cross-repo docs (read, do not copy): "
        "`RenQuant/doc/arch/multirepo-sop.md` (architecture + SOP) and "
        "`subrepo-operating-model.md` (roles).\n\n"
        "The umbrella `RenQuant` is never deleted, emptied, or rewritten.\n\n"
        "## Repositories\n\n"
        "| Repo | Remote | Role |\n|---|---|---|\n" + table + "\n\n" + _STATIC
    )


def _targets(lock: dict) -> list[Path]:
    return [Path(lock["source_repo"]["local_path"])] + [
        Path(r["local_path"]) for r in lock["subrepos"]
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 on drift, write nothing")
    args = ap.parse_args()

    lock = json.loads(LOCK_PATH.read_text())
    content = render_repo_registry(lock)
    drifted: list[str] = []
    for repo in _targets(lock):
        if not repo.exists():
            drifted.append(f"{repo.name}: missing path")
            continue
        path = repo / REGISTRY_FILENAME
        if args.check:
            if not path.exists() or path.read_text() != content:
                drifted.append(f"{repo.name}: {REGISTRY_FILENAME} out of sync")
        else:
            path.write_text(content)
            print(f"  wrote {repo.name}/{REGISTRY_FILENAME}")

    if args.check and drifted:
        print("DRIFT:\n  " + "\n  ".join(drifted))
        return 1
    print("ok" if args.check else "synced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
