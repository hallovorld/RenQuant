#!/usr/bin/env python3
"""Pin import-integrity sweep (GOAL-5 AC5, D1).

Proves that a candidate pin COMBINATION can actually serve every aliased
kernel import the pinned pipeline may lazily reach at runtime — the class
of cross-repo gap that per-repo CI is structurally blind to.

Motivating incident (2026-07-16, orchestrator #524): the F-8 bootstrap
aliased non-owned kernel stems only as ``kernel.<stem>``; the pinned
pipeline's ``pp_inference`` lazily imports
``renquant_pipeline.kernel.meta_label.task_meta_label_veto`` (which exists
only in the authoritative renquant-backtesting copy). Both repos' own CI
was green; the first full daily after the pin sync died mid-run. This
sweep fails that combination at PR time instead.

Method:
  1. Bootstrap the multirepo runtime EXACTLY as the daily does
     (``renquant_orchestrator.live_bridge.bootstrap_multirepo``) against
     the candidate lock + checkouts.
  2. AST-walk the pinned pipeline source for every import statement —
     module-level AND function-local — whose target lives in an aliased
     namespace (``kernel.*`` / ``renquant_pipeline.kernel.*``).
  3. Import each target in-process, post-bootstrap. For ``from M import
     n`` also require ``n`` to resolve as an attribute or submodule of the
     ALIASED ``M``.
  4. Any failure names the import, the source site, and which side must
     fix it. Exit 1.

Run in a FRESH interpreter per combination (aliasing mutates sys.modules);
the test harness invokes this script as a subprocess.

Usage:
  pin_import_integrity_sweep.py --lock-file subrepos.lock.json \
      --siblings /path/containing/renquant-* [--json]

``--siblings`` must contain checkouts of renquant-orchestrator,
renquant-pipeline, renquant-backtesting and renquant-common at EXACTLY the
lock pins (the bootstrap's own pin guard verifies this — a drifted
checkout fails the sweep honestly rather than validating the wrong code).
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

ALIASED_PREFIXES = ("kernel.", "renquant_pipeline.kernel.")


def collect_aliased_imports(pipeline_src: Path) -> list[dict]:
    """Every (module, [names], file, line) import target in aliased namespaces."""
    out: list[dict] = []
    for py in sorted(pipeline_src.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # a broken pinned file is itself a finding
            out.append({"module": None, "names": [], "file": str(py),
                        "line": exc.lineno or 0, "syntax_error": str(exc)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(ALIASED_PREFIXES) or any(
                        alias.name == p.rstrip(".") for p in ALIASED_PREFIXES
                    ):
                        out.append({"module": alias.name, "names": [],
                                    "file": str(py), "line": node.lineno})
            elif isinstance(node, ast.ImportFrom):
                # relative imports resolve inside the pipeline package
                # itself and cannot cross the alias boundary; skip level>0.
                if node.level or not node.module:
                    continue
                if node.module.startswith(ALIASED_PREFIXES):
                    out.append({
                        "module": node.module,
                        "names": [a.name for a in node.names if a.name != "*"],
                        "file": str(py), "line": node.lineno,
                    })
    return out


def sweep(lock_file: Path, siblings: Path) -> dict:
    orch_src = siblings / "renquant-orchestrator" / "src"
    pipeline_src = (siblings / "renquant-pipeline" / "src" / "renquant_pipeline")
    for required, label in ((orch_src, "orchestrator src"),
                            (pipeline_src, "pipeline package")):
        if not required.is_dir():
            return {"ok": False, "failures": [
                {"error": f"missing {label}: {required}"}]}
    sys.path.insert(0, str(orch_src))

    # Minimal scratch umbrella root: the bootstrap needs a repo_root for
    # path layout + the lock file; src roots resolve via --siblings.
    scratch = Path(tempfile.mkdtemp(prefix="pin-sweep-root."))
    (scratch / "backtesting" / "renquant_104").mkdir(parents=True)

    # The resolver prefers each lock entry's local_path over the siblings
    # dir — on the dev machine those point at the REAL sibling checkouts,
    # which silently substitutes the wrong code for the sweep (observed:
    # the sweep "found" a missing module that only the real sibling's
    # stale branch lacked). Rewrite every local_path to the sweep's own
    # checkout for that name, and run the pin guard fail-closed so a
    # checkout that does not match the candidate lock aborts the sweep
    # instead of validating the wrong tree.
    import json as _json  # noqa: PLC0415
    import subprocess as _sp  # noqa: PLC0415
    lock_data = _json.loads(Path(lock_file).read_text())
    provided: list[str] = []
    kept = []
    for entry in lock_data.get("subrepos", []):
        name = entry.get("name")
        checkout = siblings / str(name)
        if not (name and checkout.is_dir()):
            # The sweep validates exactly the combination it was GIVEN;
            # lock entries without a provided checkout are dropped from the
            # sweep lock (and from pin_srcs) so the strict guard verifies
            # the provided set instead of unrelated machine state.
            continue
        entry["local_path"] = str(checkout)
        # Fixture checkouts are cloned from local mirrors, so their origin
        # differs from the lock's canonical remote. Import integrity does
        # not depend on remote identity; keep the strict guard focused on
        # COMMIT identity by aligning the remote field to the checkout.
        # (In CI the checkouts come from the canonical remotes, so this
        # rewrite is a no-op there.)
        try:
            origin = _sp.run(
                ["git", "-C", str(checkout), "config", "--get", "remote.origin.url"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            if origin:
                entry["remote"] = origin
        except Exception:  # noqa: BLE001 — leave as-is; guard will report
            pass
        provided.append(name)
        kept.append(entry)
    lock_data["subrepos"] = kept
    sweep_lock = scratch / "subrepos.lock.json"
    sweep_lock.write_text(_json.dumps(lock_data, indent=2))
    lock_file = sweep_lock
    os.environ["RENQUANT_STRICT_SUBREPO_PATHS"] = "1"

    from renquant_orchestrator.live_bridge import bootstrap_multirepo  # noqa: PLC0415

    try:
        aliased = bootstrap_multirepo(
            repo_root=scratch, lock_file=lock_file, siblings=siblings,
            pin_srcs=[n for n in provided if n != "renquant-orchestrator"],
        )
    except Exception as exc:  # noqa: BLE001 — bootstrap refusal IS a finding
        chain, cur = [], exc
        while cur is not None and len(chain) < 7:
            chain.append(repr(cur))
            cur = cur.__cause__ or cur.__context__
        return {"ok": False, "aliased": [], "failures": [
            {"error": "bootstrap_multirepo failed: " + " <- ".join(chain),
             "fix_side": "renquant-orchestrator bootstrap vs lock/checkouts"}]}

    targets = collect_aliased_imports(pipeline_src)
    failures: list[dict] = []
    for t in targets:
        if t.get("syntax_error"):
            failures.append({**t, "fix_side": "renquant-pipeline (unparseable file)"})
            continue
        try:
            mod = importlib.import_module(t["module"])
        except Exception as exc:  # noqa: BLE001
            failures.append({**t, "error": f"{type(exc).__name__}: {exc}",
                             "fix_side": "alias table (orchestrator) or the "
                                         "alias target repo must provide this module"})
            continue
        for name in t["names"]:
            if hasattr(mod, name):
                continue
            try:
                importlib.import_module(f"{t['module']}.{name}")
            except Exception as exc:  # noqa: BLE001
                failures.append({
                    "module": f"{t['module']}.{name}", "names": [],
                    "file": t["file"], "line": t["line"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "fix_side": "alias target repo is missing this "
                                "submodule/attribute the pinned pipeline imports",
                })
    return {"ok": not failures, "n_targets": len(targets),
            "aliased": aliased, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-file", type=Path, required=True)
    ap.add_argument("--siblings", type=Path, required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    result = sweep(args.lock_file.resolve(), args.siblings.resolve())
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"pin import-integrity sweep: {result.get('n_targets', 0)} "
              f"aliased-namespace import target(s) checked")
        for f in result["failures"]:
            loc = f" [{f.get('file')}:{f.get('line')}]" if f.get("file") else ""
            print(f"  FAIL {f.get('module') or ''}{loc}: "
                  f"{f.get('error') or f.get('syntax_error')}\n"
                  f"       fix side: {f.get('fix_side', 'unknown')}")
        print("RESULT: " + ("PASS" if result["ok"] else "FAIL"))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
