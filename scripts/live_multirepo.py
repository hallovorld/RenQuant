#!/usr/bin/env python
"""Multi-repo live runner bridge.

Runs the real ``live.runner`` while routing lifted RenQuant modules through
``subrepos.lock.json`` local paths. This is the shared entry point for
production daily, intraday sell-only, and readonly shadow runs; set
``RQ_DAILY_RUNNER=umbrella`` in the shell wrapper to bypass this bridge and use
the umbrella baseline.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SIBLINGS = REPO.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
LOCK_FILE = REPO / "subrepos.lock.json"

# Pinned subrepo source roots. Names resolve through subrepos.lock.json
# local_path entries, RENQUANT_SUBREPO_ROOT, then SIBLINGS/name fallback.
_PIN_SRCS = [
    "renquant-common",
    "renquant-base-data",
    "renquant-artifacts",
    "renquant-strategy-104",
    "renquant-model",
    "renquant-pipeline",
    "renquant-execution",
    "renquant-backtesting",
]


def _subrepo_src_roots() -> tuple[list[Path], list[str]]:
    lock_entries: dict[str, dict] = {}
    if LOCK_FILE.exists():
        payload = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        lock_entries = {str(e["name"]): e for e in payload.get("subrepos", [])}

    roots: list[Path] = []
    missing: list[str] = []
    root_override = os.environ.get("RENQUANT_SUBREPO_ROOT")
    for name in _PIN_SRCS:
        candidates: list[Path] = []
        if root_override:
            candidates.append(Path(root_override) / name)
        if name in lock_entries and lock_entries[name].get("local_path"):
            candidates.append(Path(str(lock_entries[name]["local_path"])))
        candidates.append(SIBLINGS / name)

        src = next(
            (candidate / "src" for candidate in candidates if (candidate / "src").is_dir()),
            None,
        )
        if src is None:
            missing.append(name)
            continue
        roots.append(src)
    return roots, missing


def _bootstrap_multirepo() -> list[str]:
    """Put sibling subrepos on sys.path and alias lifted kernel modules."""
    for path in (str(REPO), str(STRATEGY_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    src_roots, missing = _subrepo_src_roots()
    if missing:
        sys.stderr.write(f"[multirepo] missing subrepo src roots: {', '.join(missing)}\n")
        if os.environ.get("RENQUANT_STRICT_SUBREPO_PATHS") == "1":
            raise SystemExit(2)
    for src in src_roots:
        if str(src) not in sys.path:
            sys.path.append(str(src))

    pk = importlib.import_module("renquant_pipeline.kernel")
    pk_dir = Path(pk.__file__).resolve().parent

    aliased: list[str] = []
    for entry in sorted(pk_dir.iterdir()):
        stem = entry.stem if entry.suffix == ".py" else entry.name
        if stem in {"__init__", "__pycache__"} or stem.startswith("."):
            continue
        if entry.suffix not in {".py", ""}:
            continue
        modname = f"kernel.{stem}"
        try:
            mod = importlib.import_module(f"renquant_pipeline.kernel.{stem}")
        except Exception:
            continue
        sys.modules[modname] = mod
        aliased.append(modname)

    try:
        bm = importlib.import_module("renquant_backtesting.meta_label")
        sys.modules["renquant_pipeline.kernel.meta_label"] = bm
        aliased.append("renquant_pipeline.kernel.meta_label<-renquant_backtesting")
    except Exception:
        try:
            um = importlib.import_module("kernel.meta_label")
            sys.modules["renquant_pipeline.kernel.meta_label"] = um
            aliased.append("renquant_pipeline.kernel.meta_label<-umbrella")
        except Exception:
            pass

    # Keep the proven fail-closed panel scoring path from the lifted
    # renquant-pipeline kernel package until the load_scorer rewrite passes
    # production parity.
    try:
        sys.modules["renquant_pipeline.panel_scoring"] = importlib.import_module(
            "renquant_pipeline.kernel.panel_pipeline.job_panel_scoring"
        )
        aliased.append(
            "renquant_pipeline.panel_scoring<-renquant_pipeline.kernel.panel_pipeline.job_panel_scoring")
    except Exception:
        pass
    return aliased


def main() -> int:
    aliased = _bootstrap_multirepo()
    sys.stderr.write(
        f"[multirepo] routed {len(aliased)} lifted modules through sibling subrepos; "
        "live.runner remains the execution handoff.\n"
    )
    if "--strategy" not in sys.argv:
        sys.argv = [sys.argv[0], "--strategy", "renquant_104"] + sys.argv[1:]
    runner = importlib.import_module("live.runner")
    return int(runner.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
