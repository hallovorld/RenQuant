#!/usr/bin/env python
"""Multi-repo daily runner — run the REAL daily through the pinned subrepos.

Goal (2026-05-27): execute the production daily (`live.runner`) but route every
LIFTED kernel module to the pinned `renquant-pipeline` subrepo, so the decision
tree genuinely runs out of the multi-repo packages — while the umbrella RenQuant
stays the untouched baseline/rollback (copy-not-move).

Bridge state: 3 modules are not yet homed into a subrepo
(`kernel.preflight`, `kernel.panel_pipeline`, `kernel.meta_label` — the model /
strategy boundary, slated for renquant-model). Those resolve from the umbrella's
own `kernel/` for now. Everything else (regime, exits, sizing, QP, selection,
rotation, the whole pipeline + jobs/tasks, decision_trace, persistence, models,
execution) comes from the pinned subrepo.

Usage (safe default = readonly-alpaca, no orders):
    python scripts/daily_multirepo.py --broker readonly-alpaca --once
For the live full run (real account, same as the umbrella daily):
    python scripts/daily_multirepo.py --broker alpaca --once
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SIBLINGS = REPO.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"

# Pinned subrepo source roots (must match subrepos.lock.json local_paths).
_PIN_SRCS = [
    "renquant-common", "renquant-base-data", "renquant-artifacts",
    "renquant-strategy-104", "renquant-model", "renquant-pipeline",
    "renquant-execution", "renquant-backtesting",
]


def _bootstrap_multirepo() -> list[str]:
    """Put pinned subrepos on path and alias every lifted kernel.* module to the
    pin. Returns the list of aliased module names (for the run report)."""
    # umbrella + strategy dir first so non-lifted kernel.* (preflight/
    # panel_pipeline/meta_label) and `live`/`adapters` still resolve locally.
    for p in (str(REPO), str(STRATEGY_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for name in _PIN_SRCS:
        src = SIBLINGS / name / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.append(str(src))

    pk = importlib.import_module("renquant_pipeline.kernel")
    pk_dir = Path(pk.__file__).resolve().parent

    aliased: list[str] = []
    # Top-level lifted kernel modules/packages (everything the pin carries).
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
            continue  # leaf with heavy optional deps — skip; umbrella fallback
        sys.modules[modname] = mod
        aliased.append(modname)

    # The pin's lifted code (e.g. pp_inference) rewrote model-boundary imports to
    # `renquant_pipeline.kernel.{meta_label,panel_pipeline}`, but those are NOT
    # homed in the pin yet (model boundary → renquant-model, pending). Bridge
    # them to the umbrella's own modules so the pipeline runs end-to-end.
    for modname in ("meta_label", "panel_pipeline"):
        try:
            um = importlib.import_module(f"kernel.{modname}")
        except Exception:
            continue
        sys.modules[f"renquant_pipeline.kernel.{modname}"] = um
        aliased.append(f"renquant_pipeline.kernel.{modname}←umbrella")

    # pp_inference does `from renquant_pipeline.panel_scoring import PanelScoringJob`,
    # but the pin's panel_scoring.py is a *consolidated rewrite* of the model
    # boundary that (unlike the umbrella) does NOT fail-close on an unfingerprinted
    # artifact — a parity gap AND a safety divergence (§5.13.15). Route it to the
    # umbrella's proven PanelScoringJob until the model boundary is homed into
    # renquant-model. The fail-closed gate is a pure artifact-metadata check, so the
    # decision is identical regardless of the (pin-provided) feature/data source.
    try:
        sys.modules["renquant_pipeline.panel_scoring"] = importlib.import_module(
            "kernel.panel_pipeline.job_panel_scoring")
        aliased.append("renquant_pipeline.panel_scoring←umbrella.job_panel_scoring")
    except Exception:
        pass
    return aliased


def main() -> int:
    aliased = _bootstrap_multirepo()
    sys.stderr.write(
        f"[multirepo] routed {len(aliased)} kernel modules to renquant-pipeline; "
        f"preflight/panel_pipeline/meta_label resolve from umbrella (bridge).\n"
    )
    # Hand off to the real production runner with the original CLI args.
    if "--strategy" not in sys.argv:
        sys.argv = [sys.argv[0], "--strategy", "renquant_104"] + sys.argv[1:]
    runner = importlib.import_module("live.runner")
    return int(runner.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
