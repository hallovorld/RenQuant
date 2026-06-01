#!/usr/bin/env python
"""Multi-repo daily runner — run the REAL daily through the pinned subrepos.

Goal (2026-05-27): execute the production daily (`live.runner`) but route every
LIFTED kernel module to the pinned `renquant-pipeline` subrepo, so the decision
tree genuinely runs out of the multi-repo packages — while the umbrella RenQuant
stays the untouched baseline/rollback (copy-not-move).

Bridge state (2026-06-01):
  * `kernel.preflight`     → renquant-pipeline.kernel.preflight (lifted)
  * `kernel.panel_pipeline`→ renquant-pipeline.kernel.panel_pipeline (lifted)
  * `renquant_pipeline.panel_scoring`
                            → renquant-pipeline.kernel.panel_pipeline.job_panel_scoring
  * `kernel.meta_label`    → renquant-backtesting.meta_label (lifted; bridged here)
  * everything else        → renquant-pipeline (already lifted)
The umbrella `kernel/` retains all of these as byte-equivalent copies for
rollback (RQ_DAILY_RUNNER=umbrella) and for any code still doing
`from kernel.x import …` outside the bootstrap.

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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from subrepo_pin_guard import enforce_or_warn, resolve_subrepo_src_roots
from subrepo_pin_guard import strict_clean_enabled

REPO = Path(__file__).resolve().parent.parent
SIBLINGS = REPO.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
LOCK_FILE = REPO / "subrepos.lock.json"

# Pinned subrepo source roots (must match subrepos.lock.json local_paths).
_PIN_SRCS = [
    "renquant-common", "renquant-base-data", "renquant-artifacts",
    "renquant-strategy-104", "renquant-model", "renquant-pipeline",
    "renquant-execution", "renquant-backtesting",
]


def _bootstrap_multirepo() -> list[str]:
    """Put pinned subrepos on path and alias every lifted kernel.* module to the
    pin. Returns the list of aliased module names (for the run report)."""
    # umbrella + strategy dir first so `live`/`adapters` and explicit rollback
    # paths still resolve locally; lifted kernel.* modules are overwritten below.
    for p in (str(REPO), str(STRATEGY_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    src_roots, pin_issues = resolve_subrepo_src_roots(
        lock_file=LOCK_FILE,
        names=_PIN_SRCS,
        siblings=SIBLINGS,
        root_override=os.environ.get("RENQUANT_SUBREPO_ROOT"),
        check_dirty=strict_clean_enabled(),
    )
    enforce_or_warn(pin_issues)
    for src in src_roots:
        if str(src) not in sys.path:
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

    # 2026-05-30: panel_pipeline + preflight are now homed in the pin (C2.9 + C2.11).
    # The pin loop above already aliases them. meta_label was lifted to
    # renquant-backtesting (C2.4) — alias that here so the pin's pp_inference path
    # `from renquant_pipeline.kernel.meta_label import …` still resolves.
    try:
        bm = importlib.import_module("renquant_backtesting.meta_label")
        sys.modules["renquant_pipeline.kernel.meta_label"] = bm
        aliased.append("renquant_pipeline.kernel.meta_label←renquant_backtesting")
    except Exception:
        # Fallback to umbrella if backtesting subrepo not on path
        try:
            um = importlib.import_module("kernel.meta_label")
            sys.modules["renquant_pipeline.kernel.meta_label"] = um
            aliased.append("renquant_pipeline.kernel.meta_label←umbrella")
        except Exception:
            pass

    # pp_inference imports `renquant_pipeline.panel_scoring.PanelScoringJob`.
    # Production daily needs the byte-equivalent fail-closed scorer job lifted
    # under renquant-pipeline.kernel.panel_pipeline, not the experimental
    # load_scorer rewrite exposed at renquant_pipeline.panel_scoring.
    try:
        sys.modules["renquant_pipeline.panel_scoring"] = importlib.import_module(
            "renquant_pipeline.kernel.panel_pipeline.job_panel_scoring")
        aliased.append(
            "renquant_pipeline.panel_scoring←renquant_pipeline.kernel.panel_pipeline.job_panel_scoring")
    except Exception:
        pass
    return aliased


def main() -> int:
    aliased = _bootstrap_multirepo()
    sys.stderr.write(
        f"[multirepo] routed {len(aliased)} kernel modules to renquant-pipeline; "
        "preflight/panel_pipeline/panel_scoring resolve from pinned subrepos; "
        "meta_label resolves from renquant-backtesting when available.\n"
    )
    # Hand off to the real production runner with the original CLI args.
    if "--strategy" not in sys.argv:
        sys.argv = [sys.argv[0], "--strategy", "renquant_104"] + sys.argv[1:]
    runner = importlib.import_module("live.runner")
    return int(runner.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
