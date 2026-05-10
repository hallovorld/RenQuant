#!/usr/bin/env python
"""One-shot fix for side configs that reference production artifact paths.

Audit fix (2026-05-09): historical side configs (strategy_config.*.json
that aren't the active production or golden) reference production
artifact paths. If accidentally invoked, they'd overwrite production.

This script rewrites every artifact_path in side configs from the
production default (e.g. `artifacts/panel-ltr.json`) to a side-aliased
path containing the config's label (e.g.
`artifacts/panel-ltr.golden.previous_2026-05-08.json`).

These aliased artifact files don't exist on disk — invoking the side
config will FAIL FAST at load time rather than silently clobber.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"

PRODUCTION_DEFAULTS = {
    "artifacts/panel-ltr.json": "panel-ltr",
    "artifacts/ngboost-head.json": "ngboost-head",
    "artifacts/panel-rank-calibration.json": "panel-rank-calibration",
}


def _label_from_filename(fname: str) -> str:
    """`strategy_config.golden.previous_2026-05-08.json` → `golden.previous_2026-05-08`."""
    return fname.replace("strategy_config.", "").replace(".json", "")


def _rewrite_paths(obj, label: str, count: list):
    """Recursively rewrite every value at any 'artifact_path' key that
    matches a production default."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "artifact_path" and isinstance(v, str) and v in PRODUCTION_DEFAULTS:
                base = PRODUCTION_DEFAULTS[v]
                obj[k] = f"artifacts/{base}.{label}.json"
                count[0] += 1
            elif isinstance(v, (dict, list)):
                _rewrite_paths(v, label, count)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                _rewrite_paths(v, label, count)


def main() -> None:
    side_files = sorted(STRATEGY_DIR.glob("strategy_config.*.json"))
    # Filter to side configs only — exclude `strategy_config.json` (it's
    # `strategy_config.json`, no `.<label>.json` suffix) and `strategy_config.golden.json`.
    skip = {"strategy_config.json", "strategy_config.golden.json"}
    side_files = [f for f in side_files if f.name not in skip]

    total_rewrites = 0
    for f in side_files:
        label = _label_from_filename(f.name)
        cfg = json.loads(f.read_text())
        count = [0]
        _rewrite_paths(cfg, label, count)
        if count[0] > 0:
            f.write_text(json.dumps(cfg, indent=2))
            print(f"  {f.name}: rewrote {count[0]} artifact_path entries → ...{label}...")
            total_rewrites += count[0]
        else:
            print(f"  {f.name}: clean (no production paths)")

    print(f"\nTotal: {total_rewrites} entries rewritten across {len(side_files)} files.")


if __name__ == "__main__":
    main()
