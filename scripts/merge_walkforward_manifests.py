#!/usr/bin/env python
"""Merge extended walkforward manifest into existing sim manifest.

After running train_walkforward_panel.py with an earlier date range
(e.g., 2022-01-01 → 2023-12-31), the artifacts land under
``artifacts/walkforward_v2/<cutoff>/`` while the existing sim manifest
points to ``artifacts/sim/walkforward_retrains/<cutoff>/``. Both are
valid pre-trained heads; we need a unified manifest spanning both
ranges for an extended OOS panel.

This script:
  1. Reads existing manifest (default: artifacts/sim/walkforward_manifest.json)
  2. Reads extended manifest (default: artifacts/walkforward_manifest_extended.json)
  3. Concatenates the retrain entries (sorted by cutoff_date, dedup)
  4. Verifies each artifact_uri exists on disk
  5. Writes merged manifest

Output:  artifacts/sim/walkforward_manifest_merged.json (or --out)

Usage:
    python scripts/merge_walkforward_manifests.py
    python scripts/merge_walkforward_manifests.py --out custom.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("merge")

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--existing", default=str(STRATEGY_DIR / "artifacts" / "sim" / "walkforward_manifest.json"))
    p.add_argument("--extended", default=str(STRATEGY_DIR / "artifacts" / "walkforward_manifest_extended.json"))
    p.add_argument("--out", default=str(STRATEGY_DIR / "artifacts" / "sim" / "walkforward_manifest_merged.json"))
    args = p.parse_args()

    existing = json.loads(Path(args.existing).read_text())
    extended = json.loads(Path(args.extended).read_text())

    log.info(f"Existing manifest: {len(existing['retrains'])} retrains")
    log.info(f"Extended manifest: {len(extended['retrains'])} retrains")

    merged_retrains = existing["retrains"] + extended["retrains"]
    # Dedupe by cutoff_date (keep first encountered = existing wins)
    seen = set()
    dedup = []
    for r in merged_retrains:
        cd = r["cutoff_date"]
        if cd in seen:
            log.warning(f"Duplicate cutoff_date {cd} — skipping")
            continue
        seen.add(cd)
        dedup.append(r)
    dedup.sort(key=lambda r: r["cutoff_date"])

    # Verify each artifact exists
    missing = []
    for r in dedup:
        uri = Path(r["artifact_uri"])
        if not uri.exists():
            missing.append(str(uri))
    if missing:
        log.error(f"Missing {len(missing)}/{len(dedup)} artifacts:")
        for m in missing[:5]:
            log.error(f"  {m}")
        raise SystemExit(1)

    merged = dict(existing)  # carry meta fields like cadence_days
    merged["retrains"] = dedup
    merged["_merge_note"] = (
        f"Merged {len(existing['retrains'])} existing + "
        f"{len(extended['retrains'])} extended (2022-2023) → "
        f"{len(dedup)} unique retrains"
    )

    Path(args.out).write_text(json.dumps(merged, indent=2))
    log.info(f"\n✓ Merged manifest: {len(dedup)} retrains")
    log.info(f"  Range: {dedup[0]['cutoff_date']} → {dedup[-1]['cutoff_date']}")
    log.info(f"  Output: {args.out}")


if __name__ == "__main__":
    main()
