#!/usr/bin/env python
"""Retrain PROD XGB on data BEFORE --train-cutoff so we get a TRULY OOS test set.

The standard prod train uses ALL panel rows with valid fwd_60d label
(date ≤ panel_max - 60d). That model is then "eval'd" on the same range —
all in-sample. To estimate true OOS skill we need an earlier cutoff and
eval on dates strictly AFTER it.

Output: backtesting/renquant_104/artifacts/walkforward_truly_oos_2024-07-01/panel-ltr.json

This DOES NOT touch the live prod artifact. It writes to a side path used
ONLY by the companion eval script + tests/test_prod_signal_truly_oos.py.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_THREADS = str(min(os.cpu_count() or 14, 14))
os.environ.setdefault("OMP_NUM_THREADS", DEFAULT_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", DEFAULT_THREADS)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("retrain-truly-oos")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cutoff", default="2024-07-01",
                   help="Drop panel rows with date >= cutoff during training")
    p.add_argument("--output-dir",
                   default="backtesting/renquant_104/artifacts/walkforward_truly_oos_2024-07-01",
                   help="Side-path directory for truly-OOS artifacts")
    args = p.parse_args()

    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_artifact = out_dir / "panel-ltr.json"

    log.info("Retraining PROD XGB with --train-cutoff %s → %s",
             args.train_cutoff, out_artifact)

    cmd = [
        sys.executable, str(REPO / "scripts/train_production_model.py"),
        "--train-cutoff", args.train_cutoff,
        "--output-path", str(out_artifact),
        "--side-label", f"truly_oos_{args.train_cutoff}",
    ]
    log.info("CMD: %s", " ".join(cmd))
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        log.error("Training failed (rc=%d)", r.returncode)
        return r.returncode

    if not out_artifact.exists():
        log.error("Training succeeded but artifact missing: %s", out_artifact)
        return 1
    log.info("✓ Wrote %s (%d bytes)", out_artifact, out_artifact.stat().st_size)

    # Stamp the cutoff into the artifact metadata for downstream eval
    art = json.loads(out_artifact.read_text())
    art["_truly_oos_train_cutoff"] = args.train_cutoff
    out_artifact.write_text(json.dumps(art))
    log.info("✓ Stamped _truly_oos_train_cutoff=%s into artifact", args.train_cutoff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
