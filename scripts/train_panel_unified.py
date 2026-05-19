#!/usr/bin/env python3
"""Unified panel-LTR training — registry-driven for future extensibility.

Per 2026-05-18 user mandate: train + inference both swap by `kind` config.
Per 2026-05-18 update: extensible registry, easy to add future models
(LightGBM / CatBoost / NGBoost / new architectures).

The registry lives in:
  backtesting/renquant_104/kernel/panel_pipeline/model_registry.py

Currently registered: xgb, patchtst.
Add a new model: decorate with @registry.register("kind") (see registry docs).

Usage:
    python scripts/train_panel_unified.py --kind xgb \\
        --dataset data/alpha158_291_fundamental_dataset.parquet \\
        --output artifacts/panel-ltr.alpha158_fund.json

    python scripts/train_panel_unified.py --kind patchtst \\
        --dataset data/transformer_v4_wl200_clean.parquet \\
        --output-dir artifacts/patchtst_unified \\
        --num-seeds 5 --epochs 10

    python scripts/train_panel_unified.py --list   # show available kinds
"""
from __future__ import annotations
import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))

from kernel.panel_pipeline.model_registry import registry  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train-panel-unified")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=registry.list() + ["LIST"],
                   help=f"Model kind. Registered: {registry.list()}")
    p.add_argument("--list", action="store_true",
                   help="List registered model kinds and exit")
    p.add_argument("--dataset",
                   help="Panel parquet path")
    p.add_argument("--label", default="fwd_60d_excess",
                   choices=["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"])
    p.add_argument("--output", default=None,
                   help="(kind=xgb) output .json artifact path")
    p.add_argument("--output-dir", default=None,
                   help="(kind=patchtst) output dir for .pt + summary")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=5,
                   help="(kind=patchtst) seeds for variance")
    p.add_argument("--epochs", type=int, default=10,
                   help="(kind=patchtst) training epochs per seed")
    p.add_argument("--seq-len", type=int, default=32,
                   help="(kind=patchtst) sequence length")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu", "auto"],
                   help="(kind=patchtst) torch device")
    args = p.parse_args()

    if args.list:
        print(f"Registered model kinds: {registry.list()}")
        for k in registry.list():
            h = registry.get(k)
            print(f"  - {k}  requires_history={h.requires_history}")
        return 0

    if not args.kind or not args.dataset:
        p.error("--kind and --dataset are required (or use --list)")

    handler = registry.get(args.kind)
    cmd = handler.train_cmd(args)
    log.info("Dispatching %s: %s", args.kind, " ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
