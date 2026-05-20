#!/usr/bin/env python3
"""Compare 5-cut × 5-seed eval results across architectures.

Usage::

    .venv/bin/python scripts/compare_arch_5cut_5seed.py \\
        --runs artifacts/hf_trainer_5cut_5seed_pt07:hf_patchtst \\
               artifacts/dlinear_5cut_5seed:dlinear

Reads each run's aggregate.csv (produced by eval_*_5cut_5seed.py scripts),
produces:
  - per-regime × per-architecture mean IC matrix
  - min-regime IC across cuts (PRIME DIRECTIVE selection metric)
  - per-cut head-to-head
  - verdict: "architecture matters" iff min-regime IC of any arch > best
    other arch + 0.005 (§5.13.4a Tier 2 threshold)
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("compare-arch")


def load_run(path: Path, arch_label: str) -> pd.DataFrame:
    """Load an aggregate.csv produced by eval_*_5cut_5seed.py."""
    p = path / "aggregate.csv"
    if not p.exists():
        log.warning("[%s] aggregate.csv missing at %s — skipping", arch_label, p)
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["arch"] = arch_label
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="Pairs <dir>:<label>, e.g. "
                        "artifacts/hf_trainer_5cut_5seed_pt07:hf_patchtst")
    p.add_argument("--tier2-threshold", type=float, default=0.005,
                   help="min-regime IC gap to declare architecture matters")
    args = p.parse_args()

    parts = []
    for spec in args.runs:
        path_str, label = spec.split(":", 1)
        parts.append(load_run(REPO / path_str, label))

    if not any(len(p) > 0 for p in parts):
        log.error("no runs loaded — exit")
        sys.exit(1)

    df = pd.concat([p for p in parts if len(p) > 0], ignore_index=True)
    log.info("loaded %d rows across %d archs", len(df), df["arch"].nunique())

    # 1. Per-arch × per-regime mean IC
    log.info("\n=== ARCH × REGIME mean IC (across cuts × seeds) ===")
    pivot = df.groupby(["arch", "regime"])["ic"].mean().unstack()
    print(pivot.to_string(float_format="%+.4f"))

    # 2. Min-regime IC per arch (PRIME DIRECTIVE selection metric)
    log.info("\n=== ARCH min-regime IC (PRIME DIRECTIVE) ===")
    real_regimes = [r for r in df["regime"].unique() if r != "_MIN_"]
    arch_min_regime = (df[df["regime"].isin(real_regimes)]
                        .groupby(["arch", "cut", "seed"])["ic"].min()
                        .reset_index()
                        .groupby("arch")["ic"]
                        .agg(["mean", "std", "count"]))
    print(arch_min_regime.to_string(float_format="%+.4f"))

    # 3. Per-cut head-to-head
    log.info("\n=== PER-CUT min-regime IC head-to-head ===")
    per_cut_min = (df[df["regime"].isin(real_regimes)]
                    .groupby(["arch", "cut", "seed"])["ic"].min()
                    .reset_index()
                    .groupby(["arch", "cut"])["ic"]
                    .mean()
                    .unstack(level="arch"))
    print(per_cut_min.to_string(float_format="%+.4f"))

    # 4. Verdict
    log.info("\n=== VERDICT ===")
    arch_scores = arch_min_regime["mean"].sort_values(ascending=False)
    if len(arch_scores) >= 2:
        best, runnerup = arch_scores.index[0], arch_scores.index[1]
        gap = arch_scores.iloc[0] - arch_scores.iloc[1]
        log.info("Best: %s = %+.4f (min-regime IC mean)", best, arch_scores.iloc[0])
        log.info("Runner-up: %s = %+.4f", runnerup, arch_scores.iloc[1])
        log.info("Gap: %+.4f (Tier 2 threshold: %.4f)", gap, args.tier2_threshold)
        if gap >= args.tier2_threshold:
            log.info("→ ARCHITECTURE MATTERS — %s beats %s by ≥ %.4f",
                     best, runnerup, args.tier2_threshold)
            log.info("→ Recommendation: invest in T2.x architectural items")
        else:
            log.info("→ ARCHITECTURE NOT BOTTLENECK — gap %+.4f < %.4f threshold",
                     gap, args.tier2_threshold)
            log.info("→ Recommendation: invest in features / labels / sample size, "
                     "NOT more architecture (per §5.12 + 'Are Transformers Effective')")
    else:
        log.warning("need ≥2 archs to compare; got %d", len(arch_scores))


if __name__ == "__main__":
    main()
