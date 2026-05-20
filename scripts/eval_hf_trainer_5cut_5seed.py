#!/usr/bin/env python3
"""5-cut × 5-seed eval of new HF Trainer + multi-task head patchtst_hf.py.

Knobs match Phase 2 DOE best point (pt_07): lr=1e-4, wd=0.3, seq_len=24.
Goal: validate that the 2026-05-19 HF Trainer refactor delivers ≥ current
shadow IC, and verify per-regime selection + distributional head work
end-to-end across all 5 walk-forward cuts × 5 seeds.

After completion, prints per-cut per-regime IC table + aggregates.
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval-5cut-5seed")

CUTS = ["cut1_covid", "cut2_fed", "cut3_inflpk", "cut4_svb", "cut5_unwind"]
SEEDS = [42, 43, 44, 45, 46]

# Phase 2 DOE best-point knobs (pt_07: lr=1e-4, wd=0.3, seq_len=24, warmup=10)
PT07 = dict(lr="1e-4", weight_decay="0.3", seq_len="24")
EPOCHS = "8"
DEVICE = "mps"

import os as _os  # noqa: E402
# Env-overridable output paths (so 3-way orchestrator can run separate
# embargo-clean variants without text-replace hacks)
_TAG = _os.environ.get("EVAL_OUT_TAG", "hf_trainer_5cut_5seed_pt07")
OUT_ROOT = REPO / f"artifacts/{_TAG}"
LOG_ROOT = REPO / f"logs/{_TAG}"


def run_one(cut: str, seed: int) -> dict:
    out_dir = OUT_ROOT / cut / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "scripts/patchtst_hf.py"),
        "--cut", cut, "--seed", str(seed),
        "--epochs", EPOCHS,
        "--seq-len", PT07["seq_len"],
        "--lr", PT07["lr"],
        "--weight-decay", PT07["weight_decay"],
        "--device", DEVICE,
        "--save-model",
        "--output-dir", str(out_dir),
    ]
    log_path = LOG_ROOT / f"{cut}_seed{seed}.log"
    log.info("[%s seed %d] START → %s", cut, seed, log_path.name)
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        log.warning("[%s seed %d] FAILED rc=%d (see %s)",
                    cut, seed, result.returncode, log_path)
        return {"status": "fail", "cut": cut, "seed": seed}
    summary_path = out_dir / f"hf_patchtst_{cut}_seed{seed}_summary.json"
    if not summary_path.exists():
        return {"status": "no_summary", "cut": cut, "seed": seed}
    summary = json.loads(summary_path.read_text())
    log.info("[%s seed %d] OK best_val_ic=%+.4f per_regime=%s",
             cut, seed, summary.get("best_val_ic", float("nan")),
             summary.get("per_regime_ic", {}))
    summary["status"] = "ok"
    return summary


def aggregate(results: list[dict]) -> None:
    """Print per-cut and overall aggregate of per-regime IC."""
    import numpy as np
    import pandas as pd

    rows = []
    for r in results:
        if r.get("status") != "ok":
            continue
        for regime, ic in r.get("per_regime_ic", {}).items():
            rows.append({"cut": r["cut"], "seed": r["seed"],
                          "regime": regime, "ic": ic})
        rows.append({"cut": r["cut"], "seed": r["seed"],
                      "regime": "_MIN_", "ic": r.get("best_val_ic", float("nan"))})

    if not rows:
        log.warning("no successful runs to aggregate")
        return

    df = pd.DataFrame(rows)
    log.info("\n=== PER-CUT × PER-REGIME mean IC ===")
    pivot = df.groupby(["cut", "regime"])["ic"].mean().unstack()
    log.info("\n" + pivot.to_string())

    log.info("\n=== PER-REGIME mean across cuts ===")
    agg = df.groupby("regime")["ic"].agg(["mean", "std", "count"])
    log.info("\n" + agg.to_string())

    out_path = OUT_ROOT / "aggregate.csv"
    df.to_csv(out_path, index=False)
    log.info("aggregate dumped: %s", out_path)


def main():
    results = []
    n_total = len(CUTS) * len(SEEDS)
    n_done = 0
    for cut in CUTS:
        for seed in SEEDS:
            r = run_one(cut, seed)
            results.append(r)
            n_done += 1
            log.info("PROGRESS %d/%d", n_done, n_total)
    log.info("ALL DONE — aggregating...")
    aggregate(results)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
