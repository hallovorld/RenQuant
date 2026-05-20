#!/usr/bin/env python3
"""DLinear 5-cut × 5-seed eval — apples-to-apples vs HF Trainer PatchTST.

Same protocol as eval_hf_trainer_5cut_5seed.py but DLinear backbone.
CPU is fine (DLinear ~200 params); runs alongside PatchTST MPS eval
without GPU contention.

Knobs:
  seq_len=24, kernel_size=5 (≈ seq_len/5, sensible decompose),
  lr=1e-3 (linear can take higher LR), epochs=8.
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
log = logging.getLogger("eval-dlinear")

CUTS = ["cut1_covid", "cut2_fed", "cut3_inflpk", "cut4_svb", "cut5_unwind"]
SEEDS = [42, 43, 44, 45, 46]

KNOBS = dict(seq_len="24", kernel_size="5", lr="1e-3", weight_decay="1e-3")
EPOCHS = "8"
DEVICE = "cpu"

OUT_ROOT = REPO / "artifacts/dlinear_5cut_5seed"
LOG_ROOT = REPO / "logs/dlinear_5cut_5seed"


def run_one(cut: str, seed: int) -> dict:
    out_dir = OUT_ROOT / cut / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "scripts/dlinear_baseline.py"),
        "--cut", cut, "--seed", str(seed),
        "--epochs", EPOCHS,
        "--seq-len", KNOBS["seq_len"],
        "--kernel-size", KNOBS["kernel_size"],
        "--lr", KNOBS["lr"],
        "--weight-decay", KNOBS["weight_decay"],
        "--device", DEVICE,
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
    summary_path = out_dir / f"dlinear_{cut}_seed{seed}_summary.json"
    if not summary_path.exists():
        return {"status": "no_summary", "cut": cut, "seed": seed}
    summary = json.loads(summary_path.read_text())
    log.info("[%s seed %d] OK best_val_ic=%+.4f per_regime=%s",
             cut, seed, summary.get("best_val_ic", float("nan")),
             summary.get("per_regime_ic", {}))
    summary["status"] = "ok"
    return summary


def aggregate(results: list[dict]) -> None:
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
    log.info("\n=== DLINEAR PER-CUT × PER-REGIME mean IC ===")
    pivot = df.groupby(["cut", "regime"])["ic"].mean().unstack()
    log.info("\n" + pivot.to_string())
    log.info("\n=== DLINEAR PER-REGIME mean across cuts ===")
    log.info("\n" + df.groupby("regime")["ic"].agg(["mean", "std", "count"]).to_string())
    df.to_csv(OUT_ROOT / "aggregate.csv", index=False)


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
