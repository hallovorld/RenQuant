#!/usr/bin/env python3
"""HF Trainer + cross-stock attention 5-cut × 5-seed eval.

Same knobs as baseline / FiLM but adds --cross-stock-attn flag (Tier 2 T2.1
iTransformer-style variate-as-token attention across tickers, Liu 2024).

Embargo: walk-forward splitter has default embargo_days=60 (P0-1 fix
2026-05-20) — train sets are clean of 60-day forward-label leakage.
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
log = logging.getLogger("eval-hf-cross-stock")

CUTS = ["cut1_covid", "cut2_fed", "cut3_inflpk", "cut4_svb", "cut5_unwind"]
SEEDS = [42, 43, 44, 45, 46]

# Phase 2 DOE best-point knobs (pt_07)
PT07 = dict(lr="1e-4", weight_decay="0.3", seq_len="24", warmup_ratio="0.1")
EPOCHS = "8"
DEVICE = "mps"

import os as _os  # noqa: E402
_TAG = _os.environ.get("EVAL_OUT_TAG", "hf_cross_stock_5cut_5seed_pt07")
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
        "--warmup-ratio", PT07["warmup_ratio"],
        "--device", DEVICE,
        "--save-model",
        "--cross-stock-attn",  # Tier 2 T2.1 — iTransformer cross-stock attention
        "--output-dir", str(out_dir),
    ]
    log_path = LOG_ROOT / f"{cut}_seed{seed}.log"
    log.info("[CSA %s seed %d] START → %s", cut, seed, log_path.name)
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        log.warning("[CSA %s seed %d] FAILED rc=%d (see %s)",
                    cut, seed, result.returncode, log_path)
        return {"status": "fail", "cut": cut, "seed": seed}
    summary_path = out_dir / f"hf_patchtst_{cut}_seed{seed}_summary.json"
    if not summary_path.exists():
        return {"status": "no_summary", "cut": cut, "seed": seed}
    summary = json.loads(summary_path.read_text())
    log.info("[CSA %s seed %d] OK best_val_ic=%+.4f per_regime=%s",
             cut, seed, summary.get("best_val_ic", float("nan")),
             summary.get("per_regime_ic", {}))
    summary["status"] = "ok"
    return summary


def aggregate(results: list[dict]) -> None:
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
    log.info("\n=== CROSS-STOCK PER-CUT × PER-REGIME mean IC ===\n%s",
             df.groupby(["cut", "regime"])["ic"].mean().unstack().to_string())
    log.info("\n=== CROSS-STOCK PER-REGIME mean across cuts ===\n%s",
             df.groupby("regime")["ic"].agg(["mean", "std", "count"]).to_string())
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
