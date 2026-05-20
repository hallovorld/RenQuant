#!/usr/bin/env python3
"""HF Trainer + FiLM regime conditioning 5-cut × 5-seed eval.

Same knobs as eval_hf_trainer_5cut_5seed.py (Phase 2 DOE best pt_07:
lr=1e-4, wd=0.3, seq_len=24, epochs=8) but adds --film-regime-cond flag.

Run AFTER eval_hf_trainer_5cut_5seed.py completes. Compare via:

    .venv/bin/python scripts/compare_arch_5cut_5seed.py \\
        --runs artifacts/hf_trainer_5cut_5seed_pt07:patchtst_baseline \\
               artifacts/hf_film_5cut_5seed_pt07:patchtst_film
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
log = logging.getLogger("eval-hf-film")

CUTS = ["cut1_covid", "cut2_fed", "cut3_inflpk", "cut4_svb", "cut5_unwind"]
SEEDS = [42, 43, 44, 45, 46]

PT07 = dict(lr="1e-4", weight_decay="0.3", seq_len="24")
EPOCHS = "8"
DEVICE = "mps"

OUT_ROOT = REPO / "artifacts/hf_film_5cut_5seed_pt07"
LOG_ROOT = REPO / "logs/hf_film_5cut_5seed_pt07"


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
        "--film-regime-cond",  # Pillar B: FiLM regime conditioning ON
        "--output-dir", str(out_dir),
    ]
    log_path = LOG_ROOT / f"{cut}_seed{seed}.log"
    log.info("[FILM %s seed %d] START → %s", cut, seed, log_path.name)
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        log.warning("[FILM %s seed %d] FAILED rc=%d (see %s)",
                    cut, seed, result.returncode, log_path)
        return {"status": "fail", "cut": cut, "seed": seed}
    summary_path = out_dir / f"hf_patchtst_{cut}_seed{seed}_summary.json"
    if not summary_path.exists():
        return {"status": "no_summary", "cut": cut, "seed": seed}
    summary = json.loads(summary_path.read_text())
    log.info("[FILM %s seed %d] OK best_val_ic=%+.4f per_regime=%s",
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
    log.info("\n=== FILM PER-CUT × PER-REGIME mean IC ===\n%s",
             df.groupby(["cut", "regime"])["ic"].mean().unstack().to_string())
    log.info("\n=== FILM PER-REGIME mean across cuts ===\n%s",
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
