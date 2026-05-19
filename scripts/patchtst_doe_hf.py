#!/usr/bin/env python3
"""HF PatchTST DOE — pyDOE2 FrFact × walk-forward × multi-seed ensemble.

Replaces scripts/patchtst_doe_sweep.py (custom-impl) per 2026-05-18
user mandate "尽量用第三方lib". Uses scripts/patchtst_hf.py as the
per-trial trainer (HF transformers backbone).

PRIME DIRECTIVE compliance:
  - Objective: bull_regime_IC (mean of {BULL_CALM, BULL_VOLATILE} via
    internal HMM, NOT pooled mean or SPY-9-grid)
  - Walk-forward N cuts each covering different regime context (incl
    SPIKED periods missing from prior 2023-only val)
  - N seeds per design point × cut for ensemble (predict-averaging)

Design (CLAUDE.md §5.14.1 stage table — Stage 1 screening):
  - pyDOE2 FrFact 2^(4-1) Resolution IV + 1 center = 9 points
  - Knob ranges tightened from prior custom-DOE main effects:
      lr            ∈ [1e-5, 1e-4]   (prior best 1e-4 on low edge)
      weight_decay  ∈ [1e-2, 3e-1]   (prior pt_6 won at high wd=1e-1)
      warmup_epochs ∈ [4, 10]        (prior pt_6 won at long warmup=6)
      seq_len       ∈ [8, 24]        (prior best seq=16)

References:
  - Box-Hunter-Hunter 2005 *Statistics for Experimenters* ch.6
  - Lakshminarayanan 2017 NIPS (deep ensembles)
  - Kelly-Gu-Xiu 2020 RFS (CSRankNorm already in patchtst_hf.py loader)
  - Kim 2021 ICLR (RevIN; HF default scaling=std handles this)

Usage::

    nohup caffeinate -i .venv/bin/python scripts/patchtst_doe_hf.py \\
        --n-seeds 3 --epochs 4 --device mps \\
        --cuts cut1_covid,cut3_inflpk,cut5_unwind \\
        > logs/patchtst_doe_hf/doe_$(date +%Y%m%d-%H%M%S).log 2>&1 &
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pyDOE2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kernel.walk_forward_splits import build_default_cuts
from kernel.hmm_regime_labels import (compute_hmm_regime_labels,
                                        per_hmm_regime_ic, bull_regime_ic)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("patchtst-doe-hf")


# Tightened knob ranges from prior custom-impl DOE main effects
KNOBS = [
    # (name, low, high, log_scale)
    ("lr",            1e-5, 1e-4, True),
    ("weight_decay",  1e-2, 3e-1, True),
    ("warmup_epochs", 4,    10,   False),
    ("seq_len",       8,    24,   False),
]


def coded_to_real(name: str, coded: float) -> float:
    spec = next(k for k in KNOBS if k[0] == name)
    _, low, high, log_scale = spec
    if log_scale:
        log_low, log_high = np.log10(low), np.log10(high)
        log_mid = (log_low + log_high) / 2.0
        log_half = (log_high - log_low) / 2.0
        return float(10.0 ** (log_mid + coded * log_half))
    mid = (low + high) / 2.0
    half = (high - low) / 2.0
    val = mid + coded * half
    if name in ("warmup_epochs", "seq_len"):
        return int(round(val))
    return float(val)


def build_design_matrix() -> pd.DataFrame:
    corners = pyDOE2.fracfact("a b c abc")  # 2^(4-1) Res IV, 8 corners
    center = np.zeros((1, 4))
    coded = np.vstack([corners, center])
    rows = []
    for i, row in enumerate(coded):
        rows.append({
            "point_id": i,
            "is_center": bool(np.all(row == 0)),
            "lr_coded":            float(row[0]),
            "weight_decay_coded":  float(row[1]),
            "warmup_epochs_coded": float(row[2]),
            "seq_len_coded":       float(row[3]),
            "lr":            coded_to_real("lr", row[0]),
            "weight_decay":  coded_to_real("weight_decay", row[1]),
            "warmup_epochs": coded_to_real("warmup_epochs", row[2]),
            "seq_len":       coded_to_real("seq_len", row[3]),
        })
    return pd.DataFrame(rows)


def _train_one_cut_seed(point_id: int, cut: str, seed: int,
                          lr: float, weight_decay: float,
                          warmup_epochs: int, seq_len: int,
                          dataset: str, epochs: int, label: str,
                          device: str, out_root: Path) -> dict:
    """Run one HF training (single cut × single seed). Returns dict with
    seed_id, val_preds_path, status."""
    out_dir = out_root / f"pt_{point_id:02d}_{cut}_seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "scripts/patchtst_hf.py"),
        "--dataset", dataset,
        "--cut", cut,
        "--label", label,
        "--seq-len", str(int(seq_len)),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--lr", str(lr),
        "--weight-decay", str(weight_decay),
        "--output-dir", str(out_dir),
        "--device", device,
    ]
    log.info("[pt %02d %s seed %d] start lr=%.1e wd=%.1e warmup=%d seq=%d",
             point_id, cut, seed, lr, weight_decay, warmup_epochs, seq_len)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-400:]
        log.warning("[pt %02d %s seed %d] FAILED rc=%d tail=%s",
                    point_id, cut, seed, result.returncode, tail)
        return {"status": "fail"}
    val_preds = out_dir / f"hf_patchtst_{cut}_seed{seed}_val_preds.parquet"
    if not val_preds.exists():
        log.warning("[pt %02d %s seed %d] no val_preds output", point_id, cut, seed)
        return {"status": "missing"}
    return {"status": "ok", "val_preds_path": str(val_preds), "seed": seed}


def _ensemble_per_cut_per_regime(seed_results: list[dict],
                                   hmm_labels: pd.DataFrame
                                   ) -> tuple[dict[str, float], float]:
    """Average predictions across seeds (Lakshminarayanan 2017), then
    compute per-HMM-regime IC + bull_regime_ic on the ensembled preds."""
    ok = [r for r in seed_results if r["status"] == "ok"]
    if not ok:
        return {}, float("nan")
    # Load all seed preds, take mean of `pred` over seeds (join on date+label)
    dfs = [pd.read_parquet(r["val_preds_path"]) for r in ok]
    # Append seed index and groupby (date,label) → mean pred
    for i, d in enumerate(dfs):
        d["seed_idx"] = i
    combo = pd.concat(dfs, ignore_index=True)
    combo["date"] = pd.to_datetime(combo["date"])
    # Ensemble: average pred across seeds for each (date, label) row
    # Since rows align row-wise (same val period, same data order), the
    # simplest assumption is positional alignment.
    n_rows = len(dfs[0])
    if not all(len(d) == n_rows for d in dfs):
        log.warning("seed val_preds row count mismatch; using first only")
        ensembled = dfs[0][["date", "pred", "label"]].copy()
    else:
        ens_pred = np.mean([d["pred"].values for d in dfs], axis=0)
        ensembled = pd.DataFrame({
            "date": dfs[0]["date"].values,
            "pred": ens_pred,
            "label": dfs[0]["label"].values,
        })
    per_regime = per_hmm_regime_ic(ensembled, hmm_labels,
                                    min_samples_per_day=5,
                                    min_days_per_regime=5)
    bull_ic = bull_regime_ic(per_regime)
    return per_regime, bull_ic


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/transformer_v4_wl200_clean.parquet")
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--cuts", default="cut1_covid,cut3_inflpk,cut5_unwind",
                   help="Comma-separated walk-forward cut names")
    p.add_argument("--n-seeds", type=int, default=3,
                   help="Seeds per (point × cut), predict-averaged")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--device", default="mps",
                   choices=["cpu", "mps", "cuda"])
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (CPU only; MPS forces 1 — GPU contention)")
    p.add_argument("--out-root", default="artifacts/patchtst_doe_hf")
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="1 point × 1 cut × 1 seed")
    args = p.parse_args()

    out_root = REPO / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    design = build_design_matrix()
    design.to_csv(out_root / "design.csv", index=False)

    cut_names = args.cuts.split(",")
    all_cuts = {c.name: c for c in build_default_cuts()}
    for c in cut_names:
        if c not in all_cuts:
            raise SystemExit(f"unknown cut {c} — available: {list(all_cuts)}")

    log.info("Design: %d points × %d cuts × %d seeds = %d trainings",
             len(design), len(cut_names), args.n_seeds,
             len(design) * len(cut_names) * args.n_seeds)

    if args.smoke:
        design = design.head(1)
        cut_names = cut_names[:1]
        args_seeds = 1
    else:
        args_seeds = args.n_seeds

    # Pre-compute HMM labels once (covers all val periods)
    hmm_labels = compute_hmm_regime_labels(REPO / "data/ohlcv/SPY/1d.parquet")
    log.info("HMM regime labels: %d dates, regimes: %s",
             len(hmm_labels), sorted(hmm_labels["regime"].unique()))

    # Build job list
    jobs = []
    for _, row in design.iterrows():
        for cut in cut_names:
            for s in range(args_seeds):
                jobs.append({
                    "point_id": int(row["point_id"]),
                    "cut": cut,
                    "seed": args.seed_base + s,
                    "lr": float(row["lr"]),
                    "weight_decay": float(row["weight_decay"]),
                    "warmup_epochs": int(row["warmup_epochs"]),
                    "seq_len": int(row["seq_len"]),
                })

    # MPS = GPU = serial only; CPU can parallelize
    workers = args.workers
    if args.device == "mps" and workers > 1:
        log.warning("device=mps + workers=%d → forcing workers=1 (GPU contention)",
                    workers)
        workers = 1
    log.info("Dispatching %d jobs (workers=%d device=%s)",
             len(jobs), workers, args.device)

    seed_results_by_pc: dict[tuple, list[dict]] = {}
    if workers == 1:
        for j in jobs:
            r = _train_one_cut_seed(
                j["point_id"], j["cut"], j["seed"],
                j["lr"], j["weight_decay"], j["warmup_epochs"], j["seq_len"],
                args.dataset, args.epochs, args.label, args.device, out_root)
            key = (j["point_id"], j["cut"])
            seed_results_by_pc.setdefault(key, []).append(r)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut2key = {}
            for j in jobs:
                fut = ex.submit(_train_one_cut_seed,
                    j["point_id"], j["cut"], j["seed"],
                    j["lr"], j["weight_decay"], j["warmup_epochs"], j["seq_len"],
                    args.dataset, args.epochs, args.label, args.device, out_root)
                fut2key[fut] = (j["point_id"], j["cut"])
            for fut in as_completed(fut2key):
                key = fut2key[fut]
                seed_results_by_pc.setdefault(key, []).append(fut.result())

    # Ensemble per (point, cut) and compute bull_regime_ic
    rows = []
    for (pid, cut), seed_rs in seed_results_by_pc.items():
        per_regime, bull_ic = _ensemble_per_cut_per_regime(seed_rs, hmm_labels)
        rows.append({
            "point_id": pid,
            "cut": cut,
            "n_seeds_ok": sum(1 for s in seed_rs if s["status"] == "ok"),
            "bull_regime_ic": bull_ic,
            "per_regime_json": json.dumps(per_regime),
        })
    runs_df = pd.DataFrame(rows)
    runs_df = runs_df.merge(
        design[["point_id", "lr", "weight_decay", "warmup_epochs", "seq_len",
                "lr_coded", "weight_decay_coded", "warmup_epochs_coded",
                "seq_len_coded", "is_center"]],
        on="point_id", how="left")
    runs_df.to_csv(out_root / "runs.csv", index=False)

    # Aggregate bull_regime_ic across cuts per point
    point_agg = runs_df.dropna(subset=["bull_regime_ic"]).groupby("point_id").agg(
        lr=("lr", "first"),
        weight_decay=("weight_decay", "first"),
        warmup_epochs=("warmup_epochs", "first"),
        seq_len=("seq_len", "first"),
        is_center=("is_center", "first"),
        bull_ic_mean=("bull_regime_ic", "mean"),
        bull_ic_std=("bull_regime_ic", "std"),
        n_cuts=("cut", "count"),
    ).reset_index().sort_values("bull_ic_mean", ascending=False)
    point_agg.to_csv(out_root / "points.csv", index=False)

    # Write summary
    md = ["# HF PatchTST DOE Summary\n"]
    md.append("**Method**: 2^(4-1) FrFact Res IV + 1 center, walk-forward × ensemble\n")
    md.append("**Backbone**: HuggingFace transformers PatchTSTModel\n")
    md.append("**Objective**: bull_regime_IC (mean of {BULL_CALM, BULL_VOLATILE})\n")
    md.append(f"**Cuts**: {cut_names}\n")
    md.append(f"**Seeds per (point × cut)**: {args_seeds} (predict-averaged ensemble)\n")
    md.append("\n## Per-Point Aggregate (sorted by bull_ic_mean)\n")
    md.append("| Point | lr | wd | warmup | seq | bull_ic_mean | bull_ic_std | n_cuts |\n|---|---|---|---|---|---|---|---|\n")
    for _, r in point_agg.iterrows():
        md.append(f"| {int(r['point_id'])} | {r['lr']:.1e} | {r['weight_decay']:.1e} | "
                  f"{int(r['warmup_epochs'])} | {int(r['seq_len'])} | "
                  f"{r['bull_ic_mean']:+.4f} | {r['bull_ic_std']:.4f} | "
                  f"{int(r['n_cuts'])} |\n")
    if not point_agg.empty:
        best = point_agg.iloc[0]
        md.append(f"\n## Best point\n- **Point {int(best['point_id'])}**: "
                  f"lr={best['lr']:.2e}, wd={best['weight_decay']:.2e}, "
                  f"warmup={int(best['warmup_epochs'])}, seq={int(best['seq_len'])}\n")
        md.append(f"- **bull_ic_mean** = {best['bull_ic_mean']:+.4f} "
                  f"± {best['bull_ic_std']:.4f} (n_cuts={int(best['n_cuts'])})\n")
    (out_root / "summary.md").write_text("".join(md))

    print(f"\nDONE: {len(jobs)} runs; summary at {out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
