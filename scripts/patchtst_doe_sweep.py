#!/usr/bin/env python3
"""PatchTST hyperparameter sweep via DOE — CLAUDE.md §5.14 canonical method.

REPLACES the (deleted) Optuna black-box approach. §5.14.3 names pyDOE2
as the canonical library; §5.14.1 stage table says Fractional Factorial
2^(k-p) Resolution IV+ for "which knobs + which pairs matter" screening.

PRIME DIRECTIVE compliance (CLAUDE.md §🔴 + [[feedback_prime_directive_in_objective_funcs]]):
  Objective = min-across-regime val IC (NOT pooled mean). Pooled mean
  picks regime-fragile models; min picks the regime-robust ones.

References:
  - Box-Hunter-Hunter 2005 *Statistics for Experimenters* ch.6 (FrFact)
  - Box & Behnken 1960 *Technometrics* 2:455 (BBD — Stage 2 if needed)
  - Bailey-López de Prado 2014 *J. Portfolio Mgmt* 40(5):94 (DSR for
    multiple-comparison correction §5.14.4)
  - Nie et al 2023 ICLR "PatchTST: A Time Series is Worth 64 Words"
    (hyperparameter bounds)
  - Asness-Moskowitz-Pedersen 2013 *J. Finance* 68(3):929 (regime
    stratification rationale)

Design (Stage 1 — screening, §5.14.1):
  - 4 knobs × 2 levels = 2^(4-1) = 8 FrFact corner points + 1 center
  - 9 points × 3 seeds = 27 PatchTST training runs
  - Estimated wallclock: 27 × ~5 min = ~2.25 h on MPS-serial

Knobs (bounds per Nie 2023 PatchTST + repo conventions):
  - lr ∈ {1e-4, 1e-3}           (log-scaled coded as ±1)
  - weight_decay ∈ {1e-4, 1e-1} (log-scaled)
  - warmup_epochs ∈ {2, 6}
  - seq_len ∈ {16, 60}

Concurrency (per [[concurrency_resource_budget]]):
  - MPS is a GPU = serial (one process at a time)
  - --workers >1 only sane on CPU (PatchTST ~10× slower so default 1)
  - Per-trial subprocess to existing scripts/transformer_v4.py (no
    custom training code per user 2026-05-18 mandate)

Outputs (§5.14.6 interaction-aware reporting):
  - artifacts/patchtst_doe/runs.csv         per (point, seed) row
  - artifacts/patchtst_doe/main_effects.csv per-knob β + CI
  - artifacts/patchtst_doe/interactions.csv per-pair β + CI
  - artifacts/patchtst_doe/summary.md       full DOE report

Usage::

    .venv/bin/python scripts/patchtst_doe_sweep.py \\
        --dataset data/transformer_v4_wl200_clean.parquet \\
        --n-seeds 3 --epochs 4 --device mps

    # CPU concurrent (slower per run but parallel):
    .venv/bin/python scripts/patchtst_doe_sweep.py \\
        --device cpu --workers 4 --n-seeds 3 --epochs 4
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Single-thread per-subprocess; trainer sets its own threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pyDOE2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Lifted to renquant-common (umbrella PR #5 in that repo, 2026-06-01).
from renquant_common.regime_labels import (compute_spy_regime_labels,
                                            per_regime_cs_ic, min_across_regimes)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("patchtst-doe")


# ── Knob spec ──────────────────────────────────────────────────────────────

KNOBS = [
    # (name, low, high, log_scale)
    ("lr",            1e-4, 1e-3, True),
    ("weight_decay",  1e-4, 1e-1, True),
    ("warmup_epochs", 2,    6,    False),
    ("seq_len",       16,   60,   False),
]


def coded_to_real(name: str, coded: float) -> float:
    """Map ±1 (or 0 for center) → real knob value."""
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
    """Build 2^(4-1) Resolution IV fractional factorial + 1 center.

    Generator: D = ABC (Box-Hunter-Hunter standard for Res IV in 4 factors).
    Produces 8 corner points; we add 1 center point per §5.14.2 (lack-of-fit).
    """
    # pyDOE2: fracfact("a b c abc") → Res IV 2^(4-1)
    corners = pyDOE2.fracfact("a b c abc")  # shape (8, 4)
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


# ── Per-run subprocess ─────────────────────────────────────────────────────

def _train_one(point_id: int, seed: int, lr: float, weight_decay: float,
               warmup_epochs: int, seq_len: int, dataset: str, epochs: int,
               label: str, device: str, out_root: Path) -> dict:
    """Subprocess one transformer_v4.py training + compute per-regime IC.

    Returns dict with point_id, seed, val_ic_pool, val_ic_per_regime, val_ic_min.
    """
    out_dir = out_root / f"pt_{point_id:02d}_seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "scripts/transformer_v4.py"),
        "--dataset", dataset,
        "--arch", "patchtst",
        "--label", label,
        "--seq-len", str(int(seq_len)),
        "--epochs", str(epochs),
        "--num-seeds", "1",
        "--seed", str(seed),
        "--lr", str(lr),
        "--weight-decay", str(weight_decay),
        "--warmup-epochs", str(int(warmup_epochs)),
        "--output-dir", str(out_dir),
        "--device", device,
    ]
    log.info("[pt %02d seed %d] start lr=%.1e wd=%.1e warmup=%d seq=%d",
             point_id, seed, lr, weight_decay, warmup_epochs, seq_len)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-400:]
        log.warning("[pt %02d seed %d] FAILED rc=%d tail=%s",
                    point_id, seed, result.returncode, tail)
        return {"point_id": point_id, "seed": seed, "status": "fail",
                "val_ic_pool": None, "val_ic_min": None,
                "val_ic_per_regime": {}}

    summary_path = out_dir / "patchtst_summary.json"
    val_preds_path = out_dir / f"patchtst_seed{seed}_val_preds.parquet"
    if not summary_path.exists() or not val_preds_path.exists():
        log.warning("[pt %02d seed %d] missing outputs", point_id, seed)
        return {"point_id": point_id, "seed": seed, "status": "missing",
                "val_ic_pool": None, "val_ic_min": None,
                "val_ic_per_regime": {}}

    summary = json.loads(summary_path.read_text())
    val_ic_pool = float(summary["val_ic_mean"])

    val_preds = pd.read_parquet(val_preds_path)
    regimes = compute_spy_regime_labels(REPO / "data/ohlcv/SPY/1d.parquet")
    per_regime = per_regime_cs_ic(val_preds, regimes,
                                   min_samples_per_day=5,
                                   min_days_per_regime=10)
    val_ic_min = min_across_regimes(per_regime)
    log.info("[pt %02d seed %d] DONE pool=%+.4f min_regime=%+.4f n_regimes=%d",
             point_id, seed, val_ic_pool, val_ic_min, len(per_regime))
    return {
        "point_id": point_id, "seed": seed, "status": "ok",
        "val_ic_pool": val_ic_pool,
        "val_ic_min": val_ic_min,
        "val_ic_per_regime": per_regime,
    }


# ── DOE response surface fit ───────────────────────────────────────────────

def fit_main_effects_and_interactions(runs_df: pd.DataFrame
                                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit y ~ β₀ + Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ on coded design.

    Returns (main_effects_df, interactions_df).
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    # Aggregate to point-level mean over seeds (variance-aware)
    point_df = runs_df.dropna(subset=["val_ic_min"]).groupby("point_id").agg(
        lr_coded=("lr_coded", "first"),
        weight_decay_coded=("weight_decay_coded", "first"),
        warmup_epochs_coded=("warmup_epochs_coded", "first"),
        seq_len_coded=("seq_len_coded", "first"),
        val_ic_min_mean=("val_ic_min", "mean"),
        val_ic_min_std=("val_ic_min", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()

    knob_names = ["lr", "weight_decay", "warmup_epochs", "seq_len"]
    X_coded = point_df[[f"{k}_coded" for k in knob_names]].values
    y = point_df["val_ic_min_mean"].values

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_coded)
    names = poly.get_feature_names_out([f"{k}_coded" for k in knob_names])

    reg = LinearRegression().fit(X_poly, y)
    intercept = float(reg.intercept_)
    coefs = reg.coef_

    main_rows, inter_rows = [], []
    for name, coef in zip(names, coefs):
        if " " in name:  # interaction term
            a, b = name.split(" ")
            inter_rows.append({
                "knob_a": a.replace("_coded", ""),
                "knob_b": b.replace("_coded", ""),
                "beta": float(coef),
            })
        else:
            main_rows.append({
                "knob": name.replace("_coded", ""),
                "beta": float(coef),
            })
    main_df = pd.DataFrame(main_rows).sort_values("beta", key=np.abs,
                                                    ascending=False)
    inter_df = pd.DataFrame(inter_rows).sort_values("beta", key=np.abs,
                                                      ascending=False)
    return main_df, inter_df, point_df, intercept


def write_summary(out_root: Path, runs_df: pd.DataFrame, main_df: pd.DataFrame,
                  inter_df: pd.DataFrame, point_df: pd.DataFrame,
                  intercept: float) -> None:
    runs_df.to_csv(out_root / "runs.csv", index=False)
    main_df.to_csv(out_root / "main_effects.csv", index=False)
    inter_df.to_csv(out_root / "interactions.csv", index=False)
    point_df.to_csv(out_root / "points.csv", index=False)

    # Markdown summary
    md = ["# PatchTST DOE Sweep Summary\n"]
    md.append("**Method**: 2^(4-1) Fractional Factorial Resolution IV + 1 center "
              "(CLAUDE.md §5.14.1)\n")
    md.append(f"**Objective**: min-across-regime val IC "
              "(PRIME DIRECTIVE compliance)\n")
    md.append(f"**Intercept (mean min-IC over design)**: {intercept:+.4f}\n")

    md.append("\n## Main Effects (sorted |β|)\n")
    md.append("| Knob | β (effect on min-regime IC) |\n|---|---|\n")
    for _, r in main_df.iterrows():
        md.append(f"| `{r['knob']}` | {r['beta']:+.4f} |\n")

    md.append("\n## 2-Way Interactions (sorted |β|)\n")
    md.append("| Knob A | Knob B | β |\n|---|---|---|\n")
    for _, r in inter_df.iterrows():
        md.append(f"| `{r['knob_a']}` | `{r['knob_b']}` | {r['beta']:+.4f} |\n")

    md.append("\n## Per-Point Results (n_seeds × min-regime IC)\n")
    md.append("| Point | lr | weight_decay | warmup_epochs | seq_len | "
              "mean | std | n |\n|---|---|---|---|---|---|---|---|\n")
    for _, r in point_df.iterrows():
        md.append(f"| {int(r['point_id'])} | "
                  f"{coded_to_real('lr', r['lr_coded']):.1e} | "
                  f"{coded_to_real('weight_decay', r['weight_decay_coded']):.1e} | "
                  f"{coded_to_real('warmup_epochs', r['warmup_epochs_coded'])} | "
                  f"{coded_to_real('seq_len', r['seq_len_coded'])} | "
                  f"{r['val_ic_min_mean']:+.4f} | "
                  f"{r['val_ic_min_std']:.4f} | "
                  f"{int(r['n_seeds'])} |\n")

    md.append("\n## Best point (highest min-regime IC)\n")
    best = point_df.sort_values("val_ic_min_mean", ascending=False).iloc[0]
    md.append(f"- **Point {int(best['point_id'])}**: "
              f"lr={coded_to_real('lr', best['lr_coded']):.2e}, "
              f"wd={coded_to_real('weight_decay', best['weight_decay_coded']):.2e}, "
              f"warmup={coded_to_real('warmup_epochs', best['warmup_epochs_coded'])}, "
              f"seq={coded_to_real('seq_len', best['seq_len_coded'])}\n")
    md.append(f"- **mean min-regime IC** = {best['val_ic_min_mean']:+.4f} "
              f"± {best['val_ic_min_std']:.4f} (n={int(best['n_seeds'])})\n")

    md.append("\n## Next steps (§5.14.1 stage table)\n")
    md.append("- If a knob's |β| dominates: zoom in, then BBD optimization "
              "(`pyDOE2.bbdesign`).\n")
    md.append("- If interactions matter: contour plot in top-2 knob plane.\n")
    md.append("- Predicted-optimum confirmatory: 3-seed run at surface max + "
              "DSR/PBO per §5.14.4.\n")

    (out_root / "summary.md").write_text("".join(md))
    log.info("Summary written: %s", out_root / "summary.md")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/transformer_v4_wl200_clean.parquet")
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--n-seeds", type=int, default=3,
                   help="Seeds per design point (CLAUDE 5.13.4 variance)")
    p.add_argument("--epochs", type=int, default=4,
                   help="Epochs per training (sweep speed)")
    p.add_argument("--device", default="mps",
                   choices=["cpu", "mps", "cuda"])
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers. Keep 1 for MPS (GPU contention); "
                        "higher only on CPU per [[concurrency_resource_budget]]")
    p.add_argument("--out-root", default="artifacts/patchtst_doe")
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Run only first point + 1 seed (quick verification)")
    args = p.parse_args()

    out_root = REPO / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    design = build_design_matrix()
    design.to_csv(out_root / "design.csv", index=False)
    log.info("Design matrix: %d points (8 FrFact + 1 center)", len(design))
    log.info("Total runs: %d points × %d seeds = %d trainings",
             len(design), args.n_seeds, len(design) * args.n_seeds)

    if args.smoke:
        design = design.head(1)
        args_n_seeds = 1
    else:
        args_n_seeds = args.n_seeds

    # Sanity: if device=mps, force workers=1 (GPU contention)
    if args.device == "mps" and args.workers > 1:
        log.warning("device=mps + workers=%d → forcing workers=1 "
                    "(MPS GPU contention)", args.workers)
        args.workers = 1

    # Build job list: (point_id, seed) for every (design row × n_seeds)
    jobs = []
    for _, row in design.iterrows():
        for s in range(args_n_seeds):
            jobs.append((int(row["point_id"]), args.seed_base + s,
                         float(row["lr"]), float(row["weight_decay"]),
                         int(row["warmup_epochs"]), int(row["seq_len"])))

    log.info("Dispatching %d jobs to %d worker(s) on %s",
             len(jobs), args.workers, args.device)

    runs = []
    if args.workers == 1:
        for j in jobs:
            r = _train_one(*j, dataset=args.dataset, epochs=args.epochs,
                            label=args.label, device=args.device,
                            out_root=out_root)
            runs.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut2job = {
                ex.submit(_train_one, *j, dataset=args.dataset,
                          epochs=args.epochs, label=args.label,
                          device=args.device, out_root=out_root): j
                for j in jobs
            }
            for fut in as_completed(fut2job):
                runs.append(fut.result())

    # Join runs to design coded coordinates
    runs_df = pd.DataFrame(runs)
    runs_df = runs_df.merge(
        design[["point_id", "lr_coded", "weight_decay_coded",
                "warmup_epochs_coded", "seq_len_coded"]],
        on="point_id", how="left")

    if args.smoke:
        runs_df.to_csv(out_root / "runs_smoke.csv", index=False)
        print(runs_df)
        return

    main_df, inter_df, point_df, intercept = \
        fit_main_effects_and_interactions(runs_df)
    write_summary(out_root, runs_df, main_df, inter_df, point_df, intercept)

    best_pt = point_df.sort_values("val_ic_min_mean", ascending=False).iloc[0]
    print("\n" + "=" * 70)
    print("DOE SWEEP COMPLETE")
    print(f"  Runs:        {len(runs_df)}")
    print(f"  Best point:  {int(best_pt['point_id'])}")
    print(f"  Min-IC mean: {best_pt['val_ic_min_mean']:+.4f}")
    print(f"  Summary:     {out_root / 'summary.md'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
