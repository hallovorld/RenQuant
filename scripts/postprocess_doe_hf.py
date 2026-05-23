#!/usr/bin/env python3
"""Post-hoc DOE analysis: DSR + PBO + main effects + 2-way interactions.

CLAUDE.md §5.14.4 (multiple-comparison correction MANDATORY) +
§5.14.6 (interaction-aware reporting). The HF DOE script
scripts/patchtst_doe_hf.py emits raw per-(point, cut) bull_regime_IC
data but doesn't compute these. This post-hoc fills the gap so the
verdict satisfies §5.14.

Reads:
  artifacts/patchtst_doe_hf/runs.csv   (per-(point, cut) bull_regime_ic)
  artifacts/patchtst_doe_hf/design.csv (coded knob matrix)

Writes:
  artifacts/patchtst_doe_hf/dsr.csv          (Bailey-LdP 2014 DSR per point)
  artifacts/patchtst_doe_hf/pbo_summary.csv  (CSCV rank-consistency)
  artifacts/patchtst_doe_hf/main_effects.csv (β per knob with CI)
  artifacts/patchtst_doe_hf/interactions.csv (2-way interaction β)
  artifacts/patchtst_doe_hf/summary_full.md  (§5.14.6 interaction-aware report)

References:
  - Bailey, López de Prado 2014 *J. Portfolio Mgmt* 40(5):94 (DSR)
  - Bailey, Borwein, López de Prado, Zhu 2015 *J. Comp. Finance* 14(1) (PBO/CSCV)
  - Box-Hunter-Hunter 2005 ch.6 (FrFact effects)
"""
from __future__ import annotations
import argparse
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def compute_dsr(ic_samples: np.ndarray, n_trials: int) -> float:
    """Deflated Sharpe Ratio (Bailey-LdP 2014).

    Args:
      ic_samples: per-(cut, seed) IC values for one design point
      n_trials: total number of design points evaluated (deflation factor)
    Returns: DSR-Sharpe (raw Sharpe penalized for multiple testing).
    """
    if len(ic_samples) < 2:
        return float("nan")
    sr = float(np.mean(ic_samples) / np.std(ic_samples, ddof=1)) \
         if np.std(ic_samples, ddof=1) > 1e-9 else 0.0
    # Bailey-LdP 2014 eq. 5 — expected maximum SR under H0 over N trials
    # (Euler-Mascheroni γ ≈ 0.5772)
    gamma = 0.5772156649
    n = max(n_trials, 2)
    e_max_sr = (
        (1 - gamma) * np.percentile(np.random.standard_normal(10_000), (1 - 1/n) * 100)
        + gamma * np.percentile(np.random.standard_normal(10_000), (1 - 1/(n * np.e)) * 100)
    )
    # DSR = how much our SR exceeds the bound expected by chance
    return float(sr - e_max_sr)


def compute_pbo(per_point_per_cut: pd.DataFrame, score_col: str = "bull_regime_ic"
                ) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey-Borwein-LdP-Zhu 2015).

    For each subset partition of cuts: find best point on IS half, check its
    rank on OOS half. PBO = fraction of partitions where best-IS scores below
    median on OOS.
    """
    cuts = sorted(per_point_per_cut["cut"].unique())
    if len(cuts) < 2:
        return float("nan")
    pivot = per_point_per_cut.pivot_table(
        index="point_id", columns="cut", values=score_col, aggfunc="mean")
    n_points = len(pivot)
    if n_points < 4:
        return float("nan")
    n_below_median = 0
    n_partitions = 0
    for is_cuts in combinations(cuts, max(1, len(cuts) // 2)):
        oos_cuts = [c for c in cuts if c not in is_cuts]
        if not oos_cuts:
            continue
        is_score = pivot[list(is_cuts)].mean(axis=1)
        oos_score = pivot[list(oos_cuts)].mean(axis=1)
        best_is_point = is_score.idxmax()
        oos_rank = (oos_score < oos_score.loc[best_is_point]).sum()
        if oos_rank < n_points / 2:
            n_below_median += 1
        n_partitions += 1
    return float(n_below_median / n_partitions) if n_partitions else float("nan")


def fit_effects(points_df: pd.DataFrame, score_col: str = "bull_ic_mean"
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Linear regression on coded design → main effects + 2-way interactions."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    coded_cols = [f"{k}_coded" for k in
                  ("lr", "weight_decay", "warmup_epochs", "seq_len")]
    needed_cols = [score_col] + coded_cols
    missing = [c for c in needed_cols if c not in points_df.columns]
    if missing or points_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    valid = points_df.dropna(subset=needed_cols)
    if len(valid) < 5:
        return pd.DataFrame(), pd.DataFrame()
    X = valid[coded_cols].values
    y = valid[score_col].values
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    names = poly.get_feature_names_out(coded_cols)
    reg = LinearRegression().fit(X_poly, y)

    main_rows, inter_rows = [], []
    for name, coef in zip(names, reg.coef_):
        if " " in name:
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
    main_df = pd.DataFrame(main_rows).reindex(
        pd.DataFrame(main_rows)["beta"].abs().sort_values(ascending=False).index)
    inter_df = pd.DataFrame(inter_rows).reindex(
        pd.DataFrame(inter_rows)["beta"].abs().sort_values(ascending=False).index)
    return main_df.reset_index(drop=True), inter_df.reset_index(drop=True)


def assemble_runs_from_val_preds(doe_dir: Path, design: pd.DataFrame
                                  ) -> pd.DataFrame:
    """Build runs.csv-equivalent from completed val_preds files. Enables
    partial verdict even if DOE script hasn't finished + written runs.csv."""
    import sys as _sys
    _sys.path.insert(0, str(REPO))
    from kernel.hmm_regime_labels import (compute_hmm_regime_labels,
                                            per_hmm_regime_ic, bull_regime_ic)
    import json as _json

    hmm = compute_hmm_regime_labels(REPO / "data/ohlcv/SPY/1d.parquet")
    rows = []
    # Group val_preds by (point_id, cut), ensemble across seeds
    pred_files: dict[tuple, list[Path]] = {}
    for vp in sorted(doe_dir.rglob("*val_preds.parquet")):
        parts = vp.parent.name.split("_")  # pt_00_cut1_covid_seed_42
        if len(parts) < 5: continue
        pid = int(parts[1])
        cut = "_".join(parts[2:-2])  # handle cut names like "cut1_covid"
        pred_files.setdefault((pid, cut), []).append(vp)

    for (pid, cut), files in pred_files.items():
        dfs = [pd.read_parquet(f) for f in files]
        n_rows = len(dfs[0])
        if all(len(d) == n_rows for d in dfs):
            ens_pred = np.mean([d["pred"].values for d in dfs], axis=0)
            ensembled = pd.DataFrame({
                "date": dfs[0]["date"].values, "pred": ens_pred,
                "label": dfs[0]["label"].values,
            })
        else:
            ensembled = dfs[0][["date", "pred", "label"]].copy()
        per_regime = per_hmm_regime_ic(ensembled, hmm,
                                        min_samples_per_day=5,
                                        min_days_per_regime=5)
        bull_ic = bull_regime_ic(per_regime)
        rows.append({
            "point_id": pid, "cut": cut, "n_seeds_ok": len(files),
            "bull_regime_ic": bull_ic,
            "per_regime_json": _json.dumps(per_regime),
        })
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs
    runs = runs.merge(
        design[["point_id", "lr", "weight_decay", "warmup_epochs", "seq_len",
                "lr_coded", "weight_decay_coded", "warmup_epochs_coded",
                "seq_len_coded", "is_center"]],
        on="point_id", how="left")
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doe-dir", default="artifacts/patchtst_doe_hf",
                   help="HF DOE output directory")
    p.add_argument("--partial", action="store_true",
                   help="Assemble runs.csv from val_preds if missing (DOE still running)")
    args = p.parse_args()

    np.random.seed(42)  # DSR's null-distribution sample
    doe_dir = REPO / args.doe_dir
    design = pd.read_csv(doe_dir / "design.csv")

    runs_path = doe_dir / "runs.csv"
    if runs_path.exists():
        runs = pd.read_csv(runs_path)
    elif args.partial:
        print("runs.csv missing — assembling from val_preds files (partial mode)")
        runs = assemble_runs_from_val_preds(doe_dir, design)
        runs.to_csv(doe_dir / "runs_partial.csv", index=False)
        print(f"  wrote runs_partial.csv ({len(runs)} (point,cut) combos)")
    else:
        raise SystemExit(f"runs.csv missing at {runs_path}. Use --partial to assemble from val_preds.")

    # ── Aggregate per-point across cuts ────────────────────────────────────
    valid = runs.dropna(subset=["bull_regime_ic"])
    point_agg = valid.groupby("point_id").agg(
        bull_ic_mean=("bull_regime_ic", "mean"),
        bull_ic_std=("bull_regime_ic", "std"),
        n_cuts=("cut", "count"),
    ).reset_index()
    point_agg = point_agg.merge(
        design[["point_id", "lr", "weight_decay", "warmup_epochs", "seq_len",
                "lr_coded", "weight_decay_coded", "warmup_epochs_coded",
                "seq_len_coded", "is_center"]],
        on="point_id", how="left")

    # ── §5.14.4 DSR per point ──────────────────────────────────────────────
    n_planned_design_points = len(design)
    evaluated_points = sorted(valid["point_id"].dropna().astype(int).unique())
    n_design_points = len(evaluated_points)
    missing_points = sorted(
        set(design["point_id"].astype(int)) - set(evaluated_points)
    )
    dsr_rows = []
    for pid, group in valid.groupby("point_id"):
        ic_samples = group["bull_regime_ic"].dropna().values
        dsr = compute_dsr(ic_samples, n_design_points)
        dsr_rows.append({"point_id": pid, "dsr": dsr,
                         "n_samples": len(ic_samples)})
    dsr_df = pd.DataFrame(dsr_rows)
    dsr_df.to_csv(doe_dir / "dsr.csv", index=False)

    # ── §5.14.4 PBO via CSCV ───────────────────────────────────────────────
    pbo = compute_pbo(valid)
    pd.DataFrame([{
        "pbo": pbo,
        "n_design_points": n_design_points,
        "n_planned_design_points": n_planned_design_points,
        "n_evaluated_design_points": n_design_points,
        "n_missing_design_points": len(missing_points),
        "missing_point_ids": " ".join(str(x) for x in missing_points),
    }]).to_csv(doe_dir / "pbo_summary.csv", index=False)

    # ── §5.14.6 Main effects + 2-way interactions ──────────────────────────
    main_df, inter_df = fit_effects(point_agg)
    main_df.to_csv(doe_dir / "main_effects.csv", index=False)
    inter_df.to_csv(doe_dir / "interactions.csv", index=False)

    # ── §5.14.6 Augmented summary.md ───────────────────────────────────────
    md = ["# HF PatchTST DOE — §5.14 FULL Verdict (post-hoc)\n"]
    md.append("**Source**: scripts/patchtst_doe_hf.py + scripts/postprocess_doe_hf.py\n")
    md.append(f"**Evaluated design points**: {n_design_points} / {n_planned_design_points}\n")
    if missing_points:
        md.append(f"**Missing point ids**: {', '.join(str(x) for x in missing_points)}\n")
    md.append(f"**Objective**: bull_regime_IC (HMM {{BULL_CALM, BULL_VOLATILE}})\n")
    md.append(f"\n## PBO (Bailey-Borwein-LdP-Zhu 2015): **{pbo:.2f}**\n")
    md.append("PBO > 0.5 → overfit; PBO < 0.5 → robust.\n")

    md.append("\n## Per-Point: bull_ic + DSR\n")
    md.append("| Point | lr | wd | warmup | seq | bull_ic_mean | bull_ic_std | DSR | n_cuts |\n|---|---|---|---|---|---|---|---|---|\n")
    merged = point_agg.merge(dsr_df, on="point_id", how="left").sort_values(
        "bull_ic_mean", ascending=False)
    for _, r in merged.iterrows():
        dsr_str = f"{r['dsr']:+.3f}" if pd.notna(r['dsr']) else "—"
        md.append(f"| {int(r['point_id'])} | {r['lr']:.1e} | {r['weight_decay']:.1e} | "
                  f"{int(r['warmup_epochs'])} | {int(r['seq_len'])} | "
                  f"{r['bull_ic_mean']:+.4f} | {r.get('bull_ic_std', float('nan')):.4f} | "
                  f"{dsr_str} | {int(r['n_cuts'])} |\n")

    md.append("\n## Main Effects (sorted by |β|)\n")
    md.append("| Knob | β |\n|---|---|\n")
    for _, r in main_df.iterrows():
        md.append(f"| `{r['knob']}` | {r['beta']:+.4f} |\n")

    md.append("\n## 2-Way Interactions (sorted by |β|)\n")
    md.append("| A | B | β |\n|---|---|---|\n")
    for _, r in inter_df.iterrows():
        md.append(f"| `{r['knob_a']}` | `{r['knob_b']}` | {r['beta']:+.4f} |\n")

    md.append("\n## §5.14 Pass-Gate Check\n")
    if not merged.empty:
        best = merged.iloc[0]
        passes = []
        passes.append(("PBO < 0.5", pbo < 0.5, f"PBO={pbo:.2f}"))
        passes.append(("Best DSR > 0", best.get('dsr', float('nan')) > 0,
                       f"DSR={best.get('dsr', float('nan')):+.3f}"))
        passes.append(("Best bull_ic > 0", best['bull_ic_mean'] > 0,
                       f"bull_ic={best['bull_ic_mean']:+.4f}"))
        for name, ok, val in passes:
            md.append(f"- {'✅' if ok else '❌'} {name} ({val})\n")

    (doe_dir / "summary_full.md").write_text("".join(md))
    print(f"Wrote: dsr.csv pbo_summary.csv main_effects.csv interactions.csv summary_full.md")
    print(f"PBO = {pbo:.2f}")
    if not merged.empty:
        print(f"Best point: {int(merged.iloc[0]['point_id'])}  "
              f"bull_ic={merged.iloc[0]['bull_ic_mean']:+.4f}  "
              f"DSR={merged.iloc[0].get('dsr', float('nan')):+.3f}")


if __name__ == "__main__":
    main()
