#!/usr/bin/env python
"""5-cut x 5-seed XGBoost baseline for PatchTST architecture screens.

This is the missing same-window tabular baseline for
``compare_arch_5cut_5seed.py``. It uses the same default cuts, label,
train-only label winsorization, and per-regime IC aggregation as the HF
PatchTST drivers, then writes the same ``aggregate.csv`` schema:

    cut,seed,regime,ic

Run after PatchTST arms finish:

    .venv/bin/python scripts/eval_xgb_5cut_5seed.py --jobs 3 --nthread 4
    .venv/bin/python scripts/compare_arch_5cut_5seed.py \
      --runs artifacts/xgb_5cut_5seed_pt07:xgb \
             artifacts/hf_trainer_5cut_5seed_pt07_clean:hf_patchtst
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval-xgb-5cut-5seed")

CUTS = ["cut1_covid", "cut2_fed", "cut3_inflpk", "cut4_svb", "cut5_unwind"]
SEEDS = [42, 43, 44, 45, 46]
OUT_ROOT = REPO / "artifacts/xgb_5cut_5seed_pt07"


def csrank_norm_per_day(panel: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Cross-sectional rank-normalize features to [-0.5, +0.5] per date."""
    panel = panel.copy()
    panel[feat_cols] = panel.groupby("date")[feat_cols].rank(pct=True) - 0.5
    panel[feat_cols] = panel[feat_cols].fillna(0.0)
    return panel


def label_winsor_bounds(
    panel: pd.DataFrame,
    label_col: str,
    *,
    fit_mask: pd.Series,
    pct: float = 0.005,
) -> tuple[float, float]:
    fit = panel.loc[fit_mask, label_col].dropna()
    if fit.empty:
        raise ValueError(f"cannot fit winsor bounds: empty train label {label_col}")
    return float(fit.quantile(pct)), float(fit.quantile(1.0 - pct))


def load_panel_with_split(dataset: Path, cut_name: str, label_col: str) -> tuple[pd.DataFrame, list[str]]:
    from kernel.walk_forward_splits import assign_split_column, build_default_cuts

    panel = pd.read_parquet(dataset)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    panel = panel.dropna(subset=[label_col]).copy()
    cut = next(c for c in build_default_cuts() if c.name == cut_name)
    panel["split_label"] = assign_split_column(panel, cut, embargo_days=60)
    excluded = {
        "date", "ticker", "split_label",
        "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess",
        "fwd_5d_excess_raw", "fwd_20d_excess_raw", "fwd_60d_excess_raw",
    }
    feat_cols = [
        c for c in panel.columns
        if c not in excluded and panel[c].dtype.kind in "fiub"
    ]
    panel = csrank_norm_per_day(panel, feat_cols)
    lo, hi = label_winsor_bounds(
        panel, label_col, fit_mask=panel["split_label"].eq("train"),
    )
    panel[label_col] = panel[label_col].clip(lower=lo, upper=hi)
    panel.attrs["label_winsor"] = {
        "enabled": True,
        "fit_split": "train",
        "lower": lo,
        "upper": hi,
    }
    return panel, feat_cols


def mean_daily_ic(frame: pd.DataFrame, score_col: str, label_col: str) -> float:
    ics: list[float] = []
    for _, g in frame.groupby("date", sort=False):
        if len(g) < 5:
            continue
        if np.allclose(g[score_col], g[score_col].iloc[0]) or np.allclose(g[label_col], g[label_col].iloc[0]):
            continue
        rho, _ = spearmanr(g[score_col], g[label_col])
        if np.isfinite(rho):
            ics.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan")


def run_one(args_dict: dict, cut: str, seed: int) -> dict:
    args = argparse.Namespace(**args_dict)
    out_dir = Path(args.output_root) / cut / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"xgb_{cut}_seed{seed}_summary.json"
    preds_path = out_dir / f"xgb_{cut}_seed{seed}_val_preds.parquet"
    if args.reuse_existing and summary_path.exists() and preds_path.exists():
        payload = json.loads(summary_path.read_text())
        payload["status"] = "ok"
        payload["skipped_existing"] = True
        return payload

    panel, feat_cols = load_panel_with_split(Path(args.dataset), cut, args.label)
    train = panel[panel["split_label"].eq("train")].reset_index(drop=True)
    val = panel[panel["split_label"].eq("val")].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError(f"{cut} seed={seed}: empty train/val split")

    params = {
        "objective": "rank:pairwise",
        "eval_metric": "rmse",
        "eta": float(args.eta),
        "max_depth": int(args.max_depth),
        "min_child_weight": float(args.min_child_weight),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "reg_lambda": float(args.reg_lambda),
        "reg_alpha": float(args.reg_alpha),
        "tree_method": "hist",
        "nthread": int(args.nthread),
        "verbosity": 0,
        "seed": int(seed),
    }
    dtrain = xgb.DMatrix(train[feat_cols].to_numpy(dtype=np.float32), label=train[args.label].to_numpy(dtype=np.float32))
    dtrain.set_group(train.groupby("date", sort=True).size().to_numpy(dtype=np.int32))
    dval = xgb.DMatrix(val[feat_cols].to_numpy(dtype=np.float32), label=val[args.label].to_numpy(dtype=np.float32))
    dval.set_group(val.groupby("date", sort=True).size().to_numpy(dtype=np.int32))
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(args.num_boost_round),
        evals=[(dval, "val")],
        early_stopping_rounds=int(args.early_stopping_rounds) if args.early_stopping_rounds else None,
        verbose_eval=False,
    )
    val = val.copy()
    val["pred"] = booster.predict(dval)
    val_preds = val[["date", "ticker", "pred", args.label]].rename(columns={args.label: "label"})
    val_preds.to_parquet(preds_path, index=False)

    from kernel.hmm_regime_labels import compute_hmm_regime_labels, per_hmm_regime_ic

    hmm = compute_hmm_regime_labels(Path(args.spy_path))
    per_regime = per_hmm_regime_ic(
        val_preds[["date", "pred", "label"]],
        hmm,
        min_samples_per_day=5,
        min_days_per_regime=int(args.min_days_per_regime),
    )
    best_val_ic = float(min(per_regime.values())) if per_regime else float("nan")
    summary = {
        "status": "ok",
        "arch": "xgb_rank_pairwise",
        "cut": cut,
        "seed": seed,
        "best_val_ic": best_val_ic,
        "pool_ic": mean_daily_ic(val_preds, "pred", "label"),
        "per_regime_ic": per_regime,
        "n_features": len(feat_cols),
        "n_train_rows": int(len(train)),
        "n_val_rows": int(len(val)),
        "params": params,
        "label_winsor": panel.attrs.get("label_winsor", {}),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def aggregate(results: list[dict], output_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for r in results:
        if r.get("status") != "ok":
            continue
        for regime, ic in (r.get("per_regime_ic") or {}).items():
            rows.append({"cut": r["cut"], "seed": r["seed"], "regime": regime, "ic": float(ic)})
        rows.append({"cut": r["cut"], "seed": r["seed"], "regime": "_MIN_", "ic": float(r["best_val_ic"])})
    df = pd.DataFrame(rows, columns=["cut", "seed", "regime", "ic"])
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / "aggregate.csv", index=False)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(REPO / "data/transformer_v4_wl200_clean.parquet"))
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--spy-path", default=str(REPO / "data/ohlcv/SPY/1d.parquet"))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--cuts", nargs="+", default=CUTS)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--reuse-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num-boost-round", type=int, default=200)
    p.add_argument("--early-stopping-rounds", type=int, default=20)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--min-child-weight", type=float, default=50.0)
    p.add_argument("--subsample", type=float, default=0.7)
    p.add_argument("--colsample-bytree", type=float, default=0.7)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.5)
    p.add_argument("--nthread", type=int, default=4)
    p.add_argument("--min-days-per-regime", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    pairs = [(cut, seed) for cut in args.cuts for seed in args.seeds]
    if args.dry_run:
        for cut, seed in pairs:
            print(f"{cut} seed={seed} -> {output_root / cut / f'seed_{seed}'}")
        return

    args_dict = vars(args)
    results: list[dict] = []
    failures: list[str] = []
    jobs = max(1, min(int(args.jobs), len(pairs)))
    if jobs == 1:
        for cut, seed in pairs:
            try:
                result = run_one(args_dict, cut, seed)
                results.append(result)
                log.info("[%s seed %d] OK min_regime_ic=%+.4f pool_ic=%+.4f",
                         cut, seed, result["best_val_ic"], result["pool_ic"])
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{cut}/seed{seed}: {exc!r}")
                log.exception("[%s seed %d] failed", cut, seed)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(run_one, args_dict, cut, seed): (cut, seed) for cut, seed in pairs}
            for fut in as_completed(futs):
                cut, seed = futs[fut]
                try:
                    result = fut.result()
                    results.append(result)
                    log.info("[%s seed %d] OK min_regime_ic=%+.4f pool_ic=%+.4f",
                             cut, seed, result["best_val_ic"], result["pool_ic"])
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{cut}/seed{seed}: {exc!r}")
                    log.exception("[%s seed %d] failed", cut, seed)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "raw_results.json").write_text(json.dumps(results, indent=2, default=str))
    df = aggregate(results, output_root)
    if not df.empty:
        log.info("\n%s", df.groupby(["cut", "regime"])["ic"].mean().unstack().to_string(float_format="%+.4f"))
        log.info("aggregate dumped: %s", output_root / "aggregate.csv")
    if failures:
        (output_root / "failures.json").write_text(json.dumps(failures, indent=2))
        raise SystemExit(f"{len(failures)} XGB baseline runs failed; see {output_root / 'failures.json'}")


if __name__ == "__main__":
    main()
