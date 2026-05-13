#!/usr/bin/env python
"""Track 2 — LightGBM trainer (mirrors train_production_model.py XGB API).

Per-cutoff walkforward training using LightGBM lambdarank objective.
Output schema is identical to XGB artifact except 'booster_raw_json'
contains LGBM model serialized via model_to_string().

Reference: Ke et al. NeurIPS 2017 "LightGBM: A Highly Efficient Gradient
Boosting Decision Tree"; Burges 2010 "From RankNet to LambdaRank to
LambdaMART" (lambdarank objective).

Usage (walkforward, mirrors train_production_model.py):
    python scripts/train_production_model_lgbm.py \\
        --train-cutoff 2024-06-01 \\
        --output-path backtesting/renquant_104/artifacts/walkforward_lgbm/2024-06-01/panel-ltr.json \\
        --side-label walkforward_lgbm_2024-06-01
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from typing import Optional
import numpy as np, pandas as pd, lightgbm as lgb
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train-lgbm")

REPO = Path(__file__).resolve().parent.parent

# LGBM params chosen to roughly mirror XGB depth=5, eta=0.05, 100 rounds.
# objective=lambdarank: pairwise ranking, identical to XGB rank:pairwise
# label_gain: trivial gain table for continuous label
PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,                # 2^5 - 1 to roughly match max_depth=5
    "min_child_samples": 50,         # matches XGB min_child_weight=50
    "feature_fraction": 0.7,         # matches XGB colsample_bytree
    "bagging_fraction": 0.7,         # matches XGB subsample
    "bagging_freq": 5,
    "verbose": -1,
    "num_threads": 8,
    "seed": 42,
    "lambdarank_truncation_level": 50,  # speeds up training without major IC loss
}
N_ROUNDS = 100
LABEL = "fwd_60d_excess"
DEFAULT_OUTPUT = REPO / "data" / "panel-ltr-prod-alpha158-fund-fwd60d-lgbm.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-cutoff", type=str, default=None,
                   help="ISO date; only rows where panel.date < cutoff are used.")
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--side-label", type=str, default=None)
    return p.parse_args()


def resolve_paths(args):
    cutoff = pd.Timestamp(args.train_cutoff) if args.train_cutoff else None
    is_wf = cutoff is not None
    if is_wf:
        if args.output_path is None:
            raise SystemExit("--train-cutoff requires --output-path")
        if "walkforward" not in args.output_path and "lgbm" not in args.output_path:
            raise SystemExit("--train-cutoff requires path to contain 'walkforward' or 'lgbm'")
        if args.side_label is None:
            raise SystemExit("--train-cutoff requires --side-label")
        out = Path(args.output_path)
    else:
        out = Path(args.output_path) if args.output_path else DEFAULT_OUTPUT
    return cutoff, out, is_wf


def load_and_slice_panel(cutoff_date):
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    # Match train_production_model.py feature columns (alpha158 + 5fund + 3PEAD + 3SUE)
    # The actual feat_cols list is set during training_panel pipeline; we mirror.
    # Use the same columns as existing prod artifact for consistency.
    from pathlib import Path as _P
    prod_art_path = _P("backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json")
    feat_cols = json.loads(prod_art_path.read_text())["feature_cols"]
    # Verify columns exist
    missing = [c for c in feat_cols if c not in panel.columns]
    if missing:
        raise SystemExit(f"Missing {len(missing)} feature cols in panel: {missing[:5]}")
    panel = panel[panel[LABEL].notna()].copy()
    if cutoff_date is not None:
        # Same logic as XGB: rows where date < cutoff AND label horizon doesn't peek
        # fwd_60d_excess label is computed using next 60 trading days → safe if
        # row date + 60 days <= cutoff
        safe_date = cutoff_date - pd.Timedelta(days=60)
        panel = panel[panel["date"] <= safe_date]
        log.info(f"Cutoff filter: cutoff={cutoff_date.date()} → safe_date={safe_date.date()} → "
                 f"rows={len(panel)} (kept rows where date ≤ safe_date)")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel, feat_cols


def build_normalization(train, feat_cols):
    X = train[feat_cols].fillna(0).values.astype(np.float64)
    mu = X.mean(axis=0); sd = X.std(axis=0)
    # NaN/inf guard
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where((sd > 1e-9) & np.isfinite(sd), sd, 1.0)
    return mu, sd


def train_lgbm(train, feat_cols):
    """Train lambdarank LGBM and return (booster, in-sample IC)."""
    mu, sd = build_normalization(train, feat_cols)
    Xn = ((train[feat_cols].fillna(0).values - mu) / sd).astype(np.float64)
    # LGBM lambdarank requires:
    #   - label: continuous gain values (or int relevance)
    #   - group: count per query (we use 'date' as query)
    # LGBM lambdarank requires INTEGER relevance levels bounded by
    # max_label_gain (default 30). XGB rank:pairwise uses continuous;
    # to make LGBM ranking fair we bucket each day's labels into deciles
    # (0..9). This preserves the relative ordering XGB cares about while
    # satisfying LGBM's discrete-relevance requirement.
    y = train[LABEL].clip(-5, 5).values.astype(np.float64)
    # Per-date decile bucketing
    train_with_label = train.copy()
    train_with_label["_clipped"] = y
    decile = (train_with_label.groupby("date")["_clipped"]
                              .rank(pct=True, method="first") * 10 - 1e-9)
    y_int = decile.clip(0, 9).astype(int).values
    # Group sizes
    dates = train["date"].values
    sort_idx = np.argsort(dates)
    Xs = Xn[sort_idx]; ys = y_int[sort_idx]; ds = dates[sort_idx]
    _, gsz = np.unique(ds, return_counts=True)
    log.info(f"Training LGBM lambdarank (params={PARAMS})...")
    train_data = lgb.Dataset(Xs, label=ys, group=gsz, free_raw_data=False)
    booster = lgb.train(PARAMS, train_data, num_boost_round=N_ROUNDS)
    # In-sample IC sanity
    pred = booster.predict(Xn)
    from scipy.stats import spearmanr
    train_check = train.copy()
    train_check["pred"] = pred
    ics = []
    for _, g in train_check.groupby("date"):
        if len(g) < 5: continue
        ic, _ = spearmanr(g["pred"], g[LABEL])
        if not np.isnan(ic): ics.append(ic)
    train_ic = float(np.mean(ics)) if ics else float("nan")
    log.info(f"In-sample train IC: {train_ic:+.4f}")
    return booster, train_ic, mu, sd


def build_artifact(booster, feat_cols, mu, sd, train_ic, cutoff_date, is_wf, side_label):
    return {
        "version": "1.0",
        "kind": "panel_ltr_lightgbm",
        "trained_date": datetime.utcnow().isoformat() + "Z",
        "feature_cols": list(feat_cols),
        "feature_means": mu.tolist(),
        "feature_stds": sd.tolist(),
        "params": dict(PARAMS),
        "best_iter": N_ROUNDS,
        "booster_raw_json": booster.model_to_string(),
        "panel_shape": [-1, len(feat_cols)],
        "label_col": LABEL,
        "lookahead_days": 60,
        "training_notes": (f"renquant_104 — panel-LTR LightGBM lambdarank | cutoff={cutoff_date} | "
                          f"side_label={side_label} | train_ic={train_ic:+.4f}"),
        "config_fingerprint_fields": [],   # walkforward skips fingerprint per §5.13.13
        "config_fingerprint": None,
        "metadata": {"model_class": "lightgbm", "lambdarank_truncation_level": PARAMS.get("lambdarank_truncation_level")},
    }


def main():
    args = parse_args()
    cutoff, out_path, is_wf = resolve_paths(args)
    train, feat_cols = load_and_slice_panel(cutoff)
    log.info(f"Train rows: {len(train)} | tickers: {train['ticker'].nunique()} | dates: {train['date'].min()} → {train['date'].max()}")
    booster, train_ic, mu, sd = train_lgbm(train, feat_cols)
    art = build_artifact(booster, feat_cols, mu, sd, train_ic, cutoff, is_wf, args.side_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(art))
    log.info(f"Walk-forward artifact — skipping fingerprint stamp (§5.13.13).")
    log.info(f"Saved artifact: {out_path}  (size={out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
