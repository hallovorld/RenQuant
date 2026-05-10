#!/usr/bin/env python
"""Train final production model: R1K + alpha158 + 5-fund + XGB d=5 e=0.05 fwd_60d.

Trained on ALL train data up through latest valid label date (current_date - 60d),
OR — when ``--train-cutoff`` is supplied — only on rows where ``date < cutoff``.
The cutoff path is the walk-forward training entrypoint (Track P3-v2,
CLAUDE.md §5.13.5: single source of truth for alpha158 panel-LTR training).

Saves artifact in PanelScorer format (kind=panel_ltr_xgboost) so existing
inference pipeline can load it.

Default output: data/panel-ltr-prod-alpha158-fund-fwd60d.json
Walk-forward output: backtesting/renquant_104/artifacts/walkforward_v2/<cutoff>/panel-ltr.json

CLI:
    # Daily production retrain (backward compat — no args)
    python scripts/train_production_model.py

    # Walk-forward per-cutoff retrain (Track P3-v2)
    python scripts/train_production_model.py \\
        --train-cutoff 2024-06-01 \\
        --output-path backtesting/renquant_104/artifacts/walkforward_v2/2024-06-01/panel-ltr.json \\
        --side-label walkforward_v2_2024-06-01
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from typing import Optional
import numpy as np, pandas as pd, xgboost as xgb
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train-prod")

REPO = Path(__file__).resolve().parent.parent
PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":8,"verbosity":0,"seed":42}
N_ROUNDS = 100
LABEL = "fwd_60d_excess"
DEFAULT_OUTPUT = REPO / "data" / "panel-ltr-prod-alpha158-fund-fwd60d.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--train-cutoff", type=str, default=None,
        help="ISO date (e.g. 2024-01-01); only rows where panel.date < cutoff are used.",
    )
    p.add_argument(
        "--output-path", type=str, default=None,
        help="Artifact output path. Defaults to data/panel-ltr-prod-alpha158-fund-fwd60d.json.",
    )
    p.add_argument(
        "--side-label", type=str, default=None,
        help="Extra training_notes tag; required when --train-cutoff is set.",
    )
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Optional[pd.Timestamp], Path, bool]:
    """Resolve cutoff + output path + is_walkforward flag.

    §5.13.13 (CRITICAL): when --train-cutoff is set, --output-path is
    REQUIRED and MUST contain "walkforward". Refuses to overwrite
    production artifact.
    """
    cutoff_date = pd.Timestamp(args.train_cutoff) if args.train_cutoff else None
    is_walkforward = cutoff_date is not None

    if is_walkforward:
        if args.output_path is None:
            raise SystemExit(
                "--train-cutoff requires --output-path (refusing to default "
                "to production artifact path)"
            )
        if "walkforward" not in args.output_path:
            raise SystemExit(
                f"--train-cutoff set but --output-path {args.output_path!r} "
                f"does not contain 'walkforward'. §5.13.13: refusing to risk "
                f"overwriting production artifact."
            )
        if args.side_label is None:
            raise SystemExit(
                "--train-cutoff requires --side-label for training_notes provenance."
            )
        out_path = Path(args.output_path)
    else:
        out_path = Path(args.output_path) if args.output_path else DEFAULT_OUTPUT

    return cutoff_date, out_path, is_walkforward


def load_and_slice_panel(cutoff_date: Optional[pd.Timestamp]) -> tuple[pd.DataFrame, list[str]]:
    """Load alpha158 panel, optionally filter by cutoff, return (train_df, feat_cols)."""
    log.info("Loading R1K + 5-fund panel (already normalized: alpha158=zscore, fund=robust-zscore)...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]

    train = panel.dropna(subset=[LABEL])
    if cutoff_date is not None:
        before = len(train)
        train = train[train["date"] < cutoff_date]
        log.info("Cutoff filter: %s — %d → %d rows (max date %s)",
                 cutoff_date.date().isoformat(), before, len(train),
                 train["date"].max().date() if len(train) else "EMPTY")
        if len(train) == 0:
            raise SystemExit(f"No training rows with date < {cutoff_date.date()}")

    log.info("Train rows: %d (panel total: %d), tickers: %d, dates: %s → %s",
             len(train), len(panel), train["ticker"].nunique(),
             train["date"].min().date(), train["date"].max().date())
    return train, feat_cols


def build_normalization(train: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the inference normalization chain stored in the artifact.

    For each feature, (mean, std) such that (raw - mean) / std = normalized value.
    alpha158 cols: from panel z-score stats; fund cols: robust z-score on train period.
    """
    ps = json.loads(Path("data/alpha158_qlib_dataset.stats.json").read_text())
    alpha_norm = dict(zip(ps["feature_cols"], zip(ps["feature_means"], ps["feature_stds"])))

    fund_cols = ["earnings_yield","book_to_price","gross_profitability","roe","asset_growth"]
    fund_raw = pd.read_parquet("data/sec_fundamentals_daily.parquet")
    fund_raw["date"] = pd.to_datetime(fund_raw["date"])
    train_dates = set(train["date"])
    fund_train_raw = fund_raw[fund_raw["date"].isin(train_dates)
                               & fund_raw["ticker"].isin(set(train["ticker"]))]
    log.info("Fund train rows for robust z-score recompute: %d", len(fund_train_raw))
    fund_norm = {}
    for c in fund_cols:
        col = fund_train_raw[c].dropna()
        med = float(col.median()) if len(col) else 0.0
        mad = float((col - med).abs().median()) if len(col) else 1.0
        scale = max(mad * 1.4826, 1e-9)
        fund_norm[c] = (med, scale)

    feat_means, feat_stds, feat_norm_kind = [], [], []
    for c in feat_cols:
        if c in alpha_norm:
            m, s = alpha_norm[c]
            feat_means.append(m); feat_stds.append(s); feat_norm_kind.append("global_z")
        elif c in fund_norm:
            m, s = fund_norm[c]
            feat_means.append(m); feat_stds.append(s); feat_norm_kind.append("robust_z")
        else:
            feat_means.append(0.0); feat_stds.append(1.0); feat_norm_kind.append("identity")
    log.info("Normalization chain: %d global_z, %d robust_z, %d identity",
             feat_norm_kind.count("global_z"), feat_norm_kind.count("robust_z"),
             feat_norm_kind.count("identity"))
    return np.array(feat_means), np.array(feat_stds), feat_norm_kind


def train_xgb(train: pd.DataFrame, feat_cols: list[str]) -> tuple[xgb.Booster, float]:
    """Train rank:pairwise XGB and return (booster, in-sample IC)."""
    Xtr = train[feat_cols].fillna(0).values.astype(np.float64)
    ytr = train[LABEL].clip(-5,5).values.astype(np.float64)

    sort_idx = np.argsort(train["date"].values)
    Xs, ys, ds = Xtr[sort_idx], ytr[sort_idx], train["date"].values[sort_idx]
    _, gsz = np.unique(ds, return_counts=True)

    log.info("Training XGB rank:pairwise (params=%s)...", PARAMS)
    dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
    booster = xgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)

    # In-sample IC sanity
    train_pred = booster.predict(xgb.DMatrix(Xtr))
    from scipy.stats import spearmanr
    train_check = train.copy()
    train_check["pred"] = train_pred
    train_ics = []
    for _, g in train_check.groupby("date"):
        if len(g) < 5: continue
        ic, _ = spearmanr(g["pred"], g[LABEL])
        if not np.isnan(ic): train_ics.append(ic)
    train_ic_mean = float(np.mean(train_ics)) if train_ics else float("nan")
    log.info("In-sample train IC: %+.4f (sanity check, not OOS)", train_ic_mean)
    return booster, train_ic_mean


def stamp_fingerprint(artifact: dict) -> str:
    """Stamp config_fingerprint_fields + config_fingerprint (production only).

    Per §5.13.13: walk-forward artifacts skip fingerprinting since they
    aren't intended to match live config — only the daily prod retrain
    does this step.
    """
    sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))
    from kernel.config_consistency import _model_relevant_fields, fingerprint_config  # noqa: PLC0415
    live_cfg_path = REPO / "backtesting/renquant_104/strategy_config.json"
    live_cfg = json.loads(live_cfg_path.read_text()) if live_cfg_path.exists() else {}
    artifact["config_fingerprint_fields"] = _model_relevant_fields(live_cfg)
    fp = fingerprint_config(live_cfg)
    artifact["config_fingerprint"] = fp
    return fp


def build_artifact(booster: xgb.Booster, feat_cols: list[str],
                   mu: np.ndarray, sd: np.ndarray, train: pd.DataFrame,
                   cutoff_date: Optional[pd.Timestamp],
                   side_label: Optional[str]) -> dict:
    """Build artifact dict, stamping cutoff_date + side_label when set."""
    raw_json = bytes(booster.save_raw(raw_format="json")).decode("utf-8")
    base_notes = (
        "alpha158 + SEC fund (5) + PEAD (3, E47 promoted 2026-05-08) on R1K "
        "291 tickers, fwd_60d label. PEAD real_signal lift +0.022 over "
        "alpha158+5fund baseline (paired §5.2 sanity passed)."
    )
    notes = base_notes + (f" [side_label={side_label}]" if side_label else "")
    artifact = {
        "version": 3,
        "kind": "panel_ltr_xgboost",
        "trained_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "feature_cols": feat_cols,
        "feature_means": mu.tolist(),
        "feature_stds":  sd.tolist(),
        "params": PARAMS,
        "best_iter": N_ROUNDS,
        "booster_raw_json": raw_json,
        "panel_shape": {
            "rows":    int(train.shape[0]),
            "tickers": int(train["ticker"].nunique()),
            "dates":   int(train["date"].nunique()),
        },
        "label_col": LABEL,
        "lookahead_days": 60,
        "training_notes": notes,
    }
    if cutoff_date is not None:
        artifact["cutoff_date"] = cutoff_date.isoformat()
    if side_label is not None:
        artifact["side_label"] = side_label
    return artifact


def main():
    args = parse_args()
    cutoff_date, out_path, is_walkforward = resolve_paths(args)

    train, feat_cols = load_and_slice_panel(cutoff_date)
    mu, sd, _ = build_normalization(train, feat_cols)
    booster, _ic = train_xgb(train, feat_cols)
    artifact = build_artifact(booster, feat_cols, mu, sd, train,
                              cutoff_date, args.side_label)

    if not is_walkforward:
        fp = stamp_fingerprint(artifact)
        log.info("Fingerprint: %s", fp)
    else:
        log.info("Walk-forward artifact — skipping fingerprint stamp (§5.13.13).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact))
    log.info("Saved artifact: %s  (size=%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    log.info("Feature cols (n=%d): %s ... %s", len(feat_cols), feat_cols[:3], feat_cols[-3:])


if __name__ == "__main__":
    main()
