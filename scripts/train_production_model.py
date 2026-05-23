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
import re
import numpy as np, pandas as pd, xgboost as xgb
from datetime import datetime
import uuid

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
    p.add_argument(
        "--label", type=str, default=None,
        help="Override LABEL column (default fwd_60d_excess). Track 6 horizon swap.",
    )
    p.add_argument(
        "--watchlist-file", type=str, default=None,
        help="JSON config or list file; filter panel rows to tickers in this watchlist. Track 1 wl retrain.",
    )
    p.add_argument(
        "--cv-n-splits", type=int, default=3,
        help="Number of purged walk-forward folds stamped into the artifact.",
    )
    p.add_argument(
        "--cv-embargo-days", type=int, default=60,
        help="Trading-day embargo between each train window and validation fold.",
    )
    p.add_argument(
        "--cutoff-embargo-days", type=int, default=None,
        help="Trading-day label embargo before --train-cutoff. Defaults to the "
             "lookahead encoded in --label, e.g. fwd_60d_excess -> 60.",
    )
    p.add_argument(
        "--skip-cv", action="store_true",
        help="Emergency only: skip OOS contract evaluation. Production strict "
             "contract will fail when this is used.",
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


def infer_label_lookahead_days(label: str) -> int:
    """Infer label lookahead from names such as fwd_60d_excess."""
    m = re.search(r"fwd_(\d+)d", str(label))
    return int(m.group(1)) if m else 60


def load_and_slice_panel(cutoff_date: Optional[pd.Timestamp],
                         watchlist_file: Optional[str] = None,
                         label_override: Optional[str] = None,
                         cutoff_embargo_days: Optional[int] = None) -> tuple[pd.DataFrame, list[str], str]:
    """Load alpha158 panel, optionally filter by cutoff/watchlist, return (train_df, feat_cols, label_used)."""
    label_used = label_override or LABEL
    log.info("Loading R1K + 5-fund panel (already normalized: alpha158=zscore, fund=robust-zscore)...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]

    # Watchlist filter (Track 1 wl retrain)
    if watchlist_file:
        wl_data = json.loads(Path(watchlist_file).read_text())
        wl = wl_data.get("watchlist") or wl_data.get("proposed_watchlist") or wl_data
        if isinstance(wl, list):
            n_before = panel["ticker"].nunique()
            panel = panel[panel["ticker"].isin(wl)].copy()
            log.info("Watchlist filter (%s): tickers %d → %d (matched %d)",
                     watchlist_file, n_before, panel["ticker"].nunique(),
                     len(set(wl) & set(panel["ticker"].unique())))

    train = panel.dropna(subset=[label_used])
    if cutoff_date is not None:
        before = len(train)
        embargo_days = (
            infer_label_lookahead_days(label_used)
            if cutoff_embargo_days is None else int(cutoff_embargo_days)
        )
        effective_cutoff = cutoff_date - pd.offsets.BDay(max(0, embargo_days))
        train = train[train["date"] < effective_cutoff]
        log.info("Cutoff filter: cutoff=%s embargo=%dBD effective<%s — %d → %d rows (max date %s)",
                 cutoff_date.date().isoformat(), embargo_days,
                 effective_cutoff.date().isoformat(), before, len(train),
                 train["date"].max().date() if len(train) else "EMPTY")
        if len(train) == 0:
            raise SystemExit(
                f"No training rows with date < {effective_cutoff.date()} "
                f"(cutoff={cutoff_date.date()}, embargo={embargo_days}BD)"
            )

    log.info("Train rows: %d (panel total: %d), tickers: %d, dates: %s → %s, label: %s",
             len(train), len(panel), train["ticker"].nunique(),
             train["date"].min().date(), train["date"].max().date(), label_used)
    return train, feat_cols, label_used


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


def train_xgb(train: pd.DataFrame, feat_cols: list[str], label: str = LABEL) -> tuple[xgb.Booster, float]:
    """Train rank:pairwise XGB and return (booster, in-sample IC). Label param added 2026-05-13."""
    Xtr = train[feat_cols].fillna(0).values.astype(np.float64)
    ytr = train[label].clip(-5,5).values.astype(np.float64)

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
        ic, _ = spearmanr(g["pred"], g[label])
        if not np.isnan(ic): train_ics.append(ic)
    train_ic_mean = float(np.mean(train_ics)) if train_ics else float("nan")
    log.info("In-sample train IC: %+.4f (sanity check, not OOS)", train_ic_mean)
    return booster, train_ic_mean


def cross_sectional_ic(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> dict:
    """Mean daily Spearman IC for a prediction vector."""
    from scipy.stats import spearmanr

    df = pd.DataFrame({"pred": pred, "y": y, "date": dates})
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g["pred"], g["y"])
        if not np.isnan(ic):
            ics.append(float(ic))
    return {
        "mean_ic": float(np.mean(ics)) if ics else float("nan"),
        "n_dates": int(len(ics)),
        "per_date_ic": ics,
    }


def evaluate_walk_forward_cv(
    train: pd.DataFrame,
    feat_cols: list[str],
    *,
    label: str = LABEL,
    n_splits: int = 3,
    embargo_days: int = 60,
) -> dict:
    """Purged expanding-window CV used for artifact contract metadata.

    Each fold trains only on dates strictly before the validation fold,
    leaving ``embargo_days`` trading dates between train and validation.
    """
    n_splits = max(1, int(n_splits))
    embargo_days = max(0, int(embargo_days))
    dates = np.array(sorted(pd.to_datetime(train["date"].unique())))
    if len(dates) < (n_splits + 1) * 5:
        raise ValueError(f"not enough dates for {n_splits} folds: {len(dates)}")

    fold_indices = np.array_split(np.arange(len(dates)), n_splits + 1)[1:]
    folds = []
    for fold_no, val_idx in enumerate(fold_indices, start=1):
        if len(val_idx) == 0:
            continue
        train_end_pos = int(val_idx[0]) - embargo_days
        if train_end_pos <= 0:
            log.warning("CV fold %d skipped: embargo leaves no train dates", fold_no)
            continue
        tr_dates = set(dates[:train_end_pos])
        va_dates = set(dates[val_idx])
        tr = train[train["date"].isin(tr_dates)]
        va = train[train["date"].isin(va_dates)]
        if tr["date"].nunique() < 20 or va.empty:
            log.warning(
                "CV fold %d skipped: n_train_dates=%d n_val_rows=%d",
                fold_no, tr["date"].nunique(), len(va),
            )
            continue

        booster, train_ic = train_xgb(tr, feat_cols, label=label)
        pred = booster.predict(xgb.DMatrix(va[feat_cols].fillna(0).values.astype(np.float64)))
        y = va[label].clip(-5, 5).values.astype(np.float64)
        ic_info = cross_sectional_ic(pred, y, va["date"].values)
        fold_ic = float(ic_info["mean_ic"])
        folds.append({
            "fold": fold_no,
            "train_start": pd.Timestamp(tr["date"].min()).date().isoformat(),
            "train_end": pd.Timestamp(tr["date"].max()).date().isoformat(),
            "val_start": pd.Timestamp(va["date"].min()).date().isoformat(),
            "val_end": pd.Timestamp(va["date"].max()).date().isoformat(),
            "n_train_rows": int(len(tr)),
            "n_val_rows": int(len(va)),
            "train_ic": float(train_ic),
            "ic": fold_ic,
            "n_ic_dates": int(ic_info["n_dates"]),
        })
        log.info("CV fold %d/%d IC=%+.4f train_dates=%d val_dates=%d",
                 fold_no, n_splits, fold_ic, tr["date"].nunique(), va["date"].nunique())

    per_fold = [f["ic"] for f in folds if np.isfinite(f["ic"])]
    if not per_fold:
        raise ValueError("walk-forward CV produced no finite folds")
    return {
        "cv_method": "purged_walk_forward",
        "cv_n_splits": n_splits,
        "cv_embargo_days": embargo_days,
        "oos_mean_ic": float(np.mean(per_fold)),
        "oos_std_ic": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
        "oos_per_fold_ic": [float(v) for v in per_fold],
        "folds": folds,
    }


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
                   side_label: Optional[str],
                   *,
                   label_used: str = LABEL,
                   train_ic: float | None = None,
                   cv_result: dict | None = None,
                   cutoff_embargo_days: int | None = None,
                   train_run_id: str | None = None) -> dict:
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
        "label_col": label_used,
        "lookahead_days": 60,
        "train_run_id": train_run_id,
        "training_train_ic": train_ic,
        "training_notes": notes,
    }
    if cv_result:
        artifact.update({
            "oos_mean_ic": cv_result.get("oos_mean_ic"),
            "oos_std_ic": cv_result.get("oos_std_ic"),
            "oos_per_fold_ic": cv_result.get("oos_per_fold_ic"),
            "cv_method": cv_result.get("cv_method"),
            "cv_n_splits": cv_result.get("cv_n_splits"),
            "cv_embargo_days": cv_result.get("cv_embargo_days"),
            "cv_folds": cv_result.get("folds"),
            "eval_ic": (
                cv_result.get("oos_per_fold_ic")[-1]
                if cv_result.get("oos_per_fold_ic") else None
            ),
        })
    if cutoff_date is not None:
        artifact["cutoff_date"] = cutoff_date.isoformat()
        artifact["cutoff_embargo_days"] = int(
            infer_label_lookahead_days(label_used)
            if cutoff_embargo_days is None else cutoff_embargo_days
        )
        artifact["effective_train_cutoff_date"] = (
            cutoff_date - pd.offsets.BDay(artifact["cutoff_embargo_days"])
        ).isoformat()
    if side_label is not None:
        artifact["side_label"] = side_label
    return artifact


def attach_inference_smoke(artifact: dict, booster: xgb.Booster,
                           feat_cols: list[str]) -> None:
    """Stamp deterministic scorer smoke evidence used by acceptance gates.

    The sample is synthetic by design: it tests serialization/load-time score
    mechanics and output diversity without depending on the current market
    data cache. Walk-forward/sim gates remain responsible for economic value.
    """
    rng = np.random.default_rng(104)
    X = rng.standard_normal((32, len(feat_cols))).astype(np.float64)
    scores = booster.predict(xgb.DMatrix(X))
    finite = np.isfinite(scores)
    md = artifact.setdefault("metadata", {})
    md["score_sample_range"] = [
        float(np.nanmin(scores)) if len(scores) else float("nan"),
        float(np.nanmax(scores)) if len(scores) else float("nan"),
    ]
    md["inference_smoke_test"] = {
        "n": int(len(scores)),
        "all_finite": bool(finite.all()) if len(scores) else False,
        "n_unique": int(len(set(np.round(scores[finite], 12)))) if finite.any() else 0,
    }


def main():
    args = parse_args()
    cutoff_date, out_path, is_walkforward = resolve_paths(args)

    train, feat_cols, label_used = load_and_slice_panel(
        cutoff_date, watchlist_file=args.watchlist_file, label_override=args.label,
        cutoff_embargo_days=args.cutoff_embargo_days,
    )
    mu, sd, _ = build_normalization(train, feat_cols)
    cv_result = None
    if not args.skip_cv:
        cv_result = evaluate_walk_forward_cv(
            train,
            feat_cols,
            label=label_used,
            n_splits=args.cv_n_splits,
            embargo_days=args.cv_embargo_days,
        )
        log.info(
            "OOS contract CV: mean_ic=%+.4f std=%+.4f folds=%s",
            cv_result["oos_mean_ic"],
            cv_result["oos_std_ic"],
            [round(x, 4) for x in cv_result["oos_per_fold_ic"]],
        )
    else:
        log.warning("--skip-cv set: artifact will not satisfy strict contract")

    booster, train_ic = train_xgb(train, feat_cols, label=label_used)
    artifact = build_artifact(booster, feat_cols, mu, sd, train,
                              cutoff_date, args.side_label,
                              label_used=label_used,
                              train_ic=train_ic,
                              cv_result=cv_result,
                              cutoff_embargo_days=args.cutoff_embargo_days,
                              train_run_id=str(uuid.uuid4())[:8])

    if not is_walkforward:
        fp = stamp_fingerprint(artifact)
        log.info("Fingerprint: %s", fp)
    else:
        log.info("Walk-forward artifact — skipping fingerprint stamp (§5.13.13).")
    attach_inference_smoke(artifact, booster, feat_cols)
    smoke = artifact.get("metadata", {}).get("inference_smoke_test", {})
    log.info(
        "Inference smoke: n=%s all_finite=%s n_unique=%s range=%s",
        smoke.get("n"),
        smoke.get("all_finite"),
        smoke.get("n_unique"),
        artifact.get("metadata", {}).get("score_sample_range"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact))
    log.info("Saved artifact: %s  (size=%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    log.info("Feature cols (n=%d): %s ... %s", len(feat_cols), feat_cols[:3], feat_cols[-3:])


if __name__ == "__main__":
    main()
