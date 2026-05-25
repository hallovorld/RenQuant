#!/usr/bin/env python
"""P4.3 — Train the meta-label XGBoost classifier.

Reads the labeled snapshot parquet from P4.2 and trains an XGBoost
binary classifier with PurgedKFold cross-validation per López de Prado
AFML 2018 ch.7 Snippets 7.3 and 7.4.

The trained artifact is written to:
    backtesting/renquant_104/artifacts/meta-label-exit.json

containing:
    {
        "version": 1,
        "kind": "meta_label_exit_xgb",
        "feature_cols": [...],
        "booster_raw_json": <xgboost serialized model>,
        "default_threshold": <float in [0,1]>,
        "cv_metrics": {
            "auc_mean": <float>, "auc_std": <float>,
            "precision_at_05_mean": <float>,
            "recall_at_05_mean": <float>,
        },
        "training_data_summary": {
            "n_events": <int>,
            "class_balance": <float>,
            "fwd_window_days": <int>,
        }
    }

References
----------
* López de Prado 2018 AFML ch.7 (PurgedKFold) + ch.20 (Meta-Labeling)
* doc/research/meta-labeling-exit-policy.md §6
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO  = Path(__file__).resolve().parent.parent
STRAT = REPO / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("meta-label-train")

PATH_TRIGGER_COLUMNS: tuple[str, ...] = (
    "trigger_stop_loss",
    "trigger_trailing_stop",
    "trigger_single_day_loss",
    "trigger_max_hold",
)


def select_path_rule_training_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return labeled rows matching MetaLabelVetoTask's inference surface.

    Historical snapshot logs set ``any_trigger=1`` for some model/QP exits
    while the runtime veto only ever sees path-rule exits. Training on those
    rows mixes a different decision problem into the classifier and weakens the
    artifact. The training set is therefore exactly labeled rows with one of
    the canonical path-trigger columns set.
    """
    if df.empty or "meta_label" not in df.columns:
        return df.iloc[0:0].copy()
    work = df[df["meta_label"].notna()].copy()
    if work.empty:
        return work
    mask = pd.Series(False, index=work.index)
    for col in PATH_TRIGGER_COLUMNS:
        if col in work.columns:
            mask |= pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int).eq(1)
    return work[mask].copy().reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--labels", required=True,
                   help="Input parquet from _meta_label_generate.py")
    p.add_argument("--out", default=str(
        STRAT / "artifacts" / "meta-label-exit.json"))
    p.add_argument("--n-splits", type=int, default=5,
                   help="PurgedKFold n_splits (default 5)")
    p.add_argument("--label-horizon-days", type=int, default=20,
                   help="Label horizon = triple-barrier vertical (default 20)")
    p.add_argument("--pct-embargo", type=float, default=0.01,
                   help="PurgedKFold embargo fraction (default 0.01)")
    p.add_argument("--max-depth",       type=int,   default=4)
    p.add_argument("--learning-rate",   type=float, default=0.05)
    p.add_argument("--n-estimators",    type=int,   default=200)
    p.add_argument("--subsample",       type=float, default=0.8)
    p.add_argument("--default-threshold", type=float, default=0.5)
    args = p.parse_args()

    # Load + filter to labelled rows
    log.info("Reading labels → %s", args.labels)
    raw_df = pd.read_parquet(args.labels)
    raw_labeled = int(raw_df["meta_label"].notna().sum()) if "meta_label" in raw_df.columns else 0
    df = select_path_rule_training_events(raw_df)
    n_events = len(df)
    if n_events < 100:
        log.error(
            "Only %d path-rule labelled events out of %d labelled rows; "
            "need ≥ 100 for meaningful training.",
            n_events, raw_labeled,
        )
        sys.exit(1)
    balance = float(df["meta_label"].mean())
    log.info("Labelled events: %d  class_balance(positive)=%.2f%%",
             n_events, balance * 100)

    # Feature columns: all FEATURE_COLUMNS except identifiers and outcomes
    from kernel.meta_label.snapshot import FEATURE_COLUMNS  # noqa: PLC0415
    feature_cols = [
        c for c in FEATURE_COLUMNS
        if c not in {"date", "ticker", "fwd_5d_ret", "fwd_20d_ret"}
    ]
    log.info("Feature count: %d  (e.g., %s …)",
             len(feature_cols), feature_cols[:5])

    # Sort by event date so PurgedKFold's contiguous folds preserve time order
    df["_event_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("_event_dt").reset_index(drop=True)

    X = df[feature_cols].fillna(0.0).values.astype(np.float64)
    y = df["meta_label"].astype(int).values

    from kernel.meta_label.purged_kfold import PurgedKFold  # noqa: PLC0415
    import xgboost as xgb  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score, precision_score, recall_score  # noqa: PLC0415

    cv = PurgedKFold(
        n_splits=args.n_splits,
        event_times=df["_event_dt"],
        label_horizon_days=args.label_horizon_days,
        pct_embargo=args.pct_embargo,
    )

    aucs:        list[float] = []
    precisions:  list[float] = []
    recalls:     list[float] = []

    log.info("Starting %d-fold PurgedKFold (horizon=%d  embargo=%.2f%%) …",
             args.n_splits, args.label_horizon_days, args.pct_embargo * 100)

    for k, (tr_idx, te_idx) in enumerate(cv.split(np.arange(len(X)))):
        if len(tr_idx) < 50 or len(te_idx) < 5:
            log.warning("Fold %d skipped (train=%d test=%d too small)",
                        k, len(tr_idx), len(te_idx))
            continue
        clf = xgb.XGBClassifier(
            max_depth        = args.max_depth,
            learning_rate    = args.learning_rate,
            n_estimators     = args.n_estimators,
            subsample        = args.subsample,
            tree_method      = "hist",
            n_jobs           = -1,
            random_state     = 42,
            eval_metric      = "auc",
            use_label_encoder= False,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        proba = clf.predict_proba(X[te_idx])[:, 1]
        pred  = (proba >= args.default_threshold).astype(int)
        try:
            auc = roc_auc_score(y[te_idx], proba)
        except Exception:
            auc = float("nan")
        prec = precision_score(y[te_idx], pred, zero_division=0)
        rec  = recall_score   (y[te_idx], pred, zero_division=0)
        aucs.append(auc)
        precisions.append(prec)
        recalls.append(rec)
        log.info("  fold %d  train=%d test=%d  AUC=%.3f  prec=%.3f  rec=%.3f",
                 k, len(tr_idx), len(te_idx), auc, prec, rec)

    if not aucs:
        log.error("No folds produced metrics; check label/data integrity.")
        sys.exit(1)

    auc_mean = float(np.nanmean(aucs)); auc_std = float(np.nanstd(aucs))
    log.info("\nCV summary — AUC %.3f ± %.3f  precision@0.5 %.3f ± %.3f  "
             "recall@0.5 %.3f ± %.3f",
             auc_mean, auc_std,
             float(np.mean(precisions)), float(np.std(precisions)),
             float(np.mean(recalls)),    float(np.std(recalls)))

    # ── Threshold sweep — find the F1-optimum threshold (not just 0.5)
    log.info("\nThreshold sweep on out-of-fold predictions …")
    # Reproduce out-of-fold predictions for threshold optimization
    oof_proba = np.full(len(X), np.nan)
    cv_again = PurgedKFold(
        n_splits=args.n_splits, event_times=df["_event_dt"],
        label_horizon_days=args.label_horizon_days, pct_embargo=args.pct_embargo,
    )
    for tr_idx, te_idx in cv_again.split(np.arange(len(X))):
        if len(tr_idx) < 50:
            continue
        clf2 = xgb.XGBClassifier(
            max_depth=args.max_depth, learning_rate=args.learning_rate,
            n_estimators=args.n_estimators, subsample=args.subsample,
            tree_method="hist", n_jobs=-1, random_state=42,
            eval_metric="auc", use_label_encoder=False,
        )
        clf2.fit(X[tr_idx], y[tr_idx])
        oof_proba[te_idx] = clf2.predict_proba(X[te_idx])[:, 1]

    threshold_table = []
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = ~np.isnan(oof_proba)
        if not mask.any():
            continue
        preds = (oof_proba[mask] >= thr).astype(int)
        prec  = precision_score(y[mask], preds, zero_division=0)
        rec   = recall_score   (y[mask], preds, zero_division=0)
        f1    = 2 * prec * rec / max(prec + rec, 1e-9)
        threshold_table.append({
            "threshold": thr, "precision": float(prec),
            "recall": float(rec), "f1": float(f1),
        })
        log.info("  thr=%.2f  prec=%.3f  rec=%.3f  f1=%.3f",
                 thr, prec, rec, f1)
    best_thr = max(threshold_table, key=lambda r: r["f1"])["threshold"] \
        if threshold_table else args.default_threshold
    log.info("F1-optimum threshold: %.2f", best_thr)

    # Final fit on ALL data (after CV evaluation)
    log.info("\nFinal fit on full %d events …", len(X))
    final_clf = xgb.XGBClassifier(
        max_depth        = args.max_depth,
        learning_rate    = args.learning_rate,
        n_estimators     = args.n_estimators,
        subsample        = args.subsample,
        tree_method      = "hist",
        n_jobs           = -1,
        random_state     = 42,
        eval_metric      = "auc",
        use_label_encoder= False,
    )
    final_clf.fit(X, y)

    # ── Feature importance (gain-based per XGBoost) ─────────────────
    log.info("\nFeature importance (top-15 by gain):")
    fi = final_clf.get_booster().get_score(importance_type="gain")
    fi_named = sorted(
        ((feature_cols[int(k.lstrip("f"))], v) for k, v in fi.items()),
        key=lambda x: -x[1],
    )
    fi_payload = [{"feature": n, "gain": float(g)} for n, g in fi_named[:30]]
    for n, g in fi_named[:15]:
        log.info("  %-30s gain=%.4f", n, g)

    booster = final_clf.get_booster()
    booster_raw = booster.save_raw(raw_format="json").decode("utf-8")

    payload = {
        "version": 1,
        "kind":    "meta_label_exit_xgb",
        "trained_date": pd.Timestamp.utcnow().date().isoformat(),
        "feature_cols":     feature_cols,
        "booster_raw_json": booster_raw,
        "default_threshold": float(best_thr),   # F1-optimum, NOT 0.5
        "cv_metrics": {
            "auc_mean":               auc_mean,
            "auc_std":                auc_std,
            "precision_at_05_mean":   float(np.mean(precisions)),
            "recall_at_05_mean":      float(np.mean(recalls)),
            "n_splits":               args.n_splits,
            "threshold_sweep":        threshold_table,
            "best_threshold_by_f1":   float(best_thr),
        },
        "feature_importance":    fi_payload,
        "training_data_summary": {
            "n_events":            n_events,
            "n_raw_labeled_events": raw_labeled,
            "training_event_filter": "path_rule_triggers_only",
            "class_balance":       balance,
            "fwd_window_days":     args.label_horizon_days,
            "feature_count":       len(feature_cols),
        },
        "references": {
            "method": "Meta-labeling per López de Prado AFML 2018 ch.20",
            "cv":     "PurgedKFold per López de Prado AFML 2018 ch.7.3-7.4",
            "labels": "Triple-barrier per López de Prado AFML 2018 ch.3.4",
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    log.info("Wrote artifact → %s", out_path)


if __name__ == "__main__":
    main()
