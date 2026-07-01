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
import os as _os
from pathlib import Path
from typing import Optional
import re

_THREAD_COUNT = str(_os.cpu_count() or 14)
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_k, _THREAD_COUNT)
_XGB_NTHREAD = int(_os.environ.get("OMP_NUM_THREADS", _THREAD_COUNT))

import numpy as np, pandas as pd, xgboost as xgb
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train-prod")

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))
from kernel.panel_pipeline.feature_transform import transform_feature_frame  # noqa: E402

PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":_XGB_NTHREAD,"verbosity":0,"seed":42}
N_ROUNDS = 100
LABEL = "fwd_60d_excess"
DEFAULT_OUTPUT = REPO / "data" / "panel-ltr-prod-alpha158-fund-fwd60d.json"
# Track B BULL_CALM-regime features. When the upstream panel build runs with
# ``--include-track-b`` the panel parquet contains these columns; this trainer
# DROPS them by default (preserving the baseline 172-feature recipe) and opts
# them back in via the ``--include-features`` flag (comma-list).
TRACK_B_FEATURES: tuple[str, ...] = (
    "mom_carry_12_1", "beta_dm", "rvar_total", "idio_vol_market",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--train-cutoff", type=str, default=None,
        help="ISO date (e.g. 2024-01-01); only rows where panel.date < cutoff are used.",
    )
    p.add_argument(
        "--train-start-date", type=str, default=None,
        help=(
            "RESEARCH-ONLY. ISO date (e.g. 2022-01-01); only rows where "
            "panel.date >= start are used. Defaults to None (use full panel "
            "history). NOT wired to any production training cron / scheduler "
            "/ default driver. Setting this flag produces a NON-PROMOTABLE "
            "artifact per the 2026-06-03 Track D negative finding "
            "(doc/research/2026-06-03-track-d-declare-done-negative.md): "
            "shortening the window lost 0.44 Sharpe vs full-history baseline "
            "because pooled-mean training under-uses high-dispersion BEAR / "
            "CHOPPY rows when the recent slice is BULL_CALM-dominated. The "
            "flag stays in-tree for replication of that negative result and "
            "for future per-regime gradient-weighting experiments. Combined "
            "with --train-cutoff: rows with start <= date < cutoff."
        ),
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
        "--fingerprint-config", type=str, default=None,
        help=(
            "Strategy config whose model-relevant fields are stamped into the "
            "artifact. Defaults to renquant_104/strategy_config.json; pass the "
            "same scoring config used by strict WF/sim when training side configs."
        ),
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
    p.add_argument(
        "--include-features", type=str, default=None,
        help=(
            "Comma-separated opt-in list of addendum feature names (e.g. "
            "'mom_carry_12_1,beta_dm,rvar_total,idio_vol_market' for Track B). "
            "Default: empty — drops every Track B column from feat_cols if "
            "present in the panel, preserving the 172-feature baseline recipe. "
            "Names that do NOT appear in the panel are an error (no silent "
            "rename translation; pin the upstream renquant-base-data version "
            "explicitly — see doc/research/2026-06-02-track-b-feature-audit.md)."
        ),
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
                         cutoff_embargo_days: Optional[int] = None,
                         include_features: Optional[list[str]] = None,
                         train_start_date: Optional[str] = None) -> tuple[pd.DataFrame, list[str], str]:
    """Load alpha158 panel, optionally filter by cutoff/watchlist, return (train_df, feat_cols, label_used).

    ``include_features``: opt-in list for Track B addendum columns. When None
    (default) any Track B column present in the panel is dropped, preserving
    the 172-feature baseline recipe. When supplied, only the listed Track B
    columns are kept (others are dropped). Has no effect on baseline alpha158
    + fund + PEAD + SUE + sentiment columns.

    ``train_start_date``: ISO date lower bound. Rows with ``panel.date <
    train_start_date`` are excluded. Used by Track D regime-drift retraining
    (post-2022 only) where older data is no longer gradient-relevant for
    recent BULL_CALM dynamics. ``None`` (default) preserves full-history
    behaviour.
    """
    label_used = label_override or LABEL
    log.info("Loading R1K + 5-fund panel (already normalized: alpha158=zscore, fund=robust-zscore)...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]
    # Track B opt-in filter: drop any Track B column the user did not opt in to.
    opted_in = set(include_features or [])
    # Fail loudly when ``--include-features`` names a column not in the panel.
    # Catches stale names (e.g. the old ``idio_vol_3f`` after the upstream
    # renquant-base-data #16 rename to ``idio_vol_market``) instead of
    # silently producing a baseline-equivalent run that drops every Track B
    # column. No silent rename translation by design — pin the upstream
    # version explicitly. See doc/research/2026-06-02-track-b-feature-audit.md.
    missing = sorted(opted_in - set(feat_cols))
    if missing:
        # Honest hint when the caller used the pre-#16 name.
        hint = ""
        if "idio_vol_3f" in missing and "idio_vol_market" in feat_cols:
            hint = (" Hint: renquant-base-data #16 renamed 'idio_vol_3f' to "
                    "'idio_vol_market' (SPY+size 2-factor residual; the prior "
                    "'_3f' suffix was a misnomer).")
        raise SystemExit(
            f"--include-features names not present in panel: {missing}. "
            f"Panel columns Track-B-relevant: "
            f"{sorted(set(feat_cols) & set(TRACK_B_FEATURES))}.{hint}"
        )
    drop_track_b = [c for c in TRACK_B_FEATURES if c in feat_cols and c not in opted_in]
    if drop_track_b:
        feat_cols = [c for c in feat_cols if c not in drop_track_b]
        log.info("Track B opt-in: dropped %d addendum columns (%s); kept %d features",
                 len(drop_track_b), drop_track_b, len(feat_cols))
    kept_track_b = [c for c in TRACK_B_FEATURES if c in feat_cols]
    if kept_track_b:
        log.info("Track B opt-in: training with %d addendum features active: %s",
                 len(kept_track_b), kept_track_b)

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
    if train_start_date is not None:
        before = len(train)
        train = train[train["date"] >= pd.Timestamp(train_start_date)]
        log.info("Train-start filter: date>=%s — %d → %d rows (min date %s)",
                 pd.Timestamp(train_start_date).date().isoformat(), before, len(train),
                 train["date"].min().date() if len(train) else "EMPTY")
        if len(train) == 0:
            raise SystemExit(
                f"No training rows with date >= {pd.Timestamp(train_start_date).date()}"
            )
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


def build_normalization(
    train: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[float | None], list[float | None]]:
    """Build the inference normalization chain stored in the artifact.

    For each feature, (mean, std) such that (raw - mean) / std = normalized value.
    alpha158 cols: from panel z-score stats; fund cols: robust z-score on train period.
    """
    ps = json.loads(Path("data/alpha158_qlib_dataset.stats.json").read_text())
    alpha_cols = list(ps["feature_cols"])
    alpha_lows = ps.get("feature_raw_clip_low") or [None] * len(alpha_cols)
    alpha_highs = ps.get("feature_raw_clip_high") or [None] * len(alpha_cols)
    if len(alpha_lows) != len(alpha_cols) or len(alpha_highs) != len(alpha_cols):
        alpha_lows = [None] * len(alpha_cols)
        alpha_highs = [None] * len(alpha_cols)
    alpha_norm = {
        c: {
            "mean": m,
            "std": s,
            "raw_clip_low": lo,
            "raw_clip_high": hi,
        }
        for c, m, s, lo, hi in zip(
            alpha_cols,
            ps["feature_means"],
            ps["feature_stds"],
            alpha_lows,
            alpha_highs,
        )
    }

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
    feat_raw_clip_low: list[float | None] = []
    feat_raw_clip_high: list[float | None] = []
    for c in feat_cols:
        if c in alpha_norm:
            rec = alpha_norm[c]
            m, s = rec["mean"], rec["std"]
            feat_means.append(m); feat_stds.append(s); feat_norm_kind.append("global_z")
            feat_raw_clip_low.append(rec["raw_clip_low"])
            feat_raw_clip_high.append(rec["raw_clip_high"])
        elif c in fund_norm:
            m, s = fund_norm[c]
            feat_means.append(m); feat_stds.append(s); feat_norm_kind.append("robust_z")
            feat_raw_clip_low.append(None)
            feat_raw_clip_high.append(None)
        else:
            feat_means.append(0.0); feat_stds.append(1.0); feat_norm_kind.append("identity")
            feat_raw_clip_low.append(None)
            feat_raw_clip_high.append(None)
    log.info("Normalization chain: %d global_z, %d robust_z, %d identity",
             feat_norm_kind.count("global_z"), feat_norm_kind.count("robust_z"),
             feat_norm_kind.count("identity"))
    return (
        np.array(feat_means),
        np.array(feat_stds),
        feat_norm_kind,
        feat_raw_clip_low,
        feat_raw_clip_high,
    )


def _feature_meta(
    mu: np.ndarray,
    sd: np.ndarray,
    kind: list[str],
    raw_clip_low: list[float | None] | None = None,
    raw_clip_high: list[float | None] | None = None,
) -> dict:
    meta = {
        "feature_means": np.asarray(mu, dtype=float).tolist(),
        "feature_stds": np.asarray(sd, dtype=float).tolist(),
        "feature_norm_kind": list(kind),
    }
    if raw_clip_low is not None and raw_clip_high is not None:
        meta["feature_raw_clip_low"] = list(raw_clip_low)
        meta["feature_raw_clip_high"] = list(raw_clip_high)
        meta["feature_raw_clip_fit_split"] = "train"
        meta["feature_preprocess_version"] = 2
    return meta


def panel_training_matrix(
    frame: pd.DataFrame,
    feat_cols: list[str],
    mu: np.ndarray,
    sd: np.ndarray,
    norm_kind: list[str],
) -> pd.DataFrame:
    return transform_feature_frame(
        frame.reindex(columns=feat_cols, fill_value=float("nan")),
        feat_cols,
        _feature_meta(mu, sd, norm_kind),
        source_space="panel",
    )


def train_xgb(
    train: pd.DataFrame,
    feat_cols: list[str],
    label: str = LABEL,
    *,
    feature_means: np.ndarray | None = None,
    feature_stds: np.ndarray | None = None,
    feature_norm_kind: list[str] | None = None,
) -> tuple[xgb.Booster, float]:
    """Train rank:pairwise XGB and return (booster, in-sample IC). Label param added 2026-05-13."""
    if feature_means is not None and feature_stds is not None and feature_norm_kind is not None:
        Xdf = panel_training_matrix(train, feat_cols, feature_means, feature_stds, feature_norm_kind)
    else:
        Xdf = train.reindex(columns=feat_cols, fill_value=0).fillna(0)
    Xtr = Xdf.values.astype(np.float64)
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

        mu, sd, norm_kind, _, _ = build_normalization(tr, feat_cols)
        booster, train_ic = train_xgb(
            tr,
            feat_cols,
            label=label,
            feature_means=mu,
            feature_stds=sd,
            feature_norm_kind=norm_kind,
        )
        Xva = panel_training_matrix(va, feat_cols, mu, sd, norm_kind)
        pred = booster.predict(xgb.DMatrix(Xva.values.astype(np.float64)))
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


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def _load_json(path: str | Path) -> object:
    return json.loads(_resolve_repo_path(path).read_text())


def _watchlist_from_payload(payload: object) -> list[str] | None:
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict):
        wl = payload.get("watchlist") or payload.get("proposed_watchlist")
        if isinstance(wl, list):
            return [str(x) for x in wl]
    return None


def build_fingerprint_config(
    *,
    fingerprint_config_path: str | None,
    watchlist_file: str | None,
    label_used: str,
    feat_cols: list[str],
) -> dict:
    """Build the exact model-relevant config stamp for this artifact.

    Walk-forward artifacts are used by strict sim/WF gates, so they must carry
    the same config contract as production artifacts. The stamp is derived from
    the requested scoring config, then normalized to the actual training inputs
    that this script controls (label horizon, objective, resolution, embedding
    columns, and optional watchlist filter).
    """
    if fingerprint_config_path:
        # 2026-05-27: a strategy-dir-relative name like "strategy_config.json"
        # previously resolved ONLY against repo root (non-existent there), then
        # build_fingerprint_config silently fell back to an empty {} config →
        # empty watchlist/sector_map → a bogus config_fingerprint (the gate then
        # fail-closed on panel_scorer_config_mismatch). Resolve against repo root
        # first, then the strategy dir, and FAIL LOUD if neither exists.
        cfg_path = _resolve_repo_path(fingerprint_config_path)
        if not cfg_path.exists():
            strategy_rel = STRATEGY_DIR / fingerprint_config_path
            if strategy_rel.exists():
                cfg_path = strategy_rel
            else:
                raise FileNotFoundError(
                    f"--fingerprint-config {fingerprint_config_path!r} not found at "
                    f"{cfg_path} or {strategy_rel}; refusing to stamp an empty-config "
                    "fingerprint (would fail-close the panel scorer)."
                )
    else:
        cfg_path = STRATEGY_DIR / "strategy_config.json"
    cfg = json.loads(cfg_path.read_text())

    if watchlist_file:
        payload = _load_json(watchlist_file)
        wl = _watchlist_from_payload(payload)
        if wl is not None:
            cfg["watchlist"] = wl
        if isinstance(payload, dict):
            for key in ("benchmark", "sector_map", "sector_etf_map"):
                if key in payload:
                    cfg[key] = payload[key]

    panel = cfg.setdefault("panel_ltr", {})
    panel["lookahead_days"] = infer_label_lookahead_days(label_used)
    panel["training_resolution"] = "daily"
    panel.setdefault("hourly", {})["enabled"] = False
    panel.setdefault("minute", {})["enabled"] = False
    panel.setdefault("asset_embeddings", {})["enabled"] = any(
        str(c).startswith("emb_") for c in feat_cols
    )
    panel.setdefault("xgb_params", {})["objective"] = PARAMS["objective"]
    return cfg


def build_sentiment_training_regime_map(
    dates,
    runtime_config: dict,
) -> dict[pd.Timestamp, str]:
    """Replay the production regime chain for training rows.

    The production alpha158 trainer is separate from PanelTrainingPipeline, but
    must honor the same sentiment runtime gate contract. We reuse the existing
    regime replay helper instead of inventing a second detector.
    """
    from types import SimpleNamespace  # noqa: PLC0415
    from training_panel.pp_panel_training import _build_training_regime_map  # noqa: PLC0415

    spy_path = REPO / "data" / "ohlcv" / "SPY" / "1d.parquet"
    if not spy_path.exists():
        raise FileNotFoundError(f"SPY OHLCV missing for sentiment gate: {spy_path}")
    spy_df = pd.read_parquet(spy_path)
    cfg = dict(runtime_config)
    cfg.setdefault("_strategy_dir", str(STRATEGY_DIR))
    ctx = SimpleNamespace(config=cfg, spy_df=spy_df, strategy_dir=STRATEGY_DIR)
    return _build_training_regime_map(ctx, dates)


def apply_sentiment_training_gate(
    train: pd.DataFrame,
    feat_cols: list[str],
    runtime_config: dict,
    regime_by_date: dict,
) -> tuple[pd.DataFrame, dict]:
    """Zero sentiment features in rows where runtime would disable sentiment."""
    from kernel.artifact_contract import sentiment_runtime_gate_requirement  # noqa: PLC0415

    req = sentiment_runtime_gate_requirement({"feature_cols": feat_cols}, runtime_config)
    if not req["required"]:
        return train, {}
    if "date" not in train.columns:
        raise ValueError("sentiment runtime gate requires date column in training panel")
    if not regime_by_date:
        raise ValueError("sentiment runtime gate requires regime labels")

    row_dates = pd.to_datetime(train["date"]).dt.normalize()
    from training_panel.pp_panel_training import _sentiment_gate_masks  # noqa: PLC0415
    try:
        row_regimes, warmup_missing, warmup_zeroed = _sentiment_gate_masks(
            row_dates,
            regime_by_date,
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc

    disabled = set(req["disabled_regimes"])
    mask = row_regimes.isin(disabled) | warmup_missing
    out = train.copy()
    for col in req["sentiment_feature_cols"]:
        if col in out.columns:
            out.loc[mask, col] = 0.0

    return out, {
        "sentiment_runtime_gate_contract": "trained_zeroing",
        "sentiment_runtime_gate_feature_cols": list(req["sentiment_feature_cols"]),
        "sentiment_runtime_gate_disabled_regimes": list(req["disabled_regimes"]),
        "sentiment_runtime_gate_zeroed_rows": int(mask.sum()),
        "sentiment_runtime_gate_warmup_zeroed_rows": int(warmup_zeroed),
        "sentiment_runtime_gate_missing_regime_policy": "warmup_zero_only",
        "sentiment_runtime_gate_policy": req["effective_policy"],
    }


def stamp_fingerprint(
    artifact: dict,
    *,
    fingerprint_config_path: str | None = None,
    watchlist_file: str | None = None,
    label_used: str = LABEL,
    feat_cols: list[str] | None = None,
) -> str:
    """Stamp config_fingerprint_fields + config_fingerprint for all artifacts."""
    sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))
    from kernel.config_consistency import (  # noqa: PLC0415
        _model_relevant_fields,
        fingerprint_config,
    )
    cfg = build_fingerprint_config(
        fingerprint_config_path=fingerprint_config_path,
        watchlist_file=watchlist_file,
        label_used=label_used,
        feat_cols=list(feat_cols or artifact.get("feature_cols") or []),
    )
    artifact["config_fingerprint_fields"] = _model_relevant_fields(cfg)
    fp = fingerprint_config(cfg)
    artifact["config_fingerprint"] = fp
    artifact.setdefault("metadata", {})["config_fingerprint_source"] = {
        "fingerprint_config_path": str(fingerprint_config_path or "strategy_config.json"),
        "watchlist_file": str(watchlist_file) if watchlist_file else None,
        "label_used": label_used,
        "feature_count": len(feat_cols or artifact.get("feature_cols") or []),
    }
    return fp


def build_artifact(booster: xgb.Booster, feat_cols: list[str],
                   mu: np.ndarray, sd: np.ndarray, train: pd.DataFrame,
                   cutoff_date: Optional[pd.Timestamp],
                   side_label: Optional[str],
                   *,
                   feature_norm_kind: list[str] | None = None,
                   feature_raw_clip_low: list[float | None] | None = None,
                   feature_raw_clip_high: list[float | None] | None = None,
                   label_used: str = LABEL,
                   train_ic: float | None = None,
                   cv_result: dict | None = None,
                   cutoff_embargo_days: int | None = None,
                   train_run_id: str | None = None,
                   sentiment_contract_metadata: dict | None = None,
                   train_start_date: Optional[str] = None) -> dict:
    """Build artifact dict, stamping cutoff_date + side_label when set.

    Always stamps ``effective_train_cutoff_date`` +
    ``effective_selection_cutoff_date`` (the binding information-set DATA
    cutoff) so the freshness monitor (orchestrator #213) and the
    renquant-pipeline P-MODEL-STALENESS gate can measure panel staleness
    against the DATA cutoff rather than soft-skipping. On the full-history
    production path this is the max LABELED training date (the fwd_60d-clipped
    panel max, derived from the frame — never wall-clock ``trained_date``); on
    the walk-forward path it is the pre-embargo cutoff boundary. See the block
    below for details.

    When ``train_start_date`` is provided (Track D regime-drift retrains),
    the artifact additionally stamps a machine-readable lower-bound
    provenance triplet so audits + gates can distinguish full-history vs
    recent-history retrains that share the same label / feature / config
    fingerprint:

    - ``train_start_date``: user-supplied ISO string (lower bound).
    - ``effective_train_start_date``: equal to ``train_start_date`` today
      (no lower-bound embargo applies; kept distinct for symmetry with
      ``effective_train_cutoff_date`` and future-proofing if a lower-bound
      embargo is ever needed).
    - ``train_window``: ``{"start": ..., "end": ...}`` mirror — ``start`` is
      the effective lower bound, ``end`` is ``effective_train_cutoff_date``
      when a cutoff is set, otherwise the observed max train date.

    When ``train_start_date`` is ``None`` these fields are OMITTED (not
    stamped as ``null``) so default full-history artifacts remain
    byte-equivalent to pre-extension output. The model-relevant
    ``config_fingerprint`` is intentionally unchanged — recipe identity
    (labels / features / config) still maps to one fingerprint; window
    provenance is metadata-level and read by gates that care about row
    coverage (§7.6 data-flow safety).
    """
    raw_json = bytes(booster.save_raw(raw_format="json")).decode("utf-8")
    lookahead_days = infer_label_lookahead_days(label_used)
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
        "feature_norm_kind": list(feature_norm_kind or ["legacy_full_z"] * len(feat_cols)),
        "feature_source_contract": {
            "raw": (
                "apply feature_raw_clip_low/high when present, then "
                "feature_means/stds, fillna, and z-clip before scoring "
                "live/sim rows"
            ),
            "panel": "apply only feature_norm_kind entries that are raw in the prebuilt panel",
        },
        "params": PARAMS,
        "best_iter": N_ROUNDS,
        "booster_raw_json": raw_json,
        "panel_shape": {
            "rows":    int(train.shape[0]),
            "tickers": int(train["ticker"].nunique()),
            "dates":   int(train["date"].nunique()),
        },
        "label_col": label_used,
        "lookahead_days": lookahead_days,
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
    if feature_raw_clip_low is not None and feature_raw_clip_high is not None:
        if len(feature_raw_clip_low) != len(feat_cols) or len(feature_raw_clip_high) != len(feat_cols):
            raise ValueError("feature_raw_clip_low/high length must match feature_cols")
        artifact["feature_raw_clip_low"] = list(feature_raw_clip_low)
        artifact["feature_raw_clip_high"] = list(feature_raw_clip_high)
        artifact["feature_raw_clip_fit_split"] = "train"
        artifact["feature_preprocess_version"] = 2
    # ── Information-set DATA cutoff provenance (freshness monitor
    # orchestrator #213 + renquant-pipeline P-MODEL-STALENESS gate) ──
    # The staleness rail MUST key on the DATA cutoff, never the wall-clock
    # ``trained_date``: a fresh trained_date over stale labeled data is NOT
    # fresh (the #210/#212 lesson). Stamp ``effective_train_cutoff_date`` on
    # BOTH paths so the gate can measure panel staleness instead of
    # soft-skipping:
    #   * walk-forward retrain (``--train-cutoff`` set): the information-set
    #     cutoff is the pre-embargo boundary ``cutoff_date - embargo``.
    #   * full-history production retrain (no cutoff): the panel was already
    #     ``dropna``'d on the fwd_60d label, so ``train["date"].max()`` IS
    #     the fwd-clipped information set — derive it from the training
    #     frame, NEVER ``datetime.now()``/``trained_date``.
    if cutoff_date is not None:
        artifact["cutoff_date"] = cutoff_date.isoformat()
        artifact["cutoff_embargo_days"] = int(
            lookahead_days if cutoff_embargo_days is None else cutoff_embargo_days
        )
        effective_train_cutoff_iso = (
            cutoff_date - pd.offsets.BDay(artifact["cutoff_embargo_days"])
        ).isoformat()
    else:
        effective_train_cutoff_iso = pd.Timestamp(train["date"].max()).isoformat()
    artifact["effective_train_cutoff_date"] = effective_train_cutoff_iso
    # The panel has NO separate held-out model-selection window (CV is purged
    # walk-forward WITHIN the clipped panel, and the final model trains on the
    # full labeled panel), so the selection cutoff equals the train cutoff.
    # Stamp the alias the freshness monitor reads first
    # (kernel/walk_forward/lean_guard.py:_selection_anchor prefers
    # ``effective_selection_cutoff_date``) so both the monitor and the gate
    # resolve the same data cutoff regardless of which field they key on.
    artifact["effective_selection_cutoff_date"] = effective_train_cutoff_iso
    if train_start_date is not None:
        start_ts = pd.Timestamp(train_start_date)
        effective_start_iso = start_ts.isoformat()
        artifact["train_start_date"] = effective_start_iso
        # No lower-bound embargo applies today; effective == requested.
        # Kept as a distinct field for symmetry with
        # ``effective_train_cutoff_date`` and future-proofing.
        artifact["effective_train_start_date"] = effective_start_iso
        # train_window mirrors {effective_start, effective_end} so audits
        # can read both bounds from one field. ``end`` echoes the
        # ``effective_train_cutoff_date`` stamped above — the walk-forward
        # boundary when a cutoff is set, else the observed max labeled date.
        artifact["train_window"] = {
            "start": effective_start_iso,
            "end":   effective_train_cutoff_iso,
        }
    if side_label is not None:
        artifact["side_label"] = side_label
    if sentiment_contract_metadata:
        artifact.update(sentiment_contract_metadata)
    # Track B recipe stamp: any active Track B feature in feat_cols pins the
    # artifact so the WF gate recipe-match check distinguishes the variant
    # from baseline. See doc/research/2026-06-02-track-b-feature-audit.md.
    track_b_active = [c for c in TRACK_B_FEATURES if c in feat_cols]
    if track_b_active:
        artifact["feature_addendum_v1"] = {
            "track_b_features_active": track_b_active,
            "source": "renquant-base-data:track_b_features",
            "memo": "doc/research/2026-06-02-track-b-feature-audit.md",
        }
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

    include_features: Optional[list[str]] = None
    if args.include_features:
        include_features = [
            c.strip() for c in args.include_features.split(",") if c.strip()
        ]
    train, feat_cols, label_used = load_and_slice_panel(
        cutoff_date, watchlist_file=args.watchlist_file, label_override=args.label,
        cutoff_embargo_days=args.cutoff_embargo_days,
        include_features=include_features,
        train_start_date=args.train_start_date,
    )
    fingerprint_cfg = build_fingerprint_config(
        fingerprint_config_path=args.fingerprint_config,
        watchlist_file=args.watchlist_file,
        label_used=label_used,
        feat_cols=feat_cols,
    )
    regime_map = build_sentiment_training_regime_map(
        train["date"].unique(),
        fingerprint_cfg,
    )
    train, sentiment_contract = apply_sentiment_training_gate(
        train,
        feat_cols,
        fingerprint_cfg,
        regime_map,
    )
    if sentiment_contract:
        log.info(
            "Sentiment gate contract: trained_zeroing rows=%d disabled_regimes=%s",
            sentiment_contract["sentiment_runtime_gate_zeroed_rows"],
            sentiment_contract["sentiment_runtime_gate_disabled_regimes"],
        )
    mu, sd, norm_kind, raw_clip_low, raw_clip_high = build_normalization(train, feat_cols)
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

    booster, train_ic = train_xgb(
        train,
        feat_cols,
        label=label_used,
        feature_means=mu,
        feature_stds=sd,
        feature_norm_kind=norm_kind,
    )
    artifact = build_artifact(booster, feat_cols, mu, sd, train,
                              cutoff_date, args.side_label,
                              feature_norm_kind=norm_kind,
                              feature_raw_clip_low=raw_clip_low,
                              feature_raw_clip_high=raw_clip_high,
                              label_used=label_used,
                              train_ic=train_ic,
                              cv_result=cv_result,
                              cutoff_embargo_days=args.cutoff_embargo_days,
                              train_run_id=str(uuid.uuid4())[:8],
                              sentiment_contract_metadata=sentiment_contract,
                              train_start_date=args.train_start_date)

    fp = stamp_fingerprint(
        artifact,
        fingerprint_config_path=args.fingerprint_config,
        watchlist_file=args.watchlist_file,
        label_used=label_used,
        feat_cols=feat_cols,
    )
    log.info("Fingerprint: %s%s", fp, " (walk-forward)" if is_walkforward else "")
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
