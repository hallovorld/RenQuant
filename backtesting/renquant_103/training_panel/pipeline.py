"""Stage-1 training orchestrator for the panel-LTR pipeline.

Strings together modules 9.1–9.7:

  1. Compute sector-ETF momentum features (9.3)
  2. Neutralize per-ticker momentum/trend features (9.3)
  3. Build cross-sectional factor bundle (9.5)
  4. Compute forward returns → residualize vs SPY/sector → Gaussianize (9.2)
  5. Apply min-history gate (9.4)
  6. Assemble panel + group_sizes + weights + missingness (9.1)
  7. Purged K-fold CV for mean IC (9.6)
  8. Fit final model on full panel (9.7)
  9. Save JSON artifact with CV metadata (9.7)

Inputs are *already-computed* per-ticker feature frames (from the existing
`training/features.py`) plus raw OHLCV. This keeps the orchestrator
agnostic to upstream data-fetching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training_panel.panel_frame import build_panel_frame
from training_panel.labels import build_labels
from training_panel.neutralization import (
    NEUTRALIZE_COLS, compute_sector_momentum, neutralize_features,
)
from training_panel.factors import build_factor_bundle
from training_panel.imputation import apply_min_history_gate
from training_panel.purged_cv import PurgedKFold, cross_validated_ic
from training_panel.ltr_model import PanelLTRModel


def _compute_fwd_returns(
    ohlcv: dict[str, pd.DataFrame], lookahead_days: int,
) -> dict[str, pd.Series]:
    """Forward simple return: (close[t+L] / close[t]) − 1, indexed at t."""
    out: dict[str, pd.Series] = {}
    for t, df in ohlcv.items():
        close = df["close"].astype(float)
        fwd = close.shift(-lookahead_days) / close - 1.0
        out[t] = fwd
    return out


def _sector_returns_by_ticker(
    ticker_sectors: dict[str, str],
    sector_etf_ohlcv: dict[str, pd.DataFrame],
    lookahead_days: int,
) -> dict[str, pd.Series]:
    """Per-ticker forward return of that ticker's sector ETF."""
    sec_fwd: dict[str, pd.Series] = {}
    for sec, df in sector_etf_ohlcv.items():
        close = df["close"].astype(float)
        sec_fwd[sec] = close.shift(-lookahead_days) / close - 1.0
    out: dict[str, pd.Series] = {}
    for t, sec in ticker_sectors.items():
        if sec in sec_fwd:
            out[t] = sec_fwd[sec]
    return out


def train_panel_model(
    watchlist: list[str],
    feature_frames: dict[str, pd.DataFrame],
    ohlcv: dict[str, pd.DataFrame],
    spy_ohlcv: pd.DataFrame,
    sector_etf_ohlcv: dict[str, pd.DataFrame],
    ticker_sectors: dict[str, str],
    listing_dates: dict[str, pd.Timestamp] | None,
    config: dict[str, Any],
    out_path: Path | str,
) -> dict:
    """End-to-end Stage-1 training. Writes JSON artifact and returns summary."""

    # ── Config (with sensible defaults) ────────────────────────────────────
    lookahead   = int(config.get("lookahead_days", 5))
    beta_window = int(config.get("beta_window", 60))
    min_history = int(config.get("min_history_days", 252))
    age_warmup  = int(config.get("age_warmup_days", 504))
    cv_splits   = int(config.get("cv_n_splits", 5))
    embargo     = int(config.get("cv_embargo_days", 5))
    num_rounds  = int(config.get("num_boost_round", 400))
    neutralize  = bool(config.get("neutralize_features", True))
    nan_cols    = list(config.get("nan_prone_cols", []))
    xgb_params  = dict(config.get("xgb_params", {}))

    # ── 1 & 2. Sector momentum + feature neutralization ───────────────────
    sec_momentum = compute_sector_momentum(sector_etf_ohlcv)
    if neutralize:
        ff = neutralize_features(
            feature_frames, sec_momentum, ticker_sectors,
            cols=NEUTRALIZE_COLS,
        )
    else:
        ff = {t: feature_frames[t].copy() for t in feature_frames}

    # ── 3. Factor bundle ──────────────────────────────────────────────────
    factor_frames = build_factor_bundle(ohlcv, spy_ohlcv)

    # ── 4. Labels: forward returns → residuals → Gaussianized ─────────────
    fwd_returns = _compute_fwd_returns(ohlcv, lookahead)
    sec_fwd_by_t = _sector_returns_by_ticker(
        ticker_sectors, sector_etf_ohlcv, lookahead,
    )
    spy_fwd = spy_ohlcv["close"].astype(float).shift(-lookahead) / spy_ohlcv["close"].astype(float) - 1.0
    labels = build_labels(
        fwd_returns, spy_fwd, sec_fwd_by_t,
        beta_window=beta_window, lookahead_days=lookahead,
    )

    # ── 5. Min-history gate (applied to features; panel_frame also gates) ─
    ff_gated = apply_min_history_gate(ff, min_history_days=0)  # no-op here
    # (panel_frame applies its own `min_history_days` drop internally)

    # ── 6. Assemble panel ────────────────────────────────────────────────
    # Restrict to requested watchlist
    ff_wl = {t: ff_gated[t] for t in watchlist if t in ff_gated}
    lab_wl = {t: labels[t] for t in watchlist if t in labels}
    sec_wl = {t: ticker_sectors[t] for t in watchlist if t in ticker_sectors}
    fac_wl = {t: factor_frames[t] for t in watchlist if t in factor_frames}

    panel, group_sizes, panel_meta = build_panel_frame(
        ff_wl, lab_wl, sec_wl,
        factor_frames=fac_wl,
        listing_dates=listing_dates,
        min_history_days=min_history,
        lookahead_days=lookahead,
        age_warmup_days=age_warmup,
        nan_prone_cols=nan_cols,
    )

    # Drop rows whose label is NaN (early warmup / end-of-history)
    label_mask = panel["label"].notna()
    panel = panel[label_mask].reset_index(drop=True)
    group_sizes = panel.groupby("date", sort=True).size().values.astype(np.int32)

    # Feature columns = everything that isn't a label/weight/id column
    exclude = {"date", "ticker", "sector", "label",
               "weight", "weight_concurrency", "weight_age"}
    feature_cols = [c for c in panel.columns if c not in exclude]

    # ── 7. Purged K-fold CV ──────────────────────────────────────────────
    def _factory():
        return PanelLTRModel(params=xgb_params)

    # cross_validated_ic expects models to have .fit/.predict sklearn-like;
    # wrap PanelLTRModel to adapt its train() → fit().
    class _SklearnAdapter:
        def __init__(self):
            self._m = PanelLTRModel(params=xgb_params)
        def fit(self, X, y, sample_weight=None):
            # Reconstruct local panel + group_sizes for the train slice.
            # X is panel[feature_cols] already — add date column for grouping.
            df = X.copy()
            # Look up date & weight from the outer panel by positional index.
            df["label"] = y
            df["date"] = panel.loc[X.index, "date"].values
            if sample_weight is not None:
                df["weight"] = sample_weight
            else:
                df["weight"] = 1.0
            df = df.sort_values(["date"], kind="mergesort").reset_index(drop=True)
            gs = df.groupby("date", sort=True).size().values.astype(np.int32)
            self._m.train(
                df, gs, feature_cols=list(X.columns),
                label_col="label", weight_col="weight",
                num_boost_round=max(num_rounds // 2, 50),
            )
        def predict(self, X):
            df = X.copy()
            return self._m.predict(df).values

    cv = PurgedKFold(
        n_splits=cv_splits, embargo_days=embargo, lookahead_days=lookahead,
    )
    cv_out = cross_validated_ic(
        _SklearnAdapter, panel, feature_cols, "label", cv,
        weight_col="weight",
    )

    # ── 8. Final fit on full panel ───────────────────────────────────────
    final_model = PanelLTRModel(params=xgb_params)
    final_out = final_model.train(
        panel, group_sizes, feature_cols=feature_cols,
        label_col="label", weight_col="weight",
        num_boost_round=num_rounds,
    )

    # ── 9. Save artifact ─────────────────────────────────────────────────
    out_path = Path(out_path)
    metadata = {
        "panel_shape": {
            "rows":    int(panel_meta["n_rows"]),
            "tickers": int(panel_meta["n_tickers"]),
            "dates":   int(panel_meta["n_dates"]),
        },
        "oos_mean_ic":     cv_out["mean_ic"],
        "oos_std_ic":      cv_out["std_ic"],
        "oos_per_fold_ic": cv_out["per_fold_ic"],
        "training_train_ic": final_out["train_ic"],
        "training_notes":  config.get("training_notes", "Stage 1 baseline"),
        "neutralize_features": neutralize,
        "lookahead_days":  lookahead,
        "beta_window":     beta_window,
        "min_history_days": min_history,
        "cv_n_splits":     cv_splits,
        "cv_embargo_days": embargo,
    }
    final_model.save(out_path, metadata=metadata)

    return {
        "mean_ic":      cv_out["mean_ic"],
        "per_fold_ic":  cv_out["per_fold_ic"],
        "artifact_path": str(out_path),
        "panel_metadata": panel_meta,
        "feature_cols": feature_cols,
    }
