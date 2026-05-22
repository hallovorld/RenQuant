"""Cross-sectional feature-matrix construction for panel-LTR inference.

Builds the **same** per-row feature layout that `training_panel/pipeline.py`
used at training time, but only for the single target date (today). The
output is a DataFrame indexed by ticker, whose columns exactly match the
artifact's `feature_cols`.

Training path (simplified):
  per-ticker feature_frame  (rsi, macd_hist, ... trend, trend_long ...)
  per-ticker factor_frame   (size_z, mom_12_1_z, beta_60d_z, resid_mom_z)
  + missingness indicators  ({col}_is_missing)

Inference path (this module):
  Same ingredients, one row per ticker, picked at the target date.

Public API::

    build_inference_matrix(feature_frames, factor_frames, today,
                           feature_cols, nan_prone_cols=None) -> DataFrame
    run_panel_inference(feature_frames, factor_frames, today,
                        artifact_path) -> pd.Series
"""
from __future__ import annotations

from pathlib import Path
import datetime as _dt

import numpy as np
import pandas as pd

from .panel_scorer import PanelScorer


def _pick_today_row(df: pd.DataFrame, today: pd.Timestamp) -> pd.Series | None:
    """Return the row dated `today`, or the most recent row on-or-before it.

    Returns None if the frame has no row on-or-before `today`.

    Audit fix FM-NEW-1 (2026-04-26 round-3): pre-fix, code used
    `np.where(mask)[0].max()` which picks the LAST POSITION matching
    the date filter. That assumed df is sorted ascending by date —
    if not (e.g. shuffled or descending), it picks the wrong row.
    Now: argmax over the masked DATES gives the LATEST DATE
    regardless of frame ordering.
    """
    if df is None or df.empty:
        return None
    idx = pd.to_datetime(df.index)
    mask = idx <= today
    if not mask.any():
        return None
    # Use argmax on the masked dates to find the position with the
    # latest date. mask is bool numpy; idx[mask] is the subset of dates
    # ≤ today. The position of the max is what we want.
    masked_idx = idx[mask]
    masked_positions = np.where(mask)[0]
    latest_pos_in_subset = masked_idx.argmax()
    return df.iloc[masked_positions[latest_pos_in_subset]]


def build_inference_matrix(
    feature_frames: dict[str, pd.DataFrame],
    factor_frames: dict[str, pd.DataFrame] | None,
    today: str | _dt.date | pd.Timestamp,
    feature_cols: list[str],
    nan_prone_cols: list[str] | None = None,
    macro_frame: "pd.DataFrame | None" = None,
    asset_embeddings: "dict[str, np.ndarray] | None" = None,
) -> pd.DataFrame:
    """One row per ticker × len(feature_cols) columns, aligned to the artifact.

    For each ticker:
      1. Pick today's row from `feature_frames[ticker]` (or most recent ≤ today)
      2. Append today's row from `factor_frames[ticker]` if provided
      3. **Bug #25 fix**: append today's macro features (broadcast — same
         value for every ticker on this date) if `macro_frame` provided
      4. Append `{col}_is_missing` indicator for each `col` in `nan_prone_cols`
      5. Select and order columns per `feature_cols`
      6. **T2-2 fix**: join per-ticker asset embedding vectors (emb_0..emb_{D-1})
         from `asset_embeddings` dict. Mirrors `build_panel_frame`'s embedding
         broadcast. Tickers absent from the artifact get 0.0 (neutral).

    The ``asset_embeddings`` parameter is the interface point for T2-2
    (per-ticker asset embedding broadcast at inference). The full
    broadcast logic lives on the ``exp/macro-v3-isolation`` experimental
    branch — the production T2-2 was rejected (NO-GO 2026-04-27, CPCV
    OOS IC −18.5%) and ``asset_embeddings`` is None in the production
    config. This signature accepts the kwarg so the call site in
    ``job_panel_scoring.py`` is type-correct on main; ignoring the value
    is the production behavior.

    Tickers with no row on-or-before `today` are skipped. Missing columns
    are filled with NaN so the resulting matrix has the exact shape the
    model expects.
    """
    today_ts = pd.Timestamp(today)
    rows: dict[str, dict] = {}

    # Bug #25 fix: pre-pick the macro values for today (or most-recent ≤ today),
    # then broadcast to every ticker. Same date → same macro values.
    macro_values: dict | None = None
    if macro_frame is not None and not macro_frame.empty:
        macro_row = _pick_today_row(macro_frame, today_ts)
        if macro_row is not None:
            macro_values = dict(macro_row)

    for ticker, ff in feature_frames.items():
        ff_row = _pick_today_row(ff, today_ts)
        if ff_row is None:
            continue
        row: dict = dict(ff_row)

        if factor_frames is not None and ticker in factor_frames:
            fac_row = _pick_today_row(factor_frames[ticker], today_ts)
            if fac_row is not None:
                for k, v in fac_row.items():
                    row[k] = v

        if macro_values is not None:
            for k, v in macro_values.items():
                # Don't let macro overwrite an existing column from per-ticker
                # data (matches build_panel_frame's collision rule, suffix '_macro').
                if k in row:
                    row[f"{k}_macro"] = v
                else:
                    row[k] = v

        if nan_prone_cols:
            for col in nan_prone_cols:
                ind = f"{col}_is_missing"
                if col in row:
                    row[ind] = int(pd.isna(row[col]))
                else:
                    row[ind] = 1

        rows[ticker] = row

    if not rows:
        return pd.DataFrame(columns=feature_cols)

    out = pd.DataFrame.from_dict(rows, orient="index")

    # T2-2 fix: join per-ticker asset embeddings (emb_0..emb_{D-1}).
    # Training path (build_panel_frame) broadcasts each ticker's embedding
    # vector onto every row of that ticker. Here we do the same: index is
    # already ticker, so a left-join suffices. Tickers absent from the
    # embedding artifact get 0.0 (same convention as build_panel_frame).
    if asset_embeddings is not None and len(asset_embeddings) > 0:
        first_emb = next(iter(asset_embeddings.values()))
        if first_emb is not None and len(first_emb) > 0:
            emb_dim = len(first_emb)
            emb_cols = [f"emb_{i}" for i in range(emb_dim)]
            emb_df = pd.DataFrame.from_dict(
                {t: list(v) for t, v in asset_embeddings.items()
                 if v is not None and len(v) == emb_dim},
                orient="index",
                columns=emb_cols,
            )
            out = out.join(emb_df, how="left")
            out[emb_cols] = out[emb_cols].fillna(0.0)

    # Guarantee exact artifact order and fill absent columns in one shot.
    # Repeated column insertion fragments pandas frames and swamps daily-run logs.
    return out.reindex(columns=feature_cols)


def run_panel_inference(
    feature_frames: dict[str, pd.DataFrame],
    factor_frames: dict[str, pd.DataFrame] | None,
    today: str | _dt.date | pd.Timestamp,
    artifact_path: str | Path,
    nan_prone_cols: list[str] | None = None,
) -> pd.Series:
    """Load the artifact, build today's matrix, return per-ticker scores.

    Returns a Series indexed by ticker. Tickers missing from the inputs
    are absent from the result.
    """
    scorer = PanelScorer.load(artifact_path)
    X = build_inference_matrix(
        feature_frames, factor_frames, today,
        feature_cols=scorer.feature_cols,
        nan_prone_cols=nan_prone_cols,
    )
    if X.empty:
        return pd.Series(dtype=float, name="panel_score")
    return scorer.score(X)
