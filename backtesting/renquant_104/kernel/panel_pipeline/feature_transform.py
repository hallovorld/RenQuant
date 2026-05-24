"""Feature-space transforms for panel-LTR artifacts.

The historical alpha158 panel stores technical alpha columns already
train-normalized, while fundamental columns are appended in raw units. Live
inference builds raw technical + raw fundamental rows. The artifact therefore
needs two explicit source-space contracts:

* ``raw``: apply every stored mean/std before scoring live/sim rows.
* ``panel``: apply only columns whose kind says the prebuilt panel is raw
  (currently ``robust_z`` fundamentals); leave already-normalized alpha columns
  alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _stats_from_metadata(metadata: dict, n: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    means = metadata.get("feature_means")
    stds = metadata.get("feature_stds")
    kinds = metadata.get("feature_norm_kind") or metadata.get("feature_norm_kinds")
    if means is None or stds is None or len(means) != n or len(stds) != n:
        return np.zeros(n, dtype=float), np.ones(n, dtype=float), ["identity"] * n
    out_kinds = list(kinds) if isinstance(kinds, list) and len(kinds) == n else ["legacy_full_z"] * n
    sd = np.asarray(stds, dtype=float)
    sd = np.where(np.isfinite(sd) & (np.abs(sd) > 1e-12), sd, 1.0)
    return np.asarray(means, dtype=float), sd, out_kinds


def transform_feature_frame(
    frame: pd.DataFrame,
    feature_cols: list[str],
    metadata: dict,
    *,
    source_space: str,
    clip: float = 5.0,
) -> pd.DataFrame:
    """Align and transform features into the scorer's training space."""
    X = frame.reindex(columns=feature_cols, fill_value=float("nan")).fillna(0.0)
    n = len(feature_cols)
    if n == 0:
        return X.astype(float)
    means, stds, kinds = _stats_from_metadata(metadata or {}, n)
    values = X.values.astype(float)

    if source_space == "raw":
        mask = np.ones(n, dtype=bool)
    elif source_space == "panel":
        mask = np.asarray([k in {"robust_z", "panel_raw_z"} for k in kinds], dtype=bool)
    else:
        raise ValueError(f"unknown feature source_space: {source_space}")

    if mask.any():
        values[:, mask] = (values[:, mask] - means[mask]) / stds[mask]
    if clip is not None and clip > 0:
        values = np.clip(values, -float(clip), float(clip))
    return pd.DataFrame(values, index=X.index, columns=feature_cols)
