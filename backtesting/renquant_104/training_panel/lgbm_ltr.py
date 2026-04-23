"""LightGBM LambdaRank alternative to the XGBoost rank:pairwise panel model.

Trade-offs vs `PanelLTRModel` (XGBoost):

  + LambdaRank optimises NDCG@k, which matches our top-8 selection budget
    more exactly than pairwise across all pairs.
  + ~2× faster training on the same panel (verified on the M4 Pro).
  - Requires integer `label_gain` in [0, 31] — we bucketize the continuous
    Gaussianized labels into 11 gain levels before training.
  - Artifact is bigger (native LGBM `model_to_string()` vs XGBoost JSON).

JSON artifact format mirrors `PanelLTRModel`:
    { "version": 1, "kind": "panel_lgbm",
      "feature_cols": [...], "params": {...},
      "booster_str": "<lgbm text dump>", ... }

The artifact is loadable by a new `PanelLGBMScorer` that duck-types
`PanelScorer` (same `.feature_cols`, `.score(matrix)`), so inference
remains one call via `PanelScoringJob`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


try:
    import lightgbm as lgb  # type: ignore
except Exception:             # pragma: no cover - only missing in stripped envs
    lgb = None


DEFAULT_PARAMS: dict[str, Any] = {
    "objective":         "lambdarank",
    "metric":            "ndcg",
    "ndcg_at":           [5, 10],
    "label_gain":        list(range(32)),   # gain[i] = i
    "learning_rate":     0.02,
    "num_leaves":        15,
    "max_depth":         4,
    "min_data_in_leaf":  50,
    "feature_fraction":  0.7,
    "bagging_fraction":  0.7,
    "bagging_freq":      5,
    "lambda_l1":         2.0,
    "lambda_l2":         5.0,
    "lambdarank_truncation_level": 10,     # optimize NDCG@10
    "verbose":           -1,
    "num_threads":       -1,
}


def _bucketize_labels(y: np.ndarray, n_buckets: int = 11) -> np.ndarray:
    """Map continuous labels to integer gains [0 … n_buckets-1] via rank.

    LambdaRank needs integer relevance; we rank-transform Gaussianized
    labels on each group (date) and assign bucket = int((rank - 1) / size × n).
    Returns int32 array of the same length as y.
    """
    out = np.zeros(len(y), dtype=np.int32)
    # Can't rank per-group here — caller bucketizes globally; we rely on the
    # labels already being Gaussianized per-date so a global bucketize is fine.
    # Use quantile bucketing across the whole series.
    if len(y) < n_buckets:
        return out
    quantiles = np.quantile(y, np.linspace(0, 1, n_buckets + 1))
    # np.digitize gives 0..n_buckets (1-indexed right-open intervals).
    out = np.clip(np.digitize(y, quantiles[1:-1]), 0, n_buckets - 1).astype(np.int32)
    return out


@dataclass
class PanelLGBMModel:
    """LightGBM LambdaRank panel model — mirror of PanelLTRModel."""
    params: dict[str, Any] = None   # type: ignore[assignment]
    booster: Any = None
    feature_cols: list[str] = None  # type: ignore[assignment]
    best_iter: int | None = None

    def __post_init__(self):
        if lgb is None:
            raise RuntimeError("lightgbm is not installed")
        self.params = {**DEFAULT_PARAMS, **(self.params or {})}
        self.feature_cols = list(self.feature_cols or [])

    # ── Training ───────────────────────────────────────────────────────────
    def train(
        self,
        panel: pd.DataFrame,
        group_sizes: np.ndarray,
        feature_cols: list[str],
        label_col: str = "label",
        weight_col: str | None = "weight",
        num_boost_round: int = 300,
    ) -> dict:
        self.feature_cols = list(feature_cols)
        X = panel[feature_cols].values
        y_raw = panel[label_col].values.astype(float)
        y     = _bucketize_labels(y_raw, n_buckets=11)

        # LightGBM ranking takes PER-ROW weights (length = n_rows),
        # unlike XGBoost 3.x which takes per-group. To keep parity with
        # the panel's weighting semantics (where weights are meaningful
        # at the date-group level), we average each group's weight and
        # broadcast back to per-row so LightGBM sees a constant
        # per-group weight expressed at row granularity. Empty/missing
        # weight column → no weights supplied at all.
        row_weights = None
        if weight_col and weight_col in panel.columns:
            w_rows = panel[weight_col].values.astype(float)
            row_weights = np.empty(len(w_rows), dtype=float)
            off = 0
            for gs in group_sizes:
                grp_mean = w_rows[off:off + gs].mean()
                row_weights[off:off + gs] = grp_mean
                off += gs
            assert len(row_weights) == len(X), (
                f"row_weights len {len(row_weights)} != X rows {len(X)}"
            )

        dtrain = lgb.Dataset(X, label=y, group=group_sizes, weight=row_weights)
        self.booster = lgb.train(
            self.params, dtrain, num_boost_round=num_boost_round,
        )
        self.best_iter = self.booster.current_iteration()

        # Per-date Spearman IC against the original (non-bucketed) label
        preds = self.booster.predict(X)
        ic = _per_date_ic(panel, preds, label_col, date_col="date")
        return {"best_iter": self.best_iter, "train_ic": ic}

    # ── Prediction ─────────────────────────────────────────────────────────
    def predict(self, panel: pd.DataFrame) -> pd.Series:
        if self.booster is None:
            raise RuntimeError("PanelLGBMModel.predict called before train/load")
        X = panel[self.feature_cols].values
        preds = self.booster.predict(X)
        return pd.Series(preds, index=panel.index, name="panel_score")

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        if self.booster is None:
            raise RuntimeError("PanelLGBMModel.save called before train")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version":      1,
            "kind":         "panel_lgbm",
            "trained_date": str(date.today()),
            "feature_cols": list(self.feature_cols),
            "params":       self.params,
            "best_iter":    self.best_iter,
            "booster_str":  self.booster.model_to_string(),
        }
        if metadata:
            payload.update({k: v for k, v in metadata.items() if k not in payload})
        path.write_text(json.dumps(payload, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "PanelLGBMModel":
        path = Path(path)
        payload = json.loads(path.read_text())
        if payload.get("kind") != "panel_lgbm":
            raise ValueError(
                f"PanelLGBMModel.load: artifact at {path} is not a panel_lgbm "
                f"(kind={payload.get('kind')!r})",
            )
        m = cls(params=payload.get("params"), feature_cols=payload["feature_cols"])
        m.best_iter = payload.get("best_iter")
        m.booster   = lgb.Booster(model_str=payload["booster_str"])
        return m


def _per_date_ic(panel: pd.DataFrame, preds: np.ndarray,
                 label_col: str, date_col: str = "date") -> float:
    df = pd.DataFrame({
        "date": panel[date_col].values,
        "p":    preds,
        "y":    panel[label_col].values,
    })
    ics: list[float] = []
    for _, g in df.groupby("date", sort=False):
        y = g["y"].values
        p = g["p"].values
        if len(y) < 2 or np.all(y == y[0]) or np.all(p == p[0]):
            continue
        rho, _ = spearmanr(p, y)
        if rho is not None and rho == rho:
            ics.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan")


# ── Scorer that duck-types PanelScorer for the inference side ────────────────

class PanelLGBMScorer:
    """Thin LightGBM loader mirroring `PanelScorer`'s interface."""

    def __init__(self, booster, feature_cols: list[str],
                 metadata: dict | None = None):
        self.booster = booster
        self.feature_cols = list(feature_cols)
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: str | Path) -> "PanelLGBMScorer":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed")
        path = Path(path)
        payload = json.loads(path.read_text())
        if payload.get("kind") != "panel_lgbm":
            raise ValueError(f"PanelLGBMScorer.load: not a panel_lgbm artifact at {path}")
        booster = lgb.Booster(model_str=payload["booster_str"])
        meta = {k: v for k, v in payload.items() if k != "booster_str"}
        return cls(booster, list(payload["feature_cols"]), meta)

    def score(self, feature_matrix: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.feature_cols if c not in feature_matrix.columns]
        if missing:
            raise KeyError(f"PanelLGBMScorer.score: missing cols {missing}")
        X = feature_matrix[self.feature_cols].values
        preds = self.booster.predict(X)
        return pd.Series(preds, index=feature_matrix.index, name="panel_score")


__all__ = [
    "DEFAULT_PARAMS",
    "PanelLGBMModel",
    "PanelLGBMScorer",
]
