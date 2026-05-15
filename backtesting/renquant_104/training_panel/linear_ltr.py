"""Linear panel-LTR model — sklearn LinearRegression wrapper.

Mirrors `training_panel/lgbm_ltr.py` interface but uses a simple OLS
linear model. Produces a JSON artifact compatible with
`kernel.panel_pipeline.panel_scorer.PanelScorer.load()` via the
`kind: panel_linear` dispatch.

Why this exists (Phase 1 of alpha158+Linear integration, 2026-05-06):
The Qlib alpha158 (158 features) + sklearn LinearRegression (MSE) recipe
gave **+0.038 test_median_ic** on RenQuant data — 3.8× the 11-feature
v4 baseline and ~84% of Qlib's csi500 published +0.045 benchmark
(per √breadth scaling). Walk-forward 3-cut showed **+29 pts mean alpha
vs SPY @ 10bp friction** (vs production XGB walk-forward −8.6 pts).

References (per CLAUDE.md §5.12a):
- Qlib `qlib.contrib.model.linear.LinearModel` (gbdt-style API). We
  mirror their MSE-on-CSZScoreNorm-label recipe: train target is
  per-day cross-sectionally z-scored forward returns; OLS handles the
  residual.
- Bryan Kelly + Gu + Xiu RFS 2020: "linear models capture most of the
  predictability; nonlinearity adds modest gain" — empirically observed
  in our cross-section.

The wire-up to inference path (`BuildFeatureMatrixTask` → `PanelScoringJob`
→ this class via `PanelScorer.load`) is identical to XGB. No changes to
calibrator, JointPortfolioQPTask, sell gates, etc.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("training_panel.linear_ltr")


class PanelLinearScorer:
    """Linear panel-LTR scorer compatible with `PanelScorer.load()`.

    Artifact format (JSON, `kind: panel_linear`):

      {
        "version": 1,
        "kind": "panel_linear",
        "feature_cols": [...],            # list of feature column names
        "coef": [...],                    # per-feature coefficients (n_features,)
        "intercept": 0.0,                 # scalar bias
        "trained_date": "2026-05-06",
        "panel_shape": {"rows": ..., "tickers": ..., "dates": ...},
        "oos_mean_ic": 0.0316,            # diagnostic
        "training_train_ic": 0.0527,
        "metadata": {...}                 # extras: training_window, label, etc.
      }
    """

    def __init__(self, coef: np.ndarray, intercept: float,
                 feature_cols: list[str], metadata: dict | None = None,
                 feature_means: np.ndarray | None = None,
                 feature_stds: np.ndarray | None = None,
                 clip_sigma: float = 5.0):
        self.coef = np.asarray(coef, dtype=float).reshape(-1)
        self.intercept = float(intercept)
        self.feature_cols = list(feature_cols)
        self.metadata = metadata or {}
        if self.coef.shape != (len(self.feature_cols),):
            raise ValueError(
                f"PanelLinearScorer: coef shape {self.coef.shape} != "
                f"({len(self.feature_cols)},)"
            )
        # Optional ZScoreNorm stats for self-contained inference. If
        # set, score_raw(raw_features) applies (x - mean) / std before
        # the linear predict. Built per Qlib `_DEFAULT_INFER_PROCESSORS`
        # spec: ProcessInf + ZScoreNorm + Fillna.
        if feature_means is not None:
            feature_means = np.asarray(feature_means, dtype=float).reshape(-1)
            if feature_means.shape != (len(self.feature_cols),):
                raise ValueError(
                    f"feature_means shape {feature_means.shape} != "
                    f"({len(self.feature_cols)},)"
                )
        if feature_stds is not None:
            feature_stds = np.asarray(feature_stds, dtype=float).reshape(-1)
            if feature_stds.shape != (len(self.feature_cols),):
                raise ValueError(
                    f"feature_stds shape {feature_stds.shape} != "
                    f"({len(self.feature_cols)},)"
                )
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.clip_sigma = float(clip_sigma)

    @classmethod
    def from_sklearn(cls, model, feature_cols: list[str],
                     metadata: dict | None = None,
                     feature_means: np.ndarray | None = None,
                     feature_stds: np.ndarray | None = None):
        """Wrap a fitted sklearn LinearRegression / Ridge / Lasso."""
        intercept = (float(model.intercept_)
                     if hasattr(model, "intercept_")
                     and not isinstance(model.intercept_, np.ndarray)
                     else 0.0)
        return cls(
            coef=model.coef_,
            intercept=intercept,
            feature_cols=feature_cols,
            metadata=metadata or {},
            feature_means=feature_means,
            feature_stds=feature_stds,
        )

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"PanelLinearScorer.load: artifact not found: {path}"
            )
        payload = json.loads(path.read_text())
        if payload.get("kind") != "panel_linear":
            raise ValueError(
                f"PanelLinearScorer.load: artifact kind={payload.get('kind')!r} "
                f"≠ 'panel_linear'"
            )
        # Keep `kind` IN metadata — downstream (ApplyScoresTask, DriftGuardTask)
        # dispatches on `scorer.metadata["kind"]`. Removing it here was a
        # bug: panel_linear dispatch silently fell through to the legacy
        # 21-feature XGB code path → DriftGuard saw 158 alpha158 cols all
        # NaN → FAIL-SAFE cleared all candidates → 0 trades over 128 days.
        meta = {k: v for k, v in payload.items()
                if k not in ("coef", "intercept", "feature_cols",
                              "version", "feature_means", "feature_stds",
                              "clip_sigma")}
        means = payload.get("feature_means")
        stds  = payload.get("feature_stds")
        return cls(
            coef=np.asarray(payload["coef"], dtype=float),
            intercept=float(payload.get("intercept", 0.0)),
            feature_cols=list(payload["feature_cols"]),
            metadata=meta,
            feature_means=np.asarray(means, dtype=float) if means is not None else None,
            feature_stds=np.asarray(stds, dtype=float) if stds is not None else None,
            clip_sigma=float(payload.get("clip_sigma", 5.0)),
        )

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged_meta = dict(self.metadata)
        if metadata:
            merged_meta.update(metadata)
        payload = {
            "version": 1,
            "kind": "panel_linear",
            "feature_cols": self.feature_cols,
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "clip_sigma": self.clip_sigma,
            **merged_meta,
        }
        if self.feature_means is not None:
            payload["feature_means"] = self.feature_means.tolist()
        if self.feature_stds is not None:
            payload["feature_stds"] = self.feature_stds.tolist()
        path.write_text(json.dumps(payload, indent=2, default=str))

    def score(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """Predict per-ticker scores: y = X @ coef + intercept.

        Expects features ALREADY normalized (consistent with PanelScorer
        protocol — caller has run ZScoreNorm). Returns Series indexed
        like `feature_matrix.index`. Raises KeyError if any feature
        column is missing.
        """
        missing = [c for c in self.feature_cols if c not in feature_matrix.columns]
        if missing:
            raise KeyError(
                f"PanelLinearScorer.score: missing columns: {missing[:5]}"
                + (f" + {len(missing) - 5} more" if len(missing) > 5 else "")
            )
        X = feature_matrix[self.feature_cols].values.astype(float)
        # Substitute NaN/inf with 0 (matches Qlib's Fillna processor)
        X = np.where(np.isfinite(X), X, 0.0)
        preds = X @ self.coef + self.intercept
        return pd.Series(preds, index=feature_matrix.index, name="panel_score")

    def score_raw(self, raw_features: pd.DataFrame) -> pd.Series:
        """Predict from RAW (un-normalized) features — applies stored
        ZScoreNorm + Fillna(0) + Clip ±clip_sigma internally.

        Mirrors Qlib `_DEFAULT_INFER_PROCESSORS` pipeline:
          ProcessInf → ZScoreNorm → Fillna(0).

        Use this when the caller has raw feature output from
        `compute_alpha158_at()` and the scorer was saved with
        feature_means + feature_stds (the production path).

        Raises ValueError if normalization stats not stored in the artifact.
        """
        if self.feature_means is None or self.feature_stds is None:
            raise ValueError(
                "PanelLinearScorer.score_raw: feature_means/feature_stds "
                "not stored in artifact. Either re-train + save with "
                "stats, or use score() with pre-normalized features."
            )
        missing = [c for c in self.feature_cols if c not in raw_features.columns]
        if missing:
            raise KeyError(
                f"PanelLinearScorer.score_raw: missing columns: {missing[:5]}"
                + (f" + {len(missing) - 5} more" if len(missing) > 5 else "")
            )
        X = raw_features[self.feature_cols].values.astype(float)
        # ProcessInf — replace ±inf with NaN
        X = np.where(np.isfinite(X), X, np.nan)
        # ZScoreNorm using stored stats
        std_safe = np.where(self.feature_stds > 1e-9, self.feature_stds, 1.0)
        X = (X - self.feature_means) / std_safe
        # Fillna(0)
        X = np.where(np.isfinite(X), X, 0.0)
        # Clip ±clip_sigma (matches build script post-norm clip)
        if self.clip_sigma > 0:
            X = np.clip(X, -self.clip_sigma, self.clip_sigma)
        preds = X @ self.coef + self.intercept
        return pd.Series(preds, index=raw_features.index, name="panel_score")
