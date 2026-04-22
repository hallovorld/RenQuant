"""Cross-sectional panel scorer — loads the Stage-1 artifact and predicts.

The training side (`training_panel/`) writes a JSON artifact with:

    { version, feature_cols, params, booster_raw_json, oos_mean_ic, ... }

`PanelScorer.load(path)` rebuilds an XGBoost booster from the embedded
JSON and exposes a single entry point::

    scores: dict[ticker, float] = scorer.score(feature_matrix)

`feature_matrix` is a DataFrame indexed by ticker with one column per
feature name in `feature_cols`. The returned scores preserve the input
index order.

Two gate helpers are provided for selection use:

    top_n_by_score(scores, n)      — largest-N by score
    probability_gate(scores, thr)  — keep score ≥ thr
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb


class PanelScorer:
    """Thin loader around a saved panel-LTR artifact."""

    def __init__(self, booster: xgb.Booster, feature_cols: list[str],
                 metadata: dict | None = None):
        self.booster = booster
        self.feature_cols = list(feature_cols)
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: str | Path) -> "PanelScorer":
        path = Path(path)
        payload = json.loads(path.read_text())
        booster = xgb.Booster()
        booster.load_model(bytearray(payload["booster_raw_json"].encode("utf-8")))
        meta = {k: v for k, v in payload.items() if k != "booster_raw_json"}
        return cls(
            booster=booster,
            feature_cols=list(payload["feature_cols"]),
            metadata=meta,
        )

    def score(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """Predict panel scores for rows of `feature_matrix`.

        Returns a Series indexed like `feature_matrix.index` (typically
        ticker symbols). Missing feature columns raise KeyError — the
        caller is responsible for aligning the matrix to the artifact's
        `feature_cols`.
        """
        missing = [c for c in self.feature_cols if c not in feature_matrix.columns]
        if missing:
            raise KeyError(
                f"PanelScorer.score: feature matrix missing columns: {missing}",
            )
        X = feature_matrix[self.feature_cols].values
        d = xgb.DMatrix(X)
        preds = self.booster.predict(d)
        return pd.Series(preds, index=feature_matrix.index, name="panel_score")


def compute_panel_scores(
    artifact_path: str | Path,
    feature_matrix: pd.DataFrame,
) -> pd.Series:
    """One-shot helper: load artifact → score → return per-ticker scores."""
    scorer = PanelScorer.load(artifact_path)
    return scorer.score(feature_matrix)


def top_n_by_score(scores: pd.Series, n: int) -> list[str]:
    """Return the top-`n` labels (indices) of `scores` by value, descending.

    NaN scores are excluded. Ties broken by input order (stable sort).
    """
    if n <= 0:
        return []
    s = scores.dropna()
    order = s.sort_values(ascending=False, kind="mergesort")
    return list(order.index[:n])


def probability_gate(scores: pd.Series, threshold: float) -> list[str]:
    """Return labels whose score is >= `threshold`, sorted high → low.

    NaN scores are excluded.
    """
    s = scores.dropna()
    passed = s[s >= threshold]
    return list(passed.sort_values(ascending=False, kind="mergesort").index)
