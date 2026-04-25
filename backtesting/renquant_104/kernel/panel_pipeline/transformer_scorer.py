"""Inference-side scorer for the transformer panel backend.

Mirrors :class:`PanelScorer` duck-typing so callers can dispatch on the
artifact kind. Loads a ``.pt`` (state_dict) + ``.json`` sidecar pair
written by :class:`PanelTransformerModel`.

Exposes::

    scorer = TransformerPanelScorer.load(path)
    scores: pd.Series = scorer.score(feature_matrix)

where ``feature_matrix`` is a DataFrame indexed by ticker with one column
per feature name in ``feature_cols``. Returned scores preserve the input
index order.

The scorer always treats the input as **a single date-group** — per-bar
inference in LEAN / live / sim gives us exactly one date's worth of
tickers, which is the shape the transformer wants.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class TransformerPanelScorer:
    """Single-date inference wrapper around a saved transformer artifact.

    Interface-compatible with :class:`PanelScorer`: ``feature_cols`` attr
    + ``score(matrix) -> pd.Series``.
    """

    def __init__(self, model, feature_cols: list[str], metadata: dict | None = None):
        self._model = model
        self.feature_cols = list(feature_cols)
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: str | Path) -> "TransformerPanelScorer":
        # Delay import so torch stays an optional dependency of the package.
        from training_panel.transformer_model import PanelTransformerModel  # noqa: PLC0415
        m = PanelTransformerModel.load(path)
        pt_path = Path(path)
        if pt_path.suffix == ".json":
            pt_path = pt_path.with_suffix(".pt")
        json_path = pt_path.with_suffix(".json")
        meta = {}
        if json_path.exists():
            import json  # noqa: PLC0415
            meta = json.loads(json_path.read_text())
        return cls(model=m, feature_cols=m.feature_cols, metadata=meta)

    def score(self, feature_matrix: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.feature_cols if c not in feature_matrix.columns]
        if missing:
            raise KeyError(
                f"TransformerPanelScorer.score: feature matrix missing columns: {missing}",
            )
        # Round-3 audit (#R3-22): empty matrix is a valid no-op (no
        # candidates this bar). Return an empty Series early to avoid the
        # zero-row-but-one-column degeneracy of fabricating `date=0` below.
        if feature_matrix.empty:
            return pd.Series([], dtype=float, name="panel_score",
                              index=feature_matrix.index)
        # Single-date-group inference: fabricate a `date` column so the
        # model's predict() groups all rows together.
        frame = feature_matrix[self.feature_cols].copy()
        frame["date"] = 0
        preds = self._model.predict(frame)
        # Model's predict returns a Series indexed like `frame` — re-align
        # to the caller's original index.
        preds.index = feature_matrix.index
        preds.name = "panel_score"
        return preds


__all__ = ["TransformerPanelScorer"]
