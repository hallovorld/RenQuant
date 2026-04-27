"""Ensemble panel scorer — average per-bar ranks of two backend scorers.

Per `doc/components/transformer-104.md §5` ship-gate: when the
transformer's OOS IC is within [1.10, 1.30)× XGBoost's, ensemble (rather
than replace). Rank averaging is scale-invariant, so a transformer model
whose raw scores sit on a different scale than XGBoost's composes
cleanly without calibration.

Loader contract: this class is NOT returned by `PanelScorer.load(single_path)`
— it's built explicitly by :func:`build_ensemble_scorer` from two
individual scorer artifacts. A dedicated Task (`LoadEnsembleScorerTask`,
added later) wires it into `PanelScoringJob` when
`ranking.panel_scoring.ensemble.enabled == true`.

Interface matches :class:`PanelScorer` + :class:`TransformerPanelScorer`:
``feature_cols`` (intersection of the two) + ``score(matrix) -> pd.Series``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class EnsemblePanelScorer:
    """Average the per-bar RANK of multiple scorers' outputs.

    For each call to :meth:`score`, each inner scorer is asked for its
    per-row score; we then rank each scorer's output independently
    (descending, average rank for ties) and average those ranks with
    per-scorer weights. The returned Series has the same index as
    ``feature_matrix``; values are mean ranks normalized to the unit
    interval so downstream tier thresholds remain meaningful.
    """

    def __init__(self, scorers: list, weights: list[float] | None = None,
                 metadata: dict | None = None):
        if not scorers:
            raise ValueError("EnsemblePanelScorer: need at least one inner scorer")
        self._scorers = list(scorers)
        if weights is None:
            weights = [1.0 / len(scorers)] * len(scorers)
        if len(weights) != len(scorers):
            raise ValueError(
                f"EnsemblePanelScorer: len(weights)={len(weights)} != "
                f"n_scorers={len(scorers)}"
            )
        wsum = float(sum(weights))
        if wsum <= 0:
            raise ValueError("EnsemblePanelScorer: weights sum must be > 0")
        self._weights = [w / wsum for w in weights]
        self.metadata = metadata or {}
        # Union across inner scorers — the caller supplies a matrix; each
        # scorer picks out the columns it needs. feature_cols exposed here
        # is the union so missing-column detection catches both backends.
        cols: list[str] = []
        seen: set[str] = set()
        for s in scorers:
            for c in getattr(s, "feature_cols", []):
                if c not in seen:
                    cols.append(c)
                    seen.add(c)
        self.feature_cols = cols

    def score(self, feature_matrix: pd.DataFrame) -> pd.Series:
        n = len(feature_matrix)
        if n == 0:
            return pd.Series([], name="panel_score", dtype=float)

        ranks = np.zeros(n, dtype=np.float64)
        for s, w in zip(self._scorers, self._weights):
            sub = feature_matrix[list(s.feature_cols)]
            raw = s.score(sub)
            # Normalized rank in [0, 1]: 1.0 for the top score, 0.0 for the
            # bottom. Ties → average rank. Single-row edge case gets 0.5.
            arr = raw.to_numpy(dtype=np.float64)
            if n == 1:
                ranks += w * 0.5
                continue
            order = np.argsort(-arr, kind="mergesort")
            # Rank by position in the sorted order (0=best), then normalize.
            rank_pos = np.empty(n, dtype=np.float64)
            rank_pos[order] = np.arange(n, dtype=np.float64)
            # Handle ties: average ranks of equal values. Using a simple
            # groupby over sorted values is enough for our small n.
            sorted_vals = arr[order]
            i = 0
            while i < n:
                j = i + 1
                while j < n and sorted_vals[j] == sorted_vals[i]:
                    j += 1
                if j - i > 1:
                    avg_rank = (i + j - 1) / 2.0
                    rank_pos[order[i:j]] = avg_rank
                i = j
            # Invert so higher-score = higher normalized rank.
            rank_norm = 1.0 - rank_pos / max(n - 1, 1)
            ranks += w * rank_norm
        return pd.Series(ranks, index=feature_matrix.index, name="panel_score")


def build_ensemble_scorer(
    artifact_paths: Iterable[str | Path],
    weights: list[float] | None = None,
    metadata: dict | None = None,
) -> EnsemblePanelScorer:
    """Build an EnsemblePanelScorer from a list of artifact paths.

    Each path is dispatched through :meth:`PanelScorer.load` so any
    supported backend (xgboost / lightgbm / transformer) can be combined.
    """
    from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
    scorers = [PanelScorer.load(Path(p)) for p in artifact_paths]
    return EnsemblePanelScorer(scorers=scorers, weights=weights, metadata=metadata)


__all__ = ["EnsemblePanelScorer", "build_ensemble_scorer"]
