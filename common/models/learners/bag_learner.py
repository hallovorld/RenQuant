"""Bootstrap Aggregating (Bagging) ensemble wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from .random_tree import RTLearner


class BagLearner:
    """Train an ensemble of base learners on bootstrap samples.

    Default configuration creates a Random Forest of :class:`RTLearner` trees.
    """

    def __init__(
        self,
        learner: type = RTLearner,
        kwargs: dict[str, Any] | None = None,
        bags: int = 20,
    ):
        kwargs = kwargs or {}
        self.learners = [learner(**kwargs) for _ in range(bags)]

    def add_evidence(self, data_x: np.ndarray, data_y: np.ndarray) -> None:
        n = data_x.shape[0]
        for learner in self.learners:
            idx = np.random.choice(n, size=n, replace=True)
            learner.add_evidence(data_x[idx], data_y[idx])

    def query(self, data_x: np.ndarray) -> np.ndarray:
        preds = np.array([learner.query(data_x) for learner in self.learners])
        return np.mean(preds, axis=0)
