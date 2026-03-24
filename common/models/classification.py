"""Classification-based trading model (Random Forest).

Labels each day by its N-day forward return as +1 (long), -1 (short),
or 0 (hold), then trains a bagged random-tree ensemble to predict the label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseModel
from .learners import BagLearner, RTLearner


class ClassificationModel(BaseModel):
    """Bagged random-tree classifier for trade signal generation.

    Parameters:
        feature_columns: Indicator columns used as features.
        lookahead: Days ahead for labeling forward returns.
        threshold: Minimum return (absolute) to label as long/short.
        leaf_size: Minimum leaf size for each tree.
        bags: Number of trees in the ensemble.
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        lookahead: int = 10,
        threshold: float = 0.04,
        leaf_size: int = 25,
        bags: int = 15,
        impact: float = 0.0,
    ):
        self.feature_columns = feature_columns or ["rsi", "macd_hist", "cci"]
        self.lookahead = lookahead
        self.threshold = threshold
        self.leaf_size = leaf_size
        self.bags = bags
        self.impact = impact
        self.learner: BagLearner | None = None

    @property
    def model_type(self) -> str:
        return "classification"

    # ── training ───────────────────────────────────────────────────────

    def _build_labels(self, close: pd.Series) -> pd.Series:
        future_ret = close.shift(-self.lookahead) / close - 1.0
        buy_thresh = self.threshold + self.impact
        sell_thresh = -(self.threshold + self.impact)
        return pd.Series(
            np.where(future_ret > buy_thresh, 1,
                     np.where(future_ret < sell_thresh, -1, 0)),
            index=close.index,
        )

    def train(self, df: pd.DataFrame, **kwargs) -> dict:
        # Shift features 1 day to avoid lookahead bias
        features = df[self.feature_columns].shift(1)
        labels = self._build_labels(df["close"])

        train_df = features.copy()
        train_df["Y"] = labels
        train_df = train_df.dropna()

        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": self.leaf_size},
            bags=self.bags,
        )
        self.learner.add_evidence(
            train_df[self.feature_columns].values,
            train_df["Y"].values,
        )

        return {
            "model_type": self.model_type,
            "train_rows": len(train_df),
            "label_distribution": {
                "long": int((train_df["Y"] == 1).sum()),
                "hold": int((train_df["Y"] == 0).sum()),
                "short": int((train_df["Y"] == -1).sum()),
            },
        }

    # ── prediction ─────────────────────────────────────────────────────

    def predict(self, state: pd.Series | pd.DataFrame) -> str:
        if self.learner is None:
            raise RuntimeError("Model not trained. Call train() first.")
        if isinstance(state, pd.Series):
            features = state[self.feature_columns].values.reshape(1, -1)
        else:
            features = state[self.feature_columns].values
            if features.ndim == 1:
                features = features.reshape(1, -1)

        pred = self.learner.query(features)[0]
        if pred > 0.5:
            return "buy"
        if pred < -0.5:
            return "sell"
        return "hold"

    # ── persistence ────────────────────────────────────────────────────

    def save(self, directory: Path, model_name: str) -> dict:
        directory = Path(directory)
        if self.learner is None:
            raise RuntimeError("Cannot save untrained model.")

        # Save each tree in the ensemble as JSON
        trees_data = []
        for i, learner in enumerate(self.learner.learners):
            trees_data.append(learner.tree.tolist())

        artifact_path = directory / f"{model_name}-rf-trees.json"
        artifact_path.write_text(json.dumps(trees_data, indent=2))

        metadata = {
            "model_name": model_name,
            "policy_type": self.model_type,
            "feature_columns": self.feature_columns,
            "lookahead": self.lookahead,
            "threshold": self.threshold,
            "leaf_size": self.leaf_size,
            "bags": self.bags,
            "impact": self.impact,
            "artifacts": {"trees": str(artifact_path)},
        }
        meta_path = directory / f"{model_name}-policy-metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata

    def load(self, directory: Path, model_name: str) -> None:
        directory = Path(directory)
        meta_path = directory / f"{model_name}-policy-metadata.json"
        metadata = json.loads(meta_path.read_text())

        self.feature_columns = metadata["feature_columns"]
        self.lookahead = metadata["lookahead"]
        self.threshold = metadata["threshold"]
        self.leaf_size = metadata["leaf_size"]
        self.bags = metadata["bags"]
        self.impact = metadata.get("impact", 0.0)

        trees_path = directory / f"{model_name}-rf-trees.json"
        trees_data = json.loads(trees_path.read_text())

        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": self.leaf_size},
            bags=len(trees_data),
        )
        for i, tree_list in enumerate(trees_data):
            self.learner.learners[i].tree = np.array(tree_list)
