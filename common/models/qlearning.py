"""Tabular Q-Learning trading model.

Discretizes continuous indicator values into bins, encodes
(bin_indicators..., holding_bucket) as a single state integer,
and learns a Q-table over training epochs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseModel
from .learners import TabularQLearner

# action_id -> target holding multiplier (1000 = full long, etc.)
ACTIONS = {0: 1000, 1: -1000, 2: 0}
ACTION_NAMES = {0: "buy", 1: "sell", 2: "hold"}


class QLearningModel(BaseModel):
    """Q-Learning with discretized indicator states.

    Parameters:
        feature_columns: Indicator columns to discretize.
        n_bins: Number of bins per indicator.
        n_epochs: Training epochs through the data.
        alpha: Learning rate.
        gamma: Discount factor.
        rar: Initial random-action rate.
        radr: Random-action decay rate.
        dyna: Dyna replay updates per step (0 = off).
        impact: Market impact for reward penalty.
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        n_bins: int = 10,
        n_epochs: int = 100,
        alpha: float = 0.2,
        gamma: float = 0.9,
        rar: float = 0.98,
        radr: float = 0.999,
        dyna: int = 0,
        impact: float = 0.0,
        commission: float = 0.0,
    ):
        self.feature_columns = feature_columns or ["rsi", "macd_hist", "cci"]
        self.n_bins = n_bins
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.impact = impact
        self.commission = commission

        self.bin_edges: dict[str, np.ndarray] | None = None
        self.qlearner: TabularQLearner | None = None

    @property
    def model_type(self) -> str:
        return "qlearning"

    # ── state encoding ─────────────────────────────────────────────────

    def _num_states(self) -> int:
        # n_bins^(n_features) * 3 holding buckets
        return (self.n_bins ** len(self.feature_columns)) * 3

    def _discretize(self, features: pd.DataFrame) -> np.ndarray:
        result = np.zeros((len(features), len(self.feature_columns)), dtype=int)
        for i, col in enumerate(self.feature_columns):
            bins = np.digitize(features[col].values, self.bin_edges[col]) - 1
            result[:, i] = np.clip(bins, 0, self.n_bins - 1)
        return result

    def _encode_state(self, bin_row: np.ndarray, holding: int) -> int:
        h = 2 if holding > 0 else (0 if holding < 0 else 1)
        state = 0
        for b in bin_row:
            state = state * self.n_bins + int(b)
        return state * 3 + h

    # ── training ───────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, **kwargs) -> dict:
        # Shift features 1 day for lookahead prevention
        features = df[self.feature_columns].shift(1)
        close = df["close"]
        daily_rets = close.pct_change().fillna(0.0)

        valid = features.dropna()
        valid_rets = daily_rets.loc[valid.index]

        # Compute bin edges from training data
        self.bin_edges = {}
        for col in self.feature_columns:
            self.bin_edges[col] = np.linspace(
                valid[col].quantile(0.01),
                valid[col].quantile(0.99),
                self.n_bins + 1,
            )[1:-1]

        disc = self._discretize(valid)

        self.qlearner = TabularQLearner(
            num_states=self._num_states(),
            num_actions=3,
            alpha=self.alpha,
            gamma=self.gamma,
            rar=self.rar,
            radr=self.radr,
            dyna=self.dyna,
        )

        for _ in range(self.n_epochs):
            holding = 0
            state = self._encode_state(disc[0], holding)
            action = self.qlearner.querysetstate(state)

            for i in range(1, len(valid)):
                target = ACTIONS[action]
                trade = target - holding
                reward = holding * valid_rets.iloc[i]
                if trade != 0:
                    reward -= abs(trade) * self.impact
                    reward -= self.commission
                holding = target
                state = self._encode_state(disc[i], holding)
                action = self.qlearner.query(state, reward)

        return {
            "model_type": self.model_type,
            "n_epochs": self.n_epochs,
            "n_states": self._num_states(),
            "train_rows": len(valid),
        }

    # ── prediction ─────────────────────────────────────────────────────

    def predict(self, state: pd.Series | pd.DataFrame) -> str:
        if self.qlearner is None or self.bin_edges is None:
            raise RuntimeError("Model not trained.")
        if isinstance(state, pd.DataFrame):
            state = state.iloc[0]

        bins = np.array([
            np.clip(np.digitize(state[col], self.bin_edges[col]) - 1, 0, self.n_bins - 1)
            for col in self.feature_columns
        ])
        holding = int(state.get("position_flag", 0)) * 1000
        s = self._encode_state(bins, holding)
        action = self.qlearner.querysetstate(s)
        return ACTION_NAMES[action]

    # ── persistence ────────────────────────────────────────────────────

    def save(self, directory: Path, model_name: str) -> dict:
        directory = Path(directory)
        if self.qlearner is None or self.bin_edges is None:
            raise RuntimeError("Cannot save untrained model.")

        q_path = directory / f"{model_name}-qtable.json"
        q_path.write_text(json.dumps(self.qlearner.Q.tolist(), indent=2))

        edges_path = directory / f"{model_name}-bin-edges.json"
        edges_path.write_text(json.dumps(
            {col: edges.tolist() for col, edges in self.bin_edges.items()},
            indent=2,
        ))

        metadata = {
            "model_name": model_name,
            "policy_type": self.model_type,
            "feature_columns": self.feature_columns,
            "n_bins": self.n_bins,
            "n_epochs": self.n_epochs,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "impact": self.impact,
            "artifacts": {
                "qtable": str(q_path),
                "bin_edges": str(edges_path),
            },
        }
        meta_path = directory / f"{model_name}-policy-metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata

    def load(self, directory: Path, model_name: str) -> None:
        directory = Path(directory)
        meta_path = directory / f"{model_name}-policy-metadata.json"
        metadata = json.loads(meta_path.read_text())

        self.feature_columns = metadata["feature_columns"]
        self.n_bins = metadata["n_bins"]
        self.n_epochs = metadata["n_epochs"]
        self.alpha = metadata["alpha"]
        self.gamma = metadata["gamma"]
        self.impact = metadata.get("impact", 0.0)

        q_path = directory / f"{model_name}-qtable.json"
        q_data = json.loads(q_path.read_text())

        edges_path = directory / f"{model_name}-bin-edges.json"
        edges_data = json.loads(edges_path.read_text())
        self.bin_edges = {col: np.array(edges) for col, edges in edges_data.items()}

        self.qlearner = TabularQLearner(
            num_states=self._num_states(),
            num_actions=3,
            alpha=self.alpha,
            gamma=self.gamma,
        )
        self.qlearner.Q = np.array(q_data)
