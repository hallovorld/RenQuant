"""Fitted Q-Iteration model with XGBoost.

Extracted from the research notebook.  Trains one XGBRegressor per action
(hold / buy / sell) over multiple FQI iterations with a discount factor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .base import BaseModel

_DEFAULT_XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    objective="reg:squarederror",
)


class FQIModel(BaseModel):
    """Fitted Q-Iteration with XGBoost function approximation.

    Parameters:
        state_columns:  Feature columns that define the state vector.
        n_iter:         Number of FQI iterations.
        gamma:          Discount factor.
        transaction_cost_bps: Transaction cost in basis points for reward.
        gate_rules:     Dict describing buy/sell gate conditions (metadata only).
        xgb_params:     Override dict for XGBRegressor hyperparameters.
    """

    def __init__(
        self,
        state_columns: list[str] | None = None,
        n_iter: int = 8,
        gamma: float = 0.95,
        transaction_cost_bps: float = 5,
        gate_rules: dict[str, str] | None = None,
        xgb_params: dict | None = None,
    ):
        self.state_columns = state_columns or [
            "macd_line", "macd_signal", "macd_hist", "rsi", "cci", "position_flag",
        ]
        self.n_iter = n_iter
        self.gamma = gamma
        self.transaction_cost_bps = transaction_cost_bps
        self.gate_rules = gate_rules or {
            "buy": "macd_line crosses above macd_signal and rsi > 50 while flat",
            "sell": "macd_line crosses below macd_signal and rsi < 50 while long",
            "hold": "always valid",
        }
        self.xgb_params = {**_DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.models: dict[str, XGBRegressor | None] = {
            "hold": None, "buy": None, "sell": None,
        }

    @property
    def model_type(self) -> str:
        return "fqi"

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def build_transitions(
        df: pd.DataFrame,
        state_columns: list[str],
        transaction_cost: float = 5e-4,
    ) -> pd.DataFrame:
        """Build state-transition records from an indicator-enriched DataFrame.

        The DataFrame must have ``buy_signal`` and ``sell_signal`` bool columns
        (gate logic), plus all columns listed in *state_columns*.
        """
        df = df.copy()
        df["next_return"] = df["close"].pct_change().shift(-1)
        df = df.iloc[:-1].copy()

        records: list[dict] = []
        for i in range(len(df) - 1):
            cur = df.iloc[i]
            nxt = df.iloc[i + 1]
            next_ret = cur["next_return"]

            for pos in (0, 1):
                valid = ["hold"]
                if pos == 0 and bool(cur["buy_signal"]):
                    valid.append("buy")
                if pos == 1 and bool(cur["sell_signal"]):
                    valid.append("sell")

                for action in valid:
                    next_pos = pos
                    if action == "buy":
                        next_pos = 1
                        reward = next_ret - transaction_cost
                    elif action == "sell":
                        next_pos = 0
                        reward = -transaction_cost
                    else:
                        reward = next_ret if pos == 1 else 0.0

                    rec: dict = {
                        "action_name": action,
                        "reward": reward,
                        "buy_signal": int(bool(cur["buy_signal"])),
                        "sell_signal": int(bool(cur["sell_signal"])),
                        "next_buy_signal": int(bool(nxt["buy_signal"])),
                        "next_sell_signal": int(bool(nxt["sell_signal"])),
                        "next_position_flag": next_pos,
                    }
                    for col in state_columns:
                        if col == "position_flag":
                            rec[col] = pos
                            rec[f"next_{col}"] = next_pos
                        else:
                            rec[col] = cur[col]
                            rec[f"next_{col}"] = nxt[col]
                    records.append(rec)

        return pd.DataFrame(records)

    def _score_valid_actions(
        self, features: pd.DataFrame, buy_sig, sell_sig, pos_flag,
    ) -> pd.DataFrame:
        zero = np.zeros(len(features))
        scores = pd.DataFrame(index=features.index)

        for action in ("hold", "buy", "sell"):
            m = self.models[action]
            raw = zero if m is None else m.predict(features)
            if action == "buy":
                scores[action] = np.where(
                    (pos_flag == 0) & (buy_sig == 1), raw, -np.inf,
                )
            elif action == "sell":
                scores[action] = np.where(
                    (pos_flag == 1) & (sell_sig == 1), raw, -np.inf,
                )
            else:
                scores[action] = raw
        return scores

    # ── training ───────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """Train FQI from a DataFrame that already has indicator + gate columns.

        *df* must contain: all ``state_columns``, ``buy_signal``, ``sell_signal``,
        and ``close``.
        """
        tc = self.transaction_cost_bps / 10_000
        transitions = self.build_transitions(df, self.state_columns, tc)

        targets = transitions["reward"].copy()
        action_names = ["hold", "buy", "sell"]

        for _ in range(self.n_iter):
            next_feats = transitions[[f"next_{c}" for c in self.state_columns]].copy()
            next_feats.columns = self.state_columns

            next_scores = self._score_valid_actions(
                next_feats,
                transitions["next_buy_signal"],
                transitions["next_sell_signal"],
                transitions["next_position_flag"],
            )
            max_next_q = next_scores.max(axis=1).replace(-np.inf, 0.0)
            targets = transitions["reward"] + self.gamma * max_next_q

            for action in action_names:
                subset = transitions[transitions["action_name"] == action]
                if subset.empty:
                    continue
                model = XGBRegressor(**self.xgb_params)
                model.fit(subset[self.state_columns], targets.loc[subset.index])
                self.models[action] = model

        return {
            "model_type": self.model_type,
            "n_iter": self.n_iter,
            "gamma": self.gamma,
            "transition_count": len(transitions),
            "action_distribution": transitions["action_name"].value_counts().to_dict(),
        }

    # ── prediction ─────────────────────────────────────────────────────

    def predict(self, state: pd.Series | pd.DataFrame) -> str:
        if isinstance(state, pd.Series):
            state = state.to_frame().T
        features = state[self.state_columns].astype(float)
        buy_sig = pd.to_numeric(state.get("buy_signal", pd.Series(0, index=state.index)))
        sell_sig = pd.to_numeric(state.get("sell_signal", pd.Series(0, index=state.index)))
        pos_flag = pd.to_numeric(state.get("position_flag", pd.Series(0, index=state.index)))

        scores = self._score_valid_actions(features, buy_sig, sell_sig, pos_flag)
        return scores.iloc[0].idxmax()

    # ── persistence ────────────────────────────────────────────────────

    def save(self, directory: Path, model_name: str) -> dict:
        directory = Path(directory)
        artifact_paths: dict[str, str] = {}

        for action, model in self.models.items():
            if model is None:
                continue
            path = directory / f"{model_name}-q-{action}.json"
            model.save_model(path)
            artifact_paths[action] = str(path)

        metadata = {
            "model_name": model_name,
            "policy_type": self.model_type,
            "actions": {"hold": 0, "buy": 1, "sell": 2},
            "state_columns": self.state_columns,
            "gamma": self.gamma,
            "transaction_cost_bps": self.transaction_cost_bps,
            "gate_rules": self.gate_rules,
            "artifacts": artifact_paths,
        }
        meta_path = directory / f"{model_name}-policy-metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata

    def load(self, directory: Path, model_name: str) -> None:
        directory = Path(directory)
        meta_path = directory / f"{model_name}-policy-metadata.json"
        metadata = json.loads(meta_path.read_text())

        self.state_columns = metadata["state_columns"]
        self.gamma = metadata["gamma"]
        self.transaction_cost_bps = metadata.get("transaction_cost_bps", 5)
        self.gate_rules = metadata.get("gate_rules", self.gate_rules)

        for action, path_str in metadata.get("artifacts", {}).items():
            path = Path(path_str)
            if not path.exists():
                path = directory / path.name
            model = XGBRegressor()
            model.load_model(path)
            self.models[action] = model
