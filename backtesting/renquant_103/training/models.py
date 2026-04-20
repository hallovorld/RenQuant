"""Training-time model classes for renquant_103.

Provides ClassificationModel, QLearningModel, XGBoostModel, ManualModel
and the create_model factory.  Requires scikit-learn / xgboost at training
time; LEAN inference uses kernel/models.py (no ML lib dependencies).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .learners import BagLearner, RTLearner, TabularQLearner


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseModel(ABC):
    @abstractmethod
    def train(self, df: pd.DataFrame, **kwargs) -> dict: ...

    @abstractmethod
    def predict(self, state: pd.Series | pd.DataFrame) -> str: ...

    def predict_bulk(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(self.predict, axis=1)

    @abstractmethod
    def save(self, directory: Path, model_name: str) -> dict: ...

    @abstractmethod
    def load(self, directory: Path, model_name: str) -> None: ...

    @property
    @abstractmethod
    def model_type(self) -> str: ...


# ── ClassificationModel ───────────────────────────────────────────────────────

class ClassificationModel(BaseModel):
    def __init__(
        self,
        feature_columns: list[str] | None = None,
        lookahead: int = 10,
        threshold: float = 0.04,
        leaf_size: int = 25,
        bags: int = 15,
        impact: float = 0.0,
        buy_threshold: float = 0.5,
        sell_threshold: float = -0.5,
    ):
        self.feature_columns = feature_columns or ["rsi", "macd_hist", "cci"]
        self.lookahead = lookahead
        self.threshold = threshold
        self.leaf_size = leaf_size
        self.bags = bags
        self.impact = impact
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.learner: BagLearner | None = None

    @property
    def model_type(self) -> str:
        return "classification"

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

    def predict(self, state: pd.Series | pd.DataFrame) -> str:
        if self.learner is None:
            raise RuntimeError("Model not trained.")
        if isinstance(state, pd.Series):
            features = state[self.feature_columns].values.reshape(1, -1)
        else:
            features = state[self.feature_columns].values
            if features.ndim == 1:
                features = features.reshape(1, -1)
        pred = self.learner.query(features)[0]
        if pred > self.buy_threshold:
            return "buy"
        if pred < self.sell_threshold:
            return "sell"
        return "hold"

    def predict_bulk(self, df: pd.DataFrame) -> pd.Series:
        if self.learner is None:
            raise RuntimeError("Model not trained.")
        features = df[self.feature_columns].values
        preds = self.learner.query(features)
        result = np.where(
            preds > self.buy_threshold, "buy",
            np.where(preds < self.sell_threshold, "sell", "hold"),
        )
        return pd.Series(result, index=df.index)

    def predict_score_bulk(self, df: pd.DataFrame) -> pd.Series:
        if self.learner is None:
            raise RuntimeError("Model not trained.")
        features = df[self.feature_columns].values
        preds = self.learner.query(features)
        return pd.Series(preds.astype(float), index=df.index)

    def save(self, directory: Path, model_name: str) -> dict:
        directory = Path(directory)
        if self.learner is None:
            raise RuntimeError("Cannot save untrained model.")
        trees_data = [learner.tree.tolist() for learner in self.learner.learners]
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
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
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
        self.buy_threshold = metadata.get("buy_threshold", 0.5)
        self.sell_threshold = metadata.get("sell_threshold", -0.5)
        trees_path = directory / f"{model_name}-rf-trees.json"
        trees_data = json.loads(trees_path.read_text())
        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": self.leaf_size},
            bags=len(trees_data),
        )
        for i, tree_list in enumerate(trees_data):
            self.learner.learners[i].tree = np.array(tree_list)


# ── QLearningModel ────────────────────────────────────────────────────────────

ACTIONS = {0: 1000, 1: -1000, 2: 0}
ACTION_NAMES = {0: "buy", 1: "sell", 2: "hold"}


class QLearningModel(BaseModel):
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

    def _num_states(self) -> int:
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

    def train(self, df: pd.DataFrame, **kwargs) -> dict:
        features = df[self.feature_columns].shift(1)
        close = df["close"]
        daily_rets = close.pct_change().fillna(0.0)
        valid = features.dropna()
        valid_rets = daily_rets.loc[valid.index]

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

        return {"model_type": self.model_type, "n_epochs": self.n_epochs,
                "n_states": self._num_states(), "train_rows": len(valid)}

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

    def predict_bulk(self, df: pd.DataFrame) -> pd.Series:
        if self.qlearner is None or self.bin_edges is None:
            raise RuntimeError("Model not trained.")
        disc = self._discretize(df[self.feature_columns])
        pos_flags = df["position_flag"].values if "position_flag" in df.columns else np.zeros(len(df), dtype=int)
        results = []
        for i in range(len(df)):
            holding = int(pos_flags[i]) * 1000
            s = self._encode_state(disc[i], holding)
            action = self.qlearner.querysetstate(s)
            results.append(ACTION_NAMES[action])
        return pd.Series(results, index=df.index)

    def predict_score_bulk(self, df: pd.DataFrame) -> pd.Series:
        if self.qlearner is None or self.bin_edges is None:
            raise RuntimeError("Model not trained.")
        disc = self._discretize(df[self.feature_columns])
        pos_flags = df["position_flag"].values if "position_flag" in df.columns else np.zeros(len(df), dtype=int)
        scores = []
        for i in range(len(df)):
            holding = int(pos_flags[i]) * 1000
            s = self._encode_state(disc[i], holding)
            q_row = self.qlearner.Q[s]
            scores.append(float(q_row[0] - q_row[1]))
        return pd.Series(scores, index=df.index)

    def save(self, directory: Path, model_name: str) -> dict:
        directory = Path(directory)
        if self.qlearner is None or self.bin_edges is None:
            raise RuntimeError("Cannot save untrained model.")
        q_path = directory / f"{model_name}-qtable.json"
        q_path.write_text(json.dumps(self.qlearner.Q.tolist(), indent=2))
        edges_path = directory / f"{model_name}-bin-edges.json"
        edges_path.write_text(json.dumps(
            {col: edges.tolist() for col, edges in self.bin_edges.items()}, indent=2,
        ))
        metadata = {
            "model_name": model_name, "policy_type": self.model_type,
            "feature_columns": self.feature_columns, "n_bins": self.n_bins,
            "n_epochs": self.n_epochs, "alpha": self.alpha, "gamma": self.gamma,
            "impact": self.impact,
            "artifacts": {"qtable": str(q_path), "bin_edges": str(edges_path)},
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
        edges_path = directory / f"{model_name}-bin-edges.json"
        self.bin_edges = {col: np.array(e) for col, e in json.loads(edges_path.read_text()).items()}
        self.qlearner = TabularQLearner(
            num_states=self._num_states(), num_actions=3,
            alpha=self.alpha, gamma=self.gamma,
        )
        self.qlearner.Q = np.array(json.loads(q_path.read_text()))


# ── XGBoostModel ──────────────────────────────────────────────────────────────

class XGBoostModel(BaseModel):
    def __init__(
        self,
        feature_columns: list[str] | None = None,
        lookahead: int = 5,
        threshold: float = 0.03,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.55,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 10,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
    ):
        self.feature_columns = feature_columns or ["rsi", "macd_hist", "cci", "bbp", "adx"]
        self.lookahead = lookahead
        self.threshold = threshold
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self._buy_model: Any = None
        self._sell_model: Any = None
        self._feature_importances: dict[str, float] = {}

    @property
    def model_type(self) -> str:
        return "xgboost"

    def _build_labels(self, close: pd.Series) -> pd.Series:
        future_ret = close.shift(-self.lookahead) / close - 1.0
        return pd.Series(
            np.where(future_ret > self.threshold, 1,
                     np.where(future_ret < -self.threshold, -1, 0)),
            index=close.index,
        )

    def train(self, df: pd.DataFrame, **kwargs) -> dict:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed — run: pip install xgboost")

        features = df[self.feature_columns].shift(1)
        labels = self._build_labels(df["close"])
        combined = features.copy()
        combined["Y"] = labels
        combined = combined.dropna()
        if len(combined) < 50:
            raise ValueError(f"Insufficient training rows: {len(combined)}")

        X = combined[self.feature_columns].values
        y = combined["Y"].values

        xgb_kwargs = dict(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
            use_label_encoder=False, eval_metric="logloss", verbosity=0,
        )
        y_buy = (y == 1).astype(int)
        y_sell = (y == -1).astype(int)
        buy_ratio  = max(1, (y_buy == 0).sum() / max(1, (y_buy == 1).sum()))
        sell_ratio = max(1, (y_sell == 0).sum() / max(1, (y_sell == 1).sum()))

        self._buy_model = XGBClassifier(scale_pos_weight=buy_ratio, **xgb_kwargs)
        self._buy_model.fit(X, y_buy)
        self._sell_model = XGBClassifier(scale_pos_weight=sell_ratio, **xgb_kwargs)
        self._sell_model.fit(X, y_sell)

        fi_buy  = self._buy_model.feature_importances_
        fi_sell = self._sell_model.feature_importances_
        self._feature_importances = {
            col: float((fi_buy[i] + fi_sell[i]) / 2)
            for i, col in enumerate(self.feature_columns)
        }
        return {
            "model_type": self.model_type, "train_rows": len(combined),
            "label_distribution": {
                "buy": int((y == 1).sum()), "hold": int((y == 0).sum()), "sell": int((y == -1).sum()),
            },
            "feature_importances": self._feature_importances,
        }

    def _score(self, X: np.ndarray) -> np.ndarray:
        p_buy  = self._buy_model.predict_proba(X)[:, 1]
        p_sell = self._sell_model.predict_proba(X)[:, 1]
        return p_buy - p_sell

    def predict(self, state: pd.Series | pd.DataFrame) -> str:
        if self._buy_model is None:
            raise RuntimeError("Model not trained.")
        if isinstance(state, pd.Series):
            X = state[self.feature_columns].values.reshape(1, -1)
        else:
            X = state[self.feature_columns].values.reshape(1, -1)
        score = self._score(X)[0]
        if score > self.buy_threshold:
            return "buy"
        if score < -self.sell_threshold:
            return "sell"
        return "hold"

    def predict_bulk(self, df: pd.DataFrame) -> pd.Series:
        if self._buy_model is None:
            raise RuntimeError("Model not trained.")
        X = df[self.feature_columns].values
        scores = self._score(X)
        result = np.where(
            scores > self.buy_threshold, "buy",
            np.where(scores < -self.sell_threshold, "sell", "hold"),
        )
        return pd.Series(result, index=df.index)

    def predict_score_bulk(self, df: pd.DataFrame) -> pd.Series:
        if self._buy_model is None:
            raise RuntimeError("Model not trained.")
        X = df[self.feature_columns].values
        return pd.Series(self._score(X), index=df.index)

    def save(self, directory: Path, model_name: str) -> dict:
        directory = Path(directory)
        if self._buy_model is None:
            raise RuntimeError("Cannot save untrained model.")
        buy_json  = self._buy_model.get_booster().save_raw(raw_format="json")
        sell_json = self._sell_model.get_booster().save_raw(raw_format="json")
        buy_path  = directory / f"{model_name}-xgb-buy.json"
        sell_path = directory / f"{model_name}-xgb-sell.json"
        buy_path.write_bytes(buy_json)
        sell_path.write_bytes(sell_json)
        metadata = {
            "model_name": model_name, "policy_type": self.model_type,
            "feature_columns": self.feature_columns, "lookahead": self.lookahead,
            "threshold": self.threshold, "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold, "n_estimators": self.n_estimators,
            "max_depth": self.max_depth, "learning_rate": self.learning_rate,
            "feature_importances": self._feature_importances,
            "artifacts": {"buy_model": str(buy_path.name), "sell_model": str(sell_path.name)},
        }
        meta_path = directory / f"{model_name}-policy-metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata

    def load(self, directory: Path, model_name: str) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed — run: pip install xgboost")
        directory = Path(directory)
        metadata = json.loads((directory / f"{model_name}-policy-metadata.json").read_text())
        self.feature_columns = metadata["feature_columns"]
        self.lookahead = metadata["lookahead"]
        self.threshold = metadata["threshold"]
        self.buy_threshold = metadata["buy_threshold"]
        self.sell_threshold = metadata["sell_threshold"]
        self._feature_importances = metadata.get("feature_importances", {})
        artifacts = metadata.get("artifacts", {})
        buy_path  = directory / artifacts.get("buy_model", f"{model_name}-xgb-buy.json")
        sell_path = directory / artifacts.get("sell_model", f"{model_name}-xgb-sell.json")
        self._buy_model = XGBClassifier()
        self._buy_model.load_model(str(buy_path))
        self._sell_model = XGBClassifier()
        self._sell_model.load_model(str(sell_path))


# ── ManualModel ───────────────────────────────────────────────────────────────

DEFAULT_SCORE_RULES = [
    {"col": "rsi",       "buy_below": 30,   "sell_above": 70},
    {"col": "macd_hist", "buy_above": 0,     "sell_below": 0},
    {"col": "cci",       "buy_below": -100,  "sell_above": 50},
]


class ManualModel(BaseModel):
    def __init__(
        self,
        score_rules: list[dict] | None = None,
        buy_threshold: int = 2,
        sell_threshold: int = -2,
        rules: dict | None = None,
    ):
        if rules is not None and score_rules is None:
            score_rules = _convert_legacy_rules(rules)
            buy_threshold = rules.get("score_buy_threshold", buy_threshold)
            sell_threshold = rules.get("score_sell_threshold", sell_threshold)
        self.score_rules = score_rules or DEFAULT_SCORE_RULES
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    @property
    def model_type(self) -> str:
        return "manual"

    def train(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "model_type": self.model_type,
            "score_rules": self.score_rules,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
        }

    def _score(self, state: pd.Series) -> int:
        score = 0
        for rule in self.score_rules:
            val = state.get(rule["col"])
            if val is None:
                continue
            if "buy_below"  in rule and val < rule["buy_below"]:  score += 1
            if "buy_above"  in rule and val > rule["buy_above"]:  score += 1
            if "sell_above" in rule and val > rule["sell_above"]: score -= 1
            if "sell_below" in rule and val < rule["sell_below"]: score -= 1
        return score

    def predict(self, state: pd.Series | pd.DataFrame) -> str:
        if isinstance(state, pd.DataFrame):
            state = state.iloc[0]
        score = self._score(state)
        if score >= self.buy_threshold:  return "buy"
        if score <= self.sell_threshold: return "sell"
        return "hold"

    def predict_bulk(self, df: pd.DataFrame) -> pd.Series:
        score = np.zeros(len(df), dtype=int)
        for rule in self.score_rules:
            col = rule["col"]
            if col not in df.columns: continue
            vals = df[col]
            if "buy_below"  in rule: score += np.where(vals < rule["buy_below"],  1, 0)
            if "buy_above"  in rule: score += np.where(vals > rule["buy_above"],  1, 0)
            if "sell_above" in rule: score += np.where(vals > rule["sell_above"], -1, 0)
            if "sell_below" in rule: score += np.where(vals < rule["sell_below"], -1, 0)
        result = np.where(
            score >= self.buy_threshold, "buy",
            np.where(score <= self.sell_threshold, "sell", "hold"),
        )
        return pd.Series(result, index=df.index)

    def predict_score_bulk(self, df: pd.DataFrame) -> pd.Series:
        score = np.zeros(len(df), dtype=float)
        for rule in self.score_rules:
            col = rule["col"]
            if col not in df.columns: continue
            vals = df[col]
            if "buy_below"  in rule: score += np.where(vals < rule["buy_below"],  1.0, 0.0)
            if "buy_above"  in rule: score += np.where(vals > rule["buy_above"],  1.0, 0.0)
            if "sell_above" in rule: score += np.where(vals > rule["sell_above"], -1.0, 0.0)
            if "sell_below" in rule: score += np.where(vals < rule["sell_below"], -1.0, 0.0)
        return pd.Series(score, index=df.index)

    def save(self, directory: Path, model_name: str) -> dict:
        directory = Path(directory)
        rules_data = {
            "score_rules": self.score_rules,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
        }
        rules_path = directory / f"{model_name}-manual-rules.json"
        rules_path.write_text(json.dumps(rules_data, indent=2))
        metadata = {
            "model_name": model_name, "policy_type": self.model_type,
            **rules_data,
            "artifacts": {"rules": str(rules_path)},
        }
        meta_path = directory / f"{model_name}-policy-metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata

    def load(self, directory: Path, model_name: str) -> None:
        directory = Path(directory)
        rules_path = directory / f"{model_name}-manual-rules.json"
        data = json.loads(rules_path.read_text())
        if "score_rules" in data:
            self.score_rules = data["score_rules"]
            self.buy_threshold = data["buy_threshold"]
            self.sell_threshold = data["sell_threshold"]
        else:
            self.score_rules = _convert_legacy_rules(data)
            self.buy_threshold = data.get("score_buy_threshold", 2)
            self.sell_threshold = data.get("score_sell_threshold", -2)


def _convert_legacy_rules(rules: dict) -> list[dict]:
    converted = []
    if "rsi_oversold" in rules:
        converted.append({"col": "rsi", "buy_below": rules["rsi_oversold"], "sell_above": rules["rsi_overbought"]})
    if "macd_bullish" in rules:
        converted.append({"col": "macd_hist", "buy_above": rules["macd_bullish"], "sell_below": rules["macd_bullish"]})
    if "cci_oversold" in rules:
        converted.append({"col": "cci", "buy_below": rules["cci_oversold"], "sell_above": rules["cci_overbought"]})
    return converted


# ── Factory ───────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "manual": ManualModel,
    "classification": ClassificationModel,
    "qlearning": QLearningModel,
    "xgboost": XGBoostModel,
}


def create_model(model_type: str, **kwargs) -> BaseModel:
    """Factory: create a model by type name."""
    cls = MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model type {model_type!r}. Available: {list(MODEL_REGISTRY)}")
    return cls(**kwargs)


__all__ = [
    "BaseModel",
    "ClassificationModel",
    "QLearningModel",
    "XGBoostModel",
    "ManualModel",
    "MODEL_REGISTRY",
    "create_model",
]
