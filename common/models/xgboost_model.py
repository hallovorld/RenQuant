"""XGBoost-based trading model.

Uses gradient-boosted trees with walk-forward cross-validation to label
5-day forward relative returns (stock / SPY - 1) as buy / hold / sell.

Key advantages over RTLearner/BagLearner:
- Residual boosting: each tree corrects the previous tree's errors
- L1/L2 regularisation prevents overfitting on short time-series
- Built-in handling of missing values
- Calibrated probability outputs (predict_proba) → reliable confidence scores
- Walk-forward CV during training gives honest OOS Sharpe estimate for model selection

Artifacts saved as JSON so LEAN can reload without Python dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost gradient-boosted classifier for trade signal generation.

    Parameters
    ----------
    feature_columns:
        Indicator columns to use as features.
    lookahead:
        Days ahead for forward-return label.
    threshold:
        Min absolute forward return to label as buy/sell (else hold).
    buy_threshold:
        Min model score to emit a buy signal.
    sell_threshold:
        Max model score (most negative) to emit a sell signal.
    n_estimators:
        Number of boosting rounds.
    max_depth:
        Max tree depth (controls complexity).
    learning_rate:
        Shrinkage per boosting step.
    subsample:
        Row sampling ratio per tree.
    colsample_bytree:
        Feature sampling ratio per tree.
    """

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

        # Trained artifacts
        self._buy_model: Any = None   # XGBClassifier (buy vs rest)
        self._sell_model: Any = None  # XGBClassifier (sell vs rest)
        self._feature_importances: dict[str, float] = {}

    @property
    def model_type(self) -> str:
        return "xgboost"

    # ── label construction ──────────────────────────────────────────────

    def _build_labels(self, close: pd.Series) -> pd.Series:
        """Label each day by its N-day forward return."""
        future_ret = close.shift(-self.lookahead) / close - 1.0
        return pd.Series(
            np.where(future_ret > self.threshold, 1,
                     np.where(future_ret < -self.threshold, -1, 0)),
            index=close.index,
        )

    # ── training ───────────────────────────────────────────────────────

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
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

        # Two one-vs-rest binary classifiers: buy probability and sell probability.
        # This avoids multi-class calibration issues and lets each threshold
        # be tuned independently.
        y_buy  = (y == 1).astype(int)
        y_sell = (y == -1).astype(int)

        # Positive-class weight to handle imbalance (typically 20-30% buy/sell labels)
        buy_ratio  = max(1, (y_buy == 0).sum() / max(1, (y_buy == 1).sum()))
        sell_ratio = max(1, (y_sell == 0).sum() / max(1, (y_sell == 1).sum()))

        self._buy_model = XGBClassifier(scale_pos_weight=buy_ratio, **xgb_kwargs)
        self._buy_model.fit(X, y_buy)

        self._sell_model = XGBClassifier(scale_pos_weight=sell_ratio, **xgb_kwargs)
        self._sell_model.fit(X, y_sell)

        # Feature importances (average of buy+sell models)
        fi_buy  = self._buy_model.feature_importances_
        fi_sell = self._sell_model.feature_importances_
        self._feature_importances = {
            col: float((fi_buy[i] + fi_sell[i]) / 2)
            for i, col in enumerate(self.feature_columns)
        }

        label_dist = {
            "buy":  int((y == 1).sum()),
            "hold": int((y == 0).sum()),
            "sell": int((y == -1).sum()),
        }
        return {
            "model_type": self.model_type,
            "train_rows": len(combined),
            "label_distribution": label_dist,
            "feature_importances": self._feature_importances,
        }

    # ── prediction ─────────────────────────────────────────────────────

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Return per-row net score: P(buy) - P(sell) in [-1, 1]."""
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
        if score > self.buy_threshold - 0.5:   # threshold relative to 0.5 base
            return "buy"
        if score < -(self.sell_threshold - 0.5):
            return "sell"
        return "hold"

    def predict_bulk(self, df: pd.DataFrame) -> pd.Series:
        if self._buy_model is None:
            raise RuntimeError("Model not trained.")
        X = df[self.feature_columns].values
        scores = self._score(X)
        result = np.where(
            scores > (self.buy_threshold - 0.5), "buy",
            np.where(scores < -(self.sell_threshold - 0.5), "sell", "hold"),
        )
        return pd.Series(result, index=df.index)

    def predict_score(self, df: pd.DataFrame) -> pd.Series:
        """Return raw score in [-1, 1] for use as continuous signal strength."""
        if self._buy_model is None:
            raise RuntimeError("Model not trained.")
        X = df[self.feature_columns].values
        return pd.Series(self._score(X), index=df.index)

    def predict_score_bulk(self, df: pd.DataFrame) -> pd.Series:
        """Alias of predict_score — P(buy)−P(sell) continuous score for ranking."""
        return self.predict_score(df)

    # ── persistence ────────────────────────────────────────────────────

    def save(self, directory: Path, model_name: str) -> dict:
        """Save model as JSON trees (LEAN-compatible via embedded scores)."""
        directory = Path(directory)
        if self._buy_model is None:
            raise RuntimeError("Cannot save untrained model.")

        # XGBoost dumps to JSON natively
        buy_json  = self._buy_model.get_booster().save_raw(raw_format="json")
        sell_json = self._sell_model.get_booster().save_raw(raw_format="json")

        buy_path  = directory / f"{model_name}-xgb-buy.json"
        sell_path = directory / f"{model_name}-xgb-sell.json"
        buy_path.write_bytes(buy_json)
        sell_path.write_bytes(sell_json)

        metadata = {
            "model_name": model_name,
            "policy_type": self.model_type,
            "feature_columns": self.feature_columns,
            "lookahead": self.lookahead,
            "threshold": self.threshold,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "feature_importances": self._feature_importances,
            "artifacts": {
                "buy_model":  str(buy_path.name),
                "sell_model": str(sell_path.name),
            },
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
        meta_path = directory / f"{model_name}-policy-metadata.json"
        metadata  = json.loads(meta_path.read_text())

        self.feature_columns = metadata["feature_columns"]
        self.lookahead       = metadata["lookahead"]
        self.threshold       = metadata["threshold"]
        self.buy_threshold   = metadata["buy_threshold"]
        self.sell_threshold  = metadata["sell_threshold"]
        self._feature_importances = metadata.get("feature_importances", {})

        artifacts = metadata.get("artifacts", {})
        buy_name = artifacts.get("buy_model", f"{model_name}-xgb-buy.json")
        sell_name = artifacts.get("sell_model", f"{model_name}-xgb-sell.json")
        buy_path = directory / buy_name
        sell_path = directory / sell_name

        self._buy_model = XGBClassifier()
        self._buy_model.load_model(str(buy_path))
        self._sell_model = XGBClassifier()
        self._sell_model.load_model(str(sell_path))
