"""Score extraction and calibration helpers for cross-model comparison.

Inference-side only — no sklearn fitting.  See training/scoring.py for
fit_probability_calibration.

Self-contained: only numpy, pandas.  No common/ imports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ScoreCalibration:
    version: int = 1
    method: str = "identity"
    score_kind: str = "raw"
    target_kind: str = "forward_relative_return_gt_threshold"
    sample_size: int = 0
    lookahead: int = 5
    threshold: float = 0.03
    base_rate: float = 0.0
    raw_min: float | None = None
    raw_max: float | None = None
    # isotonic
    x_thresholds: list[float] | None = None
    y_thresholds: list[float] | None = None
    # platt (logistic regression on standardised raw score)
    platt_coef: float | None = None
    platt_intercept: float | None = None
    platt_scale_mean: float | None = None
    platt_scale_std: float | None = None

    def calibrate(self, raw_score: float) -> float:
        if raw_score is None or not np.isfinite(raw_score):
            return 0.0

        if self.method == "identity":
            return float(raw_score)
        if self.method == "constant_probability":
            return float(np.clip(self.base_rate, 0.0, 1.0))
        if self.method == "isotonic":
            if not self.x_thresholds or not self.y_thresholds:
                return float(np.clip(self.base_rate, 0.0, 1.0))
            return float(np.clip(
                np.interp(raw_score, self.x_thresholds, self.y_thresholds),
                0.0, 1.0,
            ))
        if self.method == "platt":
            if self.platt_coef is None or self.platt_intercept is None:
                return float(np.clip(self.base_rate, 0.0, 1.0))
            scaled = raw_score
            if self.platt_scale_std and self.platt_scale_std > 0:
                scaled = (raw_score - self.platt_scale_mean) / self.platt_scale_std
            log_odds = self.platt_coef * scaled + self.platt_intercept
            return float(np.clip(1.0 / (1.0 + np.exp(-log_odds)), 0.0, 1.0))
        raise ValueError(f"Unknown calibration method: {self.method}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScoreCalibration | None":
        if not data:
            return None
        return cls(**data)


@dataclass
class ScoreEvaluation:
    signal: str
    raw_score: float
    rank_score: float


def raw_score_kind_for_model(model: Any) -> str:
    model_type = getattr(model, "model_type", "unknown")
    return {
        "manual": "vote_count",
        "classification": "bag_learner_raw",
        "qlearning": "q_buy_minus_sell",
        "xgboost": "p_buy_minus_sell",
    }.get(model_type, "raw")


def extract_raw_score(model: Any, row: pd.Series) -> float:
    df_row = row.to_frame().T
    if hasattr(model, "predict_score_bulk"):
        return float(model.predict_score_bulk(df_row).iloc[0])
    if hasattr(model, "predict_score"):
        return float(model.predict_score(df_row).iloc[0])
    return {"buy": 1.0, "hold": 0.0, "sell": -1.0}.get(model.predict(row), 0.0)


def extract_raw_scores_bulk(model: Any, features: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_score_bulk"):
        scores = model.predict_score_bulk(features)
        return pd.Series(scores, index=features.index, dtype=float)
    if hasattr(model, "predict_score"):
        scores = model.predict_score(features)
        return pd.Series(scores, index=features.index, dtype=float)
    mapped = features.apply(model.predict, axis=1).map(
        {"buy": 1.0, "hold": 0.0, "sell": -1.0}
    ).fillna(0.0)
    return pd.Series(mapped, index=features.index, dtype=float)


def evaluate_row(
    model: Any, row: pd.Series, calibration: ScoreCalibration | None = None
) -> ScoreEvaluation:
    raw_score = extract_raw_score(model, row)
    rank_score = calibration.calibrate(raw_score) if calibration else float(raw_score)
    return ScoreEvaluation(
        signal=model.predict(row),
        raw_score=float(raw_score),
        rank_score=float(rank_score),
    )


__all__ = [
    "ScoreCalibration",
    "ScoreEvaluation",
    "raw_score_kind_for_model",
    "extract_raw_score",
    "extract_raw_scores_bulk",
    "evaluate_row",
]
