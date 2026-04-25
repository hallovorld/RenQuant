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
    # expected-return regression (continuous target)
    # Fits raw_score → E[stock_return - SPY_return] over `er_lookahead` trading
    # days, in fraction units.  Used by rotation to compare candidates and held
    # positions on a dimensionally honest "8% better" rule.
    er_method: str = "none"          # "none", "isotonic", "linear", "constant"
    er_lookahead: int = 5
    er_residual_std: float | None = None
    er_x_thresholds: list[float] | None = None
    er_y_thresholds: list[float] | None = None
    er_coef: float | None = None
    er_intercept: float | None = None
    er_constant: float | None = None

    def calibrate(self, raw_score: float) -> float:
        # Audit fix SC-1 (Round 7, 2026-04-25): align NaN behaviour with
        # kernel/models.py:calibrate_score (which returns base_rate, not
        # 0.0). Pre-fix, the two calibration paths returned different
        # values on identical NaN input — kernel/models route returned
        # base_rate (e.g. 0.05) while this route returned 0.0. Downstream
        # tier gates compared rank_score < 0.10, so both paths skipped
        # the candidate, but the mismatch could surface anywhere a
        # downstream blend / weighted sum / persistence layer reads
        # rank_score directly.
        if raw_score is None or not np.isfinite(raw_score):
            return float(np.clip(self.base_rate, 0.0, 1.0))

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
            # Audit fix SC-PLATT (Round 2 deep audit, 2026-04-25): pre-fix,
            # when platt_scale_std was None or 0 (corrupt artifact, or
            # degenerate fit on constant input), the code silently fell
            # back to `scaled = raw_score`. But training/scoring.py ALWAYS
            # fits Platt on standardised inputs (StandardScaler) — so
            # `coef * raw_score + intercept` produces meaningless log-odds
            # when the scaler is missing (the coef expects standardized
            # input ~N(0,1), not raw scores in some arbitrary range).
            # Now: missing scale params → return base_rate, same as the
            # other "calibration data missing" branches.
            if (self.platt_scale_std is None
                    or not np.isfinite(self.platt_scale_std)
                    or self.platt_scale_std <= 0
                    or self.platt_scale_mean is None
                    or not np.isfinite(self.platt_scale_mean)):
                return float(np.clip(self.base_rate, 0.0, 1.0))
            scaled = (raw_score - self.platt_scale_mean) / self.platt_scale_std
            log_odds = self.platt_coef * scaled + self.platt_intercept
            return float(np.clip(1.0 / (1.0 + np.exp(-log_odds)), 0.0, 1.0))
        # Audit #69: unknown method — fall back to base_rate / identity so
        # a typo in metadata ("identy") doesn't crash production retrain.
        # Log once; downstream caller still gets a sensible probability.
        import logging
        logging.getLogger("kernel.scoring").warning(
            "ScoreCalibration: unknown method %r — returning base_rate %.3f",
            self.method, float(np.clip(self.base_rate, 0.0, 1.0)),
        )
        return float(np.clip(self.base_rate, 0.0, 1.0))

    def expected_return(
        self, raw_score: float, *, horizon_days: int | None = None
    ) -> float:
        """E[stock_return − SPY_return] in fraction units.

        Returned in `er_lookahead` units by default; pass `horizon_days` to
        scale linearly (additive-return assumption — fine for short horizons,
        breaks down past ~60 days).
        """
        if raw_score is None or not np.isfinite(raw_score):
            base = 0.0
        elif (self.er_method == "isotonic"
              and self.er_x_thresholds and self.er_y_thresholds):
            base = float(np.interp(
                raw_score, self.er_x_thresholds, self.er_y_thresholds,
            ))
        elif self.er_method == "linear" and self.er_coef is not None:
            base = float(self.er_coef * raw_score + (self.er_intercept or 0.0))
        elif self.er_method == "constant" and self.er_constant is not None:
            base = float(self.er_constant)
        else:
            base = 0.0

        if horizon_days is None or horizon_days == self.er_lookahead:
            return base
        if self.er_lookahead <= 0:
            return base
        return base * (horizon_days / self.er_lookahead)

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
    # Audit fix EVAL-NONE-CAL-NaN (Round 2 deep audit, 2026-04-25): when
    # calibration is None (uncalibrated fallback path used by tests +
    # the warm-up window before score_calibration is fit), pre-fix
    # passed raw_score through directly. If the model returned NaN
    # (extract_raw_score on a model that crashed), rank_score = NaN
    # propagated downstream and slipped past `< tier_threshold` checks.
    # The calibrated path already guards via SC-1 (returns base_rate
    # on NaN); mirror that guard on the None-calibration path.
    if calibration is not None:
        rank_score = calibration.calibrate(raw_score)
    else:
        rank_score = float(raw_score) if np.isfinite(raw_score) else 0.0
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
