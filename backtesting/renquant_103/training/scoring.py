"""Training-time score calibration fitting.

Provides fit_probability_calibration (requires sklearn) and re-exports
ScoreCalibration from kernel.scoring for convenience.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression as _LogisticRegression
    from sklearn.preprocessing import StandardScaler as _StandardScaler
except ImportError:  # pragma: no cover
    IsotonicRegression = None
    _LogisticRegression = None
    _StandardScaler = None

from kernel.scoring import ScoreCalibration, raw_score_kind_for_model

_ISOTONIC_MIN_SAMPLES = 300
_PLATT_MIN_SAMPLES = 120
_ER_LINEAR_MIN_SAMPLES = 60


def fit_probability_calibration(
    raw_scores: pd.Series,
    future_relative_returns: pd.Series,
    *,
    lookahead: int,
    threshold: float,
    score_kind: str,
) -> ScoreCalibration:
    joined = pd.DataFrame({
        "raw_score": pd.Series(raw_scores, dtype=float),
        "future_return": pd.Series(future_relative_returns, dtype=float),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if joined.empty:
        return ScoreCalibration(
            method="constant_probability", score_kind=score_kind,
            sample_size=0, lookahead=lookahead, threshold=threshold, base_rate=0.0,
        )

    targets = (joined["future_return"] > threshold).astype(int)
    raw_vals = joined["raw_score"].to_numpy(dtype=float)
    base_rate = float(targets.mean())
    n = len(joined)
    has_variance = targets.nunique() >= 2 and np.unique(raw_vals).size >= 2

    if n >= _ISOTONIC_MIN_SAMPLES and has_variance and IsotonicRegression is not None:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(raw_vals, targets.to_numpy())
        return ScoreCalibration(
            method="isotonic", score_kind=score_kind, sample_size=n,
            lookahead=lookahead, threshold=threshold, base_rate=base_rate,
            raw_min=float(np.min(raw_vals)), raw_max=float(np.max(raw_vals)),
            x_thresholds=[float(v) for v in isotonic.X_thresholds_],
            y_thresholds=[float(v) for v in isotonic.y_thresholds_],
        )

    if n >= _PLATT_MIN_SAMPLES and has_variance and _LogisticRegression is not None:
        scaler = _StandardScaler()
        raw_scaled = scaler.fit_transform(raw_vals.reshape(-1, 1)).ravel()
        lr = _LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(raw_scaled.reshape(-1, 1), targets.to_numpy())
        return ScoreCalibration(
            method="platt", score_kind=score_kind, sample_size=n,
            lookahead=lookahead, threshold=threshold, base_rate=base_rate,
            raw_min=float(np.min(raw_vals)), raw_max=float(np.max(raw_vals)),
            platt_coef=float(lr.coef_[0][0]),
            platt_intercept=float(lr.intercept_[0]),
            platt_scale_mean=float(scaler.mean_[0]),
            platt_scale_std=float(scaler.scale_[0]),
        )

    return ScoreCalibration(
        method="constant_probability", score_kind=score_kind, sample_size=n,
        lookahead=lookahead, threshold=threshold, base_rate=base_rate,
        raw_min=float(np.min(raw_vals)) if len(raw_vals) else None,
        raw_max=float(np.max(raw_vals)) if len(raw_vals) else None,
    )


def fit_expected_return_calibration(
    raw_scores: pd.Series,
    future_relative_returns: pd.Series,
    *,
    lookahead: int,
) -> dict:
    """Regress raw_score → continuous E[future_relative_return] (fraction units).

    Returns a dict of er_* fields to merge into a ScoreCalibration.  Used for
    rotation decisions: gives us "expected outperformance vs SPY over `lookahead`
    days" rather than a binary probability.

    Method selection by sample size:
      * n >= 300         — isotonic regression on continuous target
      * 60 <= n < 300    — linear (least-squares) regression
      * n < 60           — constant fallback (sample mean)
    """
    joined = pd.DataFrame({
        "raw_score":     pd.Series(raw_scores, dtype=float),
        "future_return": pd.Series(future_relative_returns, dtype=float),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if joined.empty:
        return {"er_method": "none", "er_lookahead": lookahead}

    raw_vals  = joined["raw_score"].to_numpy(dtype=float)
    targets   = joined["future_return"].to_numpy(dtype=float)
    n         = len(joined)
    has_var   = np.unique(raw_vals).size >= 2

    if n >= _ISOTONIC_MIN_SAMPLES and has_var and IsotonicRegression is not None:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_vals, targets)
        residuals = targets - iso.predict(raw_vals)
        return {
            "er_method":         "isotonic",
            "er_lookahead":      lookahead,
            "er_residual_std":   float(np.std(residuals)),
            "er_x_thresholds":   [float(v) for v in iso.X_thresholds_],
            "er_y_thresholds":   [float(v) for v in iso.y_thresholds_],
        }

    if n >= _ER_LINEAR_MIN_SAMPLES and has_var:
        slope, intercept = np.polyfit(raw_vals, targets, 1)
        residuals = targets - (slope * raw_vals + intercept)
        return {
            "er_method":         "linear",
            "er_lookahead":      lookahead,
            "er_residual_std":   float(np.std(residuals)),
            "er_coef":           float(slope),
            "er_intercept":      float(intercept),
        }

    mean_target = float(np.mean(targets))
    return {
        "er_method":       "constant",
        "er_lookahead":    lookahead,
        "er_constant":     mean_target,
        "er_residual_std": float(np.std(targets - mean_target)) if n > 1 else 0.0,
    }


__all__ = [
    "ScoreCalibration",
    "fit_probability_calibration",
    "fit_expected_return_calibration",
    "raw_score_kind_for_model",
]
