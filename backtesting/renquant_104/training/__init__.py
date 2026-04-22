"""Training package for renquant_103.

Provides model classes, learners, scoring calibration, portfolio simulation,
and GMM regime training utilities.  Requires scikit-learn and xgboost at
training time.  The LEAN Docker only uses kernel/ (no sklearn dependency).
"""
from .models import (
    BaseModel,
    ClassificationModel,
    ManualModel,
    QLearningModel,
    XGBoostModel,
    MODEL_REGISTRY,
    create_model,
)
from .scoring import ScoreCalibration, fit_probability_calibration, raw_score_kind_for_model
from .portfolio import compute_portvals, portfolio_stats
from .regime import build_gmm_features, RegimeGMM
from .features import build_training_features, build_all_training_features
from .tournament import oos_sharpe, run_tournament, run_tournament_all
from .export import export_models, retrain_live_models

__all__ = [
    "BaseModel", "ClassificationModel", "ManualModel", "QLearningModel",
    "XGBoostModel", "MODEL_REGISTRY", "create_model",
    "ScoreCalibration", "fit_probability_calibration", "raw_score_kind_for_model",
    "compute_portvals", "portfolio_stats",
    "build_gmm_features", "RegimeGMM",
    "build_training_features", "build_all_training_features",
    "oos_sharpe", "run_tournament", "run_tournament_all",
    "export_models", "retrain_live_models",
]
