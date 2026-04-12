"""Trading model library.

Six model types available::

    from common.models import ManualModel, ClassificationModel, QLearningModel, FQIModel, OptimizationModel, XGBoostModel
"""

from .base import BaseModel
from .classification import ClassificationModel
from .fqi import FQIModel
from .manual import ManualModel
from .optimization import OptimizationModel
from .qlearning import QLearningModel
from .xgboost_model import XGBoostModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "manual": ManualModel,
    "classification": ClassificationModel,
    "qlearning": QLearningModel,
    "fqi": FQIModel,
    "optimization": OptimizationModel,
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
    "FQIModel",
    "ManualModel",
    "OptimizationModel",
    "QLearningModel",
    "XGBoostModel",
    "MODEL_REGISTRY",
    "create_model",
]
