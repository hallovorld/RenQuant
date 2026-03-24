"""Abstract base class for all trading models."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseModel(ABC):
    """Every model produces a trading signal: ``"hold"``, ``"buy"``, or ``"sell"``.

    Subclasses differ in how they are trained and how they make predictions,
    but all share the same lifecycle: ``train`` → ``save`` → ``load`` → ``predict``.
    """

    @abstractmethod
    def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """Train on historical data with indicators already computed.

        Args:
            df: DataFrame with OHLCV + indicator columns.

        Returns:
            Training metadata dict (metrics, hyperparams chosen, etc.).
        """
        ...

    @abstractmethod
    def predict(self, state: pd.Series | pd.DataFrame) -> str:
        """Predict action for the current market state.

        Args:
            state: Single-row Series/DataFrame with indicator values
                   and ``position_flag``.

        Returns:
            One of ``"hold"``, ``"buy"``, ``"sell"``.
        """
        ...

    @abstractmethod
    def save(self, directory: Path, model_name: str) -> dict:
        """Export model artifacts (JSON) and return policy metadata dict."""
        ...

    @abstractmethod
    def load(self, directory: Path, model_name: str) -> None:
        """Load model artifacts from a strategy directory."""
        ...

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Identifier string: ``'manual'``, ``'classification'``, etc."""
        ...
