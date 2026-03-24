"""Strategy abstraction: composes data + indicators + model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .data import fetch_ohlcv
from .indicators import compute_indicators
from .models import BaseModel, create_model


@dataclass
class StrategyConfig:
    """Full specification for a trading strategy."""

    name: str
    symbol: str
    model_type: str
    indicator_spec: dict[str, dict[str, Any]]
    model_params: dict[str, Any] = field(default_factory=dict)
    gate_rules: dict[str, str] | None = None
    data_start: str | None = None
    data_end: str | None = None
    data_provider: str = "yfinance"
    initial_cash: float = 100_000
    transaction_cost_bps: float = 5.0

    def to_dict(self) -> dict:
        return {
            "model_name": self.name,
            "stock_symbol": self.symbol,
            "model_type": self.model_type,
            "indicator_spec": self.indicator_spec,
            "model_params": self.model_params,
            "gate_rules": self.gate_rules,
            "data_src": self.data_provider,
            "backtest_start": self.data_start,
            "backtest_end": self.data_end,
            "initial_cash": self.initial_cash,
            "transaction_cost_bps": self.transaction_cost_bps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StrategyConfig:
        return cls(
            name=d["model_name"],
            symbol=d["stock_symbol"],
            model_type=d.get("model_type", "fqi"),
            indicator_spec=d.get("indicator_spec", {
                "rsi": {"period": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "cci": {"period": 20},
            }),
            model_params=d.get("model_params", {}),
            gate_rules=d.get("gate_rules"),
            data_start=d.get("backtest_start"),
            data_end=d.get("backtest_end"),
            data_provider=d.get("data_src", "yfinance"),
            initial_cash=d.get("initial_cash", 100_000),
            transaction_cost_bps=d.get("transaction_cost_bps", 5.0),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> StrategyConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))


class Strategy:
    """Compose data ingestion, indicators, and a model into a runnable strategy."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.model: BaseModel = create_model(config.model_type, **config.model_params)

    def fetch_data(self) -> pd.DataFrame:
        """Fetch OHLCV data for the configured symbol and date range."""
        return fetch_ohlcv(
            self.config.symbol,
            start=self.config.data_start,
            end=self.config.data_end,
            provider=self.config.data_provider,
        )

    def prepare_data(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Fetch data (if not provided) and compute indicators."""
        if df is None:
            df = self.fetch_data()
        return compute_indicators(df, self.config.indicator_spec)

    def research(self, df: pd.DataFrame | None = None) -> dict:
        """Full research pipeline: fetch → indicators → train.

        Returns training metadata.
        """
        df = self.prepare_data(df)
        return self.model.train(df, symbol=self.config.symbol)

    def export(self, strategy_dir: Path) -> dict:
        """Export model artifacts + strategy_config.json to *strategy_dir*."""
        strategy_dir = Path(strategy_dir)
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Save strategy config
        self.config.save(strategy_dir / "strategy_config.json")

        # Save model artifacts
        return self.model.save(strategy_dir, self.config.name)

    def signal(self, state: pd.Series | pd.DataFrame) -> str:
        """Get trading signal for the current market state."""
        return self.model.predict(state)

    def load_model(self, strategy_dir: Path) -> None:
        """Load trained model from a strategy directory."""
        self.model.load(Path(strategy_dir), self.config.name)
