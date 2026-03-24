"""Rule-based (manual) trading model.

Scores each indicator against configurable thresholds and votes.
No machine-learning training is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .base import BaseModel

DEFAULT_RULES = {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_bullish": 0,       # macd_hist > 0
    "cci_oversold": -100,
    "cci_overbought": 50,
    "score_buy_threshold": 2,
    "score_sell_threshold": -2,
}


class ManualModel(BaseModel):
    """Indicator-threshold voting model.

    Scoring rules (each contributes +1 or -1):

    * RSI < ``rsi_oversold`` → +1 ;  RSI > ``rsi_overbought`` → -1
    * MACD hist > 0 → +1 ;  MACD hist < 0 → -1
    * CCI < ``cci_oversold`` → +1 ;  CCI > ``cci_overbought`` → -1

    If total score >= ``score_buy_threshold`` → **buy**
    If total score <= ``score_sell_threshold`` → **sell**
    Otherwise → **hold**
    """

    def __init__(self, rules: dict | None = None):
        self.rules = {**DEFAULT_RULES, **(rules or {})}

    @property
    def model_type(self) -> str:
        return "manual"

    # ── training (no-op) ───────────────────────────────────────────────

    def train(self, df: pd.DataFrame, **kwargs) -> dict:
        return {"model_type": self.model_type, "rules": self.rules}

    # ── prediction ─────────────────────────────────────────────────────

    def _score(self, state: pd.Series) -> int:
        score = 0
        rsi = state.get("rsi")
        macd_hist = state.get("macd_hist")
        cci = state.get("cci")

        if rsi is not None:
            if rsi < self.rules["rsi_oversold"]:
                score += 1
            elif rsi > self.rules["rsi_overbought"]:
                score -= 1

        if macd_hist is not None:
            if macd_hist > self.rules["macd_bullish"]:
                score += 1
            elif macd_hist < self.rules["macd_bullish"]:
                score -= 1

        if cci is not None:
            if cci < self.rules["cci_oversold"]:
                score += 1
            elif cci > self.rules["cci_overbought"]:
                score -= 1

        return score

    def predict(self, state: pd.Series | pd.DataFrame) -> str:
        if isinstance(state, pd.DataFrame):
            state = state.iloc[0]
        score = self._score(state)
        if score >= self.rules["score_buy_threshold"]:
            return "buy"
        if score <= self.rules["score_sell_threshold"]:
            return "sell"
        return "hold"

    # ── persistence ────────────────────────────────────────────────────

    def save(self, directory: Path, model_name: str) -> dict:
        directory = Path(directory)
        rules_path = directory / f"{model_name}-manual-rules.json"
        rules_path.write_text(json.dumps(self.rules, indent=2))

        metadata = {
            "model_name": model_name,
            "policy_type": self.model_type,
            "rules": self.rules,
            "artifacts": {"rules": str(rules_path)},
        }
        meta_path = directory / f"{model_name}-policy-metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata

    def load(self, directory: Path, model_name: str) -> None:
        directory = Path(directory)
        rules_path = directory / f"{model_name}-manual-rules.json"
        self.rules = json.loads(rules_path.read_text())
