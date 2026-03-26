"""Rule-based (manual) trading model.

Scores each indicator against configurable thresholds and votes.
No machine-learning training is required.

Each scoring rule is a dict::

    {"col": "rsi", "buy_below": 40, "sell_above": 70}

Available conditions (each contributes +1 when true):
  - ``buy_below``  : value < threshold → score +1  (oversold → bullish)
  - ``buy_above``  : value > threshold → score +1  (trend confirmed → bullish)
  - ``sell_below``  : value < threshold → score -1  (breakdown → bearish)
  - ``sell_above`` : value > threshold → score -1  (overbought → bearish)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseModel

DEFAULT_SCORE_RULES = [
    {"col": "rsi",       "buy_below": 30,   "sell_above": 70},
    {"col": "macd_hist", "buy_above": 0,     "sell_below": 0},
    {"col": "cci",       "buy_below": -100,  "sell_above": 50},
]


class ManualModel(BaseModel):
    """Indicator-threshold voting model.

    Each rule evaluates one indicator column and contributes +1 (bullish)
    or -1 (bearish) to a total score.  The action is determined by
    ``buy_threshold`` and ``sell_threshold``.

    Parameters:
        score_rules:    List of scoring rule dicts (see module docstring).
        buy_threshold:  Minimum score to trigger a buy signal.
        sell_threshold: Maximum score (negative) to trigger a sell signal.
        rules:          **Deprecated** — old-style dict, converted to
                        ``score_rules`` for backward compatibility.
    """

    def __init__(
        self,
        score_rules: list[dict] | None = None,
        buy_threshold: int = 2,
        sell_threshold: int = -2,
        rules: dict | None = None,
    ):
        # Backward compatibility: convert old-style rules dict
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

    # ── training (no-op) ───────────────────────────────────────────────

    def train(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "model_type": self.model_type,
            "score_rules": self.score_rules,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
        }

    # ── prediction ─────────────────────────────────────────────────────

    def _score(self, state: pd.Series) -> int:
        score = 0
        for rule in self.score_rules:
            val = state.get(rule["col"])
            if val is None:
                continue
            if "buy_below" in rule and val < rule["buy_below"]:
                score += 1
            if "buy_above" in rule and val > rule["buy_above"]:
                score += 1
            if "sell_above" in rule and val > rule["sell_above"]:
                score -= 1
            if "sell_below" in rule and val < rule["sell_below"]:
                score -= 1
        return score

    def predict(self, state: pd.Series | pd.DataFrame) -> str:
        if isinstance(state, pd.DataFrame):
            state = state.iloc[0]
        score = self._score(state)
        if score >= self.buy_threshold:
            return "buy"
        if score <= self.sell_threshold:
            return "sell"
        return "hold"

    def predict_bulk(self, df: pd.DataFrame) -> pd.Series:
        score = np.zeros(len(df), dtype=int)
        for rule in self.score_rules:
            col = rule["col"]
            if col not in df.columns:
                continue
            vals = df[col]
            if "buy_below" in rule:
                score += np.where(vals < rule["buy_below"], 1, 0)
            if "buy_above" in rule:
                score += np.where(vals > rule["buy_above"], 1, 0)
            if "sell_above" in rule:
                score += np.where(vals > rule["sell_above"], -1, 0)
            if "sell_below" in rule:
                score += np.where(vals < rule["sell_below"], -1, 0)
        result = np.where(
            score >= self.buy_threshold, "buy",
            np.where(score <= self.sell_threshold, "sell", "hold"),
        )
        return pd.Series(result, index=df.index)

    # ── persistence ────────────────────────────────────────────────────

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
            "model_name": model_name,
            "policy_type": self.model_type,
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
            # Legacy format
            self.score_rules = _convert_legacy_rules(data)
            self.buy_threshold = data.get("score_buy_threshold", 2)
            self.sell_threshold = data.get("score_sell_threshold", -2)


def _convert_legacy_rules(rules: dict) -> list[dict]:
    """Convert old-style flat rules dict to score_rules list."""
    converted = []
    if "rsi_oversold" in rules:
        converted.append({
            "col": "rsi",
            "buy_below": rules["rsi_oversold"],
            "sell_above": rules["rsi_overbought"],
        })
    if "macd_bullish" in rules:
        converted.append({
            "col": "macd_hist",
            "buy_above": rules["macd_bullish"],
            "sell_below": rules["macd_bullish"],
        })
    if "cci_oversold" in rules:
        converted.append({
            "col": "cci",
            "buy_below": rules["cci_oversold"],
            "sell_above": rules["cci_overbought"],
        })
    return converted
