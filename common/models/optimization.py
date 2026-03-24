"""Optimization-based trading model.

Uses SciPy Nelder-Mead to search over indicator parameters while
training an inner classification model.  The objective is in-sample
cumulative trading return.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from ..indicators import compute_indicators
from ..portfolio import compute_portvals
from .base import BaseModel
from .classification import ClassificationModel


class OptimizationModel(BaseModel):
    """Meta-model that optimises indicator parameters via Nelder-Mead.

    The outer loop searches over indicator windows; the inner loop trains
    a :class:`ClassificationModel`, generates trades, simulates portfolio
    performance, and returns the cumulative return as the objective.

    Parameters:
        indicator_spec_template: Base indicator spec. Keys whose values
            will be optimised should use placeholder values.
        optimizable_params: List of ``(indicator_name, param_name, init)``
            tuples that Nelder-Mead will search over.
        inner_model_kwargs: Kwargs forwarded to ClassificationModel.
        max_iter: Maximum Nelder-Mead iterations.
        start_val: Initial cash for portfolio simulation.
        commission: Commission per trade.
        impact: Market impact fraction.
    """

    def __init__(
        self,
        indicator_spec_template: dict | None = None,
        optimizable_params: list[tuple[str, str, float]] | None = None,
        inner_model_kwargs: dict | None = None,
        max_iter: int = 30,
        start_val: float = 100_000,
        commission: float = 0.0,
        impact: float = 0.0,
    ):
        self.indicator_spec_template = indicator_spec_template or {
            "rsi": {"period": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "cci": {"period": 20},
        }
        self.optimizable_params = optimizable_params or [
            ("rsi", "period", 14.0),
            ("macd", "fast", 12.0),
            ("macd", "slow", 26.0),
            ("cci", "period", 20.0),
        ]
        self.inner_model_kwargs = inner_model_kwargs or {}
        self.max_iter = max_iter
        self.start_val = start_val
        self.commission = commission
        self.impact = impact

        self.best_params: np.ndarray | None = None
        self.best_spec: dict | None = None
        self.inner_model: ClassificationModel | None = None

    @property
    def model_type(self) -> str:
        return "optimization"

    # ── helpers ────────────────────────────────────────────────────────

    def _build_spec(self, param_vector: np.ndarray) -> dict:
        """Apply rounded optimizer params onto the indicator spec template."""
        import copy
        spec = copy.deepcopy(self.indicator_spec_template)
        for i, (ind_name, param_name, _) in enumerate(self.optimizable_params):
            val = max(2, int(round(param_vector[i])))
            spec[ind_name][param_name] = val
        # Ensure MACD slow > fast
        if "macd" in spec:
            if spec["macd"].get("slow", 26) <= spec["macd"].get("fast", 12):
                spec["macd"]["slow"] = spec["macd"]["fast"] + 1
        return spec

    def _objective(self, params: np.ndarray, df_raw: pd.DataFrame, symbol: str) -> float:
        spec = self._build_spec(params)
        try:
            df = compute_indicators(df_raw, spec)
        except Exception:
            return 1.0

        inner = ClassificationModel(impact=self.impact, **self.inner_model_kwargs)
        meta = inner.train(df)
        if meta["train_rows"] < 20:
            return 1.0

        # Generate trades from the trained model
        features = df[inner.feature_columns].shift(1)
        valid = features.dropna()

        preds = inner.learner.query(valid.values)
        pred_series = pd.Series(0.0, index=df.index)
        pred_series[valid.index] = preds

        trades = pd.DataFrame(0.0, index=df.index, columns=[symbol])
        holding = 0
        for i in range(len(df)):
            p = pred_series.iloc[i]
            target = 1000 if p > 0.5 else (-1000 if p < -0.5 else holding)
            trade = target - holding
            if trade != 0:
                trades.iloc[i, 0] = float(trade)
            holding = target

        portvals = compute_portvals(
            trades, df, start_val=self.start_val,
            commission=self.commission, impact=self.impact,
        )
        cum_ret = portvals.iloc[-1] / portvals.iloc[0] - 1.0
        return -cum_ret  # minimize negative return

    # ── training ───────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, **kwargs) -> dict:
        symbol = kwargs.get("symbol", "STOCK")
        # Strip indicator columns — we'll recompute with candidate params
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        df_raw = df[[c for c in ohlcv_cols if c in df.columns]].copy()

        x0 = np.array([init for _, _, init in self.optimizable_params])
        result = optimize.minimize(
            self._objective, x0, args=(df_raw, symbol),
            method="Nelder-Mead",
            options={"maxiter": self.max_iter, "xatol": 1.0, "fatol": 0.001},
        )

        self.best_params = result.x
        self.best_spec = self._build_spec(self.best_params)

        # Retrain final inner model with best params
        df_best = compute_indicators(df_raw, self.best_spec)
        self.inner_model = ClassificationModel(impact=self.impact, **self.inner_model_kwargs)
        self.inner_model.train(df_best)

        return {
            "model_type": self.model_type,
            "best_indicator_spec": self.best_spec,
            "objective_value": float(result.fun),
            "n_iterations": int(result.nit),
        }

    # ── prediction ─────────────────────────────────────────────────────

    def predict(self, state: pd.Series | pd.DataFrame) -> str:
        if self.inner_model is None:
            raise RuntimeError("Model not trained.")
        return self.inner_model.predict(state)

    # ── persistence ────────────────────────────────────────────────────

    def save(self, directory: Path, model_name: str) -> dict:
        directory = Path(directory)
        if self.inner_model is None or self.best_spec is None:
            raise RuntimeError("Cannot save untrained model.")

        # Save inner model
        inner_meta = self.inner_model.save(directory, f"{model_name}-inner")

        metadata = {
            "model_name": model_name,
            "policy_type": self.model_type,
            "best_indicator_spec": self.best_spec,
            "best_params": self.best_params.tolist() if self.best_params is not None else None,
            "optimizable_params": [
                {"indicator": ind, "param": par, "init": init}
                for ind, par, init in self.optimizable_params
            ],
            "inner_model": inner_meta,
            "max_iter": self.max_iter,
            "impact": self.impact,
        }
        meta_path = directory / f"{model_name}-policy-metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata

    def load(self, directory: Path, model_name: str) -> None:
        directory = Path(directory)
        meta_path = directory / f"{model_name}-policy-metadata.json"
        metadata = json.loads(meta_path.read_text())

        self.best_spec = metadata["best_indicator_spec"]
        self.best_params = np.array(metadata["best_params"]) if metadata.get("best_params") else None
        self.max_iter = metadata.get("max_iter", 30)
        self.impact = metadata.get("impact", 0.0)
        self.optimizable_params = [
            (p["indicator"], p["param"], p["init"])
            for p in metadata.get("optimizable_params", [])
        ]

        self.inner_model = ClassificationModel(impact=self.impact)
        self.inner_model.load(directory, f"{model_name}-inner")
