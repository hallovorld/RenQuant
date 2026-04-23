"""NGBoost head — Normal(μ, σ) over raw residual forward returns.

Complements PanelLTRModel by producing **both** a location (μ) and a scale
(σ) per candidate. Downstream:

  score = μ − λ·σ          (σ-aware ranking, replaces Gaussianized LTR score
                            when ngboost.enabled is true)
  σ-multiplier             (scales max_position_pct by σ_p50 / σ_i)

Trained on **raw** residuals (`compute_residual_returns` output, not
Gaussianized), so σ is on the return scale and directly consumable for
sizing.

Persistence is a single JSON artifact. The underlying NGBRegressor is
pickled and base64-encoded into the JSON payload so the file remains
self-contained (no separate .pkl alongside) and callers don't have to
invent a second codec path.
"""
from __future__ import annotations

import base64
import json
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal


DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.01,
    "minibatch_frac": 1.0,
    "natural_gradient": True,
    "verbose": False,
    "random_state": 17,
}


class NGBoostHead:
    """NGBoost Normal(μ, σ) regression head.

    API mirrors PanelLTRModel: train / predict / save / load, plus
    `predict_distribution` returning a two-column DataFrame.
    """

    def __init__(self, params: dict | None = None):
        self.params: dict[str, Any] = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.regressor: NGBRegressor | None = None
        self.feature_cols: list[str] = []

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        panel: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "residual_return_raw",
        sample_weight_col: str | None = "weight",
    ) -> dict:
        self.feature_cols = list(feature_cols)
        X = panel[feature_cols].values.astype(float)
        y = panel[label_col].values.astype(float)
        sw = None
        if sample_weight_col and sample_weight_col in panel.columns:
            sw = panel[sample_weight_col].values.astype(float)

        self.regressor = NGBRegressor(Dist=Normal, **self.params)
        if sw is not None:
            self.regressor.fit(X, y, sample_weight=sw)
        else:
            self.regressor.fit(X, y)

        preds = self.regressor.pred_dist(X)
        return {
            "n_rows": int(len(y)),
            "n_features": int(len(feature_cols)),
            "train_mu_mean": float(np.mean(preds.loc)),
            "train_sigma_mean": float(np.mean(preds.scale)),
        }

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_distribution(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame[mu, sigma] indexed like `panel`."""
        if self.regressor is None:
            raise RuntimeError("NGBoostHead.predict called before train/load")
        X = panel[self.feature_cols].values.astype(float)
        d = self.regressor.pred_dist(X)
        return pd.DataFrame(
            {"mu": d.loc, "sigma": d.scale},
            index=panel.index,
        )

    def predict_mu(self, panel: pd.DataFrame) -> pd.Series:
        return self.predict_distribution(panel)["mu"].rename("mu")

    def predict_sigma(self, panel: pd.DataFrame) -> pd.Series:
        return self.predict_distribution(panel)["sigma"].rename("sigma")

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        if self.regressor is None:
            raise RuntimeError("NGBoostHead.save called before train")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        blob = base64.b64encode(pickle.dumps(self.regressor)).decode("ascii")
        payload: dict[str, Any] = {
            "version": 1,
            "kind": "ngboost_head",
            "trained_date": str(date.today()),
            "feature_cols": list(self.feature_cols),
            "params": self.params,
            "regressor_pickle_b64": blob,
        }
        if metadata:
            payload.update({k: v for k, v in metadata.items() if k not in payload})
        path.write_text(json.dumps(payload, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "NGBoostHead":
        path = Path(path)
        payload = json.loads(path.read_text())
        if payload.get("kind") != "ngboost_head":
            raise ValueError(
                f"NGBoostHead.load: artifact at {path} is not an ngboost_head "
                f"(kind={payload.get('kind')!r})",
            )
        head = cls(params=payload.get("params"))
        head.feature_cols = list(payload["feature_cols"])
        head.regressor = pickle.loads(
            base64.b64decode(payload["regressor_pickle_b64"].encode("ascii")),
        )
        return head


# ── Scoring helpers ───────────────────────────────────────────────────────────

def combined_score(mu: pd.Series, sigma: pd.Series, lambda_sigma: float) -> pd.Series:
    """score = μ − λ·σ, preserving the (ticker) index of the inputs."""
    return (mu - float(lambda_sigma) * sigma).rename("ngboost_score")


def sigma_sizing_multiplier(
    sigma: pd.Series,
    *,
    floor: float = 0.3,
    ceiling: float = 1.0,
) -> pd.Series:
    """Per-row multiplier = clip(σ_median / σ_i, floor, ceiling).

    High-σ candidates get smaller allocations. The universe median sits
    at 1.0 (no change from baseline sizing). Candidates with σ ≤ median
    are capped at `ceiling` (default 1.0 — never oversize).
    """
    s = sigma.astype(float)
    med = float(s.median())
    if not np.isfinite(med) or med <= 0.0:
        return pd.Series(1.0, index=s.index, name="sigma_mult")
    mult = med / s.replace(0.0, np.nan)
    mult = mult.clip(lower=float(floor), upper=float(ceiling))
    return mult.fillna(1.0).rename("sigma_mult")
