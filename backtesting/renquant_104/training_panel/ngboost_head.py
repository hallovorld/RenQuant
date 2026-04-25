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
        """Fit NGBRegressor(Normal) on the panel.

        Audit fixes (2026-04-25):
          N-1 / N-13 ─ drop rows with NaN/±inf in any feature column or
                       in the label. Pre-fix, these slipped through and
                       NGBoost either segfaulted or fit on garbage.
          N-22       ─ drop rows whose sample-weight is NaN/non-positive.
        """
        self.feature_cols = list(feature_cols)
        # Build a clean view: finite features, finite label, finite + non-negative weight.
        feat_arr = panel[feature_cols].to_numpy(dtype=float, copy=False)
        finite_feat_mask = np.isfinite(feat_arr).all(axis=1)
        label_arr = panel[label_col].to_numpy(dtype=float, copy=False)
        finite_label_mask = np.isfinite(label_arr)
        keep = finite_feat_mask & finite_label_mask
        if sample_weight_col and sample_weight_col in panel.columns:
            w_arr = panel[sample_weight_col].to_numpy(dtype=float, copy=False)
            keep = keep & np.isfinite(w_arr) & (w_arr >= 0.0)
        n_dropped = int(len(panel) - keep.sum())
        if n_dropped:
            import logging  # noqa: PLC0415
            logging.getLogger("ngboost").info(
                "NGBoostHead.train: dropped %d/%d rows with NaN/inf in "
                "features/label/weight", n_dropped, len(panel),
            )
        sub = panel.loc[keep]
        if len(sub) < 10:
            raise ValueError(
                f"NGBoostHead.train: too few clean rows ({len(sub)} after "
                f"NaN/inf drop). Check feature pipeline."
            )
        X = sub[feature_cols].to_numpy(dtype=float)
        y = sub[label_col].to_numpy(dtype=float)
        sw = None
        if sample_weight_col and sample_weight_col in sub.columns:
            sw = sub[sample_weight_col].to_numpy(dtype=float)

        self.regressor = NGBRegressor(Dist=Normal, **self.params)
        if sw is not None:
            self.regressor.fit(X, y, sample_weight=sw)
        else:
            self.regressor.fit(X, y)

        preds = self.regressor.pred_dist(X)
        # Audit N-4 (2026-04-25): also report fit-time IC of μ̂ vs y so
        # downstream metadata captures one usable signal-quality number
        # per training run (still no CV — see N-17 for full fix).
        try:
            from scipy.stats import spearmanr  # noqa: PLC0415
            rho, _ = spearmanr(preds.loc, y)
            train_ic = float(rho) if rho == rho else float("nan")
        except Exception:
            train_ic = float("nan")
        return {
            "n_rows": int(len(y)),
            "n_rows_dropped": n_dropped,
            "n_features": int(len(feature_cols)),
            "train_mu_mean":    float(np.mean(preds.loc)),
            "train_sigma_mean": float(np.mean(preds.scale)),
            "train_mu_ic":      train_ic,
        }

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_distribution(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame[mu, sigma] indexed like `panel`.

        Audit fix N-5 (2026-04-25): if any input row has NaN/inf in a
        feature column, NGBoost will either error or produce garbage
        predictions. Pre-fix, that exception was swallowed by
        ApplyNGBoostTask → silent NGBoost no-op. Post-fix, NaN-row
        predictions are returned as NaN (downstream can detect + skip),
        and finite rows score normally.
        """
        if self.regressor is None:
            raise RuntimeError("NGBoostHead.predict called before train/load")
        X = panel[self.feature_cols].to_numpy(dtype=float, copy=False)
        finite_mask = np.isfinite(X).all(axis=1)
        out = pd.DataFrame(
            {"mu": np.nan, "sigma": np.nan},
            index=panel.index,
            dtype=float,
        )
        if finite_mask.any():
            d = self.regressor.pred_dist(X[finite_mask])
            out.loc[panel.index[finite_mask], "mu"]    = d.loc
            out.loc[panel.index[finite_mask], "sigma"] = d.scale
        return out

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
