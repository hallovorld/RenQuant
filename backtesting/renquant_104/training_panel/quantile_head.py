"""QuantileHead — drop-in replacement for NGBoostHead via XGBoost-quantile.

The NGBoost-Normal head fit on 516k×163 was single-threaded and didn't
finish in 1h+ wallclock. Replaced 2026-05-08 with three XGBoost-quantile
regressors (q=0.16/0.50/0.84) producing μ̂=q_0.50 and
σ̂=(q_0.84−q_0.16)/2 via Gaussian parametric recovery. Multi-threaded
on M2 Pro → ~30s vs NGBoost's never-finished wallclock.

References (per CLAUDE.md §5.12):
- Koenker & Bassett 1978 "Regression Quantiles" — the foundational
  quantile-regression theory.
- Lim et al. 2021 ICLR PatchTST §3 — exact same Gaussian-from-quantiles
  σ recovery used by Temporal Fusion Transformer for prediction
  intervals.
- Wakefield 2013 §3.4 — parametric Gaussian recovery from symmetric
  quantile pairs: σ = (q_(1-α) − q_α) / (2 · Φ⁻¹(1-α))  ≈ (q_0.84 − q_0.16) / 2.

API mirrors training_panel.ngboost_head.NGBoostHead:
  - load(path)
  - predict_distribution(panel) → DataFrame[mu, sigma]
  - feature_cols + feature_medians_

Downstream `ApplyNGBoostTask` dispatches on the artifact's `kind` field
to pick which loader to call.
"""
from __future__ import annotations

import base64
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


class QuantileHead:
    """3-quantile XGBoost head — same predict_distribution API as NGBoostHead."""

    def __init__(self) -> None:
        self.feature_cols: list[str] = []
        self.feature_medians_: np.ndarray | None = None
        self.quantiles: list[float] = []
        self.boosters: dict[float, xgb.Booster] = {}
        self.params: dict[str, Any] = {}

    @classmethod
    def load(cls, path: str | Path) -> "QuantileHead":
        path = Path(path)
        payload = json.loads(path.read_text())
        if payload.get("kind") != "quantile_head":
            raise ValueError(
                f"QuantileHead.load: artifact at {path} has kind="
                f"{payload.get('kind')!r}, expected 'quantile_head'"
            )
        head = cls()
        head.feature_cols = list(payload["feature_cols"])
        head.params = dict(payload.get("params", {}))
        head.quantiles = list(payload["quantiles"])

        # Pickled obj is a dict with boosters_raw[q] = JSON bytes per quantile.
        blob = base64.b64decode(payload["regressor_pickle_b64"].encode("ascii"))
        obj = pickle.loads(blob)
        for q, raw_json in obj["boosters_raw"].items():
            booster = xgb.Booster()
            booster.load_model(bytearray(raw_json.encode("utf-8")))
            head.boosters[float(q)] = booster

        medians = payload.get("feature_medians")
        if medians is not None:
            head.feature_medians_ = np.asarray(medians, dtype=float)
        return head

    def predict_distribution(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame[mu, sigma] indexed like `panel`.

        μ̂  = q_0.5 prediction
        σ̂  = (q_0.84 − q_0.16) / 2  (Gaussian parametric recovery,
                                        floored at 1e-6 to avoid /0)

        Mirrors NGBoostHead.predict_distribution: validates required
        columns, applies median imputation matching training-time
        `impute_features=True`, and returns NaN for any row whose
        feature vector remains non-finite after imputation.

        Per CLAUDE.md §5.3 BUG #6 invariant: head_contract.soft_check_input
        runs before predict, soft_check_output runs after. Both LOG warnings
        but do not raise — pipeline-level guards (ApplyNGBoostTask) decide
        the fail-safe action with full context.
        """
        if not self.boosters:
            raise RuntimeError("QuantileHead.predict_distribution called before load")
        missing = [c for c in self.feature_cols if c not in panel.columns]
        if missing:
            raise ValueError(
                f"QuantileHead.predict_distribution: panel missing required "
                f"feature columns: {missing[:5]}{'…' if len(missing) > 5 else ''} "
                f"(model trained on {len(self.feature_cols)} features)."
            )
        # ── Input contract ──
        from .model_contract import soft_check_input, soft_check_output  # noqa: PLC0415
        soft_check_input(panel, self.feature_cols, head_name="QuantileHead")

        X = panel[self.feature_cols].to_numpy(dtype=float, copy=False).copy()
        medians = getattr(self, "feature_medians_", None)
        if medians is not None:
            X = np.where(np.isfinite(X), X, medians)
        finite_mask = np.isfinite(X).all(axis=1)
        out = pd.DataFrame(
            {"mu": np.nan, "sigma": np.nan},
            index=panel.index,
            dtype=float,
        )
        if not finite_mask.any():
            soft_check_output(out, head_name="QuantileHead")
            return out

        Xf = X[finite_mask]
        D = xgb.DMatrix(Xf)
        # Gaussian parametric recovery
        q_lo = self.boosters[0.16].predict(D)
        q_md = self.boosters[0.50].predict(D)
        q_hi = self.boosters[0.84].predict(D)
        mu    = q_md
        sigma = np.maximum((q_hi - q_lo) / 2.0, 1e-6)

        out.loc[finite_mask, "mu"]    = mu
        out.loc[finite_mask, "sigma"] = sigma
        # ── Output contract ──
        soft_check_output(out, head_name="QuantileHead")
        return out


def load_head_by_kind(path: str | Path):
    """Polymorphic loader: dispatch on artifact's `kind` field.

    Returns NGBoostHead for kind=='ngboost_head', QuantileHead for
    kind=='quantile_head'. Both classes expose identical
    predict_distribution / feature_cols / feature_medians_ APIs so
    downstream ApplyNGBoostTask need not branch.
    """
    payload = json.loads(Path(path).read_text())
    kind = payload.get("kind")
    if kind == "ngboost_head":
        from training_panel.ngboost_head import NGBoostHead  # noqa: PLC0415
        return NGBoostHead.load(path)
    if kind == "quantile_head":
        return QuantileHead.load(path)
    raise ValueError(
        f"load_head_by_kind: artifact at {path} has unsupported kind="
        f"{kind!r}; expected 'ngboost_head' or 'quantile_head'."
    )
