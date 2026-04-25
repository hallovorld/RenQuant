"""XGBoost learning-to-rank wrapper for the Stage-1 panel.

Trains `rank:pairwise` with `group = per-date row counts`, producing a
continuous per-row score. JSON-serializable for LEAN/live-runner loading
(no pickle).

Public API::

    PanelLTRModel(params=None)
        .train(panel, group_sizes, feature_cols, ...) -> dict
        .predict(panel) -> pd.Series
        .save(path, metadata)
        PanelLTRModel.load(path)

The produced artifact is a single JSON file with the booster dumped via
`booster.save_raw(raw_format="json")` so inference code can rebuild the
booster without any unpickling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr


DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "rank:pairwise",
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "lambda": 1.0,
    "alpha": 0.5,
    "tree_method": "hist",
    "nthread": -1,
    "verbosity": 0,
}


def _mean_ic(panel: pd.DataFrame, preds: np.ndarray,
             label_col: str, date_col: str = "date") -> float:
    df = pd.DataFrame({
        "date": panel[date_col].values,
        "p":    preds,
        "y":    panel[label_col].values,
    })
    ics: list[float] = []
    for _, g in df.groupby("date", sort=False):
        y = g["y"].values
        p = g["p"].values
        if len(y) < 2 or np.all(y == y[0]) or np.all(p == p[0]):
            continue
        rho, _ = spearmanr(p, y)
        if not np.isnan(rho):
            ics.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan")


class PanelLTRModel:
    """XGBoost rank:pairwise wrapper."""

    def __init__(self, params: dict | None = None,
                 monotone_constraints: dict[str, int] | None = None):
        self.params: dict[str, Any] = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.booster: xgb.Booster | None = None
        self.feature_cols: list[str] = []
        self.best_iter: int | None = None
        # Sign constraints per feature name: {"roe_z": +1, "beta_60d_z": -1, ...}
        # Sign 0 or missing entries are unconstrained. The XGBoost-formatted
        # tuple is built during train() once feature_cols is known.
        self.monotone_constraints: dict[str, int] = dict(monotone_constraints or {})

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        panel: pd.DataFrame,
        group_sizes: np.ndarray,
        feature_cols: list[str],
        label_col: str = "label",
        weight_col: str | None = "weight",
        num_boost_round: int = 400,
        early_stopping_rounds: int | None = None,
        eval_panel: pd.DataFrame | None = None,
        eval_group_sizes: np.ndarray | None = None,
    ) -> dict:
        """Fit the booster; return train/eval metadata.

        Round-3 audit (#R3-9 #R3-10): the prior signature accepted
        `early_stopping_rounds` and `eval_panel`/`eval_group_sizes` but
        the body hardcoded `evals=None, early_stopping_rounds=None` —
        the parameters were silently ignored. Default is now None
        (matching the actual behaviour); if the caller passes both
        a non-None value AND eval data, we wire it through.
        """
        self.feature_cols = list(feature_cols)

        X = panel[feature_cols].values
        y = panel[label_col].values
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(group_sizes)
        # XGBoost 3.x ranking: weights are per-group (one per query), not per-row.
        # Aggregate row-level sample weights to group means. Concurrency weight
        # is already constant within a date; age weight averages across tickers
        # per date, preserving "this date has young listings" signal.
        if weight_col and weight_col in panel.columns:
            w_rows = panel[weight_col].values.astype(float)
            group_weights = np.empty(len(group_sizes), dtype=float)
            offset = 0
            for gi, gs in enumerate(group_sizes):
                # Round-3 audit (#R3-11): guard against gs==0 — pre-fix
                # the .mean() of an empty slice produced NaN with a warning,
                # which XGBoost then treated as "no weight" silently.
                if gs <= 0:
                    group_weights[gi] = 1.0
                else:
                    group_weights[gi] = w_rows[offset:offset + gs].mean()
                offset += gs
            dtrain.set_weight(group_weights)

        deval: xgb.DMatrix | None = None
        if eval_panel is not None and eval_group_sizes is not None:
            Xe = eval_panel[feature_cols].values
            ye = eval_panel[label_col].values
            deval = xgb.DMatrix(Xe, label=ye)
            deval.set_group(eval_group_sizes)

        # XGBoost 3.x NDCG/MAP require integer relevance labels — our
        # Gaussianized labels are continuous, so we skip ranking metrics and
        # use per-date Spearman IC (computed in Python after training) instead.
        # Pass no evals ⇒ no metric evaluation inside xgboost.
        params = dict(self.params)

        # Monotone constraints: build XGBoost-format tuple string matching
        # feature_cols order. Only inject when at least one feature is
        # constrained — otherwise XGBoost's default (unconstrained) applies.
        if self.monotone_constraints:
            signs = [int(self.monotone_constraints.get(c, 0)) for c in feature_cols]
            if any(s != 0 for s in signs):
                params["monotone_constraints"] = "(" + ",".join(str(s) for s in signs) + ")"

        # Round-3 audit (#R3-9 #R3-10): plumb the function-arg early stop
        # through xgb.train when caller actually supplies eval data.
        train_kwargs: dict[str, Any] = {
            "num_boost_round": num_boost_round,
            "verbose_eval":    False,
        }
        if (deval is not None
                and early_stopping_rounds is not None
                and early_stopping_rounds > 0):
            train_kwargs["evals"] = [(dtrain, "train"), (deval, "eval")]
            train_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

        self.booster = xgb.train(params, dtrain, **train_kwargs)
        # When xgboost-side early stopping fires, `best_iteration` is set on
        # the booster. Otherwise final round.
        self.best_iter = getattr(self.booster, "best_iteration", num_boost_round - 1)

        result: dict[str, Any] = {"best_iter": self.best_iter}
        train_preds = self.booster.predict(dtrain)
        result["train_ic"] = _mean_ic(panel, train_preds, label_col)
        if deval is not None:
            eval_preds = self.booster.predict(deval)
            result["eval_ic"] = _mean_ic(eval_panel, eval_preds, label_col)
        return result

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        if self.booster is None:
            raise RuntimeError("PanelLTRModel.predict called before train/load")
        X = panel[self.feature_cols].values
        d = xgb.DMatrix(X)
        preds = self.booster.predict(d)
        return pd.Series(preds, index=panel.index, name="panel_score")

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        if self.booster is None:
            raise RuntimeError("PanelLTRModel.save called before train")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        raw = self.booster.save_raw(raw_format="json")
        # raw is bytearray; decode to str for JSON embedding
        raw_str = bytes(raw).decode("utf-8")

        payload = {
            "version": 1,
            "trained_date": str(date.today()),
            "feature_cols": list(self.feature_cols),
            "params": self.params,
            "best_iter": self.best_iter,
            "booster_raw_json": raw_str,
        }
        if metadata:
            payload.update({k: v for k, v in metadata.items() if k not in payload})
        path.write_text(json.dumps(payload, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "PanelLTRModel":
        path = Path(path)
        payload = json.loads(path.read_text())
        m = cls(params=payload.get("params"))
        m.feature_cols = list(payload["feature_cols"])
        m.best_iter = payload.get("best_iter")
        booster = xgb.Booster()
        # load_model accepts a bytearray; re-encode the JSON string
        booster.load_model(bytearray(payload["booster_raw_json"].encode("utf-8")))
        m.booster = booster
        return m
