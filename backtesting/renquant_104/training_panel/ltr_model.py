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
    # Audit fix X6 (2026-04-26): cap nthread at a reasonable bound to
    # avoid fork/OMP deadlock when training is launched from a process
    # that previously used multiprocessing (e.g. TickerPanelFeatureJob).
    # Pre-fix `nthread=-1` used all cores; on macOS with prior fork
    # context this could deadlock. Cap at 4 (≥ 2× speedup over single-
    # thread, well below deadlock threshold). Override via xgb_params.nthread.
    "nthread": 4,
    "verbosity": 0,
    # Audit fix X12 (2026-04-26 batch-3): explicit RNG seed for
    # reproducibility. Pre-fix, subsample=0.8 and colsample_bytree=0.7
    # used random sampling without a fixed seed → two consecutive
    # train() calls produced DIFFERENT models with measurably different
    # OOS IC (~±0.005). Setting seed=42 makes training bit-reproducible.
    "seed": 42,
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
        #
        # Audit fix X13 (2026-04-26 batch-3): pre-fix, the dict-to-list
        # mapping was POSITIONAL — silently produced wrong sign mappings
        # if `feature_cols` order changed between the constraint dict's
        # author and the training call. Now: validate that every dict
        # key maps to a known feature_col + log resolved signs for
        # transparency. Unknown keys → loud error.
        if self.monotone_constraints:
            unknown_keys = [k for k in self.monotone_constraints
                            if k not in feature_cols]
            if unknown_keys:
                raise ValueError(
                    f"PanelLTRModel: monotone_constraints references "
                    f"feature(s) not in training feature_cols: {unknown_keys}. "
                    f"Either remove from constraints or add to features."
                )
            signs = [int(self.monotone_constraints.get(c, 0)) for c in feature_cols]
            if any(s != 0 for s in signs):
                params["monotone_constraints"] = "(" + ",".join(str(s) for s in signs) + ")"
                resolved = {c: s for c, s in zip(feature_cols, signs) if s != 0}
                import logging  # noqa: PLC0415
                logging.getLogger("panel.ltr").info(
                    "monotone_constraints resolved: %s", resolved,
                )

        # Audit fix X1+X2 (2026-04-26, completed): Python-level early
        # stopping. XGBoost 3.x ranking objective auto-enables NDCG which
        # requires INTEGER labels — our Gaussianized labels are
        # continuous → NDCG raises `label_is_integer` (un-catchable C++
        # crash). Workaround: train in chunks, compute spearman IC per
        # chunk, manually break on patience. xgboost's incremental
        # training via `xgb_model=current_booster` lets us continue
        # training without losing state.
        if (deval is not None
                and early_stopping_rounds is not None
                and early_stopping_rounds > 0
                and eval_panel is not None):
            # Audit fix X14 (2026-04-26 round-3): chunk_size MUST be
            # smaller than early_stopping_rounds so patience can absorb
            # multiple bad chunks. Pre-fix, chunk == early_stop → patience
            # of effectively 1 chunk → broke far too aggressively. New:
            # chunk = max(5, early_stop // 4) so we get up to 4 chances
            # of no-improvement before stopping.
            chunk_size = max(5, int(early_stopping_rounds) // 4)
            # Audit fix X18 (2026-04-26 round-3): tighten improvement
            # threshold from 1e-4 (noise level — CPCV std ≈ 0.027) to
            # 1e-3 (one σ ÷ 27 ≈ real signal). Avoids spurious "best"
            # updates from numerical noise.
            min_delta_ic  = 1e-3
            best_ic       = float("-inf")
            best_booster  = None
            best_iter     = 0
            patience_left = int(early_stopping_rounds)
            cur_booster   = None
            rounds_done   = 0
            import logging  # noqa: PLC0415
            _ltr_log = logging.getLogger("panel.ltr")
            while rounds_done < num_boost_round:
                this_chunk = min(chunk_size, num_boost_round - rounds_done)
                cur_booster = xgb.train(
                    params, dtrain,
                    num_boost_round = this_chunk,
                    xgb_model       = cur_booster,
                    verbose_eval    = False,
                )
                rounds_done += this_chunk
                eval_preds = cur_booster.predict(deval)
                ic = _mean_ic(eval_panel, eval_preds, label_col)
                if ic > best_ic + min_delta_ic:
                    best_ic       = ic
                    best_iter     = rounds_done - 1
                    # Persist via byte serialization to immortalize
                    best_booster  = cur_booster.save_raw(raw_format="ubj")
                    patience_left = int(early_stopping_rounds)
                    _ltr_log.info(
                        "early-stop: rounds=%d eval_ic=%+.4f (new best)",
                        rounds_done, ic,
                    )
                else:
                    patience_left -= this_chunk
                    _ltr_log.debug(
                        "early-stop: rounds=%d eval_ic=%+.4f patience_left=%d",
                        rounds_done, ic, patience_left,
                    )
                if patience_left <= 0:
                    _ltr_log.info(
                        "early-stop fired at rounds=%d, best_iter=%d, best_ic=%+.4f",
                        rounds_done, best_iter, best_ic,
                    )
                    break
            # Restore best
            if best_booster is not None:
                self.booster = xgb.Booster()
                self.booster.load_model(bytearray(best_booster))
            else:
                self.booster = cur_booster
            self.best_iter = best_iter
        else:
            train_kwargs: dict[str, Any] = {
                "num_boost_round": num_boost_round,
                "verbose_eval":    False,
            }
            self.booster = xgb.train(params, dtrain, **train_kwargs)
            # When xgboost-side early stopping fires, `best_iteration` is set on
            # the booster. Otherwise final round.
            self.best_iter = getattr(self.booster, "best_iteration", num_boost_round - 1)

        result: dict[str, Any] = {"best_iter": self.best_iter}
        train_preds = self.booster.predict(dtrain)
        result["train_ic"] = _mean_ic(panel, train_preds, label_col)

        # Audit fix X10 (2026-04-26 batch-3): expose feature importances
        # in train metadata for downstream debugging. `gain` = average
        # loss-reduction contribution per feature; the most actionable
        # importance type for ranking models. Names: f0, f1, ... → map
        # back to feature_cols.
        try:
            scores = self.booster.get_score(importance_type="gain")
            named: dict[str, float] = {}
            for k, v in scores.items():
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if 0 <= idx < len(self.feature_cols):
                        named[self.feature_cols[idx]] = float(v)
            result["feature_importances"] = named
        except Exception:
            result["feature_importances"] = {}

        if deval is not None:
            eval_preds = self.booster.predict(deval)
            result["eval_ic"] = _mean_ic(eval_panel, eval_preds, label_col)
        return result

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        if self.booster is None:
            raise RuntimeError("PanelLTRModel.predict called before train/load")

        # Audit fix X5 (2026-04-26): validate column order — pre-fix
        # numpy positional indexing silently used WRONG features when
        # panel had different column order than train. Now: explicit
        # feature_cols indexing + missing-column check.
        missing = [c for c in self.feature_cols if c not in panel.columns]
        if missing:
            raise ValueError(
                f"PanelLTRModel.predict: panel missing required feature "
                f"columns: {missing[:5]}{'…' if len(missing) > 5 else ''} "
                f"(model trained on {len(self.feature_cols)} features)."
            )

        # to_numpy with explicit column selection guarantees order matches
        # the training feature_cols (vs `.values` which uses panel's
        # current column order).
        X = panel[self.feature_cols].to_numpy(dtype=np.float32)
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
