"""LightGBM LambdaRank alternative to the XGBoost rank:pairwise panel model.

Trade-offs vs `PanelLTRModel` (XGBoost):

  + LambdaRank optimises NDCG@k, which matches our top-8 selection budget
    more exactly than pairwise across all pairs.
  + ~2× faster training on the same panel (verified on the M4 Pro).
  - Requires integer `label_gain` in [0, 31] — we bucketize the continuous
    Gaussianized labels into 11 gain levels before training.
  - Artifact is bigger (native LGBM `model_to_string()` vs XGBoost JSON).

JSON artifact format mirrors `PanelLTRModel`:
    { "version": 1, "kind": "panel_lgbm",
      "feature_cols": [...], "params": {...},
      "booster_str": "<lgbm text dump>", ... }

The artifact is loadable by a new `PanelLGBMScorer` that duck-types
`PanelScorer` (same `.feature_cols`, `.score(matrix)`), so inference
remains one call via `PanelScoringJob`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


try:
    import lightgbm as lgb  # type: ignore
except Exception:             # pragma: no cover - only missing in stripped envs
    lgb = None


# 2026-04-27 LGBM-V2-RETRACT: v2 hyperparameter audit (exponential
# label_gain, truncation 50, num_leaves 8, min_data_in_leaf 100,
# bagging_freq 0) made things WORSE not better — OOS IC went from
# v1's 0.0193 to v2's 0.0014, train IC dropped 0.0777 → 0.0412.
#
# Two real bugs in the audit doc's diagnosis:
# (1) `bagging_freq: 0` was claimed to mean "single bag at start" but
#     per LightGBM docs (https://lightgbm.readthedocs.io/en/latest/
#     Parameters.html) IT MEANS DISABLE BAGGING ENTIRELY. Lost variance
#     reduction.
# (2) Diagnosis assumed LGBM was overfitting (because it lost to XGB
#     it MUST be overfitting). Empirically LGBM v1 had train_IC 0.077,
#     OOS_IC 0.019 — a healthy gap suggesting MIDDLE-of-bias-variance,
#     not the high-variance overfitting v2 attempted to "fix" with
#     smaller trees + higher min_data_in_leaf. v2 starved the model
#     of capacity → train IC also collapsed.
#
# Reverted to v1 defaults below. Audit doc retained for forensics.
# Conclusion: LGBM is structurally weaker than XGB rank:pairwise on
# this panel size; no tuning room within lambdarank objective. T2-1
# absolutely rejected.

DEFAULT_PARAMS: dict[str, Any] = {
    "objective":         "lambdarank",
    "metric":            "ndcg",
    "ndcg_at":           [5, 10],
    "label_gain":        list(range(32)),   # gain[i] = i (linear; revert from v2 exp)
    "learning_rate":     0.02,
    "num_leaves":        15,
    "max_depth":         4,
    "min_data_in_leaf":  50,
    "feature_fraction":  0.7,
    "bagging_fraction":  0.7,
    "bagging_freq":      5,
    "lambda_l1":         2.0,
    "lambda_l2":         5.0,
    "lambdarank_truncation_level": 10,     # NDCG@10 — v1 setting
    "verbose":           -1,
    # 2026-05-03 raise: num_threads 4 → 10 per CLAUDE.md §5.10 (saturate hardware).
    # Original LGB-NEW-5 cap was defensive against macOS fork/OMP deadlock with
    # prior multiprocessing parents; current dispatch goes through subprocess.run
    # (clean process launch), not a forked worker, so fork-OMP deadlock is not
    # the active risk. M2 Pro has 10 cores.
    "num_threads":       10,
    # Audit fix LGB-NEW-4 (2026-04-26 round-3): explicit seeds for
    # reproducibility. bagging_fraction + feature_fraction use random
    # sampling; without seed, two runs differ. Multiple seed knobs
    # because LightGBM has separate bagging_seed / feature_fraction_seed.
    "seed":              42,
    "bagging_seed":      42,
    "feature_fraction_seed": 42,
    # Audit #12 fix (2026-04-27): `data_random_seed` only affects
    # GOSS / random-forest sampling; lambdarank with gbdt + bagging
    # ignores it. Removed to avoid the false impression of an extra
    # determinism control. The above three seeds cover the actual paths.
    "deterministic":     True,
}


def _bucketize_labels(
    y: np.ndarray, n_buckets: int = 11,
    group_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Map continuous labels to integer gains [0 … n_buckets-1] via PER-GROUP rank.

    Audit fix LGB-NEW-1 (2026-04-26 round-3, 🔴 CRITICAL): pre-fix, this
    function used GLOBAL quantile bucketing across the whole panel. For
    cross-sectional ranking via LightGBM lambdarank, labels must be
    relative WITHIN a group (date) — global bucketing destroys most of
    the ranking signal:
      - Date A with returns in [-0.05, +0.05] → all in median bucket
      - Date B with returns in [+0.10, +0.30] → all in top buckets
    Within-date rank is what lambdarank's pairwise loss compares. Pre-fix
    LightGBM was effectively trained to predict GLOBAL bucket position
    rather than per-date rank — explains why it underperformed XGBoost
    so badly even with proper weights.

    Now: when `group_sizes` is provided, rank labels WITHIN each group,
    map to integer buckets per group. When not provided (legacy callers),
    fall back to global quantile bucketing with a warning.

    LambdaRank needs integer relevance; we rank-transform per-date
    labels and assign bucket = int(rank / group_size × n_buckets).
    Returns int32 array of the same length as y.
    """
    if group_sizes is None:
        # Fallback path — log warning at caller site.
        out = np.zeros(len(y), dtype=np.int32)
        if len(y) < n_buckets:
            return out
        quantiles = np.quantile(y, np.linspace(0, 1, n_buckets + 1))
        out = np.clip(np.digitize(y, quantiles[1:-1]), 0, n_buckets - 1).astype(np.int32)
        return out

    # Per-group rank-bucketing
    out = np.zeros(len(y), dtype=np.int32)
    offset = 0
    for gs in group_sizes:
        gs_int = int(gs)
        # Audit LGBM #8 fix (2026-04-27): skip degenerate (don't add to
        # offset when negative — would corrupt subsequent slices).
        if gs_int <= 0:
            continue
        slice_y = y[offset:offset + gs_int]
        # Audit LGBM #3 fix (2026-04-27): tied labels must get the SAME
        # bucket. Pre-fix `argsort.argsort` gave unique ranks even for
        # ties → trains lambdarank to enforce arbitrary tie-breaks.
        # Now: pandas Series.rank(method="dense") gives ties the same rank.
        ranks_s = pd.Series(slice_y).rank(method="dense").values - 1.0
        max_rank = float(ranks_s.max()) if len(ranks_s) > 0 else 0.0
        if max_rank > 0 and gs_int >= n_buckets:
            buckets = (ranks_s * (n_buckets - 1) / max_rank).astype(np.int32)
        else:
            # Either tiny group or all-ties: linear ranks
            buckets = ranks_s.astype(np.int32)
        out[offset:offset + gs_int] = np.clip(buckets, 0, n_buckets - 1).astype(np.int32)
        offset += gs_int
    return out


@dataclass
class PanelLGBMModel:
    """LightGBM LambdaRank panel model — mirror of PanelLTRModel."""
    params: dict[str, Any] = None   # type: ignore[assignment]
    booster: Any = None
    feature_cols: list[str] = None  # type: ignore[assignment]
    best_iter: int | None = None

    def __post_init__(self):
        if lgb is None:
            raise RuntimeError("lightgbm is not installed")
        self.params = {**DEFAULT_PARAMS, **(self.params or {})}
        # Audit LGBM #11 fix (2026-04-27): validate objective is a
        # lambdarank-family loss. Pre-fix any objective string was
        # accepted; passing "regression" + bucketized integer labels
        # would learn nonsense. Audit 2nd-round #11 tightening: use
        # explicit whitelist (was startswith("lambda") which would
        # accept typos like "lambdarank_truncation").
        ALLOWED_OBJECTIVES = {"lambdarank", "rank_xendcg"}
        obj = str(self.params.get("objective", "lambdarank")).lower()
        if obj not in ALLOWED_OBJECTIVES:
            raise ValueError(
                f"PanelLGBMModel: objective={obj!r} not in allowed set "
                f"{ALLOWED_OBJECTIVES}. Bucketized integer labels assume "
                f"listwise ranking semantics."
            )
        self.feature_cols = list(self.feature_cols or [])

    # ── Training ───────────────────────────────────────────────────────────
    def train(
        self,
        panel: pd.DataFrame,
        group_sizes: np.ndarray,
        feature_cols: list[str],
        label_col: str = "label",
        weight_col: str | None = "weight",
        num_boost_round: int = 300,
        eval_panel: pd.DataFrame | None = None,
        eval_group_sizes: np.ndarray | None = None,
        early_stopping_rounds: int | None = None,
    ) -> dict:
        """Audit fix LGB-EVAL (2026-04-26 round-3): wire eval_panel +
        early_stopping_rounds so the FinalFitTask path (which now
        provides them per the X1+X2 fix) doesn't TypeError. LightGBM's
        lambdarank handles continuous labels via _bucketize_labels →
        integer relevance, so NDCG eval works (unlike XGBoost ranking).
        """
        self.feature_cols = list(feature_cols)
        # Audit LGBM #7 fix (2026-04-27): validate feature_cols ⊆ panel.columns
        # at train time. Pre-fix, missing column raised cryptic KeyError
        # deep inside pandas. Mirror predict()'s validation.
        missing_train_cols = [c for c in feature_cols if c not in panel.columns]
        if missing_train_cols:
            raise ValueError(
                f"PanelLGBMModel.train: panel missing required feature "
                f"columns: {missing_train_cols[:5]}"
                f"{'…' if len(missing_train_cols) > 5 else ''} "
                f"(of {len(feature_cols)} expected)."
            )
        # Audit fix LGB-NEW-2 (2026-04-26 round-3): validate group_sizes
        # match panel length. Same as XGBoost X15.
        gs_sum = int(np.sum(group_sizes))
        if gs_sum != len(panel):
            raise ValueError(
                f"PanelLGBMModel.train: sum(group_sizes)={gs_sum} != "
                f"len(panel)={len(panel)}."
            )
        # Audit LGBM #4 fix (2026-04-27): unify train/predict dtype to
        # float32. Pre-fix train used default (float64), predict used
        # float32 → precision drift on tiny splits.
        X = panel[feature_cols].to_numpy(dtype=np.float32)
        y_raw = panel[label_col].values.astype(float)
        # Audit fix LGB-NEW-1 (CRITICAL): pass group_sizes for PER-DATE
        # rank bucketing. Pre-fix global bucketing destroyed signal.
        y = _bucketize_labels(y_raw, n_buckets=11, group_sizes=group_sizes)

        # LightGBM ranking takes PER-ROW weights (length = n_rows),
        # unlike XGBoost 3.x which takes per-group. To keep parity with
        # the panel's weighting semantics (where weights are meaningful
        # at the date-group level), we average each group's weight and
        # broadcast back to per-row so LightGBM sees a constant
        # per-group weight expressed at row granularity. Empty/missing
        # weight column → no weights supplied at all.
        row_weights = None
        if weight_col and weight_col in panel.columns:
            w_rows = panel[weight_col].values.astype(float)
            # Audit LGBM #1 fix (2026-04-27): NaN guard. Pre-fix, NaN
            # weights from concurrency × age product (early dates) would
            # silently propagate to LightGBM with undefined behavior.
            n_nan = int(np.isnan(w_rows).sum())
            if n_nan > 0:
                log.warning(
                    "PanelLGBMModel.train: replacing %d NaN weights with 1.0 "
                    "(typically early-date age-weight warmup)", n_nan,
                )
                w_rows = np.nan_to_num(w_rows, nan=1.0)
            row_weights = np.empty(len(w_rows), dtype=float)
            off = 0
            for gs in group_sizes:
                # Audit LGBM #6 fix (2026-04-27): skip degenerate groups
                # to avoid empty-slice .mean() RuntimeWarning.
                gs_int = int(gs)
                if gs_int <= 0:
                    continue
                grp_mean = float(w_rows[off:off + gs_int].mean())
                row_weights[off:off + gs_int] = grp_mean
                off += gs_int
            assert len(row_weights) == len(X), (
                f"row_weights len {len(row_weights)} != X rows {len(X)}"
            )
            # Audit fix LGB-WEIGHT-NORM (2026-04-25): production panel
            # weights are concurrency × age products — total mass ~10^-4.
            # XGBoost auto-normalizes per-group; LightGBM uses raw weights
            # in `sum_of_gradients × weight`. With weight ≈ 0.0003 the
            # effective gradient signal collapses → lambdarank converges
            # at iter 1, train_ic=NaN, panel-ltr.json saved with 5.5kB
            # of nothing (observed in T2-1 retrain, 2026-04-25 20:33).
            # Fix: rescale so mean(row_weights) = 1.0 — preserves the
            # RELATIVE per-group weight ratio that the panel encodes,
            # while keeping LightGBM's gradient signal normalised.
            mean_w = float(row_weights.mean())
            if mean_w > 0:
                row_weights = row_weights / mean_w

        dtrain = lgb.Dataset(X, label=y, group=group_sizes, weight=row_weights)

        # Audit fix LGB-EVAL (2026-04-26 round-3): wire eval set + early
        # stopping when caller provides eval data.
        valid_sets = [dtrain]
        valid_names = ["train"]
        callbacks: list = []
        if eval_panel is not None and eval_group_sizes is not None:
            # Audit LGBM #4 fix: same float32 cast as train.
            Xe = eval_panel[feature_cols].to_numpy(dtype=np.float32)
            ye_raw = eval_panel[label_col].values.astype(float)
            # Audit fix LGB-NEW-1: per-date rank bucketing on eval too.
            ye = _bucketize_labels(ye_raw, n_buckets=11,
                                   group_sizes=eval_group_sizes)
            deval = lgb.Dataset(Xe, label=ye, group=eval_group_sizes,
                                reference=dtrain)
            valid_sets.append(deval)
            valid_names.append("eval")
            if early_stopping_rounds is not None and early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(int(early_stopping_rounds), verbose=False)
                )

        self.booster = lgb.train(
            self.params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks or None,
        )
        # Audit LGBM #2 fix (2026-04-27): prefer best_iteration when
        # early stopping fired. Pre-fix `current_iteration()` returned
        # the LAST round (= best + early_stopping_patience), misreporting
        # peak.
        best_iter_attr = getattr(self.booster, "best_iteration", 0) or 0
        if best_iter_attr > 0:
            self.best_iter = int(best_iter_attr)
        else:
            self.best_iter = int(self.booster.current_iteration())

        # Per-date Spearman IC against the original (non-bucketed) label.
        # Audit LGBM #9 fix (2026-04-27): explicit num_iteration so older
        # LightGBM versions also use the best round.
        preds = self.booster.predict(X, num_iteration=self.best_iter)
        ic = _per_date_ic(panel, preds, label_col, date_col="date")
        return {"best_iter": self.best_iter, "train_ic": ic}

    # ── Prediction ─────────────────────────────────────────────────────────
    def predict(self, panel: pd.DataFrame) -> pd.Series:
        if self.booster is None:
            raise RuntimeError("PanelLGBMModel.predict called before train/load")
        # Audit fix LGB-NEW-3 (2026-04-26 round-3): validate column
        # presence (mirror XGBoost X5). Pre-fix, missing column raised
        # cryptic KeyError deep in pandas.
        missing = [c for c in self.feature_cols if c not in panel.columns]
        if missing:
            raise ValueError(
                f"PanelLGBMModel.predict: panel missing required feature "
                f"columns: {missing[:5]}{'…' if len(missing) > 5 else ''} "
                f"(model trained on {len(self.feature_cols)} features)."
            )
        X = panel[self.feature_cols].to_numpy(dtype=np.float32)
        preds = self.booster.predict(X)
        return pd.Series(preds, index=panel.index, name="panel_score")

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        if self.booster is None:
            raise RuntimeError("PanelLGBMModel.save called before train")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version":      1,
            "kind":         "panel_lgbm",
            "trained_date": str(date.today()),
            "feature_cols": list(self.feature_cols),
            "params":       self.params,
            "best_iter":    self.best_iter,
            "booster_str":  self.booster.model_to_string(),
        }
        if metadata:
            payload.update({k: v for k, v in metadata.items() if k not in payload})
        path.write_text(json.dumps(payload, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "PanelLGBMModel":
        path = Path(path)
        payload = json.loads(path.read_text())
        if payload.get("kind") != "panel_lgbm":
            raise ValueError(
                f"PanelLGBMModel.load: artifact at {path} is not a panel_lgbm "
                f"(kind={payload.get('kind')!r})",
            )
        m = cls(params=payload.get("params"), feature_cols=payload["feature_cols"])
        m.best_iter = payload.get("best_iter")
        m.booster   = lgb.Booster(model_str=payload["booster_str"])
        return m


def _per_date_ic(panel: pd.DataFrame, preds: np.ndarray,
                 label_col: str, date_col: str = "date") -> float:
    # Audit fix LGB-NEW-6 (2026-04-26 round-3): defensive column guard.
    for col in (date_col, label_col):
        if col not in panel.columns:
            raise KeyError(
                f"_per_date_ic: panel missing required column '{col}'. "
                f"Available: {list(panel.columns)[:10]}"
                f"{'…' if len(panel.columns) > 10 else ''}"
            )
    df = pd.DataFrame({
        "date": panel[date_col].values,
        "p":    preds,
        "y":    panel[label_col].values,
    })
    ics: list[float] = []
    for _, g in df.groupby("date", sort=False):
        y = g["y"].values
        p = g["p"].values
        # Audit fix LGB-NEW-7 (2026-04-26 round-3): np.allclose for float
        # equality (matches transformer #67).
        if (len(y) < 2
                or np.allclose(y, y[0], rtol=0, atol=1e-12)
                or np.allclose(p, p[0], rtol=0, atol=1e-12)):
            continue
        rho, _ = spearmanr(p, y)
        # Audit fix LGB-NEW-8 (2026-04-26 round-3): use np.isnan idiom.
        if rho is not None and not np.isnan(rho):
            ics.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan")


# ── Scorer that duck-types PanelScorer for the inference side ────────────────

class PanelLGBMScorer:
    """Thin LightGBM loader mirroring `PanelScorer`'s interface."""

    def __init__(self, booster, feature_cols: list[str],
                 metadata: dict | None = None):
        self.booster = booster
        self.feature_cols = list(feature_cols)
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: str | Path) -> "PanelLGBMScorer":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed")
        path = Path(path)
        payload = json.loads(path.read_text())
        if payload.get("kind") != "panel_lgbm":
            raise ValueError(f"PanelLGBMScorer.load: not a panel_lgbm artifact at {path}")
        booster = lgb.Booster(model_str=payload["booster_str"])
        meta = {k: v for k, v in payload.items() if k != "booster_str"}
        return cls(booster, list(payload["feature_cols"]), meta)

    def score(self, feature_matrix: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.feature_cols if c not in feature_matrix.columns]
        if missing:
            raise KeyError(f"PanelLGBMScorer.score: missing cols {missing}")
        X = feature_matrix[self.feature_cols].values
        preds = self.booster.predict(X)
        return pd.Series(preds, index=feature_matrix.index, name="panel_score")


__all__ = [
    "DEFAULT_PARAMS",
    "PanelLGBMModel",
    "PanelLGBMScorer",
]
