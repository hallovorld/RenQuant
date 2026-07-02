"""Cross-sectional panel scorer — loads the Stage-1 artifact and predicts.

The training side (`training_panel/`) writes a JSON artifact with:

    { version, feature_cols, params, booster_raw_json, oos_mean_ic, ... }

`PanelScorer.load(path)` rebuilds an XGBoost booster from the embedded
JSON and exposes a single entry point::

    scores: dict[ticker, float] = scorer.score(feature_matrix)

`feature_matrix` is a DataFrame indexed by ticker with one column per
feature name in `feature_cols`. The returned scores preserve the input
index order.

Two gate helpers are provided for selection use:

    top_n_by_score(scores, n)      — largest-N by score
    probability_gate(scores, thr)  — keep score ≥ thr
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xgboost as xgb


def artifact_sha256(path: str | Path) -> str:
    """Full-file artifact hash for tamper/audit checks.

    Do not use this as the scorer/calibrator pairing identity: acceptance
    tools append mutable metadata such as ``wf_gate_metadata`` after training,
    which changes the file bytes without changing the model.
    """
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


_MUTABLE_ARTIFACT_KEYS = {
    "metadata",
    "wf_gate_metadata",
    "artifact_path",
    "artifact_sha256",
    "artifact_fingerprint",
    "model_content_fingerprint",
    "config_fingerprint",
    "config_fingerprint_fields",
    "trained_date",
    "training_notes",
    "label",
    "label_col",
    "lookahead_days",
    "panel_shape",
    "n_train_rows",
    "training_train_ic",
    "val_mean_ic",
    "val_median_ic",
    "test_mean_ic",
    "test_median_ic",
    "oos_mean_ic",
    # P-PANEL-CONTRACT acceptance fields (2026-05-30 Bug D fix).
    # These are pure post-training metadata: CV bookkeeping, OOS evidence,
    # promotion gates, sentiment-contract markers, audit IDs. Stamping any
    # of these changes the JSON bytes but does NOT change the model's
    # predictions — must be excluded from model_content_fingerprint so the
    # calibrator binding survives metadata edits (previously caused 3
    # calibrator rebinds in one day).
    "cv_method",
    "cv_embargo_days",
    "cv_folds",
    "cv_n_splits",
    "oos_std_ic",
    "oos_per_fold_ic",
    "eval_ic",
    "train_run_id",
    "sentiment_runtime_gate_contract",
    "sentiment_runtime_gate_trained",
    "promotion_status",
    "promotion_gating_reason",
    "version",  # artifact-format version, not a model parameter
    "side_label",
}

_PREDICTIVE_CONTENT_HINTS = {
    "booster_raw_json",
    "feature_cols",
    "feature_columns",
    "feature_means",
    "feature_stds",
    "feature_norm_kind",
    "feature_norm_kinds",
    "feature_raw_clip_low",
    "feature_raw_clip_high",
    "coef",
    "intercept",
    "clip_sigma",
    "state_dict",
    "config_dict",
    "model_bytes",
    "model_bytes_b64",
}


def model_content_sha256(payload: dict[str, Any]) -> str:
    """Stable scorer identity over immutable model content.

    Panel artifacts are JSON files that later acquire operational metadata
    (WF gate results, file hashes, paths). Calibrators are fitted to the model
    score distribution, not to that mutable metadata. Hash only the content
    that changes the scorer's predictions.
    """
    content = {
        k: v for k, v in payload.items()
        if k not in _MUTABLE_ARTIFACT_KEYS
    }
    if not any(k in content for k in _PREDICTIVE_CONTENT_HINTS):
        raise ValueError("payload has no recognizable scorer prediction content")
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def model_content_sha256_from_path(path: str | Path) -> str:
    """Return model-content hash for JSON artifacts, full hash otherwise."""
    p = Path(path)
    try:
        payload = json.loads(p.read_text())
    except Exception:
        return artifact_sha256(p)
    if not isinstance(payload, dict):
        return artifact_sha256(p)
    try:
        return model_content_sha256(payload)
    except ValueError:
        return artifact_sha256(p)


# 2026-07-02 (#426 round 8, Codex CHANGES_REQUESTED): the canonical
# provenance-schema/recipe taxonomy shadow_scoring.py's admission gate
# validates against. Defined HERE — not in shadow_scoring.py — so every
# artifact-producing path (this module's own XGB stamping below,
# hf_patchtst_scorer.py, scripts/train_production_model.py::build_artifact)
# imports the SAME canonical source and stamps it once, at/near artifact
# construction, rather than a downstream consumer re-INFERRING a recipe from
# whichever cutoff fields happen to be present on any given artifact (round
# 7's gap: inference re-evaluates the CURRENT code's understanding against
# an artifact with no way to prove which stamping contract actually built
# it — a legacy or hand-edited artifact with a matching field combination
# was silently admitted under a contract it was never validated against).
#
# `resolve_recipe_id` is the single function every stamping site calls;
# `RECIPE_REQUIRED_AXES` is the single map every stamping site AND every
# consumer reads. A consumer must additionally re-verify (not merely trust)
# that the artifact still actually carries the fields its own stamped
# `recipe_id` claims — see shadow_scoring.py `_compute_admission`.
PROVENANCE_SCHEMA_VERSION = "v1"
RECIPE_REQUIRED_AXES: dict[str, frozenset] = {
    # hf_patchtst / patchtst (hf_patchtst_scorer.py stamps only this field),
    # or an XGB walk-forward artifact carrying only the walk-forward axis.
    "walkforward_only_v1": frozenset({"effective_train_cutoff_date"}),
    # XGB full-history (train_production_model.py::build_artifact stamps
    # this on BOTH training paths).
    "full_history_only_v1": frozenset({"label_observation_cutoff"}),
    # XGB walk-forward (train_production_model.py::build_artifact stamps
    # BOTH fields on the walk-forward path — the most complete provenance).
    "walkforward_dual_axis_v1": frozenset(
        {"label_observation_cutoff", "effective_train_cutoff_date"}
    ),
}


def resolve_recipe_id(present_confirmed_fields: Iterable[str]) -> str | None:
    """The canonical `recipe_id` for an EXACT set of confirmed provenance
    axes under `PROVENANCE_SCHEMA_VERSION`, or `None` if the combination is
    not a recognized, validated contract (fail-closed — callers must treat
    `None` as UNKNOWN, never guess or admit under an unrecognized shape)."""
    fs = frozenset(present_confirmed_fields)
    for recipe_id, required in RECIPE_REQUIRED_AXES.items():
        if fs == required:
            return recipe_id
    return None


def stamp_provenance_schema(artifact: dict, present_confirmed_fields: Iterable[str]) -> dict:
    """Mutate `artifact` in place, adding `provenance_schema_version` /
    `recipe_id` / `required_axis_fields` when `present_confirmed_fields`
    resolves to a recognized recipe — the single stamping call every
    artifact-producing path should use so the taxonomy is applied
    identically everywhere. No-ops (stamps nothing) when the combination is
    unrecognized; callers must not fabricate a recipe_id for an unrecognized
    shape. Returns `artifact` for chaining."""
    recipe_id = resolve_recipe_id(present_confirmed_fields)
    if recipe_id is not None:
        artifact["provenance_schema_version"] = PROVENANCE_SCHEMA_VERSION
        artifact["recipe_id"] = recipe_id
        artifact["required_axis_fields"] = sorted(RECIPE_REQUIRED_AXES[recipe_id])
    return artifact


def stamp_artifact_metadata(
    metadata: dict | None,
    path: str | Path,
    payload: dict[str, Any] | None = None,
) -> dict:
    """Return metadata with path + fingerprint fields for runtime contracts."""
    meta = dict(metadata or {})
    nested = meta.get("metadata")
    if isinstance(nested, dict):
        for key, value in nested.items():
            meta.setdefault(key, value)
    sha = artifact_sha256(path)
    try:
        content_sha = (
            model_content_sha256(payload)
            if isinstance(payload, dict)
            else model_content_sha256_from_path(path)
        )
    except ValueError:
        content_sha = sha
    meta.setdefault("artifact_path", str(Path(path)))
    meta.setdefault("artifact_sha256", sha)
    meta.setdefault("artifact_fingerprint", sha)
    meta.setdefault("model_content_fingerprint", content_sha)
    return meta


class PanelScorer:
    """Thin loader around a saved panel-LTR artifact."""

    def __init__(self, booster: xgb.Booster, feature_cols: list[str],
                 metadata: dict | None = None):
        self.booster = booster
        self.feature_cols = list(feature_cols)
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: str | Path):
        """Load a panel artifact — dispatches on `kind` (or file extension).

        Returns a scorer whose class matches the artifact:
          - `kind: panel_lgbm`       → PanelLGBMScorer       (LightGBM)
          - `kind: panel_transformer`→ TransformerPanelScorer (PyTorch .pt)
          - otherwise (legacy)       → PanelScorer            (XGBoost)

        All three expose the same `.feature_cols` attr + `.score(matrix)`
        method so callers (`PanelScoringJob`, tests, scripts) can treat
        them interchangeably.

        A `.pt` path is also accepted — we forward to the transformer
        loader which resolves the paired `.json` sidecar automatically.
        """
        path = Path(path)
        # 2026-05-04 audit Issue 28: explicit FileNotFoundError so the
        # caller (LoadScorerTask) gets a typed error and a useful path
        # in the message — pre-fix, json.loads on a missing file raised
        # FileNotFoundError from path.read_text() with the same path but
        # transformer .pt branch took it before the JSON path could
        # produce any error context.
        if not path.exists():
            raise FileNotFoundError(
                f"PanelScorer.load: artifact not found: {path} — "
                f"check ranking.panel_scoring.artifact_path config + "
                f"that the snapshot dir copied the side artifact "
                f"(2026-05-04 snapshot side-config fix)."
            )
        if path.suffix == ".pt":
            # 2026-05-20 fix: `.pt` no longer auto-routes to legacy custom
            # TransformerPanelScorer. HF PatchTST (scripts/patchtst_hf.py
            # --save-model, registered as kind=hf_patchtst in
            # model_registry.py 2026-05-18) saves a checkpoint with
            # `config_dict`+`feature_cols` keys and NO sidecar JSON. Pre-fix,
            # SimAdapter trying to load such an artifact failed with
            # "PanelTransformerModel.load: sidecar JSON not found" — split-
            # brain between model_registry (HF-aware) and PanelScorer.load
            # (HF-blind) per §1c violation.
            # Detect HF format via marker keys without loading state_dict.
            ckpt = None
            try:
                import torch  # noqa: PLC0415
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                pass  # fall through to legacy if checkpoint peek failed
            if isinstance(ckpt, dict) and "config_dict" in ckpt and "feature_cols" in ckpt:
                from kernel.panel_pipeline.hf_patchtst_scorer import HFPatchTSTPanelScorer  # noqa: PLC0415
                return HFPatchTSTPanelScorer.load(path)
            from kernel.panel_pipeline.transformer_scorer import TransformerPanelScorer  # noqa: PLC0415
            return TransformerPanelScorer.load(path)
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"PanelScorer.load: artifact at {path} is not valid JSON: {exc}"
            ) from exc
        kind = payload.get("kind")
        if kind == "panel_transformer":
            # JSON sidecar was passed; transformer loader will find the .pt.
            from kernel.panel_pipeline.transformer_scorer import TransformerPanelScorer  # noqa: PLC0415
            return TransformerPanelScorer.load(path)
        if kind == "panel_lgbm":
            # Delay import to keep lightgbm optional.
            from training_panel.lgbm_ltr import PanelLGBMScorer  # noqa: PLC0415
            return PanelLGBMScorer.load(path)
        if kind == "panel_linear":
            # Phase 1 (2026-05-06): alpha158 + sklearn LinearRegression.
            # +29 pts walk-forward alpha vs SPY @ 10bp friction.
            from training_panel.linear_ltr import PanelLinearScorer  # noqa: PLC0415
            return PanelLinearScorer.load(path)
        # Default: XGBoost rank:pairwise artifact
        booster = xgb.Booster()
        booster.load_model(bytearray(payload["booster_raw_json"].encode("utf-8")))
        meta = stamp_artifact_metadata(
            {k: v for k, v in payload.items() if k != "booster_raw_json"},
            path,
            payload=payload,
        )
        return cls(
            booster=booster,
            feature_cols=list(payload["feature_cols"]),
            metadata=meta,
        )

    def score(self, feature_matrix: pd.DataFrame, ctx: Any = None) -> pd.Series:
        """Predict panel scores for rows of `feature_matrix`.

        Returns a Series indexed like `feature_matrix.index` (typically
        ticker symbols). Missing feature columns raise KeyError — the
        caller is responsible for aligning the matrix to the artifact's
        `feature_cols`.

        ``ctx`` is accepted-but-ignored at this layer so the public
        scoring contract is uniform across PanelScorer + ensemble
        variants (e.g. ``RegimeEnsemblePanelScorer`` reads regime fields
        from ctx). Callers that have an ``InferenceContext`` should pass
        it through; callers without one (back-compat: shadow scoring,
        smoke tests, ``compute_panel_scores``) keep the single-arg form
        working. Pinned by Track C wiring fix (2026-06-02) — see
        ``RegimeEnsemblePanelScorer.score`` for the routing logic.

        Per CLAUDE.md §5.3 BUG #6 invariant: soft_check_input runs before
        predict, soft_check_score_series runs after. Both LOG warnings on
        degeneracy (constant features, collapsed scores) so silent
        feature-corruption bugs surface immediately.
        """
        del ctx  # accepted for signature uniformity; PanelScorer is regime-blind
        missing = [c for c in self.feature_cols if c not in feature_matrix.columns]
        if missing:
            raise KeyError(
                f"PanelScorer.score: feature matrix missing columns: {missing}",
            )
        # ── Input contract ──
        from training_panel.model_contract import (  # noqa: PLC0415
            soft_check_input, soft_check_score_series,
        )
        soft_check_input(feature_matrix, self.feature_cols, head_name="PanelScorer")

        X = feature_matrix[self.feature_cols].values
        d = xgb.DMatrix(X)
        preds = self.booster.predict(d)
        out = pd.Series(preds, index=feature_matrix.index, name="panel_score")
        # ── Output contract ──
        soft_check_score_series(out, model_name="PanelScorer")
        return out


def compute_panel_scores(
    artifact_path: str | Path,
    feature_matrix: pd.DataFrame,
) -> pd.Series:
    """One-shot helper: load artifact → score → return per-ticker scores."""
    scorer = PanelScorer.load(artifact_path)
    return scorer.score(feature_matrix)


def top_n_by_score(scores: pd.Series, n: int) -> list[str]:
    """Return the top-`n` labels (indices) of `scores` by value, descending.

    NaN scores are excluded. Ties broken by input order (stable sort).
    """
    if n <= 0:
        return []
    s = scores.dropna()
    order = s.sort_values(ascending=False, kind="mergesort")
    return list(order.index[:n])


def probability_gate(scores: pd.Series, threshold: float) -> list[str]:
    """Return labels whose score is >= `threshold`, sorted high → low.

    NaN scores are excluded.
    """
    s = scores.dropna()
    passed = s[s >= threshold]
    return list(passed.sort_values(ascending=False, kind="mergesort").index)
