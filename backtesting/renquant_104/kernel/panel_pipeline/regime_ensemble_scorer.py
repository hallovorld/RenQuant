"""Regime-specialist ensemble panel scorer (Track C, 2026-06-02).

Loads up to 4 per-regime specialist panel scorers and dispatches at score
time based on the regime detector's posterior+final regime read from the
InferenceContext. Falls back transparently to the global panel scorer
when a specialist is missing or when ``specialists`` is not configured.

Per CLAUDE.md §1 PRIME DIRECTIVE this is the structural fix for
pooled-mean signal averaging in BULL_CALM: each specialist optimizes
for its regime's return distribution rather than the gradient-weighted
average across regimes.

Config schema (under ``ranking.panel_scoring``)::

    "specialists": {
        "BULL_CALM":     "artifacts/prod/panel-ltr.alpha158_fund.bull_calm.json",
        "BEAR":          "artifacts/prod/panel-ltr.alpha158_fund.bear.json",
        "BULL_VOLATILE": "artifacts/prod/panel-ltr.alpha158_fund.bull_volatile.json",
        "CHOPPY":        "artifacts/prod/panel-ltr.alpha158_fund.choppy.json"
    },
    "specialist_confidence_threshold": 0.8,   # ≥ threshold → hard-pick
    "specialist_blend_top_k": 2               # transition → blend top-k posteriors

Back-compat: when ``specialists`` is absent the scorer load path simply
returns the legacy ``PanelScorer.load(global_path)``; existing artifacts
keep working without any consumer change.

Reference:
  - 2026-06-02 BULL_CALM signal recovery plan, Track C
  - 2026-06-02 BULL_CALM no-signal diagnostic
  - CLAUDE.md §1 PRIME DIRECTIVE, §5.1 Task/Job/Pipeline, §7.5 single source
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .panel_scorer import PanelScorer

log = logging.getLogger("kernel.panel_pipeline.regime_ensemble_scorer")

DEFAULT_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_BLEND_TOP_K = 2


class StaleSpecialistArtifact(RuntimeError):
    """Raised when a specialist artifact's recipe fingerprint disagrees with the global."""


class RegimeEnsemblePanelScorer:
    """Per-regime specialist ensemble over PanelScorer artifacts.

    Public surface mirrors ``PanelScorer`` (``feature_cols`` + ``metadata``)
    but adds a ``score(ctx, features_df)`` method that reads context
    fields ``final_regime`` / ``regime_confidence`` / ``regime_posterior``
    and dispatches:

      confidence >= threshold       → specialists[final_regime].score(X)
      confidence <  threshold       → posterior-weighted blend of the
                                      top-k regimes that HAVE specialists
      specialist missing            → fall back to global scorer

    The fallback semantics keep operations safe when only a subset of
    specialists has been trained (e.g. BULL_CALM-only Track A pilot).
    """

    def __init__(
        self,
        global_scorer: PanelScorer,
        specialists: dict[str, PanelScorer],
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        blend_top_k: int = DEFAULT_BLEND_TOP_K,
        metadata: dict | None = None,
    ) -> None:
        if global_scorer is None:
            raise ValueError("RegimeEnsemblePanelScorer requires a non-None global_scorer")
        self.global_scorer = global_scorer
        self.specialists = dict(specialists or {})
        self.confidence_threshold = float(confidence_threshold)
        self.blend_top_k = max(1, int(blend_top_k))
        # Feature col union across global + specialists; ApplyScoresTask
        # builds X via this set so every specialist's required features
        # are present at score time.
        feat_set: set[str] = set(global_scorer.feature_cols)
        for s in self.specialists.values():
            feat_set.update(s.feature_cols)
        self.feature_cols = sorted(feat_set)
        meta = dict(metadata or {})
        meta.setdefault("kind", "panel_ltr_xgboost")  # back-compat for kind-dispatching code
        meta["ensemble_kind"] = "regime_specialist"
        meta["specialist_regimes"] = sorted(self.specialists.keys())
        meta["confidence_threshold"] = self.confidence_threshold
        meta["blend_top_k"] = self.blend_top_k
        # Expose the global scorer's artifact identity so downstream
        # config-consistency checks still bind against a primary artifact.
        meta.setdefault("artifact_path", global_scorer.metadata.get("artifact_path"))
        meta.setdefault("artifact_sha256", global_scorer.metadata.get("artifact_sha256"))
        meta.setdefault("model_content_fingerprint",
                        global_scorer.metadata.get("model_content_fingerprint"))
        meta.setdefault("config_fingerprint", global_scorer.metadata.get("config_fingerprint"))
        self.metadata = meta

    # ── Loader ────────────────────────────────────────────────────────────

    @classmethod
    def load_from_config(
        cls,
        panel_cfg: dict,
        strategy_dir: str | Path | None = None,
    ) -> "RegimeEnsemblePanelScorer | PanelScorer":
        """Build either an ensemble scorer or the legacy global scorer.

        Returns the legacy ``PanelScorer`` (NOT wrapped) when no
        ``specialists`` block is configured — keeps the consumer side
        byte-equivalent for back-compat.
        """
        global_path = panel_cfg.get("artifact_path")
        if not global_path:
            raise ValueError(
                "RegimeEnsemblePanelScorer.load_from_config: "
                "ranking.panel_scoring.artifact_path is required"
            )
        global_scorer = PanelScorer.load(cls._resolve(global_path, strategy_dir))

        specs_cfg = panel_cfg.get("specialists") or {}
        if not specs_cfg:
            return global_scorer  # back-compat: no ensemble wrapping
        specialists: dict[str, PanelScorer] = {}
        for regime, art_path in specs_cfg.items():
            if not art_path:
                continue
            resolved = cls._resolve(art_path, strategy_dir)
            if not resolved.exists():
                log.warning(
                    "RegimeEnsemblePanelScorer: specialist for %s missing at %s — falling back to global",
                    regime, resolved,
                )
                continue
            sp = PanelScorer.load(resolved)
            cls._validate_specialist_compatibility(global_scorer, sp, regime)
            specialists[str(regime)] = sp
        return cls(
            global_scorer=global_scorer,
            specialists=specialists,
            confidence_threshold=float(panel_cfg.get(
                "specialist_confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)),
            blend_top_k=int(panel_cfg.get(
                "specialist_blend_top_k", DEFAULT_BLEND_TOP_K)),
        )

    @staticmethod
    def _resolve(path: str | Path, strategy_dir: str | Path | None) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        if strategy_dir:
            return Path(strategy_dir) / p
        return p

    @staticmethod
    def _validate_specialist_compatibility(
        global_scorer: PanelScorer,
        specialist: PanelScorer,
        regime: str,
    ) -> None:
        """Verify specialist's feature_cols is a subset of the global's.

        Specialists may use FEWER features than the global (regime-specific
        feature engineering) but never NEW features — that would imply a
        recipe mismatch that breaks the panel-builder contract.
        """
        gset = set(global_scorer.feature_cols)
        sset = set(specialist.feature_cols)
        extra = sorted(sset - gset)
        if extra:
            raise StaleSpecialistArtifact(
                f"specialist[{regime}] introduces feature columns absent from the global "
                f"scorer: {extra[:10]}{'…' if len(extra) > 10 else ''}. The specialist's recipe "
                "fingerprint disagrees with the global panel; refusing to load to avoid silent "
                "feature-shape miscalibration (CLAUDE.md §7.6 hardcoded-artifact-filename guard)."
            )

    # ── Score dispatch ────────────────────────────────────────────────────

    def score(self, ctx: Any, feature_matrix: pd.DataFrame) -> pd.Series:
        """Route to specialist(s) by detector regime + confidence.

        ``ctx`` must carry ``final_regime``, ``regime_confidence``, and
        ``regime_posterior`` (a dict[regime → prob]). When ``ctx`` is
        ``None`` or those fields are missing we fall back to the global
        scorer — keeps test harnesses that don't construct a full
        InferenceContext from breaking.
        """
        if ctx is None:
            return self._score_with(self.global_scorer, feature_matrix)
        final_regime = getattr(ctx, "final_regime", None) or getattr(ctx, "regime", None)
        confidence = float(getattr(ctx, "regime_confidence", None)
                           or getattr(ctx, "confidence", 0.0) or 0.0)
        posterior = dict(getattr(ctx, "regime_posterior", None) or {})

        if not self.specialists:
            return self._score_with(self.global_scorer, feature_matrix)

        # Hard-pick path: high-confidence regime call.
        if confidence >= self.confidence_threshold:
            if final_regime in self.specialists:
                log.info(
                    "RegimeEnsemblePanelScorer: hard-pick specialist=%s conf=%.3f >= %.3f",
                    final_regime, confidence, self.confidence_threshold,
                )
                return self._score_with(self.specialists[final_regime], feature_matrix)
            log.info(
                "RegimeEnsemblePanelScorer: final_regime=%s has no specialist "
                "(loaded=%s) — fallback to global",
                final_regime, sorted(self.specialists.keys()),
            )
            return self._score_with(self.global_scorer, feature_matrix)

        # Blend path: weighted average of top-k posterior entries that
        # actually have specialists loaded.
        eligible = [
            (regime, float(prob))
            for regime, prob in posterior.items()
            if regime in self.specialists and float(prob) > 0.0
        ]
        if not eligible:
            log.info(
                "RegimeEnsemblePanelScorer: conf=%.3f < %.3f but posterior has no "
                "specialist-backed regime (posterior=%s) — fallback to global",
                confidence, self.confidence_threshold, posterior,
            )
            return self._score_with(self.global_scorer, feature_matrix)
        eligible.sort(key=lambda r: r[1], reverse=True)
        top = eligible[: self.blend_top_k]
        total = sum(p for _, p in top)
        if total <= 0:
            return self._score_with(self.global_scorer, feature_matrix)
        blended: pd.Series | None = None
        weights_log: list[tuple[str, float]] = []
        for regime, prob in top:
            w = prob / total
            weights_log.append((regime, w))
            partial = self._score_with(self.specialists[regime], feature_matrix) * w
            blended = partial if blended is None else blended + partial
        log.info(
            "RegimeEnsemblePanelScorer: blend conf=%.3f weights=%s",
            confidence, weights_log,
        )
        assert blended is not None  # at least one eligible entry
        blended.name = "panel_score"
        return blended

    @staticmethod
    def _score_with(scorer: PanelScorer, feature_matrix: pd.DataFrame) -> pd.Series:
        """Run scorer.score() with only its own feature_cols subset.

        The ensemble's union ``feature_cols`` may include columns this
        specialist was not trained on; PanelScorer.score raises on
        unexpected columns? No — it raises on MISSING required cols.
        Extra columns are fine; pass the full matrix through.
        """
        missing = [c for c in scorer.feature_cols if c not in feature_matrix.columns]
        if missing:
            raise KeyError(
                "RegimeEnsemblePanelScorer._score_with: feature matrix missing "
                f"{len(missing)} columns required by specialist (first 5: {missing[:5]})"
            )
        return scorer.score(feature_matrix)


def load_panel_scorer_with_ensemble(
    panel_cfg: dict,
    strategy_dir: str | Path | None = None,
) -> RegimeEnsemblePanelScorer | PanelScorer:
    """Public entry — returns either the legacy or the ensemble scorer.

    The model_registry handler can use this in place of ``PanelScorer.load``
    when the artifact's enclosing config has a ``specialists`` block.
    """
    return RegimeEnsemblePanelScorer.load_from_config(panel_cfg, strategy_dir)


__all__ = [
    "RegimeEnsemblePanelScorer",
    "StaleSpecialistArtifact",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_BLEND_TOP_K",
    "load_panel_scorer_with_ensemble",
]
