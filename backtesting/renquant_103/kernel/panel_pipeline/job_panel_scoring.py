"""PanelScoringJob — swap in cross-sectional panel scores during inference.

Slots between CandidateJob (Phase 2) and RankingJob (Phase 3) of the
standard InferencePipeline. When the config flag
`ranking.panel_scoring.enabled` is true and a panel-LTR artifact is
configured, this Job loads the scorer, builds today's inference matrix
for every candidate ticker, and overwrites each CandidateResult's
`rank_score` in place. The existing RankingJob then blends that panel
score with rs_score using the same `ranking.blend_weights`.

Task chain::

    LoadScorerTask           read artifact path from config, cache scorer
    BuildFeatureMatrixTask   pick today's rows per candidate ticker
    ApplyScoresTask          write panel_score into CandidateResult.rank_score

The Job is a no-op when:
  • the config flag is off, OR
  • no candidates survived Phase 2 (ctx.candidates empty), OR
  • the artifact can't be loaded (logged, Job short-circuits).

Kept isolated from the Stage-1 training pipeline so revert is purely
additive: remove this file + the one-line import wiring.
"""
from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Job, Task

from .panel_scorer import PanelScorer
from .feature_matrix import build_inference_matrix

log = logging.getLogger("kernel.panel_pipeline.scoring")


# ── Task chain ────────────────────────────────────────────────────────────────

class LoadScorerTask(Task):
    """Load the PanelScorer artifact from config. Cache on ctx for reuse."""

    def run(self, ctx: InferenceContext) -> bool | None:
        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        if not panel_cfg.get("enabled", False):
            log.debug("LoadScorerTask: panel scoring disabled — skipping chain")
            return False

        # Scorer may have been pre-loaded by the adapter (live runner / LEAN)
        scorer = getattr(ctx, "_panel_scorer", None)
        if scorer is not None:
            return

        artifact_path = panel_cfg.get("artifact_path")
        if not artifact_path:
            log.warning("LoadScorerTask: panel_scoring.enabled but no artifact_path — skipping")
            return False
        try:
            p = Path(artifact_path)
            if not p.is_absolute():
                strategy_dir = ctx.config.get("_strategy_dir")
                if strategy_dir:
                    p = Path(strategy_dir) / p
            ctx._panel_scorer = PanelScorer.load(p)  # noqa: SLF001
        except Exception as exc:
            log.error("LoadScorerTask: failed to load %s — %s", artifact_path, exc)
            return False
        log.info("LoadScorerTask: loaded artifact (features=%d)",
                 len(ctx._panel_scorer.feature_cols))


class BuildFeatureMatrixTask(Task):
    """Pick today's row per candidate ticker into a single feature matrix."""

    def run(self, ctx: InferenceContext) -> bool | None:
        if not ctx.candidates:
            return False
        scorer: PanelScorer = getattr(ctx, "_panel_scorer", None)
        if scorer is None:
            return False

        feature_frames = getattr(ctx, "_panel_feature_frames", None)
        factor_frames  = getattr(ctx, "_panel_factor_frames", None)
        if feature_frames is None:
            log.warning("BuildFeatureMatrixTask: ctx has no _panel_feature_frames "
                        "(adapter must populate before RankingJob) — skipping")
            return False

        today = ctx.today
        today_ts = pd.Timestamp(today if isinstance(today, datetime.date) else today)
        candidate_tickers = {c.ticker for c in ctx.candidates}
        ff_subset = {t: feature_frames[t] for t in candidate_tickers if t in feature_frames}
        fac_subset = None
        if factor_frames is not None:
            fac_subset = {t: factor_frames[t] for t in candidate_tickers if t in factor_frames}

        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        nan_prone = list(panel_cfg.get("nan_prone_cols", []))

        X = build_inference_matrix(
            ff_subset, fac_subset, today_ts,
            feature_cols=scorer.feature_cols,
            nan_prone_cols=nan_prone,
        )
        if X.empty:
            log.warning("BuildFeatureMatrixTask: empty inference matrix — skipping")
            return False
        ctx._panel_matrix = X  # noqa: SLF001
        log.debug("BuildFeatureMatrixTask: matrix %s", X.shape)


class ApplyScoresTask(Task):
    """Score the matrix and overwrite CandidateResult.rank_score in place."""

    def run(self, ctx: InferenceContext) -> bool | None:
        scorer: PanelScorer = getattr(ctx, "_panel_scorer", None)
        X = getattr(ctx, "_panel_matrix", None)
        if scorer is None or X is None or X.empty:
            return False
        scores: pd.Series = scorer.score(X)
        for cand in ctx.candidates:
            v = scores.get(cand.ticker)
            if v is None or pd.isna(v):
                continue
            cand.rank_score = float(v)
        log.info("ApplyScoresTask: wrote panel score to %d/%d candidates",
                 int(scores.notna().sum()), len(ctx.candidates))


# ── Job ──────────────────────────────────────────────────────────────────────

class PanelScoringJob(Job):
    """Overwrite rank_score on surviving candidates with cross-sectional panel scores.

    Task chain: LoadScorer → BuildFeatureMatrix → ApplyScores
    """

    def should_skip(self, ctx: InferenceContext) -> bool:
        if not ctx.candidates:
            return True
        return not ctx.config.get("ranking", {}).get("panel_scoring", {}).get("enabled", False)

    @property
    def tasks(self) -> list[Task]:
        return [LoadScorerTask(), BuildFeatureMatrixTask(), ApplyScoresTask()]
