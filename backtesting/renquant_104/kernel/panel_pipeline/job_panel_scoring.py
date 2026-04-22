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
        p = Path(artifact_path)
        if not p.is_absolute():
            strategy_dir = ctx.config.get("_strategy_dir")
            if strategy_dir:
                p = Path(strategy_dir) / p
        try:
            ctx._panel_scorer = PanelScorer.load(p)  # noqa: SLF001
        except Exception as exc:
            log.error("LoadScorerTask: failed to load %s — %s", p, exc)
            return False
        log.info("LoadScorerTask: loaded artifact (features=%d)",
                 len(ctx._panel_scorer.feature_cols))


class BuildFeatureMatrixTask(Task):
    """Pick today's row per candidate + held ticker into a single feature matrix.

    Held positions are scored alongside candidates so rotation can compare
    them on the same cross-sectional panel scale.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        if not ctx.candidates and not ctx.holdings:
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
        target_tickers = {c.ticker for c in ctx.candidates} | set(ctx.holdings.keys())
        ff_subset = {t: feature_frames[t] for t in target_tickers if t in feature_frames}
        fac_subset = None
        if factor_frames is not None:
            fac_subset = {t: factor_frames[t] for t in target_tickers if t in factor_frames}

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
    """Score the matrix and write panel_score onto candidates AND holdings.

    For candidates the panel score also overwrites `rank_score` so the
    downstream RankingJob/SelectionJob path is unchanged. For holdings we
    only populate the new `panel_score` field — per-ticker `rank_score`
    (set by ScoreModelTask) stays intact for exit logic.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        scorer: PanelScorer = getattr(ctx, "_panel_scorer", None)
        X = getattr(ctx, "_panel_matrix", None)
        if scorer is None or X is None or X.empty:
            return False
        scores: pd.Series = scorer.score(X)

        n_cand_scored = 0
        for cand in ctx.candidates:
            v = scores.get(cand.ticker)
            if v is None or pd.isna(v):
                continue
            cand.rank_score  = float(v)
            cand.panel_score = float(v)
            n_cand_scored += 1

        n_held_scored = 0
        for ticker, hs in ctx.holdings.items():
            v = scores.get(ticker)
            if v is None or pd.isna(v):
                continue
            hs.panel_score = float(v)
            n_held_scored += 1

        log.info("ApplyScoresTask: panel scored %d/%d candidates, %d/%d holdings",
                 n_cand_scored, len(ctx.candidates),
                 n_held_scored, len(ctx.holdings))


class VetoWeakBuysTask(Task):
    """Drop candidates whose panel_score is below `buy_floor`.

    No-op when buy_floor is unset or <= -inf. Candidates without a panel
    score (e.g. missing features) are kept — RankingJob blends rs_score in.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        if not ctx.candidates:
            return False
        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        floor     = panel_cfg.get("buy_floor")
        if floor is None:
            return
        floor = float(floor)

        kept: list = []
        dropped = 0
        for cand in ctx.candidates:
            ps = cand.panel_score
            if ps is not None and ps < floor:
                dropped += 1
                continue
            kept.append(cand)

        if dropped:
            ctx.candidates = kept
            ctx.counters["panel_vetoed"] = ctx.counters.get("panel_vetoed", 0) + dropped
            log.info("VetoWeakBuysTask: dropped %d candidate(s) below panel_score=%.3f",
                     dropped, floor)


# ── Job ──────────────────────────────────────────────────────────────────────

class PanelScoringJob(Job):
    """Overwrite rank_score on surviving candidates with cross-sectional panel scores.

    Task chain: LoadScorer → BuildFeatureMatrix → ApplyScores → VetoWeakBuys
    """

    def should_skip(self, ctx: InferenceContext) -> bool:
        # Run even with no candidates so holdings can still be panel-scored
        # for rotation decisions later in the pipeline.
        if not ctx.candidates and not ctx.holdings:
            return True
        return not ctx.config.get("ranking", {}).get("panel_scoring", {}).get("enabled", False)

    @property
    def tasks(self) -> list[Task]:
        return [LoadScorerTask(), BuildFeatureMatrixTask(), ApplyScoresTask(), VetoWeakBuysTask()]
