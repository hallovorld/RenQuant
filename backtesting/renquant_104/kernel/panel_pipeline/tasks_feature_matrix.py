"""Feature-matrix Job — 4 small Tasks composing the legacy
BuildFeatureMatrixTask monolith.

User mandate (2026-05-04 §1c): split monoliths.

Composition (in `BuildFeatureMatrixJob` below):
  ResolveInferenceFramesTask    — subset frames, handle macro v1/v2
  AssembleInferenceMatrixTask   — call build_inference_matrix
  RowCoverageGateTask           — drop low-coverage rows
  DriftGuardTask                — structural vs transient NaN
"""
from __future__ import annotations

import datetime
import logging

import pandas as pd

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Job, Task

from .feature_matrix import build_inference_matrix
from .panel_scorer import PanelScorer

log = logging.getLogger("kernel.panel_pipeline.feature_matrix")


def _target_tickers(ctx: InferenceContext) -> list[str]:
    return sorted({c.ticker for c in ctx.candidates} | set(ctx.holdings.keys()))


def _set_target_only_matrix(
    ctx: InferenceContext,
    *,
    marker_col: str,
    scorer_kind: str | None,
    reason: str,
) -> bool:
    target = _target_tickers(ctx)
    ctx._panel_matrix = pd.DataFrame({marker_col: 1.0}, index=target)  # noqa: SLF001
    log.info(
        "ResolveInferenceFramesTask: %s scorer %s using target-only "
        "matrix for %d tickers",
        reason, scorer_kind, len(target),
    )
    return False


# ── 1. Resolve inference frames + macro v1/v2 silencing ─────────────────────

class ResolveInferenceFramesTask(Task):
    """Subset feature/factor/macro/asset_embeddings frames to target tickers.

    Reads:  ctx.candidates, ctx.holdings, ctx._panel_feature_frames,
             ctx._panel_factor_frames, ctx._panel_macro_frame,
             ctx._panel_asset_embeddings, ctx.config['panel_ltr']['macro']
    Writes: ctx._fm_inputs ({ff_subset, fac_subset, macro_frame,
             asset_embeddings, today_ts, target_tickers})
    """
    name = "ResolveInferenceFramesTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        if not ctx.candidates and not ctx.holdings:
            ctx._panel_matrix = None  # noqa: SLF001
            return None
        if getattr(ctx, "_panel_scorer", None) is None:
            ctx._panel_matrix = None  # noqa: SLF001
            return None
        ff = getattr(ctx, "_panel_feature_frames", None)
        if ff is None:
            scorer = getattr(ctx, "_panel_scorer", None)
            scorer_kind = (
                scorer.metadata.get("kind") if getattr(scorer, "metadata", None)
                else None
            )
            if scorer_kind in ("panel_linear", "panel_ltr_xgboost"):
                return _set_target_only_matrix(
                    ctx,
                    marker_col="__alpha158_target__",
                    scorer_kind=scorer_kind,
                    reason="alpha158",
                )
            if getattr(scorer, "requires_history", False):
                return _set_target_only_matrix(
                    ctx,
                    marker_col="__history_target__",
                    scorer_kind=scorer_kind,
                    reason="history",
                )
            log.warning("ResolveInferenceFramesTask: ctx._panel_feature_frames "
                        "missing — adapter must populate; matrix unset")
            ctx._panel_matrix = None  # noqa: SLF001
            return None
        fac = getattr(ctx, "_panel_factor_frames", None)
        macro_v = str(ctx.config.get("panel_ltr", {})
                                  .get("macro", {})
                                  .get("version", "v1")).lower()
        macro_frame = (None if macro_v == "v2"
                       else getattr(ctx, "_panel_macro_frame", None))
        target = set(_target_tickers(ctx))
        ff_sub = {t: ff[t] for t in target if t in ff}
        fac_sub = {t: fac[t] for t in target if t in fac} if fac else None
        today_ts = pd.Timestamp(ctx.today if isinstance(ctx.today, datetime.date)
                                  else ctx.today)
        ctx._fm_inputs = {  # noqa: SLF001
            "ff_subset": ff_sub, "fac_subset": fac_sub,
            "macro_frame": macro_frame, "today_ts": today_ts,
            "asset_embeddings": getattr(ctx, "_panel_asset_embeddings", None),
        }


# ── 2. Build the inference matrix ──────────────────────────────────────────

class AssembleInferenceMatrixTask(Task):
    """Call build_inference_matrix with the resolved inputs.

    Reads:  ctx._fm_inputs, ctx._panel_scorer, ctx.config['ranking']['panel_scoring']
    Writes: ctx._panel_matrix
    """
    name = "AssembleInferenceMatrixTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        inp = getattr(ctx, "_fm_inputs", None)
        if inp is None:
            return False
        scorer: PanelScorer = ctx._panel_scorer  # noqa: SLF001
        nan_prone = list(ctx.config.get("ranking", {})
                                     .get("panel_scoring", {})
                                     .get("nan_prone_cols", []))
        X = build_inference_matrix(
            inp["ff_subset"], inp["fac_subset"], inp["today_ts"],
            feature_cols=scorer.feature_cols,
            nan_prone_cols=nan_prone,
            macro_frame=inp["macro_frame"],
            asset_embeddings=inp["asset_embeddings"],
        )
        # 2026-05-05 P0 diag: log X shape + index sample to diagnose
        # the wl183 0-trade bug where ApplyScores reports 0/N scored.
        # If shape=(N,K) but cands report 0/N, X.index doesn't match
        # ctx.candidates (data path bug). If shape=(0,K), the matrix
        # builder dropped every ticker (frame issue).
        target = list(inp["ff_subset"].keys())
        log.info("AssembleInferenceMatrixTask: X.shape=%s  ff_sub=%d  "
                  "X.index[:5]=%s  target[:5]=%s",
                  X.shape, len(target),
                  list(X.index[:5]), target[:5])
        if X.empty:
            log.warning("AssembleInferenceMatrixTask: empty matrix")
            ctx._panel_matrix = None  # noqa: SLF001
            return None
        ctx._panel_matrix = X  # noqa: SLF001


# ── 3. Row-coverage filter ─────────────────────────────────────────────────

class RowCoverageGateTask(Task):
    """Drop rows whose feature coverage is below min_pct.

    Reads:  ctx._panel_matrix, ctx._panel_scorer.feature_cols, ctx.config
    Writes: ctx._panel_matrix (filtered, or None if empty after filter)
    """
    name = "RowCoverageGateTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        X = getattr(ctx, "_panel_matrix", None)
        if X is None:
            return None
        from kernel.row_coverage import coverage_from_config, filter_by_coverage
        enabled, min_pct = coverage_from_config(ctx.config)
        if not enabled:
            return
        scorer = ctx._panel_scorer  # noqa: SLF001
        # 2026-05-05 wl183 0-trade fix: inference X is ticker-indexed.
        # Without preserve_index=True, filter_by_coverage's default reset
        # to int64 0..n-1 broke every downstream `scores.get(cand.ticker)`
        # lookup → 0/N scored → 0 trades on every bar. See row_coverage.py
        # docstring for the full incident write-up.
        X, stats = filter_by_coverage(
            X, list(scorer.feature_cols), min_pct, preserve_index=True,
        )
        if stats["n_dropped"]:
            log.info("RowCoverageGateTask: dropped %d/%d (%.1f%%) "
                      "below %.0f%% coverage",
                      stats["n_dropped"], stats["n_in"],
                      stats["pct_dropped"] * 100, min_pct * 100)
        if X.empty:
            log.warning("RowCoverageGateTask: filter dropped all rows — "
                        "improve data coverage or disable")
            ctx._panel_matrix = None  # noqa: SLF001
            return None
        ctx._panel_matrix = X  # noqa: SLF001


# ── 4. Drift guard (structural vs transient NaN) ──────────────────────────

class DriftGuardTask(Task):
    """Differentiate structural drift (column never produced) from
    transient (column exists but today's data not cached yet).

    Reads:  ctx._panel_matrix, ctx._panel_scorer.feature_cols,
             ctx._fm_inputs.{ff_subset,fac_subset}, ctx.config
    Writes: ctx._panel_matrix → None when structural drift > threshold;
             ctx.candidates → [] in that case (fail-SAFE)
    """
    name = "DriftGuardTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        X = getattr(ctx, "_panel_matrix", None)
        if X is None:
            return None
        scorer = ctx._panel_scorer  # noqa: SLF001
        # Phase 3 (2026-05-06): alpha158 models use a separate inference
        # path (features built per-ticker from raw OHLCV in ApplyScoresTask).
        # The X matrix at this stage is XGB-shaped (21 cols) and doesn't
        # match scorer.feature_cols (158 alpha158 cols). Skip drift check —
        # ApplyScoresTask handles feature building for these kinds.
        scorer_kind = (scorer.metadata.get("kind")
                       if hasattr(scorer, "metadata") else None)
        if scorer_kind in ("panel_linear", "panel_ltr_xgboost"):
            return None
        # 2026-05-19 (shadow full-e2e): sequence-input scorers (PatchTST,
        # HF PatchTST) have requires_history=True and pull their own
        # per-ticker sequence data inside score_with_history(). They don't
        # consume ctx._panel_matrix, so checking the matrix columns against
        # scorer.feature_cols is meaningless. Caught in first shadow run
        # 2026-05-19 19:23: HFPatchTSTPanelScorer has 172 feature_cols but
        # _panel_matrix is the 21-col XGB shape → 100% structural drift
        # → fail-safe cleared all candidates → no_candidates.
        if getattr(scorer, "requires_history", False):
            return None
        nan_cols = [c for c in scorer.feature_cols if X[c].isna().all()]
        if not nan_cols:
            return
        inp = getattr(ctx, "_fm_inputs", {}) or {}
        produced = self._produced_cols(inp.get("ff_subset"), inp.get("fac_subset"))
        structural = [c for c in nan_cols if c not in produced]
        transient  = [c for c in nan_cols if c in produced]
        if transient:
            log.warning("DriftGuardTask: %d transient all-NaN col(s) — "
                        "XGBoost NaN-imputes; scoring continues. Cols: %s",
                        len(transient), transient[:5])
        if not structural:
            return
        thr = float(ctx.config.get("ranking", {})
                              .get("panel_scoring", {})
                              .get("max_feature_drift_pct", 0.05))
        n_total = len(scorer.feature_cols)
        if len(structural) / max(1, n_total) > thr:
            log.error("DriftGuardTask: %d/%d (%.1f%%) STRUCTURALLY missing "
                      "— FAIL-SAFE clearing candidates. First 10: %s",
                      len(structural), n_total,
                      len(structural) / max(1, n_total) * 100,
                      structural[:10])
            ctx._panel_matrix = None  # noqa: SLF001
            ctx.candidates = []
            return False

    @staticmethod
    def _produced_cols(ff_subset, fac_subset) -> set:
        out: set = set()
        for frames in (ff_subset, fac_subset or {}):
            if not frames:
                continue
            for ff in frames.values():
                if hasattr(ff, "columns"):
                    out.update(ff.columns)
                elif isinstance(ff, dict):
                    out.update(ff.keys())
        return out


# ── Job orchestrator ───────────────────────────────────────────────────────

class BuildFeatureMatrixJob(Job):
    """Phase chain that replaces the legacy 165-line BuildFeatureMatrixTask."""
    name = "BuildFeatureMatrixJob"

    @property
    def tasks(self) -> list[Task]:
        return [
            ResolveInferenceFramesTask(),
            AssembleInferenceMatrixTask(),
            RowCoverageGateTask(),
            DriftGuardTask(),
        ]


__all__ = [
    "ResolveInferenceFramesTask",
    "AssembleInferenceMatrixTask",
    "RowCoverageGateTask",
    "DriftGuardTask",
    "BuildFeatureMatrixJob",
]
