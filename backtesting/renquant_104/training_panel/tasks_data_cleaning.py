"""DataCleaning Job — three Tasks implementing Qlib's standard feature processors.

Reference: qlib/data/dataset/handler.py Alpha158 DEFAULT_PROCESSORS (Lines 37-152)
  ProcessInf        → ProcessInfTask
  RobustZScoreNorm  → RobustZScoreNormTask
  CSZScoreNorm      → CSZScoreNormFeaturesTask
  Fillna            → FillnaTask

Composition (DataCleaningJob):
  ProcessInfTask            — ±inf → NaN
  RobustZScoreNormTask      — per-feature robust z-score (train stats, clip 3σ)
  CSZScoreNormFeaturesTask  — cross-sectional z-score per date per feature
  FillnaTask                — remaining NaN → 0

Why Tasks not a monolith: each step has a distinct failure mode.
  ProcessInf failure = silently NaN-propagated gradients
  RobustZScore failure = scale mismatch between features
  CSZScore failure = cross-sectional signal suppression
  Fillna failure = NaN leaking into loss
Splitting makes each independently testable and the log pinpoints which step
produced unusual output.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .pp_panel_training import PanelJob, PanelTask
from .context import PanelTrainingContext

log = logging.getLogger("training_panel.data_cleaning")

_EPS = 1e-9


# ── 1. ProcessInfTask ──────────────────────────────────────────────────────

class ProcessInfTask(PanelTask):
    """Replace ±inf with NaN in all feature columns.

    Reads:  ctx.panel (DataFrame), ctx.feature_cols
    Writes: ctx.panel (in-place)

    Ref: Qlib ProcessInf processor — first step in the default pipeline so
    that downstream stats (mean, std, median) aren't corrupted by infinities.
    """
    name = "ProcessInfTask"

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        panel = ctx.panel
        feat_cols = ctx.feature_cols
        if panel is None or not feat_cols:
            return None
        n_inf = np.isinf(panel[feat_cols].values).sum()
        if n_inf:
            panel[feat_cols] = panel[feat_cols].replace([np.inf, -np.inf], np.nan)
            log.info("ProcessInfTask: replaced %d ±inf values with NaN", n_inf)
        else:
            log.debug("ProcessInfTask: no ±inf values found")
        ctx.panel = panel


# ── 2. RobustZScoreNormTask ────────────────────────────────────────────────

class RobustZScoreNormTask(PanelTask):
    """Per-feature robust z-score using train-only median and MAD, clip at 3σ.

    Reads:  ctx.panel, ctx.feature_cols, ctx._clean_train_mask
    Writes: ctx.panel (in-place), ctx._clean_feature_stats (dict for inference)

    Ref: Qlib RobustZScoreNorm — uses median and MAD (×1.4826 ≈ σ for normal
    distributions) instead of mean/std so single outlier tickers don't skew
    the normalization. Clip at ±3σ after normalizing.

    ctx._clean_feature_stats is stored so inference-time scoring can apply the
    same normalization with training statistics (no data leakage at test time).
    """
    name = "RobustZScoreNormTask"

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        panel = ctx.panel
        feat_cols = ctx.feature_cols
        train_mask = getattr(ctx, "_clean_train_mask", None)
        if panel is None or not feat_cols:
            return None

        if train_mask is None:
            train_mask = panel.get("split_label", pd.Series("train", index=panel.index)) == "train"
            log.warning("RobustZScoreNormTask: no _clean_train_mask, using all rows as train")

        stats: dict[str, dict] = {}
        for c in feat_cols:
            col_train = panel.loc[train_mask, c].dropna()
            med = float(col_train.median()) if len(col_train) else 0.0
            mad = float((col_train - med).abs().median()) if len(col_train) else 1.0
            scale = max(mad * 1.4826, _EPS)
            stats[c] = {"median": med, "mad_scale": scale}
            panel[c] = ((panel[c] - med) / scale).clip(-3.0, 3.0)

        ctx.panel = panel
        ctx._clean_feature_stats = stats
        log.info("RobustZScoreNormTask: normalized %d features (train rows=%d)",
                 len(feat_cols), int(train_mask.sum()))


# ── 3. CSZScoreNormFeaturesTask ────────────────────────────────────────────

class CSZScoreNormFeaturesTask(PanelTask):
    """Cross-sectional z-score per date per feature.

    Reads:  ctx.panel, ctx.feature_cols
    Writes: ctx.panel (in-place)

    Ref: Qlib CSZScoreNorm — applied to features so each day's cross-section
    has mean=0, std=1 per feature. This removes time-series trends in feature
    levels (e.g. VMA rising across a bull market) that would otherwise leak
    temporal signal into the cross-sectional ranking model.

    Deviation from Qlib: Qlib applies CSZScoreNorm to labels only by default,
    with ZScoreNorm (global) on features. We apply both — global RobustZScore
    first (RobustZScoreNormTask above), then CSZScore — matching the more
    aggressive preprocessing in MASTER (Li et al. 2024) and AlphaPortfolio.
    """
    name = "CSZScoreNormFeaturesTask"

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        panel = ctx.panel
        feat_cols = ctx.feature_cols
        if panel is None or not feat_cols or "date" not in panel.columns:
            return None

        for c in feat_cols:
            date_mean = panel.groupby("date")[c].transform("mean")
            date_std  = panel.groupby("date")[c].transform("std")
            panel[c] = (panel[c] - date_mean) / (date_std + _EPS)

        ctx.panel = panel
        log.info("CSZScoreNormFeaturesTask: cross-sectional z-scored %d features",
                 len(feat_cols))


# ── 4. FillnaTask ──────────────────────────────────────────────────────────

class FillnaTask(PanelTask):
    """Fill remaining NaN in feature columns with 0.

    Reads:  ctx.panel, ctx.feature_cols
    Writes: ctx.panel (in-place)

    Ref: Qlib Fillna processor — applied last so that clipping and normalization
    are done on real values only; NaN rows receive the cross-sectional neutral
    value (0 after z-scoring = mean of the cross-section).
    """
    name = "FillnaTask"

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        panel = ctx.panel
        feat_cols = ctx.feature_cols
        if panel is None or not feat_cols:
            return None

        n_nan_before = panel[feat_cols].isna().sum().sum()
        panel[feat_cols] = panel[feat_cols].fillna(0.0)
        if n_nan_before:
            log.info("FillnaTask: filled %d NaN values with 0", int(n_nan_before))
        ctx.panel = panel


# ── Job ────────────────────────────────────────────────────────────────────

class DataCleaningJob(PanelJob):
    """Chain the four Qlib-standard data cleaning Tasks.

    Call before any model training. Expects ctx.panel and ctx.feature_cols
    to be set. Optionally set ctx._clean_train_mask (boolean Series aligned
    to panel index) to ensure robust z-score stats are fit on training rows only.

    Steps: ProcessInf → RobustZScoreNorm → CSZScoreNormFeatures → Fillna
    """

    @property
    def tasks(self):
        return [
            ProcessInfTask(),
            RobustZScoreNormTask(),
            CSZScoreNormFeaturesTask(),
            FillnaTask(),
        ]

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        return ctx.panel is None or not ctx.feature_cols
