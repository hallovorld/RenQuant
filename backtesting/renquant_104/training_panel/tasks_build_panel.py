"""BuildPanel Job — 7 Tasks splitting the legacy 155-line BuildPanelTask
(per CLAUDE.md §1c, 2026-05-04).

Composition:
  SliceWatchlistFramesTask   — pick ff/lab/sec/fac for watchlist tickers
  AssemblePanelFrameTask     — call build_panel_frame
  MergeRawResidualsTask      — merge ctx.raw_residuals → panel
  ForwardFillImputeTask      — whitelisted slow features
  RowCoverageFilterTask      — drop low-coverage rows
  NaNFillFeaturesTask        — final NaN→0 + missingness indicators (E28 fix)
  FinalizePanelTask          — exclude/drop_cols + commit feature_cols
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .pp_panel_training import DEFAULT_DROP_COLS, PanelJob, PanelTask
from .context import PanelTrainingContext

log = logging.getLogger("training_panel.build_panel")


# ── 1. Slice frames + read panel_ltr cfg ───────────────────────────────────

class SliceWatchlistFramesTask(PanelTask):
    """Subset per-ticker frames to ctx.watchlist; resolve macro v1/v2.

    Reads:  ctx.{neutralized_frames, labels, factor_frames, ticker_sectors,
             watchlist, macro_factor_frame, asset_embeddings, listing_dates,
             config}
    Writes: ctx._bp_inputs ({ff_wl, lab_wl, sec_wl, fac_wl, macro_for_panel,
             asset_embeddings, listing_dates, cfg})
    """
    name = "SliceWatchlistFramesTask"

    def run(self, ctx: PanelTrainingContext) -> None:
        cfg = ctx.config.get("panel_ltr", {})
        macro_v = str(cfg.get("macro", {}).get("version", "v1")).lower()
        macro_for_panel = (
            None if macro_v == "v2" else ctx.macro_factor_frame
        )
        ctx._bp_inputs = {
            "ff_wl":  {t: ctx.neutralized_frames[t]
                        for t in ctx.watchlist
                        if t in ctx.neutralized_frames},
            "lab_wl": {t: ctx.labels[t] for t in ctx.watchlist if t in ctx.labels},
            "sec_wl": {t: ctx.ticker_sectors[t]
                        for t in ctx.watchlist if t in ctx.ticker_sectors},
            "fac_wl": {t: ctx.factor_frames[t]
                        for t in ctx.watchlist if t in ctx.factor_frames},
            "macro_for_panel": macro_for_panel,
            "asset_embeddings": ctx.asset_embeddings or None,
            "listing_dates":   ctx.listing_dates,
            "cfg":             cfg,
        }


# ── 2. Call build_panel_frame ──────────────────────────────────────────────

class AssemblePanelFrameTask(PanelTask):
    """Build the long-form panel + group_sizes via training_panel.panel_frame.

    Reads:  ctx._bp_inputs
    Writes: ctx._bp_panel, ctx._bp_group_sizes, ctx._bp_meta
    """
    name = "AssemblePanelFrameTask"

    def run(self, ctx: PanelTrainingContext) -> None:
        from .panel_frame import build_panel_frame, resolve_lookahead_days
        inp = ctx._bp_inputs
        cfg = inp["cfg"]
        # Track C8 / P3.3 (2026-05-10): single source of truth for lookahead.
        # `cfg` here is the panel_ltr sub-dict (set by SliceWatchlistFramesTask
        # above) — resolve_lookahead_days accepts either shape.
        lookahead = resolve_lookahead_days(cfg)
        log.info(
            "AssemblePanelFrameTask: lookahead horizon = %d trading days "
            "(from panel_ltr.lookahead_days)", lookahead,
        )
        panel, gs, meta = build_panel_frame(
            inp["ff_wl"], inp["lab_wl"], inp["sec_wl"],
            factor_frames=inp["fac_wl"],
            macro_frame=inp["macro_for_panel"],
            asset_embeddings=inp["asset_embeddings"],
            listing_dates=inp["listing_dates"],
            min_history_days=int(cfg.get("min_history_days", 252)),
            lookahead_days=lookahead,
            age_warmup_days=int(cfg.get("age_warmup_days", 504)),
            nan_prone_cols=list(cfg.get("nan_prone_cols", [])),
            training_window_years=cfg.get("training_window_years"),
            recency_weighting=cfg.get("recency_weighting"),
        )
        # Drop rows with no label (training-only data has NaN labels at tail)
        panel = panel[panel["label"].notna()].reset_index(drop=True)
        gs = panel.groupby("date", sort=True).size().to_numpy().astype(np.int32)
        ctx._bp_panel       = panel
        ctx._bp_group_sizes = gs
        ctx._bp_meta        = meta


# ── 3. Merge raw_residuals for NGBoost head ────────────────────────────────

class MergeRawResidualsTask(PanelTask):
    """Append `residual_return_raw` column from ctx.raw_residuals onto panel.

    Reads:  ctx._bp_panel, ctx.raw_residuals
    Writes: ctx._bp_panel (in place — column added if raw_residuals present)
    """
    name = "MergeRawResidualsTask"

    def run(self, ctx: PanelTrainingContext) -> None:
        if not ctx.raw_residuals:
            return
        rows = []
        for t, s in ctx.raw_residuals.items():
            rows.append(pd.DataFrame({
                "ticker": t,
                "date":   pd.to_datetime(s.index),
                "residual_return_raw": s.values,
            }))
        raw_df = pd.concat(rows, ignore_index=True)
        panel = ctx._bp_panel
        panel["date"] = pd.to_datetime(panel["date"])
        ctx._bp_panel = panel.merge(raw_df, on=["ticker", "date"], how="left")


# ── 4. Forward-fill whitelisted slow features ──────────────────────────────

class ForwardFillImputeTask(PanelTask):
    """Forward-fill whitelisted slow-moving features (per ticker).

    Reads:  ctx._bp_panel, ctx._bp_inputs.cfg.imputation
    Writes: ctx._bp_panel (filled in place)
    """
    name = "ForwardFillImputeTask"

    def run(self, ctx: PanelTrainingContext) -> None:
        imp_cfg = ctx._bp_inputs["cfg"].get("imputation", {})
        cols = list(imp_cfg.get("ffill_cols", []))
        max_gap = int(imp_cfg.get("ffill_max_gap_days", 5))
        if not cols or max_gap <= 0:
            return
        from .imputation import forward_fill_per_ticker
        before = ctx._bp_panel
        filled = forward_fill_per_ticker(before, cols, max_gap_days=max_gap)
        n_filled = (
            int(before[cols].isna().sum().sum() - filled[cols].isna().sum().sum())
            if all(c in before.columns for c in cols) else 0
        )
        log.info(
            "ForwardFillImputeTask: filled %d cells across %d cols (gap≤%dd)",
            n_filled, len(cols), max_gap,
        )
        ctx._bp_panel = filled


# ── 5. Row-coverage filter ─────────────────────────────────────────────────

class RowCoverageFilterTask(PanelTask):
    """Drop rows whose feature coverage is below min_pct.

    Reads:  ctx._bp_panel, ctx._bp_inputs.cfg, ctx._bp_feature_cols (computed
             by FinalizePanelTask but RowCov runs first — we use a pre-pass).
    Writes: ctx._bp_panel (filtered), ctx._bp_group_sizes (recomputed),
             ctx._bp_meta["row_coverage_stats"]
    """
    name = "RowCoverageFilterTask"

    def run(self, ctx: PanelTrainingContext) -> None:
        from kernel.row_coverage import coverage_from_config, filter_by_coverage
        enabled, min_pct = coverage_from_config(ctx.config)
        if not enabled:
            return
        # Compute candidate feature_cols pre-Finalize (Finalize runs after)
        panel = ctx._bp_panel
        user_drop = ctx._bp_inputs["cfg"].get("drop_cols")
        drop_cols = set(DEFAULT_DROP_COLS)
        if user_drop is not None:
            drop_cols |= set(user_drop)
        exclude = {"date", "ticker", "sector", "label",
                    "residual_return_raw",
                    "weight", "weight_concurrency", "weight_age",
                    "weight_recency"} | drop_cols
        feature_cols = [c for c in panel.columns if c not in exclude]
        if not feature_cols:
            return
        panel, stats = filter_by_coverage(panel, feature_cols, min_pct)
        log.info("RowCoverageFilterTask: dropped %d/%d (%.1f%%) "
                  "rows below %.0f%% coverage",
                  stats["n_dropped"], stats["n_in"],
                  stats["pct_dropped"] * 100, min_pct * 100)
        ctx._bp_panel = panel
        ctx._bp_group_sizes = (
            panel.groupby("date", sort=True).size().to_numpy().astype(np.int32)
        )
        if isinstance(ctx._bp_meta, dict):
            ctx._bp_meta["row_coverage_stats"] = stats


# ── 6. NaN-fill features + missingness indicators (E28 fix) ────────────────

class NaNFillFeaturesTask(PanelTask):
    """Final NaN handling for feature columns before XGB sees them.

    Closes E28 (NaN-leaf collapse: 60.8% training rows routed to same
    terminal node because XGB's default-direction sends every NaN-rich
    row down the same path). For each candidate feature column with NaN
    rate above ``missingness_threshold_pct``, append a
    ``{col}_is_missing ∈ {0,1}`` indicator so the model can still learn
    "missingness itself is informative". Then fill all remaining NaN in
    feature columns with 0.0 — z-scored features are zero-mean, so 0 is
    the natural neutral baseline.

    Reads:  ctx._bp_panel, ctx._bp_inputs.cfg.imputation
    Writes: ctx._bp_panel (NaN cells filled, indicator cols added)
    """
    name = "NaNFillFeaturesTask"

    def run(self, ctx: PanelTrainingContext) -> None:
        imp_cfg = ctx._bp_inputs["cfg"].get("imputation", {})
        fill_zero = bool(imp_cfg.get("fill_zero", False))
        add_indicators = bool(imp_cfg.get("add_missingness_indicators", False))
        # Skip if neither knob is on (preserves legacy "do nothing" default).
        if not fill_zero and not add_indicators:
            return
        threshold_pct = float(imp_cfg.get("missingness_threshold_pct", 5.0)) / 100.0

        panel = ctx._bp_panel
        user_drop = ctx._bp_inputs["cfg"].get("drop_cols")
        drop_cols = set(DEFAULT_DROP_COLS)
        if user_drop is not None:
            drop_cols |= set(user_drop)
        exclude = {"date", "ticker", "sector", "label",
                    "residual_return_raw",
                    "weight", "weight_concurrency", "weight_age",
                    "weight_recency"} | drop_cols
        feat_cols = [c for c in panel.columns
                     if c not in exclude and not c.endswith("_is_missing")]
        if not feat_cols:
            return

        nan_rates = panel[feat_cols].isna().mean()
        nan_rich = nan_rates[nan_rates > threshold_pct].index.tolist()
        n_nan_total = int(panel[feat_cols].isna().sum().sum())

        # Indicators are independent of fill: Option C = indicators only,
        # Option A = indicators + fill, Option B (rare) = fill only.
        if add_indicators and nan_rich:
            for col in nan_rich:
                ind = f"{col}_is_missing"
                if ind not in panel.columns:
                    panel[ind] = panel[col].isna().astype(np.int8)

        if fill_zero:
            panel[feat_cols] = panel[feat_cols].fillna(0.0)

        log.info(
            "NaNFillFeaturesTask: %d NaN cells / %d feat cols  "
            "indicators_added=%d  fill_zero=%s  threshold=%.1f%%",
            n_nan_total, len(feat_cols),
            len(nan_rich) if add_indicators else 0,
            fill_zero, threshold_pct * 100,
        )
        ctx._bp_panel = panel


# ── 7. Finalize: feature_cols + commit ────────────────────────────────────

class FinalizePanelTask(PanelTask):
    """Compute feature_cols (excluding label/meta/drop_cols) + commit.

    Reads:  ctx._bp_panel, ctx._bp_group_sizes, ctx._bp_meta, ctx._bp_inputs.cfg
    Writes: ctx.panel, ctx.group_sizes, ctx.panel_metadata, ctx.feature_cols
    """
    name = "FinalizePanelTask"

    def run(self, ctx: PanelTrainingContext) -> None:
        cfg = ctx._bp_inputs["cfg"]
        user_drop = cfg.get("drop_cols")
        drop_cols = set(DEFAULT_DROP_COLS)
        if user_drop is not None:
            drop_cols |= set(user_drop)
        exclude = {"date", "ticker", "sector", "label",
                    "residual_return_raw",
                    "weight", "weight_concurrency", "weight_age",
                    "weight_recency"} | drop_cols
        panel = ctx._bp_panel
        feature_cols = [c for c in panel.columns if c not in exclude]
        if drop_cols & set(panel.columns):
            log.info("FinalizePanelTask: dropped non-ranking cols %s",
                     sorted(drop_cols & set(panel.columns)))
        ctx.panel          = panel
        ctx.group_sizes    = ctx._bp_group_sizes
        ctx.panel_metadata = ctx._bp_meta
        ctx.feature_cols   = feature_cols
        log.info("FinalizePanelTask: panel=%d rows  features=%d  tickers=%d  dates=%d",
                  len(panel), len(feature_cols),
                  ctx._bp_meta.get("n_tickers"), ctx._bp_meta.get("n_dates"))


# ── Job orchestrator ───────────────────────────────────────────────────────

class BuildPanelJob(PanelJob):
    """Replaces the 155-line BuildPanelTask monolith."""
    name = "BuildPanelJob"

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        return ctx.panel is not None

    @property
    def tasks(self) -> list[PanelTask]:
        return [
            SliceWatchlistFramesTask(),
            AssemblePanelFrameTask(),
            MergeRawResidualsTask(),
            ForwardFillImputeTask(),
            RowCoverageFilterTask(),
            NaNFillFeaturesTask(),
            FinalizePanelTask(),
        ]


__all__ = [
    "SliceWatchlistFramesTask",
    "AssemblePanelFrameTask",
    "MergeRawResidualsTask",
    "ForwardFillImputeTask",
    "RowCoverageFilterTask",
    "NaNFillFeaturesTask",
    "FinalizePanelTask",
    "BuildPanelJob",
]
