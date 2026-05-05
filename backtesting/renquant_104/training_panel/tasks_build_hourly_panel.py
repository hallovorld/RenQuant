"""BuildHourlyResolutionPanel Job — 5 Tasks splitting the legacy
~160-line BuildHourlyResolutionPanelTask (per CLAUDE.md §1c).

Composition:
  LoadHourlyBarsForPanelTask   — read HourlyBarStore for watchlist + benchmark
  AssembleHourlyPanelTask      — call build_hourly_resolution_panel
  NormalizeHourlySchemaTask    — reset_index, rename, derive date/hour, drop NaN labels
  BroadcastMacroToHourlyTask   — merge macro_factor_frame onto hourly rows
  FinalizeHourlyPanelTask      — feature_cols + dtype filter + commit
"""
from __future__ import annotations

import logging

import pandas as pd

from .pp_panel_training import _resolve_cache_dir, PanelJob, PanelTask
from .context import PanelTrainingContext

log = logging.getLogger("training_panel.build_hourly_panel")


# ── 1. Load hourly bars from cache ─────────────────────────────────────────

class LoadHourlyBarsForPanelTask(PanelTask):
    """Load per-ticker hourly bars + benchmark hourly bars.

    Reads:  ctx.config['panel_ltr']['hourly'], ctx.watchlist, ctx.config['benchmark']
    Writes: ctx._hr_bars ({bars dict, bm_bars, label_horizon}), or sets
             skip-flag on missing data
    """
    name = "LoadHourlyBarsForPanelTask"

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        from kernel.intraday import HourlyBarStore
        cfg = ctx.config.get("panel_ltr", {})
        if str(cfg.get("training_resolution", "daily")).lower() != "hourly":
            return False   # daily mode — Job is a no-op
        cache_dir = _resolve_cache_dir(
            cfg.get("hourly", {}).get("cache_dir", "data/intraday"),
            ctx.config,
        )
        store = HourlyBarStore(data_dir=cache_dir)
        bars: dict = {}
        for t in (ctx.watchlist or []):
            df = store.load(t)
            if df is not None and not df.empty:
                bars[t] = df
        if not bars:
            log.warning("LoadHourlyBarsForPanelTask: no hourly bars cached")
            return False
        benchmark = ctx.config.get("benchmark", "SPY")
        bm = bars.pop(benchmark, None) or store.load(benchmark)
        ctx._hr_bars = {
            "bars": bars, "bm_bars": bm,
            "label_horizon": int(cfg.get("hourly", {}).get("label_horizon_bars", 7)),
        }


# ── 2. Build hourly panel ─────────────────────────────────────────────────

class AssembleHourlyPanelTask(PanelTask):
    """Call build_hourly_resolution_panel to produce the (ticker, datetime)
    indexed panel.

    Reads:  ctx._hr_bars
    Writes: ctx._hr_panel (None if empty/error)
    """
    name = "AssembleHourlyPanelTask"

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        from .hourly_resolution_panel import build_hourly_resolution_panel
        h = getattr(ctx, "_hr_bars", None)
        if h is None:
            return False
        panel = build_hourly_resolution_panel(
            h["bars"],
            label_horizon_bars=h["label_horizon"],
            benchmark_bars=h["bm_bars"],
            apply_wash=True,
        )
        if panel is None or panel.empty:
            log.warning("AssembleHourlyPanelTask: empty panel")
            return False
        ctx._hr_panel = panel.reset_index()


# ── 3. Normalize schema (rename, derive date/hour, drop NaN labels) ────────

class NormalizeHourlySchemaTask(PanelTask):
    """Rename level_1→datetime, derive date+hour, rename label, drop NaN.

    Reads:  ctx._hr_panel
    Writes: ctx._hr_panel (in place)
    """
    name = "NormalizeHourlySchemaTask"

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        panel = getattr(ctx, "_hr_panel", None)
        if panel is None:
            return False
        if "level_1" in panel.columns:
            panel = panel.rename(columns={"level_1": "datetime"})
        if "datetime" in panel.columns:
            dt_col = panel["datetime"]
        else:
            dt_col = pd.to_datetime(panel.iloc[:, 1])
            panel["datetime"] = dt_col
        panel["date"] = pd.to_datetime(dt_col).dt.normalize()
        panel["hour"] = pd.to_datetime(dt_col).dt.hour
        if "forward_excess_return" in panel.columns:
            panel = panel.rename(columns={"forward_excess_return": "label"})
        if "label" in panel.columns:
            panel = panel[panel["label"].notna()].reset_index(drop=True)
        ctx._hr_panel = panel


# ── 4. Broadcast macro frame onto hourly rows ─────────────────────────────

class BroadcastMacroToHourlyTask(PanelTask):
    """Merge macro_factor_frame onto hourly panel by date; ffill within ticker.

    Reads:  ctx._hr_panel, ctx.macro_factor_frame
    Writes: ctx._hr_panel (with macro cols appended)
    """
    name = "BroadcastMacroToHourlyTask"

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        macro = ctx.macro_factor_frame
        if macro is None or macro.empty:
            return
        panel = getattr(ctx, "_hr_panel", None)
        if panel is None:
            return False
        if not isinstance(macro.index, pd.DatetimeIndex):
            macro = macro.copy()
            macro.index = pd.to_datetime(macro.index)
        macro_cols = list(macro.columns)
        collisions = [c for c in macro_cols if c in panel.columns]
        if collisions:
            log.warning("BroadcastMacroToHourlyTask: collision rename — %s",
                         collisions)
            rename = {c: f"{c}_macro" for c in collisions}
            macro = macro.rename(columns=rename)
            macro_cols = [rename.get(c, c) for c in macro_cols]
        panel = panel.merge(macro, left_on="date", right_index=True, how="left")
        panel[macro_cols] = panel.groupby(
            "ticker", group_keys=False,
        )[macro_cols].ffill().fillna(0.0)
        log.info("BroadcastMacroToHourlyTask: broadcast %d macro features",
                  len(macro_cols))
        ctx._hr_panel = panel


# ── 5. Finalize: feature_cols + dtype filter + commit ─────────────────────

class FinalizeHourlyPanelTask(PanelTask):
    """Compute feature_cols (excluding non-numeric, label, datetime), commit.

    Bug #24 (TRANSFORMER-TIMESTAMP-LEAK 2026-04-26 round-7): EXCLUDE
    every non-numeric or bool column regardless of name (defensive
    against future schema additions like 'timestamp').

    Reads:  ctx._hr_panel
    Writes: ctx.panel, ctx.feature_cols
    """
    name = "FinalizeHourlyPanelTask"

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        panel = getattr(ctx, "_hr_panel", None)
        if panel is None:
            return False
        non_feature = {"ticker", "date", "hour", "datetime", "timestamp",
                       "label", "_sample_weight", "forward_excess_return"}
        candidates = [c for c in panel.columns if c not in non_feature]
        ctx.feature_cols = [
            c for c in candidates
            if pd.api.types.is_numeric_dtype(panel[c].dtype)
            and not pd.api.types.is_bool_dtype(panel[c].dtype)
        ]
        ctx.panel = panel
        log.info("FinalizeHourlyPanelTask: panel=%d rows  features=%d",
                  len(panel), len(ctx.feature_cols))


# ── Job orchestrator ──────────────────────────────────────────────────────

class BuildHourlyResolutionPanelJob(PanelJob):
    """Replaces the ~160-line BuildHourlyResolutionPanelTask monolith."""
    name = "BuildHourlyResolutionPanelJob"

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        # No-op when daily mode (matches old Task's first-line skip).
        cfg = ctx.config.get("panel_ltr", {})
        return str(cfg.get("training_resolution", "daily")).lower() != "hourly"

    @property
    def tasks(self) -> list[PanelTask]:
        return [
            LoadHourlyBarsForPanelTask(),
            AssembleHourlyPanelTask(),
            NormalizeHourlySchemaTask(),
            BroadcastMacroToHourlyTask(),
            FinalizeHourlyPanelTask(),
        ]


__all__ = [
    "LoadHourlyBarsForPanelTask",
    "AssembleHourlyPanelTask",
    "NormalizeHourlySchemaTask",
    "BroadcastMacroToHourlyTask",
    "FinalizeHourlyPanelTask",
    "BuildHourlyResolutionPanelJob",
]
