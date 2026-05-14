"""FullTrainingPipeline — orchestrates end-to-end renquant_104 retraining.

Wraps the three existing sub-pipelines (per-ticker tournament, panel-LTR,
score recalibration) into a single Job/Task chain so the whole flow is
expressible as one `FullTrainingPipeline().run(ctx)` call.

Phase layout::

    FullTrainingPipeline
      ├─ BaselineTournamentJob     wraps TrainingPipeline
      │    └─ RunBaselineTask
      │
      ├─ PanelTrainingJob          wraps PanelTrainingPipeline
      │    ├─ FetchPanelDataTask
      │    ├─ BuildPanelFeatureFramesTask
      │    └─ RunPanelTrainingTask
      │
      └─ RecalibrationJob          wraps scripts.recalibrate_scores.recalibrate
           └─ RunRecalibrationTask

All Jobs read/write `FullTrainingContext`. Tasks short-circuit by returning
False (same semantics as `kernel/pipeline/pipeline.py`).
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("kernel.pipeline.full_training")


# ── Context ──────────────────────────────────────────────────────────────────

@dataclass
class FullTrainingContext:
    """Shared state for the three end-to-end training phases."""
    config:        dict[str, Any]
    strategy:      str
    strategy_dir:  Path

    # populated by BaselineTournamentJob
    baseline_exported: list[str] = field(default_factory=list)
    baseline_ttl_skipped: list[str] = field(default_factory=list)

    # populated by PanelTrainingJob
    ohlcv_all:        dict[str, pd.DataFrame] = field(default_factory=dict)
    feature_frames:   dict[str, pd.DataFrame] = field(default_factory=dict)
    panel_summary:    dict[str, Any] | None   = None

    # populated by RecalibrationJob
    recalibrated:     bool = False

    # phase toggles — set by orchestrator from CLI flags
    skip_baseline:     bool = False
    skip_panel:        bool = False
    skip_recalibrate:  bool = False

    # Cadence override: force a full run even if today doesn't match the
    # configured weekday (used by the dedicated Sunday script + manual runs).
    force_retrain:     bool = False


# ── Task / Job ABCs ─────────────────────────────────────────────────────────

class FullTrainingTask(ABC):
    """Atomic step. Return False to short-circuit the enclosing Job's chain."""

    @abstractmethod
    def run(self, ctx: FullTrainingContext) -> "bool | None": ...

    @property
    def name(self) -> str:
        return type(self).__name__


class FullTrainingJob(ABC):
    """Phase-level Job. Default run() drives a sequential Task chain."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def should_skip(self, ctx: FullTrainingContext) -> bool:
        return False

    @property
    def tasks(self) -> list[FullTrainingTask]:
        return []

    def run(self, ctx: FullTrainingContext) -> None:
        for task in self.tasks:
            if task.run(ctx) is False:
                log.debug("[%s] chain stopped by %s", self.name, task.name)
                return


# ── Phase 1 — Baseline tournament ────────────────────────────────────────────

class RunBaselineTask(FullTrainingTask):
    """Run the per-ticker TrainingPipeline (tournament + export + correlation)."""

    def run(self, ctx: FullTrainingContext) -> "bool | None":
        from kernel.pipeline.pp_training import TrainingPipeline, TrainingContext  # noqa: PLC0415

        cfg = dict(ctx.config)
        cfg["_strategy_dir"]   = str(ctx.strategy_dir)
        # Propagate --force down to per-ticker TTL check (see pp_training.py
        # `_model_is_fresh`). When force_retrain is True, TTL is bypassed
        # and every ticker retrains.
        cfg["_force_retrain"]  = bool(ctx.force_retrain)
        tctx = TrainingContext(config=cfg)
        TrainingPipeline().run(tctx)
        ctx.baseline_exported = list(tctx.exported)
        ctx.baseline_ttl_skipped = list(tctx.ttl_skipped)
        log.info("RunBaselineTask: exported=%d  ttl_skipped=%d",
                 len(ctx.baseline_exported), len(ctx.baseline_ttl_skipped))


class BaselineTournamentJob(FullTrainingJob):
    """Phase 1 — per-ticker tournament, export, and correlation artifact."""

    def should_skip(self, ctx: FullTrainingContext) -> bool:
        return ctx.skip_baseline

    @property
    def tasks(self) -> list[FullTrainingTask]:
        return [RunBaselineTask()]


# ── Phase 2 — Panel training ────────────────────────────────────────────────

class FetchPanelDataTask(FullTrainingTask):
    """Fetch OHLCV for watchlist ∪ benchmark ∪ sector ETFs into ctx.ohlcv_all."""

    def run(self, ctx: FullTrainingContext) -> "bool | None":
        from kernel.data import fetch_ohlcv  # noqa: PLC0415

        config     = ctx.config
        watchlist  = config["watchlist"]
        benchmark  = config.get("benchmark", "SPY")
        provider   = config.get("data_src", "yfinance")
        sector_etf = config.get("sector_etf_map", {})

        needed = sorted(set(watchlist) | {benchmark} | set(sector_etf.values()))
        log.info("FetchPanelDataTask: fetching %d symbols …", len(needed))
        # Audit fix D-8 (2026-04-25): aggregate failure / empty counts +
        # report so the operator sees how many tickers actually entered
        # the panel. Pre-fix, silent shrinking left training on an
        # arbitrary universe with no signal.
        n_fail = 0
        n_empty = 0
        for sym in needed:
            try:
                df = fetch_ohlcv(sym, provider=provider)
            except Exception as exc:
                n_fail += 1
                log.warning("  %-6s fetch failed — %s", sym, exc)
                continue
            if df is None or df.empty:
                n_empty += 1
                log.warning("  %-6s empty/None", sym)
                continue
            ctx.ohlcv_all[sym] = df
        n_loaded = len(ctx.ohlcv_all)
        n_missing = len(needed) - n_loaded
        log.info(
            "FetchPanelDataTask: loaded %d / %d  (failed=%d empty=%d)",
            n_loaded, len(needed), n_fail, n_empty,
        )
        # Loud-fail when too much of the watchlist was lost — heuristic
        # threshold: tolerate up to 5% missing, otherwise stop the phase.
        watchlist_missing = sum(
            1 for s in watchlist if s not in ctx.ohlcv_all
        )
        if watchlist_missing > max(1, len(watchlist) // 20):
            log.error(
                "FetchPanelDataTask: %d / %d watchlist tickers missing OHLCV "
                "(>5%% — training would silently shrink). Aborting panel phase.",
                watchlist_missing, len(watchlist),
            )
            return False
        if benchmark not in ctx.ohlcv_all:
            log.error("FetchPanelDataTask: benchmark %s missing — stopping panel phase",
                      benchmark)
            return False


class BuildPanelFeatureFramesTask(FullTrainingTask):
    """Build per-ticker labelled feature frames into ctx.feature_frames."""

    def run(self, ctx: FullTrainingContext) -> "bool | None":
        from training.features import build_all_training_features  # noqa: PLC0415

        config         = ctx.config
        benchmark      = config.get("benchmark", "SPY")
        indicator_spec = config.get("indicator_spec", {})
        mp             = config.get("model_params", {})
        lookahead      = int(mp.get("lookahead", 5))
        threshold      = float(mp.get("threshold", 0.03))

        watchlist_present = [t for t in config["watchlist"] if t in ctx.ohlcv_all]
        features_in = {t: ctx.ohlcv_all[t] for t in watchlist_present}
        features_in[benchmark] = ctx.ohlcv_all[benchmark]

        ctx.feature_frames = build_all_training_features(
            watchlist=watchlist_present,
            ohlcv=features_in,
            indicator_spec=indicator_spec,
            lookahead=lookahead,
            threshold=threshold,
        )
        if not ctx.feature_frames:
            log.error("BuildPanelFeatureFramesTask: no feature frames built — stopping")
            return False
        log.info("BuildPanelFeatureFramesTask: %d feature frames built",
                 len(ctx.feature_frames))


class RunPanelTrainingTask(FullTrainingTask):
    """Run PanelTrainingPipeline with the prepared frames → writes panel-ltr.json."""

    def run(self, ctx: FullTrainingContext) -> "bool | None":
        from training_panel.context import PanelTrainingContext  # noqa: PLC0415
        from training_panel.pp_panel_training import PanelTrainingPipeline  # noqa: PLC0415

        config         = ctx.config
        benchmark      = config.get("benchmark", "SPY")
        sector_map     = config.get("sector_map", {})
        sector_etf_map = config.get("sector_etf_map", {})
        mp             = config.get("model_params", {})
        lookahead      = int(mp.get("lookahead", 5))

        sector_etf_ohlcv = {
            sec: ctx.ohlcv_all[etf]
            for sec, etf in sector_etf_map.items() if etf in ctx.ohlcv_all
        }
        ticker_sectors = {t: sector_map[t] for t in ctx.feature_frames if t in sector_map}

        panel_cfg = dict(config.get("panel_ltr", {}))
        panel_cfg.setdefault("lookahead_days",      lookahead)
        panel_cfg.setdefault("beta_window",         60)
        panel_cfg.setdefault("min_history_days",    252)
        panel_cfg.setdefault("age_warmup_days",     504)
        panel_cfg.setdefault("cv_n_splits",         5)
        panel_cfg.setdefault("cv_embargo_days",     lookahead)
        panel_cfg.setdefault("num_boost_round",     400)
        panel_cfg.setdefault("neutralize_features", True)
        panel_cfg.setdefault("nan_prone_cols",      [])
        panel_cfg.setdefault("xgb_params",          {})
        # 2026-05-11 sim/prod isolation: prod artifacts live in artifacts/prod/.
        # Default rebased so training without an explicit artifact_path
        # writes to the prod path the runner reads, not a flat orphan.
        panel_cfg.setdefault("artifact_path",       "artifacts/prod/panel-ltr.alpha158_fund.json")

        # §5.13.13 guard: if this is a side config (sim/research), refuse to
        # let it overwrite the production artifact via the default fallback.
        # Catches the footgun where a sim training run forgets to override
        # panel_ltr.artifact_path and silently corrupts the prod model.
        side_label = config.get("_side_config_label") or ""
        if side_label and "artifacts/prod/" in str(panel_cfg["artifact_path"]):
            raise ValueError(
                f"FullTrainingPipeline refusing to write to a prod artifact "
                f"path from a side config (label={side_label!r}). "
                f"Set panel_ltr.artifact_path to artifacts/sim/... explicitly "
                f"in your side config to avoid breaching sim/prod isolation."
            )

        artifact_out = Path(panel_cfg["artifact_path"])
        if not artifact_out.is_absolute():
            artifact_out = ctx.strategy_dir / artifact_out
        panel_cfg["artifact_path"] = str(artifact_out)
        artifact_out.parent.mkdir(parents=True, exist_ok=True)

        merged_cfg = dict(config)
        merged_cfg["panel_ltr"]     = panel_cfg
        merged_cfg["_strategy_dir"] = str(ctx.strategy_dir)

        spy_df = ctx.ohlcv_all[benchmark]
        watchlist_usable = list(ctx.feature_frames.keys())
        ohlcv_wl = {t: ctx.ohlcv_all[t] for t in watchlist_usable if t in ctx.ohlcv_all}

        # Audit fix D-4 (2026-04-25): populate listing_dates from each
        # ticker's first OHLCV bar so age-weighting is no longer dead
        # code. Pre-fix, listing_dates=None → compute_age_weight returned
        # weight=1.0 for everyone → newly-IPO'd tickers (RBLX, NVTS, MDB,
        # SOFI, SNOW, PLTR) got the same training weight as 30-yr
        # incumbents despite ~1/4 the history.
        listing_dates = {
            t: pd.Timestamp(df.index[0])
            for t, df in ohlcv_wl.items()
            if df is not None and not df.empty
        }
        pctx = PanelTrainingContext(
            config=merged_cfg,
            watchlist=watchlist_usable,
            ohlcv=dict(ohlcv_wl) | {benchmark: spy_df},
            sector_etf_ohlcv=sector_etf_ohlcv,
            ticker_sectors=ticker_sectors,
            listing_dates=listing_dates,
        )
        pctx.feature_frames = ctx.feature_frames
        # strategy_dir is a read-only property on PanelTrainingContext derived
        # from merged_cfg["_strategy_dir"] (set above) — no direct assignment.

        PanelTrainingPipeline().run(pctx)
        ctx.panel_summary = pctx.summary or {}
        log.info("RunPanelTrainingTask: mean_ic=%+.4f  artifact=%s",
                 ctx.panel_summary.get("mean_ic", 0.0),
                 ctx.panel_summary.get("artifact_path"))


class PanelTrainingJob(FullTrainingJob):
    """Phase 2 — fetch data → build feature frames → run panel-LTR pipeline."""

    def should_skip(self, ctx: FullTrainingContext) -> bool:
        return ctx.skip_panel

    @property
    def tasks(self) -> list[FullTrainingTask]:
        return [
            FetchPanelDataTask(),
            BuildPanelFeatureFramesTask(),
            RunPanelTrainingTask(),
        ]


# ── Phase 3 — Recalibration ──────────────────────────────────────────────────

class RunRecalibrationTask(FullTrainingTask):
    """Refresh per-symbol score calibrations + blend weights in strategy_config.json."""

    def run(self, ctx: FullTrainingContext) -> "bool | None":
        from scripts.recalibrate_scores import recalibrate  # noqa: PLC0415

        recalibrate(strategy=ctx.strategy, dry_run=False)
        # Config on disk was mutated — reload so later jobs see the new blend weights
        config_path = ctx.strategy_dir / "strategy_config.json"
        if config_path.exists():
            ctx.config = json.loads(config_path.read_text())
        ctx.recalibrated = True


class RecalibrationJob(FullTrainingJob):
    """Phase 3 — refresh blend weights + per-symbol calibrations."""

    def should_skip(self, ctx: FullTrainingContext) -> bool:
        return ctx.skip_recalibrate

    @property
    def tasks(self) -> list[FullTrainingTask]:
        return [RunRecalibrationTask()]


# ── Orchestrator ─────────────────────────────────────────────────────────────

def _cadence_allows_today(config: dict[str, Any], today_weekday: int) -> tuple[bool, str]:
    """Decide whether today's full retrain should run under the configured cadence.

    Returns (allowed, reason). `today_weekday` is Python's Monday=0 … Sunday=6.

    Config block under top-level "training":
        {
          "cadence": "daily" | "weekly" | "custom",   # default "daily"
          "weekly_weekday": 6,                         # used when cadence = "weekly"
          "allowed_weekdays": [1, 3, 6]                # used when cadence = "custom"
                                                       # Mon=0 … Sun=6
        }

    When the cadence gate blocks a run, the orchestrator short-circuits
    BEFORE touching baseline/panel jobs — no wasted fetches.
    """
    cfg = config.get("training", {})
    cadence = str(cfg.get("cadence", "daily")).strip().lower()
    if cadence == "daily":
        return True, "daily cadence"
    if cadence == "weekly":
        allowed_day = int(cfg.get("weekly_weekday", 6))  # Sunday default
        if today_weekday == allowed_day:
            return True, f"weekly cadence hit on weekday={today_weekday}"
        return False, (f"weekly cadence skip — today weekday={today_weekday}, "
                       f"configured weekly_weekday={allowed_day}")
    if cadence == "custom":
        raw = cfg.get("allowed_weekdays", [])
        try:
            allowed = {int(d) for d in raw}
        except (TypeError, ValueError):
            return True, f"malformed allowed_weekdays {raw!r} — defaulting to run"
        if not allowed:
            return True, "empty allowed_weekdays — defaulting to run"
        if today_weekday in allowed:
            return True, (f"custom cadence hit on weekday={today_weekday}, "
                          f"allowed={sorted(allowed)}")
        return False, (f"custom cadence skip — today weekday={today_weekday}, "
                       f"allowed_weekdays={sorted(allowed)}")
    # Unknown cadence → fail open (run) to avoid silent outage
    return True, f"unknown cadence {cadence!r} — defaulting to run"


class FullTrainingPipeline:
    """End-to-end renquant_104 retraining: tournament → panel-LTR → recalibrate."""

    def run(self, ctx: FullTrainingContext) -> FullTrainingContext:
        import datetime  # noqa: PLC0415
        allowed, reason = _cadence_allows_today(ctx.config, datetime.date.today().weekday())
        if not allowed and not ctx.force_retrain:
            log.info("FullTrainingPipeline SKIPPED — %s  (use --force to override)", reason)
            return ctx
        if not allowed and ctx.force_retrain:
            log.info("FullTrainingPipeline forced: %s", reason)
        else:
            log.debug("Cadence gate: %s", reason)

        jobs: list[FullTrainingJob] = [
            BaselineTournamentJob(),
            PanelTrainingJob(),
            RecalibrationJob(),
        ]
        t0 = time.monotonic()
        log.info("FullTrainingPipeline START  strategy=%s", ctx.strategy)
        for job in jobs:
            if job.should_skip(ctx):
                log.info("── %s SKIPPED", job.name)
                continue
            t1 = time.monotonic()
            log.info("── %s START", job.name)
            job.run(ctx)
            log.info("── %s DONE  %.1fs", job.name, time.monotonic() - t1)
        log.info("FullTrainingPipeline DONE  total=%.1fs", time.monotonic() - t0)
        return ctx
