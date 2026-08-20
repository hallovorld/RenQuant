"""PanelTrainingPipeline — parallel panel-LTR training pipeline for renquant_103.

Mirrors the Job/Task/TickerJob architecture used by
kernel/pipeline/pp_training.py: each logical phase is one Job, with
atomic steps expressed as Tasks chained inside it.

Job structure::

    PanelDataJob            (global, sequential tasks)
      ├─ FetchOHLCVTask
      └─ SectorMomentumTask

    PanelFeatureJob         (orchestrator — parallel per-ticker chain)
      └─ run_panel_ticker_parallel(
              TickerPanelFeatureJob →
              TickerPanelNeutralizeJob →
              TickerPanelFactorJob
         )

    PanelAssemblyJob        (global, sequential tasks)
      ├─ FactorZScoreTask
      ├─ LabelsTask
      └─ BuildPanelTask

    PanelModelJob           (global, sequential tasks)
      ├─ CrossValidateTask
      ├─ FinalFitTask
      └─ SaveArtifactTask
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout, as_completed
from pathlib import Path
from threading import current_thread

import numpy as np
import pandas as pd

from .context import PanelTrainingContext, TickerPanelContext

log = logging.getLogger("training_panel.pipeline")


# ── Task / Job ABCs (self-contained — same shape as pp_training.py) ───────────

class PanelTask(ABC):
    """Atomic step inside a PanelJob. Runs sequentially."""

    @abstractmethod
    def run(self, ctx: PanelTrainingContext) -> None: ...

    @property
    def name(self) -> str:
        return type(self).__name__


class PanelJob(ABC):
    """Global panel-training stage. Default run() drives a Task chain."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        return False

    @property
    def tasks(self) -> list[PanelTask]:
        return []

    def run(self, ctx: PanelTrainingContext) -> None:
        for task in self.tasks:
            task.run(ctx)


class PanelTickerJob(ABC):
    """Per-ticker stage — reads/writes TickerPanelContext only."""

    @abstractmethod
    def run(self, tc: TickerPanelContext) -> None: ...


# ── Parallel runner ───────────────────────────────────────────────────────────

def _resolve_workers(config_value: "int | None", item_count: int) -> int:
    import os
    if config_value is not None and config_value > 0:
        n = int(config_value)
    else:
        n = max(1, (os.cpu_count() or 4) - 2)
    return min(n, item_count)


def _run_panel_ticker_chain(tc: TickerPanelContext) -> None:
    """Sequential per-ticker chain: Feature → Neutralize → Factor."""
    tag = f"[{tc.ticker}|{current_thread().name}]"
    t0 = time.monotonic()
    TickerPanelFeatureJob().run(tc)
    if tc.feature_frame is None or tc.feature_frame.empty:
        log.warning("%s Feature produced no frame — skipping chain", tag)
        return
    TickerPanelNeutralizeJob().run(tc)
    TickerPanelFactorJob().run(tc)
    log.debug("%s chain DONE  %.2fs", tag, time.monotonic() - t0)


def run_panel_ticker_parallel(
    ticker_ctxs: list[TickerPanelContext],
    max_workers: "int | None" = None,
    timeout_seconds: "float | None" = None,
) -> None:
    if not ticker_ctxs:
        return
    cfg = ticker_ctxs[0].config or {}
    if max_workers is None:
        max_workers = cfg.get("parallel_workers")
    if timeout_seconds is None:
        timeout_seconds = cfg.get("parallel_ticker_timeout_seconds")
    n = _resolve_workers(max_workers, len(ticker_ctxs))
    log.info("run_panel_ticker_parallel: %d tickers, %d workers, timeout=%s",
             len(ticker_ctxs), n, timeout_seconds)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n, thread_name_prefix="panel") as ex:
        futures = {ex.submit(_run_panel_ticker_chain, tc): tc.ticker for tc in ticker_ctxs}
        for fut in as_completed(futures, timeout=None):
            ticker = futures[fut]
            try:
                fut.result(timeout=timeout_seconds)
            except _FutTimeout:
                log.error("[%s] panel chain TIMEOUT after %ss — skipped",
                          ticker, timeout_seconds)
                fut.cancel()
            except Exception as e:
                log.error("[%s] panel chain ERROR — %s: %s",
                          ticker, type(e).__name__, e)
    log.info("run_panel_ticker_parallel: DONE  %.1fs  (%d tickers)",
             time.monotonic() - t0, len(ticker_ctxs))


# ── Phase 1 — PanelDataJob + tasks ───────────────────────────────────────────

class FetchOHLCVTask(PanelTask):
    """Fetch watchlist + benchmark + sector ETFs into ctx.ohlcv."""

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.ohlcv:
            log.info("FetchOHLCVTask: ohlcv already populated — skipping")
            return
        from kernel.data import fetch_ohlcv, resolve_sample_end

        cfg = ctx.config
        start = cfg.get("sample_start")
        end   = resolve_sample_end(cfg)
        benchmark = cfg.get("benchmark", "SPY")
        sector_etf_map = cfg.get("sector_etf_map", {})

        all_syms = sorted(
            set(ctx.watchlist) | set(sector_etf_map.values()) | {benchmark}
        )
        log.info("FetchOHLCVTask: fetching %d tickers", len(all_syms))
        for sym in all_syms:
            try:
                df = fetch_ohlcv(sym, start=start, end=end)
            except Exception as exc:
                log.warning("  %s: fetch failed — %s", sym, exc)
                continue
            if df is None or df.empty:
                log.warning("  %s: empty", sym)
                continue
            ctx.ohlcv[sym] = df

        ctx.sector_etf_ohlcv = {
            sec: ctx.ohlcv[etf] for sec, etf in sector_etf_map.items()
            if etf in ctx.ohlcv
        }
        log.info("FetchOHLCVTask: loaded %d / %d  sectors=%d",
                 len(ctx.ohlcv), len(all_syms), len(ctx.sector_etf_ohlcv))


class SectorMomentumTask(PanelTask):
    """Compute per-sector momentum frames once — reused by per-ticker neutralization."""

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.sector_momentum:
            log.info("SectorMomentumTask: already populated — skipping")
            return
        if not ctx.sector_etf_ohlcv:
            log.warning("SectorMomentumTask: no sector_etf_ohlcv — skipping")
            return
        from training_panel.neutralization import compute_sector_momentum
        ctx.sector_momentum = compute_sector_momentum(ctx.sector_etf_ohlcv)
        log.info("SectorMomentumTask: %d sector frames", len(ctx.sector_momentum))


class PanelDataJob(PanelJob):
    """Phase 1 — gather market data + sector momentum.

    Task chain: FetchOHLCV → SectorMomentum
    """

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        return bool(ctx.ohlcv) and bool(ctx.sector_momentum)

    @property
    def tasks(self) -> list[PanelTask]:
        return [FetchOHLCVTask(), SectorMomentumTask()]


# ── Phase 2 — PanelFeatureJob (orchestrator) + per-ticker Jobs ───────────────

class PanelFeatureJob(PanelJob):
    """Phase 2 — parallel per-ticker chain (Feature → Neutralize → Factor)."""

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        if ctx.feature_frames and ctx.neutralized_frames and ctx.raw_factor_frames:
            log.info("PanelFeatureJob: per-ticker outputs already populated — skipping")
            return True
        return False

    def run(self, ctx: PanelTrainingContext) -> None:
        ticker_ctxs = [
            TickerPanelContext(
                ticker=t,
                ohlcv=ctx.ohlcv,
                sector_momentum=ctx.sector_momentum,
                ticker_sectors=ctx.ticker_sectors,
                config=ctx.config,
            )
            for t in ctx.watchlist if t in ctx.ohlcv
        ]
        log.info("PanelFeatureJob: launching parallel chain for %d tickers", len(ticker_ctxs))
        run_panel_ticker_parallel(ticker_ctxs)

        ctx.feature_frames = {
            tc.ticker: tc.feature_frame for tc in ticker_ctxs
            if tc.feature_frame is not None
        }
        ctx.neutralized_frames = {
            tc.ticker: tc.neutralized_frame for tc in ticker_ctxs
            if tc.neutralized_frame is not None
        }
        ctx.raw_factor_frames = {
            tc.ticker: tc.raw_factor_frame for tc in ticker_ctxs
            if tc.raw_factor_frame is not None
        }
        log.info("PanelFeatureJob: feat=%d  neutralized=%d  raw_factors=%d",
                 len(ctx.feature_frames), len(ctx.neutralized_frames),
                 len(ctx.raw_factor_frames))


class TickerPanelFeatureJob(PanelTickerJob):
    """Reuse training.features.build_training_features for one ticker."""

    def run(self, tc: TickerPanelContext) -> None:
        from training.features import build_training_features
        cfg = tc.config
        mp = cfg.get("model_params", {})
        try:
            tc.feature_frame = build_training_features(
                tc.ticker, tc.ohlcv,
                cfg.get("indicator_spec", {}),
                int(mp.get("lookahead", 5)),
                float(mp.get("threshold", 0.03)),
            )
        except Exception as exc:
            log.error("  %s: TickerPanelFeatureJob failed — %s", tc.ticker, exc)


class TickerPanelNeutralizeJob(PanelTickerJob):
    """Residualize per-ticker momentum/trend against sector-ETF momentum."""

    def run(self, tc: TickerPanelContext) -> None:
        if tc.feature_frame is None or tc.feature_frame.empty:
            return
        from training_panel.neutralization import NEUTRALIZE_COLS, neutralize_features
        cfg = tc.config.get("panel_ltr", {})
        if not cfg.get("neutralize_features", True):
            tc.neutralized_frame = tc.feature_frame.copy()
            return
        if not tc.sector_momentum or tc.ticker not in tc.ticker_sectors:
            tc.neutralized_frame = tc.feature_frame.copy()
            return
        try:
            neutralized = neutralize_features(
                {tc.ticker: tc.feature_frame},
                tc.sector_momentum,
                {tc.ticker: tc.ticker_sectors[tc.ticker]},
                cols=NEUTRALIZE_COLS,
                rolling_window=int(cfg.get("neutralize_rolling_window", 252)),
                expanding_warmup_days=int(cfg.get("neutralize_warmup_days", 252)),
            )
            tc.neutralized_frame = neutralized.get(tc.ticker, tc.feature_frame.copy())
        except Exception as exc:
            log.error("  %s: TickerPanelNeutralizeJob failed — %s", tc.ticker, exc)
            tc.neutralized_frame = tc.feature_frame.copy()


class TickerPanelFactorJob(PanelTickerJob):
    """Compute raw factor bundle for one ticker (unscaled; z-score is global)."""

    def run(self, tc: TickerPanelContext) -> None:
        benchmark = tc.config.get("benchmark", "SPY")
        if benchmark not in tc.ohlcv or tc.ticker not in tc.ohlcv:
            return
        cfg = tc.config.get("panel_ltr", {})
        mom_window  = int(cfg.get("factor_mom_window", 252))
        skip        = int(cfg.get("factor_skip", 21))
        beta_window = int(cfg.get("beta_window", 60))

        from training_panel.factors import (
            compute_size_feature, compute_momentum_12_1,
            compute_rolling_beta, compute_residual_momentum,
        )
        try:
            one = {tc.ticker: tc.ohlcv[tc.ticker]}
            size = compute_size_feature(one, None).get(tc.ticker)
            mom  = compute_momentum_12_1(one, mom_window=mom_window, skip=skip).get(tc.ticker)
            beta = compute_rolling_beta(one, tc.ohlcv[benchmark], window=beta_window).get(tc.ticker)
            rmom = compute_residual_momentum(
                one, tc.ohlcv[benchmark], window=beta_window,
                mom_window=mom_window, skip=skip,
            ).get(tc.ticker)
            idx = tc.ohlcv[tc.ticker].index
            tc.raw_factor_frame = pd.DataFrame({
                "size":      (size if size is not None else pd.Series(index=idx)).reindex(idx),
                "mom_12_1":  (mom  if mom  is not None else pd.Series(index=idx)).reindex(idx),
                "beta_60d":  (beta if beta is not None else pd.Series(index=idx)).reindex(idx),
                "resid_mom": (rmom if rmom is not None else pd.Series(index=idx)).reindex(idx),
            }, index=idx)
        except Exception as exc:
            log.error("  %s: TickerPanelFactorJob failed — %s", tc.ticker, exc)


# ── Phase 3 — PanelAssemblyJob + tasks ───────────────────────────────────────

class FactorZScoreTask(PanelTask):
    """Cross-sectional z-score of raw factor frames, per date."""

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.factor_frames:
            return
        if not ctx.raw_factor_frames:
            return
        from training_panel.factors import cross_sectional_zscore

        raw_cols = ["size", "mom_12_1", "beta_60d", "resid_mom"]
        per_col: dict[str, dict[str, pd.Series]] = {}
        for col in raw_cols:
            per_col[col] = {
                t: df[col] for t, df in ctx.raw_factor_frames.items()
                if col in df.columns
            }
        z = {col: cross_sectional_zscore(per_col[col]) for col in raw_cols}

        out: dict[str, pd.DataFrame] = {}
        for t, raw in ctx.raw_factor_frames.items():
            idx = raw.index
            out[t] = pd.DataFrame({
                "size_z":      z["size"].get(t,      pd.Series(index=idx)).reindex(idx),
                "mom_12_1_z":  z["mom_12_1"].get(t,  pd.Series(index=idx)).reindex(idx),
                "beta_60d_z":  z["beta_60d"].get(t,  pd.Series(index=idx)).reindex(idx),
                "resid_mom_z": z["resid_mom"].get(t, pd.Series(index=idx)).reindex(idx),
            }, index=idx)
        ctx.factor_frames = out
        log.info("FactorZScoreTask: z-scored %d factor frames", len(out))


class LabelsTask(PanelTask):
    """Forward returns → purged β-neutral residuals → cross-sectional Gaussianize."""

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.labels:
            return
        from training_panel.labels import build_labels

        cfg = ctx.config.get("panel_ltr", {})
        lookahead   = int(cfg.get("lookahead_days", 5))
        beta_window = int(cfg.get("beta_window", 60))
        benchmark   = ctx.config.get("benchmark", "SPY")

        spy_df = ctx.ohlcv.get(benchmark)
        if spy_df is None:
            raise RuntimeError("LabelsTask: benchmark OHLCV missing")

        spy_close = spy_df["close"].astype(float)
        spy_fwd = spy_close.shift(-lookahead) / spy_close - 1.0

        fwd_returns: dict[str, pd.Series] = {}
        for t, df in ctx.ohlcv.items():
            if t not in ctx.watchlist:
                continue
            c = df["close"].astype(float)
            fwd_returns[t] = c.shift(-lookahead) / c - 1.0

        sec_fwd_frames: dict[str, pd.Series] = {}
        for sec, df in ctx.sector_etf_ohlcv.items():
            c = df["close"].astype(float)
            sec_fwd_frames[sec] = c.shift(-lookahead) / c - 1.0
        sec_fwd_by_ticker = {
            t: sec_fwd_frames[sec] for t, sec in ctx.ticker_sectors.items()
            if sec in sec_fwd_frames
        }

        ctx.labels = build_labels(
            fwd_returns, spy_fwd, sec_fwd_by_ticker,
            beta_window=beta_window, lookahead_days=lookahead,
        )
        log.info("LabelsTask: built labels for %d tickers", len(ctx.labels))


class BuildPanelTask(PanelTask):
    """Assemble long-form panel DataFrame + group_sizes array."""

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.panel is not None:
            return
        from training_panel.panel_frame import build_panel_frame

        cfg = ctx.config.get("panel_ltr", {})
        min_history = int(cfg.get("min_history_days", 252))
        lookahead   = int(cfg.get("lookahead_days", 5))
        age_warmup  = int(cfg.get("age_warmup_days", 504))
        nan_cols    = list(cfg.get("nan_prone_cols", []))

        ff = ctx.neutralized_frames
        lab = ctx.labels
        fac = ctx.factor_frames
        sec = ctx.ticker_sectors

        ff_wl  = {t: ff[t]  for t in ctx.watchlist if t in ff}
        lab_wl = {t: lab[t] for t in ctx.watchlist if t in lab}
        sec_wl = {t: sec[t] for t in ctx.watchlist if t in sec}
        fac_wl = {t: fac[t] for t in ctx.watchlist if t in fac}

        panel, group_sizes, meta = build_panel_frame(
            ff_wl, lab_wl, sec_wl,
            factor_frames=fac_wl,
            listing_dates=ctx.listing_dates,
            min_history_days=min_history,
            lookahead_days=lookahead,
            age_warmup_days=age_warmup,
            nan_prone_cols=nan_cols,
        )
        label_mask = panel["label"].notna()
        panel = panel[label_mask].reset_index(drop=True)
        group_sizes = panel.groupby("date", sort=True).size().values.astype(np.int32)

        exclude = {"date", "ticker", "sector", "label",
                   "weight", "weight_concurrency", "weight_age"}
        feature_cols = [c for c in panel.columns if c not in exclude]

        ctx.panel = panel
        ctx.group_sizes = group_sizes
        ctx.panel_metadata = meta
        ctx.feature_cols = feature_cols
        log.info("BuildPanelTask: panel=%d rows  features=%d  tickers=%d  dates=%d",
                 len(panel), len(feature_cols), meta.get("n_tickers"), meta.get("n_dates"))


class PanelAssemblyJob(PanelJob):
    """Phase 3 — turn per-ticker outputs into a panel DataFrame.

    Task chain: FactorZScore → Labels → BuildPanel
    """

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        return ctx.panel is not None and bool(ctx.feature_cols)

    @property
    def tasks(self) -> list[PanelTask]:
        return [FactorZScoreTask(), LabelsTask(), BuildPanelTask()]


# ── Phase 4 — PanelModelJob + tasks ──────────────────────────────────────────

class CrossValidateTask(PanelTask):
    """Purged K-fold Spearman IC over the assembled panel."""

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.cv_result:
            return
        from training_panel.purged_cv import PurgedKFold, cross_validated_ic
        from training_panel.ltr_model import PanelLTRModel

        cfg = ctx.config.get("panel_ltr", {})
        cv_splits  = int(cfg.get("cv_n_splits", 5))
        embargo    = int(cfg.get("cv_embargo_days", cfg.get("lookahead_days", 5)))
        lookahead  = int(cfg.get("lookahead_days", 5))
        num_rounds = int(cfg.get("num_boost_round", 400))
        xgb_params = dict(cfg.get("xgb_params", {}))

        panel = ctx.panel
        feature_cols = ctx.feature_cols

        class _SklearnAdapter:
            def __init__(self):
                self._m = PanelLTRModel(params=xgb_params)
            def fit(self, X, y, sample_weight=None):
                df = X.copy()
                df["label"] = y
                df["date"] = panel.loc[X.index, "date"].values
                df["weight"] = sample_weight if sample_weight is not None else 1.0
                df = df.sort_values(["date"], kind="mergesort").reset_index(drop=True)
                gs = df.groupby("date", sort=True).size().values.astype(np.int32)
                self._m.train(
                    df, gs, feature_cols=list(X.columns),
                    label_col="label", weight_col="weight",
                    num_boost_round=max(num_rounds // 2, 50),
                )
            def predict(self, X):
                return self._m.predict(X.copy()).values

        cv = PurgedKFold(n_splits=cv_splits, embargo_days=embargo,
                         lookahead_days=lookahead)
        ctx.cv_result = cross_validated_ic(
            _SklearnAdapter, panel, feature_cols, "label", cv,
            weight_col="weight",
        )
        log.info("CrossValidateTask: mean_ic=%+.4f  std=%.4f",
                 ctx.cv_result["mean_ic"], ctx.cv_result["std_ic"])


class FinalFitTask(PanelTask):
    """Fit the final PanelLTRModel on the full panel."""

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.final_model is not None:
            return
        from training_panel.ltr_model import PanelLTRModel

        cfg = ctx.config.get("panel_ltr", {})
        xgb_params = dict(cfg.get("xgb_params", {}))
        num_rounds = int(cfg.get("num_boost_round", 400))

        model = PanelLTRModel(params=xgb_params)
        fit = model.train(
            ctx.panel, ctx.group_sizes,
            feature_cols=ctx.feature_cols,
            label_col="label", weight_col="weight",
            num_boost_round=num_rounds,
        )
        ctx.final_model = model
        ctx._final_fit = fit  # noqa: SLF001 — read by SaveArtifactTask
        log.info("FinalFitTask: train_ic=%+.4f", fit.get("train_ic", 0.0))


class SaveArtifactTask(PanelTask):
    """Write the JSON artifact with CV metadata + populate ctx.summary."""

    def run(self, ctx: PanelTrainingContext) -> None:
        cfg = ctx.config.get("panel_ltr", {})

        out_path = Path(cfg.get("artifact_path", "panel-ltr.json"))
        if ctx.strategy_dir and not out_path.is_absolute():
            out_path = ctx.strategy_dir / "artifacts" / out_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fit = getattr(ctx, "_final_fit", {}) or {}
        meta = {
            "panel_shape": {
                "rows":    int(ctx.panel_metadata.get("n_rows", len(ctx.panel))),
                "tickers": int(ctx.panel_metadata.get("n_tickers", 0)),
                "dates":   int(ctx.panel_metadata.get("n_dates", 0)),
            },
            "oos_mean_ic":     ctx.cv_result["mean_ic"],
            "oos_std_ic":      ctx.cv_result["std_ic"],
            "oos_per_fold_ic": ctx.cv_result["per_fold_ic"],
            "training_train_ic": fit.get("train_ic", 0.0),
            "training_notes":  cfg.get("training_notes", "Stage-1 panel pipeline"),
            "neutralize_features": cfg.get("neutralize_features", True),
            "lookahead_days":  cfg.get("lookahead_days", 5),
            "beta_window":     cfg.get("beta_window", 60),
            "min_history_days": cfg.get("min_history_days", 252),
            "cv_n_splits":     cfg.get("cv_n_splits", 5),
            "cv_embargo_days": cfg.get("cv_embargo_days", cfg.get("lookahead_days", 5)),
        }
        ctx.final_model.save(out_path, metadata=meta)
        ctx.artifact_path = out_path
        ctx.summary = {
            "mean_ic":        ctx.cv_result["mean_ic"],
            "per_fold_ic":    ctx.cv_result["per_fold_ic"],
            "artifact_path":  str(out_path),
            "panel_metadata": ctx.panel_metadata,
            "feature_cols":   ctx.feature_cols,
        }
        log.info("SaveArtifactTask: artifact → %s", out_path)


class PanelModelJob(PanelJob):
    """Phase 4 — cross-validate, fit final, save artifact.

    Task chain: CrossValidate → FinalFit → SaveArtifact
    """

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        return ctx.final_model is not None and ctx.artifact_path is not None

    @property
    def tasks(self) -> list[PanelTask]:
        return [CrossValidateTask(), FinalFitTask(), SaveArtifactTask()]


# ── Orchestrator ─────────────────────────────────────────────────────────────

class PanelTrainingPipeline:
    """Four-phase panel-LTR training pipeline."""

    def run(self, ctx: PanelTrainingContext) -> PanelTrainingContext:
        jobs: list[PanelJob] = [
            PanelDataJob(),       # FetchOHLCV + SectorMomentum
            PanelFeatureJob(),    # parallel per-ticker chain
            PanelAssemblyJob(),   # FactorZScore + Labels + BuildPanel
            PanelModelJob(),      # CrossValidate + FinalFit + SaveArtifact
        ]
        t0 = time.monotonic()
        log.info("PanelTrainingPipeline START  watchlist=%d", len(ctx.watchlist))
        for job in jobs:
            if job.should_skip(ctx):
                continue
            t1 = time.monotonic()
            log.info("── %s START", job.name)
            job.run(ctx)
            log.info("── %s DONE  %.1fs", job.name, time.monotonic() - t1)
        log.info("PanelTrainingPipeline DONE  total=%.1fs", time.monotonic() - t0)
        return ctx
