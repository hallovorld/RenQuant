"""TrainingPipeline — parallel-per-ticker training pipeline for renquant_103.

Architecture:

    Phase 1  Global (sequential)
      DataFetchJob    fetch OHLCV for all tickers
      RegimeFitJob    fit GMM, build final_regime series

    Phase 2  Per-ticker parallel  (ThreadPoolExecutor, one TickerTrainingContext each)
      TickerFeatureJob    build labelled feature frame
      TickerTournamentJob train all model types, select best
      TickerExportJob     write models/ artifacts
      TickerCalibrationJob write score_calibration metadata

    Phase 3  Global (sequential)
      CorrelationJob  120-day return correlation artifact

The four per-ticker jobs run in sequence within each ticker's worker thread, so
ticker pipelines are fully independent and all four stages fire in one pass.
Notebook cells call the global orchestrator jobs (FeatureJob, TournamentJob,
ExportJob, CalibrationJob) whose run() dispatch to the parallel worker threads.

Usage from the notebook::

    ctx = TrainingContext(config=CONFIG, ohlcv=ohlcv)  # ohlcv pre-populated → DataFetchJob skipped
    DataFetchJob().run(ctx)      # skip if ohlcv already loaded
    RegimeFitJob().run(ctx)      # skip if final_regime already set
    FeatureJob().run(ctx)        # parallel: feature+tournament+export+calibrate per ticker
    CorrelationJob().run(ctx)    # save artifact

Or via TrainingPipeline::

    TrainingPipeline().run(ctx)  # runs all phases in order
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from threading import current_thread
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("training.pipeline")


# ── Global context ─────────────────────────────────────────────────────────────

@dataclass
class TrainingContext:
    """Shared state for global (cross-ticker) training jobs."""
    config: dict[str, Any]

    # populated by DataFetchJob (or pre-filled by caller)
    ohlcv: dict[str, pd.DataFrame] = field(default_factory=dict)

    # populated by RegimeFitJob
    hurst_series: pd.Series = field(default=None)
    cusum_series: pd.Series = field(default=None)
    changepoint_dates: pd.Index = field(default=None)
    final_regime: pd.Series = field(default=None)
    final_regime_conf: pd.Series = field(default=None)
    gmm: Any = field(default=None)

    # collected from TickerTrainingContexts after parallel phase
    feature_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    exported: list[str] = field(default_factory=list)
    calibration_summary: dict[str, Any] = field(default_factory=dict)
    # Tickers whose per-ticker retrain was skipped by the model-TTL gate
    # (see `_model_is_fresh` in this module). Surfaced by FeatureJob for
    # logging + notification bodies.
    ttl_skipped: list[str] = field(default_factory=list)

    # populated by CorrelationJob
    corr_matrix: pd.DataFrame = field(default=None)

    @property
    def spy_df(self) -> pd.DataFrame | None:
        return self.ohlcv.get("SPY")

    @property
    def watchlist(self) -> list[str]:
        return self.config["watchlist"]

    @property
    def strategy_dir(self) -> Path | None:
        _sd = self.config.get("_strategy_dir")
        return Path(_sd) if _sd else None


# ── Per-ticker context ─────────────────────────────────────────────────────────

@dataclass
class TickerTrainingContext:
    """Isolated per-ticker context for the parallel training phase.

    One instance per ticker; jobs write only to this object — never to
    TrainingContext.  Results are collected back after all threads complete.
    """
    ticker: str
    ohlcv: dict[str, pd.DataFrame]  # shared read-only reference
    config: dict[str, Any]
    strategy_dir: Path | None

    # outputs — written by per-ticker jobs
    feature_frame: pd.DataFrame | None = None
    result: dict | None = None
    exported: bool = False
    calibration: dict | None = None
    ttl_skipped: bool = False   # True when model-TTL gate skipped this ticker


def _model_params_for_tournament(config: dict[str, Any]) -> dict[str, Any]:
    """Return tournament model params with post-2026-05-10 defaults restored.

    ``model_params.lookahead`` was intentionally removed from the production
    config to stop it shadowing the panel-LTR horizon.  The per-ticker
    tournament still owns a separate short-horizon baseline, historically 5
    trading days.  Without this fallback, ``TickerFeatureJob`` swallowed a
    KeyError and every ticker produced no feature frame.
    """
    mp = dict(config.get("model_params", {}) or {})
    training_cfg = config.get("training", {}) or {}
    mp.setdefault("lookahead", int(training_cfg.get("tournament_lookahead_days", 5)))
    mp.setdefault("threshold", 0.03)
    return mp


# ── Task + Job ABCs ────────────────────────────────────────────────────────────

class TrainingTask(ABC):
    """Atomic step within a TrainingJob.

    run() returns None to continue the chain.  There is no short-circuit
    (training tasks never discard work — they either succeed or raise).
    """

    @abstractmethod
    def run(self, ctx: TrainingContext) -> None: ...

    @property
    def name(self) -> str:
        return type(self).__name__


class TrainingJob(ABC):
    """Global training pipeline stage.

    Override tasks() to define a sequential TrainingTask chain — the default
    run() drives the chain.  Jobs with non-linear flow (DataFetchJob,
    CorrelationJob, etc.) override run() directly.
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    def should_skip(self, ctx: TrainingContext) -> bool:
        return False

    @property
    def tasks(self) -> list[TrainingTask]:
        return []

    def run(self, ctx: TrainingContext) -> None:
        for task in self.tasks:
            task.run(ctx)


class TrainingTickerJob(ABC):
    """Per-ticker training stage — reads/writes TickerTrainingContext only."""

    @abstractmethod
    def run(self, tc: TickerTrainingContext) -> None: ...


# ── Parallel runner ────────────────────────────────────────────────────────────

def _model_is_fresh(
    ticker: str,
    strategy_dir: "Path | None",
    ttl_days: int,
    today: "date | None" = None,
) -> "tuple[bool, str]":
    """Return (fresh, reason) — True when a per-ticker retrain can be skipped.

    Reads `{strategy_dir}/models/{TICKER}/{TICKER}-policy-metadata.json`,
    parses `trained_date`, and compares to today. Any parse failure → not
    fresh (fail open → retrain).
    """
    if ttl_days <= 0 or strategy_dir is None:
        return False, "ttl disabled"
    mp = strategy_dir / "models" / ticker / f"{ticker}-policy-metadata.json"
    if not mp.exists():
        return False, "no existing model"
    try:
        meta = json.loads(mp.read_text())
        td = meta.get("trained_date")
        if not td:
            return False, "no trained_date"
        trained = date.fromisoformat(td)
    except Exception as exc:
        return False, f"metadata parse failed: {exc}"
    today = today or date.today()
    age_days = (today - trained).days
    if age_days <= ttl_days:
        return True, f"fresh (age={age_days}d ≤ ttl={ttl_days}d)"
    return False, f"stale (age={age_days}d > ttl={ttl_days}d)"


def _run_ticker_chain(tc: TickerTrainingContext) -> None:
    """Run the full per-ticker job chain in one worker thread."""
    tag = f"[{tc.ticker}|{current_thread().name}]"
    t0 = time.monotonic()

    # Per-model TTL check (2026-04-24): when training.model_ttl_days is set
    # and this ticker's artifact trained_date is within TTL, skip the
    # tournament and keep the existing artifact. `--force` on train_104.py
    # bypasses via TrainingContext.force_retrain (propagated into config).
    ttl_days = int(tc.config.get("training", {}).get("model_ttl_days", 0) or 0)
    force    = bool(tc.config.get("_force_retrain", False))
    if ttl_days > 0 and not force:
        fresh, reason = _model_is_fresh(tc.ticker, tc.strategy_dir, ttl_days)
        if fresh:
            log.info("%s TTL skip — %s", tag, reason)
            # Audit #45: don't claim "exported" — that confuses downstream
            # counters that interpret "exported" as "wrote a fresh artifact
            # this run". The cached artifact is reused but no work was done.
            tc.ttl_skipped = True
            return

    log.info("%s FeatureJob START", tag)
    TickerFeatureJob().run(tc)
    if tc.feature_frame is None:
        log.warning("%s FeatureJob produced no frame — skipping chain", tag)
        return
    log.info("%s FeatureJob OK  %d rows", tag, len(tc.feature_frame))

    log.info("%s TournamentJob START", tag)
    TickerTournamentJob().run(tc)
    if not tc.result:
        log.warning("%s TournamentJob produced no result — skipping export", tag)
        return
    passes = tc.result.get("passes_floor", False)
    sharpe = tc.result.get("sharpe", 0.0)
    best   = tc.result.get("best_approach", "?")
    log.info("%s TournamentJob OK  best=%s sharpe=%.3f passes=%s", tag, best, sharpe, passes)

    log.info("%s ExportJob START", tag)
    TickerExportJob().run(tc)
    log.info("%s ExportJob OK  exported=%s", tag, tc.exported)

    log.info("%s CalibrationJob START", tag)
    TickerCalibrationJob().run(tc)
    log.info("%s CalibrationJob OK  cal=%s", tag, tc.calibration)

    log.info("%s chain DONE  total=%.1fs", tag, time.monotonic() - t0)


def run_ticker_parallel(
    ticker_ctxs: list[TickerTrainingContext],
    max_workers: "int | None" = None,
    timeout_seconds: "float | None" = None,
    progress_log_seconds: "float | None" = None,
) -> None:
    """Run _run_ticker_chain for each ticker in parallel via ThreadPoolExecutor.

    max_workers=None → auto (cpu_count-2, min 1).
    timeout_seconds=None → no wall-clock phase timeout.

    ThreadPoolExecutor cannot safely interrupt a running worker. If the
    per-ticker training phase exceeds timeout_seconds, fail hard before the
    caller can collect a partial model set and treat it as a valid retrain.
    """
    if not ticker_ctxs:
        return
    from .pipeline import ParallelTimeoutError, resolve_workers
    if max_workers is None or timeout_seconds is None:
        cfg = getattr(ticker_ctxs[0], "config", None) or {}
        if max_workers is None:
            max_workers = cfg.get("parallel_workers")
        if timeout_seconds is None:
            timeout_seconds = cfg.get("parallel_ticker_timeout_seconds")
        if progress_log_seconds is None:
            progress_log_seconds = cfg.get("parallel_progress_log_seconds")
    if progress_log_seconds is None:
        progress_log_seconds = 30.0
    n = resolve_workers(max_workers, len(ticker_ctxs))
    job_name = "TickerTrainingChain"
    log.info("run_ticker_parallel: %d tickers, %d workers, timeout=%s",
             len(ticker_ctxs), n, timeout_seconds)
    t0 = time.monotonic()
    ex = ThreadPoolExecutor(max_workers=n, thread_name_prefix="ticker")
    futures = {ex.submit(_run_ticker_chain, tc): tc.ticker for tc in ticker_ctxs}
    pending = set(futures)
    completed = 0
    progress_interval = max(0.01, float(progress_log_seconds or 0.0))
    next_progress = t0 + progress_interval
    abandon_executor = False
    try:
        while pending:
            now = time.monotonic()
            elapsed = now - t0
            if timeout_seconds is not None and elapsed >= float(timeout_seconds):
                pending_tickers = sorted(futures[f] for f in pending)
                for fut in pending:
                    fut.cancel()
                log.error(
                    "run_ticker_parallel: %s TIMEOUT after %.2fs — done=%d/%d "
                    "pending=%d tickers=%s; worker may still be running",
                    job_name, elapsed, completed, len(futures), len(pending_tickers),
                    pending_tickers[:20],
                )
                ex.shutdown(wait=False, cancel_futures=True)
                abandon_executor = True
                raise ParallelTimeoutError(job_name, elapsed, pending_tickers)

            wait_timeout = max(0.0, next_progress - now)
            if timeout_seconds is not None:
                wait_timeout = min(
                    wait_timeout,
                    max(0.0, float(timeout_seconds) - elapsed),
                )
            done, pending = wait(
                pending,
                timeout=wait_timeout,
                return_when=FIRST_COMPLETED,
            )

            for fut in done:
                ticker = futures[fut]
                completed += 1
                try:
                    fut.result()
                except Exception as e:
                    log.error("[%s] chain ERROR — %s: %s", ticker, type(e).__name__, e)

            now = time.monotonic()
            if pending and now >= next_progress:
                pending_tickers = sorted(futures[f] for f in pending)
                log.info(
                    "run_ticker_parallel: %s progress done=%d/%d pending=%d "
                    "elapsed=%.2fs pending_tickers=%s",
                    job_name, completed, len(futures), len(pending_tickers),
                    now - t0, pending_tickers[:10],
                )
                next_progress = now + progress_interval
    finally:
        if not abandon_executor:
            ex.shutdown(wait=True)
    elapsed = time.monotonic() - t0
    log.info("run_ticker_parallel: DONE  %.1fs total  (%d tickers)", elapsed, len(ticker_ctxs))
    print(f"  parallel phase done in {elapsed:.1f}s  ({len(ticker_ctxs)} tickers, {n} workers)")


# ── Global jobs ────────────────────────────────────────────────────────────────

class DataFetchJob(TrainingJob):
    """Fetch OHLCV for all tickers. Skipped if ohlcv already populated."""

    def should_skip(self, ctx: TrainingContext) -> bool:
        if ctx.ohlcv:
            print("DataFetchJob: ohlcv already populated — skipping")
            return True
        return False

    def run(self, ctx: TrainingContext) -> None:
        from kernel.data import fetch_ohlcv

        cfg = ctx.config
        start = cfg["sample_start"]
        end   = cfg["sample_end"]
        sector_etf = cfg.get("sector_etf_map", {})
        # Audit #46: previously hardcoded "SPY" while pp_training_full
        # honoured config.benchmark — the two pipelines disagreed when
        # benchmark != "SPY". Use the config-driven value here too.
        benchmark = cfg.get("benchmark", "SPY")
        all_tickers = sorted(
            set(ctx.watchlist) | set(sector_etf.values()) | {benchmark}
        )
        print(f"DataFetchJob: fetching {len(all_tickers)} tickers {start} → {end}")
        for ticker in all_tickers:
            try:
                df = fetch_ohlcv(ticker, start=start, end=end)
                if df is not None and not df.empty:
                    ctx.ohlcv[ticker] = df
                    print(f"  {ticker}: {len(df)} rows")
                else:
                    print(f"  {ticker}: EMPTY")
            except Exception as exc:
                print(f"  {ticker}: ERROR — {exc}")
        print(f"DataFetchJob: loaded {len(ctx.ohlcv)} / {len(all_tickers)} tickers")


class HurstCUSUMTask(TrainingTask):
    """Compute rolling Hurst exponent and CUSUM → ctx.hurst_series, ctx.cusum_series."""

    def run(self, ctx: TrainingContext) -> None:
        from training.regime import rolling_hurst, rolling_cusum  # noqa: PLC0415

        spy_df = ctx.spy_df
        if spy_df is None:
            raise RuntimeError("HurstCUSUMTask: SPY not in ohlcv — run DataFetchJob first")

        rcfg        = ctx.config["regime"]
        spy_returns = spy_df["close"].pct_change().dropna()

        ctx.hurst_series = rolling_hurst(spy_returns, window=rcfg["hurst_window"]).dropna()
        ctx.cusum_series = rolling_cusum(
            spy_returns,
            window    = rcfg["cusum_lookback"],
            threshold = rcfg["cusum_threshold"],
            drift     = rcfg["cusum_drift"],
        )
        ctx.changepoint_dates = ctx.cusum_series[ctx.cusum_series].index

        print(f"HurstCUSUMTask: CUSUM detected {len(ctx.changepoint_dates)} changepoints")
        print(f"HurstCUSUMTask: Hurst {ctx.hurst_series.min():.3f}–"
              f"{ctx.hurst_series.max():.3f} (mean={ctx.hurst_series.mean():.3f})")


class GMMFitTask(TrainingTask):
    """Fit GMM on SPY features and predict regime labels + probs → ctx.gmm."""

    def run(self, ctx: TrainingContext) -> None:
        from training.regime import build_gmm_features, RegimeGMM  # noqa: PLC0415

        spy_df = ctx.spy_df
        rcfg   = ctx.config["regime"]

        gmm_features = build_gmm_features(
            spy_df, vol_window=20, hurst_window=rcfg["hurst_window"]
        )
        ctx._gmm_data_window_start = (  # noqa: SLF001
            gmm_features.index.min().date().isoformat()
            if not gmm_features.empty else None
        )
        ctx._gmm_data_window_end = (  # noqa: SLF001
            gmm_features.index.max().date().isoformat()
            if not gmm_features.empty else None
        )
        ctx._gmm_n_train_rows = int(len(gmm_features))  # noqa: SLF001
        gmm = RegimeGMM(n_components=3, random_state=42, n_init=10)
        gmm.fit(gmm_features)
        ctx.gmm = gmm

        regime_labels, regime_probs = gmm.predict(gmm_features)
        ctx._gmm_labels = regime_labels  # noqa: SLF001
        ctx._gmm_probs  = regime_probs   # noqa: SLF001


class RegimeCombineTask(TrainingTask):
    """Combine Hurst + CUSUM + GMM + hard-BEAR rule → ctx.final_regime, ctx.final_regime_conf."""

    def run(self, ctx: TrainingContext) -> None:
        BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR = (
            "BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"
        )
        rcfg         = ctx.config["regime"]
        hurst_trend  = rcfg["hurst_trending_threshold"]
        hurst_rev    = rcfg["hurst_reversion_threshold"]
        vol_window   = rcfg["vol_realized_window"]
        bear_vol_thr = rcfg["bear_vol_threshold"]
        bear_ret_thr = rcfg["bear_return_threshold"]
        choppy_floor = rcfg.get("choppy_hurst_floor", 0.20)

        spy_returns = ctx.spy_df["close"].pct_change().dropna()
        spy_20d_vol = spy_returns.rolling(vol_window).std() * np.sqrt(252)
        spy_20d_ret = spy_returns.rolling(vol_window).sum()

        regime_labels = ctx._gmm_labels  # noqa: SLF001
        regime_probs  = ctx._gmm_probs   # noqa: SLF001

        common_idx        = ctx.hurst_series.index.intersection(regime_labels.index)
        final_regime      = pd.Series(index=common_idx, dtype=str)
        final_regime_conf = pd.Series(index=common_idx, dtype=float)

        for dt in common_idx:
            h          = ctx.hurst_series.loc[dt]
            gmm_r      = regime_labels.loc[dt]
            gmm_bear_p = regime_probs.loc[dt].get(BEAR, 0.0)
            base       = BULL_CALM if h > hurst_trend else (CHOPPY if h < hurst_rev else None)

            vol_today = float(spy_20d_vol.loc[dt]) if dt in spy_20d_vol.index else 0.0
            ret_today = float(spy_20d_ret.loc[dt]) if dt in spy_20d_ret.index else 0.0
            hard_bear = vol_today > bear_vol_thr or ret_today < bear_ret_thr

            if hard_bear or gmm_bear_p > 0.5:
                final_regime.loc[dt] = BEAR
            elif base is None:
                final_regime.loc[dt] = gmm_r if gmm_r != BEAR else BULL_VOLATILE
            else:
                final_regime.loc[dt] = base

            r = final_regime.loc[dt]
            if r == CHOPPY:
                conf = (hurst_rev - h) / max(hurst_rev - choppy_floor, 1e-6)
                final_regime_conf.loc[dt] = float(min(1.0, max(0.0, conf)))
            else:
                final_regime_conf.loc[dt] = float(regime_probs.loc[dt].get(r, 0.5))

        ctx.final_regime      = final_regime
        ctx.final_regime_conf = final_regime_conf

        print("RegimeCombineTask: Final regime distribution:")
        print(final_regime.value_counts().to_string())


class RegimeSaveTask(TrainingTask):
    """Save the GMM artifact to disk."""

    def run(self, ctx: TrainingContext) -> None:
        if not ctx.strategy_dir:
            return
        artifacts_dir = ctx.strategy_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        gmm_path = artifacts_dir / "spy-gmm-regime.json"
        ctx.gmm.save(
            gmm_path,
            as_of_date=getattr(ctx, "_gmm_data_window_end", None),
            data_window_start=getattr(ctx, "_gmm_data_window_start", None),
            data_window_end=getattr(ctx, "_gmm_data_window_end", None),
            n_train_rows=getattr(ctx, "_gmm_n_train_rows", None),
        )
        print(f"RegimeSaveTask: GMM artifact saved → {gmm_path}")


class RegimeFitJob(TrainingJob):
    """Fit Hurst + CUSUM + GMM and build daily regime series. Saves GMM artifact.

    Task chain: HurstCUSUM → GMMFit → RegimeCombine → RegimeSave
    """

    def should_skip(self, ctx: TrainingContext) -> bool:
        if ctx.final_regime is not None:
            print("RegimeFitJob: final_regime already populated — skipping")
            return True
        return False

    @property
    def tasks(self) -> list[TrainingTask]:
        return [HurstCUSUMTask(), GMMFitTask(), RegimeCombineTask(), RegimeSaveTask()]


class FeatureJob(TrainingJob):
    """Orchestrate parallel per-ticker: Feature → Tournament → Export → Calibrate."""

    def should_skip(self, ctx: TrainingContext) -> bool:
        if ctx.feature_frames:
            print("FeatureJob: feature_frames already populated — skipping")
            return True
        return False

    def run(self, ctx: TrainingContext) -> None:
        ticker_ctxs = [
            TickerTrainingContext(
                ticker=t,
                ohlcv=ctx.ohlcv,
                config=ctx.config,
                strategy_dir=ctx.strategy_dir,
            )
            for t in ctx.watchlist
        ]
        log.info("FeatureJob: launching parallel chain for %d tickers", len(ticker_ctxs))
        print(f"FeatureJob: launching parallel chain for {len(ticker_ctxs)} tickers")
        run_ticker_parallel(ticker_ctxs)

        ctx.feature_frames = {
            tc.ticker: tc.feature_frame
            for tc in ticker_ctxs if tc.feature_frame is not None
        }
        ctx.results = {
            tc.ticker: tc.result
            for tc in ticker_ctxs if tc.result
        }
        ctx.ttl_skipped = [tc.ticker for tc in ticker_ctxs if tc.ttl_skipped]
        if ctx.ttl_skipped:
            log.info("FeatureJob: %d tickers TTL-skipped (%s)",
                     len(ctx.ttl_skipped),
                     ", ".join(ctx.ttl_skipped[:10])
                     + (" ..." if len(ctx.ttl_skipped) > 10 else ""))
        ctx.exported = [tc.ticker for tc in ticker_ctxs if tc.exported]
        ctx.calibration_summary = {
            tc.ticker: tc.calibration
            for tc in ticker_ctxs if tc.calibration
        }

        passed  = sum(1 for r in ctx.results.values() if r.get("passes_floor"))
        n_feat  = len(ctx.feature_frames)
        n_exp   = len(ctx.exported)
        n_cal   = len(ctx.calibration_summary)
        log.info("FeatureJob: frames=%d  passed=%d/%d  exported=%d  calibrated=%d",
                 n_feat, passed, len(ctx.watchlist), n_exp, n_cal)
        print(f"FeatureJob: {n_feat} feature frames, "
              f"{passed}/{len(ctx.watchlist)} passed Sharpe floor, "
              f"{n_exp} exported, {n_cal} calibrated")


class TournamentJob(TrainingJob):
    """No-op: tournament runs inside FeatureJob's parallel chain."""

    def should_skip(self, ctx: TrainingContext) -> bool:
        if ctx.results:
            print("TournamentJob: results already populated by FeatureJob — skipping")
            return True
        return False

    def run(self, ctx: TrainingContext) -> None:
        print("TournamentJob: re-running standalone (not recommended — use FeatureJob)")
        from training.tournament import run_tournament_all
        ctx.results = run_tournament_all(
            ctx.watchlist, ctx.feature_frames, ctx.ohlcv, ctx.config
        )


class ExportJob(TrainingJob):
    """No-op: export runs inside FeatureJob's parallel chain."""

    def should_skip(self, ctx: TrainingContext) -> bool:
        if ctx.exported:
            print("ExportJob: exported already populated by FeatureJob — skipping")
            return True
        return False

    def run(self, ctx: TrainingContext) -> None:
        print("ExportJob: re-running standalone (not recommended — use FeatureJob)")
        from datetime import date as _date
        from training.export import export_models, retrain_live_models
        if not ctx.strategy_dir:
            return
        today = str(_date.today())
        mp = _model_params_for_tournament(ctx.config)
        ctx.exported, _ = export_models(
            ctx.results, ctx.strategy_dir, today,
            lookahead=mp["lookahead"],
            strategy_name=ctx.config.get("_strategy_name", "renquant_103"),
        )
        retrain_live_models(
            ctx.results, ctx.feature_frames, ctx.exported,
            ctx.strategy_dir, mp, ctx.config, today,
            ohlcv=ctx.ohlcv,
        )


class CorrelationJob(TrainingJob):
    """Compute 120-day return correlation and save watchlist artifact.

    Schema v2 (2026-05-10, audit fix): output is wrapped with
    `schema_version`, `as_of_date`, `data_window_start`, `data_window_end`
    so consumers (sim / LEAN) can enforce `as_of_date <= backtest_start`
    via `kernel.walk_forward.correlation_guard.assert_correlation_no_leakage`.
    Legacy v1 flat-dict format is still accepted on the read path
    (`parse_correlation_artifact` handles both) but the writer always
    emits v2 going forward.
    """

    def run(self, ctx: TrainingContext) -> None:
        close_df = pd.DataFrame({
            t: ctx.ohlcv[t]["close"] for t in ctx.watchlist if t in ctx.ohlcv
        })
        ret_df = close_df.pct_change().dropna()
        # tail(120) is the actual data window the correlation reflects;
        # capture its endpoints for the artifact metadata.
        tail = ret_df.tail(120)
        ctx.corr_matrix = tail.corr()

        if ctx.strategy_dir:
            corr_dict = {
                ticker: {
                    other: round(float(ctx.corr_matrix.loc[ticker, other]), 4)
                    for other in ctx.corr_matrix.columns
                }
                for ticker in ctx.corr_matrix.index
            }
            # Stamp data window endpoints — `as_of_date` is the latest
            # date used in the correlation computation (i.e. the upper
            # bound of the leak-free backtest window for this artifact).
            data_start = (
                tail.index.min().date().isoformat() if not tail.empty else None
            )
            data_end = (
                tail.index.max().date().isoformat() if not tail.empty else None
            )
            wrapped = {
                "schema_version": 2,
                "as_of_date": data_end,
                "data_window_start": data_start,
                "data_window_end": data_end,
                "matrix": corr_dict,
            }
            artifacts_dir = ctx.strategy_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            corr_path = artifacts_dir / "watchlist-correlation.json"
            corr_path.write_text(json.dumps(wrapped, indent=2))
            print(
                f"CorrelationJob: saved → {corr_path} "
                f"(as_of_date={data_end} window={data_start}…{data_end})"
            )


class CalibrationJob(TrainingJob):
    """No-op: calibration runs inside FeatureJob's parallel chain."""

    def should_skip(self, ctx: TrainingContext) -> bool:
        if ctx.calibration_summary:
            print("CalibrationJob: calibration_summary already populated — skipping")
            return True
        return False

    def run(self, ctx: TrainingContext) -> None:
        print("CalibrationJob: nothing to do (calibration ran in FeatureJob parallel chain)")


# ── Per-ticker jobs ────────────────────────────────────────────────────────────

class TickerFeatureJob(TrainingTickerJob):
    """Build the labelled feature frame for one ticker."""

    def run(self, tc: TickerTrainingContext) -> None:
        from training.features import build_training_features

        mp = _model_params_for_tournament(tc.config)
        try:
            tc.feature_frame = build_training_features(
                tc.ticker,
                tc.ohlcv,
                tc.config["indicator_spec"],
                mp["lookahead"],
                mp["threshold"],
            )
        except Exception as exc:
            print(f"  {tc.ticker}: TickerFeatureJob failed — {exc}")


class TickerTournamentJob(TrainingTickerJob):
    """Train all model types for one ticker; select best by OOS Sharpe."""

    def run(self, tc: TickerTrainingContext) -> None:
        from training.tournament import run_tournament, resolve_oos_cutoff

        if tc.feature_frame is None or tc.feature_frame.empty:
            return
        mp = _model_params_for_tournament(tc.config)
        try:
            result = run_tournament(
                tc.ticker,
                tc.feature_frame,
                tc.ohlcv[tc.ticker]["close"],
                tc.ohlcv["SPY"]["close"],
                mp,
                sharpe_floor=float(tc.config.get("sharpe_floor", 0.8)),
                tax_config=tc.config["tax"],
                oos_cutoff=resolve_oos_cutoff(tc.config),
            )
            for line in result.pop("_log", []):
                print(f"  [{tc.ticker}] {line}")
            tc.result = result
        except Exception as exc:
            print(f"  {tc.ticker}: TickerTournamentJob failed — {exc}")


class TickerExportJob(TrainingTickerJob):
    """Export the best model artifact for one ticker."""

    def run(self, tc: TickerTrainingContext) -> None:
        from datetime import date as _date
        from training.export import export_one_model, retrain_one_live_model

        if not tc.result or not tc.strategy_dir:
            return
        today = str(_date.today())
        mp = _model_params_for_tournament(tc.config)
        try:
            exported = export_one_model(
                tc.ticker, tc.result, tc.strategy_dir, today,
                lookahead=mp["lookahead"],
                strategy_name=tc.config.get("_strategy_name", "renquant_103"),
            )
            if exported:
                tc.exported = True
                retrain_one_live_model(
                    tc.ticker, tc.result, tc.feature_frame,
                    tc.strategy_dir, mp, tc.config, today,
                    ohlcv=tc.ohlcv,
                )
        except Exception as exc:
            print(f"  {tc.ticker}: TickerExportJob failed — {exc}")


class TickerCalibrationJob(TrainingTickerJob):
    """Fit and save score calibration metadata for one ticker."""

    def run(self, tc: TickerTrainingContext) -> None:
        if not tc.exported or not tc.result or tc.feature_frame is None:
            return

        try:
            from training.scoring import fit_probability_calibration
        except ImportError:
            return

        import json as _json

        mp = _model_params_for_tournament(tc.config)
        res = tc.result
        best = res.get("best_approach")
        model_obj = res.get(best, {}).get("model") if best else None
        if model_obj is None:
            return

        try:
            from training.tournament import resolve_oos_cutoff
            raw_scores  = model_obj.predict_score_bulk(tc.feature_frame)
            oos_start   = resolve_oos_cutoff(tc.config)
            oos_frame   = tc.feature_frame[tc.feature_frame.index >= oos_start]
            if oos_frame.empty:
                return
            oos_scores  = raw_scores.reindex(oos_frame.index).dropna()
            future_rets = oos_frame["label"].reindex(oos_scores.index)
            if len(oos_scores) < 50:
                return

            cal = fit_probability_calibration(
                oos_scores, future_rets, lookahead=mp["lookahead"]
            )
            if tc.strategy_dir:
                meta_path = (tc.strategy_dir / "models" / tc.ticker
                             / f"{tc.ticker}-policy-metadata.json")
                if meta_path.exists():
                    meta = _json.loads(meta_path.read_text())
                    meta["score_calibration"] = cal.to_dict()
                    meta_path.write_text(_json.dumps(meta, indent=2))
            tc.calibration = {"method": cal.method, "n": len(oos_scores)}
        except Exception as exc:
            # Audit #50: previously print-only — now also log so production
            # log scrapers (ntfy, log-watcher) can surface the failure.
            log.warning("[%s] TickerCalibrationJob failed — %s: %s",
                        tc.ticker, type(exc).__name__, exc)
            print(f"  {tc.ticker}: TickerCalibrationJob failed — {exc}")


# ── TrainingPipeline ───────────────────────────────────────────────────────────

class TrainingPipeline:
    """Run the full training pipeline (3 phases)."""

    def run(self, ctx: TrainingContext) -> TrainingContext:
        jobs: list[TrainingJob] = [
            DataFetchJob(),
            RegimeFitJob(),
            FeatureJob(),    # orchestrates parallel per-ticker chain
            CorrelationJob(),
        ]
        t0 = time.monotonic()
        log.info("TrainingPipeline START  watchlist=%d", len(ctx.watchlist))
        for job in jobs:
            if job.should_skip(ctx):
                continue
            t1 = time.monotonic()
            log.info("── %s START", job.name)
            print(f"\n── {job.name} ──")
            job.run(ctx)
            log.info("── %s DONE  %.1fs", job.name, time.monotonic() - t1)
        log.info("TrainingPipeline DONE  total=%.1fs", time.monotonic() - t0)
        return ctx
