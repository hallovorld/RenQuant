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
        from kernel.data import fetch_ohlcv

        cfg = ctx.config
        start = cfg.get("sample_start")
        end   = cfg.get("sample_end")
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


class LoadFundamentalsTask(PanelTask):
    """Populate ctx.fundamentals from the parquet cache (or fetch on demand).

    No-op when `panel_ltr.fundamentals.enabled` is false (default). The
    cache lives at `data/fundamentals/{SYMBOL}.parquet`; see
    `kernel/fundamentals.py` + `scripts/fetch_fundamentals.py`.
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        cfg = ctx.config.get("panel_ltr", {}).get("fundamentals", {})
        if not cfg.get("enabled", False):
            return
        if ctx.fundamentals:
            return

        from kernel.fundamentals import (  # noqa: PLC0415
            FundamentalsStore, fetch_fundamentals_watchlist,
        )

        cache_dir = cfg.get("cache_dir", "data/fundamentals")
        store = FundamentalsStore(data_dir=cache_dir)
        refetch = bool(cfg.get("refetch", False))

        # Negative-cache 2026-04-24: permanently-empty tickers (ETFs
        # without fundamentals, foreign stocks outside OpenBB coverage)
        # get skipped to avoid per-run refetch timeouts. User spec:
        # "本地数据 up to date 的时候不需要再下载了".
        skip_set = set(cfg.get("skip_tickers", []))

        out: dict[str, dict[str, float]] = {}
        for sym in ctx.watchlist:
            if sym in skip_set:
                continue
            cached = None if refetch else store.latest(sym)
            if cached is not None:
                out[sym] = cached
        missing = [s for s in ctx.watchlist
                   if s not in out and s not in skip_set]
        if missing and cfg.get("allow_fetch", True):
            log.info("LoadFundamentalsTask: fetching %d missing tickers", len(missing))
            try:
                fresh = fetch_fundamentals_watchlist(missing, store=store)
                out.update(fresh)
            except Exception as exc:
                log.warning("LoadFundamentalsTask: fetch_fundamentals_watchlist failed — %s", exc)

        ctx.fundamentals = out
        log.info("LoadFundamentalsTask: %d / %d tickers with fundamentals (%d skipped)",
                 len(out), len(ctx.watchlist), len(skip_set))


class LoadEarningsSurpriseTask(PanelTask):
    """Populate ctx.earnings_surprises from the parquet cache (or fetch).

    No-op when `panel_ltr.earnings_surprise.enabled` is false. Cache at
    `data/earnings_surprise/{SYMBOL}.parquet` (see
    `kernel/earnings_surprise.py` + `scripts/fetch_earnings_surprise.py`).
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        cfg = ctx.config.get("panel_ltr", {}).get("earnings_surprise", {})
        if not cfg.get("enabled", False):
            return
        if ctx.earnings_surprises:
            return

        from kernel.earnings_surprise import (  # noqa: PLC0415
            EarningsSurpriseStore, fetch_earnings_surprise_watchlist,
        )

        cache_dir = cfg.get("cache_dir", "data/earnings_surprise")
        store = EarningsSurpriseStore(data_dir=cache_dir)
        # Negative-cache (2026-04-24): skip tickers with no earnings
        # (ETFs, commodity funds) to avoid per-run retry timeouts.
        skip_set = set(cfg.get("skip_tickers", []))

        out: dict = {}
        for sym in ctx.watchlist:
            if sym in skip_set:
                continue
            cached = store.load(sym)
            if cached is not None and not cached.empty:
                out[sym] = cached
        missing = [s for s in ctx.watchlist
                   if s not in out and s not in skip_set]
        if missing and cfg.get("allow_fetch", True):
            log.info("LoadEarningsSurpriseTask: fetching %d missing tickers", len(missing))
            try:
                fresh = fetch_earnings_surprise_watchlist(missing)
                # fresh_df may be empty — store writes only non-empty anyway
                for sym, df in fresh.items():
                    if df is not None and not df.empty:
                        out[sym] = df
            except Exception as exc:
                log.warning("LoadEarningsSurpriseTask: fetch failed — %s", exc)

        ctx.earnings_surprises = out
        log.info("LoadEarningsSurpriseTask: %d / %d tickers with surprise history",
                 len(out), len(ctx.watchlist))


class LoadInsiderTradesTask(PanelTask):
    """Populate ctx.insider_trades from the parquet cache (or fetch from SEC).

    No-op when `panel_ltr.insider_trades.enabled` is false. Executive-only
    (isOfficer=true) Form 4 open-market transactions.
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        cfg = ctx.config.get("panel_ltr", {}).get("insider_trades", {})
        if not cfg.get("enabled", False):
            return
        if ctx.insider_trades:
            return

        from kernel.insider_trades import (  # noqa: PLC0415
            InsiderTradesStore, fetch_insider_trades_watchlist,
        )

        cache_dir = cfg.get("cache_dir", "data/insider_trades")
        max_filings = int(cfg.get("max_filings", 200))
        store = InsiderTradesStore(data_dir=cache_dir)
        # Negative-cache (2026-04-24): foreign stocks + ETFs have no
        # SEC Form 4 filings — skip to avoid EDGAR rate-limit burn.
        skip_set = set(cfg.get("skip_tickers", []))

        out: dict = {}
        for sym in ctx.watchlist:
            if sym in skip_set:
                continue
            cached = store.load(sym)
            if cached is not None and not cached.empty:
                out[sym] = cached
        missing = [s for s in ctx.watchlist
                   if s not in out and s not in skip_set]
        if missing and cfg.get("allow_fetch", True):
            log.info("LoadInsiderTradesTask: fetching %d missing tickers (rate-limited SEC)",
                     len(missing))
            try:
                fresh = fetch_insider_trades_watchlist(
                    missing, max_filings=max_filings,
                )
                for sym, df in fresh.items():
                    if df is not None and not df.empty:
                        out[sym] = df
            except Exception as exc:
                log.warning("LoadInsiderTradesTask: fetch failed — %s", exc)

        ctx.insider_trades = out
        log.info("LoadInsiderTradesTask: %d / %d tickers with insider rows",
                 len(out), len(ctx.watchlist))


class LoadHourlyBarsTask(PanelTask):
    """Populate ctx.hourly_bars from the parquet cache (Plan G).

    No-op when `panel_ltr.hourly.enabled` is false (default). Cache at
    `data/intraday/{SYMBOL}/1h.parquet`; `scripts/fetch_hourly_bars.py`
    owns populating the cache from Alpaca IEX. This task never fetches
    live — training must be reproducible offline.
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        cfg = ctx.config.get("panel_ltr", {}).get("hourly", {})
        if not cfg.get("enabled", False):
            return
        if ctx.hourly_bars:
            return

        from kernel.intraday import HourlyBarStore  # noqa: PLC0415

        cache_dir = cfg.get("cache_dir", "data/intraday")
        store = HourlyBarStore(data_dir=cache_dir)

        out: dict[str, pd.DataFrame] = {}
        for sym in ctx.watchlist:
            df = store.load(sym)
            if df is not None and not df.empty:
                out[sym] = df

        ctx.hourly_bars = out
        log.info("LoadHourlyBarsTask: %d / %d tickers with hourly bars",
                 len(out), len(ctx.watchlist))


class PanelDataJob(PanelJob):
    """Phase 1 — gather market data + sector momentum + fundamentals + earnings + insiders + hourly.

    Task chain: FetchOHLCV → SectorMomentum → LoadFundamentals
                → LoadEarningsSurprise → LoadInsiderTrades → LoadHourlyBars
    """

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        return bool(ctx.ohlcv) and bool(ctx.sector_momentum)

    @property
    def tasks(self) -> list[PanelTask]:
        return [
            FetchOHLCVTask(),
            SectorMomentumTask(),
            LoadFundamentalsTask(),
            LoadEarningsSurpriseTask(),
            LoadInsiderTradesTask(),
            LoadHourlyBarsTask(),
        ]


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
                fundamentals=ctx.fundamentals,
                earnings_surprises=ctx.earnings_surprises,
                insider_trades=ctx.insider_trades,
                hourly_bars=ctx.hourly_bars,
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
    """Compute raw factor bundle for one ticker (unscaled; z-score is global).

    Emits size / mom_12_1 / beta_60d / resid_mom time-series per ticker, plus
    four optional static-scalar columns (earnings_yield / roe /
    gross_profitability / book_to_price) broadcast to the date index when
    fundamentals are loaded. FactorZScoreTask z-scores every column below
    cross-sectionally.
    """

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
            compute_amihud_illiquidity, compute_volume_shift,
            compute_price_to_high, compute_realized_vol,
            compute_drawdown_from_peak,
            FUNDAMENTAL_COLS,
        )
        from kernel.earnings_surprise import compute_earnings_surprise_cum
        from kernel.insider_trades    import compute_insider_net_buy_cum
        try:
            one = {tc.ticker: tc.ohlcv[tc.ticker]}
            size = compute_size_feature(one, None).get(tc.ticker)
            mom  = compute_momentum_12_1(one, mom_window=mom_window, skip=skip).get(tc.ticker)
            beta = compute_rolling_beta(one, tc.ohlcv[benchmark], window=beta_window).get(tc.ticker)
            rmom = compute_residual_momentum(
                one, tc.ohlcv[benchmark], window=beta_window,
                mom_window=mom_window, skip=skip,
            ).get(tc.ticker)
            # Round 3 orthogonal factors
            amihud   = compute_amihud_illiquidity(one, window=21).get(tc.ticker)
            vol_shft = compute_volume_shift(one, short_window=20, long_window=60).get(tc.ticker)
            p2h      = compute_price_to_high(one, window=252).get(tc.ticker)
            rvol     = compute_realized_vol(one, window=20).get(tc.ticker)
            ddn      = compute_drawdown_from_peak(one, window=252).get(tc.ticker)
            idx = tc.ohlcv[tc.ticker].index
            cols: dict[str, pd.Series] = {
                "size":            (size if size is not None else pd.Series(index=idx)).reindex(idx),
                "mom_12_1":        (mom  if mom  is not None else pd.Series(index=idx)).reindex(idx),
                "beta_60d":        (beta if beta is not None else pd.Series(index=idx)).reindex(idx),
                "resid_mom":       (rmom if rmom is not None else pd.Series(index=idx)).reindex(idx),
                # Round 3: liquidity + behavioral factors
                "amihud_illiq":    (amihud   if amihud   is not None else pd.Series(index=idx)).reindex(idx),
                "volume_shift":    (vol_shft if vol_shft is not None else pd.Series(index=idx)).reindex(idx),
                "price_to_high":   (p2h      if p2h      is not None else pd.Series(index=idx)).reindex(idx),
                "realized_vol":    (rvol     if rvol     is not None else pd.Series(index=idx)).reindex(idx),
                "drawdown_peak":   (ddn      if ddn      is not None else pd.Series(index=idx)).reindex(idx),
            }
            # Fundamentals: broadcast the ticker's snapshot scalar to every bar.
            # A missing ticker → NaN series (FactorZScoreTask / sector-median
            # fill will handle it globally).
            if tc.fundamentals:
                ticker_fund = tc.fundamentals.get(tc.ticker, {})
                for col in FUNDAMENTAL_COLS:
                    val = ticker_fund.get(col, float("nan"))
                    cols[col] = pd.Series(val, index=idx)
            # Earnings surprise: trailing-4Q cumulative surprise %, ffilled
            # to the ticker's daily index. Sparse (one row per announcement)
            # -> daily step function.
            if tc.earnings_surprises:
                surprise_daily = compute_earnings_surprise_cum(
                    {tc.ticker: tc.earnings_surprises.get(tc.ticker, pd.DataFrame())},
                    {tc.ticker: tc.ohlcv[tc.ticker]},
                ).get(tc.ticker)
                cols["earnings_surprise_cum"] = (
                    surprise_daily if surprise_daily is not None
                    else pd.Series(float("nan"), index=idx)
                )
            # Insider trades: trailing-90d cumulative net executive buy (USD).
            if tc.insider_trades:
                insider_daily = compute_insider_net_buy_cum(
                    {tc.ticker: tc.insider_trades.get(tc.ticker, pd.DataFrame())},
                    {tc.ticker: tc.ohlcv[tc.ticker]},
                    trailing_days=90,
                ).get(tc.ticker)
                cols["insider_net_buy_90d"] = (
                    insider_daily if insider_daily is not None
                    else pd.Series(float("nan"), index=idx)
                )
            # Hourly aggregated features (Plan G): morning/afternoon drift,
            # VWAP premium, vol ratio, intraday realized vol, overnight gap.
            # Reindex the per-session output onto the ticker's daily index;
            # missing sessions → NaN (FactorZScoreTask handles globally).
            if tc.hourly_bars:
                from training_panel.hourly_features import (  # noqa: PLC0415
                    HOURLY_FEATURE_COLS, compute_hourly_features,
                )
                hourly_df = tc.hourly_bars.get(tc.ticker)
                if hourly_df is not None and not hourly_df.empty:
                    h_feats = compute_hourly_features(hourly_df)
                    # Normalize daily index so join works regardless of tz.
                    h_feats.index = pd.DatetimeIndex(h_feats.index).normalize()
                    daily_idx = pd.DatetimeIndex(idx).normalize()
                    for col in HOURLY_FEATURE_COLS:
                        series = h_feats[col] if col in h_feats.columns else pd.Series(dtype=float)
                        # Reindex via normalized daily dates, then re-key onto raw idx.
                        aligned = series.reindex(daily_idx)
                        aligned.index = idx
                        cols[col] = aligned
                else:
                    for col in (
                        "morning_drift", "afternoon_drift", "vwap_premium",
                        "vol_ratio", "intraday_realized_vol", "overnight_gap",
                    ):
                        cols[col] = pd.Series(float("nan"), index=idx)
            tc.raw_factor_frame = pd.DataFrame(cols, index=idx)
        except Exception as exc:
            log.error("  %s: TickerPanelFactorJob failed — %s", tc.ticker, exc)


# ── Phase 3 — PanelAssemblyJob + tasks ───────────────────────────────────────


# Default panel feature columns that should be cross-sectionally z-scored
# across tickers per date (per-ticker raw indicators whose absolute values
# aren't comparable across tickers — e.g. AAPL's RSI distribution ≠ BRK's).
# `trend*` / `rel_mom_*` live here because they are residualized against
# sector momentum (partial neutralization) but still carry per-ticker scale.
DEFAULT_CS_ZSCORE_COLS: list[str] = [
    "rsi", "adx", "williams_r", "bbp", "cci", "obv_slope", "macd_hist",
    "trend", "trend_long", "rel_mom_20d", "rel_mom_60d",
]

# Panel feature columns that should be dropped outright from the model
# input — either zero within-date variance (same value for every ticker on
# a given date, so no LTR ranking information) or uncomparable raw levels.
DEFAULT_DROP_COLS: list[str] = [
    "close",              # raw price level
    "spy_realized_vol",   # SPY context — same across tickers per date
    "spy_adx",            # ditto
    "spy_trend",          # ditto
    "hurst_proxy",        # ditto
]


class NeutralizedFeatureZScoreTask(PanelTask):
    """Cross-sectionally z-score per-ticker indicator columns per date.

    Without this step, raw indicators (RSI, ADX, CCI, …) enter the panel
    as per-ticker absolute values — the ranker then compares AAPL's RSI=30
    against BRK's RSI=30 as if they meant the same thing. Z-scoring within
    each date puts every ticker on a comparable scale.

    Only rewrites the columns listed in `panel_ltr.cs_zscore_cols`
    (defaults to DEFAULT_CS_ZSCORE_COLS). Columns that don't exist in the
    neutralized frames are skipped silently.
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        if not ctx.neutralized_frames:
            return
        from training_panel.factors import cross_sectional_zscore

        cfg       = ctx.config.get("panel_ltr", {})
        cs_cols   = list(cfg.get("cs_zscore_cols", DEFAULT_CS_ZSCORE_COLS))
        if not cs_cols:
            return

        present_cols: list[str] = []
        for col in cs_cols:
            if any(col in f.columns for f in ctx.neutralized_frames.values()):
                present_cols.append(col)

        for col in present_cols:
            per_ticker = {
                t: f[col] for t, f in ctx.neutralized_frames.items()
                if col in f.columns
            }
            z = cross_sectional_zscore(per_ticker)
            for t, frame in ctx.neutralized_frames.items():
                if t in z:
                    frame[col] = z[t].reindex(frame.index)

        log.info("NeutralizedFeatureZScoreTask: cross-sectionally z-scored %d cols  (%s)",
                 len(present_cols), present_cols)


class FactorZScoreTask(PanelTask):
    """Cross-sectional z-score of raw factor frames, per date.

    Z-scores four technical columns (size, mom_12_1, beta_60d, resid_mom).
    When ctx.fundamentals is populated, also z-scores four fundamental
    columns (earnings_yield, roe, gross_profitability, book_to_price) after
    same-sector median fill for missing values.
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.factor_frames:
            return
        if not ctx.raw_factor_frames:
            return
        from training_panel.factors import (
            cross_sectional_zscore,
            FUNDAMENTAL_COLS,
            _cross_sectional_zscore_static,
            _sector_median_fill,
        )

        raw_cols = [
            "size", "mom_12_1", "beta_60d", "resid_mom",
            # Round 3 orthogonal factors (time-series, same treatment as above)
            "amihud_illiq", "volume_shift", "price_to_high",
            "realized_vol", "drawdown_peak",
            # Round 4+ time-varying fundamentals (opt-in via config)
            "earnings_surprise_cum",
            # Round 5: SEC Form 4 executive-only insider trades (opt-in)
            "insider_net_buy_90d",
            # Plan G: hourly-bar aggregates (opt-in via panel_ltr.hourly.enabled)
            "morning_drift", "afternoon_drift", "vwap_premium",
            "vol_ratio", "intraday_realized_vol", "overnight_gap",
        ]
        per_col: dict[str, dict[str, pd.Series]] = {}
        for col in raw_cols:
            per_col[col] = {
                t: df[col] for t, df in ctx.raw_factor_frames.items()
                if col in df.columns
            }
        z = {col: cross_sectional_zscore(per_col[col]) for col in raw_cols if per_col[col]}

        # Fundamentals: static scalar per (ticker, col). Fill missing by
        # sector median, then cross-sectionally z-score across tickers once.
        fund_z_by_col: dict[str, dict[str, float]] = {}
        has_fundamentals = any(
            col in next(iter(ctx.raw_factor_frames.values())).columns
            for col in FUNDAMENTAL_COLS
        ) if ctx.raw_factor_frames else False
        if has_fundamentals:
            for col in FUNDAMENTAL_COLS:
                raw_vals = {}
                for t, df in ctx.raw_factor_frames.items():
                    if col in df.columns and not df[col].empty:
                        v = df[col].iloc[-1]   # broadcast scalar — any row works
                        raw_vals[t] = float(v) if pd.notna(v) else float("nan")
                    else:
                        raw_vals[t] = float("nan")
                filled = _sector_median_fill(raw_vals, ctx.ticker_sectors)
                fund_z_by_col[col] = _cross_sectional_zscore_static(filled)

        out: dict[str, pd.DataFrame] = {}
        for t, raw in ctx.raw_factor_frames.items():
            idx = raw.index
            cols: dict[str, pd.Series] = {
                "size_z":      z["size"].get(t,      pd.Series(index=idx)).reindex(idx),
                "mom_12_1_z":  z["mom_12_1"].get(t,  pd.Series(index=idx)).reindex(idx),
                "beta_60d_z":  z["beta_60d"].get(t,  pd.Series(index=idx)).reindex(idx),
                "resid_mom_z": z["resid_mom"].get(t, pd.Series(index=idx)).reindex(idx),
            }
            # Round 3+: append z-scored orthogonal factors
            for c in ("amihud_illiq", "volume_shift", "price_to_high",
                      "realized_vol", "drawdown_peak",
                      "earnings_surprise_cum", "insider_net_buy_90d",
                      # Plan G: hourly-bar aggregates
                      "morning_drift", "afternoon_drift", "vwap_premium",
                      "vol_ratio", "intraday_realized_vol", "overnight_gap"):
                if c in z:
                    cols[f"{c}_z"] = z[c].get(t, pd.Series(index=idx)).reindex(idx)
            for col in FUNDAMENTAL_COLS:
                if col in fund_z_by_col:
                    v = fund_z_by_col[col].get(t, float("nan"))
                    cols[f"{col}_z"] = pd.Series(v, index=idx)
            out[t] = pd.DataFrame(cols, index=idx)
        ctx.factor_frames = out
        log.info("FactorZScoreTask: z-scored %d factor frames (fundamentals=%s)",
                 len(out), has_fundamentals)


class LabelsTask(PanelTask):
    """Forward returns → purged β-neutral residuals → cross-sectional Gaussianize.

    Stores both the raw residuals (in ctx.raw_residuals, consumed by the
    optional NGBoost head) and the Gaussianized labels (in ctx.labels,
    consumed by the LTR model).
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.labels:
            return
        from training_panel.labels import (
            compute_residual_returns,
            gaussianize_cross_section,
        )

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

        ctx.raw_residuals = compute_residual_returns(
            fwd_returns, spy_fwd, sec_fwd_by_ticker,
            beta_window=beta_window, lookahead_days=lookahead,
        )
        ctx.labels = gaussianize_cross_section(ctx.raw_residuals)
        log.info("LabelsTask: built labels for %d tickers (raw residuals + gauss)",
                 len(ctx.labels))


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
            # Round 5: last-N-year slice + exponential recency weighting.
            # User spec: 5 years + half-life 252 trading days by default.
            training_window_years=cfg.get("training_window_years"),
            recency_weighting=cfg.get("recency_weighting"),
        )
        label_mask = panel["label"].notna()
        panel = panel[label_mask].reset_index(drop=True)
        group_sizes = panel.groupby("date", sort=True).size().values.astype(np.int32)

        # Attach raw residuals as a separate column for the NGBoost head.
        # Does not affect LTR training (exclude list below).
        if ctx.raw_residuals:
            raw_rows = []
            for t, s in ctx.raw_residuals.items():
                rr = pd.DataFrame({
                    "ticker": t,
                    "date":   pd.to_datetime(s.index),
                    "residual_return_raw": s.values,
                })
                raw_rows.append(rr)
            raw_df = pd.concat(raw_rows, ignore_index=True)
            panel["date"] = pd.to_datetime(panel["date"])
            panel = panel.merge(raw_df, on=["ticker", "date"], how="left")

        # User-provided drop_cols augments — does NOT replace — DEFAULT_DROP_COLS.
        # DEFAULT_DROP_COLS lists columns that are always bad LTR features
        # (raw levels, or same-value-across-tickers-per-date). Tree backends
        # are robust to raw `close`, but the transformer backend blew up
        # with NaN loss when fed the unnormalized close column — its std
        # is ~40x larger than every other (z-scored) feature, swamping the
        # attention softmax into overflow. Union ensures every backend sees
        # the same clean input.
        user_drop_cols = cfg.get("drop_cols")
        drop_cols = set(DEFAULT_DROP_COLS)
        if user_drop_cols is not None:
            drop_cols |= set(user_drop_cols)
        exclude = {"date", "ticker", "sector", "label",
                   "residual_return_raw",
                   "weight", "weight_concurrency", "weight_age",
                   "weight_recency"} | drop_cols
        feature_cols = [c for c in panel.columns if c not in exclude]
        if drop_cols & set(panel.columns):
            log.info("BuildPanelTask: dropped non-ranking cols %s",
                     sorted(drop_cols & set(panel.columns)))

        ctx.panel = panel
        ctx.group_sizes = group_sizes
        ctx.panel_metadata = meta
        ctx.feature_cols = feature_cols
        log.info("BuildPanelTask: panel=%d rows  features=%d  tickers=%d  dates=%d",
                 len(panel), len(feature_cols), meta.get("n_tickers"), meta.get("n_dates"))


class FeatureDiagnosticTask(PanelTask):
    """Log per-feature within-date std + Spearman IC vs label.

    Surfaces features that carry no cross-sectional information (std ≈ 0)
    or no predictive power (|IC| < 0.01) so they can be removed from
    `panel_ltr.drop_cols`. Pure diagnostic — does not modify the panel.
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.panel is None or not ctx.feature_cols:
            return
        from scipy.stats import spearmanr

        panel = ctx.panel
        dates = panel["date"].values
        label = panel["label"].values
        rows: list[tuple[str, float, float]] = []
        for col in ctx.feature_cols:
            vals = panel[col].values
            # Within-date std, averaged across dates (pandas is slow per-group;
            # use a group-by-transform once)
            df = panel[["date", col]].dropna(subset=[col])
            if df.empty:
                rows.append((col, 0.0, 0.0))
                continue
            std = df.groupby("date", sort=False)[col].transform("std").mean()
            # Per-date Spearman IC, averaged
            ics: list[float] = []
            for _, g in panel[["date", col, "label"]].dropna().groupby("date", sort=False):
                y = g["label"].values
                p = g[col].values
                if len(y) < 2 or (y == y[0]).all() or (p == p[0]).all():
                    continue
                rho, _ = spearmanr(p, y)
                if not np.isnan(rho):
                    ics.append(float(rho))
            mean_ic = float(np.mean(ics)) if ics else 0.0
            rows.append((col, float(std), mean_ic))

        rows.sort(key=lambda r: abs(r[2]), reverse=True)
        lines = [f"{c:<22s}  std={s:7.4f}  IC={i:+7.4f}" for c, s, i in rows]
        log.info("FeatureDiagnosticTask: per-feature within-date std + Spearman IC\n%s",
                 "\n".join("  " + ln for ln in lines))
        ctx.feature_diagnostics = [{"col": c, "within_date_std": s, "ic": i}
                                    for c, s, i in rows]


class PanelAssemblyJob(PanelJob):
    """Phase 3 — turn per-ticker outputs into a panel DataFrame.

    Task chain: NeutralizedFeatureZScore → FactorZScore → Labels → BuildPanel
    """

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        return ctx.panel is not None and bool(ctx.feature_cols)

    @property
    def tasks(self) -> list[PanelTask]:
        return [
            NeutralizedFeatureZScoreTask(),
            FactorZScoreTask(),
            LabelsTask(),
            BuildPanelTask(),
            FeatureDiagnosticTask(),
        ]


# ── Phase 4 — PanelModelJob + tasks ──────────────────────────────────────────

class CrossValidateTask(PanelTask):
    """Purged K-fold or Combinatorial Purged Spearman IC over the panel.

    Selected via `panel_ltr.cv_method`:
      - "purged" (default): PurgedKFold — 1 train/test split per fold
      - "cpcv":             Combinatorial Purged CV — C(n_splits, n_test_groups)
                             splits, yielding a distribution of IC estimates
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.cv_result:
            return
        from training_panel.purged_cv import (
            PurgedKFold, CombinatorialPurgedCV,
            cross_validated_ic, cross_validated_ic_cpcv,
        )

        cfg = ctx.config.get("panel_ltr", {})
        cv_method  = str(cfg.get("cv_method", "purged")).strip().lower()
        cv_splits  = int(cfg.get("cv_n_splits", 5))
        embargo    = int(cfg.get("cv_embargo_days", cfg.get("lookahead_days", 5)))
        lookahead  = int(cfg.get("lookahead_days", 5))
        num_rounds = int(cfg.get("num_boost_round", 400))
        backend    = str(cfg.get("backend", "xgboost")).strip().lower()

        panel = ctx.panel
        feature_cols = ctx.feature_cols

        if backend == "transformer":
            from training_panel.transformer_model import PanelTransformerModel
            tf_params = dict(cfg.get("transformer_params", {}))
            cv_epochs = max(int(cfg.get("num_boost_round", 50)) // 2, 5)

            class _SklearnAdapter:
                def __init__(self):
                    self._m = PanelTransformerModel(params=tf_params)
                    self._feature_cols: list[str] | None = None
                def fit(self, X, y, sample_weight=None):
                    df = X.copy()
                    df["label"] = y
                    df["date"] = panel.loc[X.index, "date"].values
                    df = df.sort_values(["date"], kind="mergesort").reset_index(drop=True)
                    gs = df.groupby("date", sort=True).size().values.astype(np.int32)
                    self._feature_cols = list(X.columns)
                    self._m.train(
                        df, gs, feature_cols=self._feature_cols,
                        label_col="label", weight_col=None,
                        num_boost_round=cv_epochs,
                    )
                def predict(self, X):
                    # Transformer predict requires a `date` column (or an
                    # explicit group_sizes) to batch rows into date-groups.
                    # The CV caller passes X with no date column, so we
                    # attach it here from the parent panel, then sort so
                    # groups are contiguous and aligned with the row order
                    # the model expects.
                    df = X.copy()
                    df["date"] = panel.loc[X.index, "date"].values
                    df = df.sort_values(["date"], kind="mergesort")
                    original_index = df.index
                    df = df.reset_index(drop=True)
                    preds = self._m.predict(df)
                    # Realign predictions to X's original index order.
                    preds.index = original_index
                    return preds.reindex(X.index).values
        elif backend == "lightgbm":
            from training_panel.lgbm_ltr import PanelLGBMModel
            params = dict(cfg.get("lightgbm_params", {}))

            class _SklearnAdapter:
                def __init__(self):
                    self._m = PanelLGBMModel(params=params, feature_cols=feature_cols)
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
        else:
            from training_panel.ltr_model import PanelLTRModel
            xgb_params = dict(cfg.get("xgb_params", {}))
            monotone = dict(cfg.get("monotone_constraints", {}))

            class _SklearnAdapter:
                def __init__(self):
                    self._m = PanelLTRModel(
                        params=xgb_params,
                        monotone_constraints=monotone,
                    )
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

        if cv_method == "cpcv":
            n_test_groups = int(cfg.get("cv_n_test_groups", 2))
            cv = CombinatorialPurgedCV(
                n_splits=cv_splits, n_test_groups=n_test_groups,
                embargo_days=embargo, lookahead_days=lookahead,
            )
            ctx.cv_result = cross_validated_ic_cpcv(
                _SklearnAdapter, panel, feature_cols, "label", cv,
                weight_col="weight",
            )
            q = ctx.cv_result.get("quantiles", {})
            log.info(
                "CrossValidateTask[cpcv]: mean=%+.4f std=%.4f "
                "q05=%+.4f q50=%+.4f q95=%+.4f n_splits=%d",
                ctx.cv_result["mean_ic"], ctx.cv_result["std_ic"],
                q.get("q05", 0.0), q.get("q50", 0.0), q.get("q95", 0.0),
                len(ctx.cv_result["per_fold_ic"]),
            )
        else:
            cv = PurgedKFold(
                n_splits=cv_splits, embargo_days=embargo, lookahead_days=lookahead,
            )
            ctx.cv_result = cross_validated_ic(
                _SklearnAdapter, panel, feature_cols, "label", cv,
                weight_col="weight",
            )
            log.info("CrossValidateTask[purged]: mean_ic=%+.4f  std=%.4f",
                     ctx.cv_result["mean_ic"], ctx.cv_result["std_ic"])


class FinalFitTask(PanelTask):
    """Fit the final panel model on the full panel.

    Backend selected via `panel_ltr.backend`:
      - `"xgboost"` (default): rank:pairwise via `PanelLTRModel`
      - `"lightgbm"`:            LambdaRank@10 via `PanelLGBMModel`
      - `"transformer"`:         cross-sectional attention via `PanelTransformerModel`
                                  (MPS/CUDA/CPU; hparams in `panel_ltr.transformer_params`)
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.final_model is not None:
            return

        import time as _time
        cfg     = ctx.config.get("panel_ltr", {})
        backend = str(cfg.get("backend", "xgboost")).strip().lower()
        num_rounds = int(cfg.get("num_boost_round", 400))

        t0 = _time.monotonic()
        if backend == "transformer":
            from training_panel.transformer_model import PanelTransformerModel
            tf_params = dict(cfg.get("transformer_params", {}))
            model = PanelTransformerModel(params=tf_params)
            fit = model.train(
                ctx.panel, ctx.group_sizes,
                feature_cols=ctx.feature_cols,
                label_col="label", weight_col=None,
                num_boost_round=int(tf_params.get("max_epochs", 50)),
            )
            device_used = str(model._device)    # noqa: SLF001 — surface actual device
        elif backend == "lightgbm":
            from training_panel.lgbm_ltr import PanelLGBMModel
            params = dict(cfg.get("lightgbm_params", {}))
            model = PanelLGBMModel(params=params, feature_cols=ctx.feature_cols)
            fit = model.train(
                ctx.panel, ctx.group_sizes,
                feature_cols=ctx.feature_cols,
                label_col="label", weight_col="weight",
                num_boost_round=num_rounds,
            )
            device_used = "cpu"
        else:
            from training_panel.ltr_model import PanelLTRModel
            xgb_params = dict(cfg.get("xgb_params", {}))
            monotone = dict(cfg.get("monotone_constraints", {}))
            model = PanelLTRModel(params=xgb_params, monotone_constraints=monotone)
            fit = model.train(
                ctx.panel, ctx.group_sizes,
                feature_cols=ctx.feature_cols,
                label_col="label", weight_col="weight",
                num_boost_round=num_rounds,
            )
            device_used = "cpu"
        elapsed = _time.monotonic() - t0
        ctx.final_model = model
        ctx._final_fit = fit  # noqa: SLF001 — read by SaveArtifactTask
        ctx._final_fit_elapsed_sec = elapsed  # noqa: SLF001
        ctx._final_fit_device      = device_used  # noqa: SLF001
        log.info("FinalFitTask: backend=%s  train_ic=%+.4f  elapsed=%.1fs  device=%s",
                 backend, fit.get("train_ic", 0.0), elapsed, device_used)


class SaveArtifactTask(PanelTask):
    """Write the JSON artifact with CV metadata + populate ctx.summary."""

    def run(self, ctx: PanelTrainingContext) -> None:
        cfg = ctx.config.get("panel_ltr", {})
        backend = str(cfg.get("backend", "xgboost")).strip().lower()

        # Transformer artifacts are .pt + .json sidecar pairs. To avoid
        # clobbering the XGBoost `panel-ltr.json` when training parallel
        # backends, route the transformer artifact to a distinct default
        # path (`panel-transformer.pt`) unless the user explicitly
        # overrode `panel_ltr.transformer_artifact_path`.
        if backend == "transformer":
            default_tf = cfg.get("transformer_artifact_path", "panel-transformer.pt")
            out_path = Path(default_tf)
        else:
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
            "oos_ic_quantiles": ctx.cv_result.get("quantiles"),   # only present for CPCV
            "training_train_ic": fit.get("train_ic", 0.0),
            "training_notes":  cfg.get("training_notes", "Stage-1 panel pipeline"),
            "neutralize_features": cfg.get("neutralize_features", True),
            "lookahead_days":  cfg.get("lookahead_days", 5),
            "beta_window":     cfg.get("beta_window", 60),
            "min_history_days": cfg.get("min_history_days", 252),
            "cv_method":       cfg.get("cv_method", "purged"),
            "cv_n_splits":     cfg.get("cv_n_splits", 5),
            "cv_n_test_groups": cfg.get("cv_n_test_groups"),
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

        # Record to training_runs table + JSONL (user spec: track time +
        # data volume of every retrain for audit/trend analysis).
        try:
            from kernel.persistence import get_connection, record_training_run  # noqa: PLC0415
            conn = get_connection(ctx.config, strategy_dir=ctx.strategy_dir)
            record_training_run(
                conn,
                strategy             = ctx.config.get("_strategy_name", "renquant_104"),
                artifact_type        = "panel-ltr",
                config_snapshot      = {"panel_ltr": cfg},
                oos_mean_ic          = ctx.cv_result.get("mean_ic"),
                train_ic             = (getattr(ctx, "_final_fit", {}) or {}).get("train_ic"),
                n_rows               = int(ctx.panel_metadata.get("n_rows", len(ctx.panel))),
                n_tickers            = int(ctx.panel_metadata.get("n_tickers", 0)),
                n_dates              = int(ctx.panel_metadata.get("n_dates", 0)),
                n_features           = len(ctx.feature_cols),
                feature_cols         = list(ctx.feature_cols),
                artifact_path        = str(out_path),
                elapsed_sec          = getattr(ctx, "_final_fit_elapsed_sec", None),
                trigger              = ctx.config.get("_training_trigger", "manual"),
                device               = getattr(ctx, "_final_fit_device", "cpu"),
                deterministic        = bool(cfg.get("deterministic", False)),
                training_window_years= cfg.get("training_window_years"),
            )
        except Exception as exc:
            log.warning("SaveArtifactTask: record_training_run failed — %s", exc)


class PanelModelJob(PanelJob):
    """Phase 4 — cross-validate, fit final, save artifact.

    Task chain: CrossValidate → FinalFit → SaveArtifact
    """

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        return ctx.final_model is not None and ctx.artifact_path is not None

    @property
    def tasks(self) -> list[PanelTask]:
        return [CrossValidateTask(), FinalFitTask(), SaveArtifactTask()]


# ── Phase 5 — PanelNGBoostJob (optional) ─────────────────────────────────────

class NGBoostFitTask(PanelTask):
    """Fit the NGBoostHead on raw residual forward returns.

    Reads the `residual_return_raw` column attached to the panel by
    BuildPanelTask. No-op if the column is missing or the config flag is off.
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.ngboost_head is not None:
            return
        cfg = ctx.config.get("panel_ltr", {}).get("ngboost", {})
        if not cfg.get("enabled", False):
            log.info("NGBoostFitTask: panel_ltr.ngboost.enabled=false — skipping")
            return
        if ctx.panel is None or "residual_return_raw" not in ctx.panel.columns:
            log.warning("NGBoostFitTask: panel missing residual_return_raw — skipping")
            return

        # Drop rows whose raw residual is NaN (insufficient beta history)
        sub = ctx.panel.dropna(subset=["residual_return_raw"])
        if sub.empty:
            log.warning("NGBoostFitTask: no non-NaN residual rows — skipping")
            return

        from training_panel.ngboost_head import NGBoostHead
        import time as _time

        params = dict(cfg.get("params", {}))
        head = NGBoostHead(params=params)
        t0 = _time.monotonic()
        fit = head.train(
            sub,
            feature_cols=ctx.feature_cols,
            label_col="residual_return_raw",
            sample_weight_col="weight" if "weight" in sub.columns else None,
        )
        elapsed = _time.monotonic() - t0
        ctx.ngboost_head = head
        ctx.ngboost_fit = fit
        ctx._ngboost_elapsed_sec = elapsed  # noqa: SLF001
        log.info("NGBoostFitTask: trained on %d rows  μ_mean=%.5f  σ_mean=%.5f  elapsed=%.1fs",
                 fit["n_rows"], fit["train_mu_mean"], fit["train_sigma_mean"], elapsed)


class NGBoostSaveTask(PanelTask):
    """Persist the NGBoost head next to the LTR artifact."""

    def run(self, ctx: PanelTrainingContext) -> None:
        if ctx.ngboost_head is None:
            return
        cfg = ctx.config.get("panel_ltr", {}).get("ngboost", {})
        out_name = cfg.get("artifact_path", "ngboost-head.json")
        out_path = Path(out_name)
        if ctx.strategy_dir and not out_path.is_absolute():
            out_path = ctx.strategy_dir / "artifacts" / out_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "training_notes": cfg.get("training_notes", "Stage-2 NGBoost head"),
            "train_mu_mean":   ctx.ngboost_fit.get("train_mu_mean"),
            "train_sigma_mean": ctx.ngboost_fit.get("train_sigma_mean"),
            "n_rows":          ctx.ngboost_fit.get("n_rows"),
        }
        ctx.ngboost_head.save(out_path, metadata=meta)
        ctx.ngboost_artifact_path = out_path
        log.info("NGBoostSaveTask: artifact → %s", out_path)

        try:
            from kernel.persistence import get_connection, record_training_run  # noqa: PLC0415
            conn = get_connection(ctx.config, strategy_dir=ctx.strategy_dir)
            record_training_run(
                conn,
                strategy        = ctx.config.get("_strategy_name", "renquant_104"),
                artifact_type   = "ngboost-head",
                config_snapshot = {"ngboost": cfg},
                n_rows          = ctx.ngboost_fit.get("n_rows"),
                n_features      = len(ctx.feature_cols),
                feature_cols    = list(ctx.feature_cols),
                artifact_path   = str(out_path),
                elapsed_sec     = getattr(ctx, "_ngboost_elapsed_sec", None),
                trigger         = ctx.config.get("_training_trigger", "manual"),
                device          = "cpu",
                deterministic   = False,  # NGBoost isn't deterministic by default
                notes           = (f"μ̄={ctx.ngboost_fit.get('train_mu_mean'):+.5f} "
                                   f"σ̄={ctx.ngboost_fit.get('train_sigma_mean'):.5f}"),
            )
        except Exception as exc:
            log.warning("NGBoostSaveTask: record_training_run failed — %s", exc)


class PanelNGBoostJob(PanelJob):
    """Phase 5 — train + save NGBoost Normal(μ, σ) head.

    Task chain: NGBoostFit → NGBoostSave

    Skipped entirely when `panel_ltr.ngboost.enabled` is false (default).
    """

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        cfg = ctx.config.get("panel_ltr", {}).get("ngboost", {})
        if not cfg.get("enabled", False):
            return True
        return (ctx.ngboost_head is not None
                and ctx.ngboost_artifact_path is not None)

    @property
    def tasks(self) -> list[PanelTask]:
        return [NGBoostFitTask(), NGBoostSaveTask()]


# ── Orchestrator ─────────────────────────────────────────────────────────────

class PanelTrainingPipeline:
    """Five-phase panel training pipeline (LTR + optional NGBoost head)."""

    def run(self, ctx: PanelTrainingContext) -> PanelTrainingContext:
        jobs: list[PanelJob] = [
            PanelDataJob(),       # FetchOHLCV + SectorMomentum
            PanelFeatureJob(),    # parallel per-ticker chain
            PanelAssemblyJob(),   # FactorZScore + Labels + BuildPanel
            PanelModelJob(),      # CrossValidate + FinalFit + SaveArtifact
            PanelNGBoostJob(),    # NGBoostFit + NGBoostSave (optional)
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
