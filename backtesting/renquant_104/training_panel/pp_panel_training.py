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

import json
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


def _resolve_cache_dir(cfg_value: str, ctx_config: dict) -> Path:
    """Resolve a Load*Task cache_dir to an absolute path.

    Audit P-37 (2026-04-24): two `data/<cache>/` directories existed —
    one at repo root, one at strategy_dir/data/<cache>/, with different
    contents. The relative-path resolver picks whichever happens to be
    under cwd, so notebook/LEAN/live see different data.

    Audit fix CACHE-DIR-SNAPSHOT (2026-04-26): in sim A/B mode, the
    snapshot wraps strategy_dir in a tmpdir (e.g.
    `/var/folders/.../renquant_ab_snapshot_xxx/`), so
    `strategy_dir.parent.parent` yields a non-existent tmp ancestor —
    NOT the actual repo root where `data/fundamentals/...` lives.
    Pre-fix, this caused LoadFundamentalsTask: 0/101 (silent miss)
    every sim run. Fix: when the strategy_dir-derived path doesn't
    exist, fall back to the cwd-derived path before giving up.

    Resolution order:
      1. cfg_value is absolute → use as-is
      2. ctx.config has `_strategy_dir` → resolve relative to its
         repo_root (= strategy_dir.parent.parent).
      3. If (2) doesn't exist on disk, fall back to cwd-relative.
      4. Final fallback: return whichever of (2) or (3) existed last.
    """
    p = Path(cfg_value)
    if p.is_absolute():
        return p
    strategy_dir = ctx_config.get("_strategy_dir") if isinstance(ctx_config, dict) else None
    if strategy_dir:
        repo_root = Path(strategy_dir).parent.parent
        derived = repo_root / p
        if derived.exists():
            return derived
        # Snapshot edge case: tmpdir ancestor doesn't have data/. Try cwd.
        cwd_candidate = Path.cwd() / p
        if cwd_candidate.exists():
            log.info(
                "_resolve_cache_dir: strategy_dir-derived path %s missing "
                "(snapshot context); falling back to cwd-relative %s",
                derived, cwd_candidate,
            )
            return cwd_candidate
        # Neither exists — return derived so caller's not-found logic
        # handles it (typical no-op skip for missing cache).
        return derived
    log.warning("_resolve_cache_dir: ctx config missing _strategy_dir; "
                "falling back to cwd-relative path %s", p)
    return Path.cwd() / p if not p.is_absolute() else p


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

        cache_dir = _resolve_cache_dir(cfg.get("cache_dir", "data/fundamentals"), ctx.config)
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
        # Bug 16 fix: inference_only path NEVER fetches — read from cache only.
        if missing and cfg.get("allow_fetch", True) and not getattr(ctx, "inference_only", False):
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

        cache_dir = _resolve_cache_dir(cfg.get("cache_dir", "data/earnings_surprise"), ctx.config)
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
        # Bug 16 fix: inference_only path NEVER fetches — read from cache only.
        if missing and cfg.get("allow_fetch", True) and not getattr(ctx, "inference_only", False):
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

        cache_dir = _resolve_cache_dir(cfg.get("cache_dir", "data/insider_trades"), ctx.config)
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
        # Bug 16 fix: inference_only path NEVER fetches — read from cache only.
        if missing and cfg.get("allow_fetch", True) and not getattr(ctx, "inference_only", False):
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

        cache_dir = _resolve_cache_dir(cfg.get("cache_dir", "data/intraday"), ctx.config)
        store = HourlyBarStore(data_dir=cache_dir)

        out: dict[str, pd.DataFrame] = {}
        for sym in ctx.watchlist:
            df = store.load(sym)
            if df is not None and not df.empty:
                out[sym] = df

        ctx.hourly_bars = out
        log.info("LoadHourlyBarsTask: %d / %d tickers with hourly bars",
                 len(out), len(ctx.watchlist))


class LoadMinuteBarsTask(PanelTask):
    """Populate ctx.minute_bars from the parquet cache (2026-04-24).

    Parallel to LoadHourlyBarsTask but reads 10-min bars. No-op unless
    `panel_ltr.minute.enabled` is true. Cache at
    `data/intraday/{SYMBOL}/10min.parquet`; `scripts/fetch_minute_bars.py`
    owns populating it from Alpaca IEX. Training never fetches live.
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        cfg = ctx.config.get("panel_ltr", {}).get("minute", {})
        if not cfg.get("enabled", False):
            return
        if ctx.minute_bars:
            return

        from kernel.intraday import MinuteBarStore  # noqa: PLC0415

        cache_dir = _resolve_cache_dir(cfg.get("cache_dir", "data/intraday"), ctx.config)
        store = MinuteBarStore(data_dir=cache_dir)

        out: dict[str, pd.DataFrame] = {}
        for sym in ctx.watchlist:
            df = store.load(sym)
            if df is not None and not df.empty:
                out[sym] = df

        ctx.minute_bars = out
        log.info("LoadMinuteBarsTask: %d / %d tickers with 10-min bars",
                 len(out), len(ctx.watchlist))


class LoadMacroFactorsTask(PanelTask):
    """Populate ctx.macro_factor_frame from the macro parquet cache.

    Phase 1B (2026-04-26 round-7) of the macro_factor_frame project.
    See doc/components/macro-factor-frame-design.md for full design.

    No-op when `panel_ltr.macro.enabled` is false (default — ships off).
    Cache at `data/macro/{SYMBOL}.parquet`; `scripts/fetch_macro_factors.py`
    populates from yfinance. This task never fetches live — training
    must be reproducible offline.

    Safety harness applied:
    - F1 per-symbol load isolation (build_macro_frame uses try/except)
    - F4 short-window dropping via min_window_overlap_pct
    - F5 z-score zero-variance clamp inside _rolling_z
    - F9 corrupt parquet → cache-miss

    On any exception at the task level, logs WARN and leaves
    ctx.macro_factor_frame as None — pipeline proceeds in no-macro mode.
    """

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        cfg = ctx.config.get("panel_ltr", {}).get("macro", {})
        if not cfg.get("enabled", False):
            return True   # default off
        if ctx.macro_factor_frame is not None and not ctx.macro_factor_frame.empty:
            return True   # pre-populated (testing path or inference cache)

        try:
            from kernel.macro import (  # noqa: PLC0415
                MacroFactorStore, build_macro_frame,
                DEFAULT_MACRO_SYMBOLS, DEFAULT_TRANSFORMS,
                DEFAULT_ROLLING_WINDOW,
            )

            cache_dir = _resolve_cache_dir(
                cfg.get("cache_dir", "data/macro"), ctx.config,
            )
            store = MacroFactorStore(data_dir=cache_dir)

            symbols = cfg.get("symbols", DEFAULT_MACRO_SYMBOLS)
            transforms = cfg.get("transforms", DEFAULT_TRANSFORMS)
            rolling_window = int(cfg.get("rolling_window", DEFAULT_ROLLING_WINDOW))
            min_overlap = float(cfg.get("min_window_overlap_pct", 0.95))

            # Compute training_end from panel data — use the latest date
            # we have for the SPY benchmark as a proxy.
            training_end = None
            spy = ctx.spy_df
            if spy is not None and not spy.empty:
                training_end = pd.Timestamp(spy.index.max())

            frame, metadata = build_macro_frame(
                store,
                symbols=symbols,
                transforms=transforms,
                rolling_window=rolling_window,
                min_window_overlap_pct=min_overlap,
                training_end=training_end,
            )
            ctx.macro_factor_frame = frame
            ctx.macro_metadata = metadata
            log.info(
                "LoadMacroFactorsTask: %d features (%d symbols used, %d skipped)",
                metadata.get("n_features", 0),
                len(metadata.get("symbols_used", [])),
                len(metadata.get("symbols_skipped", [])),
            )
        except Exception as exc:
            log.warning(
                "LoadMacroFactorsTask: load failed (%s) — proceeding in no-macro "
                "mode (ctx.macro_factor_frame stays None)", exc,
            )
            ctx.macro_factor_frame = None
        return True


class LoadFredMacroTask(PanelTask):
    """Tier 2 macro expansion (2026-04-27): FRED API ingestion.

    Reads `panel_ltr.fred_macro.enabled` (default False — opt-in). When
    on, builds a FRED-derived feature frame using
    `kernel.fred_macro.build_fred_frame` and concatenates its columns
    into `ctx.macro_factor_frame`. Downstream `LoadMacroPerTickerBetasTask`
    then computes per-ticker β to FRED returns alongside ETF returns
    without any further changes.

    Data layout: 22 default series × 3 transforms (level_z, chg_5d_z,
    chg_20d_z) = 66 broadcast cols. Merged onto the existing macro
    frame by date — no rename needed since FRED column names are
    series IDs lowercased (no overlap with ETF symbols).

    Look-ahead safety: `build_fred_frame` applies release-lag shifts
    in TRADING DAYS via `_to_daily_bars` (mirrors the HIGH-1 lesson
    of bar-based shifts, never calendar-day Timedelta). Monthly series
    lag 5 trading days, weekly 2.

    Quietly no-ops if config disabled, the FRED key is absent, or the
    cache is empty — so CI/test runs without a key still pass.
    """

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        cfg = ctx.config.get("panel_ltr", {}).get("fred_macro", {})
        if not cfg.get("enabled", False):
            return True
        try:
            from kernel.fred_macro import (   # noqa: PLC0415
                FredMacroStore, build_fred_frame, DEFAULT_FRED_SERIES,
                DEFAULT_ROLLING_WINDOW, _resolve_api_key,
            )
        except ImportError as exc:
            log.warning("LoadFredMacroTask: import failed (%s) — skip", exc)
            return True

        if _resolve_api_key() is None:
            log.info("LoadFredMacroTask: no FRED API key (env var or "
                     "~/.fred_api_key) — skip; cache-only path requires "
                     "data/fred/*.parquet to exist")

        cache_dir = _resolve_cache_dir(cfg.get("cache_dir", "data/fred"), ctx.config)
        # Pass api_key=None so the store loads cached parquet without
        # requiring a key. Fetching is a separate operator step
        # (scripts/fetch_fred_macro.py).
        store = FredMacroStore(cache_dir=cache_dir, api_key=None)

        # Determine target index: union of all OHLCV trading days, or
        # SPY's index as proxy if available.
        spy = ctx.spy_df
        if spy is None or spy.empty:
            log.warning("LoadFredMacroTask: no SPY index available — skip")
            return True
        target_index = pd.DatetimeIndex(spy.index)

        rolling_window = int(cfg.get("rolling_window", DEFAULT_ROLLING_WINDOW))
        min_overlap = float(cfg.get("min_window_overlap_pct", 0.95))
        training_end = pd.Timestamp(target_index.max())

        # Allow operator to override the series catalog
        cfg_series = cfg.get("series")
        if cfg_series:
            # cfg_series is list[str] of just IDs. Pad with default freq/lag.
            id_to_spec = {s[0]: s for s in DEFAULT_FRED_SERIES}
            specs = [id_to_spec.get(sid, (sid, sid, "daily", 0)) for sid in cfg_series]
        else:
            specs = list(DEFAULT_FRED_SERIES)

        try:
            fred_frame, fred_meta = build_fred_frame(
                store, target_index,
                series_specs=specs,
                rolling_window=rolling_window,
                min_window_overlap_pct=min_overlap,
                training_end=training_end,
            )
        except Exception as exc:
            log.warning("LoadFredMacroTask: build_fred_frame failed (%s) — skip", exc)
            return True

        if fred_frame.empty or fred_frame.shape[1] == 0:
            log.info("LoadFredMacroTask: empty FRED frame — skip")
            return True

        # Merge into ctx.macro_factor_frame so downstream macro v2 path
        # picks up the FRED columns automatically.
        if ctx.macro_factor_frame is None or ctx.macro_factor_frame.empty:
            ctx.macro_factor_frame = fred_frame
        else:
            existing = ctx.macro_factor_frame
            # Align on the union of dates; FRED frame already on target_index
            existing = existing.reindex(target_index).ffill()
            fred_aligned = fred_frame.reindex(target_index)
            ctx.macro_factor_frame = pd.concat([existing, fred_aligned], axis=1)

        log.info(
            "LoadFredMacroTask: merged %d FRED features (%d series used, %d skipped)",
            fred_meta.get("n_features", 0),
            len(fred_meta.get("series_used", [])),
            len(fred_meta.get("series_skipped", [])),
        )
        return True


class LoadMacroPerTickerBetasTask(PanelTask):
    """Macro v2 (2026-04-27): compute per-ticker rolling β to macro factors.

    Reads `ctx.ohlcv` + `ctx.macro_factor_frame` (already populated by
    LoadMacroFactorsTask). For each ticker, produces a DataFrame of
    rolling 60d β to each macro factor — values DIFFER per ticker on
    same date so they enter the cross-sectional rank loss properly.

    No-op when `panel_ltr.macro.version != "v2"` (default v1 → broadcast).
    Also no-op when macro_factor_frame is None / empty (covers default
    `panel_ltr.macro.enabled: false` case).

    See doc/components/macro-factor-frame-redesign.md.
    """

    def run(self, ctx: PanelTrainingContext) -> bool:
        cfg = ctx.config.get("panel_ltr", {}).get("macro", {})
        version = str(cfg.get("version", "v1")).lower()
        if version != "v2":
            return True
        if ctx.macro_factor_frame is None or ctx.macro_factor_frame.empty:
            return True
        try:
            from kernel.macro_per_ticker import (  # noqa: PLC0415
                DEFAULT_MIN_WINDOW,
                DEFAULT_ROLLING_WINDOW,
                compute_per_ticker_macro_betas,
                macro_levels_to_returns,
            )
            macro_returns = macro_levels_to_returns(ctx.macro_factor_frame)
            if macro_returns.empty:
                log.warning("LoadMacroPerTickerBetasTask: macro_levels_to_returns "
                            "produced empty DataFrame — skipping (likely no "
                            "*_level_z columns in v1 macro frame)")
                return True
            # Audit M3 fix (2026-04-27): reference module-level defaults so
            # the cfg fallback can't drift from the function's own default.
            window = int(cfg.get("rolling_window", DEFAULT_ROLLING_WINDOW))
            betas = compute_per_ticker_macro_betas(
                ctx.ohlcv, macro_returns,
                rolling_window=window,
                min_window=int(cfg.get("min_window", DEFAULT_MIN_WINDOW)),
            )
            ctx.macro_betas = betas
            n_cols = len(next(iter(betas.values())).columns) if betas else 0
            log.info("LoadMacroPerTickerBetasTask: built per-ticker β for "
                     "%d/%d tickers (%d cols each, window=%dd)",
                     len(betas), len(ctx.ohlcv), n_cols, window)
        except Exception as exc:
            log.warning("LoadMacroPerTickerBetasTask: failed — %s. "
                        "Pipeline proceeds with macro_betas={} (no v2 features)",
                        exc)
            ctx.macro_betas = {}
        return True


class LoadAssetEmbeddingsTask(PanelTask):
    """T2-2 (2026-04-27): load pre-trained asset embeddings.

    Reads `artifacts/asset-embeddings.json` (produced by
    scripts/train_asset_embeddings.py — runs weekly via cron). Skipped
    when `panel_ltr.asset_embeddings.enabled` is not true OR the
    artifact doesn't exist (no-op; no-feature case).

    See doc/components/asset-embeddings-design.md.
    """

    def run(self, ctx: PanelTrainingContext) -> bool:
        cfg = ctx.config.get("panel_ltr", {}).get("asset_embeddings", {})
        if not cfg.get("enabled", False):
            return True
        from pathlib import Path as _Path  # noqa: PLC0415
        strategy_dir = ctx.config.get("_strategy_dir")
        if strategy_dir is None:
            return True
        path = _Path(strategy_dir) / cfg.get(
            "artifact_path", "artifacts/asset-embeddings.json"
        )
        if not path.exists():
            log.info("LoadAssetEmbeddingsTask: artifact missing at %s; "
                     "pipeline proceeds without embeddings", path)
            return True
        try:
            from training_panel.asset_embeddings import (  # noqa: PLC0415
                AssetEmbeddingTrainer,
                load_embeddings_for_inference,
            )
            # Audit 2nd-round #2 fix (2026-04-27): expose staleness as a
            # ctx field so acceptance gates / dashboard can surface to
            # operator. Pre-fix, only logged warning was emitted.
            max_age_days = int(cfg.get("max_age_days", 14))
            try:
                trainer = AssetEmbeddingTrainer.load(path)
                if trainer.trained_date:
                    age_days = (
                        pd.Timestamp.utcnow().tz_localize(None).date()
                        - pd.Timestamp(trainer.trained_date).date()
                    ).days
                    ctx.asset_embeddings_age_days = int(age_days)  # type: ignore
                    if age_days > max_age_days:
                        log.warning(
                            "LoadAssetEmbeddingsTask: embeddings STALE "
                            "(%dd old > %dd threshold) — features may be "
                            "miscalibrated; re-run scripts/train_asset_embeddings.py",
                            age_days, max_age_days,
                        )
                ctx.asset_embeddings = trainer.embeddings
            except Exception:
                # Fallback to legacy loader without staleness exposure
                ctx.asset_embeddings = load_embeddings_for_inference(
                    path, max_age_days=max_age_days,
                )
            log.info("LoadAssetEmbeddingsTask: loaded embeddings for %d tickers",
                     len(ctx.asset_embeddings))
        except Exception as exc:
            # Audit 2nd-round #10 fix (2026-04-27): elevate to ERROR log
            # so operator can't miss corrupted artifact. Pipeline still
            # proceeds (degraded gracefully), but error is loud.
            log.error("LoadAssetEmbeddingsTask: load FAILED — %s. "
                      "Pipeline proceeds without embeddings (DEGRADED). "
                      "Check artifact integrity at %s",
                      exc, path)
            ctx.asset_embeddings = {}
        return True


class PanelDataJob(PanelJob):
    """Phase 1 — gather market data + sector momentum + fundamentals + earnings + insiders + hourly + minute + macro + asset_embeddings.

    Task chain: FetchOHLCV → SectorMomentum → LoadFundamentals
                → LoadEarningsSurprise → LoadInsiderTrades → LoadHourlyBars
                → LoadMinuteBars → LoadMacroFactors → LoadMacroPerTickerBetas
                → LoadAssetEmbeddings
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
            LoadMinuteBarsTask(),
            LoadMacroFactorsTask(),
            LoadFredMacroTask(),               # Tier 2 (2026-04-27)
            LoadMacroPerTickerBetasTask(),
            LoadAssetEmbeddingsTask(),
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
                minute_bars=ctx.minute_bars,
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

        # Macro v2 (2026-04-27): merge per-ticker β into raw_factor_frames
        # so they go through FactorZScoreTask (cross-sectional z-score per
        # date) along with size_z / mom_12_1_z / etc. β values DIFFER per
        # ticker on same date → enter rank loss properly.
        if ctx.macro_betas:
            n_merged = 0
            n_collision = 0
            for ticker, beta_df in ctx.macro_betas.items():
                if ticker not in ctx.raw_factor_frames or beta_df.empty:
                    continue
                fac = ctx.raw_factor_frames[ticker]
                # Reindex β to factor frame's index then concat columns
                beta_aligned = beta_df.reindex(fac.index)
                # Drop columns whose names already exist (collision guard)
                existing = set(fac.columns)
                new_cols = [c for c in beta_aligned.columns if c not in existing]
                # Audit 2nd-round #3 fix (2026-04-27): warn on collision —
                # silently dropped columns hide misconfig (v1 + v2 both
                # adding beta_*).
                dropped = [c for c in beta_aligned.columns if c in existing]
                if dropped:
                    n_collision += 1
                    log.warning(
                        "PanelFeatureJob[macro v2]: %s — dropped %d β columns "
                        "due to name collision: %s. Verify FactorZScoreTask "
                        "isn't already producing same-named columns.",
                        ticker, len(dropped), dropped[:3],
                    )
                if new_cols:
                    ctx.raw_factor_frames[ticker] = pd.concat(
                        [fac, beta_aligned[new_cols]], axis=1, copy=False,
                    )
                    n_merged += 1
            log.info(
                "PanelFeatureJob[macro v2]: merged per-ticker β into %d/%d "
                "raw_factor_frames (collisions on %d)",
                n_merged, len(ctx.raw_factor_frames), n_collision,
            )

        n_in = len(ticker_ctxs)
        n_feat   = len(ctx.feature_frames)
        n_neut   = len(ctx.neutralized_frames)
        n_factor = len(ctx.raw_factor_frames)
        log.info(
            "PanelFeatureJob: in=%d  feat=%d  neutralized=%d  raw_factors=%d",
            n_in, n_feat, n_neut, n_factor,
        )
        # Audit fix TPF-1 (2026-04-25): pre-fix the per-ticker chain
        # silently dropped tickers on exception. Now: surface the count
        # AND abort the panel phase if too many failed (>5% threshold).
        # Prevents training on a depleted universe without operator
        # awareness — same pattern as D-8 in FetchPanelDataTask.
        n_failed = n_in - n_factor   # raw_factor_frame is the union of all stages
        threshold = max(1, n_in // 20)
        if n_failed > threshold:
            log.error(
                "PanelFeatureJob: %d / %d ticker chains failed (>5%% — "
                "panel would silently shrink). Check per-ticker error "
                "logs above. Aborting panel phase.",
                n_failed, n_in,
            )
            raise RuntimeError(
                f"PanelFeatureJob: {n_failed}/{n_in} ticker chains failed; "
                f"refusing to train on a depleted universe (D-8 / TPF-1 guard)."
            )


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
        from kernel.earnings_surprise import compute_earnings_surprise_cum, compute_pead_features
        from kernel.insider_trades    import compute_insider_net_buy_cum

        # Audit P-15 (2026-04-24): granular per-stage try/except. Pre-fix
        # one blanket `try` swallowed every error → raw_factor_frame stayed
        # None → ticker silently dropped from the cross-sectional z-score.
        # Now: stage 1 (core factors) failure drops the ticker; stages 2-6
        # (fundamentals/earnings/insider/hourly/minute) failures only
        # NaN-fill those specific columns.
        idx = tc.ohlcv[tc.ticker].index
        cols: dict[str, pd.Series] = {}
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
            cols.update({
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
            })
        except Exception as exc:
            log.error("  %s: TickerPanelFactorJob[core_factors] failed — %s: %s "
                      "— ticker dropped from this panel pass",
                      tc.ticker, type(exc).__name__, exc)
            return   # without core factors no point continuing

        # Stage 2 — fundamentals (static scalar broadcast)
        try:
            # Fundamentals: broadcast the ticker's snapshot scalar to every bar.
            # A missing ticker → NaN series (FactorZScoreTask / sector-median
            # fill will handle it globally).
            if tc.fundamentals:
                ticker_fund = tc.fundamentals.get(tc.ticker, {})
                for col in FUNDAMENTAL_COLS:
                    val = ticker_fund.get(col, float("nan"))
                    cols[col] = pd.Series(val, index=idx)
        except Exception as exc:
            log.warning("  %s: TickerPanelFactorJob[fundamentals] failed — %s",
                        tc.ticker, exc)
            for col in FUNDAMENTAL_COLS:
                cols.setdefault(col, pd.Series(float("nan"), index=idx))

        # Stage 3 — earnings surprise (time-varying)
        try:
            if tc.earnings_surprises:
                surprise_daily = compute_earnings_surprise_cum(
                    {tc.ticker: tc.earnings_surprises.get(tc.ticker, pd.DataFrame())},
                    {tc.ticker: tc.ohlcv[tc.ticker]},
                ).get(tc.ticker)
                cols["earnings_surprise_cum"] = (
                    surprise_daily if surprise_daily is not None
                    else pd.Series(float("nan"), index=idx)
                )
        except Exception as exc:
            log.warning("  %s: TickerPanelFactorJob[earnings_surprise] failed — %s",
                        tc.ticker, exc)
            cols["earnings_surprise_cum"] = pd.Series(float("nan"), index=idx)

        # Stage 3b — PEAD enrichment (Track B: days_since + decay + signal)
        # Gated on `panel_ltr.pead.enabled` (default false). Three columns:
        # `days_since_earnings`, `pead_decay_weight`, `pead_signal`. The
        # last is the canonical PEAD alpha (Bernard-Thomas 1989, CJL 1996).
        # `panel_ltr.pead.feature_subset` (optional list) filters which
        # columns get added — used for ablation A/A (e.g. days_only,
        # signal_only). Default: all three.
        pead_cfg = tc.config.get("panel_ltr", {}).get("pead", {}) or {}
        if pead_cfg.get("enabled", False):
            pead_subset = pead_cfg.get("feature_subset") or [
                "days_since_earnings", "pead_decay_weight", "pead_signal",
            ]
            try:
                if tc.earnings_surprises:
                    days_d, decay_d, signal_d = compute_pead_features(
                        {tc.ticker: tc.earnings_surprises.get(tc.ticker, pd.DataFrame())},
                        {tc.ticker: tc.ohlcv[tc.ticker]},
                        decay_window_days=int(pead_cfg.get("decay_window_days", 60)),
                        max_window_days=int(pead_cfg.get("max_window_days", 90)),
                    )
                    if "days_since_earnings" in pead_subset:
                        cols["days_since_earnings"] = (days_d.get(tc.ticker)
                            if days_d.get(tc.ticker) is not None
                            else pd.Series(float("nan"), index=idx))
                    if "pead_decay_weight" in pead_subset:
                        cols["pead_decay_weight"] = (decay_d.get(tc.ticker)
                            if decay_d.get(tc.ticker) is not None
                            else pd.Series(float("nan"), index=idx))
                    if "pead_signal" in pead_subset:
                        cols["pead_signal"] = (signal_d.get(tc.ticker)
                            if signal_d.get(tc.ticker) is not None
                            else pd.Series(float("nan"), index=idx))
                else:
                    if "days_since_earnings" in pead_subset:
                        cols["days_since_earnings"] = pd.Series(float("nan"), index=idx)
                    if "pead_decay_weight" in pead_subset:
                        cols["pead_decay_weight"] = pd.Series(float("nan"), index=idx)
                    if "pead_signal" in pead_subset:
                        cols["pead_signal"] = pd.Series(float("nan"), index=idx)
            except Exception as exc:
                log.warning("  %s: TickerPanelFactorJob[pead] failed — %s",
                            tc.ticker, exc)
                if "days_since_earnings" in pead_subset:
                    cols["days_since_earnings"] = pd.Series(float("nan"), index=idx)
                if "pead_decay_weight" in pead_subset:
                    cols["pead_decay_weight"] = pd.Series(float("nan"), index=idx)
                if "pead_signal" in pead_subset:
                    cols["pead_signal"] = pd.Series(float("nan"), index=idx)

        # Stage 4 — insider trades (time-varying, trailing-90d)
        try:
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
        except Exception as exc:
            log.warning("  %s: TickerPanelFactorJob[insider_trades] failed — %s",
                        tc.ticker, exc)
            cols["insider_net_buy_90d"] = pd.Series(float("nan"), index=idx)

        # Stage 5 — hourly aggregates (P-13 reindex/tz fragility lives here too)
        try:
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
        except Exception as exc:
            log.warning("  %s: TickerPanelFactorJob[hourly] failed — %s",
                        tc.ticker, exc)
            for col in ("morning_drift", "afternoon_drift", "vwap_premium",
                         "vol_ratio", "intraday_realized_vol", "overnight_gap"):
                cols.setdefault(col, pd.Series(float("nan"), index=idx))

        # Stage 6 — minute aggregates (10-min bars)
        try:
            # 10-minute aggregated features (2026-04-24). Adds finer-grained
            # intraday structure on top of the hourly set. Column names use
            # a `m_` prefix so they don't collide with identically-named
            # hourly columns (morning_drift etc.) — both can be enabled.
            if tc.minute_bars:
                from training_panel.minute_features import (  # noqa: PLC0415
                    MINUTE_FEATURE_COLS, compute_minute_features,
                )
                minute_df = tc.minute_bars.get(tc.ticker)
                if minute_df is not None and not minute_df.empty:
                    m_feats = compute_minute_features(minute_df)
                    m_feats.index = pd.DatetimeIndex(m_feats.index).normalize()
                    daily_idx = pd.DatetimeIndex(idx).normalize()
                    for col in MINUTE_FEATURE_COLS:
                        series = m_feats[col] if col in m_feats.columns else pd.Series(dtype=float)
                        aligned = series.reindex(daily_idx)
                        aligned.index = idx
                        cols[f"m_{col}"] = aligned
                else:
                    for col in MINUTE_FEATURE_COLS:
                        cols[f"m_{col}"] = pd.Series(float("nan"), index=idx)
        except Exception as exc:
            log.warning("  %s: TickerPanelFactorJob[minute] failed — %s",
                        tc.ticker, exc)
            try:
                from training_panel.minute_features import MINUTE_FEATURE_COLS as _MFC  # noqa: PLC0415
                for col in _MFC:
                    cols.setdefault(f"m_{col}", pd.Series(float("nan"), index=idx))
            except Exception:
                pass

        # Final assembly — by this point cols always has at least the
        # core factors. Build the DataFrame even if some optional stages
        # NaN-filled; downstream cross-sectional z-score handles NaN cells.
        tc.raw_factor_frame = pd.DataFrame(cols, index=idx)


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
        # Audit P-16: don't early-return on truthy `ctx.factor_frames`.
        # Previously a partial dict left over from a prior bar would
        # silently skip this task and use stale data. Now: we only
        # short-circuit when we have a complete, watchlist-sized result
        # (one entry per ticker that has a raw_factor_frame). Anything
        # smaller → recompute fresh.
        if ctx.factor_frames and len(ctx.factor_frames) >= len(ctx.raw_factor_frames or {}):
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
            # Track B: PEAD enrichment (opt-in via panel_ltr.pead.enabled)
            "days_since_earnings", "pead_decay_weight", "pead_signal",
            # Plan G: hourly-bar aggregates (opt-in via panel_ltr.hourly.enabled)
            "morning_drift", "afternoon_drift", "vwap_premium",
            "vol_ratio", "intraday_realized_vol", "overnight_gap",
            # 2026-04-24: 10-minute-bar aggregates (opt-in via panel_ltr.minute.enabled).
            # TickerPanelFactorJob writes these m_*-prefixed columns into
            # raw_factor_frame; without them in this list they'd silently
            # drop from feature_cols. See minute_features.py for definitions.
            "m_morning_drift", "m_morning_30min_drift",
            "m_afternoon_drift", "m_closing_30min_drift",
            "m_vwap_premium", "m_vol_ratio", "m_first_hour_vol_pct",
            "m_intraday_realized_vol", "m_overnight_gap", "m_reversal_ratio",
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
                      # Track B: PEAD enrichment (opt-in via panel_ltr.pead.enabled)
                      "days_since_earnings", "pead_decay_weight", "pead_signal",
                      # Plan G: hourly-bar aggregates
                      "morning_drift", "afternoon_drift", "vwap_premium",
                      "vol_ratio", "intraday_realized_vol", "overnight_gap",
                      # 2026-04-24: 10-minute-bar aggregates (opt-in via
                      # panel_ltr.minute.enabled). Every m_* z-scored col
                      # must be listed here OR it silently drops from
                      # factor_frames — see test_session_silent_bugs.py.
                      "m_morning_drift", "m_morning_30min_drift",
                      "m_afternoon_drift", "m_closing_30min_drift",
                      "m_vwap_premium", "m_vol_ratio", "m_first_hour_vol_pct",
                      "m_intraday_realized_vol", "m_overnight_gap",
                      "m_reversal_ratio"):
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

        # Track F: optional triple-barrier label mode (Lopez de Prado AFML §3).
        # Default `fwd_5d` (production, close-to-close at lookahead). When
        # `panel_ltr.label_mode == 'triple_barrier'`, replace per-ticker
        # forward returns with the realised return at the FIRST hit among
        # {upper, lower, time} barriers. Other steps (residuals + gauss)
        # stay identical — only the per-ticker base return differs.
        label_mode = str(cfg.get("label_mode", "fwd_5d")).lower()

        spy_df = ctx.ohlcv.get(benchmark)
        if spy_df is None:
            raise RuntimeError("LabelsTask: benchmark OHLCV missing")

        spy_close = spy_df["close"].astype(float)
        spy_fwd = spy_close.shift(-lookahead) / spy_close - 1.0

        fwd_returns: dict[str, pd.Series] = {}
        if label_mode == "triple_barrier":
            from kernel.triple_barrier import (  # noqa: PLC0415
                TripleBarrierConfig, compute_triple_barrier_labels,
            )
            tb_cfg_dict = cfg.get("triple_barrier", {}) or {}
            tb_cfg = TripleBarrierConfig(
                alpha           = float(tb_cfg_dict.get("alpha",            2.0)),
                beta            = float(tb_cfg_dict.get("beta",             2.0)),
                max_horizon_days= int(tb_cfg_dict.get("max_horizon_days",   lookahead)),
                vol_window      = int(tb_cfg_dict.get("vol_window",         20)),
            )
            log.info("LabelsTask: label_mode=triple_barrier  alpha=%.2f beta=%.2f "
                     "max_horizon=%d vol_window=%d",
                     tb_cfg.alpha, tb_cfg.beta, tb_cfg.max_horizon_days, tb_cfg.vol_window)
            wl_ohlcv = {t: df for t, df in ctx.ohlcv.items() if t in ctx.watchlist}
            tb_out = compute_triple_barrier_labels(wl_ohlcv, tb_cfg)
            for t, frame in tb_out.items():
                fwd_returns[t] = frame["label"]
        else:
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

        if label_mode == "triple_barrier":
            # E24 v2 fix (2026-05-02): hit-time-matched residualization.
            # ticker_fwd is variable-horizon (1..max_horizon_days), so
            # SPY/sector benchmark forward returns must use the SAME hit
            # horizon per (ticker, t) row — not the fixed `lookahead`-day
            # window. Mixing horizons in OLS produces anti-predictive
            # labels (initial v1 fix attempt eval_ic=−0.0744 with mismatch).
            #
            # New path: compute_residual_returns_hit_aligned reads
            # `hit_days` per (ticker, t) from the triple-barrier output
            # and constructs per-row spy_fwd[i,t] = spy_close[t+hit] / spy_close[t] − 1.
            # β estimation uses the same purged rolling OLS, with purge =
            # max_horizon_days so prior-window β fit doesn't see future
            # bars. Sector orthogonalization (FWL) preserved.
            from training_panel.labels import compute_residual_returns_hit_aligned  # noqa: PLC0415
            hit_days_by_ticker = {t: tb_out[t]["hit_days"] for t in tb_out}
            sec_close_by_ticker = {
                t: ctx.sector_etf_ohlcv[sec]["close"].astype(float)
                for t, sec in ctx.ticker_sectors.items()
                if sec in ctx.sector_etf_ohlcv
            }
            ctx.raw_residuals = compute_residual_returns_hit_aligned(
                fwd_returns,
                hit_days_by_ticker,
                ctx.ohlcv[benchmark]["close"].astype(float),
                sec_close_by_ticker,
                beta_window=beta_window,
                purge_days=tb_cfg.max_horizon_days,
            )
            log.info("LabelsTask: triple_barrier mode — hit-time-matched "
                     "residualization (purge=%d, beta_window=%d)",
                     tb_cfg.max_horizon_days, beta_window)
        else:
            ctx.raw_residuals = compute_residual_returns(
                fwd_returns, spy_fwd, sec_fwd_by_ticker,
                beta_window=beta_window, lookahead_days=lookahead,
            )
        ctx.labels = gaussianize_cross_section(ctx.raw_residuals)
        log.info("LabelsTask: built labels for %d tickers (mode=%s)",
                 len(ctx.labels), label_mode)


class BuildHourlyResolutionPanelTask(PanelTask):
    """Stage C-2 (2026-04-26): hourly-resolution panel for transformer.

    No-op unless ``panel_ltr.training_resolution == 'hourly'``. When active,
    builds a panel keyed by (ticker, date, hour) using the Stage C-1
    scaffold (build_hourly_resolution_panel). Sets ctx.panel directly so
    BuildPanelTask's early-out skips its daily aggregation.

    Group sizes for the transformer's date-group attention are computed
    on (date, hour) pairs. Daily-mode group_sizes (just by date) is the
    default; nothing changes there.

    Reference: doc/components/transformer-hourly-stage-c2.md.
    """

    def run(self, ctx: "PanelTrainingContext") -> bool | None:
        cfg = ctx.config.get("panel_ltr", {})
        if str(cfg.get("training_resolution", "daily")).lower() != "hourly":
            return True   # daily path active; this task is a no-op

        from kernel.intraday import HourlyBarStore  # noqa: PLC0415
        from training_panel.hourly_resolution_panel import (  # noqa: PLC0415
            build_hourly_resolution_panel,
        )

        cache_dir = _resolve_cache_dir(
            cfg.get("hourly", {}).get("cache_dir", "data/intraday"),
            ctx.config,
        )
        store = HourlyBarStore(data_dir=cache_dir)

        watchlist = list(ctx.watchlist or [])
        bars: dict = {}
        for t in watchlist:
            df = store.load(t)
            if df is not None and not df.empty:
                bars[t] = df
        if not bars:
            log.warning(
                "BuildHourlyResolutionPanelTask: no hourly bars cached "
                "for any watchlist ticker — falling back to daily panel",
            )
            return True

        benchmark = ctx.config.get("benchmark", "SPY")
        bm_bars = bars.pop(benchmark, None)
        if bm_bars is None:
            bm_bars = store.load(benchmark)

        label_horizon = int(
            cfg.get("hourly", {}).get("label_horizon_bars", 7),
        )
        panel = build_hourly_resolution_panel(
            bars,
            label_horizon_bars=label_horizon,
            benchmark_bars=bm_bars,
            apply_wash=True,
        )
        if panel is None or panel.empty:
            log.warning(
                "BuildHourlyResolutionPanelTask: build_hourly_resolution_panel "
                "produced empty panel — falling back to daily",
            )
            return True

        # Reset index so 'ticker' + 'date' + 'hour' are columns (downstream
        # tasks expect long-format with explicit columns, not multi-index).
        panel = panel.reset_index()
        # The hourly panel index from build_hourly_resolution_panel is
        # (ticker, datetime). After reset_index columns become:
        #   ticker, level_1 (datetime), then feature cols + label.
        # Rename level_1 → datetime, derive date + hour.
        if "level_1" in panel.columns:
            panel = panel.rename(columns={"level_1": "datetime"})
        if "datetime" in panel.columns:
            dt_col = panel["datetime"]
        else:
            # Fall back: assume index 0/1 was the datetime
            dt_col = pd.to_datetime(panel.iloc[:, 1])
            panel["datetime"] = dt_col
        panel["date"] = pd.to_datetime(dt_col).dt.normalize()
        panel["hour"] = pd.to_datetime(dt_col).dt.hour

        # Drop rows where label is NaN (warmup at start of each ticker).
        if "forward_excess_return" in panel.columns:
            panel = panel.rename(columns={"forward_excess_return": "label"})
        label_mask = panel["label"].notna() if "label" in panel.columns \
                     else pd.Series([True] * len(panel))
        panel = panel[label_mask].reset_index(drop=True)

        # Phase 1D (2026-04-26): broadcast macro frame onto hourly panel.
        # Same logic as the daily path in build_panel_frame, but inlined
        # here because the hourly path doesn't go through build_panel_frame.
        # Each hourly row gets the macro value for its DATE (same value
        # across all hours of a given trading day, by design — macro is
        # daily-resolution).
        macro_frame = ctx.macro_factor_frame
        if macro_frame is not None and not macro_frame.empty:
            if not isinstance(macro_frame.index, pd.DatetimeIndex):
                macro_frame = macro_frame.copy()
                macro_frame.index = pd.to_datetime(macro_frame.index)
            macro_cols = list(macro_frame.columns)
            existing = set(panel.columns)
            collisions = [c for c in macro_cols if c in existing]
            if collisions:
                log.warning(
                    "BuildHourlyResolutionPanelTask: macro frame columns "
                    "collide with existing panel columns: %s — using "
                    "suffix '_macro'", collisions,
                )
                rename_map = {c: f"{c}_macro" for c in collisions}
                macro_frame = macro_frame.rename(columns=rename_map)
                macro_cols = [rename_map.get(c, c) for c in macro_cols]
            panel = panel.merge(
                macro_frame, left_on="date", right_index=True, how="left",
            )
            panel[macro_cols] = panel.groupby(
                "ticker", group_keys=False,
            )[macro_cols].ffill()
            panel[macro_cols] = panel[macro_cols].fillna(0.0)
            log.info(
                "BuildHourlyResolutionPanelTask: broadcast %d macro features "
                "(rows unchanged: %d)", len(macro_cols), len(panel),
            )

        # Group_sizes by (date, hour) — transformer's date-group attention
        # operates on the cross-section at a fixed time slice.
        ctx.panel = panel
        # Audit fix C2-FEATURE-COLS-EMPTY (2026-04-26): downstream
        # CrossValidateTask + FinalFitTask read ctx.feature_cols to know
        # which columns are inputs (vs. label/metadata). Pre-fix, we
        # only set ctx.panel and ctx.panel_metadata → ctx.feature_cols
        # stayed empty → transformer.fit raised
        # 'PanelTransformerModel.train: feature_cols is empty'.
        # Fix: set feature_cols to all panel columns except the known
        # non-feature ones (ticker, date, hour, datetime, label,
        # _sample_weight). The HOURLY_RES_FEATURE_COLS list from
        # hourly_resolution_panel is the canonical input set, but use
        # actual panel columns to be defensive.
        # Bug #24 fix (TRANSFORMER-TIMESTAMP-LEAK, 2026-04-26 round-7):
        # feature_cols must EXCLUDE every non-numeric column, not just
        # the explicit name list. v5 hourly transformer trained with
        # `timestamp` (datetime64[ns]) as a feature because HourlyBarStore
        # named its index "timestamp" and reset_index turned it into a
        # column not covered by the original non_feature set. PyTorch
        # silently cast datetime64 → float (Unix epoch ns) → look-ahead
        # bias + garbage signal → OOS IC = -0.0008 (vs XGBoost +0.0326).
        #
        # Defensive fix: explicit name-list AND dtype filter. Any
        # non-numeric or bool column is dropped regardless of name. Same
        # belt-and-suspenders pattern as Bug #21 in FeatureDiagnosticTask.
        non_feature = {"ticker", "date", "hour", "datetime", "timestamp",
                       "label", "_sample_weight",
                       "forward_excess_return"}
        candidate_cols = [c for c in panel.columns if c not in non_feature]
        ctx.feature_cols = [
            c for c in candidate_cols
            if pd.api.types.is_numeric_dtype(panel[c].dtype)
            and not pd.api.types.is_bool_dtype(panel[c].dtype)
        ]
        dropped_non_numeric = [c for c in candidate_cols if c not in ctx.feature_cols]
        if dropped_non_numeric:
            log.warning(
                "BuildHourlyResolutionPanelTask: dropped %d non-numeric "
                "feature_cols (would corrupt training): %s",
                len(dropped_non_numeric),
                ", ".join(f"{c}({panel[c].dtype})" for c in dropped_non_numeric[:8]),
            )
        ctx.panel_metadata = {
            "n_rows":    int(len(panel)),
            "n_tickers": int(panel["ticker"].nunique()) if "ticker" in panel.columns else 0,
            "n_dates":   int(panel["date"].nunique()) if "date" in panel.columns else 0,
            "n_hours":   int(panel["hour"].nunique()) if "hour" in panel.columns else 0,
            "resolution": "hourly",
            "label_horizon_bars": label_horizon,
            "n_features": len(ctx.feature_cols),
        }
        log.info(
            "BuildHourlyResolutionPanelTask: hourly panel rows=%d "
            "tickers=%d dates=%d hours=%d (label horizon=%dh)",
            ctx.panel_metadata["n_rows"], ctx.panel_metadata["n_tickers"],
            ctx.panel_metadata["n_dates"], ctx.panel_metadata["n_hours"],
            label_horizon,
        )
        return True


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

        # Phase 1D (2026-04-26): pass ctx.macro_factor_frame into
        # build_panel_frame so the daily training panel sees the broadcast
        # macro factor block. Symmetric with the inference path
        # (kernel/panel_pipeline/feature_matrix.py). v2 (per-ticker β)
        # mode opts out — see Bug-2 below.
        # Bug-2 fix (2026-04-27): v2 mode uses per-ticker β features
        # (already merged into factor_frames by PanelFeatureJob), so the
        # broadcast macro_frame must NOT be injected at training time —
        # the inference side already passes None in v2 mode.  Passing the
        # broadcast frame during training while omitting it at inference
        # creates a training/inference feature-set asymmetry that silently
        # degrades OOS IC.
        macro_version = cfg.get("macro", {}).get("version", "v1")
        macro_frame_for_panel = (
            None if macro_version == "v2"
            else ctx.macro_factor_frame
        )
        panel, group_sizes, meta = build_panel_frame(
            ff_wl, lab_wl, sec_wl,
            factor_frames=fac_wl,
            macro_frame=macro_frame_for_panel,
            asset_embeddings=ctx.asset_embeddings or None,
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
        # Bug #21 fix (2026-04-26 round-7): the hourly transformer's
        # Stage C-3 v2 crashed here with
        #   DTypePromotionError: DateTime64 could not be promoted by Float64
        # when a non-numeric column (datetime / object) leaked into
        # ctx.feature_cols. Defensive: filter to columns whose dtype is
        # numeric BEFORE computing any spearman/std. Non-numeric features
        # are not a meaningful diagnostic anyway — just log + skip.
        panel = ctx.panel
        numeric_feature_cols: list[str] = []
        skipped_non_numeric: list[tuple[str, str]] = []
        for col in ctx.feature_cols:
            if col not in panel.columns:
                continue
            dt = panel[col].dtype
            if pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt):
                numeric_feature_cols.append(col)
            else:
                skipped_non_numeric.append((col, str(dt)))
        if skipped_non_numeric:
            log.warning(
                "FeatureDiagnosticTask: skipping %d non-numeric feature_cols "
                "(would crash spearmanr): %s",
                len(skipped_non_numeric),
                ", ".join(f"{c}({d})" for c, d in skipped_non_numeric[:8]),
            )

        dates = panel["date"].values
        label = panel["label"].values
        rows: list[tuple[str, float, float]] = []
        for col in numeric_feature_cols:
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
                # Bug #21 defense in depth (2026-04-26): coerce to float
                # explicitly so a stray non-numeric column that slipped past
                # the upfront filter still degrades to a logged warning,
                # not a hard crash. Skip the column entirely if coercion
                # raises (caller can investigate the panel).
                try:
                    p_f = np.asarray(p, dtype=float)
                    y_f = np.asarray(y, dtype=float)
                except (TypeError, ValueError) as exc:
                    log.warning(
                        "FeatureDiagnosticTask: column %s skipped — "
                        "spearmanr coercion failed: %s", col, exc,
                    )
                    break   # this col is broken; abandon and move on
                rho, _ = spearmanr(p_f, y_f)
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
        # Stage C-2 (2026-04-26): BuildHourlyResolutionPanelTask runs FIRST.
        # When panel_ltr.training_resolution=='hourly' it sets ctx.panel
        # directly; BuildPanelTask's early-out then skips daily aggregation.
        # Default ('daily') preserves bit-for-bit existing behaviour.
        return [
            NeutralizedFeatureZScoreTask(),
            FactorZScoreTask(),
            LabelsTask(),
            BuildHourlyResolutionPanelTask(),
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
            # Audit fix #14 (2026-04-26): align CV epochs with FinalFit.
            # Pre-fix, CV used `num_boost_round // 2` while FinalFit used
            # the full count → CV's IC ≠ FinalFit's IC because they
            # trained for different durations. Now: identical epoch
            # budgets so CV reflects what the final model will look like.
            cv_epochs = int(cfg.get("num_boost_round", 50))

            class _SklearnAdapter:
                def __init__(self):
                    self._m = PanelTransformerModel(params=tf_params)
                    self._feature_cols: list[str] | None = None
                def fit(self, X, y, sample_weight=None):
                    # Audit fix #58 (2026-04-26 round-3): assert that X
                    # index aligns with parent panel index. Pre-fix, if
                    # sklearn CV passed a re-indexed X, the .loc lookup
                    # silently produced WRONG dates → wrong group_sizes.
                    missing_idx = X.index.difference(panel.index)
                    if len(missing_idx) > 0:
                        raise KeyError(
                            f"_SklearnAdapter.fit: X has indices not in parent "
                            f"panel: {list(missing_idx)[:5]}"
                            f"{'…' if len(missing_idx) > 5 else ''}"
                        )
                    df = X.copy()
                    df["label"] = y
                    df["date"] = panel.loc[X.index, "date"].values
                    df = df.sort_values(["date"], kind="mergesort").reset_index(drop=True)
                    # Audit fix #60 (2026-04-26 batch-3): .to_numpy() not .values
                    gs = df.groupby("date", sort=True).size().to_numpy().astype(np.int32)
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
                    # Audit fix #58 (2026-04-26 round-3): assert that X
                    # index aligns with parent panel index. Pre-fix, if
                    # sklearn CV passed a re-indexed X, the .loc lookup
                    # silently produced WRONG dates → wrong group_sizes.
                    missing_idx = X.index.difference(panel.index)
                    if len(missing_idx) > 0:
                        raise KeyError(
                            f"_SklearnAdapter.fit: X has indices not in parent "
                            f"panel: {list(missing_idx)[:5]}"
                            f"{'…' if len(missing_idx) > 5 else ''}"
                        )
                    df = X.copy()
                    df["label"] = y
                    df["date"] = panel.loc[X.index, "date"].values
                    df["weight"] = sample_weight if sample_weight is not None else 1.0
                    df = df.sort_values(["date"], kind="mergesort").reset_index(drop=True)
                    # Audit fix #60 (2026-04-26 batch-3): .to_numpy() not .values
                    gs = df.groupby("date", sort=True).size().to_numpy().astype(np.int32)
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
                    # Audit fix #58 (2026-04-26 round-3): assert that X
                    # index aligns with parent panel index. Pre-fix, if
                    # sklearn CV passed a re-indexed X, the .loc lookup
                    # silently produced WRONG dates → wrong group_sizes.
                    missing_idx = X.index.difference(panel.index)
                    if len(missing_idx) > 0:
                        raise KeyError(
                            f"_SklearnAdapter.fit: X has indices not in parent "
                            f"panel: {list(missing_idx)[:5]}"
                            f"{'…' if len(missing_idx) > 5 else ''}"
                        )
                    df = X.copy()
                    df["label"] = y
                    df["date"] = panel.loc[X.index, "date"].values
                    df["weight"] = sample_weight if sample_weight is not None else 1.0
                    df = df.sort_values(["date"], kind="mergesort").reset_index(drop=True)
                    # Audit fix #60 (2026-04-26 batch-3): .to_numpy() not .values
                    gs = df.groupby("date", sort=True).size().to_numpy().astype(np.int32)
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

        # Audit fix X1+X2 (2026-04-26): build eval split for XGBoost +
        # LightGBM so they can use early_stopping_rounds. Pre-fix, both
        # ran full num_boost_round=400 with no early stop → potential
        # overfit on small panels. Split: last 20% of date-groups as eval.
        # (Transformer has its own auto_eval_split inside .train().)
        eval_panel = None
        eval_group_sizes = None
        # Allow null/0 in config to disable early stopping (needed for long
        # horizons where the eval set's last N bars have NaN labels — early
        # stopping fires immediately because it can't compute IC on NaN).
        _es_cfg = cfg.get("early_stopping_rounds", 20)
        early_stop_rounds = int(_es_cfg) if _es_cfg is not None and int(_es_cfg) > 0 else 0
        # Audit fix HIGH-2 (2026-04-27): purge `lookahead` date-groups
        # between train and eval. Pre-fix, the last `lookahead` training
        # dates carried labels that reach into the eval window — early
        # stop saw an inflated eval IC and stopped sooner than ideal.
        lookahead_for_purge = int(cfg.get("lookahead_days", 5))
        if backend in ("xgboost", "lightgbm") and len(ctx.group_sizes) >= 5 + lookahead_for_purge:
            n_total = len(ctx.group_sizes)
            # BUG-CV-3 fix (2026-04-28): align early-stop eval to the last
            # CPCV fold's date-group count (1/cv_n_splits), so early-stop
            # and CPCV IC measure the SAME data slice. Pre-fix used a
            # hardcoded 20% which differs from CPCV folds (e.g. cv_n_splits=6
            # → 1/6 ≈ 16.7%). Different slices → early stop can fire on a
            # "bad" 20% while CPCV IC reports on a different period.
            cv_splits_for_eval = int(cfg.get("cv_n_splits", 6))
            n_eval = max(2, n_total // max(2, cv_splits_for_eval))
            n_train_raw = n_total - n_eval
            # Drop the last lookahead_for_purge training dates so labels
            # don't reach into eval (HIGH-2 fix).
            n_train = max(0, n_train_raw - lookahead_for_purge)
            if n_train >= 5 and n_eval >= 2:
                row_split_train = int(np.array(ctx.group_sizes[:n_train]).sum())
                row_split_eval  = int(np.array(ctx.group_sizes[:n_train_raw]).sum())
                # Eval still starts at n_train_raw (last 20% of dates),
                # but train ends at n_train (lookahead dates earlier).
                eval_panel       = ctx.panel.iloc[row_split_eval:].copy()
                eval_group_sizes = np.array(ctx.group_sizes[n_train_raw:], dtype=np.int64)
                _train_panel       = ctx.panel.iloc[:row_split_train].copy()
                _train_group_sizes = np.array(ctx.group_sizes[:n_train], dtype=np.int64)
            else:
                _train_panel       = ctx.panel
                _train_group_sizes = ctx.group_sizes
                early_stop_rounds  = 0   # not enough data for early stop
        else:
            _train_panel       = ctx.panel
            _train_group_sizes = ctx.group_sizes

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
                _train_panel, _train_group_sizes,
                feature_cols=ctx.feature_cols,
                label_col="label", weight_col="weight",
                num_boost_round=num_rounds,
                eval_panel=eval_panel,
                eval_group_sizes=eval_group_sizes,
                early_stopping_rounds=(early_stop_rounds if early_stop_rounds > 0 else None) if eval_panel is not None else None,
            )
            device_used = "cpu"
        else:
            from training_panel.ltr_model import PanelLTRModel
            xgb_params = dict(cfg.get("xgb_params", {}))
            monotone = dict(cfg.get("monotone_constraints", {}))
            model = PanelLTRModel(params=xgb_params, monotone_constraints=monotone)
            fit = model.train(
                _train_panel, _train_group_sizes,
                feature_cols=ctx.feature_cols,
                label_col="label", weight_col="weight",
                num_boost_round=num_rounds,
                eval_panel=eval_panel,
                eval_group_sizes=eval_group_sizes,
                early_stopping_rounds=(early_stop_rounds if early_stop_rounds > 0 else None) if eval_panel is not None else None,
            )
            device_used = "cpu"
        elapsed = _time.monotonic() - t0
        ctx.final_model = model
        ctx._final_fit = fit  # noqa: SLF001 — read by SaveArtifactTask
        ctx._final_fit_elapsed_sec = elapsed  # noqa: SLF001
        ctx._final_fit_device      = device_used  # noqa: SLF001
        log.info("FinalFitTask: backend=%s  train_ic=%+.4f  elapsed=%.1fs  device=%s",
                 backend, fit.get("train_ic", 0.0), elapsed, device_used)

        # BUG-CV-2 hard guard (2026-04-28): refuse to save the artifact if
        # XGBoost early stopping fired before min_best_iter rounds. With
        # eta=0.02 and best_iter=4 (the production case discovered today),
        # total shrinkage is 4 × 0.02 = 0.08 — model is essentially
        # untrained. Pre-fix this saved silently and the model rode into
        # production with random-walk-level signal.
        # Skip for transformer (best_iter semantics differ).
        # 2026-04-28 evening update: threshold lowered 20 → 5 after the
        # round-9-saturation diagnostic confirmed XGBoost rank:pairwise on
        # this panel naturally peaks at best_iter ∈ [9, 25] with healthy
        # eval IC (+0.04 to +0.07). The original 20 was pulled from the
        # "eta×best_iter ≥ 0.4 = healthy" assumption, falsified empirically.
        # 5 still protects against the pre-fix pathology (best_iter=4) and
        # any regression that drives best_iter to 0-3 (catastrophic eval
        # set issue), without blocking healthy fast-converging models.
        if backend in ("xgboost", "lightgbm"):
            min_best_iter = int(cfg.get("min_best_iter", 5))
            best_iter = fit.get("best_iter")
            if best_iter is not None and int(best_iter) < min_best_iter:
                # 2026-05-02 refinement (Task #24): the iter-count check
                # alone is a FALSE POSITIVE on strong-univariate-IC features.
                # Adding e.g. days_since_earnings (univariate IC=+0.02) makes
                # XGBoost converge on a good split by round 4-9; further
                # rounds bring zero eval-set improvement so early stopping
                # fires. The model is NOT pathological — eval_ic is at a
                # healthy plateau, just reached fast.
                #
                # Escape clause: if eval_ic at best_iter is above
                # `min_best_iter_eval_ic_floor` (default 0.02 — well above
                # CPCV noise band ±0.005), accept the model despite low
                # iter count. Pathological case (eval_ic ≈ 0) still raises.
                eval_ic_floor = float(cfg.get("min_best_iter_eval_ic_floor", 0.02))
                eval_ic = fit.get("eval_ic")
                import math as _math2  # noqa: PLC0415
                if (eval_ic is not None and _math2.isfinite(float(eval_ic))
                        and float(eval_ic) >= eval_ic_floor):
                    log.info(
                        "FinalFitTask: best_iter=%d < min_best_iter=%d but "
                        "eval_ic=%+.4f ≥ floor=%+.4f — strong-signal feature "
                        "converged early on healthy plateau, accepting.",
                        int(best_iter), min_best_iter, float(eval_ic), eval_ic_floor,
                    )
                else:
                    raise RuntimeError(
                        f"FinalFit early_stopping fired at round {best_iter} "
                        f"(< min_best_iter={min_best_iter}) AND eval_ic={eval_ic} "
                        f"< floor {eval_ic_floor} — model genuinely undertrained "
                        f"(eta×best_iter={float(cfg.get('xgb_params', {}).get('eta', cfg.get('xgb_params', {}).get('learning_rate', 0.02))) * int(best_iter):.4f} total shrinkage). "
                        f"Artifact NOT saved. Check eval-set alignment with CPCV folds "
                        f"(see BUG-CV-3 in CLAUDE.md). To bypass for diagnostic runs only, "
                        f"set panel_ltr.min_best_iter to a smaller value, OR raise "
                        f"`panel_ltr.min_best_iter_eval_ic_floor` to relax the eval_ic check."
                    )

            # External audit fix #7 (2026-04-29): best_iter alone is not enough.
            # A model can pass best_iter=5 with eval_ic ≈ 0 (uninformative). Add
            # eval_ic_floor as the second gate — both must pass to save.
            # Disabled by default (None) so legacy retrains aren't blocked; set
            # `panel_ltr.min_eval_ic` in config to enable. Recommended: 0.005
            # (above noise floor for 15-fold CPCV on 60k-row panel).
            min_eval_ic = cfg.get("min_eval_ic")
            if min_eval_ic is not None:
                import math as _math  # noqa: PLC0415
                eval_ic = fit.get("eval_ic")
                if eval_ic is None or not _math.isfinite(float(eval_ic)) \
                        or float(eval_ic) < float(min_eval_ic):
                    raise RuntimeError(
                        f"FinalFit eval_ic={eval_ic} below min_eval_ic={min_eval_ic}. "
                        f"Model converged on best_iter but is uninformative on the "
                        f"holdout — likely panel/label/feature regression upstream. "
                        f"Artifact NOT saved. To bypass for diagnostic runs, lower "
                        f"or unset `panel_ltr.min_eval_ic` in config."
                    )


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
            # External audit fix #2 (2026-04-29): run_id ties panel-ltr, ngboost,
            # and calibrator artifacts from the same train run together. Preflight
            # warns when run_ids don't match (stale artifact from different run).
            "train_run_id":    ctx.config.get("_train_run_id"),
        }
        # 2026-04-28: stamp config fingerprint so RunnerAdapter can detect
        # config/model drift at inference startup. See
        # kernel/config_consistency.py for the spec. Three incidents in
        # 24h (NGBoost macro drift / ndcg config flip / watchlist 227
        # mismatch) all of the form "config changed but model wasn't
        # retrained / vice versa". This guard catches the next one.
        try:
            from kernel.config_consistency import (  # noqa: PLC0415
                fingerprint_config, _model_relevant_fields,
            )
            meta["config_fingerprint"]        = fingerprint_config(ctx.config)
            meta["config_fingerprint_fields"] = _model_relevant_fields(ctx.config)
        except Exception as exc:
            log.warning("Config-consistency stamp failed: %s — artifact lacks "
                        "fingerprint, will trip backwards-compat path on load", exc)
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

        # Audit fix TRANSFORMER-PANEL-LTR-SHIM (2026-04-26): when backend=transformer,
        # `final_model.save()` writes panel-transformer.pt + panel-transformer.json
        # but does NOT update panel-ltr.json. Downstream fit_panel_calibrator.py
        # reads panel-ltr.json (hardcoded) and on a transformer-backend run
        # ends up reading the STALE LightGBM/XGBoost JSON from a prior backend.
        # Today's Sunday sweep crashed with SIGSEGV in this path (panel-ltr.json
        # was an LGBM tree dump, calibrator scored against transformer features
        # against LGBM trees → undefined behaviour).
        # Fix: write a thin pointer JSON to panel-ltr.json so PanelScorer.load()
        # auto-dispatches to TransformerPanelScorer via `kind` field.
        if backend == "transformer" and ctx.strategy_dir is not None:
            try:
                shim_path = ctx.strategy_dir / "artifacts" / "panel-ltr.json"

                # Audit fix #141 TRANSFORMER-CLOBBER-AUTOBAK (2026-04-26
                # round-7): before overwriting panel-ltr.json with the
                # transformer shim, snapshot the existing file to
                # panel-ltr.{prev_kind}.bak.json so the prior backend's
                # artifact is recoverable via cp without manual sweep
                # script. Pre-fix, only sunday_panel_sweep.py created
                # .bak files; a direct `train_104.py --strategy-config
                # strategy_config.hourly_transformer.json` run silently
                # destroyed the production XGBoost panel-ltr.json with
                # only the prior sweep's .xgboost.bak.json as recovery.
                # If the user had manually edited panel-ltr.json since
                # the last sweep, those edits would be lost.
                if shim_path.exists():
                    try:
                        prev = json.loads(shim_path.read_text())
                        prev_kind = str(prev.get("kind", "unknown"))
                        # Map kernel kinds to short backend labels.
                        bak_label = {
                            "panel_ltr_xgboost":     "xgboost",
                            "panel_ltr_lightgbm":    "lightgbm",
                            "panel_transformer":     "transformer",
                        }.get(prev_kind, prev_kind.replace("panel_", "").replace("_", "-"))
                        bak_path = shim_path.parent / f"panel-ltr.{bak_label}.bak.json"
                        # Only overwrite the .bak if it doesn't already match
                        # (avoid clobbering an older bak with a newer-but-shim variant).
                        if not bak_path.exists() or bak_path.read_text() != shim_path.read_text():
                            bak_path.write_text(shim_path.read_text())
                            log.info(
                                "SaveArtifactTask: pre-shim backup → %s "
                                "(prev_kind=%s)",
                                bak_path.name, prev_kind,
                            )
                    except Exception as bak_exc:
                        log.warning(
                            "SaveArtifactTask: pre-shim backup failed — %s "
                            "(continuing; .xgboost.bak.json from sweep "
                            "may still be available)",
                            bak_exc,
                        )

                pt_relative = out_path.name   # "panel-transformer.pt"
                shim = {
                    "kind": "panel_transformer",
                    "artifact_path": pt_relative,
                    "feature_cols": ctx.feature_cols,
                    "metadata": meta,
                    "shim_for_calibrator": True,
                }
                shim_path.write_text(json.dumps(shim, default=str))
                log.info("SaveArtifactTask: wrote panel-ltr.json shim → %s "
                         "(kind=panel_transformer, points to %s)",
                         shim_path, pt_relative)
            except Exception as exc:
                log.warning("SaveArtifactTask: panel-ltr.json shim failed — %s",
                            exc)

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
        # Audit fix N-2 / N-14 (2026-04-25): enable time-ordered val
        # split + early stopping. Default 20% of distinct dates → val,
        # halt when validation NLL plateaus for `early_stopping_rounds`
        # iterations. Config keys under `panel_ltr.ngboost`:
        #   val_fraction          (float, default 0.2)
        #   early_stopping_rounds (int,   default 25; 0/null disables)
        es_rounds = cfg.get("early_stopping_rounds", 25)
        if es_rounds in (0, False):
            es_rounds = None
        val_fraction = float(cfg.get("val_fraction", 0.2))

        # ── N-17 (2026-04-25 autonomous run): CPCV for NGBoost ────────
        # Pre-fix: NGBoost had NO out-of-sample IC. We persisted only
        # train_mu_ic + train σ̄ in the artifact, so the operator had no
        # signal whether μ̂ generalised. This block runs CPCV with the
        # SAME splitter family used by panel-LTR (CombinatorialPurgedCV
        # by default; configurable). The resulting `oos_*` metrics are
        # surfaced via NGBoostSaveTask so they appear in the artifact.
        # Off by default to keep retrain time bounded; enable via
        # config `panel_ltr.ngboost.cv.enabled = true`.
        cv_cfg = cfg.get("cv", {}) or {}
        cv_enabled = bool(cv_cfg.get("enabled", False))
        cv_result: dict | None = None
        if cv_enabled:
            from training_panel.purged_cv import (
                CombinatorialPurgedCV, PurgedKFold,
                cross_validated_ic, cross_validated_ic_cpcv,
            )
            cv_method   = str(cv_cfg.get("method", "cpcv")).lower()
            cv_splits   = int(cv_cfg.get("n_splits", 6))
            cv_test_grp = int(cv_cfg.get("n_test_groups", 2))
            cv_embargo  = int(cv_cfg.get("embargo_days", 5))
            cv_lookahd  = int(cv_cfg.get("lookahead_days", 5))
            # Use a smaller n_estimators in CV (halve the production
            # rounds) to keep CV time-bounded — same rationale as the
            # transformer/lgbm CV adapters.
            cv_params = dict(params)
            cv_params["n_estimators"] = max(50, int(params.get("n_estimators", 400)) // 2)

            class _NGBSklearnAdapter:
                def __init__(self_a):
                    self_a._head = NGBoostHead(params=cv_params)
                def fit(self_a, X, y, sample_weight=None):
                    df = X.copy()
                    df["residual_return_raw"] = y
                    if sample_weight is not None:
                        df["weight"] = sample_weight
                    self_a._head.train(
                        df,
                        feature_cols=list(X.columns),
                        label_col="residual_return_raw",
                        sample_weight_col="weight" if sample_weight is not None else None,
                        early_stopping_rounds=None,  # CV folds are short — skip ES
                    )
                def predict(self_a, X):
                    pred = self_a._head.predict_distribution(pd.DataFrame(
                        X.values, index=X.index, columns=X.columns,
                    ))
                    return pred["mu"].reindex(X.index).values

            t_cv = _time.monotonic()
            try:
                if cv_method == "cpcv":
                    cv = CombinatorialPurgedCV(
                        n_splits=cv_splits, n_test_groups=cv_test_grp,
                        embargo_days=cv_embargo, lookahead_days=cv_lookahd,
                    )
                    cv_result = cross_validated_ic_cpcv(
                        _NGBSklearnAdapter, sub, ctx.feature_cols,
                        "residual_return_raw", cv,
                        weight_col="weight" if "weight" in sub.columns else None,
                    )
                else:
                    cv = PurgedKFold(
                        n_splits=cv_splits, embargo_days=cv_embargo,
                        lookahead_days=cv_lookahd,
                    )
                    cv_result = cross_validated_ic(
                        _NGBSklearnAdapter, sub, ctx.feature_cols,
                        "residual_return_raw", cv,
                        weight_col="weight" if "weight" in sub.columns else None,
                    )
                log.info(
                    "NGBoostFitTask[CV %s]: mean=%+.4f std=%.4f n_splits=%d  elapsed=%.1fs",
                    cv_method, cv_result.get("mean_ic", float("nan")),
                    cv_result.get("std_ic", float("nan")),
                    len(cv_result.get("per_fold_ic", [])),
                    _time.monotonic() - t_cv,
                )
            except Exception as exc:
                log.warning("NGBoostFitTask[CV]: skipped due to %s: %s",
                            type(exc).__name__, exc)
                cv_result = None

        head = NGBoostHead(params=params)
        t0 = _time.monotonic()
        # Audit fix HIGH-3 (2026-04-27): pass lookahead_days so the
        # train/val date split purges leakage between segments.
        lookahead_for_purge = int(ctx.config.get("panel_ltr", {}).get("lookahead_days", 5))
        try:
            fit = head.train(
                sub,
                feature_cols=ctx.feature_cols,
                label_col="residual_return_raw",
                sample_weight_col="weight" if "weight" in sub.columns else None,
                val_fraction=val_fraction,
                early_stopping_rounds=int(es_rounds) if es_rounds else None,
                lookahead_days=lookahead_for_purge,
            )
        except Exception as exc:
            # Audit fix NGB-OVERFLOW-TRAIN (2026-04-28): NGBoost fit can fail
            # hard (numerical blow-up, memory, etc.) even after the input-clip
            # guard in NGBoostHead.train. Keep the pipeline alive:
            #   • If a previous ngboost-head.json artifact exists on disk,
            #     log a warning and return without touching ctx.ngboost_head.
            #     NGBoostSaveTask checks `if ctx.ngboost_head is None: return`,
            #     so the old artifact is preserved and inference continues.
            #   • If no previous artifact exists (first-ever run), log an
            #     error and return — NGBoost simply won't run today.
            # Either way the XGBoost panel-LTR artifact is unaffected.
            cfg_inner = ctx.config.get("panel_ltr", {}).get("ngboost", {})
            _art_name = cfg_inner.get("artifact_path", "ngboost-head.json")
            _art_path = Path(_art_name)
            if ctx.strategy_dir and not _art_path.is_absolute():
                _art_path = ctx.strategy_dir / "artifacts" / _art_path.name
            if _art_path.exists():
                log.warning(
                    "NGBoostFitTask: fit raised %s: %s — keeping previous "
                    "artifact at %s; XGBoost path unaffected",
                    type(exc).__name__, exc, _art_path,
                )
            else:
                log.error(
                    "NGBoostFitTask: fit raised %s: %s — no previous artifact "
                    "at %s; NGBoost will be skipped this run",
                    type(exc).__name__, exc, _art_path,
                )
            return
        if cv_result is not None:
            # Make CV metrics available to NGBoostSaveTask.
            fit["oos_mean_ic"]    = cv_result.get("mean_ic")
            fit["oos_std_ic"]     = cv_result.get("std_ic")
            fit["oos_per_fold_ic"] = cv_result.get("per_fold_ic")
            fit["oos_ic_quantiles"] = cv_result.get("quantiles", {})
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
        # 2026-04-28 evening fix (revised): side configs that set the
        # inference-side NGBoost path (`ranking.panel_scoring.ngboost.
        # artifact_path`) but inherit the training-side default
        # `artifacts/ngboost-head.json` from the prod template were
        # mis-saved to production. Initial fix preferred training-side
        # which made this worse (training-side was silently inherited).
        # Revised: **inference-side wins** when both are set differently
        # — the live runner reads from inference-side, so writing where
        # the runner reads is the only correct semantics. Training-side
        # is the fallback only when inference-side is unset.
        cfg = ctx.config.get("panel_ltr", {}).get("ngboost", {})
        cfg_infer = (ctx.config.get("ranking", {})
                                .get("panel_scoring", {})
                                .get("ngboost", {}))
        out_name_train = cfg.get("artifact_path")
        out_name_infer = cfg_infer.get("artifact_path")
        if out_name_train and out_name_infer and out_name_train != out_name_infer:
            log.warning(
                "NGBoostSaveTask: training-side path %s != inference-side path %s. "
                "Using inference-side (where the live runner reads); "
                "please reconcile in config.",
                out_name_train, out_name_infer,
            )
        out_name = out_name_infer or out_name_train or "ngboost-head.json"
        out_path = Path(out_name)
        if ctx.strategy_dir and not out_path.is_absolute():
            # Preserve the relative path AS WRITTEN.
            if out_path.parent == Path("."):
                out_path = ctx.strategy_dir / "artifacts" / out_path
            else:
                out_path = ctx.strategy_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "train_run_id":    ctx.config.get("_train_run_id"),  # audit fix #2
            "training_notes": cfg.get("training_notes", "Stage-2 NGBoost head"),
            "train_mu_mean":   ctx.ngboost_fit.get("train_mu_mean"),
            "train_sigma_mean": ctx.ngboost_fit.get("train_sigma_mean"),
            "train_mu_ic":     ctx.ngboost_fit.get("train_mu_ic"),
            "val_mu_ic":       ctx.ngboost_fit.get("val_mu_ic"),
            "best_iter":       ctx.ngboost_fit.get("best_iter"),
            "n_rows":          ctx.ngboost_fit.get("n_rows"),
            "n_rows_train":    ctx.ngboost_fit.get("n_rows_train"),
            "n_rows_val":      ctx.ngboost_fit.get("n_rows_val"),
            "n_rows_dropped":  ctx.ngboost_fit.get("n_rows_dropped"),
            # N-17 audit (2026-04-25): CPCV results when ngboost.cv.enabled.
            "oos_mean_ic":     ctx.ngboost_fit.get("oos_mean_ic"),
            "oos_std_ic":      ctx.ngboost_fit.get("oos_std_ic"),
            "oos_per_fold_ic": ctx.ngboost_fit.get("oos_per_fold_ic"),
            "oos_ic_quantiles": ctx.ngboost_fit.get("oos_ic_quantiles"),
        }
        # Strip None values to keep the artifact clean.
        meta = {k: v for k, v in meta.items() if v is not None}

        # External audit fix #6 (2026-04-29): NGBoost saver previously did
        # a direct overwrite. If the new head was bad (val_mu_ic ≈ 0,
        # corrupted feature_cols, etc.) the prior production head was
        # already gone — no recovery path. Mirror panel-LTR's snapshot +
        # acceptance pattern: write to .staging.json, hard-gate on
        # val_mu_ic floor, atomic-rename to final path on pass.
        ngb_min_val_ic = cfg.get("min_val_mu_ic")
        val_mu_ic = ctx.ngboost_fit.get("val_mu_ic")

        # Snapshot prior production artifact for rollback (if it exists).
        prior_snapshot = None
        if out_path.exists():
            import shutil  # noqa: PLC0415
            prior_snapshot = out_path.with_suffix(".pre-train.json")
            shutil.copy2(str(out_path), str(prior_snapshot))

        # Stage the new artifact at .staging.json — never touch out_path
        # until acceptance passes.
        staging_path = out_path.with_suffix(".staging.json")
        ctx.ngboost_head.save(staging_path, metadata=meta)

        if ngb_min_val_ic is not None:
            import math as _math  # noqa: PLC0415
            if val_mu_ic is None or not _math.isfinite(float(val_mu_ic)) \
                    or float(val_mu_ic) < float(ngb_min_val_ic):
                log.error(
                    "NGBoostSaveTask: val_mu_ic=%s below min_val_mu_ic=%s — "
                    "REJECTING new NGBoost head. Staging artifact left at %s "
                    "for diagnostic; prior head preserved at %s.",
                    val_mu_ic, ngb_min_val_ic, staging_path, out_path,
                )
                # Don't promote. If we had a prior snapshot, the prior
                # production artifact is still in place at out_path.
                if prior_snapshot and prior_snapshot.exists():
                    try:
                        prior_snapshot.unlink()
                    except OSError:
                        pass
                ctx.ngboost_artifact_path = None
                return

        # Promote staging → final atomically. On POSIX rename is atomic
        # within the same filesystem; if a reader has the old artifact
        # mmap'd, it keeps reading the old version until next open.
        import os as _os  # noqa: PLC0415
        _os.replace(str(staging_path), str(out_path))
        if prior_snapshot and prior_snapshot.exists():
            try:
                prior_snapshot.unlink()
            except OSError:
                pass
        ctx.ngboost_artifact_path = out_path
        log.info("NGBoostSaveTask: artifact → %s (val_mu_ic=%s)",
                 out_path, val_mu_ic)

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


# ── Phase 6 — RefreshPanelCalibratorJob ────────────────────────────────────

class RefreshPanelCalibratorTask(PanelTask):
    """Re-fit the global panel calibrator against the freshly-saved panel-LTR.

    Audit fix CAL-7 (Round 2 deep audit, 2026-04-25): pre-fix, the
    panel-LTR retrained on every daily_104 / Sunday-retrain cycle, but
    the panel-rank-calibration.json was a SEPARATE artifact rebuilt
    only by `scripts/fit_panel_calibrator.py` — a manual script not in
    the pipeline. Operators forgot to re-run it after panel retrains
    → calibrator went stale (different ticker count / different model)
    → ApplyGlobalCalibrationTask received scores from a NEW model
    distribution mapped through an OLD calibrator → systematic
    miscalibration of rank_score + expected_return.

    Now: shells out to `scripts/fit_panel_calibrator.py` after
    PanelModelJob finishes, ensuring the calibrator artifact is always
    aligned with the panel-LTR artifact it was paired with. Skipped
    when `panel_ltr.global_calibration.auto_refresh = false`.

    Failure is logged but non-fatal — the existing (stale) calibrator
    survives, so live trading doesn't crash on a calibrator-refresh
    error mid-pipeline.
    """

    def run(self, ctx: PanelTrainingContext) -> None:
        # Audit fix CAL-7-PATH (Round 4 deep audit, 2026-04-25): pre-fix,
        # this read `panel_ltr.global_calibration` but the actual config
        # lives at `ranking.panel_scoring.global_calibration` (matches
        # the runtime LoadGlobalCalibrationTask path in job_panel_scoring.py).
        # The wrong path returned `{}` → `gc_cfg.get("enabled", False)` →
        # False → silent skip → calibrator never auto-refreshed → calibrator
        # artifact stayed stale across panel retrains, mapping new model
        # scores through old-model calibration. THIS WAS WHY no buys fired
        # despite healthy panel models — calibrated rank_scores were
        # systematically miscalibrated.
        gc_cfg = (ctx.config.get("ranking", {})
                            .get("panel_scoring", {})
                            .get("global_calibration", {}))
        if not bool(gc_cfg.get("auto_refresh", True)):
            log.info("RefreshPanelCalibratorTask: auto_refresh=false — skipping")
            return
        if not gc_cfg.get("enabled", False):
            # Nothing to refresh if global calibration is off entirely.
            return
        if ctx.strategy_dir is None:
            log.warning("RefreshPanelCalibratorTask: no strategy_dir — skipping")
            return

        import subprocess as _sub  # noqa: PLC0415
        import sys as _sys         # noqa: PLC0415
        import time as _time       # noqa: PLC0415
        repo_root = ctx.strategy_dir.parent.parent
        script    = repo_root / "scripts" / "fit_panel_calibrator.py"
        if not script.exists():
            log.warning("RefreshPanelCalibratorTask: %s not found — skipping",
                        script)
            return

        strategy_name = ctx.config.get("_strategy_name", "renquant_104")
        cmd = [_sys.executable, str(script), "--strategy", strategy_name]
        # 2026-04-28 evening fix: forward the active strategy config name
        # so the calibrator uses the matching side config (and writes to
        # side calibration path) instead of always reading production.
        # Without this, side-config retrains silently fit a calibrator
        # against the (in-flight) production panel-LTR and overwrote
        # production panel-rank-calibration.json.
        scn = ctx.config.get("_strategy_config_name")
        if scn and scn != "strategy_config.json":
            cmd.extend(["--strategy-config-name", scn])
        # Forward threshold_mode from config so the calibrator subprocess
        # picks it up without requiring a separate CLI override.
        calib_threshold_mode = ctx.config.get("panel_ltr", {}).get(
            "calibrator_threshold_mode"
        )
        if calib_threshold_mode:
            cmd.extend(["--threshold-mode", calib_threshold_mode])
        log.info("RefreshPanelCalibratorTask: %s", " ".join(cmd))
        t0 = _time.monotonic()
        try:
            # Audit fix CAL-7-TIMEOUT (2026-04-25): pre-fix timeout=600s
            # was hit on the production panel run today (2494 dates × 99
            # tickers = 247K-row CPCV-15 fit takes 10-15 min). Result:
            # calibrator artifact stayed paired with the OLD panel-LTR
            # while panel itself was retrained. Bump to 1800s (30 min)
            # to give CPCV the headroom it needs.
            r = _sub.run(cmd, cwd=str(repo_root), capture_output=True,
                         text=True, timeout=1800.0)
            elapsed = _time.monotonic() - t0
            if r.returncode == 0:
                log.info("RefreshPanelCalibratorTask: refreshed  elapsed=%.1fs",
                         elapsed)
                # Surface the calibrator's pool_ic in the log for
                # operator visibility.
                for line in (r.stdout or "").splitlines():
                    if "Summary:" in line or "Saved" in line:
                        log.info("  %s", line.strip())
            else:
                log.warning(
                    "RefreshPanelCalibratorTask: failed rc=%d  elapsed=%.1fs\n"
                    "  STDERR (tail):\n%s",
                    r.returncode, elapsed,
                    "\n".join((r.stderr or "").splitlines()[-15:]),
                )
        except _sub.TimeoutExpired:
            log.warning("RefreshPanelCalibratorTask: timed out after 1800s")
        except Exception as exc:
            log.warning("RefreshPanelCalibratorTask: %s", exc)


class RefreshPanelCalibratorJob(PanelJob):
    """Phase 6 — refresh panel-rank-calibration.json after panel retrain.

    Skipped when `panel_ltr.global_calibration.{enabled,auto_refresh}`
    is false. Always runs AFTER PanelNGBoostJob so both the LTR and
    NGBoost artifacts are stable before the calibrator queries them.
    """

    def should_skip(self, ctx: PanelTrainingContext) -> bool:
        # Audit fix CAL-7-PATH (Bug J, 2026-04-25): same wrong path as the
        # task body above — corrected to ranking.panel_scoring.global_calibration.
        gc = (ctx.config.get("ranking", {})
                        .get("panel_scoring", {})
                        .get("global_calibration", {}))
        return not bool(gc.get("enabled", False)
                         and gc.get("auto_refresh", True))

    @property
    def tasks(self) -> list[PanelTask]:
        return [RefreshPanelCalibratorTask()]


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
            RefreshPanelCalibratorJob(),  # CAL-7: refresh calibrator (optional)
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
