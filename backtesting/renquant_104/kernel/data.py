"""OHLCV data fetching with local Parquet cache.

Self-contained — no common/ imports.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# Module-level logger — `fetch_intraday_bars` previously referenced an
# undefined `log` and would NameError on the timeout path it was supposed
# to handle gracefully.
log = logging.getLogger("kernel.data")


class LocalStore:
    """Read/write OHLCV data as Parquet files.

    Layout::

        {data_dir}/{SYMBOL}/{timeframe}.parquet
    """

    def __init__(self, data_dir: Path | str = "data/ohlcv"):
        self.data_dir = Path(data_dir)

    def _path(self, symbol: str, timeframe: str = "1d") -> Path:
        return self.data_dir / symbol.upper() / f"{timeframe}.parquet"

    def load(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame | None:
        """Load from local Parquet. Returns None if the file is missing."""
        path = self._path(symbol, timeframe)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if start:
            df = df.loc[start:]
        if end:
            df = df.loc[:end]
        return df if not df.empty else None

    def save(self, df: pd.DataFrame, symbol: str, timeframe: str = "1d") -> Path:
        """Save (or append) OHLCV data. Deduplicates by index.

        Audit fix DAT-ATOM (Round 2 deep audit, 2026-04-25): atomic
        write via .tmp + rename. Same as DC-2-CACHE / FU-1. Critical
        because the OHLCV cache is the FOUNDATION of every other
        feature — corruption here cascades into everything.
        """
        path = self._path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if path.exists():
            try:
                existing = pd.read_parquet(path)
            except Exception:
                # Existing cache corrupt — overwrite cleanly with new data.
                existing = None
            if existing is not None:
                if not isinstance(existing.index, pd.DatetimeIndex):
                    existing.index = pd.to_datetime(existing.index)
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep="last")]
                df = df.sort_index()

        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp)
        tmp.replace(path)
        return path

    def has_range(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: str | None = None,
        end: str | None = None,
        tolerance_days: int = 5,
    ) -> bool:
        """Check whether the local cache covers [start, end]."""
        path = self._path(symbol, timeframe)
        if not path.exists():
            return False
        df = pd.read_parquet(path)
        if df.empty:
            return False
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if start and df.index.min() > pd.Timestamp(start):
            return False
        if end and df.index.max() < pd.Timestamp(end) - pd.Timedelta(days=tolerance_days):
            return False
        return True


_default_store = LocalStore()


def fetch_ohlcv(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    provider: str = "yfinance",
    cache: bool = True,
    timeout_sec: float = 30.0,
) -> pd.DataFrame:
    """Fetch OHLCV data, using a local Parquet cache when possible.

    Has a HARD TIMEOUT on the remote fetch (default 30s). If the upstream
    (yfinance / OpenBB) hangs (classic CLOSE_WAIT socket leak), the call
    returns None instead of blocking forever. Notebook was observed
    hanging 4 hours on a yfinance call 2026-04-24 — this prevents that.
    """
    store = _default_store

    if cache and store.has_range(symbol, start=start, end=end):
        cached = store.load(symbol, start=start, end=end)
        if cached is not None:
            return cached

    if provider == "yfinance":
        def _fetch_yf():
            from openbb import obb  # lazy import — OpenBB init is slow
            kwargs: dict = {"symbol": symbol, "provider": "yfinance"}
            if start:
                kwargs["start_date"] = start
            if end:
                kwargs["end_date"] = end
            return obb.equity.price.historical(**kwargs).to_df()

        from kernel.net_safety import call_with_timeout  # noqa: PLC0415
        df = call_with_timeout(
            _fetch_yf,
            timeout_sec=timeout_sec,
            label=f"fetch_ohlcv({symbol})",
        )
        if df is None:
            raise RuntimeError(
                f"fetch_ohlcv({symbol!r}) timed out after {timeout_sec}s. "
                f"Upstream yfinance/OpenBB hung (likely CLOSE_WAIT). "
                f"Rerun after checking network; check data/ohlcv/{symbol}/1d.parquet "
                f"for cached history."
            )
    else:
        raise ValueError(f"Unknown provider {provider!r}. Supported: ['yfinance']")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df[~df.index.duplicated(keep="last")].sort_index()

    if cache:
        store.save(df, symbol)

    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]

    return df


def fetch_ohlcv_incremental(
    symbol: str,
    *,
    end: "str | None" = None,
    timeframe: str = "1d",
    timeout_sec: float = 30.0,
    store: "LocalStore | None" = None,
) -> pd.DataFrame:
    """Unified OHLCV reader: cache + incremental fetch + timeout.

    User spec 2026-04-24: "读数据应该有个 wrapper，来处理各种情况，
    保证只读增量数据，cache，以及处理各种卡住 timeout 的情况".

    Semantics:

    1. **Cache first.** Read existing parquet at
       ``data/ohlcv/{symbol}/{timeframe}.parquet``. If the cache's
       latest bar is within 2 trading days of `end` (or today if `end`
       not given), return the cache as-is — no network call.

    2. **Incremental fetch.** When the cache is stale, fetch ONLY the
       delta [cache_last_date + 1, end]. Merges into the existing cache,
       saves, returns the merged series.

    3. **Cold start.** No cache → fetch the last 10 years (reasonable
       default training window), save.

    4. **Timeout-protected.** Every network call goes through
       ``kernel.net_safety.call_with_timeout`` — the 2026-04-24 notebook
       4-hour hang must never happen again. On timeout:
         * cache exists → return stale cache with a warning
         * no cache     → raise RuntimeError

    5. **No duplicate fetch in one process.** ``_inflight_locks`` is a
       module-level dict so concurrent callers for the same symbol
       don't race (sim threads + notebook cells etc).

    Drop-in replacement for ``fetch_ohlcv`` in most call paths. Unlike
    the original, it ALWAYS returns a full series (up to the latest
    cache date), never a date-bounded slice — slicing is the caller's
    job.
    """
    import logging as _logging
    import threading
    _log_local = _logging.getLogger("kernel.data.incremental")

    store = store or _default_store

    # ── Single-process dedup: don't fetch the same symbol twice at once
    with _inflight_lock:
        sym_lock = _inflight_locks.setdefault(symbol, threading.Lock())
    with sym_lock:
        return _do_incremental_fetch(symbol, end, timeframe, timeout_sec, store, _log_local)


def _do_incremental_fetch(
    symbol: str,
    end: "str | None",
    timeframe: str,
    timeout_sec: float,
    store: "LocalStore",
    log: "logging.Logger",
) -> pd.DataFrame:
    import pandas as pd  # noqa: PLC0415

    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()

    # Load existing cache
    cache_path = store._path(symbol, timeframe)  # noqa: SLF001
    cache_exists = cache_path.exists()
    cached_df = None
    cache_last_date: "pd.Timestamp | None" = None

    if cache_exists:
        try:
            cached_df = pd.read_parquet(cache_path)
            if not isinstance(cached_df.index, pd.DatetimeIndex):
                cached_df.index = pd.to_datetime(cached_df.index)
            if not cached_df.empty:
                cache_last_date = cached_df.index.max()
        except Exception as exc:
            log.warning("Cache read failed for %s: %s — will refetch", symbol, exc)
            cached_df = None

    # Freshness check: within 2 business days of `end`
    fresh_cutoff = end_ts - pd.Timedelta(days=2)
    if cached_df is not None and cache_last_date is not None and cache_last_date >= fresh_cutoff:
        log.debug("Cache hit for %s (last=%s, end=%s)",
                  symbol, cache_last_date.date(), end_ts.date())
        return cached_df.loc[:end_ts] if end else cached_df

    # Incremental window: fetch only delta
    if cache_last_date is not None:
        fetch_start = (cache_last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        log.info("Incremental fetch for %s: [%s .. %s]", symbol, fetch_start, end_ts.date())
    else:
        # Cold start: last 10 years
        fetch_start = (end_ts - pd.Timedelta(days=365 * 10)).strftime("%Y-%m-%d")
        log.info("Cold fetch for %s: [%s .. %s] (10yr history)", symbol, fetch_start, end_ts.date())

    # Network-protected fetch
    from kernel.net_safety import call_with_timeout  # noqa: PLC0415

    def _fetch():
        from openbb import obb  # noqa: PLC0415
        kwargs = {
            "symbol":     symbol,
            "provider":   "yfinance",
            "start_date": fetch_start,
            "end_date":   end_ts.strftime("%Y-%m-%d"),
        }
        return obb.equity.price.historical(**kwargs).to_df()

    new_df = call_with_timeout(
        _fetch,
        timeout_sec=timeout_sec,
        label=f"fetch_ohlcv_incremental({symbol})",
    )

    if new_df is None:
        # Timeout path — degrade gracefully
        if cached_df is not None:
            log.warning("fetch_ohlcv_incremental(%s) timed out after %.0fs — "
                        "returning stale cache (last=%s)",
                        symbol, timeout_sec, cache_last_date.date() if cache_last_date else "?")
            return cached_df.loc[:end_ts] if end else cached_df
        raise RuntimeError(
            f"fetch_ohlcv_incremental({symbol!r}) timed out after {timeout_sec}s "
            f"with no cache available. Check network; retry later."
        )

    if new_df.empty:
        log.info("fetch_ohlcv_incremental(%s): no new bars", symbol)
        return cached_df if cached_df is not None else new_df

    # Normalize index
    if not isinstance(new_df.index, pd.DatetimeIndex):
        new_df.index = pd.to_datetime(new_df.index)

    # Merge with cache
    if cached_df is not None and not cached_df.empty:
        merged = pd.concat([cached_df, new_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = new_df.sort_index()

    # Persist
    store.save(merged, symbol, timeframe)

    return merged.loc[:end_ts] if end else merged


# Module-level lock registry — prevents concurrent fetches for same symbol
import threading as _threading  # noqa: E402
_inflight_locks: dict = {}
_inflight_lock = _threading.Lock()


def fetch_intraday_bars(
    symbols: list[str] | str,
    *,
    timeframe: str = "5Min",
    start: "datetime.datetime | None" = None,
    end: "datetime.datetime | None" = None,
    limit: int = 10_000,
    timeout_sec: float = 30.0,
    skip_tickers: "list[str] | None" = None,
) -> dict[str, pd.DataFrame]:
    """Fetch intraday bars via Alpaca's IEX feed (free tier).

    `timeframe` is an Alpaca string: "1Min", "5Min", "15Min", "1Hour", "1Day".
    `start`/`end` are datetime objects (UTC or naive — Alpaca treats naive as UTC).
    Returns `{symbol: DataFrame}` with columns [open, high, low, close, volume, ...].

    Credentials are read from the ALPACA_API_KEY / ALPACA_SECRET_KEY env vars
    (populate via .env before calling).

    Protections (2026-04-24): wrapped in `call_with_timeout` so a stalled
    Alpaca response can't hang the caller indefinitely; `skip_tickers`
    drops permanent-miss symbols before the network call.
    """
    import datetime as _dt
    import os

    if isinstance(symbols, str):
        symbols = [symbols]
    if not symbols:
        return {}

    # Negative cache — skip symbols known to return nothing
    if skip_tickers:
        skip_set = {s.upper() for s in skip_tickers}
        symbols = [s for s in symbols if s.upper() not in skip_set]
        if not symbols:
            return {}

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
    except ImportError as exc:
        raise RuntimeError("alpaca-py not installed") from exc

    key    = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "fetch_intraday_bars: ALPACA_API_KEY + ALPACA_SECRET_KEY must be set "
            "(source .env before running)",
        )

    # Parse timeframe
    tf_map = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "10Min": TimeFrame(10, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    if timeframe not in tf_map:
        raise ValueError(f"Unknown Alpaca timeframe {timeframe!r}. "
                          f"Supported: {list(tf_map.keys())}")

    now = _dt.datetime.utcnow()
    if end is None:
        end = now
    if start is None:
        # Default: last 5 market days
        start = end - _dt.timedelta(days=7)

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    # Force IEX feed — free tier can't query current-day SIP data
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf_map[timeframe],
        start=start,
        end=end,
        limit=limit,
        feed=DataFeed.IEX,
    )

    # Hard timeout: Alpaca can hang under intermittent network; don't let
    # it block the caller (intraday_sell script, live runner, etc).
    from kernel.net_safety import call_with_timeout  # noqa: PLC0415
    bars = call_with_timeout(
        lambda: client.get_stock_bars(req),
        timeout_sec=timeout_sec,
        label=f"alpaca.get_stock_bars(n={len(symbols)}, tf={timeframe})",
    )
    if bars is None:
        log.warning("fetch_intraday_bars: Alpaca timeout after %.0fs — returning empty", timeout_sec)
        return {}
    df_all = bars.df

    out: dict[str, pd.DataFrame] = {}
    if df_all is None or df_all.empty:
        return out
    # Alpaca returns a MultiIndex DataFrame (symbol, timestamp)
    for sym in symbols:
        if sym in df_all.index.get_level_values(0):
            sub = df_all.xs(sym, level=0).copy()
            out[sym] = sub
    return out


__all__ = ["LocalStore", "fetch_ohlcv", "fetch_intraday_bars"]
