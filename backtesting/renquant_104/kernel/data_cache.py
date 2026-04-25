"""Generic cached data wrapper — one pattern for all fetch paths.

User spec 2026-04-24: "读数据应该有个 wrapper，来处理各种情况，保证
只读增量数据，cache，以及处理各种卡住 timeout 的情况" + "我以后准备
用十分钟级别的数据来 train panel model，这些都要 wrap 住".

This module defines `CachedStore`, a base class all data-fetching
stores inherit from. Provides:

1. **Parquet cache** at `{cache_dir}/{key}.parquet` (or nested subdir
   for OHLCV-like data).
2. **Incremental fetch** — only fetch the delta since the cache's
   latest bar (for time-series data). Falls back to full fetch for
   non-time-series data (snapshots like fundamentals).
3. **Hard timeout** via `kernel.net_safety.call_with_timeout`.
4. **Concurrent dedup** — per-key threading.Lock so two sim threads
   fetching the same ticker serialize, the second gets the first's
   write from cache.
5. **Negative cache (skip_tickers)** — permanent misses (ETFs without
   fundamentals, foreign stocks without SEC data) are configured to
   skip — no fetch attempt per run.
6. **Freshness tolerance** — if the cache's latest bar is within
   `freshness_days`, skip network entirely.

Concrete stores (inherit / compose with this):
  * OHLCV daily (data/ohlcv/{SYM}/1d.parquet)
  * OHLCV hourly / 10-min (data/intraday/{SYM}/{1h,10min}.parquet)
  * Fundamentals (data/fundamentals/{SYM}.parquet, snapshot semantics)
  * Earnings surprise (data/earnings_surprise/{SYM}.parquet)
  * Insider trades (data/insider_trades/{SYM}.parquet)

Public API::

    from kernel.data_cache import CachedStore

    store = CachedStore(
        cache_dir="data/intraday",
        file_pattern="{symbol}/10min.parquet",
        freshness_days=0.0417,   # 1 hour for 10-min bars
        timeout_sec=30.0,
        fetch_fn=_fetch_10min_bars_from_alpaca,
    )
    df = store.get("NVDA")   # incremental fetch + cache
"""
from __future__ import annotations

import datetime
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("kernel.data_cache")


@dataclass
class CachedStore:
    """Generic cache-incremental-timeout-dedup wrapper for any data type.

    Attributes:
        cache_dir: root dir for parquet files.
        file_pattern: `{symbol}` placeholder; e.g. `"{symbol}/1d.parquet"` or
            `"{symbol}.parquet"`. Rendered via Python `.format()`.
        freshness_days: if cache's latest index value is within this many
            days of `end` (or today), skip the network call entirely.
            For daily bars use 2. For 10-min bars use 1/24. For snapshot
            data (fundamentals) use 30+.
        timeout_sec: hard timeout passed to `call_with_timeout`.
        fetch_fn: callable `(symbol, start, end) -> DataFrame`. Must
            return a DataFrame (possibly empty) or raise. `start`/`end`
            may be None for full fetches.
        time_series: when True, incremental fetch logic runs on the
            cache's max index. When False (snapshot), skip incremental —
            each fetch is full.
        skip_tickers: frozen set of symbols to never fetch (ETFs
            without fundamentals, foreign stocks without SEC data).
    """

    cache_dir:      Path
    file_pattern:   str
    fetch_fn:       Callable[..., Any]
    freshness_days: float = 2.0
    timeout_sec:    float = 30.0
    time_series:    bool  = True
    skip_tickers:   "frozenset[str]" = field(default_factory=frozenset)

    # Internal
    _locks:     dict = field(default_factory=dict)
    _lock_mut:  threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        if isinstance(self.skip_tickers, (list, tuple, set)):
            self.skip_tickers = frozenset(self.skip_tickers)

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / self.file_pattern.format(symbol=symbol)

    def _load_cache(self, symbol: str) -> "pd.DataFrame | None":
        import pandas as pd  # noqa: PLC0415
        p = self._cache_path(symbol)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            log.warning("Cache read failed for %s: %s", symbol, exc)
            return None
        if not isinstance(df.index, (pd.DatetimeIndex, pd.RangeIndex)):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
        return df

    def _save_cache(self, symbol: str, df: "pd.DataFrame") -> None:
        # Audit fix DC-2-CACHE (Round 2 deep audit, 2026-04-25): pre-fix,
        # `df.to_parquet(p)` wrote in place. If the process was killed
        # mid-write (Ctrl-C, SIGKILL, OOM), the file would be left
        # truncated/corrupt and subsequent `_load_cache` reads would
        # `pd.read_parquet` raise → cache treated as missing → cold
        # refetch every time. Worse, on some platforms the caught read
        # error returns None silently (line 107-109) so the operator
        # doesn't even see the corruption.
        # Now: write to a `.tmp` sibling and atomic-rename. Either the
        # full new file lands or the old file is preserved.
        p = self._cache_path(symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_parquet(tmp)
        tmp.replace(p)   # atomic on POSIX, best-effort on Windows

    def _get_lock(self, symbol: str) -> threading.Lock:
        with self._lock_mut:
            lk = self._locks.get(symbol)
            if lk is None:
                lk = threading.Lock()
                self._locks[symbol] = lk
            return lk

    def get(
        self,
        symbol: str,
        *,
        end: "str | datetime.datetime | None" = None,
    ) -> "pd.DataFrame | None":
        """Return cached+incremental-refreshed data for `symbol`.

        Returns None if symbol is in skip_tickers (negative cache) or
        if no cache exists AND fetch fails/times out.
        """
        if symbol in self.skip_tickers:
            log.debug("CachedStore: %s in skip_tickers — returning None", symbol)
            return None

        # Serialize concurrent calls for same symbol
        with self._get_lock(symbol):
            return self._get_unlocked(symbol, end)

    def _get_unlocked(self, symbol: str, end: Any) -> Any:
        import pandas as pd  # noqa: PLC0415

        end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.now().normalize()
        cached = self._load_cache(symbol)

        # Freshness check
        if cached is not None and not cached.empty and self.time_series:
            latest = None
            if isinstance(cached.index, pd.DatetimeIndex) and len(cached):
                latest = cached.index.max()
            if latest is not None:
                freshness_cutoff = end_ts - pd.Timedelta(days=self.freshness_days)
                if latest >= freshness_cutoff:
                    log.debug("CachedStore: %s cache fresh (latest=%s)",
                              symbol, latest)
                    return cached
                fetch_start = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                fetch_start = None
        elif cached is not None and not cached.empty and not self.time_series:
            # Snapshot: cache exists → that's enough; no staleness check
            # (caller can call .refresh() explicitly if needed).
            log.debug("CachedStore: %s snapshot cache hit", symbol)
            return cached
        else:
            # Cold start — no cache
            fetch_start = None

        # Timeout-protected fetch
        from kernel.net_safety import call_with_timeout  # noqa: PLC0415
        label = f"CachedStore[{self.file_pattern}]({symbol})"
        # Round-2 audit (#R2-16): the previous `try/except TypeError` was
        # dead code — call_with_timeout swallows ALL exceptions and
        # returns None, so the TypeError path was unreachable. Use
        # signature introspection to decide which call shape to use.
        import inspect
        try:
            sig = inspect.signature(self.fetch_fn)
            takes_start_end = len(sig.parameters) >= 3
        except (TypeError, ValueError):
            takes_start_end = True   # built-ins / callables without sig — assume yes
        if takes_start_end:
            new_df = call_with_timeout(
                self.fetch_fn, symbol, fetch_start, end_ts.strftime("%Y-%m-%d"),
                timeout_sec=self.timeout_sec, label=label,
            )
        else:
            new_df = call_with_timeout(
                self.fetch_fn, symbol,
                timeout_sec=self.timeout_sec, label=label,
            )

        if new_df is None:
            # Timeout — return stale cache if present, else None
            if cached is not None:
                log.warning("CachedStore: %s fetch timeout — returning stale cache", symbol)
                return cached
            return None

        # Merge new with old (for time series) or replace (for snapshot)
        if not isinstance(new_df.index, pd.DatetimeIndex) and self.time_series:
            try:
                new_df.index = pd.to_datetime(new_df.index)
            except Exception:
                pass

        if self.time_series and cached is not None and not cached.empty:
            merged = pd.concat([cached, new_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = new_df if len(new_df) else (cached if cached is not None else new_df)

        if len(merged):
            self._save_cache(symbol, merged)

        return merged


__all__ = ["CachedStore"]
