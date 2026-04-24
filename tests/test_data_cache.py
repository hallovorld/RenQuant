"""Generic CachedStore — cache + incremental + timeout + dedup + negative.

User spec 2026-04-24: all data fetchers should go through this
pattern. This is the generic base — concrete stores (OHLCV / hourly
bars / fundamentals / earnings / insider / future 10-min bars) will
inherit/compose with it.
"""
from __future__ import annotations

import datetime
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _ts_df(start: str, periods: int):
    """Create a synthetic time-series DataFrame."""
    import pandas as pd
    idx = pd.bdate_range(start=start, periods=periods)
    return pd.DataFrame(
        {"open": [100.0 + i for i in range(periods)],
         "close": [100.0 + i for i in range(periods)]},
        index=idx,
    )


def _snapshot_df():
    """Create a synthetic single-row snapshot."""
    import pandas as pd
    return pd.DataFrame({"earnings_yield": [0.05], "roe": [0.15]},
                         index=[pd.Timestamp("2026-04-24")])


class TestSkipTickers:
    def test_skip_ticker_returns_none_no_fetch(self, tmp_path):
        from kernel.data_cache import CachedStore
        called = []
        def _fetch(sym, start=None, end=None):
            called.append(sym)
            return _ts_df("2024-01-02", 10)
        store = CachedStore(
            cache_dir=tmp_path, file_pattern="{symbol}.parquet",
            fetch_fn=_fetch, skip_tickers=["XLF", "XLK"],
            time_series=True,
        )
        assert store.get("XLF") is None
        assert store.get("XLK") is None
        assert called == []   # no fetch attempts


class TestCacheHit:
    def test_fresh_cache_skips_network(self, tmp_path):
        from kernel.data_cache import CachedStore
        import pandas as pd

        # Pre-seed cache
        cache_root = tmp_path
        (cache_root / "NVDA.parquet").parent.mkdir(parents=True, exist_ok=True)
        seed = _ts_df("2024-01-02", 100)
        seed.to_parquet(cache_root / "NVDA.parquet")
        # "Today" is right after last date → should be considered fresh

        called = []
        def _fetch(sym, start=None, end=None):
            called.append(sym)
            return _ts_df("2024-01-02", 10)

        store = CachedStore(
            cache_dir=cache_root, file_pattern="{symbol}.parquet",
            fetch_fn=_fetch, freshness_days=5,
        )

        end_ts = seed.index.max() + pd.Timedelta(days=1)
        result = store.get("NVDA", end=end_ts)
        assert result is not None
        assert len(result) == 100
        assert called == []   # cache fresh, no network


class TestIncrementalFetch:
    def test_stale_cache_fetches_delta_only(self, tmp_path):
        from kernel.data_cache import CachedStore
        import pandas as pd

        cache_root = tmp_path
        # Seed: cache to 2024-01-15
        seed = _ts_df("2024-01-02", 10)   # ends 2024-01-15
        seed.to_parquet(cache_root / "INC.parquet")

        captured_start = []
        def _fetch(sym, start=None, end=None):
            captured_start.append(start)
            return _ts_df("2024-01-16", 5)

        store = CachedStore(
            cache_dir=cache_root, file_pattern="{symbol}.parquet",
            fetch_fn=_fetch, freshness_days=2,
        )
        result = store.get("INC", end="2024-01-22")
        # Fetch should have been asked for delta starting AFTER last cache date
        assert captured_start[0] == "2024-01-16"
        # Merged frame should have both ranges
        assert pd.Timestamp("2024-01-02") in result.index
        assert pd.Timestamp("2024-01-22") in result.index


class TestTimeoutHandling:
    def test_timeout_with_cache_returns_stale(self, tmp_path):
        from kernel.data_cache import CachedStore
        import pandas as pd

        cache_root = tmp_path
        seed = _ts_df("2024-01-02", 10)
        seed.to_parquet(cache_root / "TO.parquet")

        def _slow(sym, start=None, end=None):
            time.sleep(0.6)   # just > timeout_sec; keeps test under a second
            return _ts_df("2024-01-16", 5)

        store = CachedStore(
            cache_dir=cache_root, file_pattern="{symbol}.parquet",
            fetch_fn=_slow, timeout_sec=0.2, freshness_days=0.01,
        )
        result = store.get("TO", end="2024-02-01")   # far past seed → triggers fetch
        # Timeout → stale cache returned
        assert result is not None
        assert len(result) == 10   # original seed length

    def test_timeout_no_cache_returns_none(self, tmp_path):
        from kernel.data_cache import CachedStore

        def _slow(sym, start=None, end=None):
            time.sleep(0.6)
            return _ts_df("2024-01-16", 5)

        store = CachedStore(
            cache_dir=tmp_path, file_pattern="{symbol}.parquet",
            fetch_fn=_slow, timeout_sec=0.2,
        )
        result = store.get("NONE")
        assert result is None


class TestConcurrentDedup:
    def test_concurrent_calls_same_symbol_serialize(self, tmp_path):
        """3 concurrent threads calling the SAME symbol with an end-date
        that will be satisfied by the FIRST thread's fetch → subsequent
        threads hit fresh cache, only 1 network call total."""
        from kernel.data_cache import CachedStore
        import pandas as pd

        call_count = [0]
        mutex = threading.Lock()

        # Synthetic fetch returns bars up to "today" (fresh enough that
        # subsequent threads skip network).
        def _fetch(sym, start=None, end=None):
            with mutex:
                call_count[0] += 1
            time.sleep(0.2)
            # Return a frame ending "today" so cache is fresh post-write
            today = pd.Timestamp.now().normalize()
            return pd.DataFrame({"close": [100.0]}, index=[today])

        store = CachedStore(
            cache_dir=tmp_path, file_pattern="{symbol}.parquet",
            fetch_fn=_fetch, timeout_sec=10, freshness_days=5,
        )

        results = [None, None, None]
        def _worker(i):
            results[i] = store.get("DEDUP_T")
        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(3)]
        for t in threads: t.start()
        for t in threads: t.join()

        # Only 1 network call — subsequent threads got cache
        assert call_count[0] == 1
        for r in results:
            assert r is not None and len(r) > 0


class TestSnapshotMode:
    def test_snapshot_cache_always_hits(self, tmp_path):
        """Snapshot data (fundamentals) — cache hit doesn't check freshness."""
        from kernel.data_cache import CachedStore

        # Seed with an OLD snapshot
        seed = _snapshot_df()
        seed.to_parquet(tmp_path / "NVDA.parquet")

        called = []
        def _fetch(sym, start=None, end=None):
            called.append(sym)
            return _snapshot_df()

        store = CachedStore(
            cache_dir=tmp_path, file_pattern="{symbol}.parquet",
            fetch_fn=_fetch, time_series=False,
        )
        result = store.get("NVDA")
        assert len(result) == 1
        assert called == []   # snapshot cache hit, no fetch
