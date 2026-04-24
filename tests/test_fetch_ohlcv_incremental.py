"""Unified fetch wrapper — cache + incremental + timeout + dedup.

User spec 2026-04-24: "读数据应该有个 wrapper，来处理各种情况，保证
只读增量数据，cache，以及处理各种卡住 timeout 的情况".
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _fake_ohlcv(start_date: str, end_date: str):
    """Return a synthetic daily OHLCV frame for the requested range."""
    import pandas as pd
    idx = pd.bdate_range(start=start_date, end=end_date)
    n = len(idx)
    return pd.DataFrame({
        "open":   [100.0 + i for i in range(n)],
        "high":   [101.0 + i for i in range(n)],
        "low":    [ 99.0 + i for i in range(n)],
        "close":  [100.5 + i for i in range(n)],
        "volume": [1_000_000] * n,
    }, index=idx)


class TestColdStart:
    def test_no_cache_fetches_10yr(self, tmp_path):
        """Fresh cache → fetch 10 years of history, save."""
        import pandas as pd
        from kernel.data import fetch_ohlcv_incremental, LocalStore

        store = LocalStore(data_dir=tmp_path / "ohlcv")

        # Mock obb to return 5-year slice (we can't really test 10yr in unit)
        fake = _fake_ohlcv("2020-01-02", "2024-01-02")

        class _Resp:
            def to_df(self): return fake

        with patch("openbb.obb") as mock_obb:
            mock_obb.equity.price.historical.return_value = _Resp()
            df = fetch_ohlcv_incremental(
                "NEW_TICKER",
                end="2024-01-05",
                timeout_sec=10.0,
                store=store,
            )
            assert mock_obb.equity.price.historical.called
            assert len(df) > 0
            # Cache file should exist
            assert (tmp_path / "ohlcv" / "NEW_TICKER" / "1d.parquet").exists()


class TestIncrementalFetch:
    def test_cache_with_stale_last_date_fetches_delta_only(self, tmp_path):
        """Cache goes to 2024-01-15, request end 2024-01-20 → fetch only
        [2024-01-16, 2024-01-20], merge."""
        import pandas as pd
        from kernel.data import fetch_ohlcv_incremental, LocalStore

        store = LocalStore(data_dir=tmp_path / "ohlcv")
        # Pre-seed cache with 2024-01-02 to 2024-01-15
        existing = _fake_ohlcv("2024-01-02", "2024-01-15")
        store.save(existing, "INC_TICKER")

        # Mock: delta fetch returns just the 5-day window
        delta = _fake_ohlcv("2024-01-16", "2024-01-20")
        class _Resp:
            def to_df(self): return delta

        call_args = {}
        def _capture(**kw):
            call_args.update(kw)
            return _Resp()

        with patch("openbb.obb") as mock_obb:
            mock_obb.equity.price.historical.side_effect = _capture
            df = fetch_ohlcv_incremental(
                "INC_TICKER",
                end="2024-01-20",
                timeout_sec=10.0,
                store=store,
            )

        # Fetch was called — with START = day after last cache date
        assert call_args["start_date"] == "2024-01-16"
        assert call_args["end_date"]   == "2024-01-20"
        assert call_args["symbol"]     == "INC_TICKER"

        # Merged series should have BOTH old and new dates.
        # Note: 2024-01-20 is a Saturday → bdate_range gives last biz day 2024-01-19.
        assert pd.Timestamp("2024-01-02") in df.index
        assert pd.Timestamp("2024-01-19") in df.index

    def test_fresh_cache_skips_network(self, tmp_path):
        """Cache's last date within 2 days of `end` → return cache, no fetch."""
        import pandas as pd
        from kernel.data import fetch_ohlcv_incremental, LocalStore

        store = LocalStore(data_dir=tmp_path / "ohlcv")
        existing = _fake_ohlcv("2024-01-02", "2024-01-19")
        store.save(existing, "FRESH_TICKER")

        with patch("openbb.obb") as mock_obb:
            df = fetch_ohlcv_incremental(
                "FRESH_TICKER",
                end="2024-01-20",   # 1 day after cache last
                timeout_sec=10.0,
                store=store,
            )
            # Network NOT called — cache was fresh enough
            assert not mock_obb.equity.price.historical.called
            assert len(df) > 0


class TestTimeoutHandling:
    def test_timeout_with_cache_returns_stale(self, tmp_path):
        """Timeout + cache present → warn + return stale cache, no raise."""
        import pandas as pd
        from kernel.data import fetch_ohlcv_incremental, LocalStore

        store = LocalStore(data_dir=tmp_path / "ohlcv")
        existing = _fake_ohlcv("2024-01-02", "2024-01-05")
        store.save(existing, "TIMEOUT_WITH_CACHE")

        # Simulate hang — just barely longer than the timeout so the
        # background thread doesn't keep the xdist worker alive for 5 s
        # on each run.
        def _slow(*a, **kw):
            time.sleep(0.8)
            return MagicMock()

        with patch("openbb.obb") as mock_obb:
            mock_obb.equity.price.historical.side_effect = _slow
            df = fetch_ohlcv_incremental(
                "TIMEOUT_WITH_CACHE",
                end="2024-02-01",   # well past cache → will try to fetch
                timeout_sec=0.3,
                store=store,
            )
            # Should return stale cache, not raise
            assert len(df) > 0

    def test_timeout_no_cache_raises(self, tmp_path):
        from kernel.data import fetch_ohlcv_incremental, LocalStore

        store = LocalStore(data_dir=tmp_path / "ohlcv")

        def _slow(*a, **kw):
            time.sleep(0.8)
            return MagicMock()

        with patch("openbb.obb") as mock_obb:
            mock_obb.equity.price.historical.side_effect = _slow
            with pytest.raises(RuntimeError) as exc_info:
                fetch_ohlcv_incremental(
                    "TIMEOUT_NO_CACHE",
                    end="2024-02-01",
                    timeout_sec=0.3,
                    store=store,
                )
            assert "timed out" in str(exc_info.value).lower()


class TestDedupConcurrent:
    def test_concurrent_calls_for_same_symbol_serialize(self, tmp_path):
        """Two threads calling for the same symbol should serialize via
        the inflight lock; only ONE network call fires."""
        import pandas as pd
        from kernel.data import fetch_ohlcv_incremental, LocalStore

        store = LocalStore(data_dir=tmp_path / "ohlcv")

        call_count = [0]
        lock = threading.Lock()

        def _fake(**kw):
            with lock:
                call_count[0] += 1
            time.sleep(0.2)
            class _Resp:
                def to_df(self):
                    return _fake_ohlcv("2024-01-02", "2024-01-05")
            return _Resp()

        with patch("openbb.obb") as mock_obb:
            mock_obb.equity.price.historical.side_effect = _fake
            # Fire 3 concurrent fetches for the SAME symbol
            results = [None, None, None]
            def _worker(i):
                results[i] = fetch_ohlcv_incremental(
                    "DEDUP_TEST", end="2024-01-06", timeout_sec=5.0, store=store,
                )
            threads = [threading.Thread(target=_worker, args=(i,)) for i in range(3)]
            for t in threads: t.start()
            for t in threads: t.join()

        # First call fetches, subsequent 2 hit cache (written by first) → total 1 call
        assert call_count[0] == 1, f"expected 1 fetch, got {call_count[0]}"
        for r in results:
            assert r is not None and len(r) > 0
