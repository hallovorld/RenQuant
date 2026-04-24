"""Regression: fetch_ohlcv must not hang forever on a stuck upstream.

2026-04-24 notebook incident: notebook kernel was in state S (sleeping)
with 15+ CLOSE_WAIT sockets to yahoo.com for 4 HOURS, stuck on a
fetch_ohlcv call that had no timeout. Now fetch_ohlcv goes through
kernel.net_safety.call_with_timeout with a 30s default.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


class TestFetchOhlcvTimeout:
    def test_hanging_upstream_raises_runtime_error(self):
        """If the underlying fetch hangs > timeout, raise RuntimeError with
        a clear message, don't hang forever."""
        import pandas as pd
        from kernel.data import fetch_ohlcv
        # Inject a hanging fetch by patching the OpenBB lazy import path.
        # The fetcher function inside fetch_ohlcv imports `obb` at call
        # time, so we patch the call itself by making a slow stand-in.

        def _slow(*a, **kw):
            time.sleep(60)   # longer than our 0.5s test timeout
            return pd.DataFrame()

        # Patch call_with_timeout to actually fire — but keep it cheap.
        # Trick: give timeout_sec=0.5, inject _slow as the inner fn via
        # a module-level monkey-patch of the OpenBB historical call. That
        # path is inside a local closure, so patch obb itself.
        with patch("openbb.obb") as mock_obb:
            mock_obb.equity.price.historical.side_effect = _slow
            with pytest.raises(RuntimeError) as exc_info:
                fetch_ohlcv(
                    "STUCK_SYMBOL_DOES_NOT_EXIST",
                    cache=False,
                    timeout_sec=0.5,
                )
            assert "timed out" in str(exc_info.value).lower()

    def test_normal_fetch_still_works(self):
        """Fast fetch path (mocked) still returns a DataFrame."""
        import pandas as pd
        from kernel.data import fetch_ohlcv

        fake_df = pd.DataFrame({
            "open":   [100.0, 101.0],
            "high":   [101.0, 102.0],
            "low":    [ 99.0, 100.0],
            "close":  [100.5, 101.5],
            "volume": [1_000_000, 1_100_000],
        }, index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]))

        class _MockResp:
            def to_df(self): return fake_df

        with patch("openbb.obb") as mock_obb:
            mock_obb.equity.price.historical.return_value = _MockResp()
            result = fetch_ohlcv("FAKE_TICKER", cache=False, timeout_sec=10.0)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2

    def test_cache_hit_bypasses_timeout_path(self, tmp_path):
        """If parquet cache already covers the range, no network call
        happens at all (so no timeout risk)."""
        import pandas as pd
        from kernel.data import fetch_ohlcv, LocalStore, _default_store

        # Seed the cache
        fake_df = pd.DataFrame({
            "open": [100.0], "high": [100.0], "low": [100.0],
            "close": [100.0], "volume": [1_000_000],
        }, index=pd.DatetimeIndex(["2024-01-02"]))

        with patch.object(_default_store, "has_range", return_value=True):
            with patch.object(_default_store, "load", return_value=fake_df):
                # Even with timeout=0.01 we should return quickly because
                # the cache path doesn't touch network
                result = fetch_ohlcv("CACHED_TICKER", cache=True, timeout_sec=0.01)
                assert len(result) == 1
