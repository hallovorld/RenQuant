"""Tests for kernel/fundamentals.py — cached equity fundamentals."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.fundamentals import (  # noqa: E402
    FACTOR_COLS,
    FundamentalsStore,
    fetch_fundamentals,
    fetch_fundamentals_watchlist,
)


def _fake_provider(values: dict[str, float]):
    """Return a provider_fn stub that always yields the same snapshot."""
    def _fetch(symbol: str) -> dict[str, float]:
        return dict(values)
    return _fetch


class TestFundamentalsStore:
    def test_empty_cache_returns_none(self, tmp_path):
        store = FundamentalsStore(data_dir=tmp_path)
        assert store.load("AAPL") is None
        assert store.latest("AAPL") is None

    def test_save_and_load_roundtrip(self, tmp_path):
        store = FundamentalsStore(data_dir=tmp_path)
        idx = pd.DatetimeIndex([pd.Timestamp("2026-04-20")], name="date")
        df = pd.DataFrame([{c: 0.1 for c in FACTOR_COLS}], index=idx)
        store.save(df, "AAPL")

        loaded = store.load("AAPL")
        assert loaded is not None
        assert len(loaded) == 1
        assert set(FACTOR_COLS).issubset(loaded.columns)
        assert loaded.iloc[0]["earnings_yield"] == pytest.approx(0.1)

    def test_save_appends_and_dedupes_by_index(self, tmp_path):
        store = FundamentalsStore(data_dir=tmp_path)
        idx1 = pd.DatetimeIndex([pd.Timestamp("2026-04-20")], name="date")
        idx2 = pd.DatetimeIndex([pd.Timestamp("2026-04-21")], name="date")
        dup  = pd.DatetimeIndex([pd.Timestamp("2026-04-20")], name="date")

        store.save(pd.DataFrame([{c: 0.1 for c in FACTOR_COLS}], index=idx1), "AAPL")
        store.save(pd.DataFrame([{c: 0.2 for c in FACTOR_COLS}], index=idx2), "AAPL")
        store.save(pd.DataFrame([{c: 0.3 for c in FACTOR_COLS}], index=dup),  "AAPL")

        loaded = store.load("AAPL")
        assert len(loaded) == 2  # dedup collapses the 2026-04-20 rows
        assert loaded.iloc[0]["earnings_yield"] == pytest.approx(0.3)  # latest wins

    def test_latest_returns_float_dict(self, tmp_path):
        store = FundamentalsStore(data_dir=tmp_path)
        idx = pd.DatetimeIndex([pd.Timestamp("2026-04-20")], name="date")
        df = pd.DataFrame(
            [{"earnings_yield": 0.05, "roe": 0.18,
              "gross_profitability": 0.32, "book_to_price": 0.22}],
            index=idx,
        )
        store.save(df, "AAPL")
        latest = store.latest("AAPL")
        # Present columns come through with their values; newer columns (like
        # short_pct_float, added later) fill as NaN on older cached rows.
        assert latest["earnings_yield"] == pytest.approx(0.05)
        assert latest["roe"]            == pytest.approx(0.18)
        assert latest["gross_profitability"] == pytest.approx(0.32)
        assert latest["book_to_price"]  == pytest.approx(0.22)
        if "short_pct_float" in latest:
            import math
            assert math.isnan(latest["short_pct_float"])


class TestFetchFundamentals:
    def test_fetch_with_injected_provider(self, tmp_path):
        store = FundamentalsStore(data_dir=tmp_path)
        provider = _fake_provider({
            "earnings_yield":      0.06,
            "roe":                 0.20,
            "gross_profitability": 0.30,
            "book_to_price":       0.25,
        })
        result = fetch_fundamentals("AAPL", store=store, provider_fn=provider)
        assert result["earnings_yield"] == pytest.approx(0.06)

        # Cached row should be present
        loaded = store.load("AAPL")
        assert loaded is not None and len(loaded) == 1

    def test_cache_returns_without_provider_call(self, tmp_path):
        store = FundamentalsStore(data_dir=tmp_path)
        # Seed the cache
        idx = pd.DatetimeIndex([pd.Timestamp("2026-04-19")], name="date")
        df  = pd.DataFrame([{c: 0.11 for c in FACTOR_COLS}], index=idx)
        store.save(df, "AAPL")

        call_count = {"n": 0}
        def _never_called_provider(_sym: str) -> dict[str, float]:
            call_count["n"] += 1
            raise AssertionError("provider_fn should not be called when cache hits")

        result = fetch_fundamentals("AAPL", store=store,
                                    provider_fn=_never_called_provider)
        assert call_count["n"] == 0
        assert result["earnings_yield"] == pytest.approx(0.11)

    def test_no_cache_forces_provider(self, tmp_path):
        store = FundamentalsStore(data_dir=tmp_path)
        idx = pd.DatetimeIndex([pd.Timestamp("2026-04-19")], name="date")
        df  = pd.DataFrame([{c: 0.11 for c in FACTOR_COLS}], index=idx)
        store.save(df, "AAPL")

        provider = _fake_provider({c: 0.22 for c in FACTOR_COLS})
        result = fetch_fundamentals("AAPL", cache=False, store=store,
                                    provider_fn=provider)
        assert result["earnings_yield"] == pytest.approx(0.22)


class TestFetchWatchlist:
    def test_iterates_all_tickers(self, tmp_path):
        store = FundamentalsStore(data_dir=tmp_path)
        provider = _fake_provider({c: 0.5 for c in FACTOR_COLS})
        out = fetch_fundamentals_watchlist(
            ["AAPL", "MSFT", "NVDA"], store=store, provider_fn=provider,
        )
        assert set(out.keys()) == {"AAPL", "MSFT", "NVDA"}
        for sym, row in out.items():
            assert row["earnings_yield"] == pytest.approx(0.5)
            assert store.load(sym) is not None

    def test_errors_on_one_ticker_dont_kill_others(self, tmp_path):
        store = FundamentalsStore(data_dir=tmp_path)
        def _flaky(symbol: str) -> dict[str, float]:
            if symbol == "BAD":
                raise RuntimeError("network blip")
            return {c: 0.7 for c in FACTOR_COLS}

        out = fetch_fundamentals_watchlist(
            ["AAPL", "BAD", "MSFT"], store=store, provider_fn=_flaky,
        )
        assert "AAPL" in out and "MSFT" in out
        assert "BAD" not in out  # was dropped on error
