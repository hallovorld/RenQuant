"""Unit tests for the extracted sim price / ticker-universe resolution.

Pins adapters/sim_price.py (sim.py decomposition, S2 item 5). The SimAdapter
methods _price_frame_for / _context_price_tickers are now thin delegates; this
locks the behavior they delegate to (incl. the delegate parity that makes the
extraction behavior-preserving).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.sim_price import context_price_tickers, price_frame_for  # noqa: E402


class TestPriceFrameFor:
    def test_returns_ohlcv_frame_for_known_ticker(self):
        df = pd.DataFrame({"close": [1.0]})
        out = price_frame_for("aapl", ohlcv={"AAPL": df}, config={}, spy_df=None)
        assert out is df  # upcased key hit

    def test_blank_ticker_is_none(self):
        assert price_frame_for("", ohlcv={"AAPL": 1}, config={}, spy_df=None) is None
        assert price_frame_for(None, ohlcv={}, config={}, spy_df=None) is None

    def test_benchmark_falls_back_to_spy_df(self):
        spy = pd.DataFrame({"close": [9.0]})
        out = price_frame_for("SPY", ohlcv={}, config={"benchmark": "SPY"}, spy_df=spy)
        assert out is spy

    def test_non_benchmark_miss_is_none(self):
        spy = pd.DataFrame({"close": [9.0]})
        assert price_frame_for("NVDA", ohlcv={}, config={"benchmark": "SPY"},
                               spy_df=spy) is None

    def test_ohlcv_preferred_over_spy_fallback_for_benchmark(self):
        bench_df = pd.DataFrame({"close": [1.0]})
        spy = pd.DataFrame({"close": [9.0]})
        out = price_frame_for("SPY", ohlcv={"SPY": bench_df},
                              config={"benchmark": "SPY"}, spy_df=spy)
        assert out is bench_df


class TestContextPriceTickers:
    def test_union_deduped_and_upcased(self):
        out = context_price_tickers(
            config={"watchlist": ["aapl", "msft"], "benchmark": "SPY"},
            models={"NVDA": object()},
            sector_etf_map={"tech": "xlk"},
            holdings={"aapl": object()})  # AAPL also in watchlist → deduped
        assert out == list(dict.fromkeys(out))            # no dups
        for t in ("AAPL", "MSFT", "NVDA", "XLK", "SPY"):
            assert t in out
        assert out.count("AAPL") == 1

    def test_empty_inputs_yield_just_benchmark(self):
        out = context_price_tickers(config={"benchmark": "SPY"}, models={},
                                    sector_etf_map={}, holdings={})
        assert out == ["SPY"]

    def test_falsy_entries_skipped(self):
        out = context_price_tickers(
            config={"watchlist": ["aapl", "", None], "benchmark": "SPY"},
            models={}, sector_etf_map={"x": None}, holdings={})
        assert "AAPL" in out and "" not in out and "SPY" in out


class TestDelegateParity:
    def test_delegates_match_pure_functions(self):
        from adapters.sim import SimAdapter

        adapter = SimAdapter.__new__(SimAdapter)  # skip __init__
        spy = pd.DataFrame({"close": [9.0]})
        adapter._ohlcv = {"AAPL": pd.DataFrame({"close": [1.0]})}
        adapter._spy_df = spy
        adapter._config = {"watchlist": ["AAPL"], "benchmark": "SPY"}
        adapter._models = {"NVDA": object()}
        adapter._sector_etf_map = {"tech": "XLK"}
        adapter._holdings = {"AAPL": object()}

        assert adapter._price_frame_for("AAPL") is price_frame_for(
            "AAPL", ohlcv=adapter._ohlcv, config=adapter._config, spy_df=spy)
        assert adapter._context_price_tickers() == context_price_tickers(
            config=adapter._config, models=adapter._models,
            sector_etf_map=adapter._sector_etf_map, holdings=adapter._holdings)
