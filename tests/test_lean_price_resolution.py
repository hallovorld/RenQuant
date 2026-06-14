"""Unit tests for the extracted LEAN price / symbol resolution helpers.

Pins adapters/lean_price.py (lean.py decomposition slice 4) at the module
boundary. _current_price_for_ticker is the LEAN price surface: a ranked buy
candidate with no resolvable price is rejected downstream as size_bad_price,
so the source-order fallback (current Slice -> Security price -> OHLCV close)
and its positive-finite guarding are correctness-critical.

REGRESSION GUARD: both helpers must remain importable from BOTH
adapters.lean_price (canonical) and adapters.lean (back-compat re-export) as
the SAME object — make_context and several LeanAdapter methods call them by
the re-exported name.
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

from adapters import lean as _lean  # noqa: E402
from adapters import lean_price as _lp  # noqa: E402


def _algo(**kw):
    base = dict(symbols={}, _sector_etf_symbols={}, _benchmark=None,
               _spy_sym=None, Securities=None)
    base.update(kw)
    return NS(**base)


class TestReexportIdentity:
    def test_same_objects(self):
        assert _lean._symbol_for_ticker is _lp._symbol_for_ticker
        assert _lean._current_price_for_ticker is _lp._current_price_for_ticker


class TestSymbolForTicker:
    def test_watchlist_first(self):
        algo = _algo(symbols={"AAPL": "A_SYM"},
                     _sector_etf_symbols={"AAPL": "WRONG"})
        assert _lp._symbol_for_ticker(algo, "AAPL") == "A_SYM"

    def test_sector_etf_second(self):
        algo = _algo(_sector_etf_symbols={"XLK": "XLK_SYM"})
        assert _lp._symbol_for_ticker(algo, "XLK") == "XLK_SYM"

    def test_benchmark_third(self):
        algo = _algo(_benchmark="SPY", _spy_sym="SPY_SYM")
        assert _lp._symbol_for_ticker(algo, "SPY") == "SPY_SYM"

    def test_unknown_is_none(self):
        assert _lp._symbol_for_ticker(_algo(), "ZZZ") is None


class _Slice:
    """Minimal QCAlgorithm Slice stand-in: ContainsKey + subscript (special
    methods must live on the type, not a SimpleNamespace instance attr)."""
    def __init__(self, contains, mapping=None):
        self._contains = contains
        self._mapping = mapping or {}

    def ContainsKey(self, sym):
        return self._contains

    def __getitem__(self, sym):
        return self._mapping[sym]


class TestCurrentPriceForTicker:
    def test_slice_close_is_primary_source(self):
        algo = _algo(symbols={"AAPL": "AAPL_SYM"})
        data = _Slice(contains=True, mapping={"AAPL_SYM": NS(Close=101.0)})
        assert _lp._current_price_for_ticker(algo, data, "AAPL", {}) == 101.0

    def test_falls_back_to_security_price(self):
        algo = _algo(symbols={"AAPL": "AAPL_SYM"},
                     Securities={"AAPL_SYM": NS(Price=55.0)})
        data = _Slice(contains=False)
        assert _lp._current_price_for_ticker(algo, data, "AAPL", {}) == 55.0

    def test_falls_back_to_ohlcv_close(self):
        algo = _algo()  # no symbol → straight to OHLCV
        df = pd.DataFrame({"close": [10.0, 11.0, 12.5]})
        data = _Slice(contains=False)
        assert _lp._current_price_for_ticker(algo, data, "NVDA", {"NVDA": df}) == 12.5

    def test_non_positive_slice_price_skipped(self):
        # zero/negative Slice close must NOT be returned; fall through to OHLCV.
        algo = _algo(symbols={"NVDA": "NVDA_SYM"})
        data = _Slice(contains=True, mapping={"NVDA_SYM": NS(Close=0.0)})
        df = pd.DataFrame({"close": [9.0]})
        assert _lp._current_price_for_ticker(algo, data, "NVDA", {"NVDA": df}) == 9.0

    def test_exception_in_slice_is_swallowed(self):
        def _boom(s):
            raise RuntimeError("LEAN slice error")
        algo = _algo(symbols={"NVDA": "NVDA"})
        data = NS(ContainsKey=_boom)
        df = pd.DataFrame({"close": [7.0]})
        assert _lp._current_price_for_ticker(algo, data, "NVDA", {"NVDA": df}) == 7.0

    def test_none_when_nothing_resolves(self):
        algo = _algo()
        data = NS(ContainsKey=lambda s: False)
        assert _lp._current_price_for_ticker(algo, data, "ZZZ", {}) is None

    def test_empty_ohlcv_frame_is_none(self):
        algo = _algo()
        data = NS(ContainsKey=lambda s: False)
        assert _lp._current_price_for_ticker(
            algo, data, "ZZZ", {"ZZZ": pd.DataFrame({"close": []})}) is None

    def test_all_nan_close_column_is_none(self):
        algo = _algo()
        data = NS(ContainsKey=lambda s: False)
        df = pd.DataFrame({"close": [float("nan"), float("nan")]})
        assert _lp._current_price_for_ticker(algo, data, "ZZZ", {"ZZZ": df}) is None
