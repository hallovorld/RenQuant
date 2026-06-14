"""LEAN price / symbol resolution — lean.py decomposition slice 4.

EXTRACTED 2026-06-14 from adapters/lean.py (eng plan S2 item 5, god-file
decomposition). Pure functions that resolve a watchlist/sector/benchmark
ticker to its LEAN Symbol and to a current executable price (current Slice →
current Security price → latest OHLCV close) — so a ranked buy candidate
always has a ctx.prices entry and is not rejected downstream as
size_bad_price. No LeanAdapter state. Re-exported from lean for back-compat.
"""
from __future__ import annotations

from typing import Any

from adapters.lean_order import _positive_finite_price


def _symbol_for_ticker(algo: Any, ticker: str):
    """Resolve regular watchlist, sector ETF, or benchmark symbols."""
    sym = algo.symbols.get(ticker)
    if sym is not None:
        return sym
    sym = algo._sector_etf_symbols.get(ticker)
    if sym is not None:
        return sym
    if ticker == getattr(algo, "_benchmark", None):
        return getattr(algo, "_spy_sym", None)
    return None


def _current_price_for_ticker(
    algo: Any,
    data: Any,
    ticker: str,
    ohlcv: dict[str, Any],
) -> float | None:
    """Return the current executable price for any ticker the pipeline may size.

    SimAdapter and RunnerAdapter populate prices for all model/watchlist names.
    LEAN must do the same: a ranked buy candidate with no ``ctx.prices`` entry
    is rejected downstream as ``size_bad_price`` even if the model signal is
    valid. Price source order mirrors the execution surface: current Slice,
    current Security price, then latest OHLCV close.
    """
    sym = _symbol_for_ticker(algo, ticker)
    if sym is not None:
        try:
            if data.ContainsKey(sym):
                px = _positive_finite_price(data[sym].Close)
                if px is not None:
                    return px
        except Exception:
            pass
        securities = getattr(algo, "Securities", None)
        if securities is not None:
            try:
                px = _positive_finite_price(securities[sym].Price)
                if px is not None:
                    return px
            except Exception:
                pass

    df = ohlcv.get(ticker)
    if df is not None and not getattr(df, "empty", True):
        try:
            close = df["close"].dropna()
            if not close.empty:
                return _positive_finite_price(close.iloc[-1])
        except Exception:
            pass
    return None
