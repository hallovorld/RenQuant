"""Sim price / ticker-universe resolution — sim.py decomposition (S2 item 5).

EXTRACTED 2026-06-14 from adapters/sim.py. Pure functions that pick the bar
frame used to price a ticker, and build the universe of tickers that must
receive a current-bar price (watchlist + model names + sector ETFs + held +
benchmark + optional beta-sleeve). Keeping sim aligned with live/LEAN here is
what stops the beta sleeve silently no-op'ing only in research. SimAdapter
keeps thin method delegates; behavior is unchanged. No SimAdapter state —
self-deps are passed in.
"""
from __future__ import annotations

from typing import Any


def price_frame_for(ticker: str, *, ohlcv: dict, config: dict, spy_df: Any):
    """Return the bar frame used to price ``ticker`` in sim context, or None."""
    key = str(ticker or "").strip().upper()
    if not key:
        return None
    df = ohlcv.get(key)
    benchmark = str(config.get("benchmark") or "SPY").strip().upper()
    if df is None and key == benchmark and spy_df is not None:
        df = spy_df
    return df


def context_price_tickers(
    *,
    config: dict,
    models,
    sector_etf_map: dict,
    holdings,
) -> list[str]:
    """Ticker universe that must receive current-bar prices.

    Live and LEAN price the watchlist/model universe, sector ETFs, held
    positions, and benchmark. Keep sim aligned so optional beta sleeve logic
    cannot silently no-op only in research.
    """
    from kernel.pipeline.task_benchmark_sleeve import (  # noqa: PLC0415
        benchmark_sleeve_ticker,
    )

    from adapters.sleeve_prices import parking_sleeve_price_tickers  # noqa: PLC0415

    tickers: list[str] = []
    tickers.extend(str(t).upper() for t in config.get("watchlist", []) if t)
    tickers.extend(str(t).upper() for t in models)
    tickers.extend(str(t).upper() for t in sector_etf_map.values() if t)
    tickers.extend(str(t).upper() for t in holdings)
    benchmark = str(config.get("benchmark") or "SPY").strip().upper()
    if benchmark:
        tickers.append(benchmark)
    sleeve_ticker = benchmark_sleeve_ticker(config)
    if sleeve_ticker:
        tickers.append(sleeve_ticker)
    # Parking-sleeve legs (st104 #39 follow-up) — [] unless sleeve.enabled.
    tickers.extend(parking_sleeve_price_tickers(config))
    return list(dict.fromkeys(tickers))
