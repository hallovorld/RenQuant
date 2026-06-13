"""Runner price computation — runner.py make_context decomposition.

EXTRACTED 2026-06-13 from adapters/runner.py make_context() (eng plan
S2 item 5). Derives per-ticker prices from broker position marks, with
the RU-PRICE-1 dust guard (isfinite + 1-share floor so micro-qty
fractional-share dust can't produce an inflated mkt/qty price). Returns
(prices, broker_mark_prices); make_context then overlays OHLCV closes.
"""
from __future__ import annotations

import math


def compute_broker_mark_prices(
    positions_cache: dict, *, sell_only: bool, use_intraday_prices: bool,
) -> tuple[dict, dict]:
    """Per-ticker broker-mark prices from position market_value/qty.

    RU-PRICE-1 guard (2026-05-09): micro-qty (e.g. 1e-7 fractional shares
    from a botched fill) passed the old `qty > 0 and mkt > 0` check and
    produced an inflated mkt/qty price. Now: isfinite + a 0.5-share floor
    + a <1e6 sanity cap treat sub-share dust as 'no trustworthy price'.

    broker_mark_prices is always populated (used as the OHLCV-missing
    fallback); prices gets the mark only in sell-only / intraday modes
    (full daily runs must not mix real-time marks with daily closes).
    """
    prices: dict[str, float] = {}
    broker_mark_prices: dict[str, float] = {}
    for ticker, pos in positions_cache.items():
        qty = float(pos.get("qty", 0))
        mkt = float(pos.get("market_value", 0))
        if (math.isfinite(qty) and math.isfinite(mkt)
                and qty >= 0.5 and mkt > 0):
            px = mkt / qty
            if math.isfinite(px) and 0 < px < 1e6:
                broker_mark_prices[ticker] = px
                if sell_only or use_intraday_prices:
                    prices[ticker] = px
    return prices, broker_mark_prices
