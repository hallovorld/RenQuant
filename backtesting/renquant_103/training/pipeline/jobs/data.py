"""DataFetchJob — fetch OHLCV for watchlist + sector ETFs + SPY."""
from __future__ import annotations

from ..base import TrainingJob
from ..context import TrainingContext


class DataFetchJob(TrainingJob):
    """Fetches OHLCV for all required tickers and populates ctx.ohlcv.

    Tickers fetched: watchlist ∪ sector_etf_map.values() ∪ {SPY}
    Uses the kernel data module (fetch_ohlcv with parquet cache).
    """

    def run(self, ctx: TrainingContext) -> None:
        from kernel.data import fetch_ohlcv

        cfg = ctx.config
        watchlist   = cfg.get("watchlist", [])
        sector_etfs = list(cfg.get("sector_etf_map", {}).values())
        tickers     = list(dict.fromkeys(watchlist + sector_etfs + ["SPY"]))

        sample_start = cfg.get("sample_start", "2018-01-01")
        sample_end   = cfg.get("sample_end", ctx.today)

        ohlcv = {}
        for t in tickers:
            df = fetch_ohlcv(t, start=sample_start, end=sample_end)
            if df is not None and len(df) > 50:
                ohlcv[t] = df

        ctx.ohlcv = ohlcv
        print(f"DataFetchJob: loaded {len(ohlcv)}/{len(tickers)} tickers")
