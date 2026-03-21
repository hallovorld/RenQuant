import pandas as pd
from openbb import obb


def fetch_ohlcv(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """Fetch OHLCV data for a symbol via OpenBB.

    Args:
        symbol: Ticker symbol, e.g. "NVDA".
        start:  Start date string "YYYY-MM-DD". Defaults to provider's maximum history.
        end:    End date string "YYYY-MM-DD". Defaults to today.
        provider: OpenBB data provider (default: "yfinance").
    """
    kwargs = {"symbol": symbol, "provider": provider}
    if start:
        kwargs["start_date"] = start
    if end:
        kwargs["end_date"] = end
    return obb.equity.price.historical(**kwargs).to_df()
