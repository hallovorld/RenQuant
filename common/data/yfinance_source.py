"""YFinance / OpenBB data source."""

import pandas as pd
from openbb import obb

from .base import DataSource


class YFinanceSource(DataSource):
    """Fetch historical OHLCV via OpenBB with the yfinance provider."""

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        kwargs: dict = {"symbol": symbol, "provider": "yfinance"}
        if start:
            kwargs["start_date"] = start
        if end:
            kwargs["end_date"] = end
        return obb.equity.price.historical(**kwargs).to_df()
