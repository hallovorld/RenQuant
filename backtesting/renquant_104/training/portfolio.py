"""Portfolio simulation utilities for notebook backtests."""

import numpy as np
import pandas as pd


def compute_portvals(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    start_val: float = 100_000,
    commission: float = 0.0,
    impact: float = 0.0,
) -> pd.Series:
    """Simulate portfolio value from a trade schedule."""
    if "close" in prices.columns and len(trades.columns) == 1:
        price_frame = pd.DataFrame(
            {trades.columns[0]: prices["close"]}, index=prices.index
        )
    else:
        price_frame = prices

    dates = price_frame.index
    symbols = trades.columns.tolist()
    holdings = pd.DataFrame(0.0, index=dates, columns=symbols)
    cash = pd.Series(start_val, index=dates, dtype=float)

    for i, date in enumerate(dates):
        if i > 0:
            holdings.iloc[i] = holdings.iloc[i - 1]
            cash.iloc[i] = cash.iloc[i - 1]

        if date not in trades.index:
            continue

        for sym in symbols:
            shares = trades.at[date, sym] if date in trades.index else 0.0
            if shares == 0:
                continue
            price = price_frame.at[date, sym]
            # Audit fix P-1 (Round 8, 2026-04-25): pre-fix, NaN/inf price
            # silently propagated into cash. Once NaN, every subsequent
            # cash value was NaN → portfolio_stats returned NaN Sharpe
            # silently. Skip the trade if price isn't usable; the
            # notebook caller can flag the missing price separately.
            if not np.isfinite(price):
                continue
            if shares > 0:
                cost = price * shares * (1 + impact) + commission
                cash.iloc[i] -= cost
            else:
                proceeds = price * abs(shares) * (1 - impact) - commission
                cash.iloc[i] += proceeds
            holdings.at[date, sym] += shares

    portval = (holdings * price_frame).sum(axis=1) + cash
    portval.name = "portfolio_value"
    return portval


def portfolio_stats(portvals: pd.Series) -> dict:
    """Compute common performance metrics from a portfolio value series."""
    daily_ret = portvals.pct_change().dropna()
    cum_ret = portvals.iloc[-1] / portvals.iloc[0] - 1
    avg_daily = daily_ret.mean()
    std_daily = daily_ret.std()
    sharpe = np.sqrt(252) * avg_daily / std_daily if std_daily > 0 else 0.0
    return {
        "cumulative_return": cum_ret,
        "mean_daily_return": avg_daily,
        "std_daily_return": std_daily,
        "sharpe_ratio": sharpe,
    }


__all__ = ["compute_portvals", "portfolio_stats"]
