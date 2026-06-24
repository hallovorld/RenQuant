"""Regression test: a missing/zero market cap must yield NaN, not 1e16.

2026-06-24 bug: `mktcap + 1e-9` turned a divide-by-zero (missing
CommonStockSharesOutstanding → mktcap ≈ 0) into NetIncome / 1e-9 ≈ 2.7e16,
poisoning earnings_yield / book_to_price for ~45 tickers.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "fetch_sec_fundamentals",
    Path(__file__).resolve().parents[1] / "scripts" / "fetch_sec_fundamentals.py",
)
fsf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fsf)


def _ohlcv(repo: Path, ticker: str, dates, price):
    d = repo / "data" / "ohlcv" / ticker
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"close": [price] * len(dates)},
                 index=pd.to_datetime(dates)).to_parquet(d / "1d.parquet")


def _daily_row(ticker, date, shares):
    return {"ticker": ticker, "date": pd.Timestamp(date),
            "NetIncomeLoss": 1.0e7, "GrossProfit": 2.0e7, "Assets": 1.0e8,
            "StockholdersEquity": 5.0e7, "CommonStockSharesOutstanding": shares}


def test_zero_shares_market_cap_yields_nan_not_1e16(tmp_path):
    dates = ["2026-06-22", "2026-06-23"]
    _ohlcv(tmp_path, "GOOD", dates, 100.0)
    _ohlcv(tmp_path, "BADSHARES", dates, 100.0)
    daily = pd.DataFrame(
        [_daily_row("GOOD", d, shares=1.0e6) for d in dates]
        + [_daily_row("BADSHARES", d, shares=0.0) for d in dates]  # missing shares
    )
    out = fsf.compute_derived_features(daily, tmp_path / "data" / "ohlcv")
    out = out.set_index(["ticker", "date"])

    # GOOD: mktcap = 1e6 * 100 = 1e8 → earnings_yield = 1e7/1e8 = 0.1 (finite, sane)
    good = out.loc[("GOOD", pd.Timestamp("2026-06-23"))]
    assert np.isfinite(good["earnings_yield"]) and abs(good["earnings_yield"]) < 10
    assert np.isfinite(good["book_to_price"]) and abs(good["book_to_price"]) < 10

    # BADSHARES: mktcap ≈ 0 → earnings_yield / book_to_price must be NaN, NOT 1e16
    bad = out.loc[("BADSHARES", pd.Timestamp("2026-06-23"))]
    assert pd.isna(bad["earnings_yield"])
    assert pd.isna(bad["book_to_price"])


def test_zero_equity_roe_is_nan_not_huge(tmp_path):
    dates = ["2026-06-23"]
    _ohlcv(tmp_path, "ZEROEQ", dates, 100.0)
    row = _daily_row("ZEROEQ", "2026-06-23", shares=1.0e6)
    row["StockholdersEquity"] = 0.0  # zero equity → roe must be NaN not 1e16
    out = fsf.compute_derived_features(pd.DataFrame([row]), tmp_path / "data" / "ohlcv")
    assert pd.isna(out.iloc[-1]["roe"])
    # earnings_yield still fine (mktcap valid)
    assert np.isfinite(out.iloc[-1]["earnings_yield"])
