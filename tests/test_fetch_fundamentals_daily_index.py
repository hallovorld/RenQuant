"""Regression test for fetch_sec_fundamentals.build_daily_index.

2026-06-23 incident: the daily index was sourced only from a separately-rebuilt
alpha158 dataset; when that dataset went stale it silently capped the
fundamentals panel's end date (a "refresh" produced a file ending weeks in the
past). build_daily_index must always reach today.
"""
from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "fetch_sec_fundamentals",
    Path(__file__).resolve().parents[1] / "scripts" / "fetch_sec_fundamentals.py",
)
fsf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fsf)


def test_extends_to_today_even_when_hist_is_stale(tmp_path):
    # historical dates stop mid-Feb (the stale-artifact case); no SPY ohlcv →
    # business-day fallback must still carry the index to `today`.
    hist = pd.bdate_range("2026-01-02", "2026-02-10")
    idx = fsf.build_daily_index(hist, tmp_path, today=date(2026, 6, 23))
    assert idx.max().date() == date(2026, 6, 23)   # NOT capped at 2026-02-10
    assert idx.min().date() == date(2026, 1, 2)


def test_prefers_spy_trading_days_when_present(tmp_path):
    spy_dir = tmp_path / "data" / "ohlcv" / "SPY"
    spy_dir.mkdir(parents=True)
    spy_days = pd.bdate_range("2026-02-09", "2026-06-23")
    pd.DataFrame({"date": spy_days, "close": 1.0}).to_parquet(
        spy_dir / "1d.parquet", index=False)
    hist = pd.bdate_range("2026-01-02", "2026-02-10")
    idx = fsf.build_daily_index(hist, tmp_path, today=date(2026, 6, 23))
    assert idx.max().date() == date(2026, 6, 23)
    # SPY/business trading days only — no weekend dates anywhere in the index
    assert not any(ts.weekday() >= 5 for ts in idx)


def test_empty_history_still_reaches_today(tmp_path):
    idx = fsf.build_daily_index([], tmp_path, today=date(2026, 6, 23))
    assert idx.max().date() == date(2026, 6, 23)
