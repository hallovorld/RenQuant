"""Phase 2B borrow-cost regression — daily charge on short positions.

Per Alpaca 2026-05-14 live API research:
  - ETB (easy_to_borrow=True): default 0.005/yr (50 bps)
  - HTB (easy_to_borrow=False): default 0.05/yr (500 bps)
  - shortable=False: filtered upstream by ShortCandidateSelectionTask

Tests pin:
1. No charge when no shorts open
2. ETB charge = |short_value| × 0.005 / 252 per bar
3. HTB charge = |short_value| × 0.05 / 252 per bar
4. Missing-ticker fail-open (assume ETB)
5. Rate overridable via config
6. Cash NaN guard
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _mk_adapter(holdings: dict, *, cash: float = 100_000.0,
                ohlcv: dict | None = None,
                borrow_status: dict | None = None,
                config: dict | None = None):
    """Construct a minimal SimAdapter using __new__ so we don't load any
    artifacts; only the fields needed by _charge_daily_borrow."""
    from adapters.sim import SimAdapter
    adapter = SimAdapter.__new__(SimAdapter)
    adapter._cash = cash
    adapter.holdings = holdings
    adapter._ohlcv = ohlcv or {}
    adapter._borrow_status_cache = borrow_status or {}
    adapter._strategy_config = config or {}
    return adapter


class TestDailyBorrowCharge:

    def test_no_shorts_no_charge(self):
        from adapters.sim import SimAdapter
        adapter = _mk_adapter(
            holdings={"AAPL": SimpleNamespace(shares=100, entry_price=150.0)},
        )
        before = adapter._cash
        adapter._charge_daily_borrow(today_ts=pd.Timestamp("2024-01-15"))
        assert adapter._cash == before, "long-only → no borrow charge"

    def test_etb_default_rate(self):
        """1 short × $50,000 short value × 0.005/yr / 252 = ~$0.99/day"""
        adapter = _mk_adapter(
            holdings={"AAPL": SimpleNamespace(shares=-500, entry_price=100.0)},
            borrow_status={"AAPL": {"easy_to_borrow": True, "shortable": True}},
        )
        before = adapter._cash
        adapter._charge_daily_borrow(today_ts=pd.Timestamp("2024-01-15"))
        expected = 500 * 100.0 * 0.005 / 252.0
        actual = before - adapter._cash
        assert abs(actual - expected) < 1e-6, (
            f"ETB charge should be ~${expected:.4f}; got ${actual:.4f}"
        )

    def test_htb_default_rate(self):
        """HTB rate is 10× ETB rate"""
        adapter = _mk_adapter(
            holdings={"XYZ": SimpleNamespace(shares=-100, entry_price=200.0)},
            borrow_status={"XYZ": {"easy_to_borrow": False, "shortable": True}},
        )
        before = adapter._cash
        adapter._charge_daily_borrow(today_ts=pd.Timestamp("2024-01-15"))
        expected = 100 * 200.0 * 0.05 / 252.0
        actual = before - adapter._cash
        assert abs(actual - expected) < 1e-6

    def test_missing_ticker_fails_open_etb(self):
        """If ticker absent from borrow_status, treat as ETB (fail-open)"""
        adapter = _mk_adapter(
            holdings={"NEW": SimpleNamespace(shares=-100, entry_price=50.0)},
            borrow_status={},  # empty → fail-open
        )
        before = adapter._cash
        adapter._charge_daily_borrow(today_ts=pd.Timestamp("2024-01-15"))
        expected = 100 * 50.0 * 0.005 / 252.0  # ETB rate
        actual = before - adapter._cash
        assert abs(actual - expected) < 1e-6

    def test_config_override_rate(self):
        """borrow_rate_etb/htb overridable in config"""
        adapter = _mk_adapter(
            holdings={"AAPL": SimpleNamespace(shares=-100, entry_price=100.0)},
            borrow_status={"AAPL": {"easy_to_borrow": True, "shortable": True}},
            config={"long_short": {"borrow_rate_etb": 0.02}},  # 2% override
        )
        before = adapter._cash
        adapter._charge_daily_borrow(today_ts=pd.Timestamp("2024-01-15"))
        expected = 100 * 100.0 * 0.02 / 252.0
        actual = before - adapter._cash
        assert abs(actual - expected) < 1e-6

    def test_multiple_shorts_summed(self):
        """Multiple short positions → charges summed"""
        adapter = _mk_adapter(
            holdings={
                "AAPL": SimpleNamespace(shares=-100, entry_price=100.0),
                "MSFT": SimpleNamespace(shares=-50, entry_price=200.0),
                "GOOG": SimpleNamespace(shares=10, entry_price=300.0),  # LONG — no charge
            },
            borrow_status={
                "AAPL": {"easy_to_borrow": True, "shortable": True},
                "MSFT": {"easy_to_borrow": True, "shortable": True},
            },
        )
        before = adapter._cash
        adapter._charge_daily_borrow(today_ts=pd.Timestamp("2024-01-15"))
        expected = (100 * 100.0 + 50 * 200.0) * 0.005 / 252.0
        actual = before - adapter._cash
        assert abs(actual - expected) < 1e-6

    def test_nan_cash_skipped(self):
        """Cash NaN guard: don't poison further"""
        adapter = _mk_adapter(
            holdings={"AAPL": SimpleNamespace(shares=-100, entry_price=100.0)},
            cash=float("nan"),
            borrow_status={"AAPL": {"easy_to_borrow": True, "shortable": True}},
        )
        adapter._charge_daily_borrow(today_ts=pd.Timestamp("2024-01-15"))
        # Should not raise; cash stays NaN
        assert math.isnan(adapter._cash)

    def test_price_from_ohlcv_overrides_entry(self):
        """Mark-to-market uses today's close, not entry price"""
        idx = pd.DatetimeIndex(["2024-01-15"])
        df = pd.DataFrame({"close": [120.0]}, index=idx)
        adapter = _mk_adapter(
            holdings={"AAPL": SimpleNamespace(shares=-100, entry_price=100.0)},
            ohlcv={"AAPL": df},
            borrow_status={"AAPL": {"easy_to_borrow": True, "shortable": True}},
        )
        before = adapter._cash
        adapter._charge_daily_borrow(today_ts=pd.Timestamp("2024-01-15"))
        expected = 100 * 120.0 * 0.005 / 252.0  # uses 120, not 100
        actual = before - adapter._cash
        assert abs(actual - expected) < 1e-6
