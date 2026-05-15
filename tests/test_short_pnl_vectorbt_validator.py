"""Cross-validate SimAdapter short P&L against vectorbt (mature lib).

Per §5.12 (canonical references): every novel cash-flow path must be
validated against a mature library before scaling. vectorbt's
Portfolio.from_orders is the canonical short-aware backtest engine.

These tests pin two invariants:
  1. SimAdapter pre-tax NAV ≡ vectorbt pre-tax NAV (mechanical equivalence)
  2. SimAdapter applies §1233 ST tax on gains only (delta = expected_tax)

Bug surface: 2026-05-14 self-audit found 5 bugs in the short cash-flow
path. Without vectorbt cross-check, equity-curve correctness rested on
hand-rolled arithmetic alone. This test ensures any future regression
fails here, not silently in production.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

vbt = pytest.importorskip("vectorbt", reason="vectorbt not installed")


def _vbt_short_cover_nav(
    init_cash: float, shares: int, entry_px: float, cover_px: float, n_days: int = 30,
) -> float:
    """Run a single short-open → cover trade through vectorbt and return final NAV."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    prices = pd.Series(np.linspace(entry_px, cover_px, n_days), index=dates, name="AAPL")
    size = pd.Series(0.0, index=dates, name="AAPL")
    size.iloc[0] = -float(shares)
    size.iloc[-1] = +float(shares)
    pf = vbt.Portfolio.from_orders(
        close=prices, size=size, size_type="amount", direction="both",
        init_cash=init_cash, fees=0.0, slippage=0.0, freq="D",
    )
    return float(pf.value().iloc[-1])


def _sim_short_cover_cash(
    init_cash: float, shares: int, entry_px: float, cover_px: float, st_tax: float,
) -> float:
    """Run a single short-open → cover trade through SimAdapter and return final cash."""
    from adapters.sim import SimAdapter
    adapter = SimAdapter.__new__(SimAdapter)
    adapter._cash = init_cash
    adapter._total_fees = 0.0
    adapter._exec_enabled = False
    adapter._trade_log = []
    adapter._holdings = {}
    adapter._pos_shares = {}
    adapter._borrow_status_cache = {"AAPL": {"easy_to_borrow": True, "shortable": True}}
    adapter._strategy_config = {}
    open_ts = pd.Timestamp("2024-05-01")
    cover_ts = pd.Timestamp("2024-06-15")

    adapter._ohlcv = {"AAPL": pd.DataFrame(
        {"close": [entry_px]}, index=pd.DatetimeIndex([open_ts])
    )}
    sig = SimpleNamespace(quantity=shares, exit_type="qp_short_open")
    adapter._apply_short_open("AAPL", sig, open_ts, None)

    adapter._ohlcv = {"AAPL": pd.DataFrame(
        {"close": [cover_px]}, index=pd.DatetimeIndex([cover_ts])
    )}
    ctx = SimpleNamespace(config={"tax": {
        "short_term_rate": st_tax,
        "long_term_rate": 0.32,
        "long_term_threshold_days": 365,
    }})
    adapter._apply_buy(
        {"ticker": "AAPL", "shares": shares, "price": cover_px}, cover_ts, ctx,
    )
    return float(adapter._cash)


class TestSimAdapterMatchesVectorbtOnShorts:
    """SimAdapter pre-tax cash flow on shorts must match vectorbt exactly.

    Tax is layered AFTER the cash-flow check by subtracting expected_tax.
    """

    def test_short_profit_30day_walk(self):
        """Short 100 @ $150, cover @ $140 (linear walk). Profit = $1000."""
        init, shares, entry, cover, st = 100_000.0, 100, 150.0, 140.0, 0.50
        vbt_nav = _vbt_short_cover_nav(init, shares, entry, cover)
        sim_cash = _sim_short_cover_cash(init, shares, entry, cover, st)
        gross_pnl = (entry - cover) * shares
        tax = gross_pnl * st
        assert abs(vbt_nav - (init + gross_pnl)) < 1.0, \
            f"vectorbt pre-tax mismatch: ${vbt_nav} vs expected ${init + gross_pnl}"
        assert abs(sim_cash - (vbt_nav - tax)) < 1.0, \
            f"SimAdapter (${sim_cash}) != vectorbt (${vbt_nav}) - tax (${tax})"

    def test_short_loss_30day_walk(self):
        """Short 100 @ $150, cover @ $170. Loss = -$2000, no tax."""
        init, shares, entry, cover, st = 100_000.0, 100, 150.0, 170.0, 0.50
        vbt_nav = _vbt_short_cover_nav(init, shares, entry, cover)
        sim_cash = _sim_short_cover_cash(init, shares, entry, cover, st)
        gross_pnl = (entry - cover) * shares  # -2000
        assert gross_pnl < 0, "scenario must be a loss"
        assert abs(vbt_nav - (init + gross_pnl)) < 1.0
        # Loss: NO tax adjustment
        assert abs(sim_cash - vbt_nav) < 1.0, \
            f"loss path: SimAdapter (${sim_cash}) must equal vectorbt NAV (${vbt_nav})"

    def test_short_flat_30day_walk(self):
        """Short 100 @ $150, cover @ $150. P&L = 0, no tax."""
        init, shares, entry, cover, st = 100_000.0, 100, 150.0, 150.0, 0.50
        vbt_nav = _vbt_short_cover_nav(init, shares, entry, cover)
        sim_cash = _sim_short_cover_cash(init, shares, entry, cover, st)
        assert abs(vbt_nav - init) < 1.0
        assert abs(sim_cash - init) < 1.0

    def test_large_short_profit_no_overdraw(self):
        """Big short profit: $5K → cash + $5K - $2.5K tax = + $2.5K net."""
        init, shares, entry, cover, st = 100_000.0, 500, 100.0, 90.0, 0.50
        vbt_nav = _vbt_short_cover_nav(init, shares, entry, cover)
        sim_cash = _sim_short_cover_cash(init, shares, entry, cover, st)
        gross_pnl = (entry - cover) * shares  # 5000
        tax = gross_pnl * st  # 2500
        assert abs(sim_cash - (vbt_nav - tax)) < 1.0
        assert sim_cash > init, "should be net positive after tax"
