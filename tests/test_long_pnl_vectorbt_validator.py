"""Cross-validate SimAdapter LONG buy/sell P&L against vectorbt.

Companion to test_short_pnl_vectorbt_validator. Covers:
  1. Buy 100 @ $100, sell @ $120 after 30 days → ST gain, ST tax
  2. Buy 100 @ $100, sell @ $120 after 400 days → LT gain, LT tax
  3. Buy 100 @ $100, sell @ $80 → loss, no tax
  4. Buy 100 @ $100, partial sell 50 @ $120 → ST gain on 50 only

Each test asserts:
  - vectorbt pre-tax NAV matches expected algebraic value
  - SimAdapter cash == vectorbt NAV - expected_tax (cost basis: FIFO lots)

Pinning §5.12 invariant: mature-lib cross-check before scaling experiments.
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


def _vbt_long_nav(init_cash: float, shares: int, buy_px: float,
                  sell_px: float, sell_shares: int | None = None,
                  n_days: int = 30) -> float:
    """Run buy → (partial) sell through vectorbt; return final NAV (pre-tax)."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    prices = pd.Series(np.linspace(buy_px, sell_px, n_days), index=dates, name="AAPL")
    size = pd.Series(0.0, index=dates, name="AAPL")
    size.iloc[0] = +float(shares)
    size.iloc[-1] = -float(sell_shares or shares)
    pf = vbt.Portfolio.from_orders(
        close=prices, size=size, size_type="amount", direction="both",
        init_cash=init_cash, fees=0.0, slippage=0.0, freq="D",
    )
    return float(pf.value().iloc[-1])


def _sim_long_cash(
    init_cash: float, shares: int, buy_px: float, sell_px: float,
    hold_days: int, st_tax: float = 0.50, lt_tax: float = 0.20,
    sell_shares: int | None = None,
    tax_cash_debit_mode: str = "event_level",
    return_adapter: bool = False,
) -> float:
    """Run buy → (partial) sell through SimAdapter; return final cash."""
    from adapters.sim import SimAdapter
    adapter = SimAdapter.__new__(SimAdapter)
    adapter._cash = init_cash
    adapter._total_fees = 0.0
    adapter._exec_enabled = False
    adapter._trade_log = []
    adapter._holdings = {}
    adapter._pos_shares = {}
    adapter._strategy_config = {}
    adapter._last_sell_date = {}
    adapter._last_sell_pls = {}
    adapter._t2_queue = SimpleNamespace(
        settlement_days=0,
        add_pending=lambda *a, **k: None,
        drain=lambda *a, **k: 0.0,
        pending_total=lambda: 0.0,
    )
    buy_ts = pd.Timestamp("2024-01-01")
    sell_ts = buy_ts + pd.Timedelta(days=hold_days)

    adapter._ohlcv = {"AAPL": pd.DataFrame(
        {"close": [buy_px]}, index=pd.DatetimeIndex([buy_ts])
    )}
    ctx = SimpleNamespace(config={
        "tax": {
            "short_term_rate": st_tax,
            "long_term_rate": lt_tax,
            "long_term_threshold_days": 365,
            "cash_debit_mode": tax_cash_debit_mode,
        },
        "rotation": {"joint_actions": {"qp_tax_lot_method": "fifo"}},
    })
    adapter._apply_buy(
        {"ticker": "AAPL", "shares": shares, "price": buy_px}, buy_ts, ctx,
    )

    adapter._ohlcv = {"AAPL": pd.DataFrame(
        {"close": [sell_px]}, index=pd.DatetimeIndex([sell_ts])
    )}
    sig = SimpleNamespace(
        quantity=sell_shares,  # None = full liquidation
        exit_type="qp_close",
    )
    ctx.prices = {"AAPL": sell_px}
    adapter._apply_sell("AAPL", sig, sell_ts, ctx)
    if return_adapter:
        return adapter
    return float(adapter._cash)


class TestSimAdapterLongMatchesVectorbt:
    """SimAdapter long buy/sell cash flow must match vectorbt; tax overlay correct."""

    def test_short_term_gain(self):
        """Hold 30 days, sell at +20% gain → ST tax."""
        init, shares, buy, sell, hold = 100_000.0, 100, 100.0, 120.0, 30
        st, lt = 0.50, 0.20
        vbt_nav = _vbt_long_nav(init, shares, buy, sell, n_days=hold)
        sim_cash = _sim_long_cash(init, shares, buy, sell, hold, st, lt)
        gross_pnl = (sell - buy) * shares  # 2000
        expected_tax = gross_pnl * st  # ST: 1000
        assert abs(vbt_nav - (init + gross_pnl)) < 1.0
        assert abs(sim_cash - (vbt_nav - expected_tax)) < 1.0, \
            f"ST gain: SimAdapter ${sim_cash} != vectorbt ${vbt_nav} - tax ${expected_tax}"

    def test_long_term_gain(self):
        """Hold 400 days, sell at +20% gain → LT tax (lower rate)."""
        init, shares, buy, sell, hold = 100_000.0, 100, 100.0, 120.0, 400
        st, lt = 0.50, 0.20
        vbt_nav = _vbt_long_nav(init, shares, buy, sell, n_days=hold)
        sim_cash = _sim_long_cash(init, shares, buy, sell, hold, st, lt)
        gross_pnl = (sell - buy) * shares  # 2000
        expected_tax = gross_pnl * lt  # LT: 400
        assert abs(vbt_nav - (init + gross_pnl)) < 1.0
        assert abs(sim_cash - (vbt_nav - expected_tax)) < 1.0, \
            f"LT gain: SimAdapter ${sim_cash} != vectorbt ${vbt_nav} - LT tax ${expected_tax}. " \
            f"If sim_cash matches ST tax instead, §1233-style ST/LT switching broken."

    def test_loss_no_tax(self):
        """Hold 30 days, sell at -20% loss → no tax."""
        init, shares, buy, sell, hold = 100_000.0, 100, 100.0, 80.0, 30
        vbt_nav = _vbt_long_nav(init, shares, buy, sell, n_days=hold)
        sim_cash = _sim_long_cash(init, shares, buy, sell, hold)
        assert abs(vbt_nav - (init + (sell - buy) * shares)) < 1.0
        assert abs(sim_cash - vbt_nav) < 1.0, \
            f"Loss path: SimAdapter ${sim_cash} should equal vectorbt ${vbt_nav} (no tax on losses)"

    def test_partial_sell_st_gain(self):
        """Buy 100, sell 50 at gain → tax only on disposed 50."""
        init, shares, buy, sell, hold = 100_000.0, 100, 100.0, 120.0, 30
        sell_qty = 50
        st = 0.50
        vbt_nav = _vbt_long_nav(init, shares, buy, sell, sell_shares=sell_qty, n_days=hold)
        sim_cash = _sim_long_cash(init, shares, buy, sell, hold, st_tax=st,
                                   sell_shares=sell_qty)
        gross_pnl_disposed = (sell - buy) * sell_qty  # 1000
        expected_tax = gross_pnl_disposed * st  # 500
        # vbt_nav includes mark-to-market on remaining 50 shares at terminal price
        # That equals init - shares*buy + sell_qty*sell + (shares - sell_qty)*sell
        # = init - shares*buy + shares*sell = init + shares*(sell - buy) = vbt's NAV
        # So vbt NAV uses full unrealized; SimAdapter cash is only REALIZED.
        # We compare: SimAdapter cash + remaining_shares × buy_basis vs vbt_NAV - tax
        # OR: SimAdapter realized_cash = vbt_realized_cash - tax
        # vbt_realized_cash = init - 100*100 + 50*120 = 96_000
        # sim_realized_cash = vbt_realized - tax = 96_000 - 500 = 95_500
        vbt_realized = init - shares * buy + sell_qty * sell  # 96_000
        expected_sim_cash = vbt_realized - expected_tax
        assert abs(sim_cash - expected_sim_cash) < 1.0, \
            f"Partial sell: SimAdapter realized ${sim_cash} != expected ${expected_sim_cash}"

    def test_reporting_only_tax_keeps_trade_cash_pre_tax(self):
        """Live-like tax mode: estimate tax for reporting, but do not debit cash."""
        init, shares, buy, sell, hold = 100_000.0, 100, 100.0, 120.0, 30
        st, lt = 0.50, 0.20
        vbt_nav = _vbt_long_nav(init, shares, buy, sell, n_days=hold)
        adapter = _sim_long_cash(
            init, shares, buy, sell, hold, st, lt,
            tax_cash_debit_mode="reporting_only",
            return_adapter=True,
        )
        gross_pnl = (sell - buy) * shares
        expected_tax = gross_pnl * st
        assert abs(adapter._cash - vbt_nav) < 1.0
        sell_rows = [t for t in adapter._trade_log if t.get("action") == "sell"]
        assert len(sell_rows) == 1
        assert sell_rows[0]["tax"] == pytest.approx(expected_tax)
        assert sell_rows[0]["tax_cash_debited"] == pytest.approx(0.0)
        assert sell_rows[0]["tax_cash_debit_mode"] == "reporting_only"

    def test_unknown_tax_cash_mode_fails_closed(self):
        """A typo must not silently switch reporting-only runs to cash-debit tax."""
        from adapters.sim import _tax_cash_debit_mode

        with pytest.raises(ValueError, match="Unknown tax.cash_debit_mode"):
            _tax_cash_debit_mode({"tax": {"cash_debit_mode": "reportng_only"}})

    def test_legacy_event_cash_debit_alias_is_explicit(self):
        from adapters.sim import _tax_cash_debit_mode

        assert (
            _tax_cash_debit_mode({"tax": {"cash_debit_mode": "event_cash_debit"}})
            == "event_level"
        )
