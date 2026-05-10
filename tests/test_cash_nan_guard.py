"""Tests for the §5.13.11 cash NaN guard in SimAdapter._apply_sell.

Three scenarios:
- Normal sell: cash remains finite, equity not poisoned.
- Lots-exhausted (proceeds_basis=0) fallback path: avg-entry fallback
  applies, cash stays finite.
- Synthetic forced-NaN proceeds_basis: ValueError raised with diagnostic
  context (ticker, today, sell_shares, proceeds_basis, entry_price).

Per CLAUDE.md §5.13.3 — audit regression guard pins the invariant
"every cash mutation in _apply_sell produces a finite _cash, or raises".
"""
from __future__ import annotations

import datetime
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from adapters.sim import SimAdapter  # noqa: E402
from kernel.execution import T2CashQueue  # noqa: E402
from kernel.exits import HoldingState, TaxLot  # noqa: E402


def _make_adapter_with_holding(
    ticker: str = "AAPL",
    shares: float = 100.0,
    entry_price: float = 150.0,
    cash: float = 50_000.0,
) -> SimAdapter:
    """Construct a minimal SimAdapter with a single open position. Bypass
    __init__ to avoid heavy model loading."""
    today = datetime.date(2026, 5, 10)
    idx = pd.date_range("2026-01-01", periods=200, freq="B")
    ohlcv = {
        ticker: pd.DataFrame({"close": [200.0] * 200}, index=idx),
    }

    adapter = SimAdapter.__new__(SimAdapter)
    adapter._config = {
        "tax": {
            "short_term_rate": 0.50,
            "long_term_rate": 0.32,
            "long_term_threshold_days": 365,
        },
        "rotation": {"joint_actions": {"qp_tax_lot_method": "fifo"}},
    }
    adapter._ohlcv = ohlcv
    adapter._cash = cash
    hs = HoldingState(
        entry_price=entry_price,
        entry_date=datetime.date(2026, 1, 15),
        high_watermark=entry_price,
        shares=shares,
        lots=[TaxLot(shares=shares, price=entry_price,
                     date=datetime.date(2026, 1, 15))],
    )
    adapter._holdings = {ticker: hs}
    adapter._pos_shares = {ticker: shares}
    adapter._last_sell_date = {}
    adapter._last_sell_pls = {}
    adapter._last_stop_exit_date = {}
    adapter._trade_log = []
    # Agent A execution-model attributes — disable slippage / fees so this
    # test isolates the cash NaN guard from execution variability.
    adapter._exec_enabled = False
    adapter._exec_legacy = True
    adapter._total_fees = 0.0
    adapter._t2_queue = T2CashQueue(settlement_days=0)  # T+0 immediate credit
    return adapter


def _make_sell_sig(quantity=None, exit_type: str = "manual"):
    """Construct a minimal sell signal."""
    return SimpleNamespace(quantity=quantity, exit_type=exit_type)


class TestCashFiniteOnNormalSell:
    def test_normal_full_liquidation_keeps_cash_finite(self):
        adapter = _make_adapter_with_holding(
            ticker="AAPL", shares=100.0, entry_price=150.0, cash=50_000.0,
        )
        ctx = SimpleNamespace(
            prices={"AAPL": 200.0},
            config=adapter._config,
        )
        today_ts = pd.Timestamp("2026-05-08")  # business day
        adapter._apply_sell("AAPL", _make_sell_sig(), today_ts, ctx)

        assert math.isfinite(adapter._cash)
        # Revenue = 100*200 = 20_000; gain = 100*(200-150) = 5_000
        # hold_days = (2026-05-08 - 2026-01-15) ≈ 113d → short-term @ 0.5
        # tax = 5_000 * 0.5 = 2_500
        # delta_cash = 20_000 - 2_500 = 17_500
        # final cash = 50_000 + 17_500 = 67_500
        assert adapter._cash == pytest.approx(67_500.0, rel=1e-6)


class TestCashFiniteOnLotsExhaustedFallback:
    def test_no_lots_legacy_path_keeps_cash_finite(self):
        """When holding has no lots (legacy state), apply_sell_lots returns
        proceeds_basis = 0. The fallback (entry_price * sell_shares)
        provides a sane basis so gross_pnl + cash are finite."""
        adapter = _make_adapter_with_holding(
            ticker="MSFT", shares=50.0, entry_price=300.0, cash=20_000.0,
        )
        # Strip the lots → simulates pre-migration state.
        adapter._holdings["MSFT"].lots = []

        ctx = SimpleNamespace(
            prices={"MSFT": 350.0},
            config=adapter._config,
        )
        today_ts = pd.Timestamp("2026-05-08")
        adapter._apply_sell("MSFT", _make_sell_sig(), today_ts, ctx)

        assert math.isfinite(adapter._cash)
        # Legacy path: gross_pnl = 50*(350-300) = 2_500;
        # tax = 2_500 * 0.5 = 1_250; delta = 50*350 - 1_250 = 16_250
        # final = 20_000 + 16_250 = 36_250
        assert adapter._cash == pytest.approx(36_250.0, rel=1e-6)


class TestCashNanGuardRaisesWithDiagnostic:
    """§5.13.3 / §5.13.11 audit regression guard: forcing apply_sell_lots
    to return NaN proceeds_basis must trigger the cash guard ValueError
    with a diagnostic message naming ticker/today/sell_shares.

    Pre-fix: NaN proceeds_basis silently → NaN gross_pnl → NaN tax
    → self._cash += NaN → equity curve permanently poisoned with no error.
    """

    def test_nan_proceeds_basis_raises_with_full_context(self):
        adapter = _make_adapter_with_holding(
            ticker="NVDA", shares=80.0, entry_price=400.0, cash=30_000.0,
        )
        # Sabotage entry_price so the fallback ALSO produces NaN: this
        # ensures the NaN propagates through gross_pnl and the final
        # cash-isfinite guard catches it.
        adapter._holdings["NVDA"].entry_price = float("nan")
        # Empty lots so the fallback path is taken.
        adapter._holdings["NVDA"].lots = []

        ctx = SimpleNamespace(
            prices={"NVDA": float("nan")},  # also poison price → delta NaN
            config=adapter._config,
        )
        today_ts = pd.Timestamp("2026-05-08")

        # Force apply_sell_lots to return (NaN, 0) so even the fallback
        # cannot rescue — the guard MUST fire.
        with patch("kernel.exits.apply_sell_lots",
                   return_value=(float("nan"), 0.0)):
            with pytest.raises(ValueError) as exc_info:
                adapter._apply_sell("NVDA", _make_sell_sig(), today_ts, ctx)

        msg = str(exc_info.value)
        # Audit regression guard: every field must surface in the diagnostic.
        assert "NVDA" in msg
        assert "sell_shares" in msg
        assert "proceeds_basis" in msg
        assert "entry_price" in msg
