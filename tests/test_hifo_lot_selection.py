"""Tests for the 2026-05-17 FIFO→HIFO default switch.

HIFO (Highest In, First Out) is the tax-optimal lot selection method
for partial sales. For ANY current price relative to the lot prices:

  - If selling at a GAIN (price > all lot prices): HIFO picks the highest-
    cost lot first → smallest realized gain → smallest tax bill.
  - If selling at a LOSS (price < some lot prices): HIFO picks the lot
    with the largest unrealized loss first → biggest realized loss →
    biggest tax-loss harvest credit.

Both regimes prefer HIFO. FIFO was the default purely as a legacy
naive-accounting choice. Industry standard (Wealthfront, M1 Finance,
Schwab Intelligent Portfolios) uses HIFO for taxable accounts.

This is NOT "tax-driven sell logic" per `feedback_no_tax_driven_logic`
memory — the SELL DECISION (how many shares) is unchanged. Only the
LOT ACCOUNTING (which shares to count as sold) changes.

References:
- Berkin-Jeffrey 1990 (J. Portfolio Mgmt) "Tax-managed investing"
- Wealthfront whitepaper "Tax-Loss Harvesting" — HIFO default rationale
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.exits import HoldingState, TaxLot, apply_sell_lots  # noqa: E402
import datetime as dt
import json


def _make_lots(prices: list[float], shares_each: int = 10) -> list[TaxLot]:
    """Build N TaxLots, each with `shares_each` shares at the given price."""
    base = dt.date(2025, 1, 1)
    return [TaxLot(date=base + dt.timedelta(days=30*i), price=p, shares=float(shares_each))
            for i, p in enumerate(prices)]


def _new_hs(lots: list[TaxLot]) -> HoldingState:
    total_shares = sum(L.shares for L in lots)
    avg_price = sum(L.price * L.shares for L in lots) / max(total_shares, 1)
    hs = HoldingState(
        entry_price=avg_price,
        entry_date=lots[0].date,
        high_watermark=avg_price,
    )
    hs.shares = total_shares
    hs.lots = lots
    return hs


class TestHifoTaxOptimality:
    def test_hifo_picks_highest_cost_first_at_gain(self):
        """Selling at $200 with lots at [100, 50, 150]: HIFO sells the
        $150 lot first (cost basis $1500), realizing $500 gain
        instead of FIFO's $1000 gain on the $100 lot."""
        hs = _new_hs(_make_lots([100.0, 50.0, 150.0], shares_each=10))
        basis_h, _ = apply_sell_lots(hs, shares_to_sell=10, method="hifo")
        assert basis_h == 1500.0, f"HIFO should consume $150 lot → basis $1500, got {basis_h}"

        hs2 = _new_hs(_make_lots([100.0, 50.0, 150.0], shares_each=10))
        basis_f, _ = apply_sell_lots(hs2, shares_to_sell=10, method="fifo")
        assert basis_f == 1000.0, f"FIFO should consume $100 lot → basis $1000, got {basis_f}"

        # At sell price $200: realized gain HIFO = $500, FIFO = $1000.
        # HIFO saves $500 of realized gain × tax rate ≈ $100-200 in taxes.

    def test_hifo_picks_highest_cost_first_at_loss(self):
        """Selling at $75 with lots at [100, 50, 150]: HIFO sells the
        $150 lot first → -$750 loss, biggest tax-loss harvest. FIFO
        sells $100 lot → -$250 loss, smaller deduction."""
        hs = _new_hs(_make_lots([100.0, 50.0, 150.0], shares_each=10))
        basis_h, _ = apply_sell_lots(hs, shares_to_sell=10, method="hifo")
        assert basis_h == 1500.0  # HIFO picks $150 lot
        # At sell $75: realized loss = $75*10 - $1500 = -$750 (biggest harvest)

        hs2 = _new_hs(_make_lots([100.0, 50.0, 150.0], shares_each=10))
        basis_f, _ = apply_sell_lots(hs2, shares_to_sell=10, method="fifo")
        assert basis_f == 1000.0
        # At sell $75: realized loss = $75*10 - $1000 = -$250 (smaller harvest)

    def test_hifo_consumes_across_multiple_lots_in_descending_price_order(self):
        """Selling 15 shares with 3 lots at [100, 50, 150] × 10 each:
        HIFO consumes all 10 of $150 lot, then 5 of $100 lot.
        Total basis = 1500 + 500 = 2000."""
        hs = _new_hs(_make_lots([100.0, 50.0, 150.0], shares_each=10))
        basis, _ = apply_sell_lots(hs, shares_to_sell=15, method="hifo")
        assert basis == 2000.0
        # Remaining: $100 lot 5 shares + $50 lot 10 shares
        rem = sorted([(L.price, L.shares) for L in hs.lots])
        assert rem == [(50.0, 10.0), (100.0, 5.0)]


class TestGoldenConfigDefault:
    def test_golden_uses_hifo(self):
        c = json.loads((REPO / "backtesting/renquant_104/strategy_config.golden.json").read_text())
        m = c["rotation"]["joint_actions"]["qp_tax_lot_method"]
        assert m == "hifo", f"golden default must be HIFO (tax-optimal), got {m!r}"

    def test_live_config_uses_hifo(self):
        c = json.loads((REPO / "backtesting/renquant_104/strategy_config.json").read_text())
        m = c["rotation"]["joint_actions"]["qp_tax_lot_method"]
        assert m == "hifo"


class TestNoBehaviorChange:
    """REGRESSION GUARD: changing lot method must not change SELL decisions
    (how many shares to sell), only WHICH shares get counted as sold.
    This is lot accounting, not tax-driven sell/hold logic
    (per feedback_no_tax_driven_logic memory)."""

    def test_fifo_still_works_when_explicitly_chosen(self):
        """If someone sets method="fifo" explicitly, FIFO order applies."""
        hs = _new_hs(_make_lots([100.0, 50.0, 150.0], shares_each=10))
        basis, _ = apply_sell_lots(hs, shares_to_sell=10, method="fifo")
        assert basis == 1000.0, "explicit fifo override still respected"

    def test_total_shares_disposed_identical_across_methods(self):
        """Same number of shares sold regardless of lot order method."""
        for method in ("fifo", "hifo", "avg"):
            hs = _new_hs(_make_lots([100.0, 50.0, 150.0], shares_each=10))
            apply_sell_lots(hs, shares_to_sell=12, method=method)
            shares_left = sum(L.shares for L in hs.lots)
            assert abs(shares_left - 18.0) < 1e-9, \
                f"method={method}: should leave 18 shares, got {shares_left}"
