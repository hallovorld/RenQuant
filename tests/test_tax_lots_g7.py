"""G7 — Tax-lot tracking (FIFO / HIFO) tests.

Goal of this suite (2026-05-04, G7 deliverable):
  1. The HoldingState ``lots`` schema works: instantiate, round-trip, derive
     totals + weighted-average entry price.
  2. ``ensure_lots()`` migrates a legacy single-cost-basis holding into a
     1-element TaxLot list without changing observable state.
  3. Buy path appends a new lot; multi-buy positions accumulate lots.
  4. Sell path consumes lots according to FIFO (default) or HIFO.
  5. The QP tax-cost task uses the lot list when present (HIFO marginal),
     producing different costs than the legacy single-entry path on
     multi-lot holdings.
  6. The ``qp_tax_lot_method = 'avg'`` kill-switch falls back to legacy.

These regressions would have caught the pre-G7 single-entry assumption
(weighted average compresses old high-cost lots into the new average,
underestimating tax drag on partial sells).
"""
from __future__ import annotations

import datetime
import json
import sys
from dataclasses import asdict, field, dataclass
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.exits import (   # noqa: E402
    HoldingState, TaxLot, ensure_lots,
    apply_buy_lot, apply_sell_lots,
)


# ───────────────────────────────────────────────────────────────────────
# 1. TaxLot dataclass round-trip
# ───────────────────────────────────────────────────────────────────────

class TestTaxLotDataclass:
    def test_taxlot_dataclass_round_trips(self):
        lot = TaxLot(shares=10.0, price=100.0, date=datetime.date(2025, 1, 5))
        d = asdict(lot)
        assert d["shares"] == 10.0
        assert d["price"] == 100.0
        # date round-trip via isoformat
        s = json.dumps({**d, "date": d["date"].isoformat()})
        recovered = json.loads(s)
        lot2 = TaxLot(
            shares=recovered["shares"],
            price=recovered["price"],
            date=datetime.date.fromisoformat(recovered["date"]),
        )
        assert lot2 == lot


# ───────────────────────────────────────────────────────────────────────
# 2. ensure_lots migration
# ───────────────────────────────────────────────────────────────────────

class TestEnsureLots:
    def test_ensure_lots_migrates_legacy_holding(self):
        hs = HoldingState(
            entry_price=100.0,
            entry_date=datetime.date(2025, 6, 1),
            high_watermark=100.0,
            shares=20.0,
        )
        assert hs.lots == []
        ensure_lots(hs)
        assert len(hs.lots) == 1
        assert hs.lots[0].shares == 20.0
        assert hs.lots[0].price == 100.0
        assert hs.lots[0].date == datetime.date(2025, 6, 1)
        # Idempotent
        ensure_lots(hs)
        assert len(hs.lots) == 1

    def test_ensure_lots_skip_when_already_populated(self):
        hs = HoldingState(
            entry_price=100.0,
            entry_date=datetime.date(2025, 6, 1),
            high_watermark=100.0,
            shares=20.0,
        )
        hs.lots = [TaxLot(shares=5.0, price=80.0, date=datetime.date(2024, 1, 1))]
        ensure_lots(hs)
        # Should NOT prepend a second lot from legacy fields.
        assert len(hs.lots) == 1
        assert hs.lots[0].price == 80.0

    def test_ensure_lots_skip_when_no_real_position(self):
        hs = HoldingState(
            entry_price=0.0,
            entry_date=datetime.date(2025, 6, 1),
            high_watermark=0.0,
            shares=0.0,
        )
        ensure_lots(hs)
        assert hs.lots == []


# ───────────────────────────────────────────────────────────────────────
# 3. Buy mutation — appends a new TaxLot
# ───────────────────────────────────────────────────────────────────────

class TestBuyAppendsLot:
    def test_buy_appends_lot(self):
        hs = HoldingState(
            entry_price=0.0, entry_date=None, high_watermark=0.0, shares=0.0,
        )
        # First buy
        apply_buy_lot(hs, shares=10.0, price=100.0,
                       date=datetime.date(2025, 1, 1))
        assert len(hs.lots) == 1
        assert hs.entry_price == 100.0
        assert hs.entry_date == datetime.date(2025, 1, 1)
        # Second buy at higher price
        apply_buy_lot(hs, shares=10.0, price=120.0,
                       date=datetime.date(2025, 6, 1))
        assert len(hs.lots) == 2
        # weighted avg = (100*10 + 120*10) / 20 = 110
        assert abs(hs.weighted_avg_entry_price() - 110.0) < 1e-9
        assert hs.entry_price == 110.0
        # entry_date should remain at FIRST acquisition for tenure-based rules
        assert hs.entry_date == datetime.date(2025, 1, 1)
        # Total shares
        assert hs.total_shares() == 20.0


# ───────────────────────────────────────────────────────────────────────
# 4. Sell — FIFO (default) and HIFO
# ───────────────────────────────────────────────────────────────────────

class TestSellLotConsumption:
    def _build_three_lots(self):
        hs = HoldingState(
            entry_price=0.0, entry_date=None, high_watermark=0.0, shares=0.0,
        )
        apply_buy_lot(hs, 10, 100.0, datetime.date(2025, 1, 1))   # oldest
        apply_buy_lot(hs, 10, 120.0, datetime.date(2025, 3, 1))   # mid
        apply_buy_lot(hs, 10, 110.0, datetime.date(2025, 6, 1))   # newest
        return hs

    def test_sell_fifo_consumes_oldest(self):
        hs = self._build_three_lots()
        # Sell 15 shares: should fully consume lot1 (10 sh) + half of lot2 (5 sh).
        basis, _ = apply_sell_lots(hs, shares_to_sell=15.0, method="fifo")
        # Cost basis disposed = 10*100 + 5*120 = 1600
        assert abs(basis - 1600.0) < 1e-9
        # After consumption: 5 shares of $120 lot + 10 shares of $110 lot remain.
        assert len(hs.lots) == 2
        # First lot remaining is the half-consumed $120 lot (FIFO order
        # preserved — original index 1 still leads index 2).
        assert abs(hs.lots[0].price - 120.0) < 1e-9
        assert abs(hs.lots[0].shares - 5.0) < 1e-9
        assert abs(hs.lots[1].price - 110.0) < 1e-9
        assert abs(hs.lots[1].shares - 10.0) < 1e-9

    def test_sell_hifo_consumes_highest_cost(self):
        hs = self._build_three_lots()
        # Sell 15 shares HIFO: should fully consume the $120 lot (10 sh)
        # then half of the next-highest $110 lot (5 sh).
        basis, _ = apply_sell_lots(hs, shares_to_sell=15.0, method="hifo")
        # Cost basis disposed = 10*120 + 5*110 = 1750
        assert abs(basis - 1750.0) < 1e-9
        # Remaining: 5 sh @ $110 + 10 sh @ $100
        prices_left = sorted(L.price for L in hs.lots)
        assert prices_left == pytest.approx([100.0, 110.0])

    def test_sell_avg_method_proportional_trim(self):
        hs = self._build_three_lots()
        # avg trims proportionally — 15 of 30 shares = 50%
        apply_sell_lots(hs, shares_to_sell=15.0, method="avg")
        for L in hs.lots:
            assert abs(L.shares - 5.0) < 1e-9


# ───────────────────────────────────────────────────────────────────────
# 5. QP tax cost — lot path differs from legacy single-entry path
# ───────────────────────────────────────────────────────────────────────

from kernel.portfolio_qp.tasks import (   # noqa: E402
    ComputeBrownSmithTaxCostTask,
    _bridge_rate,
    _per_asset_tax,
    _per_asset_tax_lots,
)


@dataclass
class _Ctx:
    config: dict = field(default_factory=dict)
    holdings: dict = field(default_factory=dict)
    prices: dict = field(default_factory=dict)
    portfolio_value: float = 100_000.0
    today: datetime.date = datetime.date(2026, 5, 4)
    ytd_realized_gain_dollar: float = 0.0
    _qp_tickers: list = field(default_factory=list)
    _qp_w_current: object = None


def _qp_tax_cfg(method: str = "fifo", tax_aware: bool = True) -> dict:
    return {
        "rotation": {"joint_actions": {
            "enabled": True,
            "qp_tax_aware": tax_aware,
            "qp_tax_rate_st": 0.30,
            "qp_tax_rate_lt": 0.15,
            "qp_lt_threshold_days": 365,
            "qp_lt_bridge_window_days": 30,
            "qp_tax_lot_method": method,
        }},
    }


class TestQPTaxCostUsesLots:
    def _multi_lot_holding(self):
        """Three lots at different cost bases. Current price 130.0 →
        every lot is a winner, but $100 lot has biggest gain ($30/sh @ 30%
        ST = $9/sh tax) vs $120 lot ($10/sh × 30% = $3/sh)."""
        today = datetime.date(2026, 5, 4)
        hs = HoldingState(
            entry_price=0.0, entry_date=None, high_watermark=130.0, shares=0.0,
        )
        # All lots well within ST window (held <365d)
        apply_buy_lot(hs, 10, 100.0, datetime.date(2026, 4, 1))
        apply_buy_lot(hs, 10, 120.0, datetime.date(2026, 4, 10))
        apply_buy_lot(hs, 10, 110.0, datetime.date(2026, 4, 20))
        return hs, today

    def test_qp_tax_aware_omitted_defaults_to_reporting_only(self):
        """Tax must not enter QP decisions unless qp_tax_aware is explicit.

        User mandate: tax can be reported, but should not silently suppress
        sells or create loss-harvest trades. Missing config therefore means
        OFF, not legacy-on.
        """
        hs, today = self._multi_lot_holding()
        ctx = _Ctx(
            config={"rotation": {"joint_actions": {"enabled": True}}},
            holdings={"ABC": hs},
            prices={"ABC": 130.0},
            portfolio_value=13_000.0,
            today=today,
            _qp_tickers=["ABC"],
            _qp_w_current=np.array([0.30]),
        )
        ComputeBrownSmithTaxCostTask().run(ctx)
        assert ctx._qp_tax_cost.tolist() == [0.0]

    def test_qp_tax_cost_uses_hifo_when_lots_present(self):
        """Multi-lot holding under HIFO marginal vs legacy single-entry
        path should produce DIFFERENT costs.

        legacy path: avg entry = (100+120+110)/3 = 110, gain% = 20/110.
        HIFO path: marginal sell touches $120 lot first → smaller gain.
        """
        hs, today = self._multi_lot_holding()
        price = 130.0
        w_i = 0.30   # 30% of NAV in this position (matches 30 sh × 130 / 13000)
        nav = 30 * price / w_i   # 13000
        # Legacy (avg) — uses entry_price
        # When called with multi-lot hs, _per_asset_tax reads entry_price
        # which is the weighted average. Compute it explicitly:
        hs.entry_price = hs.weighted_avg_entry_price()
        hs.entry_date = datetime.date(2026, 4, 1)
        cost_legacy, _ = _per_asset_tax(
            hs, price, w_i, nav, today,
            0.30, 0.15, 365, 30, 0.0,
        )
        # Lot-aware HIFO: marginal on $120 lot has lower gain%.
        cost_hifo, _ = _per_asset_tax_lots(
            hs, price, w_i, nav, today,
            0.30, 0.15, 365, 30, 0.0, "hifo",
        )
        # Both positive (winners), but HIFO marginal cost should be
        # STRICTLY LESS than the legacy avg path — HIFO disposes of the
        # smallest-gain lot first.
        assert cost_legacy > 0
        assert cost_hifo > 0
        assert cost_hifo < cost_legacy, (
            f"HIFO (cost={cost_hifo:.6f}) should be < legacy "
            f"(cost={cost_legacy:.6f}) when disposing only the $120 lot "
            f"vs the avg-entry $110 basis."
        )

    def test_qp_tax_cost_fifo_picks_oldest_first(self):
        """For a 1-NAV-fraction MARGINAL sell that consumes only ONE lot,
        FIFO touches the $100 lot ($30 gain → highest tax) and HIFO
        touches the $120 lot ($10 gain → lowest tax). Costs must differ.

        Use w_i small enough that target_shares < single lot size (10).
        position is 30 sh × $130 = $3900 in a $100k portfolio → w_max=0.039.
        Pick w_i = 0.01 → target = 100 × 0.01 / 130 ≈ 7.69 sh < 10.
        """
        hs, today = self._multi_lot_holding()
        price = 130.0
        nav = 100_000.0
        w_i = 0.01    # marginal — touches only one lot
        cost_fifo, _ = _per_asset_tax_lots(
            hs, price, w_i, nav, today,
            0.30, 0.15, 365, 30, 0.0, "fifo",
        )
        cost_hifo, _ = _per_asset_tax_lots(
            hs, price, w_i, nav, today,
            0.30, 0.15, 365, 30, 0.0, "hifo",
        )
        # FIFO disposes of $100 basis lot ($30 gain × 30%) → $9/sh tax.
        # HIFO disposes of $120 basis lot ($10 gain × 30%) → $3/sh tax.
        assert cost_fifo > cost_hifo

    def test_lt_bridge_rate_is_bounded_and_decays_to_lt(self):
        st = 0.50
        lt = 0.32
        bridge = 30
        lt_days = 365

        rates = [
            _bridge_rate(st, lt, lt_days, days_held, bridge)
            for days_held in (335, 350, 364, 365)
        ]

        assert all(lt <= r <= st for r in rates)
        assert rates[0] == pytest.approx(st)
        assert rates[1] < rates[0]
        assert rates[2] < rates[1]
        assert rates[3] == pytest.approx(lt)


# ───────────────────────────────────────────────────────────────────────
# 6. Config kill-switch — qp_tax_lot_method = 'avg' falls back to legacy
# ───────────────────────────────────────────────────────────────────────

class TestAvgMethodKillSwitch:
    def test_avg_method_falls_back_to_legacy_path(self):
        """When `qp_tax_lot_method='avg'`, the task hits `_per_asset_tax`
        (single-entry), not the lot path — even when lots are present.

        Verified by: build a holding whose lot-path cost differs from
        legacy, set method='avg', confirm task output equals the legacy
        per-asset value."""
        hs = HoldingState(
            entry_price=110.0,
            entry_date=datetime.date(2026, 4, 1),
            high_watermark=130.0,
            shares=30.0,
        )
        # Multi-lot — lot-path would yield different cost.
        apply_buy_lot(hs, 10, 100.0, datetime.date(2026, 4, 1))
        apply_buy_lot(hs, 10, 120.0, datetime.date(2026, 4, 10))
        apply_buy_lot(hs, 10, 110.0, datetime.date(2026, 4, 20))
        # apply_buy_lot above will have updated entry_price to weighted avg = 110.
        assert abs(hs.entry_price - 110.0) < 1e-9

        ctx = _Ctx(
            config=_qp_tax_cfg(method="avg"),
            holdings={"AAPL": hs},
            prices={"AAPL": 130.0},
            today=datetime.date(2026, 5, 4),
            ytd_realized_gain_dollar=0.0,
        )
        ctx._qp_tickers = ["AAPL"]
        # w_current = shares × price / NAV = 30 × 130 / 100k = 0.039
        ctx._qp_w_current = np.array([30 * 130 / 100_000.0])

        ComputeBrownSmithTaxCostTask().run(ctx)
        cost_via_task = ctx._qp_tax_cost[0]

        # Independent legacy compute for parity
        cost_legacy, _ = _per_asset_tax(
            hs, 130.0, ctx._qp_w_current[0], 100_000.0, ctx.today,
            0.30, 0.15, 365, 30, 0.0,
        )
        assert abs(cost_via_task - cost_legacy) < 1e-9

    def test_qp_task_uses_lot_path_by_default(self):
        """Default config (no qp_tax_lot_method) → fifo → lot path.

        Multi-lot holding produces a cost that matches `_per_asset_tax_lots`
        with method='fifo', NOT `_per_asset_tax` (legacy)."""
        hs = HoldingState(
            entry_price=0.0, entry_date=None, high_watermark=130.0, shares=0.0,
        )
        apply_buy_lot(hs, 10, 100.0, datetime.date(2026, 4, 1))
        apply_buy_lot(hs, 10, 120.0, datetime.date(2026, 4, 10))
        apply_buy_lot(hs, 10, 110.0, datetime.date(2026, 4, 20))

        cfg = _qp_tax_cfg()   # defaults → fifo
        # Strip the explicit lot_method to exercise the .get() default.
        cfg["rotation"]["joint_actions"].pop("qp_tax_lot_method")
        ctx = _Ctx(
            config=cfg,
            holdings={"AAPL": hs},
            prices={"AAPL": 130.0},
            today=datetime.date(2026, 5, 4),
            ytd_realized_gain_dollar=0.0,
        )
        ctx._qp_tickers = ["AAPL"]
        ctx._qp_w_current = np.array([30 * 130 / 100_000.0])
        ComputeBrownSmithTaxCostTask().run(ctx)
        cost_via_task = ctx._qp_tax_cost[0]

        cost_fifo, _ = _per_asset_tax_lots(
            hs, 130.0, ctx._qp_w_current[0], 100_000.0, ctx.today,
            0.30, 0.15, 365, 30, 0.0, "fifo",
        )
        assert abs(cost_via_task - cost_fifo) < 1e-9
