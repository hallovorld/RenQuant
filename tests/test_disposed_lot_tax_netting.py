"""compute_disposed_lot_tax must net gains/losses within one sell event.

Regression suite for the mixed-sign multi-lot disposal bug found by the G4
rerun batch (first execution of the persistence-ON validation path over a
full window — the weekly gate's --no-persist never exercises it).

Pre-fix, ``compute_disposed_lot_tax`` taxed each positive-gain lot
independently and ignored losing lots, so a full exit of a topped-up
position at a price between the two lot bases produced "net loss with
positive tax" — an accounting impossibility that trips the decision-trace
integrity validator ``_sell_economics_are_valid`` (fail-closed
RuntimeError; the live daily runner calls validate_decision_trace_integrity,
so this was a latent LIVE fail-close risk).

Fixed semantics (mirrors ``compute_netted_capital_gains_tax`` exactly):
short-term lots net against short-term lots at the short-term rate,
long-term against long-term at the long-term rate, then opposite-sign
buckets offset each other (Schedule-D shape). The reported
``short_term_gross_pnl`` / ``long_term_gross_pnl`` splits stay pure
per-bucket sums.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.persistence import _sell_economics_are_valid  # noqa: E402
from kernel.portfolio import (  # noqa: E402
    compute_disposed_lot_tax,
    compute_netted_capital_gains_tax,
)

ST_RATE = 0.50
LT_RATE = 0.32


def _lot(shares: float, price: float, date: _dt.date) -> SimpleNamespace:
    return SimpleNamespace(shares=shares, price=price, date=date)


class TestVerifiedMAInstance:
    """Pin the exact verified instance: MA 2025-06-24 sim sell.

    Two disposed lots (top-up lot + original lot, full exit at a price
    between the two bases): per-lot gains +126.9676 / −193.2083, both
    short-term at rate 0.5. Pre-fix output: tax = 0.5 × 126.9676 =
    63.4838 on a gross of −66.2407 → validator trip. Post-fix: tax MUST
    be 0.0 and net == gross.
    """

    SELL_DATE = _dt.date(2025, 6, 24)
    SELL_PRICE = 560.0
    # gain = 1.0 × (560.0 − 433.0324) = +126.9676 (top-up lot, 43d hold)
    # gain = 1.0 × (560.0 − 753.2083) = −193.2083 (original lot, 287d hold)
    LOTS = [
        _lot(1.0, 433.0324, _dt.date(2025, 5, 12)),
        _lot(1.0, 753.2083, _dt.date(2024, 9, 10)),
    ]

    def _result(self) -> dict[str, float]:
        return compute_disposed_lot_tax(
            self.SELL_PRICE, self.SELL_DATE, self.LOTS, ST_RATE, LT_RATE,
        )

    def test_net_loss_event_pays_zero_tax(self):
        res = self._result()
        assert res["tax"] == 0.0

    def test_gross_split_matches_verified_instance(self):
        res = self._result()
        assert res["short_term_gross_pnl"] == pytest.approx(
            126.9676 - 193.2083, abs=1e-9,
        )
        assert res["long_term_gross_pnl"] == 0.0

    def test_net_equals_gross_when_tax_zero(self):
        res = self._result()
        gross = res["short_term_gross_pnl"] + res["long_term_gross_pnl"]
        net = gross - res["tax"]
        assert net == pytest.approx(gross)
        assert gross == pytest.approx(-66.2407, abs=1e-9)

    def test_weighted_hold_days_unchanged_by_fix(self):
        res = self._result()
        assert res["weighted_hold_days"] == pytest.approx((43 + 287) / 2.0)

    def test_fixed_outputs_pass_sell_economics_validator(self):
        res = self._result()
        gross = res["short_term_gross_pnl"] + res["long_term_gross_pnl"]
        tax = res["tax"]
        assert _sell_economics_are_valid(gross, tax, gross - tax) is True

    def test_prefix_outputs_fail_sell_economics_validator(self):
        """Deterministic reproduction: the PRE-fix numbers trip the validator.

        This is the exact triple the G4 rerun batch hit (loss with positive
        tax → invariant #3). The validator contract is correct; the tax
        computation was wrong.
        """
        gross = -66.2407
        prefix_tax = 0.5 * 126.9676  # 63.4838 — pre-fix per-lot taxation
        assert _sell_economics_are_valid(
            gross, prefix_tax, gross - prefix_tax,
        ) is False


class TestNetGainMixedSell:
    """Sibling failure mode: net-GAIN mixed sell whose per-lot tax exceeded
    net gross (validator invariant: tax cannot exceed positive gross)."""

    SELL_DATE = _dt.date(2025, 6, 24)
    SELL_PRICE = 560.0
    # gains: +126.9676 and −50.0, both short-term → net +76.9676
    LOTS = [
        _lot(1.0, 433.0324, _dt.date(2025, 5, 12)),
        _lot(1.0, 610.0, _dt.date(2024, 9, 10)),
    ]

    def _result(self) -> dict[str, float]:
        return compute_disposed_lot_tax(
            self.SELL_PRICE, self.SELL_DATE, self.LOTS, ST_RATE, LT_RATE,
        )

    def test_tax_is_rate_times_netted_sum(self):
        res = self._result()
        assert res["tax"] == pytest.approx(0.5 * (126.9676 - 50.0), abs=1e-9)

    def test_tax_within_validator_bound(self):
        # Post-fix the invariant-#4 bound holds structurally:
        # tax = rate × net_gain ≤ net_gain (rate ≤ 1) ≤ gross.
        res = self._result()
        gross = res["short_term_gross_pnl"] + res["long_term_gross_pnl"]
        assert gross > 0
        assert res["tax"] <= gross

    def test_deeper_loss_lot_prefix_exceeded_gross_now_bounded(self):
        # gains: +126.9676 and −70.0 → net +56.9676. Pre-fix tax 63.4838
        # EXCEEDS net gross (validator invariant #4 trip). Post-fix:
        # tax = 0.5 × 56.9676 = 28.4838 ≤ gross.
        lots = [
            _lot(1.0, 433.0324, _dt.date(2025, 5, 12)),
            _lot(1.0, 630.0, _dt.date(2024, 9, 10)),
        ]
        res = compute_disposed_lot_tax(
            self.SELL_PRICE, self.SELL_DATE, lots, ST_RATE, LT_RATE,
        )
        gross = res["short_term_gross_pnl"] + res["long_term_gross_pnl"]
        assert res["tax"] == pytest.approx(0.5 * (126.9676 - 70.0), abs=1e-9)
        assert res["tax"] <= gross
        assert _sell_economics_are_valid(gross, res["tax"], gross - res["tax"]) is True
        # And the pre-fix triple fails invariant #4 (tax > positive gross):
        assert _sell_economics_are_valid(gross, 63.4838, gross - 63.4838) is False


class TestAllGainAllLossRegression:
    """Uniform-sign events must behave exactly as before the fix."""

    SELL_DATE = _dt.date(2025, 6, 24)

    def test_all_gain_short_term_taxes_full_sum(self):
        lots = [
            _lot(1.0, 400.0, _dt.date(2025, 5, 12)),   # +100 ST
            _lot(2.0, 475.0, _dt.date(2025, 3, 1)),    # +50  ST
        ]
        res = compute_disposed_lot_tax(500.0, self.SELL_DATE, lots, ST_RATE, LT_RATE)
        assert res["tax"] == pytest.approx(0.5 * 150.0)
        assert res["short_term_gross_pnl"] == pytest.approx(150.0)
        assert res["long_term_gross_pnl"] == 0.0

    def test_all_gain_cross_bucket_each_bucket_taxed_at_own_rate(self):
        lots = [
            _lot(1.0, 400.0, _dt.date(2025, 5, 12)),   # +100 ST
            _lot(1.0, 300.0, _dt.date(2023, 1, 10)),   # +200 LT
        ]
        res = compute_disposed_lot_tax(500.0, self.SELL_DATE, lots, ST_RATE, LT_RATE)
        # 0.5×100 + 0.32×200 — ST gain never taxed at LT rate or vice versa
        assert res["tax"] == pytest.approx(50.0 + 64.0)
        assert res["short_term_gross_pnl"] == pytest.approx(100.0)
        assert res["long_term_gross_pnl"] == pytest.approx(200.0)

    def test_all_loss_pays_zero_tax(self):
        lots = [
            _lot(1.0, 600.0, _dt.date(2025, 5, 12)),   # −100 ST
            _lot(1.0, 550.0, _dt.date(2023, 1, 10)),   # −50  LT
        ]
        res = compute_disposed_lot_tax(500.0, self.SELL_DATE, lots, ST_RATE, LT_RATE)
        assert res["tax"] == 0.0


class TestBucketSemanticsMirrorNettedHelper:
    """Bucket netting must match compute_netted_capital_gains_tax EXACTLY.

    Same-bucket netting first; then opposite-sign buckets offset each other
    at the gaining bucket's rate (Schedule-D shape). Cross-bucket offset is
    REQUIRED for validator safety: with ST net +100 / LT net −150 the event
    gross is −50, so any positive tax would recreate the "loss with
    positive tax" trip. The per-bucket gross splits stay unmixed — losses
    in one bucket never leak into the OTHER bucket's reported gross.
    """

    SELL_DATE = _dt.date(2025, 6, 24)

    def test_st_gain_lt_loss_matches_helper(self):
        lots = [
            _lot(1.0, 400.0, _dt.date(2025, 5, 12)),   # +100 ST
            _lot(1.0, 540.0, _dt.date(2023, 1, 10)),   # −40  LT
        ]
        res = compute_disposed_lot_tax(500.0, self.SELL_DATE, lots, ST_RATE, LT_RATE)
        expected = compute_netted_capital_gains_tax(100.0, -40.0, ST_RATE, LT_RATE)
        assert expected == pytest.approx(0.5 * 60.0)  # gaining bucket's rate
        assert res["tax"] == expected
        # Reported bucket splits remain pure per-bucket sums (unmixed):
        assert res["short_term_gross_pnl"] == pytest.approx(100.0)
        assert res["long_term_gross_pnl"] == pytest.approx(-40.0)

    def test_lt_gain_st_loss_matches_helper(self):
        lots = [
            _lot(1.0, 540.0, _dt.date(2025, 5, 12)),   # −40  ST
            _lot(1.0, 400.0, _dt.date(2023, 1, 10)),   # +100 LT
        ]
        res = compute_disposed_lot_tax(500.0, self.SELL_DATE, lots, ST_RATE, LT_RATE)
        expected = compute_netted_capital_gains_tax(-40.0, 100.0, ST_RATE, LT_RATE)
        assert expected == pytest.approx(0.32 * 60.0)
        assert res["tax"] == expected

    def test_cross_bucket_net_loss_zero_tax_validator_safe(self):
        lots = [
            _lot(1.0, 400.0, _dt.date(2025, 5, 12)),   # +100 ST
            _lot(1.0, 650.0, _dt.date(2023, 1, 10)),   # −150 LT
        ]
        res = compute_disposed_lot_tax(500.0, self.SELL_DATE, lots, ST_RATE, LT_RATE)
        assert res["tax"] == 0.0
        gross = res["short_term_gross_pnl"] + res["long_term_gross_pnl"]
        assert gross == pytest.approx(-50.0)
        assert _sell_economics_are_valid(gross, res["tax"], gross) is True

    def test_same_bucket_netting_precedes_cross_offset(self):
        lots = [
            _lot(1.0, 400.0, _dt.date(2025, 5, 12)),   # +100 ST
            _lot(1.0, 530.0, _dt.date(2025, 3, 1)),    # −30  ST
            _lot(1.0, 520.0, _dt.date(2023, 1, 10)),   # −20  LT
        ]
        res = compute_disposed_lot_tax(500.0, self.SELL_DATE, lots, ST_RATE, LT_RATE)
        expected = compute_netted_capital_gains_tax(70.0, -20.0, ST_RATE, LT_RATE)
        assert expected == pytest.approx(0.5 * 50.0)
        assert res["tax"] == expected
