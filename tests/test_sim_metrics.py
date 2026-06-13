"""sim.py decomposition slice 1 — sim_metrics pure-function tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.sim_metrics import (  # noqa: E402
    _finite_attr_values,
    _finite_float,
    _mean_or_nan,
    _quantile_or_nan,
    _tax_cash_debit_amount,
    _tax_cash_debit_mode,
)


class TestFiniteHelpers:
    def test_finite_float_passthrough(self):
        assert _finite_float(3.5) == 3.5

    def test_finite_float_nan_default(self):
        assert _finite_float(float("nan"), default=-1.0) == -1.0
        assert _finite_float("x", default=0.0) == 0.0

    def test_mean_or_nan(self):
        assert _mean_or_nan([1.0, 3.0]) == 2.0
        assert math.isnan(_mean_or_nan([]))

    def test_quantile_or_nan(self):
        assert _quantile_or_nan([0.0, 1.0], 0.5) == 0.5
        assert math.isnan(_quantile_or_nan([], 0.5))

    def test_finite_attr_values_filters_nonfinite(self):
        from types import SimpleNamespace
        items = [SimpleNamespace(x=1.0), SimpleNamespace(x=float("nan")),
                 SimpleNamespace(x=3.0)]
        assert _finite_attr_values(items, "x") == [1.0, 3.0]


class TestTaxCashDebit:
    def test_mode_default_event_level(self):
        assert _tax_cash_debit_mode(None) == "event_level"
        assert _tax_cash_debit_mode({}) == "event_level"

    def test_mode_aliases(self):
        assert _tax_cash_debit_mode({"tax": {"cash_debit_mode": "none"}}) == "reporting_only"
        assert _tax_cash_debit_mode({"tax": {"cash_debit_mode": "annual_net"}}) == "reporting_only"
        assert _tax_cash_debit_mode({"tax": {"cash_debit_mode": "stress"}}) == "event_level"

    def test_amount_event_level_passes_tax(self):
        assert _tax_cash_debit_amount({"tax": {"cash_debit_mode": "event"}}, 12.0) == 12.0

    def test_amount_reporting_only_zero(self):
        assert _tax_cash_debit_amount({"tax": {"cash_debit_mode": "reporting"}}, 12.0) == 0.0

    def test_amount_nonpositive_zero(self):
        assert _tax_cash_debit_amount(None, -5.0) == 0.0
        assert _tax_cash_debit_amount(None, float("nan")) == 0.0
