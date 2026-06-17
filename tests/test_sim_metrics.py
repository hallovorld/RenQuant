"""sim.py decomposition slice 1 — sim_metrics pure-function tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

import pandas as pd  # noqa: E402

from adapters.sim_metrics import (  # noqa: E402
    _finite_attr_values,
    _finite_float,
    _mean_or_nan,
    _quantile_or_nan,
    _tax_cash_debit_amount,
    _tax_cash_debit_mode,
    activity_streak_stats,
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


def _legacy_activity_stats(trade_log, equity_df):
    """Verbatim copy of the pre-extraction in-line algorithm from
    SimAdapter.build_result — the parity oracle for the extracted function."""
    trade_dates = {
        (t["date"].date() if hasattr(t["date"], "date") else t["date"])
        for t in trade_log
    }
    eq_dates = [
        (d.date() if hasattr(d, "date") else d) for d in equity_df.index
    ] if not equity_df.empty else []
    longest_streak = 0
    current_streak = 0
    first_trade = None
    last_activity = None
    for d in eq_dates:
        if d in trade_dates:
            current_streak = 0
            last_activity = str(d)
            if first_trade is None:
                first_trade = str(d)
        else:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
    return {
        "longest_no_trade_streak": longest_streak,
        "first_trade_date": first_trade,
        "last_activity_date": last_activity,
    }


def _equity(dates):
    idx = pd.to_datetime(list(dates))
    return pd.DataFrame({"portfolio": range(len(idx))}, index=idx)


class TestActivityStreakStats:
    def test_empty_equity_curve_is_idle(self):
        out = activity_streak_stats([], _equity([]))
        assert out == {
            "longest_no_trade_streak": 0,
            "first_trade_date": None,
            "last_activity_date": None,
        }

    def test_no_trades_streak_spans_window(self):
        eq = _equity(["2026-01-02", "2026-01-03", "2026-01-04"])
        out = activity_streak_stats([], eq)
        assert out["longest_no_trade_streak"] == 3
        assert out["first_trade_date"] is None
        assert out["last_activity_date"] is None

    def test_first_and_last_activity_tracked(self):
        eq = _equity(["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
        log = [
            {"date": pd.Timestamp("2026-01-03"), "action": "buy"},
            {"date": pd.Timestamp("2026-01-05"), "action": "sell"},
        ]
        out = activity_streak_stats(log, eq)
        assert out["first_trade_date"] == "2026-01-03"
        assert out["last_activity_date"] == "2026-01-05"
        # idle days: 2026-01-02 (1 before first trade), 2026-01-04 (1 between)
        assert out["longest_no_trade_streak"] == 1

    def test_longest_streak_is_the_max_idle_run(self):
        eq = _equity([
            "2026-01-02", "2026-01-03", "2026-01-04",
            "2026-01-05", "2026-01-06", "2026-01-07",
        ])
        log = [{"date": pd.Timestamp("2026-01-02"), "action": "buy"}]
        out = activity_streak_stats(log, eq)
        # one trade on day 1, then 5 consecutive idle days
        assert out["longest_no_trade_streak"] == 5
        assert out["first_trade_date"] == "2026-01-02"
        assert out["last_activity_date"] == "2026-01-02"

    def test_date_objects_in_log_normalized_to_date(self):
        # trade-log entries carrying a python date (no .date()) still match.
        import datetime as _dt
        eq = _equity(["2026-01-02", "2026-01-03"])
        log = [{"date": _dt.date(2026, 1, 3), "action": "buy"}]
        out = activity_streak_stats(log, eq)
        assert out["last_activity_date"] == "2026-01-03"
        assert out["longest_no_trade_streak"] == 1

    def test_matches_legacy_inline_algorithm(self):
        # Replay parity: extracted fn ≡ the pre-extraction in-line block,
        # across several shapes.
        eq = _equity([
            "2026-01-02", "2026-01-03", "2026-01-04",
            "2026-01-05", "2026-01-06",
        ])
        logs = [
            [],
            [{"date": pd.Timestamp("2026-01-02"), "action": "buy"}],
            [
                {"date": pd.Timestamp("2026-01-03"), "action": "buy"},
                {"date": pd.Timestamp("2026-01-06"), "action": "sell"},
            ],
        ]
        for log in logs:
            assert activity_streak_stats(log, eq) == _legacy_activity_stats(log, eq)
        # also the empty-equity edge case
        assert activity_streak_stats([], _equity([])) == _legacy_activity_stats(
            [], _equity([]))
