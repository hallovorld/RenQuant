"""Unit tests for the extracted LEAN account / buying-power helpers.

Pins adapters/lean_account.py (lean.py decomposition slice 2) at the module
boundary, independent of the LEAN adapter integration tests. These pure
functions read a QCAlgorithm Portfolio into the buying-power + post-execution
snapshot dicts the adapter persists.

REGRESSION GUARD: the helpers (and the _BUYING_POWER_* constants) must remain
importable from BOTH adapters.lean_account (canonical) and adapters.lean
(back-compat re-export) as the SAME object — make_context (lean.py:540-ish)
and the commit decision-trace (lean.py:1177-ish) call them by the re-exported
name.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters import lean as _lean  # noqa: E402
from adapters import lean_account as _la  # noqa: E402


class TestReexportIdentity:
    def test_constants_and_fns_are_same_objects(self):
        for n in ("_BUYING_POWER_SETTLED", "_BUYING_POWER_NMBP",
                  "_BUYING_POWER_ALIASES", "_normalize_buying_power_mode",
                  "_finite_attr_float", "_lean_buying_power_snapshot",
                  "_lean_post_execution_snapshot"):
            assert getattr(_lean, n) is getattr(_la, n), n


class TestNormalizeBuyingPowerMode:
    def test_aliases_map_to_canonical(self):
        assert _la._normalize_buying_power_mode("settled") == _la._BUYING_POWER_SETTLED
        assert _la._normalize_buying_power_mode("cash") == _la._BUYING_POWER_SETTLED
        assert _la._normalize_buying_power_mode("settled_cash") == _la._BUYING_POWER_SETTLED
        assert _la._normalize_buying_power_mode("unsettled") == _la._BUYING_POWER_NMBP
        assert _la._normalize_buying_power_mode("cash_plus_unsettled") == _la._BUYING_POWER_NMBP

    def test_default_is_nmbp(self):
        assert _la._normalize_buying_power_mode(None) == _la._BUYING_POWER_NMBP
        assert _la._normalize_buying_power_mode("") == _la._BUYING_POWER_NMBP

    def test_case_and_whitespace_insensitive(self):
        assert _la._normalize_buying_power_mode("  SETTLED  ") == _la._BUYING_POWER_SETTLED

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="buying_power_mode"):
            _la._normalize_buying_power_mode("margin_4x")


class TestFiniteAttrFloat:
    def test_first_finite_attribute_wins(self):
        obj = NS(a=float("nan"), b=12.5, c=99.0)
        assert _la._finite_attr_float(obj, "a", "b", "c") == 12.5

    def test_callable_attribute_is_invoked(self):
        obj = NS(get_cash=lambda: 250.0)
        assert _la._finite_attr_float(obj, "get_cash") == 250.0

    def test_callable_raising_is_skipped(self):
        def _boom():
            raise RuntimeError("broker down")
        obj = NS(bad=_boom, good=7.0)
        assert _la._finite_attr_float(obj, "bad", "good") == 7.0

    def test_non_finite_and_unparseable_skipped(self):
        obj = NS(a=float("inf"), b="x", c=None, d=3.0)
        assert _la._finite_attr_float(obj, "a", "b", "c", "d") == 3.0

    def test_returns_none_when_nothing_finite(self):
        obj = NS(a=float("nan"), b=None)
        assert _la._finite_attr_float(obj, "a", "b", "missing") is None


class TestBuyingPowerSnapshot:
    def test_settled_mode_uses_portfolio_cash_only(self):
        algo = NS(Portfolio=NS(Cash=1000.0, NonMarginableBuyingPower=5000.0))
        bp = _la._lean_buying_power_snapshot(algo, {"execution": {"buying_power_mode": "settled"}})
        assert bp["cash"] == 1000.0
        assert bp["pending_settle_cash"] == 0.0
        assert bp["buying_power_source"] == "portfolio_cash"

    def test_nmbp_mode_uses_non_marginable_and_derives_pending(self):
        algo = NS(Portfolio=NS(Cash=1000.0, NonMarginableBuyingPower=1500.0))
        bp = _la._lean_buying_power_snapshot(algo, {"execution": {"buying_power_mode": "cash_plus_unsettled"}})
        assert bp["cash"] == 1500.0
        assert bp["settled_cash"] == 1000.0
        assert bp["pending_settle_cash"] == 500.0  # nmbp - settled
        assert bp["buying_power_source"] == "portfolio_non_marginable_buying_power"

    def test_nmbp_fallback_to_algo_pending_when_no_broker_nmbp(self):
        algo = NS(Portfolio=NS(Cash=1000.0), _pending_settle_cash=300.0)
        bp = _la._lean_buying_power_snapshot(algo, {})  # default mode = nmbp
        assert bp["cash"] == 1300.0
        assert bp["pending_settle_cash"] == 300.0
        assert bp["buying_power_source"] == "algo_pending_settle_cash"

    def test_nmbp_fallback_to_portfolio_cash_when_no_pending(self):
        algo = NS(Portfolio=NS(Cash=1000.0))
        bp = _la._lean_buying_power_snapshot(algo, {})
        assert bp["cash"] == 1000.0
        assert bp["pending_settle_cash"] == 0.0
        assert bp["buying_power_source"] == "portfolio_cash_fallback"


class TestPostExecutionSnapshot:
    def test_reads_portfolio_value_and_counts_positive_holdings(self):
        algo = NS(
            Portfolio=NS(Cash=1000.0, TotalPortfolioValue=12_000.0,
                         NonMarginableBuyingPower=1000.0),
            _holdings={"A": NS(shares=10.0), "B": NS(shares=0.0), "C": NS(shares=5.0)},
        )
        out = _la._lean_post_execution_snapshot(algo, {}, NS())
        assert out["portfolio_value"] == 12_000.0
        assert out["cash"] == 1000.0
        assert out["n_holdings"] == 2  # B has 0 shares, excluded

    def test_falls_back_to_ctx_when_portfolio_value_missing(self):
        algo = NS(Portfolio=NS(Cash=500.0), _holdings={})
        ctx = NS(portfolio_value=8_000.0)
        out = _la._lean_post_execution_snapshot(algo, {"execution": {"buying_power_mode": "settled"}}, ctx)
        assert out["portfolio_value"] == 8_000.0

    def test_falls_back_to_ctx_cash_when_portfolio_cash_missing(self):
        # Portfolio with no Cash attr → cash comes from ctx.
        algo = NS(Portfolio=NS(), _holdings={})
        ctx = NS(cash=4_321.0)
        out = _la._lean_post_execution_snapshot(algo, {"execution": {"buying_power_mode": "settled"}}, ctx)
        assert out["cash"] == 4_321.0
        assert out["settled_cash"] == 4_321.0
        assert out["pending_settle_cash"] == 0.0
