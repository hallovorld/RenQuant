"""Unit tests for the extracted sim cash / buying-power computation.

Pins adapters/sim_cash.py (sim.py decomposition, S2 item 5) at the module
boundary. These pure functions compute the cash budget the decision tree sizes
against: settled cash, optionally topped up with T+N pending-settlement
proceeds in non-marginable-buying-power (NMBP) mode. The SimAdapter methods
_pending_settle_cash / _available_buying_power are now thin delegates; this
locks the behavior they delegate to.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.sim_cash import available_buying_power, pending_settle_cash  # noqa: E402
from adapters.sim_order_helpers import (  # noqa: E402
    _BUYING_POWER_NMBP,
    _BUYING_POWER_SETTLED,
)


def _queue(total):
    return NS(pending_total=lambda: total)


class TestPendingSettleCash:
    def test_none_queue_is_zero(self):
        assert pending_settle_cash(None) == 0.0

    def test_returns_queue_pending_total(self):
        assert pending_settle_cash(_queue(250.0)) == 250.0

    def test_non_finite_pending_is_zero(self):
        assert pending_settle_cash(_queue(float("nan"))) == 0.0
        assert pending_settle_cash(_queue(float("inf"))) == 0.0


class TestAvailableBuyingPower:
    def test_settled_mode_ignores_pending(self):
        bp = available_buying_power(
            cash=1000.0, exec_enabled=True,
            buying_power_mode=_BUYING_POWER_SETTLED, t2_queue=_queue(250.0))
        assert bp == 1000.0

    def test_nmbp_mode_adds_pending(self):
        bp = available_buying_power(
            cash=1000.0, exec_enabled=True,
            buying_power_mode=_BUYING_POWER_NMBP, t2_queue=_queue(250.0))
        assert bp == 1250.0

    def test_nmbp_requires_exec_enabled(self):
        # NMBP mode but exec disabled → pending is NOT added.
        bp = available_buying_power(
            cash=1000.0, exec_enabled=False,
            buying_power_mode=_BUYING_POWER_NMBP, t2_queue=_queue(250.0))
        assert bp == 1000.0

    def test_none_and_zero_cash(self):
        assert available_buying_power(
            cash=None, exec_enabled=True, buying_power_mode=_BUYING_POWER_NMBP,
            t2_queue=None) == 0.0
        assert available_buying_power(
            cash=0.0, exec_enabled=False, buying_power_mode=_BUYING_POWER_SETTLED,
            t2_queue=None) == 0.0

    def test_non_finite_cash_is_zero(self):
        assert available_buying_power(
            cash=float("nan"), exec_enabled=False,
            buying_power_mode=_BUYING_POWER_SETTLED, t2_queue=None) == 0.0
        assert available_buying_power(
            cash=float("inf"), exec_enabled=False,
            buying_power_mode=_BUYING_POWER_SETTLED, t2_queue=None) == 0.0

    def test_nmbp_non_finite_pending_falls_back_to_cash(self):
        # a non-finite pending_total contributes 0, leaving plain cash.
        bp = available_buying_power(
            cash=1000.0, exec_enabled=True,
            buying_power_mode=_BUYING_POWER_NMBP, t2_queue=_queue(float("nan")))
        assert bp == 1000.0


class TestDelegateParity:
    """The SimAdapter delegates must produce the same numbers as calling the
    pure functions with the same self-attrs — the contract that makes the
    extraction behavior-preserving."""

    def test_delegates_match_pure_functions(self):
        from adapters.sim import SimAdapter

        adapter = SimAdapter.__new__(SimAdapter)  # skip __init__
        adapter._t2_queue = _queue(300.0)
        adapter._cash = 5000.0
        adapter._exec_enabled = True
        adapter._buying_power_mode = _BUYING_POWER_NMBP

        assert adapter._pending_settle_cash() == pending_settle_cash(adapter._t2_queue)
        assert adapter._available_buying_power() == available_buying_power(
            cash=adapter._cash, exec_enabled=adapter._exec_enabled,
            buying_power_mode=adapter._buying_power_mode, t2_queue=adapter._t2_queue)
        assert adapter._available_buying_power() == 5300.0  # 5000 + 300 pending
