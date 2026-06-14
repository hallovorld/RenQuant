"""Sim cash / buying-power computation — sim.py decomposition slice (S2 item 5).

EXTRACTED 2026-06-14 from adapters/sim.py. Pure functions for the cash budget
the decision tree may size against: settled cash, optionally topped up with
T+N pending-settlement proceeds when running the non-marginable-buying-power
mode (the live Alpaca path). SimAdapter keeps thin method delegates; behavior
is unchanged (guarded by the deterministic replay-parity harness). No
SimAdapter state — self-deps are passed in.
"""
from __future__ import annotations

import math
from typing import Any

from adapters.sim_order_helpers import _BUYING_POWER_NMBP


def pending_settle_cash(t2_queue: Any) -> float:
    """T+N proceeds not yet settled, 0.0 when no queue / non-finite."""
    if t2_queue is None:
        return 0.0
    pending = t2_queue.pending_total()
    return pending if math.isfinite(pending) else 0.0


def available_buying_power(
    *,
    cash: Any,
    exec_enabled: bool,
    buying_power_mode: str,
    t2_queue: Any,
) -> float:
    """Cash budget exposed to the decision tree for new long buys.

    ``settled_cash`` is conservative cash-account behavior. The default
    ``non_marginable_buying_power`` mirrors the live Alpaca broker path:
    executed sell proceeds replenish non-margin buying power before they have
    fully settled, while still avoiding 2x/4x margin buying power.
    """
    cash = float(cash or 0.0)
    if not math.isfinite(cash):
        return 0.0
    if exec_enabled and buying_power_mode == _BUYING_POWER_NMBP:
        cash += pending_settle_cash(t2_queue)
    return cash if math.isfinite(cash) else 0.0
