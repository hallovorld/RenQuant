"""Live <-> Sim reconciliation toolkit.

Continuous observability layer that replays every Alpaca fill through the
SimAdapter at the same timestamp and emits divergence metrics daily.

Public surface intentionally small — see ``live_sim_reconcile`` for the
helpers and ``scripts/reconcile_live_sim.py`` for the CLI.
"""
from __future__ import annotations

from .live_sim_reconcile import (
    LiveFill,
    SimDecision,
    compute_decision_divergence,
    compute_rolling_ic,
    compute_slippage,
    emit_report,
    load_live_fills,
    load_sim_decisions,
    replay_through_sim,
)

__all__ = [
    "LiveFill",
    "SimDecision",
    "compute_decision_divergence",
    "compute_rolling_ic",
    "compute_slippage",
    "emit_report",
    "load_live_fills",
    "load_sim_decisions",
    "replay_through_sim",
]
