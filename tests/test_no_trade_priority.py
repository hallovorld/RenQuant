"""Regression tests for `_no_trade_reason` priority contract.

Codex PR #48 review MED #2: when both `risk_gate_vol_dropped` and
`qp_infeasible` counters are non-zero, the ntfy body must surface the
binding QP failure, not the upstream vol gate.

The 2026-06-01 prod daily incident had counters like
``{risk_gate_vol_dropped: 10, regime_admission_blocked: ~, qp_infeasible: 1}``
and ntfy showed `no trade (risk_gate_vol_dropped(10))` instead of the
actual binding constraint.

These tests pin the new priority so refactors can't regress it silently.
"""
from __future__ import annotations

import types
import pytest

from live.runner import _no_trade_reason

#: This file IS the operator-notification contract: it calls the composition
#: helpers and asserts on what the operator would read. Membership is this
#: MARKER, deliberately applied, not a substring scan over the source — a scan
#: cannot tell a contract from a file that merely monkeypatches a helper away
#: [codex on RenQuant#601]. The workflow runs exactly the marked files, and
#: TestTheCONTRACTWorkflowActuallyRunsTheseTests asserts both directions.
pytestmark = pytest.mark.notification_contract


def _ctx(**counters):
    return types.SimpleNamespace(counters=dict(counters))


# ── Priority contract: binding constraint beats upstream drop ────────────────

def test_qp_infeasible_beats_risk_gate_vol_dropped():
    """The exact 2026-06-01 scenario: vol gate dropped 10, QP infeasible."""
    ctx = _ctx(risk_gate_vol_dropped=10, qp_infeasible=1)
    assert _no_trade_reason(ctx) == "qp_infeasible(1)"


def test_regime_admission_blocked_beats_risk_gate_vol_dropped():
    """Mid-pipeline admission block surfaces over upstream vol drop."""
    ctx = _ctx(risk_gate_vol_dropped=10, regime_admission_blocked=72)
    assert _no_trade_reason(ctx) == "regime_admission_blocked(72)"


def test_panel_scoring_fail_closed_beats_qp_infeasible_and_vol():
    """Earliest-stage fail-closed wins (it ran first and is the root cause)."""
    ctx = _ctx(
        risk_gate_vol_dropped=10,
        qp_infeasible=1,
        regime_admission_blocked=72,
        panel_scoring_fail_closed=82,
    )
    assert _no_trade_reason(ctx) == "panel_scoring_fail_closed(82)"


def test_admission_block_beats_qp_infeasible():
    """When admission blocks all candidates, QP runs against holdings only.
    The admission block is the more meaningful surface than the resulting
    QP infeasibility on holdings."""
    ctx = _ctx(regime_admission_blocked=72, qp_infeasible=1)
    assert _no_trade_reason(ctx) == "regime_admission_blocked(72)"


# ── Earliest-stage fail-closed precedence ────────────────────────────────────

def test_qp_mu_contract_block_beats_admission_and_qp():
    ctx = _ctx(
        regime_admission_blocked=72,
        qp_infeasible=1,
        qp_mu_contract_block=3,
    )
    assert _no_trade_reason(ctx) == "qp_mu_contract_block(3)"


# ── QP failure variants ──────────────────────────────────────────────────────

def test_qp_missing_solution_surfaces():
    ctx = _ctx(risk_gate_vol_dropped=5, qp_missing_solution=1)
    assert _no_trade_reason(ctx) == "qp_missing_solution(1)"


def test_qp_other_nonoptimal_surfaces():
    ctx = _ctx(qp_other_nonoptimal=2)
    assert _no_trade_reason(ctx) == "qp_other_nonoptimal(2)"


# ── risk_gate_vol_dropped is the cause only when ALONE ───────────────────────

def test_vol_gate_surfaces_only_when_alone():
    ctx = _ctx(risk_gate_vol_dropped=10)
    assert _no_trade_reason(ctx) == "risk_gate_vol_dropped(10)"


# ── Empty counters fall through to downstream causes ─────────────────────────

def test_empty_counters_returns_no_candidates_when_ranked_empty():
    ctx = _ctx()
    ctx.ranked = []
    assert _no_trade_reason(ctx) == "no_candidates"


def test_empty_counters_returns_tier_threshold_when_ranked_present():
    ctx = _ctx()
    ctx.ranked = [object(), object()]
    assert _no_trade_reason(ctx) == "tier_threshold"


# ── Bear / transition window short-circuit upfront ───────────────────────────

def test_bear_only_short_circuits_everything():
    ctx = _ctx(risk_gate_vol_dropped=10, qp_infeasible=1)
    ctx.bear_only = True
    assert _no_trade_reason(ctx) == "bear_only"


def test_transition_window_short_circuits():
    ctx = _ctx(qp_infeasible=1)
    ctx.regime_state = types.SimpleNamespace(in_transition=True)
    assert _no_trade_reason(ctx) == "transition_window"


# ── fix/size-on-settled-cash (2026-08-30): a $0 buy budget with a named ──────
# cause is the binding constraint for every candidate and is surfaced by name
# ahead of the counter rollups (after bear_only / transition_window).

def test_no_settled_cash_beats_counter_rollups():
    ctx = _ctx(risk_gate_vol_dropped=10, qp_infeasible=1, qp_zero_shares=4)
    ctx.buy_sizing_cash = {"mode": "settled_cash", "sizing_cash": 0.0,
                           "sizing_reason": "no_settled_cash"}
    assert _no_trade_reason(ctx) == "no_settled_cash"


def test_positive_budget_does_not_change_the_rollup():
    ctx = _ctx(risk_gate_vol_dropped=10, qp_infeasible=1)
    ctx.buy_sizing_cash = {"mode": "settled_cash", "sizing_cash": 33.0,
                           "sizing_reason": None}
    assert _no_trade_reason(ctx) == "qp_infeasible(1)"


def test_missing_buy_sizing_attribute_is_neutral():
    ctx = _ctx(qp_infeasible=1)
    assert _no_trade_reason(ctx) == "qp_infeasible(1)"


def test_cash_read_failure_is_named_too():
    ctx = _ctx(qp_infeasible=1)
    ctx.buy_sizing_cash = {"mode": "unavailable", "sizing_cash": 0.0,
                           "sizing_reason": "cash_read_failed"}
    assert _no_trade_reason(ctx) == "cash_read_failed"


def test_transition_window_still_outranks_no_settled_cash():
    ctx = _ctx()
    ctx.regime_state = types.SimpleNamespace(in_transition=True)
    ctx.buy_sizing_cash = {"sizing_reason": "no_settled_cash"}
    assert _no_trade_reason(ctx) == "transition_window"
