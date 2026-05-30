"""ApplyProportionalTradeTask — Task-level contract tests.

Pins:
  * Task is in JointPortfolioQPJob between SolveMarkowitzQPTask and
    EmitOrdersFromQPSolutionTask (the GP-2013 shrink must happen on the
    QP solution but before order emission)
  * N=1 / missing config → Task is no-op (legacy behaviour)
  * N>1 → ctx._qp_solution.target_w shrunk by 1/N; delta_w recomputed
  * Per-regime override wins over global default (PRIME DIRECTIVE)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.portfolio_qp.job_qp import JointPortfolioQPJob  # noqa: E402
from kernel.portfolio_qp.tasks import ApplyProportionalTradeTask  # noqa: E402


@dataclass
class FakeSol:
    target_w: np.ndarray
    delta_w: np.ndarray


@dataclass
class FakeCtx:
    regime: str = "BULL_CALM"
    config: dict = field(default_factory=dict)
    _qp_solution: FakeSol | None = None
    _qp_w_current: np.ndarray | None = None
    _qp_partial_trade_applied: bool | None = None
    _qp_partial_trade_n_days: float | None = None


def test_task_in_job_between_solve_and_emit():
    """Job wiring: SolveMarkowitzQPTask → ApplyProportionalTradeTask → EmitOrders…"""
    job = JointPortfolioQPJob()
    names = [type(t).__name__ for t in job.tasks]
    solve_i = names.index("SolveMarkowitzQPTask")
    ptt_i = names.index("ApplyProportionalTradeTask")
    emit_i = names.index("EmitOrdersFromQPSolutionTask")
    assert solve_i < ptt_i < emit_i


def test_noop_when_n_equals_one_globally():
    """Default config has no horizon → legacy all-or-nothing."""
    ctx = FakeCtx(
        regime="BULL_CALM",
        config={"regime_params": {"BULL_CALM": {}}, "rotation": {"joint_actions": {}}},
        _qp_solution=FakeSol(target_w=np.array([0.0, 0.5, 0.1]),
                              delta_w=np.array([-0.1, 0.5, 0.1])),
        _qp_w_current=np.array([0.1, 0.0, 0.0]),
    )
    out = ApplyProportionalTradeTask().run(ctx)
    assert out is None
    assert ctx._qp_partial_trade_applied is False
    # Target unchanged
    np.testing.assert_array_almost_equal(ctx._qp_solution.target_w, [0.0, 0.5, 0.1])


def test_global_default_horizon_applies_to_all_regimes():
    """global ``qp_partial_trade_horizon_days=5`` applies when regime override absent."""
    ctx = FakeCtx(
        regime="BULL_CALM",
        config={
            "regime_params": {},
            "rotation": {"joint_actions": {"qp_partial_trade_horizon_days": 5}},
        },
        _qp_solution=FakeSol(
            target_w=np.array([0.0, 0.5, 0.1]),
            delta_w=np.array([-0.1, 0.5, 0.1]),
        ),
        _qp_w_current=np.array([0.1, 0.0, 0.0]),
    )
    ApplyProportionalTradeTask().run(ctx)
    # current + (target - current)/5
    # = [0.1, 0.0, 0.0] + [-0.1/5, 0.5/5, 0.1/5]
    # = [0.08, 0.10, 0.02]
    np.testing.assert_array_almost_equal(ctx._qp_solution.target_w, [0.08, 0.10, 0.02])
    # delta_w recomputed from new target
    np.testing.assert_array_almost_equal(ctx._qp_solution.delta_w, [-0.02, 0.10, 0.02])
    assert ctx._qp_partial_trade_applied is True
    assert ctx._qp_partial_trade_n_days == 5.0


def test_regime_override_wins_over_global(monkeypatch):
    """Per-regime knob wins over global default (PRIME DIRECTIVE)."""
    ctx = FakeCtx(
        regime="BULL_CALM",
        config={
            "regime_params": {"BULL_CALM": {"qp_partial_trade_horizon_days": 2}},
            "rotation": {"joint_actions": {"qp_partial_trade_horizon_days": 20}},
        },
        _qp_solution=FakeSol(
            target_w=np.array([0.2]),
            delta_w=np.array([0.1]),
        ),
        _qp_w_current=np.array([0.1]),
    )
    ApplyProportionalTradeTask().run(ctx)
    # current=0.1, target=0.2, N=2 → partial=0.15
    np.testing.assert_array_almost_equal(ctx._qp_solution.target_w, [0.15])
    assert ctx._qp_partial_trade_n_days == 2.0


def test_skip_when_qp_solution_missing():
    ctx = FakeCtx(
        regime="BULL_CALM",
        config={"regime_params": {}, "rotation": {"joint_actions": {"qp_partial_trade_horizon_days": 5}}},
        _qp_solution=None,
        _qp_w_current=np.array([0.1]),
    )
    out = ApplyProportionalTradeTask().run(ctx)
    assert out is None
    # No mutation, no audit stamp set
    assert ctx._qp_partial_trade_applied is None


def test_skip_when_w_current_missing():
    ctx = FakeCtx(
        regime="BULL_CALM",
        config={"regime_params": {}, "rotation": {"joint_actions": {"qp_partial_trade_horizon_days": 5}}},
        _qp_solution=FakeSol(target_w=np.array([0.2]), delta_w=np.array([0.2])),
        _qp_w_current=None,
    )
    out = ApplyProportionalTradeTask().run(ctx)
    assert out is None


def test_n_equals_one_explicit_is_noop():
    """If operator sets qp_partial_trade_horizon_days=1 explicitly → still no-op."""
    ctx = FakeCtx(
        regime="BULL_CALM",
        config={"regime_params": {"BULL_CALM": {"qp_partial_trade_horizon_days": 1}},
                "rotation": {"joint_actions": {}}},
        _qp_solution=FakeSol(target_w=np.array([0.5]), delta_w=np.array([0.5])),
        _qp_w_current=np.array([0.0]),
    )
    ApplyProportionalTradeTask().run(ctx)
    np.testing.assert_array_almost_equal(ctx._qp_solution.target_w, [0.5])
    assert ctx._qp_partial_trade_applied is False
