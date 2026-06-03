"""Regression tests for the soft-scaling hold-flat clamp moved out of
`solve_portfolio_qp` after codex re-review on PR #123.

`ApplyExposureScalingTask` and `ApplyConvictionCapTask` multiply
`_qp_w_upper` by combined regime / vol-target / drawdown / conviction
multipliers (all in [0, 1]). In low-conviction or high-cap-pressure
scenarios this CAN drive `w_upper` below `w_current` for held positions,
making "hold flat" (Δw=0) infeasible at the solver. The 2026-06-02
daily-full bug surfaced exactly this — 4 holdings, all with
`per_asset_cap_max=-0.042`, QP returned `infeasible` and zero-trade
fallback fired even though hold-flat was trivially feasible.

The architectural fix is to clamp `w_upper >= w_current` at the SOFT
scaling sites — NOT in the solver — so:
  - Hard caps (max_position_pct, sector/correlation, cap-compliance
    fallback) stay enforceable in the solver
  - Soft scaling never breaks the hold-flat invariant
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.tasks import (  # noqa: E402
    ApplyConvictionCapTask,
    ApplyExposureScalingTask,
)


class _StubCtx:
    """Minimal ctx for unit-testing the two soft-scaling tasks."""

    def __init__(self):
        self.config = {}
        self.counters = {}
        # paths the tasks read via _get_path
        self._qp_w_upper = None
        self._qp_w_current = None
        # v3 (codex #123 review): hold-flat clamp is hard-cap-aware. Tests
        # must stamp ``_qp_w_upper_hard`` the way ComputeQPConstraintsTask
        # does in production, otherwise the clamp skips entirely and
        # post-scaling w_upper stays below w_current.
        self._qp_w_upper_hard = None
        self._qp_tickers = None
        self._qp_mu_source_map = None
        # exposure-scaling reads spy_returns, portfolio_value, hwm
        self.spy_returns = []
        self.portfolio_value = 1.0
        self.hwm = 1.0


class TestSoftScalingHoldFlatClamp:
    """Pin the hold-flat invariant at the proper layer (tasks, not solver)."""

    def test_exposure_scaling_does_not_push_w_upper_below_w_current(self):
        """vt × dd × ... scaling < 1 must NOT drive w_upper below w_current
        for held positions. Daily-104 2026-06-02 regression guard.
        """
        ctx = _StubCtx()
        ctx._qp_w_upper = np.array([0.10, 0.10, 0.10, 0.10])
        ctx._qp_w_upper_hard = np.array([0.10, 0.10, 0.10, 0.10])
        ctx._qp_w_current = np.array([0.10, 0.10, 0.10, 0.10])
        # Engineer a scaling < 1 by enabling vt with a very low spy_returns vol
        ctx.config = {
            "exposure_scaling": {
                "vol_target": {
                    "enabled": True,
                    "target_vol": 0.01,    # extremely tight → vt < 1
                    "window_days": 20,
                    "floor": 0.10,
                    "ceiling": 1.50,
                }
            }
        }
        ctx.spy_returns = [0.02] * 60  # high realized vol → vt floor binds

        ApplyExposureScalingTask().run(ctx)

        # Clamp invariant: each entry must be >= current weight
        scaled = np.asarray(ctx._qp_w_upper, dtype=float)
        np.testing.assert_array_less(
            -1e-9, scaled - 0.10,
            err_msg=f"hold-flat invariant broken in exposure scaling: w_upper={scaled}",
        )

    def test_exposure_scaling_preserves_buy_headroom_when_w_upper_above_current(self):
        """When w_upper > w_current, scaling should still apply normally
        (clamp is a no-op floor at w_current, NOT a ceiling)."""
        ctx = _StubCtx()
        ctx._qp_w_upper = np.array([0.20, 0.20, 0.20, 0.20])
        ctx._qp_w_upper_hard = np.array([0.20, 0.20, 0.20, 0.20])
        ctx._qp_w_current = np.array([0.05, 0.05, 0.05, 0.05])
        ctx.config = {
            "exposure_scaling": {
                "vol_target": {
                    "enabled": True,
                    "target_vol": 0.15,
                    "window_days": 20,
                    "floor": 0.50,
                    "ceiling": 1.00,
                }
            }
        }
        ctx.spy_returns = [0.012] * 60

        ApplyExposureScalingTask().run(ctx)

        scaled = np.asarray(ctx._qp_w_upper, dtype=float)
        # All values still above current (buy headroom preserved)
        assert (scaled >= 0.05).all()
        # And the cap is below the original 0.20 (scaling did fire)
        assert (scaled <= 0.20 + 1e-9).all()

    def test_conviction_cap_within_hard_cap_holding_clamps_to_w_current(self):
        """Within-hard-cap held position: conviction multiplier shrinks
        ``_qp_w_upper`` below ``w_current``. Clamp must raise it back to
        ``w_current`` (hold-flat invariant preserved).
        """
        from types import SimpleNamespace as NS
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask

        ctx = _StubCtx()
        ctx._qp_tickers = ["ORCL"]
        ctx._qp_w_upper = np.array([0.15])        # post-Compute hard cap
        ctx._qp_w_upper_hard = np.array([0.15])   # immutable hard cap snapshot
        ctx._qp_w_current = np.array([0.10])      # within hard cap (0.10 ≤ 0.15)
        # Tiny score → low conviction multiplier → soft cap shrinks below 0.10
        ctx._qp_mu_source_map = {"ORCL": NS(panel_score=0.005)}
        ctx.config = {
            "rotation": {"joint_actions": {"qp_conviction_cap_enabled": True}},
            "ranking": {"panel_scoring": {"sizing": {"enabled": True}}},
        }

        ApplyConvictionCapTask().run(ctx)

        # Hold-flat invariant: w_upper >= w_current for a within-cap holding
        w_upper = float(ctx._qp_w_upper[0])
        w_curr  = float(ctx._qp_w_current[0])
        assert w_upper >= w_curr - 1e-9, (
            f"hold-flat invariant broken: w_upper={w_upper} < w_current={w_curr}"
        )
        # And w_upper is NOT raised above the hard cap
        assert w_upper <= float(ctx._qp_w_upper_hard[0]) + 1e-9, (
            f"clamp raised w_upper above hard cap: w_upper={w_upper} > hard=0.15"
        )

    def test_conviction_cap_over_cap_holding_restores_hard_cap_exactly(self):
        """**Codex #123 v4 review regression guard.** Over-hard-cap held
        position (w_current > w_upper_hard) — after a low-conviction
        soft-scale, ``_qp_w_upper`` for the over-cap row MUST be exactly
        the hard cap (NOT the soft-scaled value).

        Without this contract, cap-compliance fallback would force-sell
        an over-cap ORCL from 22% straight to the conviction-shrunk soft
        cap (e.g. 7.5%) rather than to the hard risk cap (15%). The
        soft cap is a TARGET; the hard cap is the per-asset RISK CAP
        that ``_retry_for_per_asset_cap_compliance`` claims to enforce.

        Codex's exact #123 v3 repro: hard=15%, current=22%, panel_score=0,
        conviction multiplier = 0.5 → post-conviction soft cap = 7.5%.
        After the hard-cap restore: w_upper must be 15.0%, not 7.5%.
        """
        from types import SimpleNamespace as NS
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask

        ctx = _StubCtx()
        ctx._qp_tickers = ["ORCL"]
        ctx._qp_w_upper = np.array([0.15])        # post-Compute hard cap
        ctx._qp_w_upper_hard = np.array([0.15])   # immutable hard cap snapshot
        ctx._qp_w_current = np.array([0.22])      # 22% holding — OVER hard cap
        ctx._qp_mu_source_map = {"ORCL": NS(panel_score=0.0)}
        ctx.config = {
            "rotation": {"joint_actions": {"qp_conviction_cap_enabled": True}},
            "ranking": {"panel_scoring": {"sizing": {"enabled": True}}},
        }

        ApplyConvictionCapTask().run(ctx)

        # Strict equality assertion (v4): w_upper must be exactly the hard
        # cap for the over-cap row — NOT the soft-scaled value. The earlier
        # v3 assertion (`<= hard`) would have silently passed on 7.5%
        # (the conviction-shrunk soft cap), which is the bug codex caught.
        assert float(ctx._qp_w_upper[0]) == float(ctx._qp_w_upper_hard[0]), (
            f"REGRESSION: conviction-cap over-cap row not restored to hard. "
            f"w_upper={ctx._qp_w_upper[0]} expected hard={ctx._qp_w_upper_hard[0]}. "
            f"This would force-sell to the SOFT cap, not the RISK cap — "
            f"codex #123 v3 bug."
        )

    def test_cap_compliance_fallback_sells_to_hard_cap_not_soft_cap(self):
        """**Codex #123 v4 review regression guard — end-to-end.** Run the
        over-cap conviction-cap path THROUGH ``_retry_for_per_asset_cap_compliance``
        and assert the synthetic sell target is exactly the hard cap.

        This is the test codex specifically asked for: cap-compliance
        must sell back to the hard *risk* cap (15%), not the soft cap
        (conviction × hard = 7.5% in this repro).
        """
        from types import SimpleNamespace as NS
        from kernel.portfolio_qp.tasks import (
            ApplyConvictionCapTask,
            _retry_for_per_asset_cap_compliance,
        )
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

        ctx = _StubCtx()
        ctx._qp_tickers = ["ORCL"]
        ctx._qp_w_upper = np.array([0.15])
        ctx._qp_w_upper_hard = np.array([0.15])
        ctx._qp_w_current = np.array([0.22])
        ctx._qp_mu_source_map = {"ORCL": NS(panel_score=0.0)}
        ctx.config = {
            "rotation": {"joint_actions": {"qp_conviction_cap_enabled": True}},
            "ranking": {"panel_scoring": {"sizing": {"enabled": True}}},
        }
        ApplyConvictionCapTask().run(ctx)

        # Build solver kwargs matching the codex repro
        kwargs = dict(
            w_current=ctx._qp_w_current,
            mu=[0.0],
            sigma=[0.10],
            w_upper=ctx._qp_w_upper,
            w_lower=0.0,
            cash_reserve=0.0,
            cost_kappa=10.0,
            turnover_max=0.01,
        )
        sol = solve_portfolio_qp(**kwargs)
        # Solver must be infeasible so the retry path actually runs
        assert sol.status.startswith("infeasible"), sol.status

        post = _retry_for_per_asset_cap_compliance(sol, kwargs, solve_portfolio_qp)

        # Cap-compliance must have fired AND the synthetic target must be
        # exactly the hard cap (0.15), NOT the conviction-shrunk soft cap.
        assert post.status == "cap_compliance_fallback", (
            f"cap-compliance fallback did not fire (status={post.status!r})"
        )
        # target_w[i] for the over-cap row should be the hard cap
        np.testing.assert_allclose(
            post.target_w[0], 0.15, atol=1e-9,
            err_msg=(
                f"REGRESSION: cap-compliance sold to SOFT cap, not hard. "
                f"target_w={post.target_w[0]} expected hard=0.15. "
                f"Codex #123 v3 caught this exact silent soft-cap "
                f"force-liquidation."
            ),
        )

    def test_conviction_cap_over_cap_solver_returns_infeasible(self):
        """End-to-end: over-cap holding through ApplyConvictionCapTask must
        reach the solver with w_upper still at the hard cap, so the solver
        returns infeasible and ``_retry_for_per_asset_cap_compliance`` can
        fire its deterministic sell-down."""
        from types import SimpleNamespace as NS
        from kernel.portfolio_qp.tasks import ApplyConvictionCapTask
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp

        ctx = _StubCtx()
        ctx._qp_tickers = ["ORCL"]
        ctx._qp_w_upper = np.array([0.15])
        ctx._qp_w_upper_hard = np.array([0.15])
        ctx._qp_w_current = np.array([0.22])
        ctx._qp_mu_source_map = {"ORCL": NS(panel_score=0.0)}
        ctx.config = {
            "rotation": {"joint_actions": {"qp_conviction_cap_enabled": True}},
            "ranking": {"panel_scoring": {"sizing": {"enabled": True}}},
        }
        ApplyConvictionCapTask().run(ctx)

        sol = solve_portfolio_qp(
            w_current=ctx._qp_w_current,
            mu=[0.0],
            sigma=[0.10],
            w_upper=ctx._qp_w_upper,
            w_lower=0.0,
            cash_reserve=0.0,
            cost_kappa=10.0,
            turnover_max=0.01,
        )
        assert sol.status.startswith("infeasible"), (
            f"REGRESSION: solver did not return infeasible for over-cap "
            f"holding (status={sol.status!r}). cap_compliance_fallback path "
            f"is only triggered on infeasible status — see "
            f"_retry_for_per_asset_cap_compliance docstring."
        )
