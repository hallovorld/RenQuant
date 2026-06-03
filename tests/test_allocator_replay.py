"""Tests for the A/B replay harness (§8 Step 4b).

The harness math is pinned independently of the production WF cut
loader; tests use synthetic snapshots so the metric math
(Sharpe, MDD, turnover, per-regime split, paired daily returns) is
verifiable without artifact dependencies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.allocator_replay import (  # noqa: E402
    AllocatorReplayBar,
    paired_daily_returns,
    replay_all,
    replay_one_allocator,
)
from kernel.portfolio_qp.baseline_allocators import (  # noqa: E402
    equal_weight_top_k,
    fractional_kelly_top_k,
    inverse_vol_top_k,
)
from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot  # noqa: E402


def _snap(n: int, *, w_upper_hard=None) -> ConstraintSnapshot:
    # Permissive defaults so the per-bar mechanic tests below isolate
    # what they want to test (Sharpe / MDD / turnover accounting /
    # no-candidate flow). Constraint-enforcement tests are in
    # TestSnapshotFeasibilityValidator.
    return ConstraintSnapshot(
        n=n,
        tickers=tuple(f"T{i}" for i in range(n)),
        w_current=np.zeros(n),
        w_upper_hard=np.asarray(
            w_upper_hard if w_upper_hard is not None else np.full(n, 0.50),
            dtype=float,
        ),
        w_upper=np.asarray(
            w_upper_hard if w_upper_hard is not None else np.full(n, 0.50),
            dtype=float,
        ),
        w_lower=0.0,
        dw_max=np.full(n, 1.0),     # permissive
        cash_reserve=0.0,
        turnover_max=None,          # permissive
        drawdown=0.0,
        drawdown_limit=0.20,
        gross_max=None,
        wash_sale_mask=np.zeros(n, dtype=bool),
    )


def _bar(
    date: str,
    *,
    n: int,
    mu,
    sigma,
    fwd_return,
    regime: str | None = None,
    cost_bps: float = 0.0,
    w_upper_hard=None,
) -> AllocatorReplayBar:
    return AllocatorReplayBar(
        bar_date=date,
        snap=_snap(n, w_upper_hard=w_upper_hard),
        mu=np.asarray(mu, dtype=float),
        sigma=np.asarray(sigma, dtype=float),
        fwd_return=np.asarray(fwd_return, dtype=float),
        regime=regime,
        cost_per_trade_bps=cost_bps,
    )


class TestReplaySingleAllocatorMetrics:
    def test_constant_positive_return_yields_positive_sharpe(self):
        # 252 bars, each day +0.5% return on top-2 names → annualised
        # Sharpe should be very large (zero noise → 1/std=∞)
        # We use very-low-noise non-zero std to make Sharpe finite.
        rng = np.random.default_rng(0)
        bars = []
        for i in range(252):
            r = 0.005 + rng.normal(0, 0.0001)
            bars.append(_bar(
                f"2026-{1 + i // 21:02d}-{1 + i % 21:02d}",
                n=3,
                mu=[0.05, 0.04, 0.01],
                sigma=[0.10, 0.10, 0.10],
                fwd_return=[r, r, 0.0],
                regime="BULL_CALM",
            ))

        res = replay_one_allocator("eq", equal_weight_top_k, bars)
        assert res.bars == 252
        assert res.sharpe_annual is not None
        assert res.sharpe_annual > 5.0   # strong signal + low noise
        assert res.mean_daily_return > 0.0
        assert res.cumulative_return > 1.0   # >100% over 252 bars

    def test_no_candidate_bars_count_as_zero_return(self):
        bars = [
            _bar("2026-01-01", n=2, mu=[-0.01, -0.02], sigma=[0.1, 0.1],
                 fwd_return=[0.05, 0.05]),
            _bar("2026-01-02", n=2, mu=[0.05, 0.04], sigma=[0.1, 0.1],
                 fwd_return=[0.01, 0.01]),
        ]
        res = replay_one_allocator("eq", equal_weight_top_k, bars)
        assert res.fallback_to_no_candidates == 1
        # First bar: 0% return; second bar: 0.5·1% + 0.5·1% = 1% gross
        assert abs(res.daily_returns_net[0]) < 1e-9
        assert abs(res.daily_returns_net[1] - 0.01) < 1e-6

    def test_turnover_accounted_in_net_return(self):
        # cost_bps=10 → 10bp on |Δw|. Equal-weight top-2 → Δw = [0.5, 0.5],
        # turnover = 1.0, cost = 1.0 × 10bp = 10bp = 0.001.
        bar = _bar(
            "2026-01-01", n=2,
            mu=[0.05, 0.04], sigma=[0.10, 0.10],
            fwd_return=[0.01, 0.01],
            cost_bps=10.0,
        )
        res = replay_one_allocator("eq", equal_weight_top_k, [bar])
        # Gross: 0.5 × 0.01 + 0.5 × 0.01 = 0.01; cost: 0.001 → net 0.009
        assert abs(res.daily_returns_net[0] - 0.009) < 1e-9
        assert abs(res.mean_turnover - 1.0) < 1e-9

    def test_max_drawdown_negative(self):
        # 3 bars: +5%, -10%, +5% → equity 1.05, 0.945, 0.99225
        # MDD: from 1.05 peak to 0.945 = -10%
        bars = [
            _bar(f"2026-01-0{i+1}", n=1, mu=[0.05], sigma=[0.10],
                 fwd_return=[r], w_upper_hard=[1.0])
            for i, r in enumerate([0.05, -0.10, 0.05])
        ]
        res = replay_one_allocator("eq", equal_weight_top_k, bars)
        assert res.max_drawdown < -0.09  # ~-10%
        assert res.max_drawdown >= -0.11


class TestReplayAllAllocators:
    def test_all_three_baselines_run(self):
        rng = np.random.default_rng(42)
        bars = []
        for i in range(60):
            bars.append(_bar(
                f"2026-{1 + i // 21:02d}-{1 + i % 21:02d}",
                n=5,
                mu=rng.uniform(0.0, 0.05, 5),
                sigma=rng.uniform(0.10, 0.20, 5),
                fwd_return=rng.normal(0.001, 0.01, 5),
                regime="BULL_CALM",
            ))

        results = replay_all(
            {
                "equal_weight": equal_weight_top_k,
                "inverse_vol": inverse_vol_top_k,
                "fractional_kelly": fractional_kelly_top_k,
            },
            bars,
        )
        assert set(results) == {"equal_weight", "inverse_vol", "fractional_kelly"}
        for name, r in results.items():
            assert r.bars == 60, name
            assert len(r.daily_returns_net) == 60
            assert r.sharpe_annual is not None
            # Per-regime tracking populated
            assert "BULL_CALM" in r.per_regime
            assert len(r.per_regime["BULL_CALM"]) == 60

    def test_paired_daily_returns_aligned_by_bar(self):
        bars = []
        for i in range(10):
            bars.append(_bar(
                f"2026-01-{i+1:02d}",
                n=3,
                mu=[0.05, 0.04, 0.03],
                sigma=[0.10, 0.15, 0.20],
                fwd_return=[0.01, 0.02, 0.005],
            ))

        results = replay_all(
            {"eq": equal_weight_top_k, "iv": inverse_vol_top_k},
            bars,
        )
        paired = paired_daily_returns(results)
        assert set(paired) == {"eq", "iv"}
        assert len(paired["eq"]) == 10 and len(paired["iv"]) == 10
        # Different allocators produce different returns on the same bars
        # (inverse-vol overweights the low-σ name with the smaller fwd_return)
        assert not np.allclose(paired["eq"], paired["iv"])

    def test_per_regime_stratification(self):
        # Mixed-regime sequence
        bars = []
        for i in range(20):
            regime = "BULL_CALM" if i < 10 else "BULL_VOLATILE"
            bars.append(_bar(
                f"2026-01-{i+1:02d}",
                n=2,
                mu=[0.05, 0.04],
                sigma=[0.10, 0.10],
                fwd_return=[0.005, 0.005],
                regime=regime,
            ))
        res = replay_one_allocator("eq", equal_weight_top_k, bars)
        assert "BULL_CALM" in res.per_regime
        assert "BULL_VOLATILE" in res.per_regime
        assert len(res.per_regime["BULL_CALM"]) == 10
        assert len(res.per_regime["BULL_VOLATILE"]) == 10
        # to_dict serialisable
        d = res.to_dict()
        assert d["per_regime_n_bars"] == {"BULL_CALM": 10, "BULL_VOLATILE": 10}
        assert "per_regime_sharpe" in d


class TestReplayResultSerialisation:
    def test_to_dict_is_json_serialisable(self):
        import json
        bars = [
            _bar(f"2026-01-{i+1:02d}", n=2,
                 mu=[0.05, 0.04], sigma=[0.10, 0.10],
                 fwd_return=[0.01, 0.01], regime="BULL_CALM")
            for i in range(5)
        ]
        res = replay_one_allocator("eq", equal_weight_top_k, bars)
        d = res.to_dict()
        # Round-trip through JSON
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped["bars"] == 5
        assert round_tripped["name"] == "eq"

    def test_metrics_are_zero_when_no_bars(self):
        from kernel.portfolio_qp.allocator_replay import ReplayResult
        res = ReplayResult(name="empty", bars=0)
        assert res.sharpe_annual is None
        assert res.mean_daily_return == 0.0
        assert res.cumulative_return == 0.0
        assert res.max_drawdown == 0.0
        assert res.mean_turnover == 0.0


class TestSnapshotFeasibilityValidator:
    """**Codex #131 review HIGH-1 regression guard.** The replay
    harness must validate every allocator output against the FULL
    ConstraintSnapshot hard-constraint set and tally per-family
    violations. Step 4's gate is zero hard-constraint regressions
    vs the snapshot.
    """

    def test_check_detects_each_family(self):
        from kernel.portfolio_qp.allocator_replay import check_snapshot_feasibility
        from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot

        snap = ConstraintSnapshot(
            n=2,
            tickers=("A", "B"),
            w_current=np.array([0.0, 0.0]),
            w_upper_hard=np.array([0.30, 0.30]),
            w_upper=np.array([0.30, 0.30]),
            w_lower=0.0,
            dw_max=np.array([0.05, 0.05]),
            cash_reserve=0.0,
            turnover_max=0.05,
            drawdown=0.0,
            drawdown_limit=0.20,
            gross_max=0.20,
            wash_sale_mask=np.array([False, True]),
            sector_indicator=np.array([[1.0, 1.0]]),
            sector_cap_vec=np.array([0.20]),
            sector_names=("Tech",),
            corr_group_pairs=((0, 1, 0.20),),
        )
        target = np.array([0.50, 0.50])
        delta = target - snap.w_current
        fam = check_snapshot_feasibility(snap, target, delta)
        # Every family violated by target=[0.5,0.5] must fire
        assert fam["w_upper_hard"] == 1, fam
        assert fam["wash_sale"] == 1, fam
        assert fam["dw_max"] == 1, fam
        assert fam["turnover_max"] == 1, fam
        assert fam["sector_cap"] == 1, fam
        assert fam["corr_group_cap"] == 1, fam
        assert fam["gross_max"] == 1, fam

    def test_replay_counts_per_family_violations(self):
        """A deliberately-cheating allocator returns target=[0.5,0.5]
        on a snap with hard cap 0.20; replay must count the violations
        per family.
        """
        from kernel.portfolio_qp.baseline_allocators import AllocatorResult
        from kernel.portfolio_qp.allocator_replay import replay_one_allocator
        from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot

        def cheating(snap, *, mu, sigma=None):
            target = np.array([0.50, 0.50])
            return AllocatorResult(
                delta_w=target - snap.w_current,
                target_w=target,
                status="optimal",
                selected_indices=(0, 1),
            )

        snap = ConstraintSnapshot(
            n=2, tickers=("A", "B"),
            w_current=np.zeros(2),
            w_upper_hard=np.full(2, 0.20),
            w_upper=np.full(2, 0.20),
            w_lower=0.0,
            dw_max=np.array([0.05, 0.05]),
            cash_reserve=0.0,
            turnover_max=0.05,
            gross_max=None,
            drawdown=0.0,
            drawdown_limit=0.20,
            wash_sale_mask=np.zeros(2, dtype=bool),
        )
        bar = AllocatorReplayBar(
            bar_date="2026-01-01", snap=snap,
            mu=np.array([0.05, 0.04]),
            sigma=np.array([0.10, 0.10]),
            fwd_return=np.array([0.01, 0.01]),
            regime="BULL_CALM",
            cost_per_trade_bps=0.0,
        )
        res = replay_one_allocator("cheating", cheating, [bar])
        assert res.violations_per_family.get("w_upper_hard", 0) == 1
        assert res.violations_per_family.get("dw_max", 0) == 1
        assert res.violations_per_family.get("turnover_max", 0) == 1
        assert res.total_violations() >= 3
        assert res.cap_violations == 1

    def test_feasible_allocator_zero_violations(self):
        """An allocator that fully respects the snapshot must report
        zero violations per family."""
        from kernel.portfolio_qp.baseline_allocators import equal_weight_top_k
        from kernel.portfolio_qp.allocator_replay import replay_one_allocator
        from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot

        snap = ConstraintSnapshot(
            n=3, tickers=("A", "B", "C"),
            w_current=np.zeros(3),
            w_upper_hard=np.full(3, 0.50),
            w_upper=np.full(3, 0.50),
            w_lower=0.0,
            dw_max=np.full(3, 0.50),
            cash_reserve=0.0,
            turnover_max=None,  # permissive
            drawdown=0.0,
            drawdown_limit=0.20,
            gross_max=None,
            wash_sale_mask=np.zeros(3, dtype=bool),
        )
        bar = AllocatorReplayBar(
            bar_date="2026-01-01", snap=snap,
            mu=np.array([0.05, 0.04, 0.03]),
            sigma=np.array([0.10, 0.10, 0.10]),
            fwd_return=np.array([0.01, 0.01, 0.01]),
            regime="BULL_CALM",
            cost_per_trade_bps=0.0,
        )
        res = replay_one_allocator("eq", equal_weight_top_k, [bar])
        assert res.cap_violations == 0
        assert res.total_violations() == 0


class TestNoCandidatesAccountingFix:
    """**Codex #131 review HIGH-2 regression guard.** The allocator's
    returned ``target_w`` / ``delta_w`` ARE the action chosen; replay
    must honour them. Previously ``no_candidates`` short-circuited to
    zero return / zero turnover, silently discarding the liquidation
    cost when going to cash from a non-zero book.
    """

    def test_no_candidates_from_held_book_counts_turnover_and_cost(self):
        from kernel.portfolio_qp.baseline_allocators import equal_weight_top_k
        from kernel.portfolio_qp.allocator_replay import replay_one_allocator
        from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot

        # Held book [0.4, 0.3], all μ̂ negative → allocator returns
        # target=[0, 0] (liquidate). Replay must charge the 7bp cost.
        snap = ConstraintSnapshot(
            n=2, tickers=("A", "B"),
            w_current=np.array([0.40, 0.30]),
            w_upper_hard=np.full(2, 0.50),
            w_upper=np.full(2, 0.50),
            w_lower=0.0,
            dw_max=np.full(2, 0.50),
            cash_reserve=0.0,
            turnover_max=None,
            gross_max=None,
            drawdown=0.0,
            drawdown_limit=0.20,
            wash_sale_mask=np.zeros(2, dtype=bool),
        )
        bar = AllocatorReplayBar(
            bar_date="2026-01-01", snap=snap,
            mu=np.array([-0.01, -0.02]),  # no positive μ̂
            sigma=np.array([0.10, 0.10]),
            fwd_return=np.array([0.01, 0.01]),
            regime="BULL_CALM",
            cost_per_trade_bps=10.0,
        )
        res = replay_one_allocator("eq", equal_weight_top_k, [bar])
        assert res.fallback_to_no_candidates == 1
        # Turnover IS the L1 of the actual delta_w (liquidate 0.7 book)
        assert abs(res.turnover[0] - 0.7) < 1e-9, (
            f"turnover did not account for liquidation: {res.turnover[0]}"
        )
        # Net return: target=[0,0] → gross=0; cost = 0.7 × 10bp = 0.0007
        assert abs(res.daily_returns_net[0] - (-0.0007)) < 1e-9, (
            f"liquidation cost not deducted: {res.daily_returns_net[0]}"
        )
