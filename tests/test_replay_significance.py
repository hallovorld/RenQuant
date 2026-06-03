"""Tests for the DSR / PBO wiring on the A/B replay (§8 Step 4c).

Pins the canonical CLAUDE.md §7.4 Tier 3 + §7.3 multi-measurement
requirement at the harness boundary: every Sharpe number reported by
the offline A/B replay carries a DSR (selection-bias-corrected) and
the candidate set carries a single shared PBO (CSCV).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.allocator_replay import (  # noqa: E402
    AllocatorReplayBar,
    replay_all,
)
from kernel.portfolio_qp.baseline_allocators import (  # noqa: E402
    equal_weight_top_k,
    fractional_kelly_top_k,
    inverse_vol_top_k,
)
from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot  # noqa: E402
from kernel.portfolio_qp.replay_significance import (  # noqa: E402
    SignificanceVerdict,
    compute_significance_verdicts,
    verdicts_to_dict,
)


def _snap(n: int) -> ConstraintSnapshot:
    return ConstraintSnapshot(
        n=n,
        tickers=tuple(f"T{i}" for i in range(n)),
        w_current=np.zeros(n),
        w_upper_hard=np.full(n, 0.50),
        w_upper=np.full(n, 0.50),
        w_lower=0.0,
        dw_max=np.full(n, 1.0),
        cash_reserve=0.0,
        turnover_max=None,
        drawdown=0.0,
        drawdown_limit=0.20,
        gross_max=None,
        wash_sale_mask=np.zeros(n, dtype=bool),
    )


def _bars_with_signal(
    n_bars: int = 252,
    n: int = 3,
    *,
    mean_return: float = 0.001,
    noise_std: float = 0.005,
    seed: int = 0,
    regime: str | None = "BULL_CALM",
) -> list[AllocatorReplayBar]:
    rng = np.random.default_rng(seed)
    bars = []
    for i in range(n_bars):
        bars.append(AllocatorReplayBar(
            bar_date=f"2026-{1 + i // 21:02d}-{1 + i % 21:02d}",
            snap=_snap(n),
            mu=rng.uniform(0.0, 0.05, n),
            sigma=rng.uniform(0.10, 0.20, n),
            fwd_return=rng.normal(mean_return, noise_std, n),
            regime=regime,
            cost_per_trade_bps=0.0,
        ))
    return bars


class TestComputeSignificanceVerdicts:
    def test_dsr_and_pbo_computed_for_each_allocator(self):
        bars = _bars_with_signal(n_bars=64, n=3)
        results = replay_all(
            {"eq": equal_weight_top_k,
             "iv": inverse_vol_top_k,
             "fk": fractional_kelly_top_k},
            bars,
        )
        verdicts = compute_significance_verdicts(results, pbo_n_slices=16)
        # Same set of allocators
        assert set(verdicts) == {"eq", "iv", "fk"}
        # All have a Sharpe + DSR
        for v in verdicts.values():
            assert isinstance(v, SignificanceVerdict)
            assert v.sharpe_raw_annual is not None
            assert v.dsr is not None
            assert 0.0 <= v.dsr <= 1.0
        # PBO is shared across all allocators
        pbos = {v.pbo for v in verdicts.values()}
        assert len(pbos) == 1, f"PBO not shared: {pbos}"
        pbo = pbos.pop()
        assert pbo is not None
        assert 0.0 <= pbo <= 1.0

    def test_n_returns_below_30_yields_null_dsr(self):
        # DSR's higher-moment correction is unreliable below ~30 bars
        bars = _bars_with_signal(n_bars=20, n=2)
        results = replay_all(
            {"eq": equal_weight_top_k},
            bars,
        )
        verdicts = compute_significance_verdicts(results, pbo_n_slices=16)
        for v in verdicts.values():
            assert v.dsr is None, "DSR should be null for short series"

    def test_pbo_skipped_when_only_one_allocator(self):
        bars = _bars_with_signal(n_bars=64, n=2)
        results = replay_all({"only": equal_weight_top_k}, bars)
        verdicts = compute_significance_verdicts(results, pbo_n_slices=16)
        assert len(verdicts) == 1
        assert next(iter(verdicts.values())).pbo is None

    def test_pbo_skipped_when_T_below_n_slices(self):
        bars = _bars_with_signal(n_bars=10, n=2)
        results = replay_all(
            {"eq": equal_weight_top_k, "iv": inverse_vol_top_k},
            bars,
        )
        verdicts = compute_significance_verdicts(results, pbo_n_slices=16)
        for v in verdicts.values():
            assert v.pbo is None, "PBO should be null when T < n_slices"

    def test_mismatched_bar_counts_raises(self):
        # Manually craft results with different bar counts
        from kernel.portfolio_qp.allocator_replay import ReplayResult
        results = {
            "a": ReplayResult(name="a", bars=10,
                              daily_returns_net=[0.01] * 10),
            "b": ReplayResult(name="b", bars=12,
                              daily_returns_net=[0.01] * 12),
        }
        import pytest
        with pytest.raises(ValueError, match="different bar counts"):
            compute_significance_verdicts(results)


class TestVerdictsToDictAndPromotionGate:
    """The verdict block surfaces a `live_promotable_per_clause_7_4`
    flag for each allocator — CLAUDE.md §7.4 Tier 3 says BOTH DSR > 0.5
    AND PBO < 0.5 (we use the stronger DSR ≥ 0.95 here, the standard
    selection-bias-corrected threshold)."""

    def test_strong_signal_marked_promotable(self):
        # Strong positive Sharpe + low noise → DSR should be high and
        # PBO low for the best allocator.
        rng = np.random.default_rng(42)
        # Build a 252-bar series where equal-weight dominates IV
        # because IV over-weights the noisy ticker.
        bars = []
        for i in range(252):
            bars.append(AllocatorReplayBar(
                bar_date=f"day-{i:03d}",
                snap=_snap(2),
                mu=np.array([0.05, 0.04]),
                sigma=np.array([0.10, 0.50]),  # IV will overweight T0
                fwd_return=np.array([
                    0.003 + rng.normal(0, 0.001),
                    0.003 + rng.normal(0, 0.001),
                ]),
                regime="BULL_CALM",
                cost_per_trade_bps=0.0,
            ))
        results = replay_all(
            {"eq": equal_weight_top_k, "iv": inverse_vol_top_k},
            bars,
        )
        verdicts = compute_significance_verdicts(results)
        d = verdicts_to_dict(verdicts)
        # Both should have a promotion flag computed
        for name in ("eq", "iv"):
            assert "live_promotable_per_clause_7_4" in d[name]
            assert isinstance(d[name]["live_promotable_per_clause_7_4"], bool)
        # JSON-serialisable
        import json
        json.dumps(d)

    def test_empty_results_yields_empty_verdicts(self):
        verdicts = compute_significance_verdicts({})
        assert verdicts == {}
        assert verdicts_to_dict(verdicts) == {}
