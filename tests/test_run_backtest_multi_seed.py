"""Multi-seed harness for run_backtest (CLAUDE.md §5.13.4).

Per "Single performance number = unverified claim" (§5.13.4) any APY /
Sharpe number quoted in a commit / doc / roadmap MUST be ``mean ± std``
from ≥ 5 runs. This file tests:

1. ``run_backtest`` accepts an optional ``seed`` parameter.
2. ``run_backtest_multi_seed`` runs K sims and aggregates them into a
   :class:`MultiSeedSimResult` with sharpe_mean / sharpe_std / DSR / PBO.
3. K=1 is degenerate-but-valid (returns single-seed sim wrapped).
4. K=20 produces DSR < raw Sharpe at observed Sharpe ≈ 1.0.
5. PBO ∈ [0, 1] in multi-seed mode.
6. AUDIT REGRESSION GUARD per §5.13.3 — same K seeds + same fake sim
   functions produce identical aggregates (deterministic harness).
7. ``parallel=True`` produces an equivalent aggregate to sequential.

We avoid spinning up a real backtest (which would need full LEAN data)
by monkey-patching ``run_backtest`` with a synthetic per-seed sim that
produces controlled returns. The harness layer is what's under test.
"""
from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from sim.runner import (  # noqa: E402
    MultiSeedSimResult,
    SimResult,
    _aggregate_perf,
    _apply_seed,
    _compute_action_consistency,
    _resolve_seeds,
    _stack_returns_matrix,
    run_backtest,
    run_backtest_multi_seed,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — synthetic sim factory + monkey-patch fixture.
# ─────────────────────────────────────────────────────────────────────────────


def _make_synthetic_sim_result(
    *,
    seed: int,
    n_days: int = 252,
    sharpe_target: float = 1.0,
    apy_jitter: float = 0.0,
) -> SimResult:
    """Build a synthetic SimResult driven entirely by ``seed``.

    Used by the monkey-patch below so multi-seed tests don't need a real
    InferencePipeline run. Per-seed Sharpes scatter around ``sharpe_target``
    by construction so DSR/PBO have something to chew on.
    """
    rng = np.random.default_rng(seed)
    daily_sigma = 0.01
    daily_mu = sharpe_target * daily_sigma / math.sqrt(252)
    rets = daily_mu + daily_sigma * rng.standard_normal(n_days)
    equity = 100_000 * np.cumprod(1 + rets)
    idx = pd.date_range("2026-01-01", periods=n_days, freq="B")
    eq_df = pd.DataFrame({"portfolio": equity, "regime": "RISK_ON"}, index=idx)
    final = float(equity[-1])
    total_ret = final / 100_000.0 - 1.0
    apy = (1.0 + total_ret) ** (252 / n_days) - 1.0 + apy_jitter
    sharpe = float(rets.mean() / rets.std(ddof=1) * math.sqrt(252))
    # Synthetic trade log — alternating buys/sells per seed-derived rng.
    trade_log = []
    for i in range(0, n_days, 30):
        date = idx[i]
        ticker = "AAPL" if rng.random() > 0.5 else "MSFT"
        action = "buy" if i % 60 == 0 else "sell"
        trade_log.append(
            {"date": str(date.date()), "ticker": ticker, "action": action}
        )
    return SimResult(
        equity_df=eq_df,
        trade_log=trade_log,
        rotation_log=[],
        final_value=final,
        total_return=total_ret,
        apy=apy,
        win_rate=0.5,
        avg_hold=10.0,
        avg_pnl=0.01,
        total_tax=0.0,
        exit_reasons={},
        rotations=[],
        sharpe=sharpe,
        sortino=sharpe * 1.2,
        calmar=sharpe * 0.5,
        max_dd=0.10,
        ann_vol=0.15,
    )


@pytest.fixture
def patch_run_backtest(monkeypatch):
    """Replace ``run_backtest`` inside ``sim.runner`` with the synthetic
    factory. The harness's ``_run_one_seed`` calls module-level
    ``run_backtest``, so monkey-patching the module attribute is enough."""
    from sim import runner as _runner

    captured: list[int] = []

    def _fake(*, seed=None, **_kwargs) -> SimResult:
        # Record observed seed so reproducibility tests can verify
        # the harness threaded the seed through correctly.
        captured.append(int(seed) if seed is not None else -1)
        return _make_synthetic_sim_result(seed=seed if seed is not None else 0)

    monkeypatch.setattr(_runner, "run_backtest", _fake)
    return captured


# ─────────────────────────────────────────────────────────────────────────────
# C4.1 — run_backtest accepts seed param.
# ─────────────────────────────────────────────────────────────────────────────


class TestRunBacktestSeedSignature:
    def test_seed_param_present_with_optional_int_default_none(self):
        sig = inspect.signature(run_backtest)
        assert "seed" in sig.parameters
        param = sig.parameters["seed"]
        assert param.default is None, \
            "seed default must be None to preserve legacy non-deterministic call sites"

    def test_apply_seed_idempotent_on_none(self):
        # Calling with None must not raise and must not pin globals.
        _apply_seed(None)
        _apply_seed(None)  # second call equally OK

    def test_apply_seed_pins_numpy_rng(self):
        _apply_seed(42)
        a = np.random.random(5)
        _apply_seed(42)
        b = np.random.random(5)
        np.testing.assert_array_equal(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# C4.2 — _resolve_seeds normalization.
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveSeeds:
    def test_int_K_expands_to_range(self):
        assert _resolve_seeds(5) == [0, 1, 2, 3, 4]

    def test_explicit_list_used_as_is(self):
        assert _resolve_seeds([7, 13, 42]) == [7, 13, 42]

    def test_negative_K_raises(self):
        with pytest.raises(ValueError):
            _resolve_seeds(0)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            _resolve_seeds([])

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            _resolve_seeds("not a list")  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# C4.3 — run_backtest_multi_seed K=1, K=5, K=20.
# ─────────────────────────────────────────────────────────────────────────────


def _multi_seed_kwargs(seeds, parallel=False):
    """Minimal kwargs — the monkey-patched run_backtest ignores them."""
    return dict(
        seeds=seeds,
        parallel=parallel,
        config={},
        strategy_dir=Path("/tmp/dummy"),
        ohlcv={},
        spy_df=pd.DataFrame(),
        sector_etf_map={},
    )


class TestMultiSeedHarnessShape:
    def test_K1_reduces_to_single_sim(self, patch_run_backtest):
        msr = run_backtest_multi_seed(**_multi_seed_kwargs(seeds=1))
        assert isinstance(msr, MultiSeedSimResult)
        assert msr.n_seeds == 1
        assert msr.seeds == [0]
        # K=1 → no PBO and DSR can come from single-seed compute_perf_triple
        assert math.isnan(msr.pbo), "PBO must be NaN at K=1"
        # sharpe_std needs ≥ 2 data points, so it's NaN at K=1 — see
        # _aggregate_perf using ddof=1.
        assert math.isnan(msr.sharpe_std)
        assert math.isfinite(msr.sharpe_mean)

    def test_K5_produces_mean_and_std(self, patch_run_backtest):
        msr = run_backtest_multi_seed(**_multi_seed_kwargs(seeds=5))
        assert msr.n_seeds == 5
        assert math.isfinite(msr.sharpe_mean)
        assert math.isfinite(msr.sharpe_std), \
            "K=5 must produce finite sharpe_std (mean ± std required by §5.13.4)"
        assert math.isfinite(msr.apy_mean)
        assert math.isfinite(msr.apy_std)

    def test_K20_dsr_drops_below_observed_sharpe(self, patch_run_backtest):
        """At observed annual Sharpe ≈ 1.0 across K=20 seeds, the
        ensemble-deflated DSR must be visibly below 1.0 — proving the
        selection-bias deflator fires (CLAUDE.md §5.13.4)."""
        msr = run_backtest_multi_seed(**_multi_seed_kwargs(seeds=20))
        assert msr.n_seeds == 20
        assert math.isfinite(msr.dsr)
        # Headline (seed=0) Sharpe is approximately 1.0 by construction;
        # DSR with n_trials=20 should land < 1.0 (i.e. less-than-certain
        # of true edge).
        assert msr.dsr < 1.0, (
            f"DSR={msr.dsr:.4f} >= 1.0 — selection-bias deflator silently "
            f"muted at K=20; §5.13.4 not enforced"
        )

    def test_pbo_in_unit_interval(self, patch_run_backtest):
        msr = run_backtest_multi_seed(**_multi_seed_kwargs(seeds=10))
        assert math.isfinite(msr.pbo)
        assert 0.0 <= msr.pbo <= 1.0, \
            f"PBO must lie in [0, 1], got {msr.pbo}"


# ─────────────────────────────────────────────────────────────────────────────
# C4.4 — §5.13.3 AUDIT REGRESSION GUARD: deterministic harness.
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiSeedReproducibility:
    """§5.13.3 — every fix names its class-of-bug invariant. Here the
    invariant is: same seeds + same per-seed sim function → same aggregate.
    A future regression that, e.g., shuffles seeds before dispatch or
    introduces a wallclock-dependent step would break this test."""

    def test_same_seeds_produce_same_aggregate(self, patch_run_backtest):
        msr_a = run_backtest_multi_seed(
            **_multi_seed_kwargs(seeds=[1, 2, 3, 4, 5]))
        msr_b = run_backtest_multi_seed(
            **_multi_seed_kwargs(seeds=[1, 2, 3, 4, 5]))
        # All aggregate floats must match bit-for-bit (or both be NaN).
        for field in ("sharpe_mean", "sharpe_std", "apy_mean", "apy_std",
                      "dsr", "pbo", "majority_vote_action_consistency"):
            a = getattr(msr_a, field)
            b = getattr(msr_b, field)
            if math.isnan(a) and math.isnan(b):
                continue
            assert a == b, f"{field}: {a} != {b} — harness is non-deterministic"


# ─────────────────────────────────────────────────────────────────────────────
# C4.5 — parallel=True equivalent to parallel=False (just faster).
# ─────────────────────────────────────────────────────────────────────────────


class TestParallelEquivalent:
    def test_parallel_matches_sequential_aggregate(self, patch_run_backtest):
        seq = run_backtest_multi_seed(**_multi_seed_kwargs(seeds=[10, 20, 30, 40, 50]))
        par = run_backtest_multi_seed(**_multi_seed_kwargs(
            seeds=[10, 20, 30, 40, 50], parallel=True))
        # Same per-seed inputs → same aggregates regardless of dispatch.
        for field in ("sharpe_mean", "sharpe_std", "apy_mean", "apy_std",
                      "dsr", "pbo"):
            a, b = getattr(seq, field), getattr(par, field)
            if math.isnan(a) and math.isnan(b):
                continue
            assert a == b, f"{field}: seq={a} par={b}"


# ─────────────────────────────────────────────────────────────────────────────
# C4.6 — _stack_returns_matrix + _compute_action_consistency edge cases.
# ─────────────────────────────────────────────────────────────────────────────


class TestStackingHelpers:
    def test_stack_returns_matrix_with_K1_returns_none(self):
        r = _make_synthetic_sim_result(seed=0)
        assert _stack_returns_matrix([r]) is None

    def test_stack_returns_matrix_K3_shape(self):
        results = [_make_synthetic_sim_result(seed=s) for s in range(3)]
        m = _stack_returns_matrix(results)
        assert m is not None
        # All synthetic sims share the same date index → no truncation.
        assert m.shape == (251, 3)  # 252 days minus first NaN from pct_change

    def test_action_consistency_K1_is_nan(self):
        r = _make_synthetic_sim_result(seed=0)
        assert math.isnan(_compute_action_consistency([r]))

    def test_action_consistency_in_unit_interval(self):
        results = [_make_synthetic_sim_result(seed=s) for s in range(5)]
        c = _compute_action_consistency(results)
        if math.isfinite(c):
            assert 0.0 <= c <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# C4.7 — MultiSeedSimResult.headline returns first per-seed result.
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiSeedSimResultHeadline:
    def test_headline_is_first_per_seed_result(self, patch_run_backtest):
        msr = run_backtest_multi_seed(**_multi_seed_kwargs(seeds=[3, 7, 11]))
        assert msr.headline is msr.per_seed_results[0]

    def test_headline_raises_on_empty(self):
        empty = MultiSeedSimResult(per_seed_results=[], seeds=[])
        with pytest.raises(RuntimeError):
            _ = empty.headline
