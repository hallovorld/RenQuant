"""Tests for kernel.metrics.pbo (CSCV).

Reference: Bailey, Borwein, López de Prado, Zhu 2017,
"The Probability of Backtest Overfitting", JCF.

Spec from track-M2 prompt:
  - PBO ≈ 0.5 for random predictors
  - PBO < 0.3 for genuine signal
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from renquant_common.metrics.pbo import (  # noqa: E402
    _logit_omega,
    _slice_sharpe,
    _split_indices,
    probability_of_backtest_overfitting,
)


class TestSplitIndices:
    def test_even_split(self):
        slices = _split_indices(100, 10)
        assert len(slices) == 10
        assert sum(s.size for s in slices) == 100
        assert all(s.size == 10 for s in slices)

    def test_uneven_distributes_remainder(self):
        slices = _split_indices(103, 10)
        sizes = [s.size for s in slices]
        assert sum(sizes) == 103
        # max-min difference at most 1
        assert max(sizes) - min(sizes) <= 1


class TestSliceSharpe:
    def test_shape(self):
        rng = np.random.default_rng(0)
        M = rng.normal(size=(40, 8))
        sr = _slice_sharpe(M)
        assert sr.shape == (8,)

    def test_zero_std_yields_nan(self):
        M = np.zeros((20, 3))
        M[:, 1] = 1.0  # constant column
        sr = _slice_sharpe(M)
        assert np.isnan(sr).all()


class TestLogitOmega:
    def test_median_rank_is_zero(self):
        # For N=99, median rank is 50; ω̄ = 50/100 = 0.5 → λ = 0.
        assert _logit_omega(50, 99) == pytest.approx(0.0, abs=1e-12)

    def test_top_rank_positive(self):
        assert _logit_omega(99, 99) > 0


class TestPBORandom:
    def test_random_strategies_pbo_around_half_in_expectation(self):
        """Pure noise → PBO close to 0.5 averaged across seeds.

        Per-seed PBO has substantial variance at T=256/S=8 (range 0.15..0.75
        observed); the population value is what's centered on 0.5.
        """
        per_seed_pbo = []
        for seed in (1, 7, 42, 100, 200, 2026):
            rng = np.random.default_rng(seed)
            M = rng.normal(0.0, 0.01, size=(256, 20))
            per_seed_pbo.append(
                probability_of_backtest_overfitting(M, n_slices=8)
            )
        mean_pbo = float(np.mean(per_seed_pbo))
        # Mean over 6 seeds should land near 0.5 (±0.2 tolerance).
        assert 0.3 <= mean_pbo <= 0.7, (
            f"mean PBO={mean_pbo:.3f} per_seed={per_seed_pbo}"
        )

    def test_signal_pbo_strictly_lower_than_noise_pbo(self):
        """Paired sanity: same seed/shape, adding genuine drift to one
        column must reduce PBO."""
        rng_noise = np.random.default_rng(99)
        rng_signal = np.random.default_rng(99)
        T, N = 384, 12
        M_noise = rng_noise.normal(0.0, 0.01, size=(T, N))
        M_signal = rng_signal.normal(0.0, 0.01, size=(T, N))
        M_signal[:, 0] += 0.005  # +50 bps daily on strategy 0
        pbo_noise = probability_of_backtest_overfitting(M_noise, n_slices=8)
        pbo_signal = probability_of_backtest_overfitting(M_signal, n_slices=8)
        assert pbo_signal < pbo_noise, (
            f"signal pbo={pbo_signal:.3f} not below noise pbo={pbo_noise:.3f}"
        )


class TestPBOSignal:
    def test_one_genuinely_skilled_strategy_pbo_low(self):
        """If one strategy has a real edge, PBO should drop below 0.3."""
        rng = np.random.default_rng(7)
        T, N = 512, 16
        M = rng.normal(0.0, 0.01, size=(T, N))
        # Strategy 0 has consistent +5 bps daily drift on top of noise.
        M[:, 0] += 0.005
        pbo = probability_of_backtest_overfitting(M, n_slices=8)
        assert pbo < 0.3, f"genuine signal should give pbo<0.3, got {pbo}"


class TestPBOValidation:
    def test_rejects_single_strategy(self):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.zeros((100, 1)), n_slices=8)

    def test_rejects_odd_slices(self):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(
                np.random.default_rng(0).normal(size=(100, 5)), n_slices=7
            )

    def test_rejects_too_few_rows(self):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(
                np.random.default_rng(0).normal(size=(8, 5)), n_slices=16
            )

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.array([0.1, 0.2, 0.3]))

    def test_max_combinations_subsampling(self):
        rng = np.random.default_rng(11)
        M = rng.normal(size=(256, 10))
        # n_slices=16 -> C(16,8)=12870 combos; subsample 200.
        pbo_full = probability_of_backtest_overfitting(M, n_slices=16)
        pbo_sub = probability_of_backtest_overfitting(
            M, n_slices=16, max_combinations=200, rng=np.random.default_rng(11),
        )
        # Sub-sample estimate should be in the same ballpark.
        assert abs(pbo_sub - pbo_full) < 0.15
