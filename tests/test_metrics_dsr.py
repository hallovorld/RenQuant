"""Tests for kernel.metrics.deflated_sharpe.

Reference: Bailey & López de Prado 2014, SSRN 2460551.

Includes a §5.13.4 AUDIT REGRESSION GUARD: at observed_sharpe=1.0 with
n_trials > 50, DSR must be < 1.0 (single-number claims are forbidden).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.metrics.deflated_sharpe import (  # noqa: E402
    EULER_MASCHERONI,
    annualized_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    sharpe_std_error,
)


class TestSharpeStdError:
    def test_iid_normal_recovers_one_over_sqrt_n_minus_1(self):
        # SR=0, skew=0, excess kurt=0 → variance = 1/(n-1).
        se = sharpe_std_error(0.0, 101, 0.0, 0.0)
        assert se == pytest.approx(math.sqrt(1.0 / 100), rel=1e-9)

    def test_negative_skew_inflates_variance(self):
        # SR_obs > 0 with negative skew → variance term grows
        # (the −γ_3 · SR_obs term flips sign and adds positive mass).
        se_zero_skew = sharpe_std_error(0.5, 100, 0.0, 0.0)
        se_neg_skew = sharpe_std_error(0.5, 100, -1.0, 0.0)
        assert se_neg_skew > se_zero_skew

    def test_excess_kurtosis_input_converts_to_raw(self):
        # ek=0 (normal raw=3) → coefficient (3-1)/4=0.5; ek=3 → (6-1)/4=1.25
        # so ek=3 should yield larger variance when SR_obs != 0.
        se_normal = sharpe_std_error(0.5, 100, 0.0, 0.0)
        se_heavy = sharpe_std_error(0.5, 100, 0.0, 3.0)
        assert se_heavy > se_normal

    def test_raises_on_n_below_2(self):
        with pytest.raises(ValueError):
            sharpe_std_error(0.5, 1, 0.0, 0.0)


class TestExpectedMaxSharpe:
    def test_n_trials_one_is_zero(self):
        # Eq. 12 degenerates: Z⁻¹(0) = -inf — handled by short-circuit.
        assert expected_max_sharpe(1, 0.01) == 0.0

    def test_more_trials_gives_higher_max(self):
        v = 0.001
        e10 = expected_max_sharpe(10, v)
        e1000 = expected_max_sharpe(1000, v)
        assert e1000 > e10 > 0

    def test_zero_variance_gives_zero(self):
        assert expected_max_sharpe(100, 0.0) == 0.0

    def test_uses_euler_mascheroni(self):
        # Sanity-check the constant matches the published value.
        assert EULER_MASCHERONI == pytest.approx(0.5772156649, abs=1e-8)


class TestDeflatedSharpeRatio:
    def test_single_trial_recovers_raw_significance(self):
        # n_trials=1, SR_obs=2/sqrt(99) → ~0.2 per-period; should give
        # a high DSR since selection penalty is zero.
        n = 100
        sr = 0.2
        dsr = deflated_sharpe_ratio(sr, n, n_trials=1, skew=0.0, excess_kurtosis=0.0)
        assert 0.5 < dsr <= 1.0

    def test_high_trials_deflates_a_modest_sharpe(self):
        # SR_obs=1.0/sqrt(252) (daily, annualized 1.0), n=252, trials=1000.
        # Should NOT be significant — exactly the spec scenario.
        sr = 1.0 / math.sqrt(252)
        dsr = deflated_sharpe_ratio(
            sr, n_returns=252, n_trials=1000, skew=0.0, excess_kurtosis=0.0
        )
        assert dsr < 0.5

    def test_negative_skew_lowers_dsr(self):
        sr = 0.15
        n, trials = 200, 50
        dsr_pos = deflated_sharpe_ratio(sr, n, trials, skew=+0.5, excess_kurtosis=0.0)
        dsr_neg = deflated_sharpe_ratio(sr, n, trials, skew=-0.5, excess_kurtosis=0.0)
        assert dsr_neg < dsr_pos

    def test_5_13_4_audit_regression_guard_n_trials_gt_50(self):
        """§5.13.4: a Sharpe of 1.0 (annualized) with n_trials > 50
        cannot produce DSR ≥ 1.0 — the selection penalty must bite.
        """
        sr_per_period = 1.0 / math.sqrt(252)
        for n_trials in (51, 100, 500, 1000):
            dsr = deflated_sharpe_ratio(
                sr_per_period, n_returns=252,
                n_trials=n_trials, skew=0.0, excess_kurtosis=0.0,
            )
            assert dsr < 1.0, (
                f"DSR={dsr} at n_trials={n_trials} — single-number claim "
                f"would have escaped §5.13.4 deflation."
            )

    def test_dsr_returns_probability_in_unit_interval(self):
        for sr in (0.05, 0.10, 0.20):
            dsr = deflated_sharpe_ratio(
                sr, n_returns=252, n_trials=100,
                skew=0.0, excess_kurtosis=0.0,
            )
            assert 0.0 <= dsr <= 1.0

    def test_raises_on_zero_trials(self):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(0.1, 100, 0, 0.0, 0.0)


class TestAnnualizedSharpe:
    def test_iid_normal_synthetic(self):
        rng = np.random.default_rng(42)
        # mean 0.001/day, std 0.01/day → annualized SR ≈ sqrt(252) * 0.1 ≈ 1.587
        # 10k samples: finite-sample drift in mean/std typically ~10-20% on SR.
        rets = rng.normal(0.001, 0.01, size=10_000)
        ann = annualized_sharpe(rets, periods_per_year=252)
        assert abs(ann - 1.587) < 0.3

    def test_zero_std_returns_nan(self):
        flat = np.full(100, 0.001)
        assert math.isnan(annualized_sharpe(flat))

    def test_too_few_returns_nan(self):
        assert math.isnan(annualized_sharpe(np.array([0.01])))
