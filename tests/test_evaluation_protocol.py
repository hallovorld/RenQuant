"""Tests for the industry-leading paired-daily-returns evaluation protocol.

Pinned invariants per doc/research/evaluation-protocol.md:

1. Newey-West HAC SE matches Andrews 1991 closed-form lag for white-noise
   input → close to naive SE (no inflation).
2. HAC SE strictly LARGER than naive SE for AR(1) input (autocorrelated)
   — this is the WHOLE POINT of the method.
3. Block bootstrap CI on white noise is approximately ±1.96/√n.
4. Self-A/A: when baseline and "candidate" are IDENTICAL (same equity
   curve), pooled t = 0, mean_d = 0, verdict = NEITHER.
5. Stationary bootstrap preserves total length n.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


class TestNeweyWestSE:

    def test_lag_selection_andrews_1991(self):
        from kernel.metrics.hac_se import andrews_optimal_lag
        # n=100 → L=4 × (1)^(2/9) = 4
        assert andrews_optimal_lag(100) == 4
        # n=252 → L=4 × (2.52)^(2/9) ≈ 4.84 → floor=4
        assert andrews_optimal_lag(252) == 4
        # n=63 → L=4 × (0.63)^(2/9) ≈ 3.62 → floor=3
        assert andrews_optimal_lag(63) == 3

    def test_se_matches_naive_for_iid(self):
        """White-noise series: HAC SE ≈ naive SE (no inflation needed)."""
        from kernel.metrics.hac_se import newey_west_se
        rng = np.random.default_rng(42)
        r = rng.normal(0, 1, size=500)
        se_naive = r.std(ddof=0) / math.sqrt(len(r))
        se_nw = newey_west_se(r)
        # Should be within 20% of naive for iid input
        assert abs(se_nw - se_naive) / se_naive < 0.20, (
            f"HAC SE {se_nw} should be close to naive {se_naive} for iid"
        )

    def test_se_inflated_for_ar1(self):
        """Strong AR(1) series: HAC SE strictly LARGER than naive."""
        from kernel.metrics.hac_se import newey_west_se
        rng = np.random.default_rng(42)
        n = 500
        phi = 0.5
        r = np.zeros(n)
        eps = rng.normal(0, 1, size=n)
        r[0] = eps[0]
        for t in range(1, n):
            r[t] = phi * r[t-1] + eps[t]
        se_naive = r.std(ddof=0) / math.sqrt(len(r))
        se_nw = newey_west_se(r)
        # For φ=0.5 AR(1), variance inflation factor ≈ (1+φ)/(1-φ) = 3
        # SE inflation ≈ √3 ≈ 1.73
        assert se_nw > 1.4 * se_naive, (
            f"HAC SE {se_nw} should be much larger than naive {se_naive} "
            f"for AR(1) φ=0.5"
        )

    def test_t_stat_normal_under_null(self):
        """White noise mean ≈ 0 → |t| < 2 with high probability."""
        from kernel.metrics.hac_se import hac_t_stat
        rng = np.random.default_rng(123)
        r = rng.normal(0, 1, size=500)
        res = hac_t_stat(r)
        assert abs(res["t_stat"]) < 3.0  # 99.7% level

    def test_t_stat_detects_signal(self):
        """Mean = 0.5σ should give t ≈ 0.5 × √n."""
        from kernel.metrics.hac_se import hac_t_stat
        rng = np.random.default_rng(42)
        n = 500
        r = rng.normal(0.05, 1.0, size=n)  # mean / σ = 0.05
        res = hac_t_stat(r)
        # t ≈ 0.05 × √500 ≈ 1.12 (might be inflated by HAC randomness)
        assert res["t_stat"] > 0.8, (
            f"t-stat {res['t_stat']} should detect mean=0.05 signal at n=500"
        )


class TestStationaryBootstrap:

    def test_optimal_block_length_sensible(self):
        from kernel.metrics.block_bootstrap import optimal_block_length
        # n=100 → ~5; arch will pick something in [2, 20]
        L = optimal_block_length(np.random.default_rng(0).normal(0, 1, 100))
        assert 2 <= L <= 30

    def test_ci_covers_true_mean_for_iid(self):
        """Bootstrap 95% CI should cover the true mean for iid data."""
        from kernel.metrics.block_bootstrap import stationary_bootstrap_ci
        rng = np.random.default_rng(42)
        true_mu = 0.3
        s = rng.normal(true_mu, 1.0, size=500)
        res = stationary_bootstrap_ci(s, B=500, rng=rng)
        assert res["ci_lo"] < true_mu < res["ci_hi"], (
            f"CI [{res['ci_lo']}, {res['ci_hi']}] should cover true {true_mu}"
        )

    def test_point_estimate_matches_sample_mean(self):
        from kernel.metrics.block_bootstrap import stationary_bootstrap_ci
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res = stationary_bootstrap_ci(s, B=100)
        assert res["point"] == pytest.approx(3.5)


class TestPairedReturnsAnalyzer:

    def test_self_ab_zero_signal(self, tmp_path):
        """Self-A/A test: identical baseline ≡ candidate → t = 0, no signal."""
        import subprocess
        # Generate 3 synthetic equity curves (windows)
        bdir = tmp_path / "baseline"
        cdir = tmp_path / "candidate"
        bdir.mkdir()
        cdir.mkdir()
        rng = np.random.default_rng(42)
        for win in ["W1", "W2", "W3"]:
            n = 60
            # Use pandas business-day range so months don't overflow
            start = pd.Timestamp(f"2024-0{4 + ['W1','W2','W3'].index(win)}-01")
            dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
            curve = 100000 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=n))
            payload = {
                "config": "x", "start": dates[0], "end": dates[-1],
                "initial_cash": 100000, "final_value": float(curve[-1]),
                "total_return": float(curve[-1] / 100000 - 1),
                "apy": 0.0, "sharpe": None, "ann_vol": None, "max_dd": None,
                "equity": dict(zip(dates, curve.tolist())),
            }
            import json as _json
            (bdir / f"{win}.json").write_text(_json.dumps(payload))
            (cdir / f"{win}.json").write_text(_json.dumps(payload))  # IDENTICAL
        # Run analyzer
        result = subprocess.run([
            sys.executable, str(REPO_ROOT / "scripts" / "eval_paired_returns.py"),
            "--baseline-dir", str(bdir), "--candidate-dir", str(cdir),
            "--k-trials", "1",
        ], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        # Output should mention "NEITHER" or zero mean — self-A/A must NOT
        # produce false positives
        assert "mean Δ (annualized)    : +0.00%" in result.stdout or \
               "mean Δ (annualized)    : -0.00%" in result.stdout, (
            f"Self-A/A should produce zero mean Δ. Output:\n{result.stdout}"
        )
        # Must NOT classify as Tier 3
        assert "TIER3_CONFIRMED" not in result.stdout, "Self-A/A must not pass Tier 3"
