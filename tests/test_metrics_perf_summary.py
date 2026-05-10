"""Tests for kernel.metrics.perf_summary.compute_perf_triple."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.metrics.perf_summary import compute_perf_triple  # noqa: E402


class TestSingleSeriesMode:
    def test_returns_all_seven_keys(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0.0005, 0.01, size=252)
        out = compute_perf_triple(rets, n_trials=10)
        assert set(out) == {
            "sharpe", "sharpe_mean", "sharpe_std",
            "dsr", "pbo", "n_returns", "n_trials",
        }

    def test_pbo_is_nan_without_multi_seed(self):
        rng = np.random.default_rng(1)
        rets = rng.normal(0.0, 0.01, size=252)
        out = compute_perf_triple(rets, n_trials=5)
        assert math.isnan(out["pbo"])
        assert math.isnan(out["sharpe_std"])
        assert out["sharpe_mean"] == pytest.approx(out["sharpe"])

    def test_dsr_in_unit_interval(self):
        rng = np.random.default_rng(2)
        rets = rng.normal(0.0008, 0.01, size=252)
        out = compute_perf_triple(rets, n_trials=100)
        assert 0.0 <= out["dsr"] <= 1.0

    def test_n_returns_strips_nans(self):
        rets = np.array([0.01, np.nan, 0.005, np.nan, -0.002, 0.003])
        out = compute_perf_triple(rets, n_trials=1)
        assert out["n_returns"] == 4


class TestMultiSeedMode:
    def test_pbo_populated_when_multi_seed_provided(self):
        rng = np.random.default_rng(3)
        T, K = 256, 8
        M = rng.normal(0.0, 0.01, size=(T, K))
        headline = M[:, 0]
        out = compute_perf_triple(
            headline, n_trials=K, multi_seed_returns=M, pbo_n_slices=8,
        )
        assert math.isfinite(out["pbo"])
        assert math.isfinite(out["sharpe_mean"])
        assert math.isfinite(out["sharpe_std"])

    def test_sharpe_mean_std_match_per_seed_distribution(self):
        rng = np.random.default_rng(4)
        T, K = 252, 5
        # Five seeds with slight drift differences.
        M = rng.normal(0.0005, 0.01, size=(T, K))
        out = compute_perf_triple(
            M[:, 0], n_trials=K, multi_seed_returns=M, pbo_n_slices=8,
        )
        # Expected sharpe_mean from manual annualization
        per_seed = []
        for k in range(K):
            mu, sd = M[:, k].mean(), M[:, k].std(ddof=1)
            per_seed.append(mu / sd * math.sqrt(252))
        assert out["sharpe_mean"] == pytest.approx(np.mean(per_seed), rel=1e-9)
        assert out["sharpe_std"] == pytest.approx(np.std(per_seed, ddof=1), rel=1e-9)

    def test_rejects_single_seed_matrix(self):
        rng = np.random.default_rng(5)
        with pytest.raises(ValueError):
            compute_perf_triple(
                rng.normal(size=128), n_trials=1,
                multi_seed_returns=rng.normal(size=(128, 1)),
            )


class TestPerfTripleSampleScenario:
    """The exact scenario M2 prompt asks for: dump a sample triple."""

    def test_sample_synthetic_run_prints_clean(self, capsys):
        rng = np.random.default_rng(2026)
        T, K = 252, 8
        # 8 seeds of a strategy with weak edge (~0.2 SR) over noise.
        M = rng.normal(0.0003, 0.01, size=(T, K))
        out = compute_perf_triple(
            M[:, 0], n_trials=20, multi_seed_returns=M, pbo_n_slices=8,
        )
        # Sanity bounds — none should be NaN here.
        for k in ("sharpe", "sharpe_mean", "sharpe_std", "dsr", "pbo"):
            assert math.isfinite(out[k]), f"{k} unexpectedly NaN"
        # Print for the M2 deliverable record.
        msg = (
            f"sharpe={out['sharpe']:.3f} "
            f"mean±std={out['sharpe_mean']:.3f}±{out['sharpe_std']:.3f} "
            f"dsr={out['dsr']:.3f} pbo={out['pbo']:.3f} "
            f"n_returns={out['n_returns']} n_trials={out['n_trials']}"
        )
        print(msg)
        captured = capsys.readouterr()
        assert "sharpe=" in captured.out
