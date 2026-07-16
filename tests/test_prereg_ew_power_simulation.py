"""Tests for the §4.6 pre-activation power simulation components."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prereg_ew_power_simulation import (  # noqa: E402
    MDE_BPS,
    mbb_block_length,
    mbb_lower_bound,
    rejection_rate,
    simulate_ar1,
)


class TestBlockLength:

    def test_matches_prereg_formula(self):
        # §4.1 step 2: b = ceil(1.75 * max_holding_days)
        assert mbb_block_length(20) == 35
        assert mbb_block_length(10) == 18  # ceil(17.5)

    def test_cap_at_40_holding_days(self):
        assert mbb_block_length(60) == mbb_block_length(40) == 70


class TestSimulateAr1:

    def test_shape_mean_and_marginal_std(self):
        rng = np.random.default_rng(7)
        x = simulate_ar1(400, 300, mean_bps=5.0, sigma_bps=20.0, phi=0.5,
                         rng=rng)
        assert x.shape == (400, 300)
        assert abs(x.mean() - 5.0) < 1.0
        assert abs(x.std() - 20.0) < 1.5

    def test_phi_zero_is_iid(self):
        rng = np.random.default_rng(7)
        x = simulate_ar1(200, 500, 0.0, 10.0, 0.0, rng=rng)
        lag1 = np.mean([np.corrcoef(row[:-1], row[1:])[0, 1] for row in x])
        assert abs(lag1) < 0.05

    def test_invalid_phi_rejected(self):
        rng = np.random.default_rng(7)
        with pytest.raises(ValueError):
            simulate_ar1(1, 10, 0.0, 10.0, 1.0, rng=rng)


class TestMbbLowerBound:

    def test_lower_bound_below_sample_mean(self):
        rng = np.random.default_rng(3)
        d = rng.normal(5.0, 10.0, size=240)
        lb = mbb_lower_bound(d, b=35, rng=rng, n_boot=2000)
        assert lb < d.mean()

    def test_constant_series_bound_equals_value(self):
        rng = np.random.default_rng(3)
        d = np.full(240, 4.2)
        assert mbb_lower_bound(d, b=35, rng=rng, n_boot=500) == pytest.approx(4.2)

    def test_block_longer_than_series_rejected(self):
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError):
            mbb_lower_bound(np.zeros(30), b=35, rng=rng, n_boot=10)

    def test_deterministic_under_seed(self):
        d = np.random.default_rng(11).normal(0, 10, size=240)
        lb1 = mbb_lower_bound(d, 35, np.random.default_rng(42), n_boot=1000)
        lb2 = mbb_lower_bound(d, 35, np.random.default_rng(42), n_boot=1000)
        assert lb1 == lb2


class TestRejectionRate:

    def test_seed_determinism(self):
        r1 = rejection_rate(20, 120, MDE_BPS, 25.0, 0.35, 35, seed=9,
                            n_boot=200)
        r2 = rejection_rate(20, 120, MDE_BPS, 25.0, 0.35, 35, seed=9,
                            n_boot=200)
        assert r1 == r2

    def test_huge_effect_always_rejects(self):
        # true mean 50 bps >> MDE: every trial's lower bound clears 3 bps
        res = rejection_rate(30, 240, 50.0, 10.0, 0.0, 35, seed=5, n_boot=300)
        assert res["rate"] == 1.0

    def test_null_rarely_rejects(self):
        res = rejection_rate(50, 240, 0.0, 25.0, 0.35, 35, seed=5, n_boot=300)
        assert res["rate"] <= 0.10


class TestSmokeRun:

    def test_quick_protocol_end_to_end(self, tmp_path):
        out = tmp_path / "r.json"
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "prereg_ew_power_simulation.py"),
             "--quick", "--seed", "1", "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        # exit code 1 = gate FAIL is a valid, expected outcome
        assert proc.returncode in (0, 1), proc.stderr
        results = json.loads(out.read_text())
        assert results["section"] == "4.6"
        assert "verdict" in results and "type_i" in results
        assert results["frozen_parameters"]["mde_bps_per_session"] == 3.0
