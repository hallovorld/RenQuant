"""Tests for the §4.6 v6 BCa-bound pre-activation simulation.

Covers the properties the amendment requires PLUS the load-bearing empirical
finding this module produced:

* the v6 BCa bound provably REUSES the merged v5 MBB resampling (byte-identical
  resample means → identical percentile bound under an equal seed);
* the BCa construction reduces to the raw percentile bound when the bootstrap
  distribution is symmetric and the acceleration is zero (a sanity check that the
  BCa correction is the ONLY difference);
* the deep-null (mu=0) rejection is tiny → the sim is sound, so an elevated
  boundary rate is a genuine calibration fact, not Monte-Carlo noise;
* determinism under a fixed seed;
* the v6 §4.6 gate logic — the method-calibration bar (type-I <= 0.10 AND 95% CI
  upper <= 0.12) AND power >= 0.80 → activation-ready, else the CORRECT INFEASIBLE
  branch (uncalibrated inference vs conservative-noise);
* THE FINDING — BCa does NOT control the boundary type-I: it stays ~0.14 (T=240),
  essentially equal to the anti-conservative percentile bound, because the flaw is
  variance-scale (few-blocks MBB under-estimates the mean's sampling variance), not
  the median-bias / skewness BCa corrects. The gate therefore correctly returns
  INFEASIBLE — INFERENCE METHOD UNCALIBRATED (→ v7 studentized-t). This is asserted,
  not papered over.

All statistical assertions use FIXED seeds so the numbers are exact / non-flaky;
trial and bootstrap counts are reduced for test speed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import prereg_ew_power_simulation as base  # noqa: E402  (merged primitives)
import prereg_ew_power_simulation_v6 as v6mod  # noqa: E402
from prereg_ew_power_simulation_v6 import (  # noqa: E402
    GO_VALID,
    INFEASIBLE_INFERENCE,
    INFEASIBLE_NOISE,
    MEE_BPS,
    PE_BPS,
    SYNTHETIC_DEFAULT,
    TYPE_I_CI_UPPER_MAX,
    TYPE_I_REQUIREMENT,
    PilotNull,
    block_jackknife_acceleration,
    boundary_rejection_rates,
    mbb_bca_lower_bound,
    rejection_rate,
    run_protocol_v6,
    type_i_calibrated,
    v6_outcome,
    validate_boundary_type_i,
)

SCRIPT = SCRIPTS / "prereg_ew_power_simulation_v6.py"
P = SYNTHETIC_DEFAULT


class TestReusesMergedResampling:

    def test_bootstrap_means_mirror_merged_primitive(self):
        # The v6 BCa bound reads its correction off _mbb_bootstrap_means, which
        # MIRRORS the merged mbb_lower_bound resampling. Under an equal seed the
        # alpha-quantile of the mirror MUST equal the merged bound exactly, proving
        # the resampling is reused (not silently re-derived).
        d = np.random.default_rng(1).normal(3.0, 25.0, 240)
        r1 = np.random.default_rng(42)
        means = v6mod._mbb_bootstrap_means(d, 35, r1, n_boot=5000)
        mirror_pct = float(np.quantile(means, 0.10))
        r2 = np.random.default_rng(42)
        merged = base.mbb_lower_bound(d, 35, r2, n_boot=5000, alpha=0.10)
        assert mirror_pct == merged

    def test_block_length_over_series_rejected(self):
        with pytest.raises(ValueError):
            v6mod._mbb_bootstrap_means(np.zeros(30), 35, np.random.default_rng(0))


class TestBcaConstruction:

    def test_reduces_to_percentile_when_symmetric_zero_acceleration(self, monkeypatch):
        # z0 = 0 (symmetric: exactly half the means below the observed mean) and a = 0
        # ⇒ alpha1 = Phi(z_alpha) = 0.10 ⇒ BCa == raw percentile. Confirms the BCa
        # correction is the SOLE difference from v5's bound. (Uses a large means
        # vector so the degenerate-tail clamp 0.5/n_boot — negligible at B=10000 —
        # does not perturb alpha1.)
        monkeypatch.setattr(v6mod, "block_jackknife_acceleration", lambda d, b: 0.0)
        theta = 0.0
        below = -np.arange(1, 1001, dtype=float)
        above = np.arange(1, 1001, dtype=float)
        means = np.concatenate([below, above])  # 2000 pts, exactly 50% below theta
        d = np.full(40, theta)  # observed mean = theta
        lb = mbb_bca_lower_bound(d, 35, np.random.default_rng(0), alpha=0.10, means=means)
        assert lb == pytest.approx(float(np.quantile(means, 0.10)))

    def test_negative_acceleration_makes_bound_more_conservative(self, monkeypatch):
        # a < 0 with z0 = 0 lowers the effective percentile below 0.10 ⇒ a LOWER
        # (more conservative) bound than the raw percentile.
        monkeypatch.setattr(v6mod, "block_jackknife_acceleration", lambda d, b: -0.05)
        theta = 0.0
        means = np.linspace(-10.0, 10.0, 2001)  # symmetric ⇒ z0 = 0
        d = np.full(40, theta)
        lb = mbb_bca_lower_bound(d, 35, np.random.default_rng(0), alpha=0.10, means=means)
        assert lb < float(np.quantile(means, 0.10))

    def test_acceleration_zero_on_constant_series(self):
        assert block_jackknife_acceleration(np.ones(100), 35) == 0.0

    def test_bca_bound_deterministic_under_seed(self):
        d = np.random.default_rng(2).normal(3.0, 25.0, 240)
        a = mbb_bca_lower_bound(d, 35, np.random.default_rng(7), n_boot=1500)
        b = mbb_bca_lower_bound(d, 35, np.random.default_rng(7), n_boot=1500)
        assert a == b


class TestTypeICalibratedGate:

    @pytest.mark.parametrize("rate,ci_upper,expected", [
        (0.08, 0.11, True),    # point + CI both under the bars
        (0.10, 0.12, True),    # exactly on both bars ⇒ pass
        (0.105, 0.11, False),  # point estimate exceeds 0.10
        (0.10, 0.121, False),  # CI upper exceeds 0.12 (the MC-noise margin)
    ])
    def test_two_bar_acceptance(self, rate, ci_upper, expected):
        cell = {"rate": rate, "ci95": [max(0.0, rate - 0.01), ci_upper]}
        assert type_i_calibrated(cell) is expected

    def test_constants_frozen(self):
        assert TYPE_I_REQUIREMENT == 0.10
        assert TYPE_I_CI_UPPER_MAX == 0.12


class TestV6OutcomeLogic:

    @pytest.mark.parametrize("calibrated,feasible,outcome", [
        (True, True, GO_VALID),
        (False, True, INFEASIBLE_INFERENCE),   # calibration gate dominates
        (False, False, INFEASIBLE_INFERENCE),  # ...even when power also fails
        (True, False, INFEASIBLE_NOISE),
    ])
    def test_outcome_mapping(self, calibrated, feasible, outcome):
        assert v6_outcome(calibrated, feasible) == outcome


class TestDeterminismAndSanity:

    def test_rejection_rate_deterministic(self):
        a = rejection_rate(60, 200, PE_BPS, 25.0, 0.35, 35, MEE_BPS, seed=9,
                           n_boot=400, method="bca")
        b = rejection_rate(60, 200, PE_BPS, 25.0, 0.35, 35, MEE_BPS, seed=9,
                           n_boot=400, method="bca")
        assert a == b

    def test_deep_null_barely_rejects(self):
        # mu = 0 (far below the MEE=3 threshold): BCa almost never rejects ⇒ the sim
        # is sound and the elevated boundary rate is a GENUINE calibration fact.
        r = rejection_rate(1500, 240, 0.0, P.marginal_sigma_bps, P.ar1_phi,
                           P.block_length, MEE_BPS, seed=24, n_boot=1500, method="bca")
        assert r["rate"] == pytest.approx(0.0120, abs=1e-4)
        assert r["rate"] < 0.05

    def test_huge_effect_always_rejects(self):
        r = rejection_rate(30, 240, 50.0, 10.0, 0.0, 35, MEE_BPS, seed=5,
                           n_boot=300, method="bca")
        assert r["rate"] == 1.0

    def test_bca_power_increases_with_T(self):
        kw = dict(mean_bps=PE_BPS, sigma_bps=P.marginal_sigma_bps, phi=P.ar1_phi,
                  b=P.block_length, threshold_bps=MEE_BPS, seed=11, n_boot=1500,
                  method="bca")
        r240 = rejection_rate(500, 240, **kw)["rate"]
        r480 = rejection_rate(500, 480, **kw)["rate"]
        assert r480 > r240


class TestFindingBcaDoesNotControl:

    def test_paired_percentile_matches_merged_v5_number(self):
        # The paired percentile arm reproduces the merged v5 boundary type-I EXACTLY
        # (0.1393 at seed 23, T=240, 1500/1500) — the anchor the BCa arm is compared
        # against on the SAME resamples.
        res = boundary_rejection_rates(1500, 240, MEE_BPS, P.marginal_sigma_bps,
                                       P.ar1_phi, P.block_length, MEE_BPS, seed=23,
                                       n_boot=1500)
        assert res["percentile"]["rate"] == pytest.approx(0.1393, abs=1e-4)

    def test_bca_does_not_control_boundary_type_i(self):
        # THE FINDING. On the SAME resamples, BCa's boundary type-I (0.1427) is NOT
        # below the percentile bound's (0.1393) — it is essentially equal (and here
        # marginally worse). Both sit ~0.14 with 95% CI lower bounds ABOVE the
        # nominal 0.10, so BCa does NOT calibrate the bound: the anti-conservatism is
        # variance-scale, not the median-bias / skewness BCa corrects.
        res = boundary_rejection_rates(1500, 240, MEE_BPS, P.marginal_sigma_bps,
                                       P.ar1_phi, P.block_length, MEE_BPS, seed=23,
                                       n_boot=1500)
        assert res["bca"]["rate"] == pytest.approx(0.1427, abs=1e-4)
        assert res["bca"]["rate"] > TYPE_I_REQUIREMENT        # BCa still over-rejects
        assert res["bca"]["ci95"][0] > TYPE_I_REQUIREMENT     # CI excludes nominal α
        # BCa does not MATERIALLY move the boundary type-I vs the percentile bound.
        assert abs(res["percentile"]["rate"] - res["bca"]["rate"]) < 0.02
        assert type_i_calibrated(res["bca"]) is False

    def test_validate_returns_uncalibrated_verdict(self):
        # The full validation grid (small config) returns BCa-does-not-control on the
        # synthetic default: every cell's BCa boundary type-I stays anti-conservative.
        val = validate_boundary_type_i(P, trials=1200, boots=1200,
                                       ts=(240, 480), seed=20260719)
        assert val["bca_controls_boundary_type_i"] is False
        assert val["percentile_anticonservative"] is True
        for name, cell in val["cells"].items():
            assert cell["bca"]["rate"] > TYPE_I_REQUIREMENT, name
            assert cell["bca_cell_calibrated"] is False, name


class TestProtocolBranches:

    _real = PilotNull("d" * 64, 0.35, 20.0, 20, "pilot-fit:real")

    @staticmethod
    def _fake(power_rate, type_i_rate, ci_upper=None):
        ciu = type_i_rate if ci_upper is None else ci_upper

        def _f(n_trials, t_sessions, mean_bps, *a, **k):
            if mean_bps == PE_BPS:
                rate = power_rate
                return {"rate": rate, "ci95": [rate, rate], "rejections": 0,
                        "trials": n_trials}
            rate = type_i_rate
            return {"rate": rate, "ci95": [max(0.0, rate - 0.005), ciu],
                    "rejections": 0, "trials": n_trials}
        return _f

    def test_calibrated_and_feasible_is_go(self, monkeypatch):
        monkeypatch.setattr(v6mod, "rejection_rate", self._fake(0.85, 0.08))
        res = run_protocol_v6(self._real, seed=1, trials=10, boots=10)
        assert res["verdict"]["outcome"] == GO_VALID
        assert res["verdict"]["activation_ready"] is True
        assert res["power"]["final_t"] == 240

    def test_uncalibrated_is_inference_infeasible(self, monkeypatch):
        # BCa boundary type-I 0.14 > 0.10 ⇒ INFEASIBLE — INFERENCE METHOD UNCALIBRATED
        # even though power is feasible. This is the finding's protocol outcome.
        monkeypatch.setattr(v6mod, "rejection_rate", self._fake(0.85, 0.14))
        res = run_protocol_v6(self._real, seed=1, trials=10, boots=10)
        assert res["verdict"]["outcome"] == INFEASIBLE_INFERENCE
        assert res["verdict"]["activation_ready"] is False
        assert res["calibration_gate"]["inference_calibrated"] is False

    def test_ci_upper_margin_blocks_borderline(self, monkeypatch):
        # Point estimate passes (0.10) but the 95% CI upper (0.13) exceeds 0.12 ⇒
        # still uncalibrated. The MC-noise margin bites.
        monkeypatch.setattr(v6mod, "rejection_rate",
                            self._fake(0.85, 0.10, ci_upper=0.13))
        res = run_protocol_v6(self._real, seed=1, trials=10, boots=10)
        assert res["verdict"]["outcome"] == INFEASIBLE_INFERENCE

    def test_calibrated_but_power_infeasible_is_noise(self, monkeypatch):
        monkeypatch.setattr(v6mod, "rejection_rate", self._fake(0.50, 0.08))
        res = run_protocol_v6(self._real, seed=1, trials=10, boots=10)
        assert res["verdict"]["outcome"] == INFEASIBLE_NOISE
        assert res["power"]["final_t"] is None

    def test_synthetic_never_activation_ready(self, monkeypatch):
        monkeypatch.setattr(v6mod, "rejection_rate", self._fake(0.90, 0.05))
        res = run_protocol_v6(SYNTHETIC_DEFAULT, seed=1, trials=10, boots=10)
        assert res["verdict"]["gate_pass"] is True
        assert res["verdict"]["activation_ready"] is False

    def test_no_parameter_walked(self, monkeypatch):
        monkeypatch.setattr(v6mod, "rejection_rate", self._fake(0.85, 0.14))
        res = run_protocol_v6(self._real, seed=1, trials=10, boots=10)
        assert res["decision_rule"]["mee_bps_per_session"] == 3.0
        assert res["decision_rule"]["pe_bps_per_session"] == 6.0
        assert res["activation_simulation_block"]["mbb_block_length"] == 35


class TestFrozenOutputs:

    def test_activation_block_has_all_required_fields(self):
        res = run_protocol_v6(SYNTHETIC_DEFAULT, seed=2, trials=40, boots=200)
        block = res["activation_simulation_block"]
        for key in ("inference_method", "type_i_rate", "type_i_ci_95",
                    "inference_calibrated", "power_at_pe", "power_ci_95",
                    "final_n_sessions", "final_n_blocks", "mbb_block_length",
                    "dgp_parameters", "pilot_manifest_digest",
                    "simulation_code_commit", "simulation_seed"):
            assert key in block, key
        assert block["inference_method"] == "bca"
        assert block["simulation_seed"] == 2


class TestCli:

    def test_protocol_quick_smoke(self, tmp_path):
        out = tmp_path / "r.json"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--quick", "--seed", "1",
             "--out", str(out)], capture_output=True, text=True, timeout=300)
        assert proc.returncode in (0, 1), proc.stderr
        res = json.loads(out.read_text())
        assert res["protocol_version"] == "v6"
        assert res["inference_method"] == "bca"
        assert res["decision_rule"]["mee_bps_per_session"] == 3.0
        assert "calibration_gate" in res

    def test_validate_mode_smoke(self, tmp_path):
        out = tmp_path / "v.json"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--validate", "--seed", "20260719",
             "--trials", "400", "--boots", "600", "--out", str(out)],
            capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr
        res = json.loads(out.read_text())
        assert "validation" in res
        assert set(res["validation"]["cells"]) == {
            "iid_T240", "iid_T480", "ar1_T240", "ar1_T480"}
        assert res["resolved_outcome"] in (
            GO_VALID, INFEASIBLE_NOISE, INFEASIBLE_INFERENCE)
