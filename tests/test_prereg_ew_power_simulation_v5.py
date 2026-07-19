"""Tests for the §4.6 v5 pre-activation power / type-I simulation.

Covers the four properties the task requires plus the honest finding the sim
surfaces:

* the v5 block-length rule (b capped at 40 sessions) and how it differs from
  the #485 (v3) rule (mhd capped at 40 → b up to 70);
* the pilot-calibrated null INPUT CONTRACT (long-run-variance sizing);
* power increases with T and with effect size;
* the infeasibility branch fires under high injected variance and does NOT walk
  any parameter;
* the v3-vs-v5 rule difference: v5 power at PE is NOT coverage-capped, unlike the
  v3 power-at-threshold (≈ α);
* determinism under a fixed seed;
* FINDING — the FROZEN §4.1 percentile MBB is mildly anti-conservative: the
  deployment rule's boundary type-I (μ = MEE) runs ~0.13–0.15 (> the nominal
  0.10), worst with few blocks. The sim's gate correctly refuses activation.

All statistical assertions use FIXED seeds so the numbers below are exact and
non-flaky; trial/bootstrap counts are reduced for test speed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import prereg_ew_power_simulation as v3  # noqa: E402  (#485 module, for the rule differential)
import prereg_ew_power_simulation_v5 as v5mod  # noqa: E402
from prereg_ew_power_simulation_v5 import (  # noqa: E402
    GO_VALID,
    INFEASIBLE,
    MEE_BPS,
    PE_BPS,
    SYNTHETIC_DEFAULT,
    TYPE_I_EXCEEDED,
    PilotNull,
    mbb_block_length,
    rejection_rate,
    run_protocol,
)

SCRIPT = SCRIPTS / "prereg_ew_power_simulation_v5.py"


class TestBlockLengthV5:

    def test_default_holding_matches_v3(self):
        # For the pilot's ~20-session holding period, v5 and v3 agree: b = 35.
        assert mbb_block_length(20) == 35 == v3.mbb_block_length(20)

    def test_ceil_rule(self):
        assert mbb_block_length(10) == 18  # ceil(17.5)
        assert mbb_block_length(22) == 39  # ceil(38.5), below the cap

    def test_v5_caps_b_at_40_sessions(self):
        # v5 caps b itself at 40 sessions; the multiply happens first.
        assert mbb_block_length(23) == 40  # ceil(40.25) -> capped
        assert mbb_block_length(30) == 40
        assert mbb_block_length(40) == 40

    def test_rule_differs_from_v3_for_long_holdings(self):
        # THE rule change: v3 capped max_holding_days at 40 (b up to 70);
        # v5 caps b at 40. They diverge once mhd > ~22.
        assert v3.mbb_block_length(40) == 70
        assert mbb_block_length(40) == 40
        assert mbb_block_length(40) != v3.mbb_block_length(40)

    def test_invalid_holding_rejected(self):
        with pytest.raises(ValueError):
            mbb_block_length(0)


class TestPilotNullContract:

    def test_marginal_roundtrip(self):
        p = PilotNull.from_marginal(25.0, 0.35, 20)
        assert p.marginal_sigma_bps == pytest.approx(25.0, abs=1e-9)
        assert p.long_run_std_upper90_bps == pytest.approx(36.0288, abs=1e-3)
        assert p.block_length == 35

    def test_long_run_variance_identity(self):
        # AR(1): long_run_var = marginal_var * (1+phi)/(1-phi).
        p = PilotNull.from_marginal(18.0, 0.5, 12)
        phi = p.ar1_phi
        lrv_from_marginal = p.marginal_sigma_bps ** 2 * (1 + phi) / (1 - phi)
        assert lrv_from_marginal == pytest.approx(
            p.long_run_std_upper90_bps ** 2, rel=1e-9)

    def test_synthetic_default_flagged(self):
        assert SYNTHETIC_DEFAULT.is_synthetic is True
        assert SYNTHETIC_DEFAULT.block_length == 35
        assert SYNTHETIC_DEFAULT.marginal_sigma_bps == pytest.approx(25.0, abs=1e-6)

    def test_from_json_roundtrip(self, tmp_path):
        payload = {
            "pilot_manifest_digest": "a" * 64,
            "ar1_phi": 0.42,
            "long_run_std_upper90_bps": 30.5,
            "max_holding_days": 15,
            "source": "pilot-fit:test",
        }
        f = tmp_path / "fit.json"
        f.write_text(json.dumps(payload))
        p = PilotNull.from_json(f)
        assert p.pilot_manifest_digest == "a" * 64
        assert p.ar1_phi == 0.42
        assert p.is_synthetic is False

    @pytest.mark.parametrize("phi,std,mhd", [
        (1.0, 30.0, 20),   # phi must be < 1
        (-0.1, 30.0, 20),  # phi must be >= 0
        (0.3, 0.0, 20),    # std must be > 0
        (0.3, 30.0, 0),    # mhd must be >= 1
    ])
    def test_validation_rejects_bad_inputs(self, phi, std, mhd):
        with pytest.raises(ValueError):
            PilotNull("x" * 64, phi, std, mhd, "src")


class TestRejectionRate:

    def test_deterministic_under_seed(self):
        a = rejection_rate(60, 200, PE_BPS, 25.0, 0.35, 35, MEE_BPS, seed=9,
                           n_boot=400)
        b = rejection_rate(60, 200, PE_BPS, 25.0, 0.35, 35, MEE_BPS, seed=9,
                           n_boot=400)
        assert a == b

    def test_huge_effect_always_rejects(self):
        r = rejection_rate(30, 240, 50.0, 10.0, 0.0, 35, MEE_BPS, seed=5,
                           n_boot=300)
        assert r["rate"] == 1.0

    def test_power_increases_with_effect_size(self):
        # Synthetic-default DGP, T=240, fixed seed 7.
        p = SYNTHETIC_DEFAULT
        kw = dict(sigma_bps=p.marginal_sigma_bps, phi=p.ar1_phi,
                  b=p.block_length, threshold_bps=MEE_BPS, seed=7, n_boot=1200)
        r_null = rejection_rate(400, 240, 0.0, **kw)["rate"]        # 0.0100
        r_mee = rejection_rate(400, 240, MEE_BPS, **kw)["rate"]     # 0.1300
        r_pe = rejection_rate(400, 240, PE_BPS, **kw)["rate"]       # 0.5550
        assert r_null < r_mee < r_pe
        assert r_null < 0.05 < r_mee < 0.30 < r_pe

    def test_power_increases_with_T(self):
        p = SYNTHETIC_DEFAULT
        kw = dict(mean_bps=PE_BPS, sigma_bps=p.marginal_sigma_bps,
                  phi=p.ar1_phi, b=p.block_length, threshold_bps=MEE_BPS,
                  seed=11, n_boot=1500)
        r240 = rejection_rate(500, 240, **kw)["rate"]  # 0.5360
        r480 = rejection_rate(500, 480, **kw)["rate"]  # 0.7020
        assert r480 > r240 + 0.05


class TestTypeIBoundary:

    def test_boundary_is_alpha_scale(self):
        # "It's the null boundary": at mu=MEE the rejection sits at the alpha
        # SCALE — an order of magnitude above the deep null (mu=0), far below
        # the PE power. This is the structural property the sim validates.
        p = SYNTHETIC_DEFAULT
        kw = dict(sigma_bps=p.marginal_sigma_bps, phi=p.ar1_phi,
                  b=p.block_length, threshold_bps=MEE_BPS, n_boot=1500)
        boundary = rejection_rate(1500, 240, MEE_BPS, seed=23, **kw)["rate"]
        deep_null = rejection_rate(1500, 240, 0.0, seed=24, **kw)["rate"]
        assert deep_null < 0.05
        assert 0.08 < boundary < 0.20
        assert boundary > 5 * deep_null

    def test_frozen_percentile_mbb_is_anticonservative(self):
        # FINDING: the FROZEN §4.1 percentile MBB does NOT deliver type-I <= 0.10
        # at the boundary — it runs ~0.14 here (few blocks: b=35, 7 blocks over
        # T=240). The 95% simulation CI excludes the nominal 0.10, so this is a
        # calibration fact, not Monte-Carlo noise. The sim's gate must refuse
        # activation on it (never silently pass).
        p = SYNTHETIC_DEFAULT
        r = rejection_rate(1500, 240, MEE_BPS, p.marginal_sigma_bps, p.ar1_phi,
                           p.block_length, MEE_BPS, seed=23, n_boot=1500)
        assert r["rate"] > 0.10           # 0.1393
        assert r["ci95"][0] > 0.10        # CI lower bound 0.122 > 0.10


class TestV3VsV5Rule:

    def test_v5_power_at_pe_not_coverage_capped(self):
        # v3 sized power AT the rule threshold (mu == threshold == MDE) → ≈ α by
        # one-sided-bound coverage. v5 sizes power at PE = 2*MEE, ABOVE the
        # threshold, so it is NOT coverage-capped and is materially larger.
        p = SYNTHETIC_DEFAULT
        kw = dict(sigma_bps=p.marginal_sigma_bps, phi=p.ar1_phi,
                  b=p.block_length, threshold_bps=MEE_BPS, seed=7, n_boot=1200)
        v3_style_power_at_threshold = rejection_rate(400, 240, MEE_BPS, **kw)["rate"]
        v5_power_at_pe = rejection_rate(400, 240, PE_BPS, **kw)["rate"]
        assert v3_style_power_at_threshold < 0.20         # ≈ α (capped)
        assert v5_power_at_pe > 0.40                        # not capped
        assert v5_power_at_pe > 3 * v3_style_power_at_threshold


class TestProtocolBranches:

    def test_infeasible_under_high_variance(self):
        hi = PilotNull.from_marginal(80.0, 0.35, 20)
        res = run_protocol(hi, seed=6, trials=80, boots=250)
        v = res["verdict"]
        assert v["outcome"] == INFEASIBLE
        assert v["power_feasible"] is False
        assert v["activation_ready"] is False
        assert res["power"]["final_t"] is None
        # No parameter walked: MEE/PE/block are the frozen inputs verbatim.
        assert res["decision_rule"]["mee_bps_per_session"] == 3.0
        assert res["decision_rule"]["pe_bps_per_session"] == 6.0
        assert res["activation_simulation_block"]["final_n_sessions"] is None
        assert res["activation_simulation_block"]["mbb_block_length"] == 35

    def test_feasible_power_still_blocked_by_type_i(self):
        # Low variance → power reaches 0.80 at T=240, but the frozen percentile
        # MBB's type-I still exceeds 0.10 → the sim SEPARATES the two
        # characteristics and returns TYPE_I_EXCEEDED (not GO).
        low = PilotNull.from_marginal(8.0, 0.35, 20)
        res = run_protocol(low, seed=5, trials=200, boots=500)
        assert res["power"]["final_t"] == 240
        assert res["power"]["pass"] is True
        assert res["type_i"]["pass"] is False
        assert res["verdict"]["outcome"] == TYPE_I_EXCEEDED

    def test_outcome_logic_all_branches(self, monkeypatch):
        # Exercise the branch mapping deterministically by scripting the
        # rejection rates (power at mu=PE, type-I at mu=MEE).
        real_pilot = PilotNull("d" * 64, 0.35, 20.0, 20, "pilot-fit:real")

        def fake(power_rate, type_i_rate):
            def _f(n_trials, t_sessions, mean_bps, *a, **k):
                rate = power_rate if mean_bps == PE_BPS else type_i_rate
                return {"rate": rate, "ci95": [rate, rate],
                        "trials": n_trials, "rejections": int(rate * n_trials)}
            return _f

        # feasible power + controlled type-I -> GO_VALID, activation-ready
        monkeypatch.setattr(v5mod, "rejection_rate", fake(0.85, 0.08))
        res = run_protocol(real_pilot, seed=1, trials=10, boots=10)
        assert res["verdict"]["outcome"] == GO_VALID
        assert res["verdict"]["activation_ready"] is True
        assert res["power"]["final_t"] == 240

        # feasible power + type-I > 0.10 -> TYPE_I_EXCEEDED
        monkeypatch.setattr(v5mod, "rejection_rate", fake(0.85, 0.15))
        res = run_protocol(real_pilot, seed=1, trials=10, boots=10)
        assert res["verdict"]["outcome"] == TYPE_I_EXCEEDED
        assert res["verdict"]["activation_ready"] is False

        # power never reaches 0.80 -> INFEASIBLE (even with good type-I)
        monkeypatch.setattr(v5mod, "rejection_rate", fake(0.50, 0.08))
        res = run_protocol(real_pilot, seed=1, trials=10, boots=10)
        assert res["verdict"]["outcome"] == INFEASIBLE
        assert res["power"]["final_t"] is None

    def test_synthetic_pilot_never_activation_ready(self, monkeypatch):
        # Even if characteristics passed, a synthetic-default pilot can never be
        # activation-ready — a real blinded pilot fit is mandatory (§4.7).
        def _f(n_trials, t_sessions, mean_bps, *a, **k):
            rate = 0.90 if mean_bps == PE_BPS else 0.05
            return {"rate": rate, "ci95": [rate, rate], "trials": n_trials,
                    "rejections": int(rate * n_trials)}
        monkeypatch.setattr(v5mod, "rejection_rate", _f)
        res = run_protocol(SYNTHETIC_DEFAULT, seed=1, trials=10, boots=10)
        assert res["verdict"]["gate_pass"] is True
        assert res["verdict"]["activation_ready"] is False


class TestFrozenOutputs:

    def test_activation_block_has_all_required_fields(self):
        res = run_protocol(SYNTHETIC_DEFAULT, seed=2, trials=40, boots=200)
        block = res["activation_simulation_block"]
        for key in ("type_i_rate", "type_i_ci_95", "power_at_pe", "power_ci_95",
                    "final_n_sessions", "final_n_blocks", "mbb_block_length",
                    "dgp_parameters", "pilot_manifest_digest",
                    "simulation_code_commit", "simulation_seed"):
            assert key in block, key
        assert block["pilot_manifest_digest"] == SYNTHETIC_DEFAULT.pilot_manifest_digest
        assert block["simulation_seed"] == 2


class TestCli:

    def test_quick_smoke_end_to_end(self, tmp_path):
        out = tmp_path / "r.json"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--quick", "--seed", "1",
             "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        # exit 1 (gate not passable under the frozen inference) is a valid outcome
        assert proc.returncode in (0, 1), proc.stderr
        res = json.loads(out.read_text())
        assert res["section"] == "4.6"
        assert res["protocol_version"] == "v5"
        assert res["decision_rule"]["mee_bps_per_session"] == 3.0
        assert res["decision_rule"]["pe_bps_per_session"] == 6.0
        assert "activation_simulation_block" in res

    def test_pilot_fit_file_is_consumed(self, tmp_path):
        fit = tmp_path / "fit.json"
        fit.write_text(json.dumps({
            "pilot_manifest_digest": "b" * 64,
            "ar1_phi": 0.3,
            "long_run_std_upper90_bps": 20.0,
            "max_holding_days": 12,
            "source": "pilot-fit:unit",
        }))
        out = tmp_path / "r.json"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--quick", "--seed", "3",
             "--pilot-fit", str(fit), "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode in (0, 1), proc.stderr
        res = json.loads(out.read_text())
        assert res["pilot_fit"]["is_synthetic"] is False
        assert res["pilot_fit"]["pilot_manifest_digest"] == "b" * 64
        assert res["pilot_fit"]["mbb_block_length"] == mbb_block_length(12)
