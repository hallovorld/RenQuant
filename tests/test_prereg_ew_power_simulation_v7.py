"""Tests for the §4.6 v7 studentized-t-bound pre-activation simulation.

Covers the properties the amendment requires PLUS the load-bearing empirical
finding this module produced:

* the studentized-t bound REUSES the merged v5/v6 moving-block resample DRAW
  structure (same rng draws → same block start indices), i.e. it studentizes the
  SAME bootstrap, it does not re-invent the resampling;
* the per-resample block-based SE is the non-overlapping batch-means estimator,
  which for the MEAN is algebraically the delete-one-block jackknife SE — asserted
  equal;
* the strengthened §4.6 acceptance gate — point <= 0.10 AND 95% MC-CI upper <= 0.12
  AND a one-sided Clopper-Pearson 95% upper bound <= alpha, in EVERY frozen DGP
  cell (conjunction / multiplicity) at a fixed N (no optional stopping);
* the frozen DGP suite generators (persistence, GARCH volatility clustering, heavy
  t5 tails, skew, missing sessions) are all matched to the same marginal std;
* the deep-null (mu=0) rejection is tiny → the sim is sound;
* determinism under a fixed seed;
* THE FINDING — studentized-t does NOT control the boundary type-I: it roughly
  HALVES the percentile excess (~0.15 → ~0.11) but still does not clear <= 0.10 at
  the few-blocks operating point (b=35 over T=240 ≈ 7 blocks; SE uses L=6 batches),
  so the gate correctly returns INFEASIBLE — INFERENCE METHOD UNCALIBRATED (→ v8).
  This is asserted, not papered over.

Statistical assertions use FIXED seeds so the numbers are non-flaky; trial and
bootstrap counts are reduced for test speed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import prereg_ew_power_simulation_v7 as v7mod  # noqa: E402
from prereg_ew_power_simulation_v7 import (  # noqa: E402
    ALPHA,
    CP_ACCEPT_CONF,
    DGP_SUITE,
    GO_VALID,
    INFEASIBLE_INFERENCE,
    INFEASIBLE_NOISE,
    MEE_BPS,
    PE_BPS,
    SE_FLOOR,
    SYNTHETIC_DEFAULT,
    TYPE_I_CI_UPPER_MAX,
    TYPE_I_REQUIREMENT,
    PilotNull,
    _batch_means_se,
    _cp_upper,
    _mbb_resample_matrix,
    rejection_rate,
    run_protocol_v7,
    simulate_dgp,
    studentized_block_lower_bound,
    type_i_accepted,
    v7_outcome,
    validate_boundary_type_i,
)

SCRIPT = SCRIPTS / "prereg_ew_power_simulation_v7.py"
P = SYNTHETIC_DEFAULT


class TestBoundConstruction:

    def test_reuses_mbb_draw_structure(self):
        # The studentized bound resamples via _mbb_resample_matrix, whose draw
        # structure (rng.integers(0, T-b+1, (n_boot, n_blocks))) matches the merged
        # primitive. Under an equal seed the sampled block-start grid is identical.
        d = np.random.default_rng(1).normal(3.0, 25.0, 240)
        import prereg_ew_power_simulation as base
        r1 = np.random.default_rng(42)
        mat = _mbb_resample_matrix(d, 35, r1, n_boot=200)
        # merged mbb_lower_bound draws the SAME starts first; reconstruct them.
        r2 = np.random.default_rng(42)
        n_blocks = int(np.ceil(240 / 35))
        starts = r2.integers(0, 240 - 35 + 1, size=(200, n_blocks))
        idx = (starts[:, :, None] + np.arange(35)[None, None, :]).reshape(200, -1)
        expected = d[idx[:, :240]]
        assert np.array_equal(mat, expected)
        assert base.mbb_lower_bound  # primitive importable / reused elsewhere

    def test_batch_means_equals_delete_block_jackknife(self):
        # For the MEAN, se via non-overlapping batch means == delete-one-block
        # jackknife SE over the same blocks. Assert the identity directly.
        rng = np.random.default_rng(3)
        d = rng.normal(3.0, 25.0, 240)
        b = 35
        se_bm = float(_batch_means_se(d[None, :], b)[0])
        L = 240 // b
        Y = d[:L * b].reshape(L, b).mean(axis=1)
        grand = Y.mean()
        theta_j = (L * grand - Y) / (L - 1)          # delete-one-block means
        var_jack = ((L - 1) / L) * np.sum((theta_j - theta_j.mean()) ** 2)
        assert se_bm == pytest.approx(float(np.sqrt(var_jack)), rel=1e-12)

    def test_bound_requires_two_blocks(self):
        with pytest.raises(ValueError):
            _batch_means_se(np.zeros((1, 40)), 35)  # floor(40/35)=1 block

    def test_bound_deterministic_under_seed(self):
        d = np.random.default_rng(2).normal(3.0, 25.0, 240)
        a = studentized_block_lower_bound(d, 35, np.random.default_rng(7), n_boot=1500)
        b = studentized_block_lower_bound(d, 35, np.random.default_rng(7), n_boot=1500)
        assert a == b

    def test_bound_is_below_point_estimate(self):
        # A one-sided LOWER bound must sit below the sample mean.
        d = np.random.default_rng(4).normal(8.0, 25.0, 240)
        lb = studentized_block_lower_bound(d, 35, np.random.default_rng(1), n_boot=1500)
        assert lb < float(d.mean())

    def test_degenerate_series_se_floor(self):
        # Constant series ⇒ se_hat and every se* are 0 → floored → finite bound == mean.
        d = np.full(240, 3.0)
        lb = studentized_block_lower_bound(d, 35, np.random.default_rng(0), n_boot=500)
        assert np.isfinite(lb)
        assert lb == pytest.approx(3.0, abs=1e-6)


class TestClopperPearsonGate:

    def test_cp_upper_matches_beta_inverse(self):
        assert _cp_upper(50, 1000) == pytest.approx(
            float(stats.beta.ppf(CP_ACCEPT_CONF, 51, 950)), rel=1e-12)

    def test_cp_upper_all_success_is_one(self):
        assert _cp_upper(100, 100) == 1.0

    def test_cp_upper_above_point_estimate(self):
        # The one-sided upper bound exceeds the point estimate.
        assert _cp_upper(100, 1000) > 0.10

    @pytest.mark.parametrize("k,n,accept", [
        (400, 5000, True),    # rate 0.08, CP upper < 0.10 ⇒ accept
        (560, 5000, False),   # rate 0.112 ⇒ reject (the v7 finding regime)
        (500, 5000, False),   # rate exactly 0.10 ⇒ CP upper > 0.10 ⇒ reject
    ])
    def test_type_i_accepted_rule(self, k, n, accept):
        rate = k / n
        half = 1.96 * (rate * (1 - rate) / n) ** 0.5
        cell = {"rate": rate, "ci95": [rate - half, rate + half],
                "cp95_upper": _cp_upper(k, n), "rejections": k, "trials": n}
        got, _ = type_i_accepted(cell)
        assert got is accept

    def test_ci_upper_margin_still_required(self):
        # Even if the CP rule passed, a wide 95% MC CI upper (> 0.12) must block —
        # both retained bars AND the binomial rule are required.
        cell = {"rate": 0.09, "ci95": [0.06, 0.125], "cp95_upper": 0.099,
                "rejections": 9, "trials": 100}
        got, detail = type_i_accepted(cell)
        assert got is False
        assert detail["binomial_accept"] is True  # CP passed, retained bar failed

    def test_constants_frozen(self):
        assert TYPE_I_REQUIREMENT == 0.10
        assert TYPE_I_CI_UPPER_MAX == 0.12
        assert ALPHA == 0.10
        assert CP_ACCEPT_CONF == 0.95
        assert SE_FLOOR == 1e-9


class TestOutcomeLogic:

    @pytest.mark.parametrize("calibrated,feasible,outcome", [
        (True, True, GO_VALID),
        (False, True, INFEASIBLE_INFERENCE),
        (False, False, INFEASIBLE_INFERENCE),
        (True, False, INFEASIBLE_NOISE),
    ])
    def test_outcome_mapping(self, calibrated, feasible, outcome):
        assert v7_outcome(calibrated, feasible) == outcome


class TestDgpSuite:

    def test_suite_has_core_and_breadth(self):
        names = [n for n, _, _ in DGP_SUITE]
        assert "iid" in names and "ar1" in names           # task-required core
        # codex breadth: persistence, vol-clustering, heavy tails, skew, missing
        for extra in ("ar1_persist", "garch", "heavy_t5", "skew", "missing"):
            assert extra in names, extra

    @pytest.mark.parametrize("kind,phi", [
        ("gaussian", 0.0), ("gaussian", 0.35), ("t5", 0.35), ("skew", 0.35),
        ("garch", 0.35), ("missing", 0.35)])
    def test_marginal_std_matched(self, kind, phi):
        # Every DGP is matched to the SAME marginal std (25 bps) so cells differ
        # only in dependence / tails / skew — a like-for-like calibration domain.
        rng = np.random.default_rng(11)
        x = simulate_dgp(kind, 4000, 300, 3.0, 25.0, phi, rng)
        assert x.shape == (4000, 300)
        assert x.std() == pytest.approx(25.0, rel=0.12)
        assert x.mean() == pytest.approx(3.0, abs=1.0)

    def test_t5_has_heavier_tails_than_gaussian(self):
        rng = np.random.default_rng(12)
        g = simulate_dgp("gaussian", 2000, 300, 0.0, 25.0, 0.0, rng)
        t = simulate_dgp("t5", 2000, 300, 0.0, 25.0, 0.0, rng)
        assert stats.kurtosis(t.ravel()) > stats.kurtosis(g.ravel()) + 0.5

    def test_skew_is_skewed(self):
        rng = np.random.default_rng(13)
        s = simulate_dgp("skew", 2000, 300, 0.0, 25.0, 0.0, rng)
        assert abs(stats.skew(s.ravel())) > 0.3

    def test_missing_preserves_length(self):
        rng = np.random.default_rng(14)
        m = simulate_dgp("missing", 100, 240, 3.0, 25.0, 0.35, rng)
        assert m.shape == (100, 240)


class TestDeterminismAndSanity:

    def test_rejection_rate_deterministic(self):
        kw = dict(n_boot=400, method="studentized", dgp_kind="gaussian")
        a = rejection_rate(60, 240, PE_BPS, 25.0, 0.35, 35, MEE_BPS, seed=9, **kw)
        b = rejection_rate(60, 240, PE_BPS, 25.0, 0.35, 35, MEE_BPS, seed=9, **kw)
        assert a == b

    def test_deep_null_barely_rejects(self):
        # mu = 0 (far below MEE=3): the bound almost never rejects ⇒ sim is sound and
        # the elevated boundary rate is a GENUINE calibration fact.
        r = rejection_rate(1500, 240, 0.0, P.marginal_sigma_bps, P.ar1_phi,
                           P.block_length, MEE_BPS, seed=24, n_boot=1500,
                           method="studentized")
        assert r["rate"] < 0.03

    def test_huge_effect_always_rejects(self):
        r = rejection_rate(30, 240, 60.0, 10.0, 0.0, 35, MEE_BPS, seed=5,
                           n_boot=300, method="studentized")
        assert r["rate"] == 1.0

    def test_power_increases_with_T(self):
        kw = dict(mean_bps=PE_BPS, sigma_bps=P.marginal_sigma_bps, phi=P.ar1_phi,
                  b=P.block_length, threshold_bps=MEE_BPS, seed=11, n_boot=1500,
                  method="studentized", dgp_kind="gaussian")
        r240 = rejection_rate(500, 240, **kw)["rate"]
        r480 = rejection_rate(500, 480, **kw)["rate"]
        assert r480 > r240


class TestFindingStudentizedDoesNotControl:

    def test_studentized_below_percentile_but_over_alpha(self):
        # THE FINDING. On the binding T=240 boundary cell the studentized bound's
        # type-I is materially BELOW the percentile bound's (it corrects most of the
        # variance-scale excess) yet still ABOVE the nominal 0.10 — so it does NOT
        # calibrate the gate.
        common = dict(t_sessions=240, mean_bps=MEE_BPS, sigma_bps=P.marginal_sigma_bps,
                      phi=P.ar1_phi, b=P.block_length, threshold_bps=MEE_BPS,
                      n_boot=1500, dgp_kind="gaussian")
        stud = rejection_rate(1500, seed=23, method="studentized", **common)
        pct = rejection_rate(1500, seed=23, method="percentile", **common)
        assert stud["rate"] < pct["rate"]                 # corrects most of the excess
        assert stud["rate"] > TYPE_I_REQUIREMENT          # but still over-rejects
        assert pct["rate"] > 0.12                          # percentile is worse

    def test_validate_returns_uncalibrated_verdict(self):
        # Small-config validation on the core cells: studentized does NOT control.
        core = tuple(s for s in DGP_SUITE if s[0] in ("iid", "ar1"))
        val = validate_boundary_type_i(P, trials=1500, boots=1500, ts=(240, 480),
                                       seed=20260719, suite=core)
        assert val["studentized_controls_all_cells"] is False
        assert val["percentile_anticonservative"] is True
        # every studentized cell is materially below its percentile peer
        for name, cell in val["cells"].items():
            assert cell["studentized"]["rate"] <= cell["percentile"]["rate"] + 0.01, name


class TestProtocolBranches:

    _real = PilotNull("d" * 64, 0.35, 20.0, 20, "pilot-fit:real")

    @staticmethod
    def _fake(power_rate, type_i_rate, ci_upper=None, cp_upper=None):
        ciu = type_i_rate if ci_upper is None else ci_upper

        def _f(n_trials, t_sessions, mean_bps, *a, **k):
            if mean_bps == PE_BPS:
                return {"rate": power_rate, "ci95": [power_rate, power_rate],
                        "cp95_upper": min(1.0, power_rate + 0.02), "rejections": 0,
                        "trials": n_trials}
            cpu = type_i_rate if cp_upper is None else cp_upper
            return {"rate": type_i_rate, "ci95": [max(0.0, type_i_rate - 0.005), ciu],
                    "cp95_upper": cpu, "rejections": int(type_i_rate * n_trials),
                    "trials": n_trials}
        return _f

    def test_calibrated_and_feasible_is_go(self, monkeypatch):
        monkeypatch.setattr(v7mod, "rejection_rate", self._fake(0.85, 0.07, cp_upper=0.09))
        res = run_protocol_v7(self._real, seed=1, trials=10, boots=10)
        assert res["verdict"]["outcome"] == GO_VALID
        assert res["verdict"]["activation_ready"] is True

    def test_uncalibrated_is_inference_infeasible(self, monkeypatch):
        # studentized boundary type-I 0.11 (CP upper 0.12 > alpha) ⇒ uncalibrated,
        # even though power is feasible. This is the finding's protocol outcome.
        monkeypatch.setattr(v7mod, "rejection_rate", self._fake(0.85, 0.11, cp_upper=0.12))
        res = run_protocol_v7(self._real, seed=1, trials=10, boots=10)
        assert res["verdict"]["outcome"] == INFEASIBLE_INFERENCE
        assert res["calibration_gate"]["inference_calibrated"] is False

    def test_binomial_rule_blocks_point_on_alpha(self, monkeypatch):
        # Point exactly 0.10 but CP upper 0.108 > alpha ⇒ the strengthened rule blocks
        # it (a point estimate that lands on alpha does NOT prove control).
        monkeypatch.setattr(v7mod, "rejection_rate",
                            self._fake(0.85, 0.10, ci_upper=0.115, cp_upper=0.108))
        res = run_protocol_v7(self._real, seed=1, trials=10, boots=10)
        assert res["verdict"]["outcome"] == INFEASIBLE_INFERENCE

    def test_calibrated_but_power_infeasible_is_noise(self, monkeypatch):
        monkeypatch.setattr(v7mod, "rejection_rate", self._fake(0.50, 0.07, cp_upper=0.09))
        res = run_protocol_v7(self._real, seed=1, trials=10, boots=10)
        assert res["verdict"]["outcome"] == INFEASIBLE_NOISE
        assert res["power"]["final_t"] is None

    def test_synthetic_never_activation_ready(self, monkeypatch):
        monkeypatch.setattr(v7mod, "rejection_rate", self._fake(0.90, 0.05, cp_upper=0.07))
        res = run_protocol_v7(SYNTHETIC_DEFAULT, seed=1, trials=10, boots=10)
        assert res["verdict"]["gate_pass"] is True
        assert res["verdict"]["activation_ready"] is False

    def test_no_parameter_walked(self, monkeypatch):
        monkeypatch.setattr(v7mod, "rejection_rate", self._fake(0.85, 0.11, cp_upper=0.12))
        res = run_protocol_v7(self._real, seed=1, trials=10, boots=10)
        assert res["decision_rule"]["mee_bps_per_session"] == 3.0
        assert res["decision_rule"]["pe_bps_per_session"] == 6.0


class TestCli:

    def test_protocol_quick_smoke(self, tmp_path):
        out = tmp_path / "r.json"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--quick", "--seed", "1", "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode in (0, 1), proc.stderr
        res = json.loads(out.read_text())
        assert res["protocol_version"] == "v7"
        assert res["inference_method"] == "studentized"
        assert res["decision_rule"]["mee_bps_per_session"] == 3.0

    def test_validate_core_only_smoke(self, tmp_path):
        out = tmp_path / "v.json"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--validate", "--core-only", "--seed",
             "20260719", "--trials", "400", "--boots", "600", "--out", str(out)],
            capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr
        res = json.loads(out.read_text())
        assert set(res["validation"]["cells"]) == {
            "iid_T240", "iid_T480", "ar1_T240", "ar1_T480"}
        assert res["resolved_outcome"] in (
            GO_VALID, INFEASIBLE_NOISE, INFEASIBLE_INFERENCE)
