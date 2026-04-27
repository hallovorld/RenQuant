"""Tests for kernel/model_acceptance.py (round-7, 2026-04-26).

User spec: "我们有没有机制进行模型accpetance verification，如果不通过的话，
继续用原来的模型跑E2E？这关系到工程稳定性和可用性！"

Without acceptance gates, a retrain produces a panel-ltr.json that
auto-replaces production. If the new model is broken (calibrator
collapsed, OOS IC negative, schema changed unexpectedly), live trades
use it from the next bar. These tests pin the gate semantics so the
defense doesn't drift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.model_acceptance import (   # noqa: E402
    AcceptanceGate,
    AcceptanceVerdict,
    DEFAULT_GATES,
    GateResult,
    ModelAcceptanceGate,
    promote,
    reject,
    rollback,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_artifact(path: Path, **kwargs) -> Path:
    """Write a synthetic artifact JSON with desired metadata."""
    data = {
        "kind":         kwargs.pop("kind", "panel_ltr_xgboost"),
        "feature_cols": kwargs.pop("feature_cols", ["a", "b", "c"]),
        "metadata": {
            "oos_mean_ic":    kwargs.pop("oos_mean_ic", 0.05),
            "n_unique_prob_y": kwargs.pop("n_unique_prob_y", 10),
            "pool_ic":        kwargs.pop("pool_ic", 0.01),
            **kwargs,
        },
    }
    path.write_text(json.dumps(data))
    return path


# ── G1 Schema compatibility ───────────────────────────────────────────────────

class TestG1Schema:
    def test_same_length_passes(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", feature_cols=["a", "b", "c"])
        a = _write_artifact(tmp_path / "a.json", feature_cols=["a", "b", "c"])
        v = ModelAcceptanceGate().evaluate(s, a)
        g1 = next(r for r in v.results if r.name == "G1_schema")
        assert g1.passed

    def test_superset_passes(self, tmp_path):
        """Adding macro features = superset = OK."""
        s = _write_artifact(tmp_path / "s.json", feature_cols=["a", "b", "c", "vix_level_z"])
        a = _write_artifact(tmp_path / "a.json", feature_cols=["a", "b", "c"])
        v = ModelAcceptanceGate().evaluate(s, a)
        g1 = next(r for r in v.results if r.name == "G1_schema")
        assert g1.passed
        assert "superset" in g1.detail.lower()

    def test_shrinkage_fails(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", feature_cols=["a", "b"])
        a = _write_artifact(tmp_path / "a.json", feature_cols=["a", "b", "c"])
        v = ModelAcceptanceGate().evaluate(s, a)
        g1 = next(r for r in v.results if r.name == "G1_schema")
        assert not g1.passed

    def test_arbitrary_diff_fails(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", feature_cols=["x", "y", "z"])
        a = _write_artifact(tmp_path / "a.json", feature_cols=["a", "b", "c"])
        v = ModelAcceptanceGate().evaluate(s, a)
        g1 = next(r for r in v.results if r.name == "G1_schema")
        assert not g1.passed

    def test_missing_feature_cols_fails(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"kind": "x", "metadata": {}}))
        v = ModelAcceptanceGate().evaluate(path)
        g1 = next(r for r in v.results if r.name == "G1_schema")
        assert not g1.passed
        assert "missing feature_cols" in g1.detail


# ── G2 Calibrator non-collapse ────────────────────────────────────────────────

class TestG2CalibratorUnique:
    def test_5_unique_passes(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", n_unique_prob_y=5)
        v = ModelAcceptanceGate().evaluate(s)
        g2 = next(r for r in v.results if r.name == "G2_calibrator_unique")
        assert g2.passed

    def test_4_unique_fails(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", n_unique_prob_y=4)
        v = ModelAcceptanceGate().evaluate(s)
        g2 = next(r for r in v.results if r.name == "G2_calibrator_unique")
        assert not g2.passed

    def test_missing_metric_skips(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", n_unique_prob_y=None)
        # Remove the key entirely
        d = json.loads(s.read_text())
        d["metadata"].pop("n_unique_prob_y", None)
        s.write_text(json.dumps(d))
        v = ModelAcceptanceGate().evaluate(s)
        g2 = next(r for r in v.results if r.name == "G2_calibrator_unique")
        assert g2.passed   # skip = pass-open


# ── G3 Pool IC positive ───────────────────────────────────────────────────────

class TestG3PoolIcPositive:
    def test_positive_passes(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", pool_ic=0.001)
        v = ModelAcceptanceGate().evaluate(s)
        g3 = next(r for r in v.results if r.name == "G3_pool_ic_positive")
        assert g3.passed

    def test_zero_fails(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", pool_ic=0.0)
        v = ModelAcceptanceGate().evaluate(s)
        g3 = next(r for r in v.results if r.name == "G3_pool_ic_positive")
        assert not g3.passed

    def test_negative_fails(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", pool_ic=-0.005)
        v = ModelAcceptanceGate().evaluate(s)
        g3 = next(r for r in v.results if r.name == "G3_pool_ic_positive")
        assert not g3.passed


# ── G4 OOS IC vs prior ────────────────────────────────────────────────────────

class TestG4VsPrior:
    def test_no_prior_positive_passes(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", oos_mean_ic=0.04)
        v = ModelAcceptanceGate().evaluate(s)   # no active
        g4 = next(r for r in v.results if r.name == "G4_oos_ic_vs_prior")
        assert g4.passed

    def test_no_prior_negative_fails(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", oos_mean_ic=-0.005)
        v = ModelAcceptanceGate().evaluate(s)
        g4 = next(r for r in v.results if r.name == "G4_oos_ic_vs_prior")
        assert not g4.passed

    def test_5pct_degradation_passes(self, tmp_path):
        """Phase 1 (2026-04-26): default tightened 30% → 5%.
        Prior 0.05, new 0.0476 → 4.8% drop (just under 5% threshold)."""
        s = _write_artifact(tmp_path / "s.json", oos_mean_ic=0.0476)
        a = _write_artifact(tmp_path / "a.json", oos_mean_ic=0.05)
        v = ModelAcceptanceGate().evaluate(s, a)
        g4 = next(r for r in v.results if r.name == "G4_oos_ic_vs_prior")
        assert g4.passed

    def test_18pct_degradation_fails_at_5pct_default(self, tmp_path):
        """Phase 1 case: this is the actual macro-vs-prod case
        (0.0482 → 0.0393, -18.5%). Pre-Phase-1 (30% default) ACCEPTED
        this; Phase 1 (5% default) REJECTS — protecting prod from a
        worse model auto-promoting."""
        s = _write_artifact(tmp_path / "s.json", oos_mean_ic=0.0393)
        a = _write_artifact(tmp_path / "a.json", oos_mean_ic=0.0482)
        v = ModelAcceptanceGate().evaluate(s, a)
        g4 = next(r for r in v.results if r.name == "G4_oos_ic_vs_prior")
        assert not g4.passed

    def test_50pct_degradation_fails(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", oos_mean_ic=0.025)
        a = _write_artifact(tmp_path / "a.json", oos_mean_ic=0.05)
        v = ModelAcceptanceGate().evaluate(s, a)
        g4 = next(r for r in v.results if r.name == "G4_oos_ic_vs_prior")
        assert not g4.passed

    def test_v5_negative_ic_blocked(self, tmp_path):
        """The actual v5 case: prior +0.0326, new -0.0008 → must FAIL."""
        s = _write_artifact(tmp_path / "s.json", oos_mean_ic=-0.0008)
        a = _write_artifact(tmp_path / "a.json", oos_mean_ic=0.0326)
        v = ModelAcceptanceGate().evaluate(s, a)
        g4 = next(r for r in v.results if r.name == "G4_oos_ic_vs_prior")
        assert not g4.passed
        assert not v.all_hard_passed   # full verdict: REJECT


# ── G5/G6 (smoke + range) — soft pass when missing ────────────────────────────

class TestG5G6SmokeOptional:
    def test_g5_skips_when_no_range(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json")
        v = ModelAcceptanceGate().evaluate(s)
        g5 = next(r for r in v.results if r.name == "G5_score_range")
        assert g5.passed   # skip = pass-open

    def test_g5_blocks_constant_output(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", score_sample_range=[0.5, 0.5005])
        a = _write_artifact(tmp_path / "a.json", score_sample_range=[0.0, 1.0])
        v = ModelAcceptanceGate().evaluate(s, a)
        g5 = next(r for r in v.results if r.name == "G5_score_range")
        assert not g5.passed   # tiny span → fail

    def test_g6_smoke_test_blocks_nan(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json",
                             inference_smoke_test={"n": 32, "all_finite": False})
        v = ModelAcceptanceGate().evaluate(s)
        g6 = next(r for r in v.results if r.name == "G6_inference_smoke")
        assert not g6.passed


# ── Aggregated verdict + soft warnings ────────────────────────────────────────

class TestVerdictAggregation:
    def test_all_hard_pass_no_soft_warns(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json",
                             oos_mean_ic=0.04, n_unique_prob_y=15, pool_ic=0.01)
        v = ModelAcceptanceGate().evaluate(s)
        assert v.all_hard_passed
        assert len(v.hard_failures()) == 0

    def test_g7_below_floor_blocks_promotion(self, tmp_path):
        """Phase 1 (2026-04-26): G7 hardened from soft → hard. IC below
        floor (0.02) now BLOCKS — pre-Phase-1 it passed with soft warning."""
        s = _write_artifact(tmp_path / "s.json",
                             oos_mean_ic=0.005, n_unique_prob_y=10, pool_ic=0.001)
        v = ModelAcceptanceGate().evaluate(s)
        assert not v.all_hard_passed
        g7 = next(r for r in v.results if r.name == "G7_oos_ic_floor")
        assert g7.severity == "hard"
        assert not g7.passed

    def test_g8_still_soft_by_default(self, tmp_path):
        """G8 (per-bar variance) remains soft. Phase 1 only hardened G7."""
        g8 = next(g for g in DEFAULT_GATES if g.name == "G8_per_ticker_variance")
        assert g8.severity == "soft"


# ── Phase 1: config-driven thresholds ─────────────────────────────────────────

class TestConfigDrivenThresholds:
    """Phase 1 (2026-04-26): operators can override gate thresholds via
    `acceptance` config block in strategy_config.json without forking
    gate code. Used during exploratory rebuilds (relax G4) or strict
    promotion windows (raise G7 floor)."""

    def test_g4_max_degradation_configurable(self, tmp_path):
        """Loosen G4 to 25%: macro-vs-prod (-18.5%) now passes."""
        s = _write_artifact(tmp_path / "s.json", oos_mean_ic=0.0393)
        a = _write_artifact(tmp_path / "a.json", oos_mean_ic=0.0482)
        # Default 5% → fails
        v_default = ModelAcceptanceGate().evaluate(s, a)
        g4_default = next(r for r in v_default.results if r.name == "G4_oos_ic_vs_prior")
        assert not g4_default.passed
        # Configured 25% → passes
        v_loose = ModelAcceptanceGate(config={"g4_max_degradation": 0.25}).evaluate(s, a)
        g4_loose = next(r for r in v_loose.results if r.name == "G4_oos_ic_vs_prior")
        assert g4_loose.passed

    def test_g7_severity_configurable_to_soft(self, tmp_path):
        """Operator can downgrade G7 to soft for an exploratory run."""
        s = _write_artifact(tmp_path / "s.json", oos_mean_ic=0.005)
        v = ModelAcceptanceGate(config={"g7_severity": "soft"}).evaluate(s)
        g7 = next(r for r in v.results if r.name == "G7_oos_ic_floor")
        assert g7.severity == "soft"
        assert not g7.passed   # still failing — but as a soft warning
        # G7 soft means it's not in hard_failures
        hard_fails = v.hard_failures()
        assert not any(r.name == "G7_oos_ic_floor" for r in hard_fails)

    def test_g7_floor_configurable_higher(self, tmp_path):
        """Raise G7 floor to 0.05: previously-passing 0.03 now fails."""
        s = _write_artifact(tmp_path / "s.json", oos_mean_ic=0.03)
        v_default = ModelAcceptanceGate().evaluate(s)
        g7_default = next(r for r in v_default.results if r.name == "G7_oos_ic_floor")
        assert g7_default.passed
        v_strict = ModelAcceptanceGate(config={"g7_floor": 0.05}).evaluate(s)
        g7_strict = next(r for r in v_strict.results if r.name == "G7_oos_ic_floor")
        assert not g7_strict.passed

    def test_empty_config_uses_defaults(self, tmp_path):
        """An empty acceptance config block must NOT crash; falls back
        to Phase-1 hardened defaults."""
        s = _write_artifact(tmp_path / "s.json", oos_mean_ic=0.04)
        v = ModelAcceptanceGate(config={}).evaluate(s)
        # G4 + G7 should both pass (0.04 > 0.02 floor, no prior)
        g4 = next(r for r in v.results if r.name == "G4_oos_ic_vs_prior")
        g7 = next(r for r in v.results if r.name == "G7_oos_ic_floor")
        assert g4.passed and g7.passed
        assert g7.severity == "hard"   # Phase 1 default

    def test_strategy_config_block_loadable(self):
        """The acceptance block we wrote into strategy_config.json must
        be loadable + valid (no typos in keys / values)."""
        config_path = REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.json"
        cfg = json.loads(config_path.read_text())
        acc = cfg.get("acceptance", {})
        assert acc.get("enabled") is True
        assert acc.get("g4_max_degradation") == 0.05
        assert acc.get("g7_severity") == "hard"
        # Sanity: building gates from this config doesn't crash
        gates = ModelAcceptanceGate(config=acc).gates
        # Phase 1 had 8 gates; Phase 2 added G9/G10/G11 → 11
        assert len(gates) == 11


# ── Atomic swap (promote / reject / rollback) ─────────────────────────────────

class TestAtomicSwap:
    def test_promote_swaps_files(self, tmp_path):
        active = tmp_path / "panel-ltr.json"
        active.write_text(json.dumps({"kind": "old"}))
        staging = tmp_path / "panel-ltr.staging.json"
        staging.write_text(json.dumps({"kind": "new"}))
        promote(staging, active)
        assert json.loads(active.read_text())["kind"] == "new"
        prev = active.with_suffix(".previous.json")
        assert prev.exists()
        assert json.loads(prev.read_text())["kind"] == "old"
        assert not staging.exists()

    def test_promote_no_prior_works(self, tmp_path):
        active = tmp_path / "panel-ltr.json"
        staging = tmp_path / "panel-ltr.staging.json"
        staging.write_text(json.dumps({"kind": "new"}))
        promote(staging, active)
        assert json.loads(active.read_text())["kind"] == "new"

    def test_promote_missing_staging_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            promote(tmp_path / "missing.json", tmp_path / "active.json")

    def test_reject_archives_staging(self, tmp_path):
        active = tmp_path / "panel-ltr.json"
        active.write_text(json.dumps({"kind": "GOOD"}))
        staging = tmp_path / "panel-ltr.staging.json"
        staging.write_text(json.dumps({"kind": "BAD"}))
        verdict = AcceptanceVerdict(
            all_hard_passed=False,
            results=[GateResult("G_x", "hard", False, 0.0, 1.0, "test")],
        )
        reject(staging, tmp_path / "_acceptance_log", verdict)
        assert not staging.exists()
        assert json.loads(active.read_text())["kind"] == "GOOD"   # active untouched
        archived = list((tmp_path / "_acceptance_log").glob("*REJECTED*"))
        assert len(archived) >= 1

    def test_rollback_restores_previous(self, tmp_path):
        active = tmp_path / "panel-ltr.json"
        active.write_text(json.dumps({"kind": "BAD-NEW"}))
        prev = tmp_path / "panel-ltr.previous.json"
        prev.write_text(json.dumps({"kind": "GOOD-OLD"}))
        rollback(active)
        assert json.loads(active.read_text())["kind"] == "GOOD-OLD"
        assert not prev.exists()
        # Auto-archive of the bad-new should exist
        archive_dir = tmp_path / "_acceptance_log"
        assert archive_dir.exists()
        bads = list(archive_dir.glob("auto-rollback-*"))
        assert len(bads) >= 1


# ── Verdict summary string format (for ntfy / log) ────────────────────────────

class TestVerdictSummary:
    def test_summary_includes_all_gates_and_verdict(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json",
                             oos_mean_ic=0.04, n_unique_prob_y=10, pool_ic=0.01)
        v = ModelAcceptanceGate().evaluate(s)
        out = v.summary()
        for gate in DEFAULT_GATES:
            assert gate.name in out
        assert "VERDICT: ACCEPT" in out

    def test_summary_marks_reject_clearly(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json",
                             oos_mean_ic=-0.001, n_unique_prob_y=2, pool_ic=-0.005)
        v = ModelAcceptanceGate().evaluate(s)
        out = v.summary()
        assert "VERDICT: REJECT" in out


# ── Defensive: gate raising an exception ──────────────────────────────────────

class TestGateException:
    def test_buggy_gate_returns_failure_result(self, tmp_path):
        """If a gate's check() raises, verdict treats it as hard fail
        (defense in depth)."""
        def buggy(staging, active):
            raise RuntimeError("boom")

        s = _write_artifact(tmp_path / "s.json")
        gates = [AcceptanceGate("buggy_gate", "hard", buggy)]
        v = ModelAcceptanceGate(gates=gates).evaluate(s)
        r = next(r for r in v.results if r.name == "buggy_gate")
        assert not r.passed
        assert "RuntimeError" in r.detail
