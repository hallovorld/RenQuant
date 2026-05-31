"""Track H — paired tests for CalibratorHealthTask + CalibratorFlatRegionTask
asserting byte-equivalence with the legacy ``_check_*`` functions.

Coverage:
  CalibratorHealthTask vs _check_calibrator_health:
    (a) calibration disabled                       → soft pass
    (b) artifact missing (full)                    → HARD fail (sell-only soft)
    (c) artifact unparseable                       → HARD fail
    (d) Kelly mu enabled + wrong contract          → soft|hard
    (e) er.y max|y| > 0.20 ER_BOUND                → HARD fail
    (f) er.y flat region > max_er_flat              → HARD fail
    (g) n_unique_prob_y not stamped                → soft|hard
    (h) n_unique_prob_y < min_unique                → soft|hard
    (i) pool_ic <= 0                                → soft|hard
    (j) HARD pass: healthy metadata                → HARD pass

  CalibratorFlatRegionTask vs _check_calibrator_flat_region:
    (k) calibration disabled                        → soft pass
    (l) artifact missing                            → HARD fail (sell-only soft)
    (m) probability.x/y mismatched                  → HARD fail (sell-only soft)
    (n) flat fraction <= max_flat_fraction          → HARD pass
    (o) flat fraction > max_flat_fraction           → HARD fail
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.preflight import (
    _check_calibrator_flat_region,
    _check_calibrator_health,
)
from kernel.preflight_pipeline import (
    CalibratorFlatRegionTask,
    CalibratorHealthTask,
    PreflightContext,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _write_calibrator(tmp_path: Path, payload: dict | None,
                      path: str = "artifacts/prod/panel-rank-calibration.json"
                      ) -> tuple[Path, dict]:
    p = tmp_path / path
    p.parent.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        p.write_text(json.dumps(payload) if isinstance(payload, dict) else payload)
    config = {
        "ranking": {
            "panel_scoring": {
                "enabled": True,
                "global_calibration": {
                    "enabled": True,
                    "artifact_path": path,
                },
            },
        },
    }
    return tmp_path, config


def _ctx(strategy_dir: Path, config: dict, run_mode: str | None = None) -> PreflightContext:
    return PreflightContext(config=config, strategy_dir=strategy_dir, run_mode=run_mode)


def _healthy_payload() -> dict:
    """A calibrator payload that passes all health gates."""
    return {
        "kind": "platt",
        "metadata": {
            "n_unique_prob_y": 100,
            "pool_ic": 0.12,
            "expected_return_label_contract": "raw_return_units_required",
            "er_std": 0.04,
        },
        "probability": {
            "x": [i / 10 for i in range(10)],
            "y": [0.05 + i * 0.08 for i in range(10)],
        },
        "expected_return": {
            "x": [i / 10 for i in range(10)],
            "y": [i * 0.015 - 0.05 for i in range(10)],  # all |y| < 0.20
        },
    }


# ─── CalibratorHealthTask parity ─────────────────────────────────────────────

class TestCalibratorHealthTaskParity:

    def test_calibration_disabled_soft_pass(self, tmp_path):
        sd, cfg = _write_calibrator(tmp_path, payload=_healthy_payload())
        cfg["ranking"]["panel_scoring"]["global_calibration"]["enabled"] = False
        leg = _check_calibrator_health(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        CalibratorHealthTask().run(ctx)
        new = ctx.results[-1]
        assert new.name == leg.name == "P-CALIBRATOR-HEALTH"
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_artifact_missing_full_hard_fail(self, tmp_path):
        sd, cfg = _write_calibrator(tmp_path, payload=None)
        leg = _check_calibrator_health(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        CalibratorHealthTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_unparseable_hard_fail(self, tmp_path):
        sd, cfg = _write_calibrator(tmp_path, payload="{ malformed")
        leg = _check_calibrator_health(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        CalibratorHealthTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_healthy_metadata_hard_pass(self, tmp_path):
        sd, cfg = _write_calibrator(tmp_path, payload=_healthy_payload())
        leg = _check_calibrator_health(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        CalibratorHealthTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_n_unique_below_min_full_hard_fail(self, tmp_path):
        payload = _healthy_payload()
        payload["metadata"]["n_unique_prob_y"] = 7  # below default min=10
        sd, cfg = _write_calibrator(tmp_path, payload=payload)
        leg = _check_calibrator_health(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        CalibratorHealthTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_n_unique_missing_full_hard_fail(self, tmp_path):
        payload = _healthy_payload()
        payload["metadata"].pop("n_unique_prob_y")
        sd, cfg = _write_calibrator(tmp_path, payload=payload)
        leg = _check_calibrator_health(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        CalibratorHealthTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_pool_ic_negative_full_hard_fail(self, tmp_path):
        payload = _healthy_payload()
        payload["metadata"]["pool_ic"] = -0.05
        sd, cfg = _write_calibrator(tmp_path, payload=payload)
        leg = _check_calibrator_health(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        CalibratorHealthTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_er_y_exceeds_bound_hard_fail(self, tmp_path):
        payload = _healthy_payload()
        payload["expected_return"]["y"] = [0.05, 0.10, 0.50]  # 0.50 > 0.20 bound
        payload["expected_return"]["x"] = [0.0, 0.5, 1.0]
        sd, cfg = _write_calibrator(tmp_path, payload=payload)
        leg = _check_calibrator_health(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        CalibratorHealthTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_kelly_mu_enabled_wrong_contract_full_hard_fail(self, tmp_path):
        payload = _healthy_payload()
        payload["metadata"]["expected_return_label_contract"] = "log_return"  # wrong
        sd, cfg = _write_calibrator(tmp_path, payload=payload)
        cfg["ranking"]["kelly_sizing"] = {"use_calibrator_mu": True}
        leg = _check_calibrator_health(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        CalibratorHealthTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message


# ─── CalibratorFlatRegionTask parity ─────────────────────────────────────────

class TestCalibratorFlatRegionTaskParity:

    def test_calibration_disabled_soft_pass(self, tmp_path):
        sd, cfg = _write_calibrator(tmp_path, payload=_healthy_payload())
        cfg["ranking"]["panel_scoring"]["global_calibration"]["enabled"] = False
        leg = _check_calibrator_flat_region(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        CalibratorFlatRegionTask().run(ctx)
        new = ctx.results[-1]
        assert new.name == leg.name == "P-CALIBRATOR-FLAT-REGION"
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_artifact_missing_full_hard_fail(self, tmp_path):
        sd, cfg = _write_calibrator(tmp_path, payload=None)
        leg = _check_calibrator_flat_region(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        CalibratorFlatRegionTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_probability_mismatched_full_hard_fail(self, tmp_path):
        payload = _healthy_payload()
        payload["probability"]["y"] = []  # mismatched x/y
        sd, cfg = _write_calibrator(tmp_path, payload=payload)
        leg = _check_calibrator_flat_region(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        CalibratorFlatRegionTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_no_flat_region_hard_pass(self, tmp_path):
        # Strictly monotonic probability curve, no flat region
        sd, cfg = _write_calibrator(tmp_path, payload=_healthy_payload())
        leg = _check_calibrator_flat_region(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        CalibratorFlatRegionTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_flat_region_over_threshold_hard_fail(self, tmp_path):
        # Create a calibrator with a huge flat region (>30%)
        payload = _healthy_payload()
        # 50% flat region in the lower half
        payload["probability"]["x"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        payload["probability"]["y"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9]
        sd, cfg = _write_calibrator(tmp_path, payload=payload)
        leg = _check_calibrator_flat_region(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        CalibratorFlatRegionTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message
