"""Track H — paired tests for ConfigFingerprintTask + MetaLabelArtifactContractTask
asserting byte-equivalence with the legacy ``_check_*`` functions on
documented branches.

These are the two largest checks (129 lines each); test coverage focuses on
the principal branches with byte-equivalence asserts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.preflight import (
    _check_config_fingerprint,
    _check_meta_label_artifact_contract,
)
from kernel.preflight_pipeline import (
    ConfigFingerprintTask,
    MetaLabelArtifactContractTask,
    PreflightContext,
    build_preflight_pipeline,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_strategy_dir(tmp_path: Path, artifact_payload=None,
                       artifact_path: str = "artifacts/prod/panel-ltr.alpha158_fund.json"
                       ) -> tuple[Path, dict]:
    art = tmp_path / artifact_path
    art.parent.mkdir(parents=True, exist_ok=True)
    if artifact_payload is not None:
        art.write_text(json.dumps(artifact_payload) if isinstance(artifact_payload, dict)
                       else artifact_payload)
    config = {
        "ranking": {"panel_scoring": {"artifact_path": artifact_path,
                                       "kind": "panel_ltr_xgboost"}}
    }
    return tmp_path, config


def _ctx(strategy_dir: Path, config: dict, run_mode: str | None = None) -> PreflightContext:
    return PreflightContext(config=config, strategy_dir=strategy_dir, run_mode=run_mode)


# ─── ConfigFingerprintTask parity ────────────────────────────────────────────

class TestConfigFingerprintTaskParity:

    def test_artifact_missing_hard_fail(self, tmp_path):
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=None)
        leg = _check_config_fingerprint(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        ConfigFingerprintTask().run(ctx)
        new = ctx.results[-1]
        assert new.name == leg.name == "P-CONFIG-FP"
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_artifact_unparseable_hard_fail(self, tmp_path):
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload="{ malformed")
        leg = _check_config_fingerprint(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        ConfigFingerprintTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_stored_fingerprint_missing_full_hard_fail(self, tmp_path):
        # Artifact has no config_fingerprint stamped
        payload = {"kind": "panel_ltr_xgboost", "feature_cols": ["KMID"]}
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=payload)
        leg = _check_config_fingerprint(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        ConfigFingerprintTask().run(ctx)
        new = ctx.results[-1]
        # The legacy may return soft (if config_consistency raises) OR hard;
        # whichever it does, the new should match.
        assert new.severity == leg.severity
        assert new.ok is leg.ok
        assert new.message == leg.message

    def test_fingerprint_mismatch_full_hard_fail(self, tmp_path):
        # Artifact has a fingerprint that won't match live computed fingerprint
        payload = {
            "kind": "panel_ltr_xgboost",
            "feature_cols": ["KMID"],
            "config_fingerprint": "sha256:nonsense-stored",
            "config_fingerprint_fields": {"watchlist": ["A", "B"]},
        }
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=payload)
        cfg["watchlist"] = ["X", "Y"]
        leg = _check_config_fingerprint(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        ConfigFingerprintTask().run(ctx)
        new = ctx.results[-1]
        # Bytewise — match whatever legacy returns
        assert new.severity == leg.severity
        assert new.ok is leg.ok
        assert new.message == leg.message


# ─── MetaLabelArtifactContractTask parity ────────────────────────────────────

class TestMetaLabelArtifactContractTaskParity:

    def test_meta_label_disabled_soft_pass(self, tmp_path):
        cfg = {"ranking": {"meta_label": {"enabled": False}}}
        leg = _check_meta_label_artifact_contract(config=cfg, strategy_dir=tmp_path)
        ctx = _ctx(tmp_path, cfg)
        MetaLabelArtifactContractTask().run(ctx)
        new = ctx.results[-1]
        assert new.name == leg.name == "P-META-LABEL"
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_enabled_but_no_artifact_path_full_hard_fail(self, tmp_path):
        cfg = {"ranking": {"meta_label": {"enabled": True}}}
        leg = _check_meta_label_artifact_contract(
            config=cfg, strategy_dir=tmp_path, run_mode="full")
        ctx = _ctx(tmp_path, cfg, run_mode="full")
        MetaLabelArtifactContractTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_artifact_missing_full_hard_fail(self, tmp_path):
        rel = "artifacts/prod/meta-label.json"
        cfg = {
            "ranking": {
                "meta_label": {
                    "enabled": True,
                    "artifact_path": rel,
                },
            },
        }
        # artifact file does NOT exist
        leg = _check_meta_label_artifact_contract(
            config=cfg, strategy_dir=tmp_path, run_mode="full")
        ctx = _ctx(tmp_path, cfg, run_mode="full")
        MetaLabelArtifactContractTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_valid_contract_hard_pass(self, tmp_path):
        rel = "artifacts/prod/meta-label.json"
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "kind": "meta_label_exit_xgb",
            "feature_cols": ["a", "b"],
            "booster_raw_json": "{\"version\":1}",
            "default_threshold": 0.5,
            "cv_metrics": {"auc_mean": 0.65},
            "training_data_summary": {
                "n_events": 500,
                "fwd_window_days": 10,
                "class_balance": 0.4,
            },
        }))
        cfg = {
            "ranking": {
                "meta_label": {
                    "enabled": True,
                    "artifact_path": rel,
                    "min_auc": 0.5,
                    "min_events": 100,
                },
            },
        }
        leg = _check_meta_label_artifact_contract(
            config=cfg, strategy_dir=tmp_path, run_mode="full")
        ctx = _ctx(tmp_path, cfg, run_mode="full")
        MetaLabelArtifactContractTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message


# ─── COMPLETE pipeline test: all 18 checks fire in order ─────────────────────

class TestFullPipeline:
    """The Track H goal: complete preflight via the new T/J/P pipeline."""

    def test_full_pipeline_has_18_checks(self, tmp_path):
        ctx = PreflightContext(config={}, strategy_dir=tmp_path)
        pipeline = build_preflight_pipeline()
        results = pipeline.run(ctx, strict=False)
        names = {r.name for r in results}
        expected = {
            "P-MODEL-ARTIFACT", "P-PANEL-CONTRACT", "P-BEST-ITER",
            "P-WF-GATE", "P-REGIME-IC",
            "P-CONFIG-FP", "P-WATCHLIST", "P-SECTOR-MAP", "P-CORR-METADATA",
            "P-NEWS-SENTIMENT-FRESHNESS",
            "P-CALIBRATOR-HEALTH", "P-CALIBRATOR-FLAT-REGION",
            "P-FEATURE-COVER", "P-RUN-ID",
            "P-META-LABEL",
            # P-BROKER-FILL-FRESHNESS added 2026-06-02 (audit finding 9).
            "P-STATE-FILE", "P-BROKER-CONNECT", "P-BROKER-FILL-FRESHNESS",
        }
        assert names == expected
        assert len(results) == 18

    def test_full_pipeline_order_mirrors_legacy(self, tmp_path):
        """Ordering should match kernel.preflight.run_preflight's ALL_CHECKS."""
        ctx = PreflightContext(config={}, strategy_dir=tmp_path)
        results = build_preflight_pipeline().run(ctx, strict=False)
        names = [r.name for r in results]
        # Artifact group first
        assert names[0:3] == ["P-MODEL-ARTIFACT", "P-PANEL-CONTRACT", "P-BEST-ITER"]
        # Gate group next
        assert names[3:5] == ["P-WF-GATE", "P-REGIME-IC"]
        # Identity group
        assert names[5:10] == ["P-CONFIG-FP", "P-WATCHLIST", "P-SECTOR-MAP",
                               "P-CORR-METADATA", "P-NEWS-SENTIMENT-FRESHNESS"]
        # Calibrator
        assert names[10:12] == ["P-CALIBRATOR-HEALTH", "P-CALIBRATOR-FLAT-REGION"]
        # NGBoost-aux
        assert names[12:14] == ["P-FEATURE-COVER", "P-RUN-ID"]
        # Meta-label
        assert names[14] == "P-META-LABEL"
        # State + broker last (P-BROKER-FILL-FRESHNESS added 2026-06-02
        # audit finding 9; runs after BROKER-CONNECT so the connection is
        # up before the freshness query).
        assert names[15:18] == [
            "P-STATE-FILE", "P-BROKER-CONNECT", "P-BROKER-FILL-FRESHNESS",
        ]
