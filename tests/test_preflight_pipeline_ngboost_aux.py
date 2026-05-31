"""Track H — paired tests for FeatureCoverageTask + ArtifactRunIdAlignmentTask
asserting byte-equivalence with the legacy ``_check_*`` functions.

Both checks share the ``_ngboost_activation()`` skip-if-disabled gate; tests
exercise the gate first then the substantive branches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.preflight import (
    _check_artifact_run_id_alignment,
    _check_feature_coverage,
)
from kernel.preflight_pipeline import (
    ArtifactRunIdAlignmentTask,
    FeatureCoverageTask,
    PreflightContext,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _write_artifacts(tmp_path: Path, panel_payload=None, ngb_payload=None,
                     panel_path="artifacts/prod/panel-ltr.alpha158_fund.json",
                     ngb_path="artifacts/prod/ngboost-head.json"):
    p_path = tmp_path / panel_path
    p_path.parent.mkdir(parents=True, exist_ok=True)
    if panel_payload is not None:
        p_path.write_text(json.dumps(panel_payload))
    n_path = tmp_path / ngb_path
    n_path.parent.mkdir(parents=True, exist_ok=True)
    if ngb_payload is not None:
        n_path.write_text(json.dumps(ngb_payload))
    return p_path, n_path


def _config(panel_path: str, ngb_path: str, ngb_enabled: bool = True) -> dict:
    return {
        "panel_ltr": {"artifact_path": panel_path},
        "ranking": {
            "panel_scoring": {
                "ngboost": {
                    "enabled": ngb_enabled,
                    "artifact_path": ngb_path,
                },
            },
        },
    }


def _ctx(strategy_dir: Path, config: dict, run_mode: str | None = None) -> PreflightContext:
    return PreflightContext(config=config, strategy_dir=strategy_dir, run_mode=run_mode)


# ─── FeatureCoverageTask parity ──────────────────────────────────────────────

class TestFeatureCoverageTaskParity:

    def test_ngboost_disabled_soft_pass(self, tmp_path):
        cfg = _config("a.json", "b.json", ngb_enabled=False)
        leg = _check_feature_coverage(config=cfg, strategy_dir=tmp_path)
        ctx = _ctx(tmp_path, cfg)
        FeatureCoverageTask().run(ctx)
        new = ctx.results[-1]
        assert new.name == leg.name == "P-FEATURE-COVER"
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_artifacts_present_full_coverage_hard_pass(self, tmp_path):
        panel_payload = {"feature_cols": ["a", "b", "c", "d"]}
        ngb_payload = {"feature_cols": ["a", "b", "c"]}  # subset
        p_path, n_path = _write_artifacts(tmp_path, panel_payload, ngb_payload)
        cfg = _config(str(p_path.relative_to(tmp_path)),
                       str(n_path.relative_to(tmp_path)))
        leg = _check_feature_coverage(config=cfg, strategy_dir=tmp_path)
        ctx = _ctx(tmp_path, cfg)
        FeatureCoverageTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_ngboost_features_missing_from_panel_full_hard_fail(self, tmp_path):
        panel_payload = {"feature_cols": ["a", "b"]}
        ngb_payload = {"feature_cols": ["a", "b", "c", "d", "e"]}  # 3 missing
        p_path, n_path = _write_artifacts(tmp_path, panel_payload, ngb_payload)
        cfg = _config(str(p_path.relative_to(tmp_path)),
                       str(n_path.relative_to(tmp_path)))
        leg = _check_feature_coverage(config=cfg, strategy_dir=tmp_path,
                                       run_mode="full")
        ctx = _ctx(tmp_path, cfg, run_mode="full")
        FeatureCoverageTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message


# ─── ArtifactRunIdAlignmentTask parity ───────────────────────────────────────

class TestArtifactRunIdAlignmentTaskParity:

    def test_ngboost_disabled_soft_pass(self, tmp_path):
        cfg = _config("a.json", "b.json", ngb_enabled=False)
        leg = _check_artifact_run_id_alignment(config=cfg, strategy_dir=tmp_path)
        ctx = _ctx(tmp_path, cfg)
        ArtifactRunIdAlignmentTask().run(ctx)
        new = ctx.results[-1]
        assert new.name == leg.name == "P-RUN-ID"
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_artifacts_aligned_hard_pass(self, tmp_path):
        panel_payload = {"feature_cols": ["a"], "train_run_id": "20260530-abc123"}
        ngb_payload = {"feature_cols": ["a"], "train_run_id": "20260530-abc123"}
        p_path, n_path = _write_artifacts(tmp_path, panel_payload, ngb_payload)
        cfg = _config(str(p_path.relative_to(tmp_path)),
                       str(n_path.relative_to(tmp_path)))
        leg = _check_artifact_run_id_alignment(config=cfg, strategy_dir=tmp_path)
        ctx = _ctx(tmp_path, cfg)
        ArtifactRunIdAlignmentTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_run_id_mismatch_full_hard_fail(self, tmp_path):
        panel_payload = {"feature_cols": ["a"], "train_run_id": "20260530-aaa"}
        ngb_payload = {"feature_cols": ["a"], "train_run_id": "20260530-bbb"}
        p_path, n_path = _write_artifacts(tmp_path, panel_payload, ngb_payload)
        cfg = _config(str(p_path.relative_to(tmp_path)),
                       str(n_path.relative_to(tmp_path)))
        leg = _check_artifact_run_id_alignment(config=cfg, strategy_dir=tmp_path,
                                                run_mode="full")
        ctx = _ctx(tmp_path, cfg, run_mode="full")
        ArtifactRunIdAlignmentTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_run_id_missing_on_panel_full_hard_fail(self, tmp_path):
        panel_payload = {"feature_cols": ["a"]}  # no train_run_id
        ngb_payload = {"feature_cols": ["a"], "train_run_id": "20260530-bbb"}
        p_path, n_path = _write_artifacts(tmp_path, panel_payload, ngb_payload)
        cfg = _config(str(p_path.relative_to(tmp_path)),
                       str(n_path.relative_to(tmp_path)))
        leg = _check_artifact_run_id_alignment(config=cfg, strategy_dir=tmp_path,
                                                run_mode="full")
        ctx = _ctx(tmp_path, cfg, run_mode="full")
        ArtifactRunIdAlignmentTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message
