"""R3 — T/J/P shape regression tests for stamp_panel_contract_missing_fields.

Pins the Pipeline composition + each Task's single-responsibility contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stamp_panel_contract_missing_fields import (  # noqa: E402
    AggregatePerFoldICTask,
    ComputeJob,
    ComputeStampsTask,
    LoadArtifactTask,
    LoadJob,
    LoadManifestTask,
    StampContext,
    WriteJob,
    WriteStampedArtifactTask,
    build_pipeline,
)


@pytest.fixture
def per_cut_artifact(tmp_path):
    p = tmp_path / "per_cut.json"
    p.write_text(json.dumps({
        "kind": "panel_ltr_xgboost",
        "oos_mean_ic": 0.05,
    }))
    return p


@pytest.fixture
def manifest(tmp_path, per_cut_artifact):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({
        "retrains": [
            {"artifact_uri": str(per_cut_artifact), "cutoff_date": "2024-01-02"},
            {"artifact_uri": str(per_cut_artifact), "cutoff_date": "2024-07-01"},
        ],
    }))
    return p


@pytest.fixture
def prod_artifact(tmp_path):
    p = tmp_path / "panel-ltr.json"
    p.write_text(json.dumps({
        "kind": "panel_ltr_xgboost",
        "trained_date": "2026-05-18",
        "config_fingerprint": "sha256:14586756d4f67691",
        "label_col": "fwd_60d_excess",
    }))
    return p


def test_pipeline_has_three_ordered_jobs():
    p = build_pipeline()
    assert p.name == "StampPanelContract"
    assert [type(j).__name__ for j in p.jobs] == ["LoadJob", "ComputeJob", "WriteJob"]


def test_load_job_tasks():
    assert [type(t).__name__ for t in LoadJob().tasks] == [
        "LoadArtifactTask", "LoadManifestTask",
    ]


def test_compute_job_tasks():
    assert [type(t).__name__ for t in ComputeJob().tasks] == [
        "AggregatePerFoldICTask", "ComputeStampsTask",
    ]


def test_write_job_tasks():
    assert [type(t).__name__ for t in WriteJob().tasks] == ["WriteStampedArtifactTask"]


def test_load_artifact_task_populates_ctx(prod_artifact, manifest):
    ctx = StampContext(artifact_path=prod_artifact, manifest_path=manifest)
    LoadArtifactTask().run(ctx)
    assert ctx.artifact is not None
    assert ctx.artifact["kind"] == "panel_ltr_xgboost"


def test_load_manifest_task_raises_on_missing(tmp_path, prod_artifact):
    ctx = StampContext(
        artifact_path=prod_artifact,
        manifest_path=tmp_path / "does_not_exist.json",
    )
    with pytest.raises(RuntimeError, match="manifest not found"):
        LoadManifestTask().run(ctx)


def test_aggregate_per_fold_ic_task_populates_ctx(prod_artifact, manifest):
    ctx = StampContext(artifact_path=prod_artifact, manifest_path=manifest)
    AggregatePerFoldICTask().run(ctx)
    assert ctx.per_fold_ic == [0.05, 0.05]


def test_aggregate_per_fold_ic_task_raises_on_empty(prod_artifact, tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"retrains": []}))
    ctx = StampContext(artifact_path=prod_artifact, manifest_path=empty)
    with pytest.raises(RuntimeError, match="no usable per-fold"):
        AggregatePerFoldICTask().run(ctx)


def test_compute_stamps_task_assembles_full_dict(prod_artifact, manifest):
    ctx = StampContext(artifact_path=prod_artifact, manifest_path=manifest)
    LoadArtifactTask().run(ctx)
    AggregatePerFoldICTask().run(ctx)
    ComputeStampsTask().run(ctx)
    assert ctx.stamps is not None
    assert set(ctx.stamps) == {
        "train_run_id", "oos_mean_ic", "oos_std_ic", "oos_per_fold_ic",
        "cv_method", "cv_embargo_days", "sentiment_runtime_gate_contract",
    }
    assert ctx.stamps["oos_mean_ic"] == 0.05


def test_write_stamped_artifact_task_skip_on_dry_run(prod_artifact, manifest):
    before = prod_artifact.read_text()
    ctx = StampContext(
        artifact_path=prod_artifact,
        manifest_path=manifest,
        dry_run=True,
    )
    LoadArtifactTask().run(ctx)
    AggregatePerFoldICTask().run(ctx)
    ComputeStampsTask().run(ctx)
    WriteStampedArtifactTask().run(ctx)
    # dry-run did not touch the file
    assert prod_artifact.read_text() == before


def test_write_stamped_artifact_task_is_idempotent(prod_artifact, manifest):
    ctx = StampContext(artifact_path=prod_artifact, manifest_path=manifest)
    build_pipeline().run(ctx)
    # mutate then re-run; existing field preserved
    after = json.loads(prod_artifact.read_text())
    after["oos_mean_ic"] = 999.0
    prod_artifact.write_text(json.dumps(after))
    ctx2 = StampContext(artifact_path=prod_artifact, manifest_path=manifest)
    build_pipeline().run(ctx2)
    assert json.loads(prod_artifact.read_text())["oos_mean_ic"] == 999.0


def test_full_pipeline_end_to_end(prod_artifact, manifest):
    ctx = StampContext(artifact_path=prod_artifact, manifest_path=manifest)
    result = build_pipeline().run(ctx)
    assert result.ok is True
    assert [s.job_name for s in result.steps] == ["LoadJob", "ComputeJob", "WriteJob"]
    after = json.loads(prod_artifact.read_text())
    for k in ("train_run_id", "oos_mean_ic", "cv_method", "cv_embargo_days"):
        assert k in after
