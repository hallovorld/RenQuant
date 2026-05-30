"""Tests for scripts/stamp_panel_contract_missing_fields.py.

§5.13.5: any prod-touching script gets a regression test. This script writes
the 6 P-PANEL-CONTRACT strict fields onto an artifact JSON; it must be
idempotent (re-running doesn't overwrite existing values) and the per-fold
IC aggregation must reflect the manifest's recipe-equivalent retrains.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stamp_panel_contract_missing_fields import (  # noqa: E402
    aggregate_per_fold_ic_from_manifest,
    stamp_artifact,
    synthetic_train_run_id,
)


@pytest.fixture
def per_cut_artifact(tmp_path):
    """A minimal per-cut artifact carrying oos_mean_ic."""
    p = tmp_path / "per_cut.json"
    p.write_text(json.dumps({
        "kind": "panel_ltr_xgboost",
        "config_fingerprint": "sha256:abc",
        "oos_mean_ic": 0.05,
        "oos_per_fold_ic": [0.04, 0.06, 0.05],
    }))
    return p


@pytest.fixture
def manifest(tmp_path, per_cut_artifact):
    """A minimal walkforward manifest pointing at 3 per-cut artifacts."""
    p = tmp_path / "manifest.json"
    # Three retrain rows pointing at the SAME per-cut artifact for simplicity.
    p.write_text(json.dumps({
        "retrains": [
            {"artifact_uri": str(per_cut_artifact), "cutoff_date": "2024-01-02"},
            {"artifact_uri": str(per_cut_artifact), "cutoff_date": "2024-07-01"},
            {"artifact_uri": str(per_cut_artifact), "cutoff_date": "2025-04-01"},
        ],
    }))
    return p


@pytest.fixture
def prod_artifact(tmp_path):
    """A minimal prod GBDT artifact that LACKS the 6 strict fields."""
    p = tmp_path / "panel-ltr.json"
    p.write_text(json.dumps({
        "kind": "panel_ltr_xgboost",
        "trained_date": "2026-05-18",
        "config_fingerprint": "sha256:14586756d4f67691",
        "label_col": "fwd_60d_excess",
        "feature_cols": ["a", "b", "c"],
    }))
    return p


def test_aggregate_per_fold_ic_returns_list(manifest, per_cut_artifact):
    out = aggregate_per_fold_ic_from_manifest(manifest)
    assert isinstance(out, list)
    assert len(out) == 3  # one per retrain row pointing at the per-cut artifact
    assert all(isinstance(x, float) for x in out)
    assert all(x == 0.05 for x in out)


def test_synthetic_train_run_id_is_deterministic():
    a = synthetic_train_run_id(
        config_fingerprint="sha256:abc",
        trained_date="2026-05-18",
        label_col="fwd_60d_excess",
    )
    b = synthetic_train_run_id(
        config_fingerprint="sha256:abc",
        trained_date="2026-05-18",
        label_col="fwd_60d_excess",
    )
    assert a == b
    assert a.startswith("synthetic_")
    assert len(a) == len("synthetic_") + 16


def test_synthetic_train_run_id_changes_on_recipe_drift():
    a = synthetic_train_run_id(
        config_fingerprint="sha256:abc",
        trained_date="2026-05-18",
        label_col="fwd_60d_excess",
    )
    c = synthetic_train_run_id(
        config_fingerprint="sha256:def",  # different fp
        trained_date="2026-05-18",
        label_col="fwd_60d_excess",
    )
    assert a != c


def test_stamp_artifact_writes_six_fields(prod_artifact, manifest):
    stamps = stamp_artifact(prod_artifact, manifest)
    assert set(stamps.keys()) == {
        "train_run_id", "oos_mean_ic", "oos_std_ic", "oos_per_fold_ic",
        "cv_method", "cv_embargo_days", "sentiment_runtime_gate_contract",
    }
    # After write, the artifact has them
    after = json.loads(prod_artifact.read_text())
    for k in stamps:
        assert k in after, f"{k} not written"


def test_stamp_artifact_is_idempotent(prod_artifact, manifest):
    """Re-running must NOT overwrite already-present fields."""
    stamp_artifact(prod_artifact, manifest)
    after_first = json.loads(prod_artifact.read_text())
    # Mutate an existing field to a sentinel value
    after_first["oos_mean_ic"] = 999.0
    prod_artifact.write_text(json.dumps(after_first))
    # Re-stamp
    stamp_artifact(prod_artifact, manifest)
    after_second = json.loads(prod_artifact.read_text())
    # Existing field preserved
    assert after_second["oos_mean_ic"] == 999.0


def test_stamp_artifact_preserves_existing_fields(prod_artifact, manifest):
    stamp_artifact(prod_artifact, manifest)
    after = json.loads(prod_artifact.read_text())
    # Pre-existing keys still there
    assert after["kind"] == "panel_ltr_xgboost"
    assert after["trained_date"] == "2026-05-18"
    assert after["config_fingerprint"] == "sha256:14586756d4f67691"
    assert after["feature_cols"] == ["a", "b", "c"]


def test_stamp_artifact_dry_run_does_not_write(prod_artifact, manifest):
    before = prod_artifact.read_text()
    stamps = stamp_artifact(prod_artifact, manifest, dry_run=True)
    assert prod_artifact.read_text() == before
    # dry-run still returns the computed values
    assert "oos_mean_ic" in stamps


def test_stamp_artifact_raises_on_empty_manifest(prod_artifact, tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"retrains": []}))
    with pytest.raises(RuntimeError, match="no usable per-fold"):
        stamp_artifact(prod_artifact, empty)
