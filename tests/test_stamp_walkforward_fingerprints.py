from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.stamp_walkforward_fingerprints import stamp_manifest


def _artifact(feature_cols: list[str] | None = None) -> dict:
    cols = feature_cols or ["feat_a", "feat_b"]
    return {
        "kind": "panel_ltr_xgboost",
        "feature_cols": cols,
        "feature_means": [0.0] * len(cols),
        "feature_stds": [1.0] * len(cols),
        "feature_norm_kind": ["legacy_full_z"] * len(cols),
        "feature_source_contract": {"panel": "test"},
        "label_col": "fwd_60d_excess",
        "lookahead_days": 60,
        "params": {"objective": "rank:pairwise", "max_depth": 5},
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_stamp_manifest_repairs_missing_fingerprint_after_recipe_validation(tmp_path):
    artifact_path = _write_json(tmp_path / "wf/2024-01-02/panel-ltr.json", _artifact())
    calibrator_path = _write_json(
        tmp_path / "wf/2024-01-02/panel-rank-calibration.json",
        {
            "version": 1,
            "kind": "global_panel_calibration",
            "probability": {"x": [0.0], "y": [0.5]},
            "expected_return": {"x": [0.0], "y": [0.0]},
            "metadata": {"scorer_artifact_fingerprint": "sha256:old"},
        },
    )
    reference_path = _write_json(tmp_path / "candidate/panel-ltr.json", _artifact())
    manifest_path = _write_json(tmp_path / "manifest.json", {
        "retrains": [{
            "cutoff_date": "2024-01-02",
            "trained_date": "2024-01-03",
            "artifact_uri": str(artifact_path),
            "calibrator_uri": str(calibrator_path),
            "lookahead_days": 60,
        }],
    })
    config_path = _write_json(tmp_path / "strategy_config.json", {
        "watchlist": ["AAA", "BBB"],
        "benchmark": "SPY",
        "sector_map": {"AAA": "tech", "BBB": "finance"},
        "sector_etf_map": {"tech": "XLK", "finance": "XLF"},
        "panel_ltr": {"lookahead_days": 60, "xgb_params": {"objective": "rank:pairwise"}},
    })

    summary = stamp_manifest(
        manifest_path=manifest_path,
        fingerprint_config=config_path,
        reference_artifact=reference_path,
    )

    repaired = json.loads(artifact_path.read_text())
    repaired_cal = json.loads(calibrator_path.read_text())
    assert summary["n_stamped"] == 1
    assert summary["n_calibrators_stamped"] == 1
    assert repaired["config_fingerprint"].startswith("sha256:")
    assert repaired["config_fingerprint_fields"]["watchlist"] == ["AAA", "BBB"]
    assert (
        repaired_cal["metadata"]["scorer_model_content_fingerprint"]
        == repaired_cal["metadata"]["scorer_artifact_fingerprint"]
    )
    assert repaired_cal["metadata"]["scorer_artifact_sha256"].startswith("sha256:")


def test_stamp_manifest_refuses_recipe_mismatch(tmp_path):
    artifact_path = _write_json(tmp_path / "wf/2024-01-02/panel-ltr.json", _artifact())
    reference_path = _write_json(
        tmp_path / "candidate/panel-ltr.json",
        _artifact(["feat_a", "different_feature"]),
    )
    manifest_path = _write_json(tmp_path / "manifest.json", {
        "retrains": [{
            "cutoff_date": "2024-01-02",
            "trained_date": "2024-01-03",
            "artifact_uri": str(artifact_path),
            "lookahead_days": 60,
        }],
    })
    config_path = _write_json(tmp_path / "strategy_config.json", {
        "watchlist": ["AAA"],
        "sector_map": {"AAA": "tech"},
        "sector_etf_map": {"tech": "XLK"},
        "panel_ltr": {"lookahead_days": 60, "xgb_params": {"objective": "rank:pairwise"}},
    })

    with pytest.raises(ValueError, match="recipe validation failed"):
        stamp_manifest(
            manifest_path=manifest_path,
            fingerprint_config=config_path,
            reference_artifact=reference_path,
        )
