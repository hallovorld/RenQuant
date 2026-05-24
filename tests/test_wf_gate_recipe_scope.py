"""Unit tests for WF gate scope and recipe matching."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_wf_gate.py"


def _load_module():
    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_wf_gate_under_test", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _artifact(features: list[str]) -> dict:
    return {
        "kind": "panel_ltr_xgboost",
        "feature_cols": features,
        "label_col": "fwd_60d_excess",
        "lookahead_days": 60,
        "params": {"objective": "rank:pairwise", "eta": 0.05},
    }


def test_manifest_recipe_usage_accepts_matching_samples(tmp_path: Path):
    mod = _load_module()
    candidate = tmp_path / "candidate.json"
    sample = tmp_path / "sample.json"
    manifest = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(_artifact(["a", "b"])))
    sample.write_text(json.dumps(_artifact(["a", "b"])))
    manifest.write_text(json.dumps({
        "retrains": [
            {"artifact_uri": str(sample), "cutoff_date": "2024-01-01"},
            {"artifact_uri": str(sample), "cutoff_date": "2024-02-01"},
        ]
    }))

    usage = mod._manifest_recipe_usage(manifest, candidate)

    assert usage["recipe_validated"] is True
    assert usage["candidate_n_features"] == 2


def test_manifest_recipe_usage_rejects_feature_drift(tmp_path: Path):
    mod = _load_module()
    candidate = tmp_path / "candidate.json"
    sample = tmp_path / "sample.json"
    manifest = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(_artifact(["a", "b", "sentiment"])))
    sample.write_text(json.dumps(_artifact(["a", "b"])))
    manifest.write_text(json.dumps({
        "retrains": [
            {"artifact_uri": str(sample), "cutoff_date": "2024-01-01"},
            {"artifact_uri": str(sample), "cutoff_date": "2024-02-01"},
        ]
    }))

    usage = mod._manifest_recipe_usage(manifest, candidate)

    assert usage["recipe_validated"] is False
    report = usage["manifest_sample_reports"][0]
    assert report["missing_features_vs_candidate"] == ["sentiment"]


def test_manifest_recipe_usage_checks_all_rows_not_just_samples(tmp_path: Path):
    """A drifted non-sampled manifest row must fail promotion scope.

    Regression target: the old implementation inspected only first/middle/last
    rows, so a bad row at position 1 of 5 was invisible.
    """
    mod = _load_module()
    candidate = tmp_path / "candidate.json"
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    manifest = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(_artifact(["a", "b"])))
    good.write_text(json.dumps(_artifact(["a", "b"])))
    bad.write_text(json.dumps(_artifact(["a", "b", "leaky_extra"])))
    manifest.write_text(json.dumps({
        "retrains": [
            {"artifact_uri": str(good), "cutoff_date": "2024-01-01"},
            {"artifact_uri": str(bad), "cutoff_date": "2024-01-22"},
            {"artifact_uri": str(good), "cutoff_date": "2024-02-12"},
            {"artifact_uri": str(good), "cutoff_date": "2024-03-04"},
            {"artifact_uri": str(good), "cutoff_date": "2024-03-25"},
        ]
    }))

    usage = mod._manifest_recipe_usage(manifest, candidate)

    assert usage["recipe_validated"] is False
    assert any(
        "leaky_extra" in r.get("extra_features_vs_candidate", [])
        for r in usage["manifest_sample_reports"]
    )


def test_manifest_recipe_usage_accepts_patchtst_pt_sidecars(tmp_path: Path):
    mod = _load_module()
    candidate = tmp_path / "candidate.pt"
    sample = tmp_path / "sample.pt"
    manifest = tmp_path / "manifest.json"
    candidate.write_bytes(b"candidate checkpoint")
    sample.write_bytes(b"sample checkpoint")
    payload = {
        "kind": "hf_patchtst",
        "feature_cols": ["a", "b"],
        "label_col": "fwd_60d_excess",
        "lookahead_days": 60,
        "training_contract": {
            "hyperparameters": {
                "seq_len": 32,
                "patch_length": 4,
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
            },
        },
    }
    candidate.with_name(candidate.name + ".metadata.json").write_text(json.dumps(payload))
    sample.with_name(sample.name + ".metadata.json").write_text(json.dumps(payload))
    manifest.write_text(json.dumps({
        "retrains": [{"artifact_uri": str(sample), "cutoff_date": "2024-01-01"}],
    }))

    usage = mod._manifest_recipe_usage(manifest, candidate)

    assert usage["recipe_validated"] is True
    assert usage["candidate_n_features"] == 2


def test_static_sanity_contract_rejects_artifact_without_cutoff() -> None:
    mod = _load_module()
    artifact = {
        "kind": "panel_ltr_xgboost",
        "trained_date": "2026-05-23",
        "lookahead_days": 60,
    }

    result = mod._validate_static_sanity_oos_contract(
        artifact,
        mod.pd.Timestamp("2024-02-02"),
    )

    assert result["passed"] is False
    assert "missing effective training cutoff" in result["reason"]


def test_static_sanity_contract_requires_cutoff_plus_lookahead_before_eval() -> None:
    mod = _load_module()
    artifact = {
        "kind": "panel_ltr_xgboost",
        "train_cutoff": "2024-02-01",
        "lookahead_days": 60,
    }

    result = mod._validate_static_sanity_oos_contract(
        artifact,
        mod.pd.Timestamp("2024-02-02"),
    )

    assert result["passed"] is False
    assert "cutoff + lookahead" in result["reason"]


def test_manifest_sanity_uses_effective_cutoff_without_double_embargo(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """WF sanity must mirror WalkForwardModelLoader's safe-date contract.

    Regression target: the manifest entry below is safe on 2024-02-02 because
    the artifact trained only through 2023-10-31 and uses a 60-business-day
    forward label. A stale sanity check incorrectly used cutoff_date
    2024-01-23 + 60BDay and failed closed.
    """
    mod = _load_module()
    strategy_path = str(REPO / "backtesting" / "renquant_104")
    if strategy_path not in sys.path:
        sys.path.insert(0, strategy_path)

    from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415

    candidate = tmp_path / "candidate.json"
    sample = tmp_path / "sample.json"
    manifest = tmp_path / "manifest.json"
    payload = _artifact(["feature_a"])
    candidate.write_text(json.dumps(payload))
    sample.write_text(json.dumps(payload))
    manifest.write_text(json.dumps({
        "retrains": [{
            "cutoff_date": "2024-01-23T00:00:00",
            "effective_train_cutoff_date": "2023-10-31T00:00:00",
            "trained_date": "2024-01-23T00:00:00",
            "lookahead_days": 60,
            "artifact_uri": str(sample),
        }]
    }))

    class FakeScorer:
        metadata = {
            "feature_means": [0.0],
            "feature_stds": [1.0],
            "feature_norm_kind": ["identity"],
        }

        def score(self, frame):
            return pd.Series([0.1] * len(frame), index=frame.index)

    monkeypatch.setattr(
        PanelScorer,
        "load",
        staticmethod(lambda path: FakeScorer()),
    )

    val = pd.DataFrame({
        "date": pd.to_datetime(["2024-02-02", "2024-02-02"]),
        "ticker": ["AAA", "BBB"],
        "feature_a": [1.0, 2.0],
    })

    mu, meta = mod._score_manifest_sanity(
        val,
        ["feature_a"],
        manifest,
        candidate,
        payload,
    )

    assert list(mu) == [0.1, 0.1]
    assert meta["n_oos_dates"] == 1
    assert "effective_train_cutoff_date" in meta["cutoff_contract"]


def test_manifest_sanity_scores_history_scorer_with_past_only_history(
    monkeypatch, tmp_path: Path,
):
    mod = _load_module()
    strategy_path = str(REPO / "backtesting" / "renquant_104")
    if strategy_path not in sys.path:
        sys.path.insert(0, strategy_path)

    from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415

    candidate = tmp_path / "candidate.json"
    sample = tmp_path / "sample.json"
    manifest = tmp_path / "manifest.json"
    payload = {
        "kind": "hf_patchtst",
        "feature_cols": ["feature_a"],
        "label_col": "fwd_5d_excess",
        "lookahead_days": 0,
        "params": {"seq_len": 3},
    }
    candidate.write_text(json.dumps(payload))
    sample.write_text(json.dumps(payload))
    manifest.write_text(json.dumps({
        "retrains": [{
            "cutoff_date": "2024-01-01T00:00:00",
            "effective_train_cutoff_date": "2024-01-01T00:00:00",
            "trained_date": "2024-01-02T00:00:00",
            "lookahead_days": 0,
            "artifact_uri": str(sample),
        }]
    }))

    class FakeHistoryScorer:
        requires_history = True
        seq_len = 3
        metadata = {"kind": "hf_patchtst"}

        def score_with_history(self, history, target_tickers):
            assert history["date"].max() < pd.Timestamp("2024-01-10")
            assert target_tickers == ["AAA", "BBB"]
            return pd.Series({"AAA": 0.2, "BBB": 0.4})

    monkeypatch.setattr(
        PanelScorer,
        "load",
        staticmethod(lambda path: FakeHistoryScorer()),
    )

    panel = pd.DataFrame({
        "date": pd.to_datetime(
            [
                "2024-01-05", "2024-01-06", "2024-01-07",
                "2024-01-05", "2024-01-06", "2024-01-07",
                "2024-01-10", "2024-01-10",
            ]
        ),
        "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "AAA", "BBB"],
        "feature_a": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 9.0, 9.1],
    })
    val = panel[panel["date"].eq(pd.Timestamp("2024-01-10"))].copy()

    mu, meta = mod._score_manifest_sanity(
        val,
        ["feature_a"],
        manifest,
        candidate,
        payload,
        panel_history=panel,
    )

    assert list(mu) == [0.2, 0.4]
    assert meta["n_history_scorer_artifacts"] == 1


def test_manifest_safe_last_label_prefers_effective_train_cutoff() -> None:
    mod = _load_module()
    entry = SimpleNamespace(
        cutoff_date=mod.pd.Timestamp("2024-01-23"),
        effective_train_cutoff_date=mod.pd.Timestamp("2023-10-31"),
        lookahead_days=60,
    )

    safe = mod._manifest_entry_safe_last_label_date(entry)

    assert safe == mod.pd.Timestamp("2024-01-23")


def test_recipe_fingerprint_ignores_execution_only_xgb_params() -> None:
    """Hardware/threading changes must not invalidate historical WF recipes."""
    mod = _load_module()
    old_hw = _artifact(["a", "b"])
    new_hw = _artifact(["a", "b"])
    old_hw["params"] = {
        "objective": "rank:pairwise",
        "eta": 0.05,
        "max_depth": 5,
        "nthread": 8,
        "verbosity": 0,
    }
    new_hw["params"] = {
        "objective": "rank:pairwise",
        "eta": 0.05,
        "max_depth": 5,
        "nthread": 14,
        "verbosity": 2,
    }

    assert mod._recipe_fingerprint(old_hw) == mod._recipe_fingerprint(new_hw)

    changed_learning_param = _artifact(["a", "b"])
    changed_learning_param["params"] = {
        "objective": "rank:pairwise",
        "eta": 0.10,
        "max_depth": 5,
        "nthread": 14,
        "verbosity": 0,
    }
    assert (
        mod._recipe_fingerprint(old_hw)
        != mod._recipe_fingerprint(changed_learning_param)
    )
