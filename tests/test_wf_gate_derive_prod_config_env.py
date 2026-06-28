"""Guards for run_wf_gate's prod-config selection in --derive-config-from-prod.

The weekly WF-promote wrapper evaluates non-PatchTST candidates (e.g. GBDT)
by exporting RENQUANT_STRATEGY_CONFIG to the GBDT/shadow prod config. The
derive + parity path in run_wf_gate.py must honor that env var; otherwise it
derives from the PatchTST PRIMARY config and the derived eval config keeps
ranking.panel_scoring.kind=hf_patchtst while pointing at a GBDT artifact, so the
scorer-kind/artifact parity guard fires on every GBDT candidate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "scripts", REPO / "backtesting" / "renquant_104"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_wf_gate  # noqa: E402
from wf_config_builder import build_wf_config_from_prod  # noqa: E402
from wf_config_parity import evaluate_wf_config_parity  # noqa: E402


def test_prod_config_path_defaults_to_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RENQUANT_STRATEGY_CONFIG", raising=False)
    assert run_wf_gate._prod_config_path() == run_wf_gate.STRATEGY_DIR / "strategy_config.json"


def test_prod_config_path_honors_relative_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RENQUANT_STRATEGY_CONFIG", "strategy_config.shadow.json")
    assert run_wf_gate._prod_config_path() == (
        run_wf_gate.STRATEGY_DIR / "strategy_config.shadow.json"
    )


def test_prod_config_path_honors_absolute_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    abs_cfg = tmp_path / "gbdt_prod.json"
    monkeypatch.setenv("RENQUANT_STRATEGY_CONFIG", str(abs_cfg))
    assert run_wf_gate._prod_config_path() == abs_cfg


def _artifact(path: Path, cols: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": "panel_ltr_xgboost", "feature_cols": cols}))
    return path


def _prod_config(artifact_path: str, *, kind: str) -> dict:
    """Minimal prod config sufficient for the kind/feature parity checks."""
    return {
        "ranking": {
            "panel_scoring": {
                "enabled": True,
                "kind": kind,
                "artifact_path": artifact_path,
                "buy_floor": "adaptive_mean_std",
                "rotation_advantage": 0.0,
            }
        }
    }


def test_gbdt_prod_config_derives_xgb_kind_and_parity_passes(tmp_path: Path) -> None:
    """End-to-end: deriving from the GBDT prod config keeps kind=xgb -> parity ok.

    Mirrors what _prod_config_path() now selects when RENQUANT_STRATEGY_CONFIG
    points at the GBDT/shadow config: the derived eval config inherits the GBDT
    scorer kind, so it matches the GBDT (panel_ltr_xgboost) candidate artifact.
    """
    candidate = _artifact(tmp_path / "artifacts/prod/panel-ltr.json", ["f1", "f2"])
    wf_art = _artifact(tmp_path / "artifacts/wf/cut/panel-ltr.json", ["f1", "f2"])
    manifest = tmp_path / "artifacts/sim/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "retrains": [{"artifact_uri": str(wf_art), "cutoff_date": "2024-01-01"}]
    }))

    gbdt_prod = _prod_config(str(candidate), kind="xgb")
    base_wf = {
        "ranking": {"panel_scoring": {"artifact_path": "artifacts/sim/placeholder.json"}},
        "walkforward": {"enabled": True, "manifest_path": str(manifest)},
    }

    derived = build_wf_config_from_prod(
        gbdt_prod,
        manifest_path=str(manifest),
        base_wf_config=base_wf,
        strategy_dir=tmp_path,
    )
    # kind is inherited from the (GBDT) prod config, not the base/placeholder.
    assert derived["ranking"]["panel_scoring"]["kind"] == "xgb"

    prod_path = tmp_path / "gbdt_prod.json"
    prod_path.write_text(json.dumps(gbdt_prod, indent=2))
    derived_path = tmp_path / "derived.json"
    derived_path.write_text(json.dumps(derived, indent=2))

    result = evaluate_wf_config_parity(
        prod_path,
        derived_path,
        candidate_artifact=candidate,
        strategy_dir=tmp_path,
    )
    assert result["passed"] is True, result["issues"]


def test_mismatched_prod_kind_fails_kind_parity(tmp_path: Path) -> None:
    """Regression: deriving a GBDT candidate from the PatchTST prod -> kind drift.

    Reproduces the bug at the parity layer: when the derived eval config is
    built from the PatchTST PRIMARY (kind=hf_patchtst) but the GBDT/shadow config
    (kind=xgb) is the correct production reference for a GBDT candidate, the
    ranking.panel_scoring.kind semantic path diverges and parity fails. The env
    fix keeps both sides on the same (GBDT) prod config so this can't happen.
    """
    candidate = _artifact(tmp_path / "artifacts/prod/panel-ltr.json", ["f1", "f2"])
    wf_art = _artifact(tmp_path / "artifacts/wf/cut/panel-ltr.json", ["f1", "f2"])
    manifest = tmp_path / "artifacts/sim/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "retrains": [{"artifact_uri": str(wf_art), "cutoff_date": "2024-01-01"}]
    }))

    # Bug shape: derived config built from PatchTST prod keeps kind=hf_patchtst,
    # but the correct GBDT prod reference declares kind=xgb for this candidate.
    patchtst_prod = _prod_config(str(candidate), kind="hf_patchtst")
    gbdt_prod = _prod_config(str(candidate), kind="xgb")
    base_wf = {
        "ranking": {"panel_scoring": {"artifact_path": "artifacts/sim/placeholder.json"}},
        "walkforward": {"enabled": True, "manifest_path": str(manifest)},
    }

    derived = build_wf_config_from_prod(
        patchtst_prod,
        manifest_path=str(manifest),
        base_wf_config=base_wf,
        strategy_dir=tmp_path,
    )
    assert derived["ranking"]["panel_scoring"]["kind"] == "hf_patchtst"

    gbdt_prod_path = tmp_path / "gbdt_prod.json"
    gbdt_prod_path.write_text(json.dumps(gbdt_prod, indent=2))
    derived_path = tmp_path / "derived.json"
    derived_path.write_text(json.dumps(derived, indent=2))

    result = evaluate_wf_config_parity(
        gbdt_prod_path,
        derived_path,
        candidate_artifact=candidate,
        strategy_dir=tmp_path,
    )
    assert result["passed"] is False
    assert any(
        i["path"] == "ranking.panel_scoring.kind" for i in result["issues"]
    ), result["issues"]


def test_run_wf_gate_uses_prod_config_helper_at_both_sites() -> None:
    """Both the derive site and the parity site must route through the helper."""
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    # The hardcoded primary literal must no longer be used to pick the prod ref.
    assert 'prod_cfg_path = STRATEGY_DIR / "strategy_config.json"' not in src
    assert src.count("_prod_config_path()") >= 2
