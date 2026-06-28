"""Guards for run_wf_gate's prod-config selection in --derive-config-from-prod.

The ONE converged parity contract (shared with renquant-backtesting #58's
``wf_config_builder.select_prod_reference_for_candidate``): select the
production reference whose scorer kind MATCHES the candidate's declared kind
(read from artifact metadata, never a path suffix), and use that SAME reference
for BOTH derivation and the parity check. A GBDT/xgb candidate is compared
against the GBDT/shadow config; a PatchTST candidate against the PatchTST
primary. ``RENQUANT_STRATEGY_CONFIG`` is honored but validated against the
candidate kind — a mismatch (or an unknown kind) FAILS CLOSED, so a genuine
prod-vs-candidate mismatch never passes.
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


# ── candidate-matched reference selection ─────────────────────────────────────


def test_prod_config_path_defaults_to_primary_without_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy callers (no candidate kind) keep the env/primary fallback."""
    monkeypatch.delenv("RENQUANT_STRATEGY_CONFIG", raising=False)
    assert run_wf_gate._prod_config_path() == (
        run_wf_gate.STRATEGY_DIR / "strategy_config.json"
    )


def test_prod_config_path_maps_gbdt_kind_to_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RENQUANT_STRATEGY_CONFIG", raising=False)
    assert run_wf_gate._prod_config_path(candidate_kind="panel_ltr_xgboost") == (
        run_wf_gate.STRATEGY_DIR / "strategy_config.shadow.json"
    )


def test_prod_config_path_maps_patchtst_kind_to_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RENQUANT_STRATEGY_CONFIG", raising=False)
    assert run_wf_gate._prod_config_path(candidate_kind="hf_patchtst") == (
        run_wf_gate.STRATEGY_DIR / "strategy_config.json"
    )


def test_prod_config_path_unknown_kind_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RENQUANT_STRATEGY_CONFIG", raising=False)
    with pytest.raises(ValueError, match="no production reference"):
        run_wf_gate._prod_config_path(candidate_kind="some_unknown_scorer")


def test_prod_config_path_env_validated_against_candidate_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A validated env override matching the candidate kind is honored."""
    shadow = tmp_path / "shadow.json"
    shadow.write_text(json.dumps({"ranking": {"panel_scoring": {"kind": "xgb"}}}))
    monkeypatch.setenv("RENQUANT_STRATEGY_CONFIG", str(shadow))
    assert run_wf_gate._prod_config_path(candidate_kind="panel_ltr_xgboost") == shadow


def test_prod_config_path_env_kind_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An env override whose kind != candidate kind FAILS CLOSED (no smuggling)."""
    primary = tmp_path / "primary.json"
    primary.write_text(
        json.dumps({"ranking": {"panel_scoring": {"kind": "hf_patchtst"}}})
    )
    monkeypatch.setenv("RENQUANT_STRATEGY_CONFIG", str(primary))
    with pytest.raises(ValueError, match="does not match the candidate kind"):
        run_wf_gate._prod_config_path(candidate_kind="panel_ltr_xgboost")


# ── same-selected-reference contract: positive then negative ──────────────────


def _artifact(path: Path, cols: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": "panel_ltr_xgboost", "feature_cols": cols}))
    return path


def _prod_config(artifact_path: str, *, kind: str) -> dict:
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


def test_gbdt_candidate_with_gbdt_reference_passes(tmp_path: Path) -> None:
    """POSITIVE: GBDT candidate derived from the GBDT/shadow reference passes.

    Same selected reference for derivation and parity; the GBDT/shadow config
    already declares kind=xgb so no mutation is needed.
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
    # kind inherited from the (GBDT) prod config — NOT mutated.
    assert derived["ranking"]["panel_scoring"]["kind"] == "xgb"

    prod_path = tmp_path / "gbdt_prod.json"
    prod_path.write_text(json.dumps(gbdt_prod, indent=2))
    derived_path = tmp_path / "derived.json"
    derived_path.write_text(json.dumps(derived, indent=2))

    result = evaluate_wf_config_parity(
        prod_path,  # SAME selected reference used for derivation
        derived_path,
        candidate_artifact=candidate,
        strategy_dir=tmp_path,
    )
    assert result["passed"] is True, result["issues"]


def test_gbdt_candidate_against_patchtst_reference_stays_non_promotable(
    tmp_path: Path,
) -> None:
    """NEGATIVE (parity layer): derived PatchTST kind vs the correct GBDT
    reference → ``ranking.panel_scoring.kind`` diverges and parity FAILS.

    Reproduces the bug shape the converged selection now prevents: a GBDT
    candidate whose derived config kept ``kind=hf_patchtst`` (inherited from the
    PatchTST primary), compared against the GBDT/shadow reference that is the
    correct production semantics for that candidate (``kind=xgb``). The semantic
    kind path diverges, so the run is non-promotable. The selection-layer
    fail-closed test below stops this from ever reaching parity in practice.
    """
    candidate = _artifact(tmp_path / "artifacts/prod/panel-ltr.json", ["f1", "f2"])
    wf_art = _artifact(tmp_path / "artifacts/wf/cut/panel-ltr.json", ["f1", "f2"])
    manifest = tmp_path / "artifacts/sim/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "retrains": [{"artifact_uri": str(wf_art), "cutoff_date": "2024-01-01"}]
    }))

    patchtst_prod = _prod_config(str(candidate), kind="hf_patchtst")
    gbdt_prod = _prod_config(str(candidate), kind="xgb")
    base_wf = {
        "ranking": {"panel_scoring": {"artifact_path": "artifacts/sim/placeholder.json"}},
        "walkforward": {"enabled": True, "manifest_path": str(manifest)},
    }

    # Bug shape: derived from the PatchTST primary keeps kind=hf_patchtst.
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
        gbdt_prod_path,  # the correct GBDT reference for this candidate
        derived_path,
        candidate_artifact=candidate,
        strategy_dir=tmp_path,
    )
    assert result["passed"] is False
    assert any(
        i.get("path") == "ranking.panel_scoring.kind" for i in result["issues"]
    ), result["issues"]


def test_selection_layer_blocks_mismatch_before_parity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NEGATIVE (selection layer): the converged contract refuses to compare a
    GBDT candidate against a PatchTST reference at all — fail closed.

    This is the real-world defense: ``weekly_wf_promote.sh`` must export the
    GBDT/shadow config for a GBDT candidate; if it points at (or defaults to)
    the PatchTST primary, ``_prod_config_path`` raises rather than silently
    deriving a non-production-equivalent config.
    """
    primary = tmp_path / "primary.json"
    primary.write_text(
        json.dumps({"ranking": {"panel_scoring": {"kind": "hf_patchtst"}}})
    )
    monkeypatch.setenv("RENQUANT_STRATEGY_CONFIG", str(primary))
    with pytest.raises(ValueError):
        run_wf_gate._prod_config_path(candidate_kind="panel_ltr_xgboost")


def test_run_wf_gate_uses_prod_config_helper_at_both_sites() -> None:
    """Both the derive site and the parity site must route through the helper,
    passing the candidate kind so the kind-matched reference is selected."""
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    # The hardcoded primary literal must no longer be used to pick the prod ref.
    assert 'prod_cfg_path = STRATEGY_DIR / "strategy_config.json"' not in src
    # Both sites pass the candidate kind into the selector.
    assert src.count("_prod_config_path(candidate_kind=artifact.get(\"kind\"))") >= 2
