"""Behavioural guards for orch#799 option A: derive the xgb GBDT production
reference from a pinned blend config's ``ranking.panel_scoring.components[0]``.

The weekly WF-promote gate refused every week once the pinned primary became
``kind=blend`` (``run_wf_gate._prod_config_path`` fails closed on a non-xgb
reference). ``scripts/derive_gbdt_wf_reference.py`` derives a ``kind=xgb`` VIEW of
the SAME pinned recipe from component[0] so the gate can proceed, WITHOUT ever
consulting the banned umbrella working copy / sibling checkout, and WITHOUT
weakening any WF / sanity / parity gate.

Cases (per orch#799 task):
  (a) blend primary with an xgb component[0]      -> resolves component[0].
  (b) blend whose component[0] is NOT xgb          -> still fails closed.
  (c) the umbrella working-copy / sibling path      -> never consulted.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HELPER_SRC = REPO / "scripts" / "derive_gbdt_wf_reference.py"
REAL_CONFIG_CONSISTENCY = (
    REPO / "backtesting" / "renquant_104" / "kernel" / "config_consistency.py"
)

_GBDT_LEG = "artifacts/prod/panel-ltr.alpha158_fund.json"


def _load_helper():
    spec = importlib.util.spec_from_file_location("derive_gbdt_wf_reference_under_test", HELPER_SRC)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPER = _load_helper()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_strategy_dir(root: Path, *, with_leg_artifact: bool = True) -> Path:
    """Minimal strategy dir: the real (stdlib-only) config_consistency for the
    fingerprint-invariance guard, plus an optional GBDT leg artifact so the
    existence check can pass."""
    strategy_dir = root / "backtesting" / "renquant_104"
    (strategy_dir / "kernel").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REAL_CONFIG_CONSISTENCY, strategy_dir / "kernel" / "config_consistency.py")
    if with_leg_artifact:
        leg = strategy_dir / _GBDT_LEG
        leg.parent.mkdir(parents=True, exist_ok=True)
        leg.write_text(json.dumps({"kind": "panel_ltr_xgboost", "feature_cols": ["a", "b"]}))
    return strategy_dir


def _blend_config(component0: dict) -> dict:
    return {
        "watchlist": ["AAPL", "MSFT", "NVDA"],
        "benchmark": "SPY",
        "panel_ltr": {"lookahead_days": 60, "xgb_params": {"objective": "rank:pairwise"}},
        "ranking": {
            "panel_scoring": {
                "kind": "blend",
                "artifact_path": _GBDT_LEG,
                "buy_floor": 0.5,
                "sizing": {"mode": "sigma"},
                "components": [
                    component0,
                    {"kind": "momentum_residual",
                     "artifact_path": "artifacts/momentum/ledger.jsonl"},
                ],
            }
        },
    }


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# (a) blend primary with an xgb component[0] -> resolves component[0]
# ---------------------------------------------------------------------------


def test_a_blend_kind_absent_leg_artifact_path_resolves(tmp_path):
    """component[0] has no explicit kind but its artifact_path is the panel-ltr
    GBDT leg (the live served shape) -> derive a kind=xgb reference."""
    strategy_dir = _make_strategy_dir(tmp_path)
    pinned = _write(tmp_path / "pinned" / "strategy_config.json", _blend_config(
        {"artifact_path": _GBDT_LEG,
         "expected_config_fingerprint": "sha256:deadbeef",
         "_role": "component 0 = PRODUCTION panel scorer (rank:pairwise xgb)"},
    ))
    out = tmp_path / "logs" / "derived.json"

    result = HELPER.derive_reference(pinned, strategy_dir, out)

    assert out.exists()
    derived = json.loads(out.read_text())
    panel = derived["ranking"]["panel_scoring"]
    assert panel["kind"] == "xgb"                       # gate's kind-match passes
    assert panel["artifact_path"] == _GBDT_LEG          # points at component[0] leg
    assert "components" not in panel                    # blend-only key dropped
    assert panel["_derived_gbdt_reference"]["pinned_config"] == str(pinned)
    assert result["config_fingerprint_invariance_verified"] is True

    # Fingerprint invariance: the derived reference is the SAME recipe.
    fp = HELPER._load_fingerprint_config(strategy_dir)
    assert fp is not None
    assert fp(json.loads(pinned.read_text())) == fp(derived)


def test_a_blend_explicit_xgb_kind_component0_resolves(tmp_path):
    """component[0] with an explicit xgb-family kind also resolves."""
    strategy_dir = _make_strategy_dir(tmp_path)
    pinned = _write(tmp_path / "pinned" / "strategy_config.json", _blend_config(
        {"kind": "panel_ltr_xgboost", "artifact_path": _GBDT_LEG},
    ))
    out = tmp_path / "logs" / "derived.json"

    HELPER.derive_reference(pinned, strategy_dir, out)
    panel = json.loads(out.read_text())["ranking"]["panel_scoring"]
    assert panel["kind"] == "xgb"
    assert panel["artifact_path"] == _GBDT_LEG


# ---------------------------------------------------------------------------
# (b) blend whose component[0] is NOT xgb -> still fails closed
# ---------------------------------------------------------------------------


def test_b_component0_not_xgb_fails_closed(tmp_path):
    strategy_dir = _make_strategy_dir(tmp_path)
    pinned = _write(tmp_path / "pinned" / "strategy_config.json", _blend_config(
        {"kind": "momentum_residual", "artifact_path": "artifacts/momentum/ledger.jsonl"},
    ))
    out = tmp_path / "logs" / "derived.json"

    with pytest.raises(HELPER.DeriveError, match="not the xgb GBDT leg"):
        HELPER.derive_reference(pinned, strategy_dir, out)
    assert not out.exists()               # nothing written on a fail-closed


def test_b_xgb_component0_but_leg_artifact_absent_fails_closed(tmp_path):
    """An xgb component[0] whose referenced leg artifact does not exist cannot
    form a WF-comparable reference -> fail closed."""
    strategy_dir = _make_strategy_dir(tmp_path, with_leg_artifact=False)
    pinned = _write(tmp_path / "pinned" / "strategy_config.json", _blend_config(
        {"artifact_path": _GBDT_LEG, "_role": "rank:pairwise xgb"},
    ))
    out = tmp_path / "logs" / "derived.json"

    with pytest.raises(HELPER.DeriveError, match="leg artifact absent"):
        HELPER.derive_reference(pinned, strategy_dir, out)
    assert not out.exists()


def test_b_non_blend_primary_fails_closed(tmp_path):
    strategy_dir = _make_strategy_dir(tmp_path)
    cfg = _blend_config({"kind": "xgb", "artifact_path": _GBDT_LEG})
    cfg["ranking"]["panel_scoring"]["kind"] = "hf_patchtst"
    pinned = _write(tmp_path / "pinned" / "strategy_config.json", cfg)
    out = tmp_path / "logs" / "derived.json"

    with pytest.raises(HELPER.DeriveError, match="neither"):
        HELPER.derive_reference(pinned, strategy_dir, out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# (c) the umbrella working-copy / sibling path is never consulted
# ---------------------------------------------------------------------------


def test_c_working_copy_config_is_never_consulted(tmp_path):
    """A kind=xgb working-copy config sitting right next to the strategy dir must
    have ZERO influence: the derived reference comes from the pinned blend's
    component[0], and its provenance names the pinned config only."""
    strategy_dir = _make_strategy_dir(tmp_path)

    # The banned A8 working copy: kind=xgb, pointing at a DIFFERENT artifact.
    banned = _write(strategy_dir / "strategy_config.shadow.json", {
        "ranking": {"panel_scoring": {
            "kind": "xgb",
            "artifact_path": "artifacts/patchtst_shadow/DIVERGED_working_copy.json",
        }},
    })

    pinned = _write(tmp_path / "pinned" / "strategy_config.json", _blend_config(
        {"artifact_path": _GBDT_LEG, "_role": "rank:pairwise xgb"},
    ))
    out = tmp_path / "logs" / "derived.json"

    result = HELPER.derive_reference(pinned, strategy_dir, out)
    panel = json.loads(out.read_text())["ranking"]["panel_scoring"]

    # Derived FROM the pinned blend's component[0], NOT the working copy.
    assert panel["artifact_path"] == _GBDT_LEG
    assert panel["artifact_path"] != "artifacts/patchtst_shadow/DIVERGED_working_copy.json"
    assert result["pinned_config"] == str(pinned)
    assert str(banned) not in json.dumps(panel)


def test_c_helper_api_has_no_working_copy_source_param():
    """Structural guard: the only config source is the pinned config; the API
    exposes no working-copy / sibling parameter to smuggle one in."""
    params = set(inspect.signature(HELPER.derive_reference).parameters)
    assert params == {"pinned_config_path", "strategy_dir", "out_path"}
    src = HELPER_SRC.read_text()
    # The strategy dir is used for artifact existence + fingerprint only, never
    # as a config reference source.
    assert "working copy" in src.lower()          # the ban is documented
    assert "components[0]" in src or "component[0]" in src


# ---------------------------------------------------------------------------
# CLI contract: stdout carries ONLY the derived path (shell capture is clean).
# ---------------------------------------------------------------------------


def test_cli_prints_only_path_on_success(tmp_path):
    strategy_dir = _make_strategy_dir(tmp_path)
    pinned = _write(tmp_path / "pinned" / "strategy_config.json", _blend_config(
        {"artifact_path": _GBDT_LEG, "_role": "rank:pairwise xgb"},
    ))
    out = tmp_path / "logs" / "derived.json"

    proc = subprocess.run(
        [sys.executable, str(HELPER_SRC),
         "--pinned-config", str(pinned),
         "--strategy-dir", str(strategy_dir),
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(out)             # clean, single-line path
    assert out.exists()


def test_cli_fails_closed_nonzero_and_writes_nothing(tmp_path):
    strategy_dir = _make_strategy_dir(tmp_path)
    pinned = _write(tmp_path / "pinned" / "strategy_config.json", _blend_config(
        {"kind": "momentum_residual", "artifact_path": "artifacts/momentum/ledger.jsonl"},
    ))
    out = tmp_path / "logs" / "derived.json"

    proc = subprocess.run(
        [sys.executable, str(HELPER_SRC),
         "--pinned-config", str(pinned),
         "--strategy-dir", str(strategy_dir),
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""                   # no phantom path emitted
    assert "FAIL CLOSED" in proc.stderr
    assert not out.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
