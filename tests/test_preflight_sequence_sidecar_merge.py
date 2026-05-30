"""Bug fix regression — preflight reads merged sidecar payloads for .pt artifacts.

2026-05-30: the pre-fix order put ``_summary.json`` first, so wf_gate_metadata
written into ``.pt.metadata.json`` by the WF gate runner was invisible to
preflight (the older summary always existed and short-circuited the read).
P-WF-GATE incorrectly reported "WF gate metadata absent" even when the gate
had stamped real Sharpe + sanity numbers.

Post-fix: ``_load_sequence_sidecar`` MERGES every existing sidecar so both
the training-time summary (best_val_ic, config_fingerprint) and the gate-time
wf_gate_metadata survive in the same payload.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.preflight import _load_sequence_sidecar, _sequence_sidecar_paths


@pytest.fixture
def fake_artifact_with_both_sidecars(tmp_path):
    """Create a fake .pt + both sidecars (the production layout)."""
    pt = tmp_path / "hf_patchtst_all_seed42_model.pt"
    pt.write_bytes(b"fake")
    # Summary (training stamp)
    summary = tmp_path / "hf_patchtst_all_seed42_summary.json"
    summary.write_text(json.dumps({
        "arch": "hf_patchtst",
        "best_val_ic": 0.0182,
        "n_features": 169,
        "config_fingerprint": "sha256:abc",
    }))
    # Metadata sidecar (WF gate stamp)
    pt_meta = tmp_path / "hf_patchtst_all_seed42_model.pt.metadata.json"
    pt_meta.write_text(json.dumps({
        "kind": "hf_patchtst",
        "metadata": {
            "wf_gate_metadata": {
                "passed": False,
                "wf_3cut_sharpe_mean": 0.65,
                "wf_reason": "FAIL: …",
                "sanity_regime_ic": {"passed": False, "reason": "regime sanity IC failed: BEAR"},
            },
        },
    }))
    return pt


def test_sidecar_order_metadata_first():
    """Order priority: .pt.metadata.json before _summary.json (post-fix)."""
    paths = _sequence_sidecar_paths(Path("/tmp/foo_model.pt"))
    names = [p.name for p in paths]
    assert names == ["foo_model.pt.metadata.json", "foo_summary.json"]


def test_load_sequence_sidecar_merges_both(fake_artifact_with_both_sidecars):
    """Merged payload contains BOTH wf_gate_metadata AND best_val_ic."""
    payload, primary = _load_sequence_sidecar(fake_artifact_with_both_sidecars)
    # From metadata sidecar:
    assert "metadata" in payload
    wf = payload["metadata"]["wf_gate_metadata"]
    assert wf["passed"] is False
    assert wf["wf_3cut_sharpe_mean"] == 0.65
    assert wf["sanity_regime_ic"]["passed"] is False
    # From summary sidecar:
    assert payload["best_val_ic"] == 0.0182
    assert payload["n_features"] == 169
    assert payload["config_fingerprint"] == "sha256:abc"
    # primary points to first existing sidecar in priority order
    assert primary.name == "hf_patchtst_all_seed42_model.pt.metadata.json"


def test_load_sequence_sidecar_only_summary(tmp_path):
    """When only the summary exists, payload contains its fields with no metadata."""
    pt = tmp_path / "hf_patchtst_all_seed42_model.pt"
    pt.write_bytes(b"fake")
    summary = tmp_path / "hf_patchtst_all_seed42_summary.json"
    summary.write_text(json.dumps({"arch": "hf_patchtst", "best_val_ic": 0.05}))
    payload, primary = _load_sequence_sidecar(pt)
    assert payload["best_val_ic"] == 0.05
    assert "metadata" not in payload
    assert primary.name == "hf_patchtst_all_seed42_summary.json"


def test_load_sequence_sidecar_only_metadata(tmp_path):
    """When only the .pt.metadata.json exists, wf_gate_metadata still surfaces."""
    pt = tmp_path / "x_model.pt"
    pt.write_bytes(b"fake")
    md = tmp_path / "x_model.pt.metadata.json"
    md.write_text(json.dumps({"metadata": {"wf_gate_metadata": {"passed": True}}}))
    payload, _ = _load_sequence_sidecar(pt)
    assert payload["metadata"]["wf_gate_metadata"]["passed"] is True


def test_load_sequence_sidecar_neither_raises(tmp_path):
    pt = tmp_path / "ghost_model.pt"
    pt.write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="missing sequence sidecar"):
        _load_sequence_sidecar(pt)
