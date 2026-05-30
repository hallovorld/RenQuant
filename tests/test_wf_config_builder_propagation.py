"""Regression test for wf_config_builder.build_wf_config_from_prod propagation.

The 2026-05-30 incident: v2 gate stamped passed=False even though the
user-supplied base config had wf_gate.sanity_regime_ic_required=False,
because the derive operation stripped the wf_gate block when producing
the prod-semantic config. The gate's _read_wf_gate_relax() then read from
the derived (empty) config and defaulted to strict.

The fix: build_wf_config_from_prod now copies the base config's wf_gate
block verbatim into the derived output. This test pins that behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from wf_config_builder import build_wf_config_from_prod  # noqa: E402


@pytest.fixture
def minimal_prod_cfg():
    return {
        "ranking": {"panel_scoring": {"enabled": True}},
        "walkforward": {},
    }


def test_wf_gate_block_propagated_from_base(minimal_prod_cfg, tmp_path):
    base = dict(minimal_prod_cfg)
    base["wf_gate"] = {
        "benchmark_required": False,
        "regime_required": False,
        "sanity_regime_ic_required": False,
    }
    derived = build_wf_config_from_prod(
        minimal_prod_cfg,
        manifest_path="dummy.json",
        base_wf_config=base,
        strategy_dir=tmp_path,
    )
    assert "wf_gate" in derived, "wf_gate block must propagate"
    assert derived["wf_gate"]["benchmark_required"] is False
    assert derived["wf_gate"]["regime_required"] is False
    assert derived["wf_gate"]["sanity_regime_ic_required"] is False


def test_wf_gate_block_absent_when_base_lacks_it(minimal_prod_cfg, tmp_path):
    """Base without wf_gate → derived without wf_gate (preserves strict default)."""
    base = dict(minimal_prod_cfg)  # no wf_gate key
    derived = build_wf_config_from_prod(
        minimal_prod_cfg,
        manifest_path="dummy.json",
        base_wf_config=base,
        strategy_dir=tmp_path,
    )
    assert "wf_gate" not in derived


def test_wf_gate_block_deep_copied_not_aliased(minimal_prod_cfg, tmp_path):
    """Mutating the derived dict must not retroactively mutate the base."""
    base = dict(minimal_prod_cfg)
    base["wf_gate"] = {"benchmark_required": False}
    derived = build_wf_config_from_prod(
        minimal_prod_cfg,
        manifest_path="dummy.json",
        base_wf_config=base,
        strategy_dir=tmp_path,
    )
    derived["wf_gate"]["benchmark_required"] = True  # mutate derived
    # Base must be untouched
    assert base["wf_gate"]["benchmark_required"] is False
