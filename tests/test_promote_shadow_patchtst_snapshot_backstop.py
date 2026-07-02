"""SUCCESSFUL-SWAP wiring test for promote_shadow_patchtst.py's snapshot
freshness backstop (Codex PR #432 round 5).

Complements the isolated `_apply_snapshot_freshness_backstop` unit tests in
tests/test_promote_shadow_patchtst.py: those prove the function's own
behavior; THIS file proves run_promote() actually reaches it on exactly the
successful `--apply` swap path. It drives a REAL executed swap in a fixture
repo — write-new copy of the candidate .pt + sidecar, superseded-config
backup, atomic pin rewrite, promote log — and asserts:

  * stale snapshot -> rc flips RC_OK -> RC_GATE_FAILED AFTER the swap, the
    verdict still records the PROMOTED pin, and the on-disk config keeps
    the NEW pin (never auto-reverted for a stale-snapshot finding alone);
  * fresh snapshot -> rc stays RC_OK;
  * dry-run (no swap) -> the backstop is never consulted.

Per the round-4 review, the snapshot checker is INJECTED (a fake
promote_pin module in sys.modules, the seam the backstop lazily imports)
rather than rendered against live sources, and only the two heavy external
seams unrelated to the backstop — the fingerprint stamp subprocess and the
live-fingerprint read — are stubbed; freshness/advance/sanity gates run for
real off fixture sidecars. check_snapshot_freshness's own render/diff logic
stays covered non-mocked in tests/test_promote_pin.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "promote_shadow_patchtst_swap_backstop_target",
    REPO / "scripts" / "promote_shadow_patchtst.py")
psp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = psp  # dataclasses need the module resolvable by name
_SPEC.loader.exec_module(psp)


def _sidecar(pt_path: Path, *, train: str, sel: str, wf_ic: float | None = None) -> None:
    meta = {
        "trained_date": train,
        "effective_train_cutoff_date": train,
        "effective_selection_cutoff_date": sel,
        "lookahead_days": 5,
        "training_contract": {"label_col": "fwd_5d"},
    }
    if wf_ic is not None:
        meta["wf_ic"] = wf_ic
    Path(str(pt_path) + ".metadata.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture()
def swap_fixture(tmp_path, monkeypatch):
    """Fixture repo with a served hf_patchtst pin (older cutoffs) + an
    advancing candidate; every gate satisfiable without torch/network."""
    repo = tmp_path
    cfg_dir = repo / "backtesting" / "renquant_104"
    served_dir = cfg_dir / "artifacts" / "served"
    served_dir.mkdir(parents=True)

    served_pt = served_dir / "current.pt"
    served_pt.write_bytes(b"served-weights")
    _sidecar(served_pt, train="2026-05-01", sel="2026-05-01")

    cand_pt = repo / "candidates" / "retrain_20260601.pt"
    cand_pt.parent.mkdir(parents=True)
    cand_pt.write_bytes(b"candidate-weights")
    _sidecar(cand_pt, train="2026-06-01", sel="2026-06-01", wf_ic=0.05)

    served_config = cfg_dir / "strategy_config.shadow.json"
    served_config.write_text(json.dumps({
        "ranking": {"panel_scoring": {
            "kind": "hf_patchtst",
            "artifact_path": os.path.relpath(served_pt, cfg_dir),
        }},
    }), encoding="utf-8")

    monkeypatch.setattr(psp, "REPO", repo)
    # The two heavy seams UNRELATED to the backstop under test: the external
    # stamp subprocess (§3.3) and the live-fingerprint sidecar read.
    monkeypatch.setattr(psp, "stamp_fingerprint", lambda *a, **k: (0, "stub stamp OK"))
    monkeypatch.setattr(psp, "live_config_fingerprint", lambda *a, **k: None)

    args = argparse.Namespace(
        served_config="backtesting/renquant_104/strategy_config.shadow.json",
        pin_key="ranking.panel_scoring.artifact_path",
        wf_manifest="unused-manifest.json",
        candidate=str(cand_pt),
        served_root="artifacts/patchtst_shadow",
        stamp_script="scripts/stamp_patchtst_fingerprint.py",
        sources_json="[]",   # no source SLAs -> freshness rides the cutoff advance
        fast_ceiling_days=psp.FAST_CEILING_DAYS,
        sanity_floor=0.0,
        resource_max_seconds=120.0,
        resource_max_rss_mb=4096.0,
        allow_non_fresh=False,
        reason=None,
        skip_inference_gate=True,
        apply=True,
        check=False,
        json=False,
        now=dt.date(2026, 7, 2),
    )
    return repo, served_config, args


def _inject_checker(monkeypatch, fresh: bool, msg: str) -> list:
    calls: list[tuple] = []

    def fake_check(python, repo=None):
        calls.append((python, repo))
        return fresh, msg

    fake_pp = types.ModuleType("promote_pin")
    fake_pp.check_snapshot_freshness = fake_check
    monkeypatch.setitem(sys.modules, "promote_pin", fake_pp)
    return calls


def _pin_on_disk(served_config: Path) -> str:
    cfg = json.loads(served_config.read_text(encoding="utf-8"))
    return cfg["ranking"]["panel_scoring"]["artifact_path"]


def test_successful_swap_then_stale_snapshot_fails_rc_without_revert(
        swap_fixture, monkeypatch):
    repo, served_config, args = swap_fixture
    calls = _inject_checker(monkeypatch, False,
                            "ACTION REQUIRED: snapshot STALE (injected)")

    rep = psp.run_promote(args)

    # The swap itself succeeded and is recorded...
    assert rep.verdict.startswith("PROMOTED"), rep.verdict
    assert rep.promoted_pin, "a real swap must have produced a new pin"
    new_pt = (served_config.parent / rep.promoted_pin).resolve()
    assert new_pt.read_bytes() == b"candidate-weights"
    assert rep.superseded_backup and Path(rep.superseded_backup).exists()
    # ...the backstop ran exactly once, against THIS repo...
    assert len(calls) == 1
    assert calls[0][1] == repo
    # ...and flipped ONLY the exit code, not the swap.
    assert rep.rc == psp.RC_GATE_FAILED
    assert "snapshot STALE (injected)" in rep.verdict
    assert _pin_on_disk(served_config) == rep.promoted_pin, (
        "the on-disk pin must keep the NEW artifact — a stale snapshot is "
        "reported, never auto-reverted"
    )


def test_successful_swap_with_fresh_snapshot_keeps_rc_ok(swap_fixture, monkeypatch):
    repo, served_config, args = swap_fixture
    calls = _inject_checker(monkeypatch, True, "snapshot fresh (injected)")

    rep = psp.run_promote(args)

    assert rep.rc == psp.RC_OK
    assert rep.verdict.startswith("PROMOTED")
    assert "snapshot fresh (injected)" in rep.verdict
    assert len(calls) == 1
    assert _pin_on_disk(served_config) == rep.promoted_pin


def test_dry_run_never_invokes_the_backstop(swap_fixture, monkeypatch):
    """No swap -> no production state changed -> the backstop must not fire
    (and must not flip a clean dry-run verdict)."""
    repo, served_config, args = swap_fixture
    args.apply = False
    old_pin = _pin_on_disk(served_config)
    calls = _inject_checker(monkeypatch, False, "should never be consulted")

    rep = psp.run_promote(args)

    assert rep.rc == psp.RC_OK
    assert rep.verdict.startswith("DRY-RUN OK")
    assert not calls
    assert _pin_on_disk(served_config) == old_pin
