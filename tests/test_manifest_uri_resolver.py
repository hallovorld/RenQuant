"""Bounded manifest-URI resolver — contract tests (PR #421 review).

The resolver replaces a walk-arbitrary-ancestors / first-existing lookup that
made a validated artifact's identity depend on the machine's surrounding
filesystem. These tests pin the bounded contract the reviewer required:

  * resolve only against a small ORDERED set of known roots,
  * enforce containment (reject traversal outside the roots),
  * reject ambiguity (two existing candidates with different digests),
  * bind the resolved file to an expected digest where present,
  * stay deterministic across checkout relocation / surrounding files.

This module imports only ``kernel.manifest_uri_resolver`` (stdlib-only) so it
runs without the heavier walk-forward pipeline dependencies.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.manifest_uri_resolver import (  # noqa: E402
    ManifestUriResolutionError,
    resolve_manifest_uri,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strategy_with_manifest(base: Path) -> tuple[Path, Path]:
    """Build ``<strategy>/artifacts/sim/walkforward_manifest.json`` and return
    ``(strategy_root, manifest_path)``."""
    strategy = base
    manifest = strategy / "artifacts" / "sim" / "walkforward_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}")
    return strategy, manifest


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --- pass-through ----------------------------------------------------------

def test_absolute_uri_untouched(tmp_path):
    _strategy, manifest = _strategy_with_manifest(tmp_path)
    abs_uri = str(tmp_path / "elsewhere" / "model.pt")
    out = resolve_manifest_uri(manifest, abs_uri)
    assert out == Path(abs_uri)
    assert Path(out).is_absolute()


def test_scheme_uri_untouched(tmp_path):
    _strategy, manifest = _strategy_with_manifest(tmp_path)
    uri = "s3://bucket/walkforward/panel-ltr.json"
    out = resolve_manifest_uri(manifest, uri)
    assert out == uri  # returned verbatim, still a str


# --- (1) manifest-relative resolves ---------------------------------------

def test_manifest_relative_uri_resolves(tmp_path):
    _strategy, manifest = _strategy_with_manifest(tmp_path)
    real = _write(manifest.parent / "calib" / "panel-rank-calibration.json", "{}")
    out = resolve_manifest_uri(manifest, "calib/panel-rank-calibration.json")
    assert out == real
    assert Path(out).exists()


# --- (2) repo/strategy-relative resolves ----------------------------------

def test_strategy_dir_relative_uri_resolves(tmp_path):
    strategy, manifest = _strategy_with_manifest(tmp_path)
    uri = "artifacts/walkforward_gbdt_prod_recipe_v2/2023-10-02/panel-ltr.json"
    real = _write(strategy / uri, '{"model": "gbdt"}')
    out = resolve_manifest_uri(manifest, uri)
    assert out == real
    assert Path(out).exists()
    # Explicitly NOT the doubled ``artifacts/sim/artifacts/...`` prefix.
    assert out != manifest.parent / uri


# --- fallback: nothing exists → manifest-relative (meaningful not-found) ---

def test_missing_uri_falls_back_to_manifest_relative(tmp_path):
    _strategy, manifest = _strategy_with_manifest(tmp_path)
    out = resolve_manifest_uri(manifest, "artifacts/nope/panel-ltr.json")
    assert out == manifest.parent / "artifacts/nope/panel-ltr.json"
    assert not Path(out).exists()


# --- (3) traversal outside allowed roots → rejected ------------------------

def test_traversal_outside_roots_rejected(tmp_path):
    _strategy, manifest = _strategy_with_manifest(tmp_path)
    with pytest.raises(ManifestUriResolutionError):
        resolve_manifest_uri(manifest, "../../../../../../../../etc/passwd")


def test_traversal_just_outside_strategy_rejected(tmp_path):
    # A single ``..`` off the strategy root still escapes every allowed root.
    _strategy, manifest = _strategy_with_manifest(tmp_path)
    with pytest.raises(ManifestUriResolutionError):
        resolve_manifest_uri(manifest, "../sibling/model.json")


# --- (4) two conflicting candidates → error --------------------------------

def test_conflicting_candidates_rejected(tmp_path):
    strategy, manifest = _strategy_with_manifest(tmp_path)
    uri = "shared/model.json"
    # Same relative path exists under BOTH roots with DIFFERENT content.
    _write(manifest.parent / uri, '{"id": "manifest-copy"}')   # root1
    _write(strategy / uri, '{"id": "strategy-copy"}')          # root2
    with pytest.raises(ManifestUriResolutionError):
        resolve_manifest_uri(manifest, uri)


def test_identical_candidates_not_ambiguous(tmp_path):
    strategy, manifest = _strategy_with_manifest(tmp_path)
    uri = "shared/model.json"
    same = '{"id": "same-bytes"}'
    _write(manifest.parent / uri, same)   # root1
    _write(strategy / uri, same)          # root2
    out = resolve_manifest_uri(manifest, uri)
    # Deterministic: manifest-folder root wins the tie (ordered first).
    assert out == manifest.parent / uri


# --- (5) deterministic across relocation; ignores surrounding filesystem ----

def _build_relocatable(base: Path) -> tuple[Path, Path, str, Path, Path]:
    strategy = base / "checkout" / "renquant_104"
    manifest = strategy / "artifacts" / "sim" / "wf.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}")
    uri = "artifacts/walkforward_gbdt/2023-10-02/panel-ltr.json"
    real = _write(strategy / uri, '{"model": "in-strategy"}')
    # A same-named DECOY in an ancestor ABOVE the strategy root — the
    # "surrounding filesystem" a walk-arbitrary-ancestors resolver would grab.
    decoy = _write(base / uri, '{"model": "surrounding-fs-decoy"}')
    return strategy, manifest, uri, real, decoy


def test_deterministic_across_relocation(tmp_path):
    s1, m1, uri, real1, _d1 = _build_relocatable(tmp_path / "loc_a")
    s2, m2, _uri, real2, _d2 = _build_relocatable(tmp_path / "loc_b")

    out1 = resolve_manifest_uri(m1, uri)
    out2 = resolve_manifest_uri(m2, uri)

    assert out1 == real1
    assert out2 == real2
    assert Path(out1).read_text() == '{"model": "in-strategy"}'
    # Same relative resolution under a different absolute prefix.
    assert out1.relative_to(s1) == out2.relative_to(s2)


def test_same_named_file_outside_roots_never_selected(tmp_path):
    # No in-strategy copy; only a decoy above the strategy root. The bounded
    # resolver must NOT return it (a parent-walking resolver would have).
    strategy = tmp_path / "renquant_104"
    manifest = strategy / "artifacts" / "sim" / "wf.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}")
    uri = "artifacts/walkforward_gbdt/2023-10-02/panel-ltr.json"
    decoy = _write(tmp_path / uri, '{"model": "decoy"}')

    out = resolve_manifest_uri(manifest, uri)

    assert out != decoy
    assert not Path(out).exists()  # honest not-found, not the surrounding decoy


# --- (6) digest binding ----------------------------------------------------

def test_digest_mismatch_rejected(tmp_path):
    strategy, manifest = _strategy_with_manifest(tmp_path)
    uri = "artifacts/walkforward_gbdt/2023-10-02/panel-ltr.json"
    _write(strategy / uri, '{"model": "gbdt"}')
    wrong = "sha256:" + "0" * 64
    with pytest.raises(ManifestUriResolutionError):
        resolve_manifest_uri(manifest, uri, expected_digest=wrong)


def test_digest_match_accepted(tmp_path):
    strategy, manifest = _strategy_with_manifest(tmp_path)
    uri = "artifacts/walkforward_gbdt/2023-10-02/panel-ltr.json"
    real = _write(strategy / uri, '{"model": "gbdt"}')
    good = "sha256:" + _sha256(real)
    out = resolve_manifest_uri(manifest, uri, expected_digest=good)
    assert out == real


def test_digest_ignored_when_absent(tmp_path):
    # No expected_digest supplied → bounded resolution still applies, no raise.
    strategy, manifest = _strategy_with_manifest(tmp_path)
    uri = "artifacts/walkforward_gbdt/2023-10-02/panel-ltr.json"
    real = _write(strategy / uri, '{"model": "gbdt"}')
    assert resolve_manifest_uri(manifest, uri, expected_digest=None) == real


# --- call-site delegation: the WF loader shares the same contract ----------

def test_loader_resolve_uri_delegates(tmp_path):
    # The walk-forward package __init__ pulls the pipeline subrepo; skip where
    # it isn't assembled (bare clone) — it IS present on the real WF-gate venv.
    pytest.importorskip("renquant_pipeline")
    from kernel.walk_forward.loader import WalkForwardModelLoader  # noqa: PLC0415

    strategy, manifest = _strategy_with_manifest(tmp_path)
    manifest.write_text('{"retrains": []}')
    uri = "artifacts/walkforward_gbdt/2023-10-02/panel-ltr.json"
    real = _write(strategy / uri, "{}")
    loader = WalkForwardModelLoader(manifest)
    assert loader._resolve_uri(uri) == real
    with pytest.raises(ManifestUriResolutionError):
        loader._resolve_uri("../../../../../../etc/passwd")
