"""Bounded manifest-URI resolver — contract tests (PR #421 review).

The resolver replaces a walk-arbitrary-ancestors / first-existing lookup that
made a validated artifact's identity depend on the machine's surrounding
filesystem. These tests pin the bounded contract the reviewer required:

  * resolve only against a small ORDERED set of known roots,
  * enforce lexical containment (reject ``..`` traversal outside the roots),
  * enforce realpath containment on existing candidates (reject an in-root
    symlink whose real target escapes every allowed root — round 3),
  * reject an absolute path outside every allowed root unless the caller
    passes ``allow_external=True`` (round 3),
  * reject ambiguity (two existing candidates with different digests),
  * bind the resolved file to an expected digest where present,
  * fail closed on a missing digest once ``require_digest`` is set, and gate
    that on the ``digest_required()`` compatibility-window date (round 3),
  * stay deterministic across checkout relocation / surrounding files.

This module imports only ``kernel.manifest_uri_resolver`` (stdlib-only) so it
runs without the heavier walk-forward pipeline dependencies.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import timedelta
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.manifest_uri_resolver import (  # noqa: E402
    ARTIFACT_DIGEST_REQUIRED_AFTER,
    ManifestUriResolutionError,
    digest_required,
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

def test_absolute_uri_within_root_returned_as_path(tmp_path):
    # Round 3: an absolute URI is no longer returned blindly — its realpath
    # must be under an allowed root. Here it is (the strategy root IS
    # tmp_path), so it resolves without needing allow_external. See
    # test_absolute_uri_outside_roots_rejected below for the rejection case.
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


# --- (3b) in-root symlink escape → rejected (PR #421 review, blocking pt 1) -

def test_symlink_escape_rejected(tmp_path):
    # The URI is lexically contained (passes _is_within), but the file is a
    # symlink whose REAL target lives outside every allowed root. The bounded
    # resolver must reject it rather than silently digesting/loading the
    # external target.
    strategy = tmp_path / "renquant_104"
    manifest = strategy / "artifacts" / "sim" / "wf.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}")
    external = _write(tmp_path / "external" / "model.json", '{"model": "external"}')
    uri = "artifacts/walkforward_gbdt/2023-10-02/panel-ltr.json"
    link_path = strategy / uri
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(external)

    with pytest.raises(ManifestUriResolutionError):
        resolve_manifest_uri(manifest, uri)


def test_symlink_within_roots_resolves(tmp_path):
    # Contrast case: a symlink whose real target is ALSO inside an allowed
    # root is fine — only escape is rejected, not symlinks per se.
    strategy = tmp_path / "renquant_104"
    manifest = strategy / "artifacts" / "sim" / "wf.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}")
    real = _write(strategy / "real" / "model.json", '{"model": "in-strategy"}')
    uri = "artifacts/walkforward_gbdt/2023-10-02/panel-ltr.json"
    link_path = strategy / uri
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(real)

    out = resolve_manifest_uri(manifest, uri)

    assert Path(out).read_text() == '{"model": "in-strategy"}'


# --- (3c) absolute path outside roots → rejected unless allow_external -----
# (PR #421 review, blocking point 2)

def test_absolute_uri_outside_roots_rejected(tmp_path):
    strategy = tmp_path / "renquant_104"
    manifest = strategy / "artifacts" / "sim" / "wf.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}")
    external = _write(tmp_path / "external" / "model.pt", "external-bytes")

    with pytest.raises(ManifestUriResolutionError):
        resolve_manifest_uri(manifest, str(external))


def test_absolute_uri_outside_roots_allowed_with_allow_external(tmp_path):
    strategy = tmp_path / "renquant_104"
    manifest = strategy / "artifacts" / "sim" / "wf.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}")
    external = _write(tmp_path / "external" / "model.pt", "external-bytes")

    out = resolve_manifest_uri(manifest, str(external), allow_external=True)

    assert Path(out) == external


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


# --- (7) require_digest fail-closed (PR #421 review, blocking point 3) -----

def test_require_digest_missing_rejected(tmp_path):
    # Model-validation path past the compatibility window: no expected_digest
    # supplied at all → fails closed rather than accepting "found a file".
    strategy, manifest = _strategy_with_manifest(tmp_path)
    uri = "artifacts/walkforward_gbdt/2023-10-02/panel-ltr.json"
    _write(strategy / uri, '{"model": "gbdt"}')
    with pytest.raises(ManifestUriResolutionError):
        resolve_manifest_uri(manifest, uri, require_digest=True)


def test_require_digest_present_accepted(tmp_path):
    strategy, manifest = _strategy_with_manifest(tmp_path)
    uri = "artifacts/walkforward_gbdt/2023-10-02/panel-ltr.json"
    real = _write(strategy / uri, '{"model": "gbdt"}')
    good = "sha256:" + _sha256(real)
    out = resolve_manifest_uri(
        manifest, uri, expected_digest=good, require_digest=True
    )
    assert out == real


def test_require_digest_false_still_tolerates_missing(tmp_path):
    # Before the compatibility window closes, require_digest defaults False
    # (caller-gated by digest_required()) — unstamped manifests keep working.
    strategy, manifest = _strategy_with_manifest(tmp_path)
    uri = "artifacts/walkforward_gbdt/2023-10-02/panel-ltr.json"
    real = _write(strategy / uri, '{"model": "gbdt"}')
    out = resolve_manifest_uri(manifest, uri, require_digest=False)
    assert out == real


def test_digest_required_date_boundary():
    # digest_required(now) is False strictly before the cutoff and True on
    # and after it. now is injectable so this is deterministic (no freezing
    # the real clock / no dependency on wall-clock "today").
    before = ARTIFACT_DIGEST_REQUIRED_AFTER - timedelta(days=1)
    on = ARTIFACT_DIGEST_REQUIRED_AFTER
    after = ARTIFACT_DIGEST_REQUIRED_AFTER + timedelta(days=1)
    assert digest_required(before) is False
    assert digest_required(on) is True
    assert digest_required(after) is True


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
