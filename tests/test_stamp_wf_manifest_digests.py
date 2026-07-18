"""WF corpus manifests carry per-entry digests that survive enforcement.

PR #497 review finding: ``kernel/manifest_uri_resolver.py`` closes its digest
compatibility window on ``ARTIFACT_DIGEST_REQUIRED_AFTER`` (2026-09-01) — the
model-validation path then requires a stamped ``artifact_sha256`` per manifest
entry and fails closed without one. The committed corpus manifests were 0/39
stamped. These tests pin the repaired state (AC-5 of the #497 design doc):

* both committed corpus manifests are fully stamped;
* the resolver ACCEPTS every stamped entry with the enforcement date forced
  past 2026-09-01 (date injection via ``digest_required(now=...)``, exactly
  the loader's ``require = digest_required()`` wiring);
* a missing digest under enforcement fails closed;
* a tampered artifact (one-byte flip) is REFUSED with a digest mismatch;
* the stamped manifests keep the ``json.dumps(indent=2)`` shape with digest
  keys directly after their ``*_uri`` keys (the added-lines-only diff shape).

Imports only ``kernel.manifest_uri_resolver`` (stdlib-only) — no pandas / WF
pipeline dependencies.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.manifest_uri_resolver import (  # noqa: E402
    ARTIFACT_DIGEST_REQUIRED_AFTER,
    ManifestUriResolutionError,
    digest_required,
    resolve_manifest_uri,
)

_SIM_DIR = _STRATEGY_DIR / "artifacts" / "sim"
_MANIFEST_V1 = _SIM_DIR / "walkforward_manifest.json"
_MANIFEST_V2 = _SIM_DIR / "walkforward_manifest_v2_20260602.json"
_DIGEST_FIELDS = (
    ("artifact_uri", "artifact_sha256"),
    ("calibrator_uri", "calibrator_sha256"),
)


def _entries(manifest: Path) -> list[dict]:
    payload = json.loads(manifest.read_text())
    retrains = payload["retrains"]
    assert isinstance(retrains, list) and retrains
    return retrains


def _enforced() -> bool:
    """The loader's ``require = digest_required()`` with the date forced
    one day past the compatibility window (deterministic date injection)."""
    return digest_required(
        now=ARTIFACT_DIGEST_REQUIRED_AFTER + timedelta(days=1)
    )


def _deployed_v1_manifest() -> "Path | None":
    """The deployed manifest location encoded in v1's absolute URIs.

    v1 carries absolute URIs into the operator tree; resolver roots must
    derive from that manifest path (as the live loader's do). Returns None
    when that tree is not present on the running machine.
    """
    first_uri = Path(_entries(_MANIFEST_V1)[0]["artifact_uri"])
    if not first_uri.is_absolute():
        return _MANIFEST_V1
    for parent in first_uri.parents:
        candidate = parent / "walkforward_manifest.json"
        if parent.name == "sim" and candidate.exists():
            return candidate
    return None


def test_enforcement_window_date_semantics():
    day_before = ARTIFACT_DIGEST_REQUIRED_AFTER - timedelta(days=1)
    assert not digest_required(now=day_before)
    assert digest_required(now=ARTIFACT_DIGEST_REQUIRED_AFTER)
    assert _enforced()


@pytest.mark.parametrize("manifest", [_MANIFEST_V1, _MANIFEST_V2],
                         ids=["v1", "v2_20260602"])
def test_manifests_fully_stamped(manifest: Path):
    for i, entry in enumerate(_entries(manifest)):
        for uri_key, digest_key in _DIGEST_FIELDS:
            if not entry.get(uri_key):
                continue
            digest = entry.get(digest_key)
            assert isinstance(digest, str) and len(digest) == 64, (
                f"entry[{i}] {uri_key} has no 64-hex {digest_key}"
            )
            assert digest == digest.strip().lower()
            int(digest, 16)  # raises if not hex


def test_resolver_accepts_stamped_v2_under_enforcement():
    """AC-5: every v2 entry resolves with require_digest forced on."""
    require = _enforced()
    for entry in _entries(_MANIFEST_V2):
        for uri_key, digest_key in _DIGEST_FIELDS:
            uri = entry.get(uri_key)
            if not uri:
                continue
            resolved = resolve_manifest_uri(
                _MANIFEST_V2,
                uri,
                expected_digest=entry[digest_key],
                # Task #82: the loader now enforces BOTH digest fields
                # (calibrator_as_of binds calibrator_sha256 the same way
                # _resolve_model_uri binds artifact_sha256) — require both,
                # exactly as it wires.
                require_digest=require,
                digest_field=digest_key,
            )
            assert isinstance(resolved, Path) and resolved.exists()


def test_resolver_accepts_stamped_v1_under_enforcement():
    """v1 (absolute deployed-tree URIs) resolves where that tree exists."""
    deployed = _deployed_v1_manifest()
    if deployed is None:
        pytest.skip("deployed operator tree for v1 absolute URIs not present")
    require = _enforced()
    for entry in _entries(_MANIFEST_V1):
        resolved = resolve_manifest_uri(
            deployed,
            entry["artifact_uri"],
            expected_digest=entry["artifact_sha256"],
            require_digest=require,
        )
        assert isinstance(resolved, Path) and resolved.exists()


def test_missing_digest_fails_closed_under_enforcement():
    entry = _entries(_MANIFEST_V2)[0]
    with pytest.raises(ManifestUriResolutionError, match="artifact_sha256"):
        resolve_manifest_uri(
            _MANIFEST_V2,
            entry["artifact_uri"],
            expected_digest=None,
            require_digest=_enforced(),
        )


def test_tampered_artifact_refused(tmp_path: Path):
    """One-byte flip in a referenced artifact → digest mismatch error."""
    entry = _entries(_MANIFEST_V2)[0]
    uri = entry["artifact_uri"]
    source = _STRATEGY_DIR / uri
    manifest = tmp_path / "strategy" / "artifacts" / "sim" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    copy = tmp_path / "strategy" / uri
    copy.parent.mkdir(parents=True)
    payload = bytearray(source.read_bytes())
    copy.write_bytes(payload)

    # Positive control: the untampered copy passes under enforcement.
    resolved = resolve_manifest_uri(
        manifest, uri,
        expected_digest=entry["artifact_sha256"], require_digest=True,
    )
    assert isinstance(resolved, Path) and resolved.exists()

    payload[0] ^= 0x01  # one-byte flip
    copy.write_bytes(payload)
    with pytest.raises(ManifestUriResolutionError, match="digest"):
        resolve_manifest_uri(
            manifest, uri,
            expected_digest=entry["artifact_sha256"], require_digest=True,
        )


@pytest.mark.parametrize("manifest", [_MANIFEST_V1, _MANIFEST_V2],
                         ids=["v1", "v2_20260602"])
def test_stamp_preserved_manifest_format(manifest: Path):
    """Stamps changed the manifests by added lines only: the file keeps its
    ``json.dumps(indent=2)`` byte shape and each digest key sits immediately
    after its ``*_uri`` key."""
    raw = manifest.read_text()
    payload = json.loads(raw)
    body = json.dumps(payload, indent=2)
    assert raw in (body, body + "\n")
    for entry in payload["retrains"]:
        keys = list(entry)
        for uri_key, digest_key in _DIGEST_FIELDS:
            if entry.get(uri_key):
                assert keys.index(digest_key) == keys.index(uri_key) + 1
