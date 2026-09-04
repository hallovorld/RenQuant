"""The manifest the weekly promote gate ACTUALLY resolves is digest-stamped.

2026-07-18 stamped `walkforward_manifest.json` and
`walkforward_manifest_v2_20260602.json` ahead of the resolver's
`ARTIFACT_DIGEST_REQUIRED_AFTER = 2026-09-01` enforcement. The weekly promote
gate (`scripts/weekly_wf_promote.sh`, `WF_MANIFEST=`) had already moved on
2026-06-16 to `walkforward_manifest_gbdt_prod_recipe_v2.calibrated.json`,
which was 0/43 stamped — so on 2026-09-01 every retrain's WF simulation died
in `WalkForwardModelLoader.model_as_of` with `ManifestUriResolutionError`
("compatibility window closed 2026-09-01") and the gate reported
"3/3 sim cuts failed execution" as an ordinary reject for three days.

These tests pin the repaired state for the manifest the gate names, and pin
the NAMING itself: the manifest referenced by the promote script must be one
of the stamped manifests, so a future manifest switch cannot silently
re-open the gap.

Stdlib + `kernel.manifest_uri_resolver` only (no pandas / WF pipeline).
"""
from __future__ import annotations

import hashlib
import json
import re
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
_MANIFEST_PROD_V2 = _SIM_DIR / "walkforward_manifest_gbdt_prod_recipe_v2.calibrated.json"
_PROMOTE_SCRIPT = _REPO_ROOT / "scripts" / "weekly_wf_promote.sh"
_STAMPED_MANIFESTS = frozenset({
    "artifacts/sim/walkforward_manifest.json",
    "artifacts/sim/walkforward_manifest_v2_20260602.json",
    "artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.calibrated.json",
})
_DIGEST_FIELDS = (
    ("artifact_uri", "artifact_sha256"),
    ("calibrator_uri", "calibrator_sha256"),
)


def _entries(manifest: Path) -> list[dict]:
    retrains = json.loads(manifest.read_text())["retrains"]
    assert isinstance(retrains, list) and retrains
    return retrains


def _enforced() -> bool:
    """The loader's ``require = digest_required()`` with the date forced one
    day past the window — the exact call that killed the 09-01..09-03 runs."""
    return digest_required(now=ARTIFACT_DIGEST_REQUIRED_AFTER + timedelta(days=1))


def _promote_gate_manifest() -> str:
    """The one `WF_MANIFEST="..."` assignment in the weekly promote script."""
    text = _PROMOTE_SCRIPT.read_text()
    found = re.findall(r'^WF_MANIFEST="([^"]+)"\s*$', text, flags=re.M)
    assert len(found) == 1, f"expected exactly one WF_MANIFEST assignment, got {found}"
    return found[0]


def test_promote_gate_names_a_stamped_manifest():
    """The 07-18 stamp validated two manifests the gate no longer used."""
    named = _promote_gate_manifest()
    assert named in _STAMPED_MANIFESTS, (
        f"weekly_wf_promote.sh resolves {named!r}, which is not in the stamped "
        f"set {sorted(_STAMPED_MANIFESTS)} — stamp it (scripts/"
        f"stamp_wf_manifest_digests.py) and add it here before the gate runs"
    )
    assert named == "artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.calibrated.json"


def test_prod_recipe_v2_manifest_fully_stamped():
    for i, entry in enumerate(_entries(_MANIFEST_PROD_V2)):
        for uri_key, digest_key in _DIGEST_FIELDS:
            assert entry.get(uri_key), f"entry[{i}] has no {uri_key}"
            digest = entry.get(digest_key)
            assert isinstance(digest, str) and len(digest) == 64, (
                f"entry[{i}] {uri_key} has no 64-hex {digest_key}"
            )
            assert digest == digest.strip().lower()
            int(digest, 16)


def test_resolver_accepts_every_prod_recipe_v2_entry_under_enforcement():
    """AC: the exact post-window loader call succeeds for all 43 entries,
    both digest fields, against the COMMITTED corpus bytes."""
    require = _enforced()
    assert require
    n = 0
    for entry in _entries(_MANIFEST_PROD_V2):
        for uri_key, digest_key in _DIGEST_FIELDS:
            resolved = resolve_manifest_uri(
                _MANIFEST_PROD_V2,
                entry[uri_key],
                expected_digest=entry[digest_key],
                require_digest=require,
                digest_field=digest_key,
            )
            assert isinstance(resolved, Path) and resolved.exists()
            n += 1
    assert n == 86


def test_missing_digest_on_prod_recipe_v2_fails_closed_under_enforcement():
    """The 2026-09-01..03 failure, reproduced deterministically."""
    entry = _entries(_MANIFEST_PROD_V2)[0]
    with pytest.raises(ManifestUriResolutionError, match="compatibility window closed"):
        resolve_manifest_uri(
            _MANIFEST_PROD_V2,
            entry["artifact_uri"],
            expected_digest=None,
            require_digest=_enforced(),
        )


def test_committed_corpus_is_the_fingerprint_stamped_state():
    """Step 3.5 of the promote gate (`stamp_walkforward_fingerprints.py`)
    rewrites any referenced scorer whose `config_fingerprint` differs and any
    calibrator whose scorer binding differs. The committed bytes must already
    BE that state, or the first run after a checkout rewrites the files and
    every digest above mismatches (fail-closed, gate dark again)."""
    for entry in _entries(_MANIFEST_PROD_V2):
        scorer_path = _STRATEGY_DIR / entry["artifact_uri"]
        scorer = json.loads(scorer_path.read_text())
        assert scorer.get("config_fingerprint"), f"{entry['artifact_uri']}: no config_fingerprint"
        cal = json.loads((_STRATEGY_DIR / entry["calibrator_uri"]).read_text())
        meta = cal.get("metadata") or {}
        bound = str(meta.get("scorer_artifact_sha256") or "")
        assert bound.startswith("sha256:"), f"{entry['calibrator_uri']}: no scorer_artifact_sha256 binding"
        assert bound[len("sha256:"):] == hashlib.sha256(scorer_path.read_bytes()).hexdigest(), (
            f"{entry['calibrator_uri']}: scorer binding is not the committed scorer's bytes"
        )
        assert bound[len("sha256:"):] == entry["artifact_sha256"]


def test_prod_recipe_v2_manifest_keeps_added_lines_only_shape():
    raw = _MANIFEST_PROD_V2.read_text()
    payload = json.loads(raw)
    body = json.dumps(payload, indent=2)
    assert raw in (body, body + "\n")
    for entry in payload["retrains"]:
        keys = list(entry)
        for uri_key, digest_key in _DIGEST_FIELDS:
            assert keys.index(digest_key) == keys.index(uri_key) + 1
