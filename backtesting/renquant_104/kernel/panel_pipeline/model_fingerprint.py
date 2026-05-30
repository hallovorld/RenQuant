"""Compute MODEL-content fingerprints (not artifact-file fingerprints).

Bug D fix (2026-05-30): the calibrator validator already prefers
``model_content_fingerprint`` over ``artifact_fingerprint`` (per
``_fingerprint_values`` priority in job_panel_scoring.py:1469). The gap
was that training scripts only stamped the file-bytes fingerprint, which
is invalidated by every metadata edit (cv_method stamp, P-PANEL-CONTRACT
fields, doc-string update, etc.). 3 rebinds today.

This module gives the canonical way to compute a STABLE fingerprint of
the model (the actual predictor), invariant to JSON/sidecar metadata edits:

  * JSON GBDT artifact:  ``sha256(booster_raw_json field)``
                          — the model lives inside this string field;
                            everything else in the JSON is metadata.

  * PyTorch .pt sequence checkpoint:  ``sha256(file_bytes)``
                          — the .pt IS the model (state_dict serialised by
                            torch.save). Metadata lives in the SIDECAR JSON
                            (``foo.pt.metadata.json``), so .pt bytes are
                            already stable across metadata edits.

  * Pickled or other formats:  ``sha256(file_bytes)`` — same reasoning.

Use ``compute_model_fingerprint(path)`` from training scripts (calibrator
fitters, WF gate) to stamp the calibrator's metadata, and from the scorer
loader to expose it in the scorer's runtime metadata. The validator then
matches them, and the operator never has to rebind again when only
metadata changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def compute_model_fingerprint(scorer_path: Path | str) -> str:
    """Return a stable sha256 of the MODEL bytes (not the artifact bytes).

    Returns the empty string when the path is missing or unrecognised. Callers
    that want strict behaviour should check the return value and fail closed.
    """
    p = Path(scorer_path)
    if not p.exists():
        return ""

    if p.suffix == ".json":
        # GBDT panel-LTR JSON. Hash the booster_raw_json string — the actual
        # XGBoost model serialisation. All other JSON fields are metadata.
        try:
            artifact = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return ""
        booster_raw = artifact.get("booster_raw_json")
        if not isinstance(booster_raw, str) or not booster_raw:
            # Legacy artifacts without booster_raw_json — fall back to file
            # hash so older folds still validate. Document the divergence.
            return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
        return "sha256:" + hashlib.sha256(booster_raw.encode("utf-8")).hexdigest()

    # .pt / .pth / .bin / other binary: the file IS the model.
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def model_fingerprint_kind(scorer_path: Path | str) -> str:
    """Diagnostic: report which path the fingerprint took."""
    p = Path(scorer_path)
    if p.suffix == ".json":
        try:
            artifact = json.loads(p.read_text())
        except Exception:
            return "file_bytes_json_parse_failed"
        return "booster_raw_json" if isinstance(artifact.get("booster_raw_json"), str) else "file_bytes_legacy_json"
    return "file_bytes_binary"
