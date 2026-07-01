"""Sim artifact-metadata helpers — sim.py decomposition slice 2.

EXTRACTED 2026-06-13 from adapters/sim.py (eng plan S2 item 5). Pure
helpers for reading artifact/model metadata: artifact kind, history
seq_len, model type, inference-forbidden column drop (the leakage guard
that strips label/fwd_* columns before scoring), and manifest URI
resolution. No SimAdapter state. Re-exported from sim for back-compat.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from kernel.decision_trace import model_type_from_artifact
from kernel.manifest_uri_resolver import resolve_manifest_uri

# Leakage guard: columns that must never reach an inference feature frame.
_FORBIDDEN_HISTORY_COL_PREFIXES = ("fwd_",)
_FORBIDDEN_HISTORY_COLS = {"label", "split_label"}


def _artifact_kind(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    meta = payload.get("metadata") if isinstance(payload, dict) else None
    if isinstance(meta, dict) and meta.get("kind"):
        return str(meta.get("kind"))
    if isinstance(payload, dict) and payload.get("kind"):
        return str(payload.get("kind"))
    return None


def _history_seq_len_from_artifact(path: Path) -> int | None:
    """Best-effort sequence length probe without loading a Torch checkpoint."""
    candidates = [
        path.with_name(path.name + ".metadata.json"),
        path.with_name(path.stem + "_metadata.json"),
        path.with_name(path.stem + "_summary.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        contract = payload.get("training_contract") or {}
        hparams = contract.get("hyperparameters") or {}
        raw = payload.get("seq_len") or hparams.get("seq_len")
        if raw:
            return int(raw)
    return None


def _model_type_from_artifact(model: Any) -> str | None:
    """Extract readable model type from dict/object artifacts for audit rows."""
    return model_type_from_artifact(model)


def _drop_inference_forbidden_cols(df: pd.DataFrame) -> pd.DataFrame:
    forbidden = [
        c for c in df.columns
        if c in _FORBIDDEN_HISTORY_COLS
        or any(str(c).startswith(prefix) for prefix in _FORBIDDEN_HISTORY_COL_PREFIXES)
    ]
    return df.drop(columns=forbidden) if forbidden else df


def _resolve_manifest_uri(manifest_path: Path, uri: str) -> Path:
    """Resolve a manifest artifact URI via the shared bounded resolver.

    Thin wrapper over ``kernel.walk_forward.uri_resolver.resolve_manifest_uri``
    so this call site shares the single URI contract (bounded known roots,
    containment, ambiguity rejection) with the WF loader and the gate script,
    instead of a drifting local copy. Manifest URIs are normally
    manifest-folder-relative, but orchestrator-built WF manifests emit
    strategy-dir-relative URIs (``artifacts/walkforward_.../panel-ltr.json``);
    the shared resolver handles both against an ordered set of known roots.
    """
    return Path(resolve_manifest_uri(manifest_path, uri))
