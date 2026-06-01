"""Walk-forward model loading + leakage guards.

P1 (2026-05-10) public surface:
    RetrainEntry              — single retrain record (frozen dataclass)
    WalkForwardManifest       — list of RetrainEntry + cadence metadata
    WalkForwardModelLoader    — wraps a manifest, exposes model_as_of
    read_manifest / write_manifest — JSON I/O helpers
    assert_no_leakage         — the single-source-of-truth leakage check

Per CLAUDE.md §5.13.5 (single source of truth): both legacy static-model
sim and walk-forward sim share `assert_no_leakage` from this module.
Adding a parallel implementation requires deleting this one first.
"""
from __future__ import annotations

from renquant_pipeline.kernel.walk_forward.correlation_guard import (
    assert_correlation_no_leakage,
    parse_correlation_artifact,
)
from renquant_pipeline.kernel.walk_forward.gmm_guard import (
    assert_gmm_no_leakage,
    gmm_artifact_as_of,
)
from renquant_pipeline.kernel.walk_forward.leakage_guard import assert_no_leakage
from renquant_pipeline.kernel.walk_forward.lean_guard import assert_lean_panel_no_leakage
from kernel.walk_forward.loader import (
    RetrainEntry,
    WalkForwardModelLoader,
)
from kernel.walk_forward.manifest import (
    WalkForwardManifest,
    read_manifest,
    write_manifest,
)

__all__ = [
    "assert_no_leakage",
    "assert_lean_panel_no_leakage",
    "assert_correlation_no_leakage",
    "parse_correlation_artifact",
    "assert_gmm_no_leakage",
    "gmm_artifact_as_of",
    "RetrainEntry",
    "WalkForwardModelLoader",
    "WalkForwardManifest",
    "read_manifest",
    "write_manifest",
]
