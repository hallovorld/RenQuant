"""Build a minimal PreflightPipeline (state + broker) for Track H proof-of-life.

This factory is the migration entry point. Today it returns a pipeline with
just StateFileTask + BrokerConnectTask wired into one Job. As remaining 14
``_check_*`` functions migrate (one per PR), each lands as a new Task in an
appropriate Job, and ``build_minimal_preflight_pipeline`` is renamed
``build_preflight_pipeline`` and replaces ``kernel.preflight.run_preflight``.
"""
from __future__ import annotations

from .base import PreflightJob, PreflightPipeline
from .tasks.artifact import BestIterTask, ModelArtifactTask, PanelContractTask
from .tasks.broker import BrokerConnectTask
from .tasks.calibrator import CalibratorFlatRegionTask, CalibratorHealthTask
from .tasks.correlation import CorrelationMetadataTask
from .tasks.gate import RegimeLayeredICTask, WfGateMetadataTask
from .tasks.sector_map import SectorMapCoverageTask
from .tasks.state import StateFileTask
from .tasks.watchlist import WatchlistSizeTask


class _ArtifactJob(PreflightJob):
    """Artifact group — checks the active scorer artifact exists, parses,
    carries the contract metadata, and was trained to a healthy best_iter."""

    tasks = [
        ModelArtifactTask(),
        PanelContractTask(),
        BestIterTask(),
    ]


class _GateJob(PreflightJob):
    """WF gate + regime-layered IC — the production trust boundary
    (CLAUDE.md prime directive: regime-conditional evidence required)."""

    tasks = [WfGateMetadataTask(), RegimeLayeredICTask()]


class _IdentityJob(PreflightJob):
    """Identity-of-trained-model group — watchlist consistency, sector-map
    coverage, correlation metadata. (config_fingerprint lands in a follow-up
    PR as it's the most complex single check at 129 lines.)"""

    tasks = [
        WatchlistSizeTask(),
        SectorMapCoverageTask(),
        CorrelationMetadataTask(),
    ]


class _CalibratorJob(PreflightJob):
    """Calibrator health + structural flat-region checks. Sits between
    identity and state+broker because they ALL operate on the calibrator
    artifact and share the global_calibration-disabled soft-skip path."""

    tasks = [CalibratorHealthTask(), CalibratorFlatRegionTask()]


class _StateAndBrokerJob(PreflightJob):
    """State + broker connectivity — final checks before live decisions."""

    tasks = [StateFileTask(), BrokerConnectTask()]


def build_minimal_preflight_pipeline() -> PreflightPipeline:
    """Return a PreflightPipeline holding the migrated checks (12/16 so far).

    Job order mirrors ``kernel.preflight.run_preflight``'s ALL_CHECKS list:
    artifact → WF gate → regime IC → identity (watchlist + sector + corr) →
    calibrator (health + flat-region) → state + broker. Remaining:
    config_fingerprint, feature_coverage, artifact_run_id_alignment,
    meta_label.
    """
    return PreflightPipeline(jobs=[
        _ArtifactJob(),
        _GateJob(),
        _IdentityJob(),
        _CalibratorJob(),
        _StateAndBrokerJob(),
    ])
