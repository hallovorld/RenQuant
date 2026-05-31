"""Build the Track H PreflightPipeline."""
from __future__ import annotations

from .base import PreflightJob, PreflightPipeline
from .tasks.artifact import BestIterTask, ModelArtifactTask, PanelContractTask
from .tasks.broker import BrokerConnectTask
from .tasks.calibrator import CalibratorFlatRegionTask, CalibratorHealthTask
from .tasks.config_fingerprint import ConfigFingerprintTask
from .tasks.correlation import CorrelationMetadataTask
from .tasks.feature_coverage import FeatureCoverageTask
from .tasks.gate import RegimeLayeredICTask, WfGateMetadataTask
from .tasks.meta_label import MetaLabelArtifactContractTask
from .tasks.run_id import ArtifactRunIdAlignmentTask
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
    """Identity-of-trained-model group — config fingerprint, watchlist
    consistency, sector-map coverage, correlation metadata."""

    tasks = [
        ConfigFingerprintTask(),
        WatchlistSizeTask(),
        SectorMapCoverageTask(),
        CorrelationMetadataTask(),
    ]


class _CalibratorJob(PreflightJob):
    """Calibrator health + structural flat-region checks. Sits between
    identity and state+broker because they ALL operate on the calibrator
    artifact and share the global_calibration-disabled soft-skip path."""

    tasks = [CalibratorHealthTask(), CalibratorFlatRegionTask()]


class _NgboostAuxJob(PreflightJob):
    """NGBoost-dependent auxiliary checks: feature coverage between the
    NGBoost head + panel-LTR scorer, and train_run_id alignment. Both checks
    share the ``_ngboost_activation()`` skip-if-disabled gate."""

    tasks = [FeatureCoverageTask(), ArtifactRunIdAlignmentTask()]


class _MetaLabelJob(PreflightJob):
    """Meta-label exit-veto contract — runs only when ranking.meta_label.enabled.
    Stands alone as its own Job because (a) the artifact is independent from
    panel + NGBoost, and (b) the contract has many distinct failure modes."""

    tasks = [MetaLabelArtifactContractTask()]


class _StateAndBrokerJob(PreflightJob):
    """State + broker connectivity — final checks before live decisions."""

    tasks = [StateFileTask(), BrokerConnectTask()]


def build_preflight_pipeline() -> PreflightPipeline:
    """Return the FULL PreflightPipeline holding ALL 16 migrated checks.

    Jobs run in semantic dependency order. ``kernel.preflight.run_preflight``
    preserves the legacy returned-list order by sorting the results after the
    pipeline run.

    This replaces the prior ``build_minimal_preflight_pipeline()`` factory
    (kept as alias for back-compat). Track H migration COMPLETE at this PR.
    Follow-up: retire ``kernel.preflight.run_preflight`` functional path
    and make it a thin wrapper around ``PreflightPipeline.run`` + config-
    driven thresholds.
    """
    return PreflightPipeline(jobs=[
        _ArtifactJob(),
        _GateJob(),
        _IdentityJob(),
        _CalibratorJob(),
        _NgboostAuxJob(),
        _MetaLabelJob(),
        _StateAndBrokerJob(),
    ])


# Back-compat alias — older imports still work
def build_minimal_preflight_pipeline() -> PreflightPipeline:
    """DEPRECATED alias — use ``build_preflight_pipeline`` instead."""
    return build_preflight_pipeline()
