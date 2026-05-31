"""Preflight T/J/P refactor — Track H scaffolding.

Drop-in architecture for ``kernel.preflight.run_preflight`` migration:

  - PreflightContext: dataclass shared across all Tasks (read-mostly)
  - PreflightTask: subclass of the canonical kernel.pipeline.Task ABC, with
    a ``check_name``/``severity`` contract that maps to PreflightCheck
  - PreflightJob: groups related PreflightTasks; runs sequentially
  - PreflightPipeline: orchestrates Jobs in declaration order; ``run`` returns
    list[PreflightCheck] identical in shape to the legacy ``run_preflight``

Migration strategy (one PR per group):
  1. THIS PR: scaffolding + 2 example Tasks (StateFileTask, BrokerConnectTask)
     proven byte-equivalent to legacy ``_check_state_file`` / ``_check_broker_connect``
     by paired tests.
  2. Later PRs lift remaining 14 ``_check_*`` functions into Tasks.
  3. Final PR retires the legacy functional layout; ``run_preflight`` becomes
     a thin wrapper around ``PreflightPipeline.run``.
"""
from .ctx import PreflightContext
from .base import PreflightTask, PreflightJob, PreflightPipeline
from .tasks.state import StateFileTask
from .tasks.broker import BrokerConnectTask
from .tasks.artifact import BestIterTask, ModelArtifactTask, PanelContractTask
from .tasks.gate import RegimeLayeredICTask, WfGateMetadataTask
from .tasks.sector_map import SectorMapCoverageTask
from .tasks.watchlist import WatchlistSizeTask
from .tasks.correlation import CorrelationMetadataTask
from .tasks.calibrator import CalibratorFlatRegionTask, CalibratorHealthTask
from .tasks.feature_coverage import FeatureCoverageTask
from .tasks.run_id import ArtifactRunIdAlignmentTask
from .tasks.config_fingerprint import ConfigFingerprintTask
from .tasks.meta_label import MetaLabelArtifactContractTask
from .pipeline import build_minimal_preflight_pipeline, build_preflight_pipeline

__all__ = [
    "PreflightContext",
    "PreflightTask",
    "PreflightJob",
    "PreflightPipeline",
    "StateFileTask",
    "BrokerConnectTask",
    "ModelArtifactTask",
    "PanelContractTask",
    "BestIterTask",
    "WfGateMetadataTask",
    "RegimeLayeredICTask",
    "SectorMapCoverageTask",
    "WatchlistSizeTask",
    "CorrelationMetadataTask",
    "CalibratorHealthTask",
    "CalibratorFlatRegionTask",
    "FeatureCoverageTask",
    "ArtifactRunIdAlignmentTask",
    "ConfigFingerprintTask",
    "MetaLabelArtifactContractTask",
    "build_minimal_preflight_pipeline",
    "build_preflight_pipeline",
]
