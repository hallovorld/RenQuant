"""Preflight T/J/P refactor — Track H complete pipeline.

Drop-in architecture for ``kernel.preflight.run_preflight`` migration:

  - PreflightContext: dataclass shared across all Tasks (read-mostly)
  - PreflightTask: subclass of the canonical kernel.pipeline.Task ABC, with
    a ``check_name``/``severity`` contract that maps to PreflightCheck
  - PreflightJob: groups related PreflightTasks; runs sequentially
  - PreflightPipeline: orchestrates Jobs in declaration order; ``run`` returns
    list[PreflightCheck] identical in shape to the legacy ``run_preflight``

All 16 checks are represented as Tasks. ``run_preflight`` is wired as a thin
wrapper in the follow-up PR so production callers keep the legacy API while
the business logic moves behind Task/Job/Pipeline boundaries.
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
