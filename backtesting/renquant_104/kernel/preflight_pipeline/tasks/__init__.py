"""PreflightTask implementations (Track H migration target).

Each Task corresponds to one of the legacy ``_check_*`` functions in
``kernel.preflight``. As checks lift over (one per follow-up PR), import them
here to surface a clean public symbol set.
"""
from .state import StateFileTask
from .broker import BrokerConnectTask
from .broker_fill_freshness import BrokerFillFreshnessTask
from .artifact import BestIterTask, ModelArtifactTask, PanelContractTask
from .gate import RegimeLayeredICTask, WfGateMetadataTask
from .sector_map import SectorMapCoverageTask
from .watchlist import WatchlistSizeTask
from .correlation import CorrelationMetadataTask
from .calibrator import CalibratorFlatRegionTask, CalibratorHealthTask
from .feature_coverage import FeatureCoverageTask
from .run_id import ArtifactRunIdAlignmentTask
from .config_fingerprint import ConfigFingerprintTask
from .meta_label import MetaLabelArtifactContractTask

__all__ = [
    "StateFileTask",
    "BrokerConnectTask",
    "BrokerFillFreshnessTask",
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
]
