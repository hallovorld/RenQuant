"""Panel-LTR inference kernel package — parallel to kernel/pipeline/.

Swaps the per-ticker `ScoreBuyTask` + `RankingJob` pair for a single
cross-sectional `ComputePanelScoresTask` + `PanelRankGateTask` pair
driven by a trained `PanelLTRModel` artifact.

This package is `common/`-free so it can run under LEAN and the live
runner. The training side lives in `training_panel/` (notebook / cron).

Entry points::

    from kernel.panel_pipeline import (
        PanelScorer,               # thin loader around the saved artifact
        compute_panel_scores,      # pure function: (artifact, feature_matrix) → scores
        top_n_by_score,            # rank gate: keep top-N
        probability_gate,          # rank gate: keep score ≥ threshold
    )
"""
from __future__ import annotations

from .panel_scorer import (
    PanelScorer,
    compute_panel_scores,
    top_n_by_score,
    probability_gate,
)
from .feature_matrix import (
    build_inference_matrix,
    run_panel_inference,
)
from .job_panel_scoring import (
    PanelScoringJob,
    LoadScorerTask,
    BuildFeatureMatrixTask,
    ApplyScoresTask,
    VetoWeakBuysTask,
    LoadGlobalCalibrationTask,
    ApplyGlobalCalibrationTask,
    LoadNGBoostTask,
    ApplyNGBoostTask,
)

__all__ = [
    "PanelScorer",
    "compute_panel_scores",
    "top_n_by_score",
    "probability_gate",
    "build_inference_matrix",
    "run_panel_inference",
    "PanelScoringJob",
    "LoadScorerTask",
    "BuildFeatureMatrixTask",
    "ApplyScoresTask",
    "VetoWeakBuysTask",
    "LoadGlobalCalibrationTask",
    "ApplyGlobalCalibrationTask",
    "LoadNGBoostTask",
    "ApplyNGBoostTask",
]
