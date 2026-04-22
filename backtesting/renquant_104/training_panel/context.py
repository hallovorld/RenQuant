"""Context dataclasses for the panel-LTR training pipeline.

Mirrors the shape of kernel/pipeline/pp_training.py's TrainingContext +
TickerTrainingContext, but with panel-specific fields: sector momentum
frames, neutralized feature frames, raw factor bundles, the assembled
panel, CV results, and the final model.

Kept deliberately separate from TrainingContext so callers who only run
the per-ticker path (existing renquant_103) are unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class PanelTrainingContext:
    """Shared state for global (cross-ticker) panel-training jobs."""
    config: dict[str, Any]

    # ── Inputs (populated by DataFetchJob or pre-filled by caller) ────────
    watchlist: list[str] = field(default_factory=list)
    ohlcv: dict[str, pd.DataFrame] = field(default_factory=dict)
    sector_etf_ohlcv: dict[str, pd.DataFrame] = field(default_factory=dict)
    ticker_sectors: dict[str, str] = field(default_factory=dict)
    listing_dates: dict[str, pd.Timestamp] | None = None

    # ── Phase 1 outputs ───────────────────────────────────────────────────
    sector_momentum: dict[str, pd.DataFrame] = field(default_factory=dict)

    # ── Phase 2 collected from per-ticker contexts ────────────────────────
    feature_frames: dict[str, pd.DataFrame] = field(default_factory=dict)       # pre-neutralize
    neutralized_frames: dict[str, pd.DataFrame] = field(default_factory=dict)   # post-neutralize
    raw_factor_frames: dict[str, pd.DataFrame] = field(default_factory=dict)    # pre-zscore

    # ── Phase 3 outputs ───────────────────────────────────────────────────
    factor_frames: dict[str, pd.DataFrame] = field(default_factory=dict)        # post-zscore
    labels: dict[str, pd.Series] = field(default_factory=dict)
    panel: pd.DataFrame | None = None
    group_sizes: Any = None                     # np.ndarray[int32]
    panel_metadata: dict = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)
    cv_result: dict = field(default_factory=dict)
    final_model: Any = None                     # PanelLTRModel
    artifact_path: Path | None = None
    summary: dict = field(default_factory=dict)

    @property
    def spy_df(self) -> pd.DataFrame | None:
        return self.ohlcv.get(self.config.get("benchmark", "SPY"))

    @property
    def strategy_dir(self) -> Path | None:
        _sd = self.config.get("_strategy_dir")
        return Path(_sd) if _sd else None


@dataclass
class TickerPanelContext:
    """Per-ticker context for the parallel Phase-2 chain.

    One instance per ticker. Jobs write only to this object; results are
    collected back into PanelTrainingContext after all workers complete.
    """
    ticker: str
    ohlcv: dict[str, pd.DataFrame]                  # shared read-only
    sector_momentum: dict[str, pd.DataFrame]        # shared read-only
    ticker_sectors: dict[str, str]                  # shared read-only
    config: dict[str, Any]

    # Outputs — written by per-ticker jobs
    feature_frame: pd.DataFrame | None = None
    neutralized_frame: pd.DataFrame | None = None
    raw_factor_frame: pd.DataFrame | None = None
