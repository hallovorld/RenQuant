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
    fundamentals: dict[str, dict[str, float]] = field(default_factory=dict)   # {ticker: {factor: value}}
    # Earnings surprise history per ticker (sparse — one row per announcement);
    # TickerPanelFactorJob forward-fills to daily via compute_earnings_surprise_cum.
    earnings_surprises: dict[str, pd.DataFrame] = field(default_factory=dict)
    # SEC Form 4 executive-only insider trades per ticker (sparse — one row
    # per transaction). TickerPanelFactorJob produces daily trailing-90d net
    # buy via compute_insider_net_buy_cum.
    insider_trades: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Raw hourly OHLCV bars per ticker (Plan G). TickerPanelFactorJob
    # aggregates via `training_panel.hourly_features.compute_hourly_features`
    # into six daily factor columns. Empty dict → hourly panel features off.
    hourly_bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Raw 10-min OHLCV bars per ticker (2026-04-24 extension). Aggregated
    # via `training_panel.minute_features.compute_minute_features` into
    # ~10 daily factor columns. Empty dict → minute panel features off.
    minute_bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Bug 16 fix (2026-04-24): when set True (inference path), Load*Task
    # NEVER auto-fetches missing data — read from cache only. Training
    # path leaves this False so LoadFundamentals/EarningsSurprise/Insider
    # can fetch new tickers if needed. Inference must remain offline-fast.
    inference_only: bool = False

    # Macro factor frame (Phase 1B, 2026-04-26 round-7). Date-indexed
    # DataFrame with z-scored macro features (VIX, HYG, UUP, ...). When
    # populated, BuildPanelTask broadcasts these per-date values to every
    # row of the panel. None → no-macro mode (default; ships off-by-flag).
    # See doc/components/macro-factor-frame-design.md.
    macro_factor_frame: pd.DataFrame | None = None
    macro_metadata: dict = field(default_factory=dict)

    # ── Phase 2 collected from per-ticker contexts ────────────────────────
    feature_frames: dict[str, pd.DataFrame] = field(default_factory=dict)       # pre-neutralize
    neutralized_frames: dict[str, pd.DataFrame] = field(default_factory=dict)   # post-neutralize
    raw_factor_frames: dict[str, pd.DataFrame] = field(default_factory=dict)    # pre-zscore

    # ── Phase 3 outputs ───────────────────────────────────────────────────
    factor_frames: dict[str, pd.DataFrame] = field(default_factory=dict)        # post-zscore
    labels: dict[str, pd.Series] = field(default_factory=dict)
    raw_residuals: dict[str, pd.Series] = field(default_factory=dict)           # pre-Gaussianization (NGBoost label)
    panel: pd.DataFrame | None = None
    group_sizes: Any = None                     # np.ndarray[int32]
    panel_metadata: dict = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)
    feature_diagnostics: list[dict] = field(default_factory=list)  # per-feature std + IC
    cv_result: dict = field(default_factory=dict)
    final_model: Any = None                     # PanelLTRModel
    artifact_path: Path | None = None
    summary: dict = field(default_factory=dict)

    # ── Phase 5 (NGBoost head — optional) ────────────────────────────────
    ngboost_head: Any = None                    # NGBoostHead | None
    ngboost_artifact_path: Path | None = None
    ngboost_fit: dict = field(default_factory=dict)

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
    fundamentals: dict[str, dict[str, float]] = field(default_factory=dict)  # shared read-only
    earnings_surprises: dict = field(default_factory=dict)                     # shared read-only {ticker: DataFrame}
    insider_trades:     dict = field(default_factory=dict)                     # shared read-only {ticker: DataFrame}
    hourly_bars:        dict = field(default_factory=dict)                     # shared read-only {ticker: hourly OHLCV}
    minute_bars:        dict = field(default_factory=dict)                     # shared read-only {ticker: 10-min OHLCV}

    # Outputs — written by per-ticker jobs
    feature_frame: pd.DataFrame | None = None
    neutralized_frame: pd.DataFrame | None = None
    raw_factor_frame: pd.DataFrame | None = None
