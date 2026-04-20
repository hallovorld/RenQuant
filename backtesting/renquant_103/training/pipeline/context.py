"""TrainingContext — shared state passed between all training pipeline jobs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingContext:
    # ── Required inputs ────────────────────────────────────────────────────────
    config: dict
    strategy_dir: Path
    today: str   # ISO date string "YYYY-MM-DD"

    # ── Populated by DataFetchJob ──────────────────────────────────────────────
    # {ticker: OHLCV DataFrame}; includes SPY + sector ETFs + watchlist
    ohlcv: dict[str, Any] = field(default_factory=dict)

    # ── Populated by RegimeFitJob ──────────────────────────────────────────────
    # spy-gmm-regime.json artifact path (saved by job)
    gmm_artifact_path: Path | None = None

    # ── Populated by FeatureJob ────────────────────────────────────────────────
    # {ticker: labelled feature DataFrame}
    feature_frames: dict[str, Any] = field(default_factory=dict)

    # ── Populated by TournamentJob ─────────────────────────────────────────────
    # {ticker: result dict from run_tournament()}
    tournament_results: dict[str, dict] = field(default_factory=dict)

    # ── Populated by ExportJob ─────────────────────────────────────────────────
    exported: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    # ── Populated by CorrelationJob ────────────────────────────────────────────
    corr_dict: dict = field(default_factory=dict)

    # ── Populated by CalibrationJob ────────────────────────────────────────────
    # Updated blend_weights written back to strategy_config.json by the job
    blend_weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
