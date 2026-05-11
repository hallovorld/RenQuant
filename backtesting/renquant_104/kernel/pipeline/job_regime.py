"""RegimeJob — 3-layer regime detection + optional 12M trend overlay."""
from __future__ import annotations

from .pipeline import Job, Task
from .task_regime import HurstTask, CUSUMTask, GMMTask, BEAROverrideTask, RegimeFinalizeTask
from .task_trend_overlay import TrendOverlayTask


class RegimeJob(Job):
    """Task chain: Hurst → CUSUM → GMM → BEAROverride → TrendOverlay → Finalize

    The 2026-05-11 R-03 addition (TrendOverlayTask) is a Hurst-Ooi-Pedersen
    2017 12-month SPY trend filter wired *between* BEAROverride and
    Finalize. It only escalates ``state.hard_bear`` (never demotes it),
    so the canonical BEAR-resolution branch in RegimeFinalizeTask picks
    up the override with no extra coupling. Disabled by default —
    enabled via ``regime.trend_overlay.enabled = true``.
    """

    @property
    def tasks(self) -> list[Task]:
        return [
            HurstTask(),
            CUSUMTask(),
            GMMTask(),
            BEAROverrideTask(),
            TrendOverlayTask(),
            RegimeFinalizeTask(),
        ]
