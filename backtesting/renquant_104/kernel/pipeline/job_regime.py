"""RegimeJob — 3-layer regime detection."""
from __future__ import annotations

from .pipeline import Job, Task
from .task_regime import HurstTask, CUSUMTask, GMMTask, BEAROverrideTask, RegimeFinalizeTask


class RegimeJob(Job):
    """Task chain: Hurst → CUSUM → GMM → BEAROverride → Finalize"""

    @property
    def tasks(self) -> list[Task]:
        return [HurstTask(), CUSUMTask(), GMMTask(), BEAROverrideTask(), RegimeFinalizeTask()]
