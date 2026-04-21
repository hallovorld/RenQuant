"""RegimeJob — 3-layer regime detection (Hurst → CUSUM → GMM → BEAR-override → finalize).

Task chain:
    HurstTask         Layer 1: rolling Hurst → state.hurst, state.hurst_regime
    CUSUMTask         Layer 2: changepoint detection → state.countdown, state.in_transition
    GMMTask           Layer 3: GMM posterior → state.gmm_probs
    BEAROverrideTask  Hard vol/return override → state.hard_bear
    RegimeFinalizeTask Resolve final regime + confidence → ctx.regime, ctx.confidence
"""
from __future__ import annotations

from ..pipeline import Job, Task
from ..tasks.regime import (
    HurstTask, CUSUMTask, GMMTask, BEAROverrideTask, RegimeFinalizeTask,
)


class RegimeJob(Job):
    """Detect market regime via Hurst + CUSUM + GMM and update ctx.

    Task chain: Hurst → CUSUM → GMM → BEAROverride → Finalize
    """

    @property
    def tasks(self) -> list[Task]:
        return [
            HurstTask(),
            CUSUMTask(),
            GMMTask(),
            BEAROverrideTask(),
            RegimeFinalizeTask(),
        ]
