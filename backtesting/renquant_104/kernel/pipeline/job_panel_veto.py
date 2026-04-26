"""PanelRankVetoJob — flag-gated veto of model_sell exits.

Runs at the start of Phase 3, immediately after PanelScoringJob
populates rank_score on holdings. Vetoes per-ticker `model_sell` exits
when the held's panel rank_score is above the configured threshold —
on the theoretical observation that a per-ticker XGBoost saying SELL
is misaligned when the panel-LTR says the held is the strongest in
the universe.

See task_panel_veto.py for full motivation + reference list.
"""
from __future__ import annotations

from .context import InferenceContext
from .pipeline import Job, Task
from .task_panel_veto import PanelRankVetoTask


class PanelRankVetoJob(Job):
    """Single-task wrapper. Default off; opt-in via config."""

    def should_skip(self, ctx: InferenceContext) -> bool:
        cfg = ((ctx.config.get("model_sell") or {})
                       .get("panel_veto") or {})
        if not cfg.get("enabled", False):
            return True
        if not ctx.exits:
            return True
        return False

    @property
    def tasks(self) -> list[Task]:
        return [PanelRankVetoTask()]
