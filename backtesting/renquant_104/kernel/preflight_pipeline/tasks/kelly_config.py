"""Kelly sizing config sanity checks."""
from __future__ import annotations

from kernel.preflight import (  # noqa: PLC0415 (legacy bridge)
    PreflightCheck,
    _check_kelly_sigma_horizon_config,
)

from ..base import PreflightTask
from ..ctx import PreflightContext


class KellySigmaHorizonTask(PreflightTask):
    """Validate the optional Kelly σ horizon override before live runs."""

    check_name = "P-KELLY-SIGMA-HORIZON"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        return _check_kelly_sigma_horizon_config(
            ctx.config, ctx.strategy_dir, ctx.run_mode,
        )
