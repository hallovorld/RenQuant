"""P-RUN-ID — panel-ltr and ngboost-head share the same train_run_id.

Migrated from kernel.preflight._check_artifact_run_id_alignment.
External audit fix #2 (2026-04-29): without run_id, one artifact can
silently come from a different training run.
"""
from __future__ import annotations

import json

from kernel.preflight import (  # noqa: PLC0415 (legacy bridge)
    PreflightCheck,
    _ngboost_activation,
    _resolve_artifact_path,
    _soft_for_sell_only,
)

from ..base import PreflightTask
from ..ctx import PreflightContext


class ArtifactRunIdAlignmentTask(PreflightTask):
    """P-RUN-ID — panel-ltr and ngboost head must share train_run_id.

    Behavior parity with ``_check_artifact_run_id_alignment``.
    """

    check_name = "P-RUN-ID"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        panel_cfg = ctx.config.get("panel_ltr", {})
        ltr_rel = panel_cfg.get(
            "artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json"
        )
        ngb_cfg, per_regime_activates, ngb_potentially_active = (
            _ngboost_activation(ctx.config)
        )
        if not ngb_potentially_active:
            return PreflightCheck(
                self.check_name, "soft", True,
                "NGBoost disabled globally + no per-regime overlay activates — skip",
            )
        ngb_rel = ngb_cfg.get("artifact_path")
        if not ngb_rel:
            return _soft_for_sell_only(
                self.check_name,
                "NGBoost can activate but ranking.panel_scoring.ngboost.artifact_path "
                f"is missing (per_regime={per_regime_activates}); cannot verify "
                "panel/NGBoost train_run_id alignment",
                run_mode=ctx.run_mode,
                details={"per_regime_activates": per_regime_activates},
            )
        return self._compare_run_ids(ltr_rel, ngb_rel, ctx)

    def _compare_run_ids(self, ltr_rel: str, ngb_rel: str,
                         ctx: PreflightContext) -> PreflightCheck:
        ltr_path = _resolve_artifact_path(ctx.strategy_dir, ltr_rel)
        ngb_path = _resolve_artifact_path(ctx.strategy_dir, ngb_rel)
        for p in (ltr_path, ngb_path):
            if not p.exists():
                return _soft_for_sell_only(
                    self.check_name,
                    f"artifact missing: {p}; cannot verify panel/NGBoost "
                    "train_run_id alignment",
                    run_mode=ctx.run_mode,
                    details={"panel_path": str(ltr_path),
                             "ngboost_path": str(ngb_path)},
                )
        try:
            ltr_id = json.loads(ltr_path.read_text()).get("train_run_id")
            ngb_id = json.loads(ngb_path.read_text()).get("train_run_id")
        except Exception as exc:  # noqa: BLE001
            return _soft_for_sell_only(
                self.check_name,
                f"unreadable: {exc}",
                run_mode=ctx.run_mode,
                details={"panel_path": str(ltr_path),
                         "ngboost_path": str(ngb_path)},
            )
        if ltr_id is None or ngb_id is None:
            return _soft_for_sell_only(
                self.check_name,
                "run_id not stamped on panel or NGBoost artifact; full/buy "
                "cannot mix unstamped μ/σ with panel scores",
                run_mode=ctx.run_mode,
                details={"panel_train_run_id": ltr_id,
                         "ngboost_train_run_id": ngb_id},
            )
        if ltr_id != ngb_id:
            return _soft_for_sell_only(
                self.check_name,
                f"run_id mismatch: panel-ltr={ltr_id} ngboost={ngb_id}. "
                "NGBoost μ/σ may be from a different training run — Kelly "
                "sizing potentially corrupted. Retrain recommended.",
                run_mode=ctx.run_mode,
                details={"panel_train_run_id": ltr_id,
                         "ngboost_train_run_id": ngb_id},
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"run_id aligned ({ltr_id})",
        )
