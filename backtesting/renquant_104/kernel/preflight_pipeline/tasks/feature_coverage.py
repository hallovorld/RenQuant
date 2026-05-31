"""P-FEATURE-COVER — NGBoost head's feature_cols are a subset of panel's.

Migrated from kernel.preflight._check_feature_coverage.
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


class FeatureCoverageTask(PreflightTask):
    """P-FEATURE-COVER — static check on artifact metadata that the NGBoost
    head and panel-LTR scorer agree on feature_cols. The actual runtime
    drift detector in ApplyNGBoostTask catches the dynamic case.

    Behavior parity with ``_check_feature_coverage``. Threshold:
    ``ranking.panel_scoring.ngboost.max_feature_drift_pct`` (default 0.05).
    """

    check_name = "P-FEATURE-COVER"

    # Default threshold preserved from legacy positional parameter.
    DEFAULT_FEATURE_DRIFT_PCT = 0.05

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        panel_cfg = ctx.config.get("panel_ltr", {})
        panel_rel = panel_cfg.get(
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
                f"is missing (per_regime={per_regime_activates}); full/buy cannot "
                "silently fall back to panel-only scoring",
                run_mode=ctx.run_mode,
                details={"per_regime_activates": per_regime_activates},
            )
        return self._compare_feature_sets(panel_rel, ngb_rel, ngb_cfg, ctx)

    def _compare_feature_sets(self, panel_rel: str, ngb_rel: str,
                              ngb_cfg: dict,
                              ctx: PreflightContext) -> PreflightCheck:
        panel_p = _resolve_artifact_path(ctx.strategy_dir, panel_rel)
        ngb_p = _resolve_artifact_path(ctx.strategy_dir, ngb_rel)
        if not panel_p.exists() or not ngb_p.exists():
            return _soft_for_sell_only(
                self.check_name,
                f"artifact missing: panel={panel_p.exists()} "
                f"ngb={ngb_p.exists()}",
                run_mode=ctx.run_mode,
                details={"panel_path": str(panel_p), "ngboost_path": str(ngb_p)},
            )
        try:
            panel_meta = json.loads(panel_p.read_text())
            ngb_meta = json.loads(ngb_p.read_text())
        except Exception as exc:  # noqa: BLE001
            return _soft_for_sell_only(
                self.check_name,
                f"unreadable: {exc}",
                run_mode=ctx.run_mode,
                details={"panel_path": str(panel_p), "ngboost_path": str(ngb_p)},
            )
        panel_feats = set(panel_meta.get("feature_cols") or [])
        ngb_feats = set(ngb_meta.get("feature_cols") or [])
        if not ngb_feats:
            return _soft_for_sell_only(
                self.check_name,
                "NGBoost feature_cols not stamped; full/buy cannot validate "
                "runtime feature parity",
                run_mode=ctx.run_mode,
                details={"ngboost_path": str(ngb_p)},
            )
        missing = ngb_feats - panel_feats
        pct = len(missing) / max(1, len(ngb_feats))
        feature_drift_pct = float(
            ngb_cfg.get("max_feature_drift_pct", self.DEFAULT_FEATURE_DRIFT_PCT)
        )
        allow_partial = bool(ngb_cfg.get("allow_partial_feature_fill", False))
        if missing and (not allow_partial or pct > feature_drift_pct):
            policy = (
                "partial fill disabled"
                if not allow_partial else
                f"missing_pct={pct:.1%} > max_feature_drift_pct="
                f"{feature_drift_pct:.1%}"
            )
            return _soft_for_sell_only(
                self.check_name,
                f"NGBoost expects {len(ngb_feats)} feats, "
                f"{len(missing)} ({pct:.1%}) missing from panel — "
                f"{policy}; retrain NGBoost head against current panel "
                f"pipeline. First 5 missing: {sorted(missing)[:5]}",
                run_mode=ctx.run_mode,
                details={"missing_count": len(missing),
                         "missing_pct": pct,
                         "first_missing": sorted(missing)[:10],
                         "allow_partial_feature_fill": allow_partial},
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"NGBoost feature coverage OK ({len(ngb_feats)} feats, "
            f"{len(missing)} missing = {pct:.1%})",
        )
