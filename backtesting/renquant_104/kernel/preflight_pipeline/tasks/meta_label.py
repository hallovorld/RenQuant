"""P-META-LABEL — meta-label artifact contract for the path-rule exit veto.

Migrated from kernel.preflight._check_meta_label_artifact_contract. Decomposed
per §1c into validation helpers.
"""
from __future__ import annotations

import json

from kernel.preflight import (  # noqa: PLC0415 (legacy bridge)
    PreflightCheck,
    _finite_float,
    _resolve_artifact_path,
    _soft_for_sell_only,
)

from ..base import PreflightTask
from ..ctx import PreflightContext


class MetaLabelArtifactContractTask(PreflightTask):
    """P-META-LABEL — exit-veto artifact validation.

    Meta-label is an optional path-rule exit veto. When enabled for buy/full
    runs, a missing/corrupt artifact silently turns the decision tree back into
    the un-vetoed stop path. Sell-only runs are exempt so raw risk exits stay
    armed.

    Behavior parity with ``_check_meta_label_artifact_contract``.
    """

    check_name = "P-META-LABEL"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        cfg = ((ctx.config.get("ranking") or {}).get("meta_label") or {})
        if not bool(cfg.get("enabled", False)):
            return PreflightCheck(
                self.check_name, "soft", True,
                "ranking.meta_label disabled; artifact contract not applicable",
            )
        rel = cfg.get("artifact_path")
        if not rel:
            return _soft_for_sell_only(
                self.check_name,
                "ranking.meta_label.enabled=true but artifact_path is missing; "
                "full/buy cannot silently fall back to un-vetoed path exits",
                run_mode=ctx.run_mode,
            )
        p = _resolve_artifact_path(ctx.strategy_dir, rel)
        if not p.exists():
            return _soft_for_sell_only(
                self.check_name,
                f"ranking.meta_label.enabled=true but artifact missing at {p}; "
                "full/buy cannot silently fall back to un-vetoed path exits",
                run_mode=ctx.run_mode,
            )
        try:
            payload = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            return _soft_for_sell_only(
                self.check_name,
                f"meta-label artifact unreadable at {p}: {exc}",
                run_mode=ctx.run_mode,
            )
        return self._validate_contract(payload, cfg, p, ctx)

    def _validate_contract(self, payload: dict, cfg: dict, p,
                           ctx: PreflightContext) -> PreflightCheck:
        errors: list[str] = []
        # Type / shape contract
        if payload.get("kind") != "meta_label_exit_xgb":
            errors.append(f"kind={payload.get('kind')!r} != 'meta_label_exit_xgb'")
        feature_cols = payload.get("feature_cols") or []
        if not isinstance(feature_cols, list) or not feature_cols:
            errors.append("feature_cols missing/empty")
        if not isinstance(payload.get("booster_raw_json"), str) or not payload.get(
            "booster_raw_json"
        ):
            errors.append("booster_raw_json missing/empty")
        # Threshold validity
        default_threshold = _finite_float(payload.get("default_threshold"))
        cfg_threshold = _finite_float(cfg.get("threshold", default_threshold))
        if default_threshold is None:
            errors.append("default_threshold missing/non-finite")
        if cfg_threshold is None or not (0.0 <= cfg_threshold <= 1.0):
            errors.append(f"config threshold invalid: {cfg.get('threshold')!r}")
        # Model quality
        cv = payload.get("cv_metrics") or {}
        auc = _finite_float(cv.get("auc_mean"))
        min_auc = _finite_float(cfg.get("min_auc", 0.50))
        if auc is None:
            errors.append("cv_metrics.auc_mean missing/non-finite")
        elif min_auc is not None and auc < min_auc:
            errors.append(f"cv_metrics.auc_mean={auc:.4f} < min_auc={min_auc:.4f}")
        # Training data summary
        summary = payload.get("training_data_summary") or {}
        n_events_i, fwd_window_i = self._extract_training_counts(summary, cfg, errors)
        class_balance = _finite_float(summary.get("class_balance"))
        if class_balance is None or not (0.0 < class_balance < 1.0):
            errors.append("training_data_summary.class_balance missing/out-of-range")

        if errors:
            return _soft_for_sell_only(
                self.check_name,
                "meta-label artifact contract failed: " + "; ".join(errors),
                run_mode=ctx.run_mode,
                details={"artifact_path": str(p), "errors": errors},
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"meta-label artifact ok: n_events={n_events_i}, auc={auc:.4f}, "
            f"threshold={cfg_threshold:.2f}, fwd_window_days={fwd_window_i}",
            details={
                "artifact_path": str(p),
                "n_events": n_events_i,
                "auc_mean": auc,
                "threshold": cfg_threshold,
                "fwd_window_days": fwd_window_i,
                "feature_count": len(feature_cols),
            },
        )

    def _extract_training_counts(self, summary: dict, cfg: dict,
                                  errors: list) -> tuple[int, int]:
        try:
            n_events_i = int(summary.get("n_events"))
        except (TypeError, ValueError):
            n_events_i = 0
        min_events = int(cfg.get("min_events", 100))
        if n_events_i < min_events:
            errors.append(
                f"training_data_summary.n_events={n_events_i} < "
                f"min_events={min_events}"
            )
        try:
            fwd_window_i = int(summary.get("fwd_window_days"))
        except (TypeError, ValueError):
            fwd_window_i = 0
        if fwd_window_i <= 0:
            errors.append(
                "training_data_summary.fwd_window_days missing/non-positive"
            )
        return n_events_i, fwd_window_i
