"""Calibrator-group Tasks — health + flat-region.

Migrated from kernel.preflight._check_calibrator_health +
_check_calibrator_flat_region. Both checks gate the global-calibration
artifact against the bug classes that produced incidents in 2026-05:

  • n_unique_prob_y collapse (2026-05-04)  → CalibratorHealthTask
  • expected_return.y > 0.20 ER_BOUND      → CalibratorHealthTask
  • probability.y flat region > 30%         → CalibratorFlatRegionTask
  • Kelly/QP consumes wrong contract        → CalibratorHealthTask
"""
from __future__ import annotations

import json
import logging

from kernel.preflight import (  # noqa: PLC0415 (legacy bridge)
    PreflightCheck,
    _is_global_calibration_enabled,
    _resolve_artifact_path,
    _soft_for_sell_only,
)

from ..base import PreflightTask
from ..ctx import PreflightContext

log = logging.getLogger("preflight_pipeline.calibrator")


def _calibrator_artifact_path(ctx: PreflightContext):
    """Resolve the calibrator artifact path — used by both Tasks."""
    panel_cfg = ctx.config.get("panel_ltr", {})
    cal_cfg = ((ctx.config.get("ranking", {})
                          .get("panel_scoring", {})
                          .get("global_calibration", {})) or {})
    rel = (
        cal_cfg.get("artifact_path")
        or panel_cfg.get("calibrator_artifact_path")
        or "artifacts/prod/panel-rank-calibration.json"
    )
    return _resolve_artifact_path(ctx.strategy_dir, rel)


class CalibratorHealthTask(PreflightTask):
    """P-CALIBRATOR-HEALTH — runtime equivalent of the training-side
    probability-head-collapse guard. Closes the gap where an artifact was
    saved BEFORE the training-time guard was added (2026-05-04 production
    incident: n_unique_prob_y=7).
    """

    check_name = "P-CALIBRATOR-HEALTH"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        if not _is_global_calibration_enabled(ctx.config):
            return PreflightCheck(
                self.check_name, "soft", True,
                "global_calibration disabled; health gate not applicable",
            )
        # Note: parity with legacy — legacy uses ``strategy_dir / rel`` directly
        # (without _resolve_artifact_path) for the existence check, so we mirror.
        panel_cfg = ctx.config.get("panel_ltr", {})
        cal_cfg = ((ctx.config.get("ranking", {})
                              .get("panel_scoring", {})
                              .get("global_calibration", {})) or {})
        rel = cal_cfg.get("artifact_path", "artifacts/prod/panel-rank-calibration.json")
        p = ctx.strategy_dir / rel
        if not p.exists():
            return _soft_for_sell_only(
                self.check_name,
                f"global_calibration.enabled=true but calibrator artifact "
                f"absent at {p}",
                run_mode=ctx.run_mode,
            )
        try:
            payload = json.loads(p.read_text())
            meta = payload.get("metadata", {}) or {}
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "hard", False, f"unreadable: {exc}",
            )
        return self._evaluate(payload, meta, panel_cfg, ctx)

    def _evaluate(self, payload: dict, meta: dict, panel_cfg: dict,
                  ctx: PreflightContext) -> PreflightCheck:
        n_unique = meta.get("n_unique_prob_y")
        pool_ic = meta.get("pool_ic")
        # μ-mu contract gate (Kelly sizing consumer)
        gate = self._check_kelly_mu_contract(meta, ctx)
        if gate is not None:
            return gate
        # er.y range/flat checks (P0 2026-05-15)
        gate = self._check_er_y_range_and_flatness(payload, ctx)
        if gate is not None:
            return gate
        # n_unique + pool_ic thresholds
        return self._check_thresholds(n_unique, pool_ic, panel_cfg, ctx)

    def _check_kelly_mu_contract(self, meta: dict,
                                 ctx: PreflightContext) -> PreflightCheck | None:
        kelly_cfg = (ctx.config.get("ranking", {}) or {}).get("kelly_sizing", {}) or {}
        if not bool(kelly_cfg.get("use_calibrator_mu", False)):
            return None
        er_contract = meta.get("expected_return_label_contract")
        if er_contract == "raw_return_units_required":
            return None
        return _soft_for_sell_only(
            self.check_name,
            "ranking.kelly_sizing.use_calibrator_mu=true but calibrator "
            f"expected_return_label_contract={er_contract!r}; QP/Kelly "
            "would consume a non-return label as expected-return μ. "
            "Refit calibrator with raw return labels before buy/full.",
            run_mode=ctx.run_mode,
            details={
                "expected_return_label_contract": er_contract,
                "required_contract": "raw_return_units_required",
                "use_calibrator_mu": True,
                "er_std": meta.get("er_std"),
            },
        )

    def _check_er_y_range_and_flatness(self, payload: dict,
                                        ctx: PreflightContext) -> PreflightCheck | None:
        try:
            er_y = payload.get("expected_return", {}).get("y", []) or []
            er_x = payload.get("expected_return", {}).get("x", []) or []
            if not er_y:
                return None
            er_max_abs = max(abs(float(v)) for v in er_y
                             if v is not None and v == v)  # NaN-safe
            ER_BOUND = 0.20  # matches GlobalPanelCalibration.load() clip
            if er_max_abs > ER_BOUND + 1e-9:
                return PreflightCheck(
                    self.check_name, "hard", False,
                    f"calibrator expected_return.y has max|y|={er_max_abs:.4f} > "
                    f"{ER_BOUND} sanity bound. CLAUDE.md §5.13.12 violation: "
                    f"artifact was not clipped at train site. Kelly sizing on "
                    f"this calibrator would produce broken position weights. "
                    f"Refit via scripts/fit_calibrator_alpha158_fund.py before "
                    f"live trade.",
                    details={"max_abs_er_y": er_max_abs,
                             "bound": ER_BOUND, "n_knots": len(er_y)},
                )
            from kernel.calibrator_quality import flat_region_stats  # noqa: PLC0415
            er_flat = flat_region_stats(er_x, er_y)
            max_er_flat = float(
                ctx.config.get("panel_ltr", {})
                .get("calibrator_health", {})
                .get("max_expected_return_flat_fraction", 0.30)
            )
            if er_flat["fraction"] > max_er_flat:
                return PreflightCheck(
                    self.check_name, "hard", False,
                    f"calibrator expected_return.y has flat region spanning "
                    f"{er_flat['fraction']*100:.1f}% of x-domain "
                    f"(>{max_er_flat*100:.0f}%). Kelly/QP consumes this curve "
                    f"as μ, so a plateau ties candidate target weights even "
                    f"when probability.y is healthy. Refit calibrator with "
                    f"smooth bounded ER head.",
                    details={"expected_return_flat_fraction": er_flat["fraction"],
                             "longest_flat_span": er_flat["longest_span"],
                             "x_total": er_flat["x_total"],
                             "max_expected_return_flat_fraction": max_er_flat},
                )
        except (TypeError, ValueError) as exc:
            log.warning("P-CALIBRATOR-HEALTH: could not check er.y bounds: %s", exc)
        return None

    def _check_thresholds(self, n_unique, pool_ic, panel_cfg: dict,
                          ctx: PreflightContext) -> PreflightCheck:
        health_cfg = panel_cfg.get("calibrator_health", {}) or {}
        min_unique = int(health_cfg.get("min_unique_prob_y", 10))
        calibration_enabled = _is_global_calibration_enabled(ctx.config)
        if n_unique is None:
            return _soft_for_sell_only(
                self.check_name,
                "n_unique_prob_y not stamped; cannot verify probability-head "
                "granularity",
                run_mode=ctx.run_mode,
                details={"pool_ic": pool_ic,
                         "global_calibration_enabled": calibration_enabled},
            )
        if int(n_unique) < min_unique:
            return _soft_for_sell_only(
                self.check_name,
                f"n_unique_prob_y={n_unique} < min_unique_prob_y={min_unique}; "
                "calibrator probability head collapsed and buy ranking is "
                "ineffective",
                run_mode=ctx.run_mode,
                details={"n_unique_prob_y": n_unique,
                         "min_unique_prob_y": min_unique, "pool_ic": pool_ic},
            )
        if pool_ic is not None and float(pool_ic) <= 0:
            return _soft_for_sell_only(
                self.check_name,
                f"pool_ic={pool_ic} <= 0; calibrator anti-correlated with labels",
                run_mode=ctx.run_mode,
                details={"n_unique_prob_y": n_unique, "pool_ic": pool_ic},
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"n_unique_prob_y={n_unique} ≥ {min_unique}, pool_ic={pool_ic}",
            details={"n_unique_prob_y": n_unique, "pool_ic": pool_ic},
        )


class CalibratorFlatRegionTask(PreflightTask):
    """P-CALIBRATOR-FLAT-REGION — structural check that probability curve has
    no flat region wider than ``max_flat_fraction`` of x-domain.

    Reference: ``doc/research/2026-05-18-mcd-rebuy-incident.md`` —
    isotonic regression can create wide flat regions where the underlying
    signal is weak; those flat regions tie up to 79% of candidates at one
    probability and produce MCD-style rebuy.
    """

    check_name = "P-CALIBRATOR-FLAT-REGION"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        panel_cfg = ctx.config.get("panel_ltr", {})
        p = _calibrator_artifact_path(ctx)
        if not _is_global_calibration_enabled(ctx.config):
            return PreflightCheck(
                self.check_name, "soft", True,
                "global_calibration disabled; flat-region gate not applicable",
            )
        if not p.exists():
            return _soft_for_sell_only(
                self.check_name,
                f"global_calibration.enabled=true but calibrator artifact "
                f"missing at {p}",
                run_mode=ctx.run_mode,
            )
        try:
            cal = json.loads(p.read_text())
            pr = cal.get("probability", {})
            x = pr.get("x", [])
            y = pr.get("y", [])
            if not x or not y or len(x) != len(y):
                return _soft_for_sell_only(
                    self.check_name,
                    "probability.x/y missing or mismatched",
                    run_mode=ctx.run_mode,
                )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return _soft_for_sell_only(
                self.check_name,
                f"could not parse calibrator: {exc}",
                run_mode=ctx.run_mode,
            )
        return self._evaluate_flat_region(cal, x, y, panel_cfg, ctx)

    def _evaluate_flat_region(self, cal: dict, x: list, y: list,
                              panel_cfg: dict,
                              ctx: PreflightContext) -> PreflightCheck:
        health_cfg = panel_cfg.get("calibrator_health", {}) or {}
        max_flat_fraction = float(health_cfg.get("max_flat_fraction", 0.30))
        # DRY: shared with the training fit script + test impls
        from kernel.calibrator_quality import flat_region_stats  # noqa: PLC0415
        stats = flat_region_stats(x, y)
        flat_frac = stats["fraction"]
        if flat_frac > max_flat_fraction:
            return PreflightCheck(
                self.check_name, "hard", False,
                f"calibrator has flat region spanning {flat_frac*100:.1f}% of "
                f"x-domain (>{max_flat_fraction*100:.0f}%). All μ̂ in that "
                "region map to one probability → ranking degenerates → "
                "tie-broken buys (MCD-rebuy class). Refit with method=platt "
                "or shrink flat region. See "
                "doc/research/2026-05-18-mcd-rebuy-incident.md.",
                details={"longest_flat_span": stats["longest_span"],
                         "x_total": stats["x_total"],
                         "flat_fraction": flat_frac,
                         "max_flat_fraction": max_flat_fraction,
                         "calibrator_kind": cal.get("kind", "unknown")},
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"largest flat region {flat_frac*100:.1f}% ≤ "
            f"{max_flat_fraction*100:.0f}% of x-domain (n_knots={len(x)})",
            details={"flat_fraction": flat_frac,
                     "max_flat_fraction": max_flat_fraction},
        )
