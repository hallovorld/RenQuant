"""Artifact-group Tasks — model_artifact + panel_contract + best_iter.

Migrated from kernel.preflight._check_model_artifact,
_check_panel_artifact_contract, _check_best_iter. Behavior parity asserted by
tests/test_preflight_pipeline.py.
"""
from __future__ import annotations

import json

from kernel.preflight import (  # noqa: PLC0415 (legacy bridge — helpers stay
                                # in preflight.py until later PR retires them)
    PreflightCheck,
    _active_panel_config,
    _active_panel_kind,
    _check_sequence_artifact_contract,
    _is_sell_only_run,
    _is_sequence_artifact,
    _resolve_artifact_path,
)

from ..base import PreflightTask
from ..ctx import PreflightContext


class ModelArtifactTask(PreflightTask):
    """P-MODEL-ARTIFACT — active scorer artifact exists + parses.

    Behavior parity with ``_check_model_artifact``:
      - artifact missing → HARD fail
      - sequence artifact + nonempty file → HARD pass
      - sequence artifact + empty file → HARD fail
      - JSON artifact + unparseable → HARD fail
      - JSON artifact + parses → HARD pass, details carry best_iter + oos_mean_ic
    """

    check_name = "P-MODEL-ARTIFACT"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        panel_cfg = _active_panel_config(ctx.config)
        kind = _active_panel_kind(ctx.config, panel_cfg)
        rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
        p = _resolve_artifact_path(ctx.strategy_dir, rel)
        if not p.exists():
            return PreflightCheck(
                self.check_name, "hard", False,
                f"artifact missing: {p}",
            )
        if _is_sequence_artifact(kind, p):
            if p.stat().st_size <= 0:
                return PreflightCheck(
                    self.check_name, "hard", False,
                    f"{kind} checkpoint is empty: {p}",
                )
            return PreflightCheck(
                self.check_name, "hard", True,
                f"loaded {kind} checkpoint {p.name}",
                details={"path": str(p), "kind": kind, "bytes": p.stat().st_size},
            )
        try:
            meta = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "hard", False,
                f"artifact unreadable {p.name}: {exc}",
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"loaded {p.name}",
            details={
                "path": str(p),
                "best_iter": meta.get("best_iter"),
                "oos_mean_ic": meta.get("oos_mean_ic"),
            },
        )


class PanelContractTask(PreflightTask):
    """P-PANEL-CONTRACT — panel artifact carries evidence metadata.

    Behavior parity with ``_check_panel_artifact_contract``. Full/buy strict,
    sell-only soft so risk exits aren't blocked by missing buy-side evidence.
    """

    check_name = "P-PANEL-CONTRACT"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        panel_cfg = _active_panel_config(ctx.config)
        kind = _active_panel_kind(ctx.config, panel_cfg)
        rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
        p = _resolve_artifact_path(ctx.strategy_dir, rel)
        if not p.exists():
            return PreflightCheck(
                self.check_name, "hard", False, f"artifact missing: {p}",
            )
        strict_contract = bool(
            ctx.config.get("preflight", {})
            .get("artifact_contract", {})
            .get("strict", not _is_sell_only_run(ctx.run_mode))
        )
        if _is_sequence_artifact(kind, p):
            return _check_sequence_artifact_contract(
                kind=kind,
                artifact_path=p,
                strict_contract=strict_contract,
            )
        try:
            payload = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "hard", False, f"unreadable: {exc}",
            )
        from kernel.artifact_contract import validate_panel_artifact_contract  # noqa: PLC0415
        result = validate_panel_artifact_contract(
            payload,
            strict=strict_contract,
            runtime_config=ctx.config,
        )
        severity = "hard" if strict_contract else "soft"
        if not result.ok:
            if _is_sell_only_run(ctx.run_mode):
                return PreflightCheck(
                    self.check_name, "soft", True,
                    "panel artifact contract failed: " + "; ".join(result.errors)
                    + "; sell-only risk exits are allowed, new buys remain blocked",
                    details=result.details | {"warnings": result.warnings},
                )
            return PreflightCheck(
                self.check_name, severity, False,
                "; ".join(result.errors),
                details=result.details | {"warnings": result.warnings},
            )
        msg = "contract ok"
        if result.warnings:
            msg = "contract legacy-compatible; " + "; ".join(result.warnings[:3])
        return PreflightCheck(
            self.check_name, severity, True, msg, details=result.details,
        )


class BestIterTask(PreflightTask):
    """P-BEST-ITER — model's ``best_iter`` ≥ ``min_best_iter`` (BUG-CV-2 class).

    Behavior parity with ``_check_best_iter``. Includes the strong-univariate-IC
    plateau escape clause (2026-05-04 P0): if best_iter is low but eval_ic is
    healthy, the model is accepted (mirrors FinalFitTask's training-time guard).
    """

    check_name = "P-BEST-ITER"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        panel_cfg = _active_panel_config(ctx.config)
        kind = _active_panel_kind(ctx.config, panel_cfg)
        rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
        p = _resolve_artifact_path(ctx.strategy_dir, rel)
        if not p.exists():
            return PreflightCheck(
                self.check_name, "hard", False, f"artifact missing: {p}",
            )
        if _is_sequence_artifact(kind, p):
            return PreflightCheck(
                self.check_name, "soft", True,
                f"best_iter not applicable to sequence artifact ({kind})",
            )
        try:
            meta = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "hard", False, f"unreadable: {exc}",
            )
        bi = meta.get("best_iter")
        if bi is None:
            return PreflightCheck(
                self.check_name, "soft", True,
                "best_iter not stamped in artifact (legacy pre-2026-04-28); skip",
            )
        min_bi = int(panel_cfg.get("min_best_iter", 5))
        if int(bi) < min_bi:
            return self._evaluate_low_best_iter(meta, panel_cfg, bi, min_bi)
        return PreflightCheck(
            self.check_name, "hard", True,
            f"best_iter={bi} ≥ {min_bi}",
            details={"best_iter": bi},
        )

    def _evaluate_low_best_iter(
        self, meta: dict, panel_cfg: dict, bi, min_bi: int,
    ) -> PreflightCheck:
        """Plateau escape clause — accept if eval_ic ≥ floor despite low bi."""
        import math as _math  # noqa: PLC0415

        eval_ic_floor = float(panel_cfg.get("min_best_iter_eval_ic_floor", 0.02))
        eval_ic = meta.get("eval_ic")
        try:
            eval_ic_f = float(eval_ic) if eval_ic is not None else None
        except (TypeError, ValueError):
            eval_ic_f = None
        if (eval_ic_f is not None and _math.isfinite(eval_ic_f)
                and eval_ic_f >= eval_ic_floor):
            return PreflightCheck(
                self.check_name, "hard", True,
                f"best_iter={bi} < {min_bi} but eval_ic={eval_ic_f:+.4f} ≥ "
                f"floor={eval_ic_floor:+.4f} — strong-univariate-IC plateau, accepting",
                details={"best_iter": bi, "min_best_iter": min_bi,
                         "eval_ic": eval_ic_f, "eval_ic_floor": eval_ic_floor},
            )
        return PreflightCheck(
            self.check_name, "hard", False,
            f"best_iter={bi} < min_best_iter={min_bi} AND "
            f"eval_ic={eval_ic} < floor={eval_ic_floor:+.4f}. "
            f"Model undertrained (early-stop fired in round {bi}). "
            f"Retrain required, OR confirm eval_ic is stamped in the artifact "
            f"(SaveArtifactTask must include 'eval_ic' in meta).",
            details={"best_iter": bi, "min_best_iter": min_bi,
                     "eval_ic": eval_ic, "eval_ic_floor": eval_ic_floor},
        )
