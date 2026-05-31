"""P-CORR-METADATA — buy/full runs require stamped correlation metadata.

Migrated from kernel.preflight._check_correlation_artifact_metadata.
"""
from __future__ import annotations

import json

from kernel.preflight import (  # noqa: PLC0415 (legacy bridge)
    PreflightCheck,
    _correlation_artifact_path,
    _soft_for_sell_only,
)

from ..base import PreflightTask
from ..ctx import PreflightContext


class CorrelationMetadataTask(PreflightTask):
    """P-CORR-METADATA — buy/full runs require stamped correlation metadata.

    Behavior parity with ``_check_correlation_artifact_metadata``. Live may
    use the freshest correlation matrix, so this preflight does NOT compare
    ``as_of_date`` with ``backtest_start`` — it ensures the artifact can prove
    its data window when strict sims or LEAN acceptance consume it.
    """

    check_name = "P-CORR-METADATA"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        p = _correlation_artifact_path(ctx.config, ctx.strategy_dir)
        details = {"path": str(p)}
        if not p.exists():
            return _soft_for_sell_only(
                self.check_name,
                f"correlation artifact missing at {p}",
                run_mode=ctx.run_mode,
                details=details,
            )
        try:
            raw = json.loads(p.read_text())
            from kernel.walk_forward import parse_correlation_artifact  # noqa: PLC0415
            matrix, as_of = parse_correlation_artifact(raw)
        except Exception as exc:  # noqa: BLE001
            return _soft_for_sell_only(
                self.check_name,
                f"correlation artifact unreadable at {p}: {exc}",
                run_mode=ctx.run_mode,
                details=details,
            )
        details.update({"as_of_date": as_of, "n_tickers": len(matrix)})
        return self._validate_as_of(ctx, p, as_of, details)

    def _validate_as_of(self, ctx: PreflightContext, p,
                        as_of, details: dict) -> PreflightCheck:
        if as_of is None:
            legacy_allowed = bool(
                (ctx.config.get("regime", {}) or {})
                .get("allow_legacy_correlation_without_as_of", False)
            )
            if legacy_allowed:
                return PreflightCheck(
                    self.check_name, "soft", True,
                    "correlation artifact missing as_of_date; explicit legacy "
                    "override enabled",
                    details=details,
                )
            return _soft_for_sell_only(
                self.check_name,
                f"correlation artifact missing as_of_date at {p}; strict "
                "full/buy preflight fails closed because leakage cannot be "
                "verified",
                run_mode=ctx.run_mode,
                details=details,
            )
        try:
            from kernel.walk_forward.leakage_guard import _to_timestamp  # noqa: PLC0415
            _to_timestamp(as_of, label="correlation as_of_date")
        except Exception as exc:  # noqa: BLE001
            return _soft_for_sell_only(
                self.check_name,
                f"correlation artifact has invalid as_of_date={as_of!r}: {exc}",
                run_mode=ctx.run_mode,
                details=details,
            )
        matrix_size = details["n_tickers"]
        return PreflightCheck(
            self.check_name, "hard", True,
            f"correlation artifact stamped as_of_date={as_of} "
            f"({matrix_size} tickers)",
            details=details,
        )
