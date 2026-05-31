"""P-CONFIG-FP — live config's fingerprint matches artifact's stored fp.

Migrated from kernel.preflight._check_config_fingerprint. Catches:
watchlist drift, lookahead change, objective change, asset_embeddings flip
— the four-incidents class from 2026-04-27/28.

Most complex single check at 129 lines legacy. Decomposed per §1c into
sub-helpers ≤50 lines each.
"""
from __future__ import annotations

import json

from kernel.preflight import (  # noqa: PLC0415 (legacy bridge)
    PreflightCheck,
    _active_panel_config,
    _active_panel_kind,
    _check_sector_map_coverage,
    _is_sequence_artifact,
    _load_sequence_sidecar,
    _resolve_artifact_path,
    _soft_for_sell_only,
)

from ..base import PreflightTask
from ..ctx import PreflightContext


class ConfigFingerprintTask(PreflightTask):
    """P-CONFIG-FP — fingerprint match against artifact stamped fp.

    Behavior parity with ``_check_config_fingerprint``.
    """

    check_name = "P-CONFIG-FP"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        panel_cfg = _active_panel_config(ctx.config)
        kind = _active_panel_kind(ctx.config, panel_cfg)
        rel = panel_cfg.get(
            "artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json"
        )
        p = _resolve_artifact_path(ctx.strategy_dir, rel)
        if not p.exists():
            return PreflightCheck(
                self.check_name, "hard", False, f"artifact missing: {p}",
            )
        meta = self._load_meta(kind, p, ctx)
        if isinstance(meta, PreflightCheck):
            return meta  # error short-circuit
        return self._evaluate(meta, kind, p, ctx)

    def _load_meta(self, kind: str, p, ctx: PreflightContext):
        """Return meta dict or a PreflightCheck on error."""
        if _is_sequence_artifact(kind, p):
            try:
                meta, _sidecar = _load_sequence_sidecar(p)
                return meta
            except Exception as exc:  # noqa: BLE001
                return _soft_for_sell_only(
                    self.check_name,
                    f"{kind} sequence sidecar unavailable for fingerprint check: "
                    f"{exc}; P-PANEL-CONTRACT handles checkpoint validity",
                    run_mode=ctx.run_mode,
                )
        try:
            return json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "hard", False, f"unreadable: {exc}",
            )

    def _evaluate(self, meta: dict, kind: str, p,
                  ctx: PreflightContext) -> PreflightCheck:
        try:
            from kernel.config_consistency import (  # noqa: PLC0415
                _model_relevant_fields,
                fingerprint_config,
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "soft", True,
                f"config_consistency module unavailable: {exc} — skip",
            )
        live_fp = fingerprint_config(ctx.config)
        stored = meta.get("config_fingerprint")
        if stored is None:
            target = "sequence sidecar" if _is_sequence_artifact(kind, p) else "artifact"
            return _soft_for_sell_only(
                self.check_name,
                f"{target} lacks config fingerprint; full/buy runs require "
                "stamped sector/config metadata",
                run_mode=ctx.run_mode,
                details={"live": live_fp},
            )
        if stored == live_fp:
            return PreflightCheck(
                self.check_name, "hard", True,
                f"fingerprint match {live_fp}",
            )
        return self._handle_mismatch(meta, live_fp, stored,
                                     _model_relevant_fields, ctx)

    def _handle_mismatch(self, meta: dict, live_fp: str, stored: str,
                         model_relevant_fields_fn,
                         ctx: PreflightContext) -> PreflightCheck:
        stored_sub = meta.get("config_fingerprint_fields") or {}
        # Defensive: 2026-05-08 alpha158_fund artifact stores fingerprint_fields
        # as a LIST of field names (not a value dict like legacy 21-feat
        # format). When that's the case we cannot compute a per-field diff.
        if not isinstance(stored_sub, dict):
            return _soft_for_sell_only(
                self.check_name,
                f"fingerprint_fields stored as {type(stored_sub).__name__}, "
                f"not dict — can't diff. live={live_fp} stored={stored}",
                run_mode=ctx.run_mode,
            )
        diff_keys = []
        live_sub = model_relevant_fields_fn(ctx.config)
        for k in sorted(set(live_sub) | set(stored_sub)):
            if live_sub.get(k) != stored_sub.get(k):
                diff_keys.append(k)
        msg = (
            f"fingerprint mismatch: live={live_fp} stored={stored} "
            f"diff_fields={diff_keys}"
        )
        details = {
            "live": live_fp,
            "stored": stored,
            "diff_fields": diff_keys,
            "run_mode": ctx.run_mode,
        }
        return self._classify_mismatch(diff_keys, stored_sub, msg, details, ctx)

    def _classify_mismatch(self, diff_keys: list, stored_sub: dict,
                           msg: str, details: dict,
                           ctx: PreflightContext) -> PreflightCheck:
        normalized_mode = str(ctx.run_mode or "").lower().replace("_", "-")
        if normalized_mode.startswith("sell-only"):
            return PreflightCheck(
                self.check_name, "soft", True,
                msg + " Sell-only risk exits are allowed; new buys remain "
                "blocked until the artifact is retrained/stamped against the "
                "live config.",
                details=details,
            )
        legacy_sector_fields = {"sector_map", "sector_etf_map"}
        if (
            diff_keys
            and set(diff_keys).issubset(legacy_sector_fields)
            and not any(k in stored_sub for k in legacy_sector_fields)
        ):
            return self._legacy_sector_branch(msg, details, ctx)
        return PreflightCheck(
            self.check_name, "hard", False, msg, details=details,
        )

    def _legacy_sector_branch(self, msg: str, details: dict,
                              ctx: PreflightContext) -> PreflightCheck:
        # Delegate to the legacy sector check for now — preserves bytewise
        # parity. Once SectorMapCoverageTask is the only impl, this will call
        # SectorMapCoverageTask().check(ctx) directly.
        sector_check = _check_sector_map_coverage(
            ctx.config, ctx.strategy_dir, ctx.run_mode,
        )
        details = details | {
            "legacy_missing_sector_fields": True,
            "sector_coverage_ok": sector_check.ok,
            "sector_coverage_severity": sector_check.severity,
            "sector_coverage_message": sector_check.message,
            "sector_coverage_details": sector_check.details,
        }
        if sector_check.ok:
            return _soft_for_sell_only(
                self.check_name,
                msg + " Legacy artifact lacks sector fingerprint fields added "
                "after training. Full/buy runs require retrain/stamp even when "
                "P-SECTOR-MAP coverage is currently OK.",
                run_mode=ctx.run_mode,
                details=details,
            )
        return PreflightCheck(
            self.check_name, "hard", False,
            msg + " Legacy artifact lacks sector fingerprint fields, and "
            "P-SECTOR-MAP did not pass; fix sector metadata or retrain/stamp "
            "before enabling buy mode.",
            details=details,
        )
