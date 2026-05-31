"""P-WATCHLIST — config watchlist length consistent with training.

Migrated from kernel.preflight._check_watchlist_size.
"""
from __future__ import annotations

import json

from kernel.preflight import (  # noqa: PLC0415 (legacy bridge)
    PreflightCheck,
    _active_panel_config,
    _active_panel_kind,
    _is_sequence_artifact,
    _load_sequence_sidecar,
    _resolve_artifact_path,
)

from ..base import PreflightTask
from ..ctx import PreflightContext


class WatchlistSizeTask(PreflightTask):
    """P-WATCHLIST — every config watchlist ticker must be one the artifact was
    trained on (and vice versa). Mismatches catch drift across promotion.

    Behavior parity with ``_check_watchlist_size``.
    """

    check_name = "P-WATCHLIST"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        wl = ctx.config.get("watchlist") or []
        panel_cfg = _active_panel_config(ctx.config)
        kind = _active_panel_kind(ctx.config, panel_cfg)
        rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
        p = _resolve_artifact_path(ctx.strategy_dir, rel)
        if not p.exists():
            return PreflightCheck(
                self.check_name, "hard", False, f"artifact missing: {p}",
            )
        if _is_sequence_artifact(kind, p):
            return self._check_sequence_artifact(wl, p, kind)
        return self._check_json_artifact(wl, p)

    def _check_sequence_artifact(self, wl: list, p, kind: str) -> PreflightCheck:
        try:
            meta, _sidecar = _load_sequence_sidecar(p)
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "soft", True,
                f"{kind} summary unavailable for watchlist check: {exc}; "
                f"live={len(wl)} ticker(s)",
            )
        fields = meta.get("config_fingerprint_fields") or {}
        if not isinstance(fields, dict):
            fields = {}
        trained_wl = fields.get("watchlist") or []
        if trained_wl:
            return self._compare_watchlists(wl, trained_wl)
        return PreflightCheck(
            self.check_name, "soft", True,
            f"trained watchlist not stamped for sequence artifact; "
            f"live={len(wl)} ticker(s)",
        )

    def _check_json_artifact(self, wl: list, p) -> PreflightCheck:
        try:
            meta = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "hard", False, f"unreadable: {exc}",
            )
        fields = meta.get("config_fingerprint_fields") or {}
        # alpha158_fund artifact may store fingerprint_fields as a LIST of
        # field names (no values). Treat as not-stamped.
        if not isinstance(fields, dict):
            fields = {}
        trained_wl = fields.get("watchlist") or []
        if not trained_wl:
            return PreflightCheck(
                self.check_name, "soft", True,
                f"trained watchlist not stamped; live={len(wl)} ticker(s)",
            )
        return self._compare_watchlists(wl, trained_wl)

    def _compare_watchlists(self, wl: list, trained_wl: list) -> PreflightCheck:
        if set(wl) != set(trained_wl):
            only_live = sorted(set(wl) - set(trained_wl))[:5]
            only_trained = sorted(set(trained_wl) - set(wl))[:5]
            return PreflightCheck(
                self.check_name, "hard", False,
                f"watchlist mismatch live={len(wl)} trained={len(trained_wl)} "
                f"in_live_not_trained={only_live} "
                f"in_trained_not_live={only_trained}",
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"watchlist match (n={len(wl)})",
        )
