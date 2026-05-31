"""Gate-group Tasks — WF gate metadata + regime-layered IC.

Migrated from kernel.preflight._check_wf_gate_metadata + _check_regime_layered_ic.
Behavior parity asserted by tests/test_preflight_pipeline_gate.py.

These are the LARGEST checks in the suite (131 + 131 lines each). They share
many helpers (_wf_metadata_from_payload, _soft_for_sell_only, etc.) which stay
in kernel.preflight via bridge import until the final retirement PR.
"""
from __future__ import annotations

import json

from kernel.preflight import (  # noqa: PLC0415 (legacy bridge)
    PreflightCheck,
    _active_panel_config,
    _active_panel_kind,
    _finite_float,
    _is_sell_only_run,
    _is_sequence_artifact,
    _load_sequence_sidecar,
    _resolve_artifact_path,
    _soft_for_sell_only,
    _wf_metadata_from_payload,
)

from ..base import PreflightTask
from ..ctx import PreflightContext


class WfGateMetadataTask(PreflightTask):
    """P-WF-GATE — refuse new risk on a known-failed WF artifact.

    Behavior parity with ``_check_wf_gate_metadata``. The contract:
      - artifact missing → soft pass (handled by P-MODEL-ARTIFACT)
      - sidecar unreadable → soft|hard per run_mode (P-PANEL-CONTRACT handles)
      - wf metadata absent → soft|hard per run_mode (full/buy strict)
      - passed=False → HARD fail (sell-only soft pass with warning)
      - passed=True with missing sanity / missing required numerics → soft|hard
        per run_mode (with config-controlled wf_gate.sanity_regime_ic_required
        relaxation, default True = strict)
      - passed=True + all required evidence present → HARD pass
    """

    check_name = "P-WF-GATE"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        panel_cfg = _active_panel_config(ctx.config)
        kind = _active_panel_kind(ctx.config, panel_cfg)
        rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
        p = _resolve_artifact_path(ctx.strategy_dir, rel)
        if not p.exists():
            return PreflightCheck(
                self.check_name, "soft", True,
                f"artifact missing at {p}; "
                "P-MODEL-ARTIFACT/P-PANEL-CONTRACT will handle",
            )
        try:
            if _is_sequence_artifact(kind, p):
                payload, _sidecar = _load_sequence_sidecar(p)
            else:
                payload = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            sell_only = _is_sell_only_run(ctx.run_mode)
            return PreflightCheck(
                self.check_name,
                "soft" if sell_only else "hard",
                True if sell_only else False,
                f"artifact/sidecar unreadable: {exc}; P-PANEL-CONTRACT will handle",
            )
        wf = _wf_metadata_from_payload(payload)
        if not wf:
            return _soft_for_sell_only(
                self.check_name,
                "WF gate metadata absent; full/buy runs require stamped WF "
                "Sharpe and SPY comparison evidence",
                run_mode=ctx.run_mode,
            )
        return self._evaluate_wf(wf, ctx)

    def _evaluate_wf(self, wf: dict, ctx: PreflightContext) -> PreflightCheck:
        passed = wf.get("passed")
        details = {
            "passed": passed,
            "run_mode": ctx.run_mode,
            "wf_3cut_sharpe_mean": wf.get("wf_3cut_sharpe_mean"),
            "wf_3cut_apy_mean": wf.get("wf_3cut_apy_mean"),
            "spy_sharpe_mean": wf.get("spy_sharpe_mean"),
            "strategy_minus_spy_sharpe_mean": wf.get("strategy_minus_spy_sharpe_mean"),
            "wf_reason": wf.get("wf_reason"),
            "run_at": wf.get("run_at"),
        }
        if passed is False:
            return self._fail_with_evidence(wf, details, ctx.run_mode)
        if passed is True:
            return self._validate_passed(wf, details, ctx)
        return _soft_for_sell_only(
            self.check_name,
            "WF gate metadata present but no boolean passed field; next "
            "promotion must stamp pass/fail evidence",
            run_mode=ctx.run_mode,
            details=details,
        )

    def _fail_with_evidence(self, wf: dict, details: dict,
                            run_mode: str | None) -> PreflightCheck:
        if _is_sell_only_run(run_mode):
            return PreflightCheck(
                self.check_name, "soft", True,
                "active panel artifact carries failed WF gate evidence; "
                "sell-only risk exits are allowed, but new buys remain blocked "
                "until a WF-passing artifact is promoted.",
                details=details,
            )
        return PreflightCheck(
            self.check_name, "hard", False,
            "active panel artifact carries failed WF gate evidence: "
            f"wf_sharpe_mean={wf.get('wf_3cut_sharpe_mean')} "
            f"spy_sharpe_mean={wf.get('spy_sharpe_mean')} "
            f"reason={wf.get('wf_reason')}. Refusing new live decisions until "
            "a WF-passing artifact is promoted or buy mode is explicitly isolated "
            "to shadow/research.",
            details=details,
        )

    def _validate_passed(self, wf: dict, details: dict,
                         ctx: PreflightContext) -> PreflightCheck:
        sanity = wf.get("sanity_regime_ic") if isinstance(wf.get("sanity_regime_ic"), dict) else None
        details["sanity_regime_passed"] = sanity.get("passed") if sanity else None
        sanity_required = bool(
            (ctx.config.get("wf_gate") or {}).get("sanity_regime_ic_required", True)
        )
        if sanity_required and (not sanity or sanity.get("passed") is not True):
            details["sanity_regime_ic"] = sanity
            return _soft_for_sell_only(
                self.check_name,
                "WF gate passed=true but missing/failed regime sanity IC "
                "evidence; rerun WF gate so regime-layered placebo/IC is "
                "part of the acceptance verdict",
                run_mode=ctx.run_mode,
                details=details,
            )
        required = {
            "wf_3cut_sharpe_mean": wf.get("wf_3cut_sharpe_mean"),
            "spy_sharpe_mean": wf.get("spy_sharpe_mean"),
            "strategy_minus_spy_sharpe_mean": wf.get("strategy_minus_spy_sharpe_mean"),
        }
        missing = [k for k, v in required.items() if _finite_float(v) is None]
        if missing:
            details["missing_required_numeric"] = missing
            return _soft_for_sell_only(
                self.check_name,
                "WF gate passed=true but missing/non-finite required evidence: "
                + ", ".join(missing),
                run_mode=ctx.run_mode,
                details=details,
            )
        if "n_cuts_beat_spy_sharpe" not in wf:
            details["missing_required"] = ["n_cuts_beat_spy_sharpe"]
            return _soft_for_sell_only(
                self.check_name,
                "WF gate passed=true but missing SPY cut-count evidence "
                "(n_cuts_beat_spy_sharpe)",
                run_mode=ctx.run_mode,
                details=details,
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"WF gate passed: wf_sharpe_mean={wf.get('wf_3cut_sharpe_mean')} "
            f"spy_sharpe_mean={wf.get('spy_sharpe_mean')}",
            details=details,
        )


class RegimeLayeredICTask(PreflightTask):
    """P-REGIME-IC — model must carry per-regime rank-IC signal evidence.

    Behavior parity with ``_check_regime_layered_ic``. Same configurable
    relaxation via ``strategy_config.wf_gate.sanity_regime_ic_required``
    (default True = strict).
    """

    check_name = "P-REGIME-IC"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        panel_cfg = _active_panel_config(ctx.config)
        kind = _active_panel_kind(ctx.config, panel_cfg)
        rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
        p = _resolve_artifact_path(ctx.strategy_dir, rel)
        if not p.exists():
            return PreflightCheck(
                self.check_name, "soft", True,
                f"artifact missing at {p}; "
                "P-MODEL-ARTIFACT/P-PANEL-CONTRACT will handle",
            )
        try:
            if _is_sequence_artifact(kind, p):
                payload, _sidecar = _load_sequence_sidecar(p)
            else:
                payload = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            return _soft_for_sell_only(
                self.check_name,
                f"artifact/sidecar unreadable: {exc}; "
                "P-PANEL-CONTRACT will handle",
                run_mode=ctx.run_mode,
            )
        wf = _wf_metadata_from_payload(payload)
        tm = wf.get("trade_monotonicity") if isinstance(wf.get("trade_monotonicity"), dict) else {}
        sanity = wf.get("sanity_regime_ic") if isinstance(wf.get("sanity_regime_ic"), dict) else {}
        details = self._initial_details(p, ctx.run_mode, tm, sanity)
        admission_cfg = panel_cfg.get("regime_admission", {}) or {}
        require_sanity = bool(admission_cfg.get("require_sanity_regime_ic", True))
        return self._evaluate(wf, tm, sanity, details, ctx, require_sanity)

    def _initial_details(self, artifact_path, run_mode, tm: dict, sanity: dict) -> dict:
        return {
            "run_mode": run_mode,
            "artifact": str(artifact_path),
            "passed": tm.get("passed") if tm else None,
            "sanity_passed": sanity.get("passed") if sanity else None,
            "pooled": tm.get("pooled") if tm else None,
            "min_n_per_regime": tm.get("min_n_per_regime") if tm else None,
            "min_spearman": tm.get("min_spearman") if tm else None,
        }

    def _evaluate(self, wf: dict, tm: dict, sanity: dict, details: dict,
                  ctx: PreflightContext, require_sanity: bool) -> PreflightCheck:
        if not tm:
            return _soft_for_sell_only(
                self.check_name,
                "regime-layered IC/monotonicity evidence absent from WF metadata",
                run_mode=ctx.run_mode,
                details=details,
            )
        if require_sanity and not sanity:
            return _soft_for_sell_only(
                self.check_name,
                "regime sanity IC evidence absent from WF metadata",
                run_mode=ctx.run_mode,
                details=details,
            )
        if sanity and sanity.get("passed") is False:
            details["sanity_regime_ic"] = sanity
            sanity_required = bool(
                (ctx.config.get("wf_gate") or {}).get("sanity_regime_ic_required", True)
            )
            if sanity_required:
                return _soft_for_sell_only(
                    self.check_name,
                    f"regime sanity IC failed: {sanity.get('reason', 'unknown')}",
                    run_mode=ctx.run_mode,
                    details=details,
                )
            details["sanity_regime_ic_relaxed"] = True
        return self._evaluate_regimes(wf, tm, details, ctx)

    def _evaluate_regimes(self, wf: dict, tm: dict, details: dict,
                          ctx: PreflightContext) -> PreflightCheck:
        regimes_raw = tm.get("regimes")
        if isinstance(regimes_raw, dict):
            regimes = regimes_raw
        elif isinstance(regimes_raw, list):
            regimes = {
                str(stats.get("regime", f"regime_{idx}")): stats
                for idx, stats in enumerate(regimes_raw)
                if isinstance(stats, dict)
            }
        else:
            regimes = {}
        eligible = {
            regime: stats for regime, stats in regimes.items()
            if isinstance(stats, dict) and bool(stats.get("eligible", False))
        }
        failed = {
            regime: stats for regime, stats in eligible.items()
            if not bool(stats.get("passed", False))
        }
        details["eligible_regimes"] = sorted(eligible)
        details["failed_regimes"] = sorted(failed)
        details["regimes"] = regimes
        if not eligible:
            return _soft_for_sell_only(
                self.check_name,
                "no regime has enough OOS trades for regime-layered IC validation",
                run_mode=ctx.run_mode,
                details=details,
            )
        if tm.get("passed") is False or failed:
            regime_evidence_required = bool(
                (ctx.config.get("wf_gate") or {}).get("sanity_regime_ic_required", True)
            )
            if regime_evidence_required:
                return _soft_for_sell_only(
                    self.check_name,
                    "regime-layered IC/monotonicity failed for eligible regimes: "
                    + ", ".join(sorted(failed) or sorted(eligible)),
                    run_mode=ctx.run_mode,
                    details=details,
                )
            details["trade_monotonicity_relaxed"] = True
        pooled = tm.get("pooled") if isinstance(tm.get("pooled"), dict) else {}
        pooled_spearman = _finite_float(pooled.get("spearman")) if pooled else None
        return PreflightCheck(
            self.check_name, "hard", True,
            "regime-layered IC/monotonicity passed for eligible regimes "
            f"{sorted(eligible)}; pooled_spearman={pooled_spearman}",
            details=details,
        )
