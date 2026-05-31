"""Pre-flight smoke test — runs at the start of every cron invocation.

Catches the class of bugs where a config / artifact / state file drifts
out of sync with what the runner assumes. Each check returns a
PreflightCheck result; any HARD failure raises PreflightFailed which
the live runner converts into an ntfy alert + abort (no orders placed).

Why this exists (2026-04-28):
  - 4-27: NGBoost feature drift (macro cols missing) → 0 buy
  - 4-28a: watchlist 227 vs model 103 mismatch → 06:32 fingerprint alert
  - 4-28b: production model best_iter=4 (untrained) → +0.0418 IC was
    random-walk noise
  - 4-28c: 103 launchd plist crashed every day on TypeError
  Each was different but ALL would have been caught by a 30-second
  pre-flight run at cron startup.

Checks (each returns ok / soft-warn / hard-fail):
  1. P-MODEL-ARTIFACT   — panel-ltr.json + ngboost-head.json exist + parse
  2. P-BEST-ITER        — best_iter ≥ min_best_iter (BUG-CV-2)
  3. P-CONFIG-FP        — config fingerprint matches artifact's stored fp
                          (BUG-CV-G7-mismatch class)
  4. P-WATCHLIST        — config watchlist size matches training watchlist
	  5. P-WF-GATE          — active artifact must not carry failed WF gate
	                          evidence
	  6. P-CORR-METADATA    — correlation artifact must be stamped with
	                          as_of_date before buy/full runs
	  7. P-FEATURE-COVER    — NGBoost head's feature_cols all present in
	                          current panel pipeline output (≥ 95%)
	  8. P-STATE-FILE       — live_state.{broker}.json parses (or absent
	                          which is fine — first run)
	  9. P-BROKER-CONNECT   — broker.connect() / get_account_value() works
	                          (only if broker is provided; skipped in dry-run)

Usage in live/runner.py:
    from kernel.preflight import run_preflight, PreflightFailed
    try:
        run_preflight(config, broker, strategy_dir)
    except PreflightFailed as e:
        log.error("PRE-FLIGHT FAILED: %s", e)
        ntfy(...)
        sys.exit(2)
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("kernel.preflight")


def _resolve_artifact_path(strategy_dir: Path, rel: str | Path) -> Path:
    """Resolve config artifact paths relative to the strategy directory."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return strategy_dir / p


def _patchtst_summary_path(path: Path) -> Path:
    """Return the lightweight JSON sidecar expected for HF PatchTST .pt files."""
    if path.name.endswith("_model.pt"):
        return path.with_name(path.name[:-len("_model.pt")] + "_summary.json")
    return path.with_suffix(".summary.json")


def _sequence_sidecar_paths(path: Path) -> list[Path]:
    """Candidate metadata sidecars for non-JSON sequence artifacts.

    Order is significant: ``_load_sequence_sidecar`` returns the first hit and
    ``_load_sequence_sidecar_merged`` MERGES later hits over earlier ones.

    2026-05-30 bug: pre-fix order put ``_summary.json`` FIRST, so wf_gate_metadata
    written into ``.pt.metadata.json`` by the WF gate runner was invisible to
    preflight (summary always existed and short-circuited the read). Reversed
    the order so the gate-written sidecar wins. Summary still provides the
    training contract (best_val_ic / n_features / config_fingerprint).
    """
    out = [path.with_name(path.name + ".metadata.json"), _patchtst_summary_path(path)]
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _load_sequence_sidecar(path: Path) -> tuple[dict, Path]:
    """Return the merged sidecar payload + the first existing source path.

    Pre-2026-05-30: returned only the FIRST existing sidecar, so e.g. wf_gate
    stamps in ``.pt.metadata.json`` were hidden when ``_summary.json`` existed.
    Post-fix: merges every existing sidecar (last writer wins per key) so the
    training-time summary fields (best_val_ic etc.) and the gate-time
    wf_gate_metadata both surface in the same payload.
    """
    merged: dict = {}
    first_path: Path | None = None
    for sidecar in reversed(_sequence_sidecar_paths(path)):
        if not sidecar.exists():
            continue
        side = json.loads(sidecar.read_text())
        if first_path is None:
            first_path = sidecar
        # Shallow merge — top-level keys from later iterations override earlier.
        # 'metadata' sub-dict gets deep-merged so wf_gate_metadata stamps survive.
        for k, v in side.items():
            if k == "metadata" and isinstance(v, dict) and isinstance(merged.get("metadata"), dict):
                merged_md = dict(merged["metadata"])
                merged_md.update(v)
                merged["metadata"] = merged_md
            else:
                merged[k] = v
    if not merged:
        raise FileNotFoundError(
            "missing sequence sidecar; checked "
            + ", ".join(str(p) for p in _sequence_sidecar_paths(path))
        )
    # Return the first sidecar path found in priority order (the one whose
    # presence triggered the original lookup; used for error messages only).
    assert first_path is not None
    primary = next((p for p in _sequence_sidecar_paths(path) if p.exists()), first_path)
    return merged, primary


def _wf_metadata_from_payload(payload: dict) -> dict:
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    wf = meta.get("wf_gate_metadata") if isinstance(meta.get("wf_gate_metadata"), dict) else None
    if wf is None and isinstance(payload.get("wf_gate_metadata"), dict):
        wf = payload.get("wf_gate_metadata")
    return wf or {}


def _active_panel_config(config: dict) -> dict:
    """Return the artifact config used by the active scoring path."""
    return (
        config.get("ranking", {})
        .get("panel_scoring", {})
        or config.get("panel_ltr", {})
        or {}
    )


def _active_panel_kind(config: dict, panel_cfg: dict | None = None) -> str:
    panel_cfg = panel_cfg or _active_panel_config(config)
    return str(
        panel_cfg.get("kind")
        or config.get("panel_ltr", {}).get("backend")
        or "xgb"
    )


def _is_sequence_artifact(kind: str, path: Path) -> bool:
    return kind in {"hf_patchtst", "patchtst"} or path.suffix == ".pt"


def _normalized_run_mode(run_mode: str | None) -> str:
    return str(run_mode or "").strip().lower().replace("_", "-")


def _is_sell_only_run(run_mode: str | None) -> bool:
    return _normalized_run_mode(run_mode).startswith("sell-only")


def _is_global_calibration_enabled(config: dict) -> bool:
    return bool(
        (config.get("ranking", {})
               .get("panel_scoring", {})
               .get("global_calibration", {}) or {})
        .get("enabled", False)
    )


def _ngboost_activation(config: dict) -> tuple[dict, list[str], bool]:
    """Return NGBoost config, activating regime overlays, and active flag."""
    ngb_cfg = (
        (config.get("ranking", {}) or {})
        .get("panel_scoring", {})
        .get("ngboost", {})
        or {}
    )
    regime_params = config.get("regime_params", {}) or {}
    per_regime_activates = [
        str(regime) for regime, params in regime_params.items()
        if isinstance(params, dict)
        and isinstance(params.get("ngboost"), dict)
        and params["ngboost"].get("enabled") is True
    ]
    return ngb_cfg, per_regime_activates, (
        bool(ngb_cfg.get("enabled", False)) or bool(per_regime_activates)
    )


def _soft_for_sell_only(
    name: str,
    message: str,
    *,
    run_mode: str | None,
    details: dict | None = None,
) -> PreflightCheck:
    if _is_sell_only_run(run_mode):
        msg = f"{message}; sell-only risk exits are allowed, new buys remain blocked"
        return PreflightCheck(name, "soft", True, msg, details=details or {})
    return PreflightCheck(name, "hard", False, message, details=details or {})


def _finite_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _check_sequence_artifact_contract(
    *,
    kind: str,
    artifact_path: Path,
    strict_contract: bool,
) -> PreflightCheck:
    """Backend-aware contract for non-JSON sequence scorers.

    HF PatchTST checkpoints are PyTorch zip/pickle files, not JSON panel-LTR
    artifacts. Preflight must validate their lightweight JSON sidecar instead
    of trying to decode the binary checkpoint as UTF-8.
    """
    if not artifact_path.exists():
        return PreflightCheck(
            "P-PANEL-CONTRACT", "hard", False,
            f"{kind} checkpoint missing: {artifact_path}",
        )
    if artifact_path.stat().st_size <= 0:
        return PreflightCheck(
            "P-PANEL-CONTRACT", "hard", False,
            f"{kind} checkpoint is empty: {artifact_path}",
        )

    try:
        summary, summary_path = _load_sequence_sidecar(artifact_path)
    except Exception as exc:
        return PreflightCheck(
            "P-PANEL-CONTRACT", "hard", False,
            f"{kind} summary sidecar missing/unreadable: {exc}",
        )

    arch = summary.get("arch")
    if arch and arch != kind:
        return PreflightCheck(
            "P-PANEL-CONTRACT", "hard", False,
            f"{kind} summary arch mismatch: arch={arch!r}",
        )
    best_val_ic = summary.get("best_val_ic")
    n_features = summary.get("n_features")
    try:
        best_val_ic_f = float(best_val_ic)
    except (TypeError, ValueError):
        best_val_ic_f = float("nan")
    try:
        n_features_i = int(n_features)
    except (TypeError, ValueError):
        n_features_i = 0

    errors = []
    if not math.isfinite(best_val_ic_f):
        errors.append("best_val_ic missing or non-finite")
    if n_features_i <= 0:
        errors.append("n_features missing or non-positive")
    if strict_contract and summary.get("seed") is None:
        errors.append("seed missing")
    if strict_contract and summary.get("cut") is None:
        errors.append("cut missing")
    if strict_contract and summary.get("config_fingerprint") is None:
        errors.append("config_fingerprint missing")
    if errors:
        return PreflightCheck(
            "P-PANEL-CONTRACT", "hard", False,
            f"{kind} sidecar contract failed: {'; '.join(errors)}",
            details={"summary_path": str(summary_path), "checkpoint": str(artifact_path)},
        )
    return PreflightCheck(
        "P-PANEL-CONTRACT", "hard", True,
        f"{kind} checkpoint contract ok: val_ic={best_val_ic_f:+.4f} "
        f"n_features={n_features_i}",
        details={
            "kind": kind,
            "checkpoint": str(artifact_path),
            "summary_path": str(summary_path),
            "best_val_ic": best_val_ic_f,
            "n_features": n_features_i,
            "seed": summary.get("seed"),
            "cut": summary.get("cut"),
            "config_fingerprint": summary.get("config_fingerprint"),
            "trained_watchlist_n": summary.get("trained_watchlist_n"),
        },
    )


@dataclass
class PreflightCheck:
    name:     str
    severity: str    # "hard" | "soft"
    ok:       bool
    message:  str = ""
    details:  dict = field(default_factory=dict)


class PreflightFailed(RuntimeError):
    """Raised when any HARD check fails. Caught by runner.main()."""

    def __init__(self, failures: list[PreflightCheck]):
        self.failures = failures
        super().__init__(self._format(failures))

    @staticmethod
    def _format(failures: list[PreflightCheck]) -> str:
        lines = [f"{len(failures)} hard pre-flight check(s) failed:"]
        for c in failures:
            lines.append(f"  ✗ {c.name}: {c.message}")
        lines.append(
            "Live runner aborting. No orders placed. "
            "Investigate and re-run after fix."
        )
        return "\n".join(lines)


# ── Individual checks ──────────────────────────────────────────────────────

def _check_model_artifact(config: dict, strategy_dir: Path) -> PreflightCheck:
    """P-MODEL-ARTIFACT: active scorer artifact exists + parses."""
    panel_cfg = _active_panel_config(config)
    kind = _active_panel_kind(config, panel_cfg)
    rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = _resolve_artifact_path(strategy_dir, rel)
    if not p.exists():
        return PreflightCheck(
            "P-MODEL-ARTIFACT", "hard", False,
            f"artifact missing: {p}",
        )
    if _is_sequence_artifact(kind, p):
        if p.stat().st_size <= 0:
            return PreflightCheck(
                "P-MODEL-ARTIFACT", "hard", False,
                f"{kind} checkpoint is empty: {p}",
            )
        return PreflightCheck(
            "P-MODEL-ARTIFACT", "hard", True,
            f"loaded {kind} checkpoint {p.name}",
            details={"path": str(p), "kind": kind, "bytes": p.stat().st_size},
        )
    try:
        meta = json.loads(p.read_text())
    except Exception as exc:
        return PreflightCheck(
            "P-MODEL-ARTIFACT", "hard", False,
            f"artifact unreadable {p.name}: {exc}",
        )
    return PreflightCheck(
        "P-MODEL-ARTIFACT", "hard", True,
        f"loaded {p.name}",
        details={"path": str(p), "best_iter": meta.get("best_iter"),
                 "oos_mean_ic": meta.get("oos_mean_ic")},
    )


def _check_panel_artifact_contract(
    config: dict,
    strategy_dir: Path,
    run_mode: str | None = None,
) -> PreflightCheck:
    """P-PANEL-CONTRACT: panel artifact carries evidence metadata.

    Full/buy paths are strict by default. Sell-only remains soft so risk exits
    are not blocked by missing buy-side evidence.
    """
    panel_cfg = _active_panel_config(config)
    kind = _active_panel_kind(config, panel_cfg)
    rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = _resolve_artifact_path(strategy_dir, rel)
    if not p.exists():
        return PreflightCheck("P-PANEL-CONTRACT", "hard", False, f"artifact missing: {p}")
    strict_contract = bool(
        config.get("preflight", {})
        .get("artifact_contract", {})
        .get("strict", not _is_sell_only_run(run_mode))
    )
    if _is_sequence_artifact(kind, p):
        return _check_sequence_artifact_contract(
            kind=kind,
            artifact_path=p,
            strict_contract=strict_contract,
        )
    try:
        payload = json.loads(p.read_text())
    except Exception as exc:
        return PreflightCheck("P-PANEL-CONTRACT", "hard", False, f"unreadable: {exc}")
    from kernel.artifact_contract import validate_panel_artifact_contract  # noqa: PLC0415
    result = validate_panel_artifact_contract(
        payload,
        strict=strict_contract,
        runtime_config=config,
    )
    severity = "hard" if strict_contract else "soft"
    if not result.ok:
        if _is_sell_only_run(run_mode):
            return PreflightCheck(
                "P-PANEL-CONTRACT", "soft", True,
                "panel artifact contract failed: " + "; ".join(result.errors)
                + "; sell-only risk exits are allowed, new buys remain blocked",
                details=result.details | {"warnings": result.warnings},
            )
        return PreflightCheck(
            "P-PANEL-CONTRACT", severity, False,
            "; ".join(result.errors),
            details=result.details | {"warnings": result.warnings},
        )
    msg = "contract ok"
    if result.warnings:
        msg = "contract legacy-compatible; " + "; ".join(result.warnings[:3])
    return PreflightCheck(
        "P-PANEL-CONTRACT", severity, True, msg, details=result.details,
    )


def _check_wf_gate_metadata(
    config: dict,
    strategy_dir: Path,
    run_mode: str | None = None,
) -> PreflightCheck:
    """P-WF-GATE: a known-failed WF artifact must not open new risk.

    CLAUDE.md makes weekly WF + sanity gates the production trust boundary.
    Pre-fix, the active artifact could carry ``metadata.wf_gate_metadata`` with
    ``passed=false`` while runtime preflight ignored it and continued to buy.
    That is worse than missing evidence: it is known negative evidence.

    Sell-only runs are risk-reduction paths. They must keep working even when
    buy-side evidence fails, otherwise the same guard that blocks bad entries
    can also block exits.
    """
    panel_cfg = _active_panel_config(config)
    kind = _active_panel_kind(config, panel_cfg)
    rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = _resolve_artifact_path(strategy_dir, rel)
    if not p.exists():
        return PreflightCheck(
            "P-WF-GATE", "soft", True,
            f"artifact missing at {p}; P-MODEL-ARTIFACT/P-PANEL-CONTRACT will handle",
        )
    try:
        if _is_sequence_artifact(kind, p):
            payload, _sidecar = _load_sequence_sidecar(p)
        else:
            payload = json.loads(p.read_text())
    except Exception as exc:
        return PreflightCheck(
            "P-WF-GATE", "soft" if _is_sell_only_run(run_mode) else "hard",
            True if _is_sell_only_run(run_mode) else False,
            f"artifact/sidecar unreadable: {exc}; P-PANEL-CONTRACT will handle",
        )
    wf = _wf_metadata_from_payload(payload)
    if not wf:
        return _soft_for_sell_only(
            "P-WF-GATE",
            "WF gate metadata absent; full/buy runs require stamped WF Sharpe "
            "and SPY comparison evidence",
            run_mode=run_mode,
        )
    passed = wf.get("passed")
    details = {
        "passed": passed,
        "run_mode": run_mode,
        "wf_3cut_sharpe_mean": wf.get("wf_3cut_sharpe_mean"),
        "wf_3cut_apy_mean": wf.get("wf_3cut_apy_mean"),
        "spy_sharpe_mean": wf.get("spy_sharpe_mean"),
        "strategy_minus_spy_sharpe_mean": wf.get("strategy_minus_spy_sharpe_mean"),
        "wf_reason": wf.get("wf_reason"),
        "run_at": wf.get("run_at"),
    }
    if passed is False:
        if _is_sell_only_run(run_mode):
            return PreflightCheck(
                "P-WF-GATE", "soft", True,
                "active panel artifact carries failed WF gate evidence; "
                "sell-only risk exits are allowed, but new buys remain blocked "
                "until a WF-passing artifact is promoted.",
                details=details,
            )
        return PreflightCheck(
            "P-WF-GATE", "hard", False,
            "active panel artifact carries failed WF gate evidence: "
            f"wf_sharpe_mean={wf.get('wf_3cut_sharpe_mean')} "
            f"spy_sharpe_mean={wf.get('spy_sharpe_mean')} "
            f"reason={wf.get('wf_reason')}. Refusing new live decisions until "
            "a WF-passing artifact is promoted or buy mode is explicitly isolated "
            "to shadow/research.",
            details=details,
        )
    if passed is True:
        sanity = wf.get("sanity_regime_ic") if isinstance(wf.get("sanity_regime_ic"), dict) else None
        details["sanity_regime_passed"] = sanity.get("passed") if sanity else None
        # Config-controlled relaxation (default True preserves strict behaviour).
        # Operator opt-in via strategy_config.wf_gate.sanity_regime_ic_required.
        sanity_required = bool(
            (config.get("wf_gate") or {}).get("sanity_regime_ic_required", True)
        )
        if sanity_required and (not sanity or sanity.get("passed") is not True):
            details["sanity_regime_ic"] = sanity
            return _soft_for_sell_only(
                "P-WF-GATE",
                "WF gate passed=true but missing/failed regime sanity IC "
                "evidence; rerun WF gate so regime-layered placebo/IC is "
                "part of the acceptance verdict",
                run_mode=run_mode,
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
                "P-WF-GATE",
                "WF gate passed=true but missing/non-finite required evidence: "
                + ", ".join(missing),
                run_mode=run_mode,
                details=details,
            )
        if "n_cuts_beat_spy_sharpe" not in wf:
            details["missing_required"] = ["n_cuts_beat_spy_sharpe"]
            return _soft_for_sell_only(
                "P-WF-GATE",
                "WF gate passed=true but missing SPY cut-count evidence "
                "(n_cuts_beat_spy_sharpe)",
                run_mode=run_mode,
                details=details,
            )
        return PreflightCheck(
            "P-WF-GATE", "hard", True,
            f"WF gate passed: wf_sharpe_mean={wf.get('wf_3cut_sharpe_mean')} "
            f"spy_sharpe_mean={wf.get('spy_sharpe_mean')}",
            details=details,
        )
    return _soft_for_sell_only(
        "P-WF-GATE",
        "WF gate metadata present but no boolean passed field; next promotion "
        "must stamp pass/fail evidence",
        run_mode=run_mode,
        details=details,
    )


def _check_regime_layered_ic(
    config: dict,
    strategy_dir: Path,
    run_mode: str | None = None,
) -> PreflightCheck:
    """P-REGIME-IC: WF artifact must carry regime-layered signal evidence.

    The contract is deliberately separate from the portfolio/QP contract:
    the model must first prove rank monotonicity / rank-IC style signal inside
    regimes with enough samples. QP can size eligible alpha; it cannot create
    alpha from missing or failed regime evidence.
    """
    panel_cfg = _active_panel_config(config)
    kind = _active_panel_kind(config, panel_cfg)
    rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = _resolve_artifact_path(strategy_dir, rel)
    if not p.exists():
        return PreflightCheck(
            "P-REGIME-IC", "soft", True,
            f"artifact missing at {p}; P-MODEL-ARTIFACT/P-PANEL-CONTRACT will handle",
        )
    try:
        if _is_sequence_artifact(kind, p):
            payload, _sidecar = _load_sequence_sidecar(p)
        else:
            payload = json.loads(p.read_text())
    except Exception as exc:
        return _soft_for_sell_only(
            "P-REGIME-IC",
            f"artifact/sidecar unreadable: {exc}; P-PANEL-CONTRACT will handle",
            run_mode=run_mode,
        )
    wf = _wf_metadata_from_payload(payload)
    tm = wf.get("trade_monotonicity") if isinstance(wf.get("trade_monotonicity"), dict) else {}
    sanity = wf.get("sanity_regime_ic") if isinstance(wf.get("sanity_regime_ic"), dict) else {}
    admission_cfg = panel_cfg.get("regime_admission", {}) or {}
    require_sanity = bool(admission_cfg.get("require_sanity_regime_ic", True))
    details = {
        "run_mode": run_mode,
        "artifact": str(p),
        "passed": tm.get("passed") if tm else None,
        "sanity_passed": sanity.get("passed") if sanity else None,
        "pooled": tm.get("pooled") if tm else None,
        "min_n_per_regime": tm.get("min_n_per_regime") if tm else None,
        "min_spearman": tm.get("min_spearman") if tm else None,
    }
    if not tm:
        return _soft_for_sell_only(
            "P-REGIME-IC",
            "regime-layered IC/monotonicity evidence absent from WF metadata",
            run_mode=run_mode,
            details=details,
        )
    if require_sanity and not sanity:
        return _soft_for_sell_only(
            "P-REGIME-IC",
            "regime sanity IC evidence absent from WF metadata",
            run_mode=run_mode,
            details=details,
        )
    if sanity and sanity.get("passed") is False:
        details["sanity_regime_ic"] = sanity
        # Config-controlled relaxation (default True preserves strict behaviour).
        # Operator opt-in via strategy_config.wf_gate.sanity_regime_ic_required.
        sanity_required = bool(
            (config.get("wf_gate") or {}).get("sanity_regime_ic_required", True)
        )
        if sanity_required:
            return _soft_for_sell_only(
                "P-REGIME-IC",
                f"regime sanity IC failed: {sanity.get('reason', 'unknown')}",
                run_mode=run_mode,
                details=details,
            )
        # Relax mode: log the failure but proceed.
        details["sanity_regime_ic_relaxed"] = True
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
            "P-REGIME-IC",
            "no regime has enough OOS trades for regime-layered IC validation",
            run_mode=run_mode,
            details=details,
        )
    if tm.get("passed") is False or failed:
        # Same relax flag as the sanity-failure branch: both are
        # regime-layered-evidence guards. Operator opt-in via
        # strategy_config.wf_gate.sanity_regime_ic_required.
        regime_evidence_required = bool(
            (config.get("wf_gate") or {}).get("sanity_regime_ic_required", True)
        )
        if regime_evidence_required:
            return _soft_for_sell_only(
                "P-REGIME-IC",
                "regime-layered IC/monotonicity failed for eligible regimes: "
                + ", ".join(sorted(failed) or sorted(eligible)),
                run_mode=run_mode,
                details=details,
            )
        details["trade_monotonicity_relaxed"] = True
    pooled = tm.get("pooled") if isinstance(tm.get("pooled"), dict) else {}
    pooled_spearman = _finite_float(pooled.get("spearman")) if pooled else None
    return PreflightCheck(
        "P-REGIME-IC", "hard", True,
        "regime-layered IC/monotonicity passed for eligible regimes "
        f"{sorted(eligible)}; pooled_spearman={pooled_spearman}",
        details=details,
    )


def _check_best_iter(config: dict, strategy_dir: Path) -> PreflightCheck:
    """P-BEST-ITER: model's best_iter ≥ min_best_iter (BUG-CV-2 class).

    Production was discovered to have best_iter=4 today (4 × 0.02 eta =
    0.08 total shrinkage = essentially untrained). This check refuses
    to trade on an undertrained model.
    """
    panel_cfg = _active_panel_config(config)
    kind = _active_panel_kind(config, panel_cfg)
    rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = _resolve_artifact_path(strategy_dir, rel)
    if not p.exists():
        return PreflightCheck(
            "P-BEST-ITER", "hard", False, f"artifact missing: {p}",
        )
    if _is_sequence_artifact(kind, p):
        return PreflightCheck(
            "P-BEST-ITER", "soft", True,
            f"best_iter not applicable to sequence artifact ({kind})",
        )
    try:
        meta = json.loads(p.read_text())
    except Exception as exc:
        return PreflightCheck(
            "P-BEST-ITER", "hard", False, f"unreadable: {exc}",
        )
    bi = meta.get("best_iter")
    if bi is None:
        # Older artifacts (e.g. transformer backend) may not stamp this.
        return PreflightCheck(
            "P-BEST-ITER", "soft", True,
            "best_iter not stamped in artifact (legacy pre-2026-04-28); skip",
        )
    min_bi = int(panel_cfg.get("min_best_iter", 5))
    if int(bi) < min_bi:
        # 2026-05-04 (P0 fix): mirror the FinalFitTask training-time
        # escape clause. best_iter < min_best_iter is a FALSE POSITIVE
        # on strong-univariate-IC features (XGBoost converges by round
        # 4-9 and further rounds add zero eval-set improvement). If
        # eval_ic at best_iter is healthy (≥ floor, default 0.02),
        # accept the model. This keeps the runtime guard symmetric
        # with the training-time guard — pre-fix, training accepted
        # the model + saved the artifact, then preflight refused to
        # load it = strategy never trades. Pathological case
        # (eval_ic ≈ 0 or missing from artifact) still fails-safe.
        eval_ic_floor = float(panel_cfg.get("min_best_iter_eval_ic_floor", 0.02))
        eval_ic = meta.get("eval_ic")
        try:
            eval_ic_f = float(eval_ic) if eval_ic is not None else None
        except (TypeError, ValueError):
            eval_ic_f = None
        import math as _math
        if (eval_ic_f is not None and _math.isfinite(eval_ic_f)
                and eval_ic_f >= eval_ic_floor):
            return PreflightCheck(
                "P-BEST-ITER", "hard", True,
                f"best_iter={bi} < {min_bi} but eval_ic={eval_ic_f:+.4f} ≥ "
                f"floor={eval_ic_floor:+.4f} — strong-univariate-IC plateau, accepting",
                details={"best_iter": bi, "min_best_iter": min_bi,
                         "eval_ic": eval_ic_f, "eval_ic_floor": eval_ic_floor},
            )
        return PreflightCheck(
            "P-BEST-ITER", "hard", False,
            f"best_iter={bi} < min_best_iter={min_bi} AND "
            f"eval_ic={eval_ic} < floor={eval_ic_floor:+.4f}. "
            f"Model undertrained (early-stop fired in round {bi}). "
            f"Retrain required, OR confirm eval_ic is stamped in the artifact "
            f"(SaveArtifactTask must include 'eval_ic' in meta).",
            details={"best_iter": bi, "min_best_iter": min_bi,
                     "eval_ic": eval_ic, "eval_ic_floor": eval_ic_floor},
        )
    return PreflightCheck(
        "P-BEST-ITER", "hard", True,
        f"best_iter={bi} ≥ {min_bi}",
        details={"best_iter": bi},
    )


def _check_config_fingerprint(
    config: dict,
    strategy_dir: Path,
    run_mode: str | None = None,
) -> PreflightCheck:
    """P-CONFIG-FP: live config's fingerprint matches artifact's stored fp.

    Catches: watchlist drift, lookahead change, objective change,
    asset_embeddings flip — the four-incidents class from 2026-04-27/28.
    """
    panel_cfg = _active_panel_config(config)
    kind = _active_panel_kind(config, panel_cfg)
    rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = _resolve_artifact_path(strategy_dir, rel)
    if not p.exists():
        return PreflightCheck(
            "P-CONFIG-FP", "hard", False, f"artifact missing: {p}",
        )
    if _is_sequence_artifact(kind, p):
        try:
            meta, _sidecar = _load_sequence_sidecar(p)
        except Exception as exc:
            return _soft_for_sell_only(
                "P-CONFIG-FP",
                f"{kind} sequence sidecar unavailable for fingerprint check: {exc}; "
                "P-PANEL-CONTRACT handles checkpoint validity",
                run_mode=run_mode,
            )
    else:
        try:
            meta = json.loads(p.read_text())
        except Exception as exc:
            return PreflightCheck(
                "P-CONFIG-FP", "hard", False, f"unreadable: {exc}",
            )
    try:
        from kernel.config_consistency import (  # noqa: PLC0415
            fingerprint_config, _model_relevant_fields,
        )
    except Exception as exc:
        return PreflightCheck(
            "P-CONFIG-FP", "soft", True,
            f"config_consistency module unavailable: {exc} — skip",
        )
    live_fp = fingerprint_config(config)
    stored = meta.get("config_fingerprint")
    if stored is None:
        target = "sequence sidecar" if _is_sequence_artifact(kind, p) else "artifact"
        return _soft_for_sell_only(
            "P-CONFIG-FP",
            f"{target} lacks config fingerprint; full/buy runs require stamped "
            "sector/config metadata",
            run_mode=run_mode,
            details={"live": live_fp},
        )
    if stored == live_fp:
        return PreflightCheck(
            "P-CONFIG-FP", "hard", True,
            f"fingerprint match {live_fp}",
        )
    stored_sub = meta.get("config_fingerprint_fields") or {}
    # Defensive: 2026-05-08 alpha158_fund artifact stores
    # config_fingerprint_fields as a LIST of field names (not a value
    # dict like the legacy 21-feat format). When that's the case we
    # cannot compute a per-field diff. In full/buy mode this is missing
    # contract evidence; sell-only is allowed for risk exits.
    if not isinstance(stored_sub, dict):
        return _soft_for_sell_only(
            "P-CONFIG-FP",
            f"fingerprint_fields stored as {type(stored_sub).__name__}, "
            f"not dict — can't diff. live={live_fp} stored={stored}",
            run_mode=run_mode,
        )
    diff_keys = []
    live_sub = _model_relevant_fields(config)
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
        "run_mode": run_mode,
    }
    normalized_mode = str(run_mode or "").lower().replace("_", "-")
    if normalized_mode.startswith("sell-only"):
        return PreflightCheck(
            "P-CONFIG-FP", "soft", True,
            msg + " Sell-only risk exits are allowed; new buys remain blocked "
            "until the artifact is retrained/stamped against the live config.",
            details=details,
        )
    legacy_sector_fields = {"sector_map", "sector_etf_map"}
    if (
        diff_keys
        and set(diff_keys).issubset(legacy_sector_fields)
        and not any(k in stored_sub for k in legacy_sector_fields)
    ):
        sector_check = _check_sector_map_coverage(config, strategy_dir, run_mode)
        details = details | {
            "legacy_missing_sector_fields": True,
            "sector_coverage_ok": sector_check.ok,
            "sector_coverage_severity": sector_check.severity,
            "sector_coverage_message": sector_check.message,
            "sector_coverage_details": sector_check.details,
        }
        if sector_check.ok:
            return _soft_for_sell_only(
                "P-CONFIG-FP",
                msg + " Legacy artifact lacks sector fingerprint fields added "
                "after training. Full/buy runs require retrain/stamp even when "
                "P-SECTOR-MAP coverage is currently OK.",
                run_mode=run_mode,
                details=details,
            )
        return PreflightCheck(
            "P-CONFIG-FP", "hard", False,
            msg + " Legacy artifact lacks sector fingerprint fields, and "
            "P-SECTOR-MAP did not pass; fix sector metadata or retrain/stamp "
            "before enabling buy mode.",
            details=details,
        )
    return PreflightCheck("P-CONFIG-FP", "hard", False, msg, details=details)


def _check_watchlist_size(config: dict, strategy_dir: Path) -> PreflightCheck:
    """P-WATCHLIST: config watchlist length consistent with training."""
    wl = config.get("watchlist") or []
    panel_cfg = _active_panel_config(config)
    kind = _active_panel_kind(config, panel_cfg)
    rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = _resolve_artifact_path(strategy_dir, rel)
    if not p.exists():
        return PreflightCheck(
            "P-WATCHLIST", "hard", False, f"artifact missing: {p}",
        )
    if _is_sequence_artifact(kind, p):
        try:
            meta, _sidecar = _load_sequence_sidecar(p)
        except Exception as exc:
            return PreflightCheck(
                "P-WATCHLIST", "soft", True,
                f"{kind} summary unavailable for watchlist check: {exc}; "
                f"live={len(wl)} ticker(s)",
            )
        fields = meta.get("config_fingerprint_fields") or {}
        if not isinstance(fields, dict):
            fields = {}
        trained_wl = fields.get("watchlist") or []
        if trained_wl:
            if set(wl) != set(trained_wl):
                only_live = sorted(set(wl) - set(trained_wl))[:5]
                only_trained = sorted(set(trained_wl) - set(wl))[:5]
                return PreflightCheck(
                    "P-WATCHLIST", "hard", False,
                    f"watchlist mismatch live={len(wl)} trained={len(trained_wl)} "
                    f"in_live_not_trained={only_live} "
                    f"in_trained_not_live={only_trained}",
                )
            return PreflightCheck(
                "P-WATCHLIST", "hard", True,
                f"watchlist match (n={len(wl)})",
            )
        return PreflightCheck(
            "P-WATCHLIST", "soft", True,
            f"trained watchlist not stamped for sequence artifact; live={len(wl)} ticker(s)",
        )
    try:
        meta = json.loads(p.read_text())
    except Exception as exc:
        return PreflightCheck(
            "P-WATCHLIST", "hard", False, f"unreadable: {exc}",
        )
    fields = meta.get("config_fingerprint_fields") or {}
    # Defensive: alpha158_fund artifact may store fingerprint_fields as a
    # LIST of field names (no values). Treat as not-stamped.
    if not isinstance(fields, dict):
        fields = {}
    trained_wl = fields.get("watchlist") or []
    if not trained_wl:
        return PreflightCheck(
            "P-WATCHLIST", "soft", True,
            f"trained watchlist not stamped; live={len(wl)} ticker(s)",
        )
    if set(wl) != set(trained_wl):
        only_live = sorted(set(wl) - set(trained_wl))[:5]
        only_trained = sorted(set(trained_wl) - set(wl))[:5]
        return PreflightCheck(
            "P-WATCHLIST", "hard", False,
            f"watchlist mismatch live={len(wl)} trained={len(trained_wl)} "
            f"in_live_not_trained={only_live} in_trained_not_live={only_trained}",
        )
    return PreflightCheck(
        "P-WATCHLIST", "hard", True,
        f"watchlist match (n={len(wl)})",
    )


def _check_sector_map_coverage(
    config: dict,
    strategy_dir: Path,
    run_mode: str | None = None,
) -> PreflightCheck:
    """P-SECTOR-MAP: every buyable ticker must have sector metadata.

    Panel-LTR inference uses sector data in three places: sector-neutralized
    features, relative strength vs sector ETF, and QP sector caps. Missing
    entries are not benign; they silently turn a stock into "no sector" and
    let it avoid sector-aware controls. Sell-only runs are allowed because
    this check protects new entries, not risk exits.
    """
    panel_enabled = bool(
        config.get("ranking", {})
        .get("panel_scoring", {})
        .get("enabled", False)
    )
    required = bool(
        config.get("risk", {}).get(
            "require_sector_map_for_buys",
            panel_enabled,
        )
    )
    if not required:
        return PreflightCheck(
            "P-SECTOR-MAP", "soft", True,
            "sector-map coverage not required for this strategy/config",
        )

    normalized_mode = str(run_mode or "").lower().replace("_", "-")
    watchlist = list(config.get("watchlist") or [])
    sector_map = config.get("sector_map", {}) or {}
    benchmark = config.get("benchmark", "SPY")
    buyable = [t for t in watchlist if t != benchmark]
    missing = sorted(
        t for t in buyable
        if not isinstance(sector_map.get(t), str) or not sector_map.get(t)
    )
    sectors = sorted({v for v in sector_map.values() if isinstance(v, str) and v})
    sector_etfs = config.get("sector_etf_map", {}) or {}
    unmapped_sectors = sorted(s for s in sectors if s not in sector_etfs)
    details = {
        "watchlist_size": len(watchlist),
        "buyable_size": len(buyable),
        "missing_count": len(missing),
        "missing_sample": missing[:20],
        "unmapped_sectors": unmapped_sectors[:20],
        "run_mode": run_mode,
    }
    if missing or unmapped_sectors:
        msg = (
            f"sector metadata incomplete: {len(missing)}/{len(buyable)} "
            f"buyable watchlist tickers missing sector_map entries "
            f"(sample={missing[:10]}), {len(unmapped_sectors)} sector(s) "
            f"missing sector_etf_map entries (sample={unmapped_sectors[:10]}). "
            "Missing sector metadata disables relative-strength context and "
            "QP sector caps for those names."
        )
        if normalized_mode.startswith("sell-only"):
            return PreflightCheck(
                "P-SECTOR-MAP", "soft", True,
                msg + " Sell-only risk exits are allowed; new buys remain blocked.",
                details=details,
            )
        return PreflightCheck("P-SECTOR-MAP", "hard", False, msg, details=details)

    return PreflightCheck(
        "P-SECTOR-MAP", "hard", True,
        f"sector coverage OK ({len(buyable)} buyable tickers, "
        f"{len(sectors)} sectors mapped)",
        details=details,
    )


def _correlation_artifact_path(config: dict, strategy_dir: Path) -> Path:
    regime_cfg = config.get("regime", {}) or {}
    rel = regime_cfg.get("correlation_artifact", "prod/watchlist-correlation.json")
    p = Path(str(rel))
    if p.is_absolute():
        return p
    return strategy_dir / "artifacts" / p


def _check_correlation_artifact_metadata(
    config: dict,
    strategy_dir: Path,
    run_mode: str | None = None,
) -> PreflightCheck:
    """P-CORR-METADATA: buy/full runs require stamped correlation metadata.

    Live may use the freshest correlation matrix, so this preflight check
    does not compare `as_of_date` with `backtest_start`. It ensures the same
    artifact can prove its data window when strict sims or LEAN acceptance
    consume it.
    """
    p = _correlation_artifact_path(config, strategy_dir)
    details = {"path": str(p)}
    if not p.exists():
        return _soft_for_sell_only(
            "P-CORR-METADATA",
            f"correlation artifact missing at {p}",
            run_mode=run_mode,
            details=details,
        )
    try:
        raw = json.loads(p.read_text())
        from kernel.walk_forward import parse_correlation_artifact  # noqa: PLC0415
        matrix, as_of = parse_correlation_artifact(raw)
    except Exception as exc:
        return _soft_for_sell_only(
            "P-CORR-METADATA",
            f"correlation artifact unreadable at {p}: {exc}",
            run_mode=run_mode,
            details=details,
        )

    details.update({"as_of_date": as_of, "n_tickers": len(matrix)})
    if as_of is None:
        legacy_allowed = bool(
            (config.get("regime", {}) or {})
            .get("allow_legacy_correlation_without_as_of", False)
        )
        if legacy_allowed:
            return PreflightCheck(
                "P-CORR-METADATA", "soft", True,
                "correlation artifact missing as_of_date; explicit legacy override enabled",
                details=details,
            )
        return _soft_for_sell_only(
            "P-CORR-METADATA",
            f"correlation artifact missing as_of_date at {p}; strict full/buy "
            "preflight fails closed because leakage cannot be verified",
            run_mode=run_mode,
            details=details,
        )

    try:
        from kernel.walk_forward.leakage_guard import _to_timestamp  # noqa: PLC0415
        _to_timestamp(as_of, label="correlation as_of_date")
    except Exception as exc:
        return _soft_for_sell_only(
            "P-CORR-METADATA",
            f"correlation artifact has invalid as_of_date={as_of!r}: {exc}",
            run_mode=run_mode,
            details=details,
        )
    return PreflightCheck(
        "P-CORR-METADATA", "hard", True,
        f"correlation artifact stamped as_of_date={as_of} ({len(matrix)} tickers)",
        details=details,
    )


def _check_feature_coverage(
    config: dict, strategy_dir: Path,
    feature_drift_pct: float = 0.05,
    run_mode: str | None = None,
) -> PreflightCheck:
    """P-FEATURE-COVER: NGBoost head's feature_cols are present.

    This is a STATIC check on artifact metadata — checks that the
    NGBoost head and the panel-LTR scorer agree on feature_cols. The
    actual runtime drift detector in ApplyNGBoostTask catches the
    dynamic case.
    """
    panel_cfg = config.get("panel_ltr", {})
    panel_rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")

    # 2026-05-17 Bug fix: a per-regime overlay can activate NGB even when
    # the global flag is False. Catch that at preflight instead of waiting
    # for runtime drift guards to clear every candidate.
    ngb_cfg, per_regime_activates, ngb_potentially_active = _ngboost_activation(config)
    if not ngb_potentially_active:
        return PreflightCheck(
            "P-FEATURE-COVER", "soft", True,
            "NGBoost disabled globally + no per-regime overlay activates — skip",
        )
    ngb_rel = ngb_cfg.get("artifact_path")
    if not ngb_rel:
        return _soft_for_sell_only(
            "P-FEATURE-COVER",
            "NGBoost can activate but ranking.panel_scoring.ngboost.artifact_path "
            f"is missing (per_regime={per_regime_activates}); full/buy cannot "
            "silently fall back to panel-only scoring",
            run_mode=run_mode,
            details={"per_regime_activates": per_regime_activates},
        )

    panel_p = _resolve_artifact_path(strategy_dir, panel_rel)
    ngb_p   = _resolve_artifact_path(strategy_dir, ngb_rel)
    if not panel_p.exists() or not ngb_p.exists():
        return _soft_for_sell_only(
            "P-FEATURE-COVER",
            f"artifact missing: panel={panel_p.exists()} ngb={ngb_p.exists()}",
            run_mode=run_mode,
            details={"panel_path": str(panel_p), "ngboost_path": str(ngb_p)},
        )
    try:
        panel_meta = json.loads(panel_p.read_text())
        ngb_meta   = json.loads(ngb_p.read_text())
    except Exception as exc:
        return _soft_for_sell_only(
            "P-FEATURE-COVER",
            f"unreadable: {exc}",
            run_mode=run_mode,
            details={"panel_path": str(panel_p), "ngboost_path": str(ngb_p)},
        )
    panel_feats = set(panel_meta.get("feature_cols") or [])
    ngb_feats   = set(ngb_meta.get("feature_cols")   or [])
    if not ngb_feats:
        return _soft_for_sell_only(
            "P-FEATURE-COVER",
            "NGBoost feature_cols not stamped; full/buy cannot validate "
            "runtime feature parity",
            run_mode=run_mode,
            details={"ngboost_path": str(ngb_p)},
        )
    missing = ngb_feats - panel_feats
    pct = len(missing) / max(1, len(ngb_feats))
    feature_drift_pct = float(
        ngb_cfg.get("max_feature_drift_pct", feature_drift_pct)
    )
    allow_partial = bool(ngb_cfg.get("allow_partial_feature_fill", False))
    if missing and (not allow_partial or pct > feature_drift_pct):
        policy = (
            "partial fill disabled"
            if not allow_partial else
            f"missing_pct={pct:.1%} > max_feature_drift_pct={feature_drift_pct:.1%}"
        )
        return _soft_for_sell_only(
            "P-FEATURE-COVER",
            f"NGBoost expects {len(ngb_feats)} feats, "
            f"{len(missing)} ({pct:.1%}) missing from panel — "
            f"{policy}; retrain NGBoost head against current panel pipeline. "
            f"First 5 missing: {sorted(missing)[:5]}",
            run_mode=run_mode,
            details={"missing_count": len(missing),
                     "missing_pct": pct,
                     "first_missing": sorted(missing)[:10],
                     "allow_partial_feature_fill": allow_partial},
        )
    return PreflightCheck(
        "P-FEATURE-COVER", "hard", True,
        f"NGBoost feature coverage OK ({len(ngb_feats)} feats, "
        f"{len(missing)} missing = {pct:.1%})",
    )


def _check_state_file(
    config: dict, strategy_dir: Path, broker_name: str | None,
) -> PreflightCheck:
    """P-STATE-FILE: live_state.{broker}.json parses (or absent)."""
    if not broker_name:
        return PreflightCheck(
            "P-STATE-FILE", "soft", True, "no broker_name (dry-run); skip",
        )
    try:
        from kernel.state_paths import resolve_live_state_read  # noqa: PLC0415
    except Exception as exc:
        return PreflightCheck(
            "P-STATE-FILE", "soft", True,
            f"state_paths unavailable: {exc}; skip",
        )
    p, _used_legacy = resolve_live_state_read(strategy_dir, broker_name)
    if not p.exists():
        return PreflightCheck(
            "P-STATE-FILE", "soft", True,
            f"state file absent at {p.name} (first run?)",
        )
    try:
        json.loads(p.read_text())
    except Exception as exc:
        return PreflightCheck(
            "P-STATE-FILE", "hard", False,
            f"state file unreadable {p.name}: {exc}",
        )
    return PreflightCheck(
        "P-STATE-FILE", "hard", True, f"loaded {p.name}",
    )


def _check_broker_connect(broker: Any) -> PreflightCheck:
    """P-BROKER-CONNECT: connect + get_account_value works."""
    if broker is None:
        return PreflightCheck(
            "P-BROKER-CONNECT", "soft", True,
            "no broker (dry-run); skip",
        )
    try:
        broker.connect()
        eq = float(broker.get_account_value())
        return PreflightCheck(
            "P-BROKER-CONNECT", "hard", True,
            f"broker connected, equity=${eq:.2f}",
        )
    except Exception as exc:
        return PreflightCheck(
            "P-BROKER-CONNECT", "hard", False,
            f"broker connect failed: {exc}",
        )


def _check_artifact_run_id_alignment(
    config: dict, strategy_dir: Path, run_mode: str | None = None,
) -> PreflightCheck:
    """P-RUN-ID: panel-ltr and ngboost-head share the same train_run_id.

    External audit fix #2 (2026-04-29): without run_id, one artifact can
    silently come from a different training run (e.g. a side-config retrain
    overwriting production NGBoost). A mismatch means μ/σ was fit on a
    different panel feature distribution than the scorer — Kelly sizing
    corrupted. Hard for full/buy when NGBoost can activate; sell-only may
    still run risk exits without the μ/σ head.
    """
    panel_cfg  = config.get("panel_ltr", {})
    ltr_rel    = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    ngb_cfg, per_regime_activates, ngb_potentially_active = _ngboost_activation(config)
    if not ngb_potentially_active:
        return PreflightCheck(
            "P-RUN-ID", "soft", True,
            "NGBoost disabled globally + no per-regime overlay activates — skip",
        )
    ngb_rel = ngb_cfg.get("artifact_path")
    if not ngb_rel:
        return _soft_for_sell_only(
            "P-RUN-ID",
            "NGBoost can activate but ranking.panel_scoring.ngboost.artifact_path "
            f"is missing (per_regime={per_regime_activates}); cannot verify "
            "panel/NGBoost train_run_id alignment",
            run_mode=run_mode,
            details={"per_regime_activates": per_regime_activates},
        )
    ltr_path   = _resolve_artifact_path(strategy_dir, ltr_rel)
    ngb_path   = _resolve_artifact_path(strategy_dir, ngb_rel)
    for p in (ltr_path, ngb_path):
        if not p.exists():
            return _soft_for_sell_only(
                "P-RUN-ID",
                f"artifact missing: {p}; cannot verify panel/NGBoost "
                "train_run_id alignment",
                run_mode=run_mode,
                details={"panel_path": str(ltr_path), "ngboost_path": str(ngb_path)},
            )
    try:
        ltr_id = json.loads(ltr_path.read_text()).get("train_run_id")
        ngb_id = json.loads(ngb_path.read_text()).get("train_run_id")
    except Exception as exc:
        return _soft_for_sell_only(
            "P-RUN-ID",
            f"unreadable: {exc}",
            run_mode=run_mode,
            details={"panel_path": str(ltr_path), "ngboost_path": str(ngb_path)},
        )
    if ltr_id is None or ngb_id is None:
        return _soft_for_sell_only(
            "P-RUN-ID",
            "run_id not stamped on panel or NGBoost artifact; full/buy "
            "cannot mix unstamped μ/σ with panel scores",
            run_mode=run_mode,
            details={"panel_train_run_id": ltr_id, "ngboost_train_run_id": ngb_id},
        )
    if ltr_id != ngb_id:
        return _soft_for_sell_only(
            "P-RUN-ID",
            f"run_id mismatch: panel-ltr={ltr_id} ngboost={ngb_id}. "
            f"NGBoost μ/σ may be from a different training run — Kelly "
            f"sizing potentially corrupted. Retrain recommended.",
            run_mode=run_mode,
            details={"panel_train_run_id": ltr_id, "ngboost_train_run_id": ngb_id},
        )
    return PreflightCheck(
        "P-RUN-ID", "hard", True,
        f"run_id aligned ({ltr_id})",
    )


def _check_meta_label_artifact_contract(
    config: dict,
    strategy_dir: Path,
    run_mode: str | None = None,
) -> PreflightCheck:
    """P-META-LABEL: enabled exit-veto path requires a usable artifact.

    Meta-label is an optional path-rule exit veto. When enabled for buy/full
    runs, a missing/corrupt artifact would silently turn the decision tree back
    into the un-vetoed stop path, exactly the false-positive stop-loss class we
    are auditing. Sell-only runs are allowed to keep raw risk exits armed.
    """
    cfg = ((config.get("ranking") or {}).get("meta_label") or {})
    if not bool(cfg.get("enabled", False)):
        return PreflightCheck(
            "P-META-LABEL", "soft", True,
            "ranking.meta_label disabled; artifact contract not applicable",
        )
    rel = cfg.get("artifact_path")
    if not rel:
        return _soft_for_sell_only(
            "P-META-LABEL",
            "ranking.meta_label.enabled=true but artifact_path is missing; "
            "full/buy cannot silently fall back to un-vetoed path exits",
            run_mode=run_mode,
        )
    p = _resolve_artifact_path(strategy_dir, rel)
    if not p.exists():
        return _soft_for_sell_only(
            "P-META-LABEL",
            f"ranking.meta_label.enabled=true but artifact missing at {p}; "
            "full/buy cannot silently fall back to un-vetoed path exits",
            run_mode=run_mode,
        )
    try:
        payload = json.loads(p.read_text())
    except Exception as exc:
        return _soft_for_sell_only(
            "P-META-LABEL",
            f"meta-label artifact unreadable at {p}: {exc}",
            run_mode=run_mode,
        )

    errors: list[str] = []
    if payload.get("kind") != "meta_label_exit_xgb":
        errors.append(f"kind={payload.get('kind')!r} != 'meta_label_exit_xgb'")
    feature_cols = payload.get("feature_cols") or []
    if not isinstance(feature_cols, list) or not feature_cols:
        errors.append("feature_cols missing/empty")
    if not isinstance(payload.get("booster_raw_json"), str) or not payload.get("booster_raw_json"):
        errors.append("booster_raw_json missing/empty")
    default_threshold = _finite_float(payload.get("default_threshold"))
    cfg_threshold = _finite_float(cfg.get("threshold", default_threshold))
    if default_threshold is None:
        errors.append("default_threshold missing/non-finite")
    if cfg_threshold is None or not (0.0 <= cfg_threshold <= 1.0):
        errors.append(f"config threshold invalid: {cfg.get('threshold')!r}")

    cv = payload.get("cv_metrics") or {}
    auc = _finite_float(cv.get("auc_mean"))
    min_auc = _finite_float(cfg.get("min_auc", 0.50))
    if auc is None:
        errors.append("cv_metrics.auc_mean missing/non-finite")
    elif min_auc is not None and auc < min_auc:
        errors.append(f"cv_metrics.auc_mean={auc:.4f} < min_auc={min_auc:.4f}")

    summary = payload.get("training_data_summary") or {}
    n_events = summary.get("n_events")
    try:
        n_events_i = int(n_events)
    except (TypeError, ValueError):
        n_events_i = 0
    min_events = int(cfg.get("min_events", 100))
    if n_events_i < min_events:
        errors.append(f"training_data_summary.n_events={n_events_i} < min_events={min_events}")
    fwd_window = summary.get("fwd_window_days")
    try:
        fwd_window_i = int(fwd_window)
    except (TypeError, ValueError):
        fwd_window_i = 0
    if fwd_window_i <= 0:
        errors.append("training_data_summary.fwd_window_days missing/non-positive")
    class_balance = _finite_float(summary.get("class_balance"))
    if class_balance is None or not (0.0 < class_balance < 1.0):
        errors.append("training_data_summary.class_balance missing/out-of-range")

    if errors:
        return _soft_for_sell_only(
            "P-META-LABEL",
            "meta-label artifact contract failed: " + "; ".join(errors),
            run_mode=run_mode,
            details={"artifact_path": str(p), "errors": errors},
        )
    return PreflightCheck(
        "P-META-LABEL", "hard", True,
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


# ── Orchestrator ───────────────────────────────────────────────────────────

ALL_CHECKS = (
    _check_model_artifact,
    _check_panel_artifact_contract,
    _check_wf_gate_metadata,
    _check_regime_layered_ic,
    _check_best_iter,
    _check_config_fingerprint,
    _check_watchlist_size,
    _check_sector_map_coverage,
    _check_correlation_artifact_metadata,
    _check_feature_coverage,
    _check_state_file,
    _check_broker_connect,
    _check_artifact_run_id_alignment,  # audit fix #2 — soft check
    _check_meta_label_artifact_contract,
    None,  # _check_calibrator_health — registered below to keep ALL_CHECKS readable
)


def _check_calibrator_health(
    config: dict,
    strategy_dir: Path,
    run_mode: str | None = None,
) -> "PreflightCheck":
    """P-CALIBRATOR-HEALTH (2026-05-05 parity fix): runtime equivalent of the
    training-side `fit_global_calibrator` "probability head collapsed to N
    unique values" guard.

    Today's diagnostic (2026-05-04 e2e) found `n_unique_prob_y = 7` in the
    production calibrator: only 7 distinct calibrated probabilities across
    235K training rows. Result at runtime: top 10 candidates all tied at
    rank_score=0.2579, no real ranking → strategy can't differentiate. The
    training-time guard catches this AT FIT but the artifact had been saved
    BEFORE that guard was added; runtime had no way to detect the
    degradation. This check closes that gap.

    Hard-fail when:
      * artifact missing or unparseable
      * `n_unique_prob_y < min_unique_prob_y` (default 10)
    Soft-warn when:
      * `pool_ic <= 0` (calibrator anti-correlated with labels)

    Tunable via config.panel_ltr.calibrator_health.min_unique_prob_y.
    Backwards-compat: pre-2026-05 artifacts without n_unique_prob_y in
    metadata get a soft skip (can't verify, log a warning).
    """
    panel_cfg = config.get("panel_ltr", {})
    cal_cfg = ((config.get("ranking", {})
                       .get("panel_scoring", {})
                       .get("global_calibration", {})) or {})
    rel = cal_cfg.get("artifact_path", "artifacts/prod/panel-rank-calibration.json")
    p = strategy_dir / rel
    calibration_enabled = _is_global_calibration_enabled(config)
    if not calibration_enabled:
        return PreflightCheck(
            "P-CALIBRATOR-HEALTH", "soft", True,
            "global_calibration disabled; health gate not applicable",
        )
    if not p.exists():
        return _soft_for_sell_only(
            "P-CALIBRATOR-HEALTH",
            f"global_calibration.enabled=true but calibrator artifact absent at {p}",
            run_mode=run_mode,
        )
    try:
        payload = json.loads(p.read_text())
        meta = payload.get("metadata", {}) or {}
    except Exception as exc:
        return PreflightCheck(
            "P-CALIBRATOR-HEALTH", "hard", False, f"unreadable: {exc}",
        )
    n_unique = meta.get("n_unique_prob_y")
    pool_ic = meta.get("pool_ic")

    kelly_cfg = (config.get("ranking", {}) or {}).get("kelly_sizing", {}) or {}
    if bool(kelly_cfg.get("use_calibrator_mu", False)):
        er_contract = meta.get("expected_return_label_contract")
        if er_contract != "raw_return_units_required":
            return _soft_for_sell_only(
                "P-CALIBRATOR-HEALTH",
                "ranking.kelly_sizing.use_calibrator_mu=true but calibrator "
                f"expected_return_label_contract={er_contract!r}; QP/Kelly "
                "would consume a non-return label as expected-return μ. "
                "Refit calibrator with raw return labels before buy/full.",
                run_mode=run_mode,
                details={
                    "expected_return_label_contract": er_contract,
                    "required_contract": "raw_return_units_required",
                    "use_calibrator_mu": True,
                    "er_std": meta.get("er_std"),
                },
            )

    # 2026-05-15 P0 ADDITION: range-bound check on expected_return.y.
    # Catches the bug class that caused the rank_score saturation incident:
    # calibrator artifacts with er.y up to +1.00 (= +100% expected return)
    # would feed catastrophically wrong μ into Kelly when use_calibrator_mu=
    # true. Hard-fail before live trade so a bad artifact never reaches QP.
    # Threshold matches the train-site clip (2026-05-15 Phase 4 commit).
    try:
        er_y = payload.get("expected_return", {}).get("y", []) or []
        er_x = payload.get("expected_return", {}).get("x", []) or []
        if er_y:
            er_max_abs = max(abs(float(v)) for v in er_y
                             if v is not None and v == v)  # NaN-safe
            ER_BOUND = 0.20  # matches GlobalPanelCalibration.load() clip
            if er_max_abs > ER_BOUND + 1e-9:
                return PreflightCheck(
                    "P-CALIBRATOR-HEALTH", "hard", False,
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
                config.get("panel_ltr", {})
                .get("calibrator_health", {})
                .get("max_expected_return_flat_fraction", 0.30)
            )
            if er_flat["fraction"] > max_er_flat:
                return PreflightCheck(
                    "P-CALIBRATOR-HEALTH", "hard", False,
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
    health_cfg = panel_cfg.get("calibrator_health", {}) or {}
    min_unique = int(health_cfg.get("min_unique_prob_y", 10))
    if n_unique is None:
        return _soft_for_sell_only(
            "P-CALIBRATOR-HEALTH",
            "n_unique_prob_y not stamped; cannot verify probability-head granularity",
            run_mode=run_mode,
            details={"pool_ic": pool_ic, "global_calibration_enabled": calibration_enabled},
        )
    if int(n_unique) < min_unique:
        return _soft_for_sell_only(
            "P-CALIBRATOR-HEALTH",
            f"n_unique_prob_y={n_unique} < min_unique_prob_y={min_unique}; "
            "calibrator probability head collapsed and buy ranking is ineffective",
            run_mode=run_mode,
            details={"n_unique_prob_y": n_unique, "min_unique_prob_y": min_unique,
                     "pool_ic": pool_ic},
        )
    if pool_ic is not None and float(pool_ic) <= 0:
        return _soft_for_sell_only(
            "P-CALIBRATOR-HEALTH",
            f"pool_ic={pool_ic} <= 0; calibrator anti-correlated with labels",
            run_mode=run_mode,
            details={"n_unique_prob_y": n_unique, "pool_ic": pool_ic},
        )
    return PreflightCheck(
        "P-CALIBRATOR-HEALTH", "hard", True,
        f"n_unique_prob_y={n_unique} ≥ {min_unique}, pool_ic={pool_ic}",
        details={"n_unique_prob_y": n_unique, "pool_ic": pool_ic},
    )


def _check_calibrator_flat_region(
    config: dict,
    strategy_dir: Path,
    run_mode: str | None = None,
) -> "PreflightCheck":
    """P-CALIBRATOR-FLAT-REGION (2026-05-18, MCD-rebuy incident):
    structural check that calibrator's probability curve has no
    flat region wider than `max_flat_fraction` of the x-domain.

    Why: isotonic regression can create wide flat regions where the
    underlying signal is weak (e.g. low scores don't reliably predict
    low returns). Those flat regions tie up to 79% of candidates at
    one probability → ranking degenerates → top-K is tie-broken by
    panel_score / ticker order → MCD-style rebuy ensues.

    Hard-fail when the largest flat segment spans > max_flat_fraction
    (default 0.30 = 30% of x-domain). Operator can override via
    config.panel_ltr.calibrator_health.max_flat_fraction.

    Reference: doc/research/2026-05-18-mcd-rebuy-incident.md
    """
    panel_cfg = config.get("panel_ltr", {})
    cal_cfg = ((config.get("ranking", {})
                       .get("panel_scoring", {})
                       .get("global_calibration", {})) or {})
    rel = (
        cal_cfg.get("artifact_path")
        or panel_cfg.get("calibrator_artifact_path")
        or "artifacts/prod/panel-rank-calibration.json"
    )
    p = _resolve_artifact_path(strategy_dir, rel)
    calibration_enabled = _is_global_calibration_enabled(config)
    if not calibration_enabled:
        return PreflightCheck(
            "P-CALIBRATOR-FLAT-REGION", "soft", True,
            "global_calibration disabled; flat-region gate not applicable",
        )
    if not p.exists():
        return _soft_for_sell_only(
            "P-CALIBRATOR-FLAT-REGION",
            f"global_calibration.enabled=true but calibrator artifact missing at {p}",
            run_mode=run_mode,
        )
    try:
        cal = json.loads(p.read_text())
        pr = cal.get("probability", {})
        x = pr.get("x", [])
        y = pr.get("y", [])
        if not x or not y or len(x) != len(y):
            return _soft_for_sell_only(
                "P-CALIBRATOR-FLAT-REGION",
                "probability.x/y missing or mismatched",
                run_mode=run_mode,
            )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return _soft_for_sell_only(
            "P-CALIBRATOR-FLAT-REGION",
            f"could not parse calibrator: {exc}",
            run_mode=run_mode,
        )

    health_cfg = panel_cfg.get("calibrator_health", {}) or {}
    max_flat_fraction = float(health_cfg.get("max_flat_fraction", 0.30))

    # 2026-05-18 user audit: DRY — extracted to kernel/calibrator_quality.py
    # to prevent silent drift between preflight + fit_script + test impls.
    from kernel.calibrator_quality import flat_region_stats  # noqa: PLC0415
    stats = flat_region_stats(x, y)
    flat_frac = stats["fraction"]
    if flat_frac > max_flat_fraction:
        return PreflightCheck(
            "P-CALIBRATOR-FLAT-REGION", "hard", False,
            f"calibrator has flat region spanning {flat_frac*100:.1f}% of "
            f"x-domain (>{max_flat_fraction*100:.0f}%). All μ̂ in that region "
            f"map to one probability → ranking degenerates → tie-broken buys "
            f"(MCD-rebuy class). Refit with method=platt or shrink flat region. "
            f"See doc/research/2026-05-18-mcd-rebuy-incident.md.",
            details={"longest_flat_span": stats["longest_span"],
                     "x_total": stats["x_total"],
                     "flat_fraction": flat_frac,
                     "max_flat_fraction": max_flat_fraction,
                     "calibrator_kind": cal.get("kind", "unknown")},
        )
    return PreflightCheck(
        "P-CALIBRATOR-FLAT-REGION", "hard", True,
        f"largest flat region {flat_frac*100:.1f}% ≤ {max_flat_fraction*100:.0f}% "
        f"of x-domain (n_knots={len(x)})",
        details={"flat_fraction": flat_frac, "max_flat_fraction": max_flat_fraction},
    )


# Replace the placeholder in ALL_CHECKS with the actual function.
# Also append the flat-region check.
ALL_CHECKS = tuple(c if c is not None else _check_calibrator_health for c in ALL_CHECKS)
ALL_CHECKS = ALL_CHECKS + (_check_calibrator_flat_region,)


# Canonical ALL_CHECKS ordering — preserves legacy run_preflight result order
# even though the new PreflightPipeline groups Tasks semantically (so Task
# execution order differs from this).
_LEGACY_CHECK_ORDER: tuple[str, ...] = (
    "P-MODEL-ARTIFACT",
    "P-PANEL-CONTRACT",
    "P-WF-GATE",
    "P-REGIME-IC",
    "P-BEST-ITER",
    "P-CONFIG-FP",
    "P-WATCHLIST",
    "P-SECTOR-MAP",
    "P-CORR-METADATA",
    "P-FEATURE-COVER",
    "P-STATE-FILE",
    "P-BROKER-CONNECT",
    "P-RUN-ID",
    "P-META-LABEL",
    "P-CALIBRATOR-HEALTH",
    "P-CALIBRATOR-FLAT-REGION",
)


def run_preflight(
    config: dict,
    broker: Any = None,
    strategy_dir: Path | str | None = None,
    broker_name: str | None = None,
    *,
    strict: bool = True,
    run_mode: str | None = None,
) -> list[PreflightCheck]:
    """Run all preflight checks via the new PreflightPipeline.

    Track H follow-up (2026-05-30): this function is now a thin wrapper
    around ``kernel.preflight_pipeline.build_preflight_pipeline()``. The
    16 ``_check_*`` legacy functions are kept in this module as the
    implementation that the new Tasks bridge to — deleting them is a
    later cleanup step.

    Contract preserved exactly:
      • Returns ``list[PreflightCheck]`` in ``_LEGACY_CHECK_ORDER`` (the
        original ALL_CHECKS tuple order), NOT the Job/Task declaration
        order from the new pipeline. Operator log markers, however, appear
        in pipeline (semantic) order during execution — cosmetic-only
        change for log readers.
      • Raises ``PreflightFailed`` if any HARD check fails and
        ``strict=True``.
      • The PreflightFailed.failures list is ordered to match
        ``_LEGACY_CHECK_ORDER``.
    """
    if strategy_dir is None:
        raise ValueError("run_preflight requires strategy_dir")
    sd = Path(strategy_dir)
    if broker is not None and broker_name is None:
        broker_name = getattr(broker, "broker_name", None)
    effective_run_mode = run_mode or config.get("_run_mode")

    # Deferred import — avoids circular import (preflight_pipeline imports
    # from this module via bridge).
    from kernel.preflight_pipeline import (  # noqa: PLC0415
        PreflightContext,
        build_preflight_pipeline,
    )
    ctx = PreflightContext(
        config=config,
        strategy_dir=sd,
        broker=broker,
        broker_name=broker_name,
        run_mode=effective_run_mode,
    )
    # Pipeline.run logs each marker inside PreflightTask.run(); we still
    # want strict-mode hard-fail to raise PreflightFailed AFTER we sort
    # to legacy order, so request strict=False here and re-evaluate below.
    pipeline = build_preflight_pipeline()
    results = pipeline.run(ctx, strict=False)
    results.sort(key=lambda r: _LEGACY_CHECK_ORDER.index(r.name)
                 if r.name in _LEGACY_CHECK_ORDER else len(_LEGACY_CHECK_ORDER))
    hard_failures = [r for r in results if r.severity == "hard" and not r.ok]
    if hard_failures and strict:
        raise PreflightFailed(hard_failures)
    return results
