#!/usr/bin/env python
"""Build production-semantic walk-forward eval configs.

Walk-forward configs need a few evaluation-only paths (manifest, static
placeholder artifact, calibration artifact), but their decision semantics must
match production exactly. Hand-edited side configs drifted on buy floors,
tax-lot method, sector maps, and regime params; this builder makes the allowed
differences explicit.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any

from wf_config_parity import evaluate_wf_config_parity


REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"

EXPERIMENT_OVERRIDE_PATHS = (
    "rotation.joint_actions.qp_admission_gate.max_sigma",
    "rotation.joint_actions.qp_admission_gate.max_sigma_by_regime",
    "rotation.joint_actions.qp_admission_gate.topup_max_sigma",
    "rotation.joint_actions.qp_admission_gate.topup_max_sigma_by_regime",
    "rotation.joint_actions.qp_admission_gate.min_expected_return",
    "rotation.joint_actions.qp_admission_gate.min_expected_return_by_regime",
    "rotation.joint_actions.qp_admission_gate.min_expected_excess_return",
    "rotation.joint_actions.qp_admission_gate.min_expected_excess_return_by_regime",
    "rotation.joint_actions.qp_admission_gate.topup_min_expected_return",
    "rotation.joint_actions.qp_admission_gate.topup_min_expected_return_by_regime",
    "rotation.joint_actions.qp_admission_gate.topup_min_expected_excess_return",
    "rotation.joint_actions.qp_admission_gate.topup_min_expected_excess_return_by_regime",
)


def _get_path(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_path(obj: dict[str, Any], dotted: str, value: Any) -> None:
    cur: Any = obj
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = cur.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise TypeError(f"cannot set {dotted}: {part} is not a mapping")
        cur = nxt
    cur[parts[-1]] = value


def _first_manifest_artifact(manifest_path: Path) -> str | None:
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("retrains", []) if isinstance(payload, dict) else payload
    if not rows:
        return None
    raw = str((rows[0] or {}).get("artifact_uri") or "")
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        try:
            return str(path.relative_to(STRATEGY_DIR))
        except ValueError:
            return str(path)
    return raw


def _resolve_strategy_path(raw: str | None, strategy_dir: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else strategy_dir / path


def _semantic_overrides_in_base(
    prod_config: dict[str, Any],
    base_wf_config: dict[str, Any],
) -> list[str]:
    """Return explicit experiment overrides that differ from production.

    ``build_wf_config_from_prod`` intentionally starts from production
    semantics. If an operator hands it a side config with an explicit
    experiment knob, silently dropping that knob creates a false A/B result.
    Require a separate opt-in before preserving those semantic differences.
    """
    out: list[str] = []
    for dotted in EXPERIMENT_OVERRIDE_PATHS:
        base_value = _get_path(base_wf_config, dotted)
        if base_value is None:
            continue
        prod_value = _get_path(prod_config, dotted)
        if base_value != prod_value:
            out.append(dotted)
    return out


def build_wf_config_from_prod(
    prod_config: dict[str, Any],
    *,
    manifest_path: str,
    base_wf_config: dict[str, Any] | None = None,
    strategy_dir: Path = STRATEGY_DIR,
    preserve_experiment_overrides: bool = False,
) -> dict[str, Any]:
    """Return a WF eval config with production decision semantics.

    Allowed non-semantic differences:
      - ``walkforward`` manifest dispatch.
      - ``ranking.panel_scoring.artifact_path`` placeholder for diagnostics;
        SimAdapter uses the manifest when walkforward is enabled.
      - ``ranking.panel_scoring.global_calibration.*`` artifact paths. These
        are evaluation artifacts and must remain point-in-time / sim-scoped,
        not production full-sample paths.
      - ``regime.gmm_artifact`` and ``regime.correlation_artifact`` because
        historical simulations must use point-in-time regime/risk artifacts
        whose ``as_of_date`` is no later than the backtest start.
      - ``ranking.panel_scoring.regime_admission.enabled`` is disabled only
        for WF acceptance runs because that gate consumes the
        ``wf_gate_metadata`` which this run is generating. Live/preflight still
        require the stamped evidence before buy/QP admission.
      - Shadow-model tracking is disabled because it has no trade-decision
        effect and can make WF gates spend hours in auxiliary PyTorch inference.
      - Label/metadata fields starting with ``_`` or ``__``.
    """
    cfg = copy.deepcopy(prod_config)
    base = base_wf_config or {}
    overrides = _semantic_overrides_in_base(prod_config, base)
    if overrides and not preserve_experiment_overrides:
        joined = ", ".join(overrides)
        raise ValueError(
            "base WF config contains semantic experiment override(s) that "
            "would be dropped by production-semantic derivation: "
            f"{joined}. Re-run with preserve_experiment_overrides=True "
            "or the CLI flag --preserve-experiment-overrides for an "
            "explicit diagnostic/non-promotable A/B run."
        )

    wf_base = copy.deepcopy(base.get("walkforward") or {})
    wf_base.update({
        "enabled": True,
        "manifest_path": manifest_path,
        "fail_on_no_model": bool(wf_base.get("fail_on_no_model", True)),
    })
    cfg["walkforward"] = wf_base

    manifest_abs = _resolve_strategy_path(manifest_path, strategy_dir)
    placeholder = _first_manifest_artifact(manifest_abs) if manifest_abs else None
    if placeholder:
        _set_path(cfg, "ranking.panel_scoring.artifact_path", placeholder)

    # Preserve point-in-time calibration artifact paths from the WF base config.
    # Production calibrators are usually fitted on the full current panel, so
    # copying them into historical WF would create a subtle look-ahead channel.
    for dotted in (
        "ranking.panel_scoring.global_calibration.artifact_path",
        "ranking.panel_scoring.global_calibration.regime_conditional.artifact_pattern",
        "regime.gmm_artifact",
        "regime.correlation_artifact",
    ):
        value = _get_path(base, dotted)
        if value is not None:
            _set_path(cfg, dotted, value)

    if preserve_experiment_overrides:
        preserved: list[str] = []
        for dotted in overrides:
            value = _get_path(base, dotted)
            if value is not None:
                _set_path(cfg, dotted, copy.deepcopy(value))
                preserved.append(dotted)
        if preserved:
            cfg["_experiment_overrides_preserved"] = preserved
            cfg["_experiment_overrides_note"] = (
                "Diagnostic WF A/B only. These semantic differences mean the "
                "generated config is not production-equivalent and cannot be "
                "promotion evidence until production config carries the same "
                "knobs and parity passes."
            )

    panel_cfg = cfg.setdefault("ranking", {}).setdefault("panel_scoring", {})
    regime_admission = panel_cfg.setdefault("regime_admission", {})
    regime_admission["enabled"] = False
    regime_admission["_wf_disabled_reason"] = (
        "WF acceptance generates trade_monotonicity/sanity_regime_ic metadata; "
        "requiring pre-existing wf_gate_metadata here creates a circular "
        "zero-trade evaluation. Live/preflight keep the production fail-closed "
        "admission contract."
    )
    if "shadow_models" in panel_cfg:
        panel_cfg["shadow_models"] = []
        panel_cfg["_wf_shadow_disabled_reason"] = (
            "WF acceptance validates production trade decisions; shadow models "
            "are non-decision diagnostics and are disabled to avoid auxiliary "
            "PyTorch inference stalls."
        )

    for key in ("__label", "_side_config_label", "_generated_note", "_backtest_start_note"):
        if key in base:
            cfg[key] = base[key]
    cfg["_generated_note"] = (
        "Generated by scripts/wf_config_builder.py from production config; "
        "only walkforward/artifact/calibration/regime paths may differ."
    )
    cfg["_generated_at_utc"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prod-config", default=str(STRATEGY_DIR / "strategy_config.json"))
    parser.add_argument("--base-wf-config", required=True,
                        help="Existing WF config to borrow manifest/calibrator paths from.")
    parser.add_argument("--out", required=True,
                        help="Output config path. Relative paths resolve under strategy dir.")
    parser.add_argument("--candidate-artifact", default=None,
                        help="Optional candidate artifact for feature-contract parity.")
    parser.add_argument("--preserve-experiment-overrides", action="store_true",
                        help="Explicitly carry whitelisted diagnostic semantic "
                             "overrides from the base WF config. This is for "
                             "A/B exploration and will normally fail prod/WF "
                             "parity until production config matches.")
    args = parser.parse_args()

    prod_path = Path(args.prod_config)
    if not prod_path.is_absolute():
        prod_path = STRATEGY_DIR / prod_path
    base_path = Path(args.base_wf_config)
    if not base_path.is_absolute():
        base_path = STRATEGY_DIR / base_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = STRATEGY_DIR / out_path

    prod = json.loads(prod_path.read_text())
    base = json.loads(base_path.read_text())
    manifest = ((base.get("walkforward") or {}).get("manifest_path"))
    if not manifest:
        raise SystemExit(f"base WF config has no walkforward.manifest_path: {base_path}")

    cfg = build_wf_config_from_prod(
        prod,
        manifest_path=str(manifest),
        base_wf_config=base,
        strategy_dir=STRATEGY_DIR,
        preserve_experiment_overrides=args.preserve_experiment_overrides,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2, sort_keys=False) + "\n")

    candidate = Path(args.candidate_artifact) if args.candidate_artifact else None
    result = evaluate_wf_config_parity(
        prod_path,
        out_path,
        candidate_artifact=candidate,
        strategy_dir=STRATEGY_DIR,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
