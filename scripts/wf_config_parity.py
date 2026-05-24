#!/usr/bin/env python
"""Validate walk-forward config parity against production config.

The guard is intentionally stricter than model-fingerprint checks. A WF
config may use different artifact paths and a retrain manifest, but it must
not silently change decision semantics such as buy floors, QP knobs, regime
params, tax mode, or scorer kind.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"


SEMANTIC_PATHS = [
    "ranking.panel_scoring.enabled",
    "ranking.panel_scoring.kind",
    "ranking.panel_scoring.buy_floor",
    "ranking.panel_scoring.rotation_advantage",
    "ranking.panel_scoring.sizing",
    "ranking.panel_scoring.sigma_sizing",
    "ranking.panel_scoring.ngboost.enabled",
    "ranking.panel_scoring.ngboost.score_mode",
    "ranking.panel_scoring.ngboost.lambda_sigma",
    "ranking.kelly_sizing",
    "rotation.joint_actions",
    "risk.panel_exit",
    "risk.stop_loss_anchor_policy",
    "regime_params",
    "max_concurrent_positions",
    "max_positions_per_sector",
    "wash_sale_days",
    "execution",
    "tax",
    "defensive_tickers",
    "sector_map",
    "tiered_thresholds",
]

IGNORED_KEYS = {
    "artifact_path",
    "manifest_path",
    "trained_date",
}


def _get_path(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _clean(v)
            for k, v in sorted(value.items())
            if not str(k).startswith("_") and str(k) not in IGNORED_KEYS
        }
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def _normalize_kind(value: Any) -> Any:
    kind = str(value or "").lower()
    aliases = {
        "panel_ltr_xgboost": "xgb",
        "xgboost": "xgb",
    }
    return aliases.get(kind, kind)


def _resolve_strategy_path(raw: str | None, strategy_dir: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else strategy_dir / path


def _artifact_feature_cols(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    return list(payload.get("feature_cols") or [])


def _manifest_artifact_paths(config: dict[str, Any], strategy_dir: Path) -> list[Path]:
    wf_cfg = config.get("walkforward", {}) or {}
    manifest_path = _resolve_strategy_path(wf_cfg.get("manifest_path"), strategy_dir)
    if manifest_path is None or not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("retrains", []) if isinstance(payload, dict) else payload
    if not rows:
        return []
    sample_rows = [rows[0], rows[len(rows) // 2], rows[-1]]
    out: list[Path] = []
    seen: set[str] = set()
    for row in sample_rows:
        raw = str((row or {}).get("artifact_uri") or "")
        if not raw or raw in seen:
            continue
        seen.add(raw)
        path = Path(raw)
        out.append(path if path.is_absolute() else strategy_dir / path)
    return out


def _configured_artifact_path(config: dict[str, Any], strategy_dir: Path) -> Path | None:
    panel = (config.get("ranking", {}) or {}).get("panel_scoring", {}) or {}
    return _resolve_strategy_path(panel.get("artifact_path"), strategy_dir)


def _feature_contract_issues(
    *,
    prod_config: dict[str, Any],
    wf_config: dict[str, Any],
    candidate_artifact: Path | None,
    strategy_dir: Path,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    prod_artifact = candidate_artifact or _configured_artifact_path(prod_config, strategy_dir)
    if prod_artifact is None or not prod_artifact.exists():
        issues.append({
            "path": "artifact",
            "reason": f"prod/candidate artifact not found: {prod_artifact}",
        })
        return issues
    prod_cols = _artifact_feature_cols(prod_artifact)
    if not prod_cols:
        issues.append({"path": "artifact.feature_cols", "reason": "candidate has no feature_cols"})
        return issues

    wf_paths = _manifest_artifact_paths(wf_config, strategy_dir)
    static = _configured_artifact_path(wf_config, strategy_dir)
    if not wf_paths and static is not None:
        wf_paths = [static]
    if not wf_paths:
        issues.append({
            "path": "walkforward.manifest_path",
            "reason": "no WF manifest artifacts or static artifact path found",
        })
        return issues

    prod_set = set(prod_cols)
    for path in wf_paths:
        if not path.exists():
            issues.append({"path": "walkforward.artifact", "reason": f"missing {path}"})
            continue
        cols = _artifact_feature_cols(path)
        if cols != prod_cols:
            issues.append({
                "path": "artifact.feature_cols",
                "artifact": str(path),
                "prod_n_features": len(prod_cols),
                "wf_n_features": len(cols),
                "missing_vs_prod": sorted(prod_set - set(cols))[:20],
                "extra_vs_prod": sorted(set(cols) - prod_set)[:20],
            })
    return issues


def evaluate_wf_config_parity(
    prod_config_path: Path,
    wf_config_path: Path,
    *,
    candidate_artifact: Path | None = None,
    strategy_dir: Path = STRATEGY_DIR,
) -> dict[str, Any]:
    prod = json.loads(prod_config_path.read_text())
    wf = json.loads(wf_config_path.read_text())
    issues: list[dict[str, Any]] = []

    for dotted in SEMANTIC_PATHS:
        prod_value = _clean(_get_path(prod, dotted))
        wf_value = _clean(_get_path(wf, dotted))
        if dotted == "ranking.panel_scoring.kind":
            prod_value = _normalize_kind(prod_value)
            wf_value = _normalize_kind(wf_value)
        if prod_value != wf_value:
            issues.append({
                "path": dotted,
                "prod": prod_value,
                "wf": wf_value,
            })

    issues.extend(_feature_contract_issues(
        prod_config=prod,
        wf_config=wf,
        candidate_artifact=candidate_artifact,
        strategy_dir=strategy_dir,
    ))

    return {
        "passed": not issues,
        "prod_config": str(prod_config_path),
        "wf_config": str(wf_config_path),
        "candidate_artifact": str(candidate_artifact) if candidate_artifact else None,
        "issues": issues,
        "checked_paths": SEMANTIC_PATHS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prod-config", default=str(STRATEGY_DIR / "strategy_config.json"))
    parser.add_argument("--wf-config", required=True)
    parser.add_argument("--candidate-artifact", default=None)
    args = parser.parse_args()

    prod_path = Path(args.prod_config)
    wf_path = Path(args.wf_config)
    if not wf_path.is_absolute():
        wf_path = STRATEGY_DIR / wf_path
    candidate = Path(args.candidate_artifact) if args.candidate_artifact else None
    result = evaluate_wf_config_parity(
        prod_path,
        wf_path,
        candidate_artifact=candidate,
        strategy_dir=STRATEGY_DIR,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
