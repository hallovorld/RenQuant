#!/usr/bin/env python
"""Stamp strict config fingerprints onto existing walk-forward artifacts.

This is a repair tool for the pre-2026-05-25 WF artifacts that were trained by
``train_production_model.py`` while that script still skipped config
fingerprints for walk-forward outputs. It never changes model weights. It only
adds the same config contract that strict full/WF scoring already requires.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from scripts import train_production_model as train_prod  # noqa: E402
from scripts.run_wf_gate import _manifest_recipe_usage  # noqa: E402
from kernel.panel_pipeline.panel_scorer import model_content_sha256  # noqa: E402


def _resolve_strategy_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else STRATEGY_DIR / p


def _manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("retrains", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest has no retrain rows: {manifest_path}")
    return [r for r in rows if isinstance(r, dict)]


def _artifact_paths(manifest_path: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for row in _manifest_rows(manifest_path):
        uri = row.get("artifact_uri")
        if not uri:
            continue
        path = _resolve_strategy_path(str(uri))
        key = path.resolve() if path.exists() else path
        if key not in seen:
            seen.add(key)
            paths.append(path)
    if not paths:
        raise ValueError(f"manifest has no artifact_uri rows: {manifest_path}")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"manifest artifact(s) missing: {missing[:5]}")
    return paths


def _artifact_calibrator_pairs(manifest_path: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for row in _manifest_rows(manifest_path):
        artifact_uri = row.get("artifact_uri")
        calibrator_uri = row.get("calibrator_uri") or row.get("calibration_uri")
        if not artifact_uri or not calibrator_uri:
            continue
        scorer = _resolve_strategy_path(str(artifact_uri))
        calibrator = _resolve_strategy_path(str(calibrator_uri))
        if not scorer.exists():
            raise FileNotFoundError(f"manifest scorer missing: {scorer}")
        if not calibrator.exists():
            raise FileNotFoundError(f"manifest calibrator missing: {calibrator}")
        pairs.append((scorer, calibrator))
    return pairs


def _label_for_artifact(artifact: dict[str, Any]) -> str:
    label = artifact.get("label_col")
    if isinstance(label, str) and label:
        return label
    lookahead = int(artifact.get("lookahead_days") or 60)
    return f"fwd_{lookahead}d_excess"


def validate_recipe(manifest_path: Path, reference_artifact: Path | None) -> dict[str, Any]:
    """Require recipe parity before metadata repair."""
    if reference_artifact is None:
        return {
            "recipe_validated": False,
            "reason": "reference_artifact is required for safe stamping",
        }
    usage = _manifest_recipe_usage(manifest_path, reference_artifact)
    if not bool(usage.get("recipe_validated")):
        raise ValueError(
            "refusing to stamp WF fingerprints: manifest recipe validation "
            f"failed ({usage.get('reason')})"
        )
    return usage


def _scorer_identity(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text())
    content_fp = model_content_sha256(payload)
    file_fp = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return content_fp, file_fp


def _stamp_calibrator_binding(
    *,
    scorer_path: Path,
    calibrator_path: Path,
    dry_run: bool,
) -> bool:
    content_fp, file_fp = _scorer_identity(scorer_path)
    payload = json.loads(calibrator_path.read_text())
    metadata = payload.setdefault("metadata", {})
    updates = {
        "scorer_artifact": str(scorer_path),
        "scorer_model_content_fingerprint": content_fp,
        "scorer_artifact_fingerprint": content_fp,
        "scorer_artifact_sha256": file_fp,
    }
    changed = any(metadata.get(k) != v for k, v in updates.items())
    if changed and not dry_run:
        metadata.update(updates)
        calibrator_path.write_text(json.dumps(payload))
    return changed


def stamp_manifest(
    *,
    manifest_path: Path,
    fingerprint_config: Path,
    reference_artifact: Path | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stamp all scorer artifacts referenced by ``manifest_path``."""
    recipe_usage = validate_recipe(manifest_path, reference_artifact)
    paths = _artifact_paths(manifest_path)
    stamped: list[str] = []
    unchanged: list[str] = []
    calibrators_stamped: list[str] = []

    for path in paths:
        artifact = json.loads(path.read_text())
        before = artifact.get("config_fingerprint")
        fp = train_prod.stamp_fingerprint(
            artifact,
            fingerprint_config_path=str(fingerprint_config),
            label_used=_label_for_artifact(artifact),
            feat_cols=list(artifact.get("feature_cols") or []),
        )
        if before == fp:
            unchanged.append(str(path))
            continue
        stamped.append(str(path))
        if not dry_run:
            path.write_text(json.dumps(artifact))

    for scorer_path, calibrator_path in _artifact_calibrator_pairs(manifest_path):
        if _stamp_calibrator_binding(
            scorer_path=scorer_path,
            calibrator_path=calibrator_path,
            dry_run=dry_run,
        ):
            calibrators_stamped.append(str(calibrator_path))

    return {
        "manifest_path": str(manifest_path),
        "fingerprint_config": str(fingerprint_config),
        "reference_artifact": str(reference_artifact) if reference_artifact else None,
        "recipe_usage": recipe_usage,
        "n_artifacts": len(paths),
        "n_stamped": len(stamped),
        "n_unchanged": len(unchanged),
        "n_calibrators_stamped": len(calibrators_stamped),
        "dry_run": bool(dry_run),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--fingerprint-config", required=True)
    p.add_argument("--reference-artifact", required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = stamp_manifest(
        manifest_path=_resolve_strategy_path(args.manifest),
        fingerprint_config=_resolve_strategy_path(args.fingerprint_config),
        reference_artifact=_resolve_strategy_path(args.reference_artifact),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
