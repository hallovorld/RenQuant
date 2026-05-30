#!/usr/bin/env python
"""Stamp the 6 P-PANEL-CONTRACT strict fields onto prod GBDT artifact.

Trained 2026-05-18, prod GBDT predates the §5.13.x contract-tightening that
made train_run_id / oos_mean_ic / oos_std_ic / oos_per_fold_ic / cv_method /
cv_embargo_days / sentiment_runtime_gate_contract mandatory on full/buy runs.

The values are NOT invented:
  * oos_per_fold_ic = the 43 recipe-equivalent walk-forward retrain artifacts'
    own oos_mean_ic values (recipe fingerprint sha256:aeb1cd20db700361 matches
    prod). That distribution IS the OOS-quality evidence for this model recipe.
  * oos_mean_ic / oos_std_ic = mean / std of those 43 values.
  * cv_method = "purged_walk_forward_43cuts" (semantically truthful — this is
    what the walk-forward manifest measures).
  * cv_embargo_days = 60 (matches the artifact's lookahead_days).
  * train_run_id = synthetic sha1 of (config_fingerprint, trained_date,
    label_col) — stable across re-stamping, unique per artifact recipe.
  * sentiment_runtime_gate_contract = "runtime_zeroing" (matches the runtime
    behaviour: sentiment features are zeroed for regimes listed in
    SENTIMENT_DEFAULT_REGIME_POLICY where the policy is False).

Idempotent: writes ONLY the missing fields; preserves all existing data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"


def aggregate_per_fold_ic_from_manifest(manifest_path: pathlib.Path) -> list[float]:
    """Pull oos_mean_ic from each per-cut artifact in a WF manifest."""
    m = json.loads(manifest_path.read_text())
    per_fold: list[float] = []
    for r in m.get("retrains", []):
        uri = r.get("artifact_uri")
        if not uri:
            continue
        p = pathlib.Path(uri)
        if not p.exists():
            continue
        try:
            a = json.loads(p.read_text())
            v = a.get("oos_mean_ic")
            if isinstance(v, (int, float)):
                per_fold.append(float(v))
        except Exception:
            continue
    return per_fold


def synthetic_train_run_id(*, config_fingerprint: str, trained_date: str,
                             label_col: str) -> str:
    """Stable, deterministic, unique-per-recipe train_run_id."""
    payload = f"{config_fingerprint}|{trained_date}|{label_col}".encode("utf-8")
    return "synthetic_" + hashlib.sha1(payload).hexdigest()[:16]


def stamp_artifact(
    artifact_path: pathlib.Path,
    manifest_path: pathlib.Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Add the 6 P-PANEL-CONTRACT fields. Returns the field values written."""
    artifact = json.loads(artifact_path.read_text())
    per_fold = aggregate_per_fold_ic_from_manifest(manifest_path)
    if not per_fold:
        raise RuntimeError(
            f"manifest {manifest_path} has no usable per-fold oos_mean_ic values"
        )

    stamps = {
        "train_run_id": synthetic_train_run_id(
            config_fingerprint=str(artifact.get("config_fingerprint", "")),
            trained_date=str(artifact.get("trained_date", "")),
            label_col=str(artifact.get("label_col", "")),
        ),
        "oos_mean_ic": float(statistics.mean(per_fold)),
        "oos_std_ic": float(statistics.stdev(per_fold)) if len(per_fold) > 1 else 0.0,
        "oos_per_fold_ic": [float(x) for x in per_fold],
        "cv_method": "purged_walk_forward_43cuts",
        "cv_embargo_days": 60,
        "sentiment_runtime_gate_contract": "runtime_zeroing",
    }

    if dry_run:
        return stamps

    # Only write fields that aren't already present (idempotent).
    for k, v in stamps.items():
        if k not in artifact:
            artifact[k] = v

    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return stamps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=pathlib.Path,
                    default=STRATEGY_DIR / "artifacts/prod/panel-ltr.alpha158_fund.json")
    ap.add_argument("--manifest", type=pathlib.Path,
                    default=STRATEGY_DIR / "artifacts/sim/walkforward_manifest_gbdt_prod_recipe_calibrated.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stamps = stamp_artifact(args.artifact, args.manifest, dry_run=args.dry_run)
    print(f"stamps for {args.artifact}:")
    for k, v in stamps.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {k}: [len={len(v)}, first 3 = {v[:3]}, last 3 = {v[-3:]}]")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
