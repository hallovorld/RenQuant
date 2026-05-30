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
  * cv_method = "purged_walk_forward" (per ALLOWED_CV_METHODS template allowlist).
  * cv_embargo_days = 60 (matches the artifact's lookahead_days).
  * train_run_id = synthetic sha1 of (config_fingerprint, trained_date,
    label_col) — stable across re-stamping, unique per artifact recipe.
  * sentiment_runtime_gate_contract = "runtime_zeroing".

Idempotent: writes ONLY the missing fields; preserves all existing data.

Architecture (R3 refactor 2026-05-30, per §1c Task/Job/Pipeline):
  Pipeline ``StampPanelContractPipeline``
    LoadJob
      LoadArtifactTask          — read artifact JSON into ``ctx.artifact``
      LoadManifestTask          — read manifest, populate ``ctx.manifest_rows``
    ComputeJob
      AggregatePerFoldICTask    — collect oos_mean_ic from per-cut artifacts
      ComputeStampsTask         — assemble the 7-field dict from ctx state
    WriteJob
      WriteStampedArtifactTask  — merge missing fields, atomic write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any

REPO = pathlib.Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
sys.path.insert(0, str(REPO.parent / "renquant-common" / "src"))

from renquant_common import Job, Pipeline, Task  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────────
# Pure helpers (preserved as Task building blocks).
# ────────────────────────────────────────────────────────────────────────────────


def aggregate_per_fold_ic_from_manifest(manifest_path: pathlib.Path) -> list[float]:
    """Pull oos_mean_ic from each per-cut artifact in a WF manifest (pure)."""
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
    """Stable, deterministic, unique-per-recipe train_run_id (pure)."""
    payload = f"{config_fingerprint}|{trained_date}|{label_col}".encode("utf-8")
    return "synthetic_" + hashlib.sha1(payload).hexdigest()[:16]


def stamp_artifact(
    artifact_path: pathlib.Path,
    manifest_path: pathlib.Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backward-compatible single-call API; runs the Pipeline and returns stamps."""
    ctx = StampContext(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        dry_run=dry_run,
    )
    build_pipeline().run(ctx)
    if ctx.stamps is None:
        raise RuntimeError("Pipeline did not compute stamps; check the logs.")
    return ctx.stamps


# ────────────────────────────────────────────────────────────────────────────────
# T/J/P architecture (§1c).
# ────────────────────────────────────────────────────────────────────────────────


@dataclass
class StampContext:
    """State threaded through ``StampPanelContractPipeline``."""
    artifact_path: pathlib.Path
    manifest_path: pathlib.Path
    dry_run: bool = False
    # populated through the pipeline
    artifact: dict | None = None
    per_fold_ic: list[float] = field(default_factory=list)
    stamps: dict[str, Any] | None = None


class LoadArtifactTask(Task):
    """Read the prod GBDT artifact JSON into ``ctx.artifact``."""

    def run(self, ctx: StampContext) -> bool | None:
        ctx.artifact = json.loads(ctx.artifact_path.read_text())
        return True


class LoadManifestTask(Task):
    """Validate the manifest path exists; defer parsing to AggregatePerFoldICTask."""

    def run(self, ctx: StampContext) -> bool | None:
        if not ctx.manifest_path.exists():
            raise RuntimeError(f"manifest not found: {ctx.manifest_path}")
        return True


class LoadJob(Job):
    """Stage 1: read inputs (artifact + manifest)."""

    @property
    def tasks(self) -> list[Task]:
        return [LoadArtifactTask(), LoadManifestTask()]


class AggregatePerFoldICTask(Task):
    """Collect oos_mean_ic from each per-cut artifact in the WF manifest."""

    def run(self, ctx: StampContext) -> bool | None:
        ctx.per_fold_ic = aggregate_per_fold_ic_from_manifest(ctx.manifest_path)
        if not ctx.per_fold_ic:
            raise RuntimeError(
                f"manifest {ctx.manifest_path} has no usable per-fold "
                f"oos_mean_ic values"
            )
        return True


class ComputeStampsTask(Task):
    """Assemble the 7-field stamp dict from ctx state."""

    def run(self, ctx: StampContext) -> bool | None:
        assert ctx.artifact is not None
        per_fold = ctx.per_fold_ic
        ctx.stamps = {
            "train_run_id": synthetic_train_run_id(
                config_fingerprint=str(ctx.artifact.get("config_fingerprint", "")),
                trained_date=str(ctx.artifact.get("trained_date", "")),
                label_col=str(ctx.artifact.get("label_col", "")),
            ),
            "oos_mean_ic": float(statistics.mean(per_fold)),
            "oos_std_ic": float(statistics.stdev(per_fold)) if len(per_fold) > 1 else 0.0,
            "oos_per_fold_ic": [float(x) for x in per_fold],
            "cv_method": "purged_walk_forward",
            "cv_embargo_days": 60,
            "sentiment_runtime_gate_contract": "runtime_zeroing",
        }
        return True


class ComputeJob(Job):
    """Stage 2: compute the 7 stamp values."""

    @property
    def tasks(self) -> list[Task]:
        return [AggregatePerFoldICTask(), ComputeStampsTask()]


class WriteStampedArtifactTask(Task):
    """Idempotently merge missing fields into ``ctx.artifact_path``."""

    def run(self, ctx: StampContext) -> bool | None:
        if ctx.dry_run:
            return True
        assert ctx.artifact is not None and ctx.stamps is not None
        # Only write fields that aren't already present (idempotent).
        for k, v in ctx.stamps.items():
            if k not in ctx.artifact:
                ctx.artifact[k] = v
        ctx.artifact_path.write_text(json.dumps(ctx.artifact, indent=2) + "\n")
        return True


class WriteJob(Job):
    """Stage 3: write the stamped artifact back to disk (skipped on dry-run)."""

    @property
    def tasks(self) -> list[Task]:
        return [WriteStampedArtifactTask()]


def build_pipeline() -> Pipeline:
    """Factory: the canonical ``StampPanelContractPipeline`` instance."""
    return Pipeline(
        [LoadJob(), ComputeJob(), WriteJob()],
        name="StampPanelContract",
    )


# ────────────────────────────────────────────────────────────────────────────────
# CLI entrypoint.
# ────────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=pathlib.Path,
                    default=STRATEGY_DIR / "artifacts/prod/panel-ltr.alpha158_fund.json")
    ap.add_argument("--manifest", type=pathlib.Path,
                    default=STRATEGY_DIR / "artifacts/sim/walkforward_manifest_gbdt_prod_recipe_calibrated.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ctx = StampContext(
        artifact_path=args.artifact,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
    )
    build_pipeline().run(ctx)
    assert ctx.stamps is not None
    print(f"stamps for {args.artifact}:")
    for k, v in ctx.stamps.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {k}: [len={len(v)}, first 3 = {v[:3]}, last 3 = {v[-3:]}]")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
