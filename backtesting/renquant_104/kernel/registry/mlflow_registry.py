"""MLflow registry helpers — split into small, single-responsibility units.

Per CLAUDE.md §1c every helper is ≤ 50 lines, single-responsibility, and
has a paired unit test in tests/test_mlflow_registry.py.

This module is intentionally a THIN wrapper over mlflow's public API.
We expose just enough surface to:

  1. point MLflow at a local file backend (no server needed),
  2. open a run as a context manager and stamp params,
  3. atomically log a local file as an artifact + JSON meta sidecar,
  4. resolve `mlflow://<run-id>/<artifact_path>` URIs back to local paths,
  5. register a model in MLflow Model Registry against a run.

Kept stdlib-only at import time so that strategy code that doesn't touch
the registry never pays the mlflow import cost. mlflow itself is loaded
lazily inside each helper.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("kernel.registry")

# `mlflow://<run_id>/<artifact_path>` — run_id is 32 hex chars per MLflow.
_MLFLOW_URI_RE = re.compile(r"^mlflow://([0-9a-fA-F]{32})/(.+)$")


# ── URI helpers ───────────────────────────────────────────────────────────────

def is_mlflow_uri(uri: str) -> bool:
    """Return True if `uri` is a well-formed `mlflow://<run_id>/<path>` URI."""
    if not isinstance(uri, str):
        return False
    return _MLFLOW_URI_RE.match(uri) is not None


def parse_mlflow_uri(uri: str) -> tuple[str, str]:
    """Split `mlflow://<run_id>/<path>` into (run_id, artifact_path).

    Raises ValueError if `uri` is not a well-formed mlflow URI.
    """
    m = _MLFLOW_URI_RE.match(uri or "")
    if not m:
        raise ValueError(f"Not an mlflow URI: {uri!r}")
    return m.group(1), m.group(2)


# ── Tracking init ────────────────────────────────────────────────────────────

def init_tracking(uri: str = "file:./mlruns") -> str:
    """Set MLFLOW_TRACKING_URI for the current process.

    Returns the resolved URI string (handy in tests). Idempotent — calling
    twice with the same URI is a no-op. Imports mlflow lazily so this
    module is safe to import in environments without mlflow installed
    (the call itself will error if mlflow is missing).
    """
    import mlflow  # noqa: PLC0415
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    log.debug("init_tracking: MLFLOW_TRACKING_URI=%s", uri)
    return uri


# ── Run lifecycle ────────────────────────────────────────────────────────────

@contextlib.contextmanager
def start_run(experiment: str,
              params: dict[str, Any] | None = None,
              run_name: str | None = None) -> Iterator[str]:
    """Context manager: open an MLflow run, yield its run_id.

    On enter: ensure the experiment exists, start the run, log `params`
    (one mlflow.log_param per key — each value is coerced to str so we
    don't choke on numpy scalars).

    On exit: mlflow's `with` semantics close the run automatically. If
    the body raises, the run is marked FAILED.
    """
    import mlflow  # noqa: PLC0415
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        if params:
            for k, v in params.items():
                try:
                    mlflow.log_param(k, str(v))
                except Exception as exc:  # noqa: BLE001
                    log.warning("log_param(%s) failed: %s", k, exc)
        log.info("start_run: experiment=%s run_id=%s", experiment, run_id)
        yield run_id


# ── Artifact logging ─────────────────────────────────────────────────────────

def _write_meta_sidecar(meta: dict[str, Any], dest_dir: Path,
                         meta_filename: str) -> Path:
    """Write `meta` as JSON to dest_dir/meta_filename, return path.

    Single-responsibility: serialize a dict to disk for upload. Used by
    log_artifact_with_meta. Sorted keys + indent=2 so diffs across runs
    are reviewable.
    """
    meta_path = dest_dir / meta_filename
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True,
                                     default=str))
    return meta_path


def log_artifact_with_meta(run_id: str,
                           local_path: str | Path,
                           artifact_path: str = "",
                           meta: dict[str, Any] | None = None) -> str:
    """Upload a local file to an MLflow run as an artifact, with JSON meta.

    Uses MlflowClient (run-id keyed) so this works whether or not a run
    is currently active in the calling thread — typical caller is inside
    `with start_run(...)`, but training pipelines may also call it later.
    Both the artifact and its `<artifact>.meta.json` sidecar are uploaded
    against the same run_id, so a reader of the run sees them as one
    bundle.

    Returns the canonical mlflow URI (`mlflow://<run_id>/<dest>`) for the
    primary artifact (not the meta sidecar).
    """
    from mlflow.tracking import MlflowClient  # noqa: PLC0415
    src = Path(local_path)
    if not src.exists():
        raise FileNotFoundError(f"log_artifact_with_meta: missing {src}")
    artifact_subdir = artifact_path.strip("/")
    dest_relpath = f"{artifact_subdir}/{src.name}" if artifact_subdir else src.name
    client = MlflowClient()
    client.log_artifact(run_id, str(src),
                        artifact_path=artifact_subdir or None)
    if meta is not None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            meta_filename = f"{src.name}.meta.json"
            _write_meta_sidecar(meta, tdp, meta_filename)
            client.log_artifact(run_id, str(tdp / meta_filename),
                                artifact_path=artifact_subdir or None)
    log.info("log_artifact_with_meta: run=%s → %s", run_id, dest_relpath)
    return f"mlflow://{run_id}/{dest_relpath}"


# ── URI resolution ───────────────────────────────────────────────────────────

def resolve_uri(uri: str | Path) -> Path:
    """Return a local Path for a URI.

    Accepts:
      * `mlflow://<run_id>/<artifact_path>` → downloads via mlflow client
        and returns the local cached path.
      * a plain local path (str / Path) → returned as Path unchanged.

    Raises FileNotFoundError if the resolved path doesn't exist after
    download/passthrough. mlflow's download_artifacts handles its own
    caching, so repeated resolves of the same uri are cheap.
    """
    if isinstance(uri, Path):
        if not uri.exists():
            raise FileNotFoundError(f"resolve_uri: local path missing: {uri}")
        return uri
    if not is_mlflow_uri(uri):
        p = Path(uri)
        if not p.exists():
            raise FileNotFoundError(f"resolve_uri: local path missing: {p}")
        return p
    run_id, artifact_path = parse_mlflow_uri(uri)
    import mlflow  # noqa: PLC0415
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path,
    )
    p = Path(local)
    if not p.exists():
        raise FileNotFoundError(f"resolve_uri: download produced no file at {p}")
    return p


# ── Model Registry ───────────────────────────────────────────────────────────

def register_model(name: str, run_id: str,
                   stage: str | None = None,
                   artifact_path: str = "model") -> dict[str, Any]:
    """Register the artifact at `runs:/run_id/artifact_path` as `name`.

    Uses MlflowClient directly so this works with plain logged artifacts
    (mlflow 3.x's top-level `register_model` requires an MLmodel-flavored
    logged model; our calibrators / xgboost JSON / parquet files don't
    have that flavour).

    Returns a dict with `name`, `version`, `run_id`, and `stage`. Stage
    transitions use the still-supported `transition_model_version_stage`
    API (will be deprecated in mlflow 4.x; migrate when that lands).

    Caller is responsible for having logged the artifact at `artifact_path`
    inside `run_id` first — typically via `log_artifact_with_meta`.
    """
    from mlflow.tracking import MlflowClient  # noqa: PLC0415
    from mlflow.exceptions import RestException  # noqa: PLC0415
    client = MlflowClient()
    try:
        client.create_registered_model(name)
    except (RestException, Exception) as exc:  # noqa: BLE001
        # Already-exists is the common case; swallow only that.
        if "already exists" not in str(exc).lower() and \
                "RESOURCE_ALREADY_EXISTS" not in str(exc):
            raise
    source_uri = f"runs:/{run_id}/{artifact_path}"
    mv = client.create_model_version(name=name, source=source_uri,
                                      run_id=run_id)
    if stage:
        client.transition_model_version_stage(
            name=name, version=mv.version, stage=stage,
        )
    return {
        "name":    name,
        "version": str(mv.version),
        "run_id":  run_id,
        "stage":   stage,
    }
