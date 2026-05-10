"""kernel.registry — MLflow-backed artifact registry foundation.

Goal: every model / calibrator / data parquet becomes traceable via an
MLflow run-id without breaking existing local-file readers.

Public surface:
    init_tracking(uri="file:./mlruns")     — set MLFLOW_TRACKING_URI
    start_run(experiment, params)          — context manager → run_id
    log_artifact_with_meta(run_id, ...)    — atomic upload + meta
    resolve_uri(uri)                       — mlflow://run-id/path or local
    register_model(name, run_id, stage)    — Model Registry handle

See README.md in this directory for the artifact migration order.
"""
from __future__ import annotations

from .mlflow_registry import (  # noqa: F401
    init_tracking,
    start_run,
    log_artifact_with_meta,
    resolve_uri,
    register_model,
    is_mlflow_uri,
    parse_mlflow_uri,
)

__all__ = [
    "init_tracking",
    "start_run",
    "log_artifact_with_meta",
    "resolve_uri",
    "register_model",
    "is_mlflow_uri",
    "parse_mlflow_uri",
]
