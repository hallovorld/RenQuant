"""Artifact contracts and run provenance for renquant_104.

This module is intentionally small and stdlib-only. It gives training,
preflight, and daily-run persistence a shared vocabulary for:

* validating that panel artifacts carry enough out-of-sample evidence,
* hashing the exact artifacts/config used by a run, and
* recording data freshness without trusting markdown notes or console logs.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PANEL_REQUIRED_FIELDS = (
    "feature_cols",
    "trained_date",
    "config_fingerprint",
    "panel_shape",
    "lookahead_days",
)

PANEL_STRICT_FIELDS = (
    "train_run_id",
    "oos_mean_ic",
    "oos_std_ic",
    "oos_per_fold_ic",
    "cv_method",
    "cv_embargo_days",
)

_VOLATILE_CONFIG_KEYS = {
    "_strategy_dir",
    "_strategy_config_name",
    "_train_run_id",
}


@dataclass
class ContractResult:
    """Structured validation result used by preflight and tests."""

    name: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError(f"{self.name} contract failed: {self.errors}")


def sha256_file(path: str | Path) -> str | None:
    """Return ``sha256:<hex>`` for an existing file, else ``None``."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def hash_jsonable(obj: Any) -> str:
    """Stable SHA256 hash of a JSON-like object."""
    blob = json.dumps(
        _strip_volatile(obj),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def resolve_artifact_paths(
    config: dict[str, Any],
    strategy_dir: str | Path,
) -> dict[str, Path]:
    """Collect artifact paths referenced by config.

    Keys are dotted config paths, with two stable aliases for the primary
    runtime artifacts:

    * ``panel`` for ``ranking.panel_scoring.artifact_path`` or
      ``panel_ltr.artifact_path``
    * ``global_calibration`` for the global panel-rank calibrator
    """
    sd = Path(strategy_dir)
    paths: dict[str, Path] = {}

    for key, raw in _iter_artifact_refs(config):
        if raw is None or raw == "":
            continue
        paths[key] = _resolve_path(sd, raw)

    panel_raw = (
        config.get("ranking", {})
        .get("panel_scoring", {})
        .get("artifact_path")
        or config.get("panel_ltr", {}).get("artifact_path")
    )
    if panel_raw:
        paths["panel"] = _resolve_path(sd, panel_raw)

    cal_raw = (
        config.get("ranking", {})
        .get("panel_scoring", {})
        .get("global_calibration", {})
        .get("artifact_path")
        or config.get("panel_ltr", {}).get("calibrator_artifact_path")
    )
    if cal_raw:
        paths["global_calibration"] = _resolve_path(sd, cal_raw)

    return paths


def validate_panel_artifact_contract(
    payload: dict[str, Any],
    *,
    strict: bool = False,
) -> ContractResult:
    """Validate evidence-bearing metadata on a panel-LTR artifact."""
    errors: list[str] = []
    warnings: list[str] = []

    required = list(PANEL_REQUIRED_FIELDS)
    if strict:
        required.extend(PANEL_STRICT_FIELDS)

    for key in required:
        if _is_missing(payload.get(key)):
            (errors if strict or key in PANEL_REQUIRED_FIELDS else warnings).append(
                f"missing {key}"
            )

    feature_cols = payload.get("feature_cols") or []
    if not isinstance(feature_cols, list) or not feature_cols:
        errors.append("feature_cols must be a non-empty list")

    folds = payload.get("oos_per_fold_ic")
    if folds is not None:
        if not isinstance(folds, list) or not folds:
            errors.append("oos_per_fold_ic must be a non-empty list when present")
        elif not all(_finite_number(v) for v in folds):
            errors.append("oos_per_fold_ic contains non-finite values")

    for key in ("oos_mean_ic", "oos_std_ic", "eval_ic", "training_train_ic"):
        if key in payload and not _is_missing(payload.get(key)):
            if not _finite_number(payload.get(key)):
                errors.append(f"{key} must be finite")

    lookahead = payload.get("lookahead_days")
    embargo = payload.get("cv_embargo_days")
    if not _is_missing(lookahead) and not _is_missing(embargo):
        try:
            if int(embargo) < int(lookahead):
                errors.append(
                    f"cv_embargo_days={embargo} < lookahead_days={lookahead}"
                )
        except (TypeError, ValueError):
            errors.append("lookahead_days and cv_embargo_days must be integers")

    panel_shape = payload.get("panel_shape") or {}
    rows = panel_shape.get("rows") if isinstance(panel_shape, dict) else None
    if rows is not None:
        try:
            if int(rows) <= 0:
                errors.append("panel_shape.rows must be positive")
        except (TypeError, ValueError):
            errors.append("panel_shape.rows must be an integer")

    if not strict:
        for key in PANEL_STRICT_FIELDS:
            if _is_missing(payload.get(key)):
                warnings.append(f"missing {key}; next retrain must stamp it")

    return ContractResult(
        name="panel_artifact",
        ok=not errors,
        errors=errors,
        warnings=warnings,
        details={
            "n_features": len(feature_cols) if isinstance(feature_cols, list) else 0,
            "trained_date": payload.get("trained_date"),
            "lookahead_days": payload.get("lookahead_days"),
            "cv_embargo_days": payload.get("cv_embargo_days"),
            "oos_mean_ic": payload.get("oos_mean_ic"),
        },
    )


def validate_feature_contract(
    expected_cols: Iterable[str],
    available_cols: Iterable[str],
    *,
    policy: str = "error",
) -> ContractResult:
    """Check that runtime feature columns cover the artifact contract."""
    expected = list(expected_cols or [])
    available = set(available_cols or [])
    missing = sorted(c for c in expected if c not in available)
    ok = not missing or policy == "warn"
    return ContractResult(
        name="feature_contract",
        ok=ok,
        errors=[f"missing {len(missing)} feature column(s)"] if missing and not ok else [],
        warnings=[f"missing {len(missing)} feature column(s)"] if missing and ok else [],
        details={"missing": missing[:25], "missing_count": len(missing)},
    )


def build_run_bundle(
    config: dict[str, Any],
    strategy_dir: str | Path,
    *,
    run_id: str,
    run_type: str,
    ctx: Any | None = None,
    broker_mode: str | None = None,
) -> dict[str, Any]:
    """Build a compact provenance bundle for one inference run."""
    sd = Path(strategy_dir)
    paths = resolve_artifact_paths(config, sd)
    artifact_paths = {k: str(v) for k, v in sorted(paths.items())}
    artifact_hashes = {k: sha256_file(v) for k, v in sorted(paths.items())}
    watchlist = sorted(config.get("watchlist") or [])

    bundle = {
        "schema_version": 1,
        "run_id": run_id,
        "run_type": run_type,
        "broker_mode": broker_mode,
        "config_hash": hash_jsonable(config),
        "watchlist_hash": hash_jsonable(watchlist),
        "watchlist_size": len(watchlist),
        "artifact_paths": artifact_paths,
        "artifact_hashes": artifact_hashes,
        "pipeline_flags": {},
        "data_max_dates": {},
    }

    panel_path = paths.get("panel")
    panel_payload = _read_json(panel_path) if panel_path else None
    if isinstance(panel_payload, dict):
        contract = validate_panel_artifact_contract(panel_payload, strict=False)
        bundle["panel_contract"] = {
            "ok": contract.ok,
            "errors": contract.errors,
            "warnings": contract.warnings,
            "details": contract.details,
        }

    if ctx is not None:
        bundle["pipeline_flags"] = {
            "buy_blocked": bool(getattr(ctx, "buy_blocked", False)),
            "skip_buys": bool(getattr(ctx, "skip_buys", False)),
            "bear_only": bool(getattr(ctx, "bear_only", False)),
            "regime": getattr(ctx, "regime", None),
            "confidence": getattr(ctx, "confidence", None),
        }
        bundle["data_max_dates"] = _data_max_dates(getattr(ctx, "ohlcv", {}) or {})

    return bundle


def _iter_artifact_refs(config: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(config, dict):
        for key, value in config.items():
            dotted = f"{prefix}.{key}" if prefix else key
            if key in {"artifact_path", "artifact_pattern", "gate_b_artifact_path", "calibrator_artifact_path"}:
                yield dotted, value
            yield from _iter_artifact_refs(value, dotted)
    elif isinstance(config, list):
        for i, value in enumerate(config):
            yield from _iter_artifact_refs(value, f"{prefix}[{i}]")


def _resolve_path(strategy_dir: Path, raw: str) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    repo_root = strategy_dir.parent.parent
    repo_candidate = repo_root / p
    if str(raw).startswith(("backtesting/", "data/", "models/", "scripts/")):
        return repo_candidate
    return strategy_dir / p


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            str(k): _strip_volatile(v)
            for k, v in obj.items()
            if str(k) not in _VOLATILE_CONFIG_KEYS
        }
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == []


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _data_max_dates(ohlcv: dict[str, Any]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for ticker, frame in ohlcv.items():
        idx = getattr(frame, "index", None)
        if idx is None or len(idx) == 0:
            out[str(ticker)] = None
            continue
        try:
            max_dt = idx.max()
            if hasattr(max_dt, "date"):
                max_dt = max_dt.date()
            out[str(ticker)] = str(max_dt)
        except Exception:
            out[str(ticker)] = None
    return out
