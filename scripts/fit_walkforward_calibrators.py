#!/usr/bin/env python
"""Fit one calibrator per walk-forward scorer and stamp the manifest.

Walk-forward simulation dispatches a different scorer artifact by date. The
calibrator must move with that scorer; a single static calibrator is a foreign
calibration surface and strict inference correctly rejects it.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"


def _resolve_strategy_path(raw: str | Path, *, base: Path = STRATEGY_DIR) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else base / p


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("retrains"), list):
        raise ValueError(f"manifest must be a dict with retrains list: {path}")
    return payload


def _calibrator_path(root: Path, cutoff: pd.Timestamp) -> Path:
    return root / cutoff.date().isoformat() / "panel-rank-calibration.json"


def _existing_artifact_window(out_path: Path) -> float | None:
    """Read the calibrator artifact's stamped window-years, or ``None`` if
    the file is missing / unreadable / lacks the metadata.

    Reuse gating depends on this:  if a legacy artifact lacks the stamp,
    we must NOT reuse it under a non-legacy requested window — the manifest
    would lie about the fit data. Returning ``None`` makes the caller force
    a refit (or refuse with a clear error).
    """
    if not out_path.exists():
        return None
    try:
        data = json.loads(out_path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for key in ("calibrator_window_years", "window_years"):
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return None
    return None


def _stamp_window_into_artifact(out_path: Path, window_years: float) -> None:
    """Inject ``calibrator_window_years`` into the artifact JSON the fitter
    subprocess just produced. The fitter scripts don't know about the
    orchestrator's window-policy concept (they only see ``--data-start`` /
    ``--data-end``), so we post-process here so future reuse decisions can
    audit the fit policy from the artifact itself.
    """
    data = json.loads(out_path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(
            f"calibrator artifact at {out_path} is not a JSON object; "
            f"cannot stamp calibrator_window_years"
        )
    data["calibrator_window_years"] = float(window_years)
    out_path.write_text(json.dumps(data, indent=2) + "\n")


def _date_window(
    cutoff: pd.Timestamp,
    years: float,
    lookahead_days: int,
) -> tuple[str | None, str]:
    effective_cutoff = cutoff - pd.offsets.BDay(max(0, int(lookahead_days)))
    if years <= 0:
        start = None
    else:
        days = int(float(years) * 365.25)
        start = effective_cutoff - pd.Timedelta(days=days)
    end = effective_cutoff
    return (start.date().isoformat() if start is not None else None,
            end.date().isoformat())


def _fit_one(
    row: dict[str, Any],
    *,
    calibrator_root: Path,
    training_window_years: float,
    calibrator_window_years: float | None,
    method: str,
    panel: str | None,
    raw_label_panel: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    # Bug G (2026-05-31): compressed per-cut calibrator output (range 0.07-0.13
    # at fit time vs daily-shadow's 0.49) tracks to the calibrator-fit data
    # window being narrower than what's needed for the calibrator to see a
    # representative score-vs-label distribution. ``training_window_years``
    # controls the model-training horizon (3.0 keeps the model from over-fitting
    # ancient regimes); ``calibrator_window_years`` decouples the calibrator
    # window so it can widen (default 0.0 → full history up to the per-cut
    # cutoff-minus-lookahead boundary) without changing the model window.
    # ``None`` preserves the legacy "use training_window_years for both" path.
    effective_years = (
        float(calibrator_window_years)
        if calibrator_window_years is not None
        else float(training_window_years)
    )
    cutoff = pd.Timestamp(row["cutoff_date"])
    scorer_path = Path(str(row["artifact_uri"]))
    out_path = _calibrator_path(calibrator_root, cutoff)
    if out_path.exists() and not overwrite:
        # Reuse gate: refuse to stamp the manifest with a window the cached
        # artifact wasn't actually fit on. Without this guard, a rerun with
        # ``--calibrator-window-years 0.0`` against a directory containing
        # legacy 3y artifacts would silently claim full-history calibration
        # while the underlying files are still 3y — Bug G regression risk.
        existing_window = _existing_artifact_window(out_path)
        if existing_window is None:
            raise RuntimeError(
                f"Refusing to reuse legacy calibrator at {out_path} that lacks "
                f"calibrator_window_years metadata. Re-run with --overwrite to "
                f"refit and stamp the window, or remove the legacy artifact."
            )
        if abs(existing_window - effective_years) > 1e-9:
            raise RuntimeError(
                f"Refusing to reuse {out_path}: existing window "
                f"{existing_window} != requested {effective_years}. Re-run "
                f"with --overwrite to refit at the new window."
            )
        stamped = copy.deepcopy(row)
        stamped["calibrator_uri"] = str(out_path)
        stamped["calibrator_data_start"], stamped["calibrator_data_end"] = _date_window(
            cutoff,
            effective_years,
            int(row.get("lookahead_days", 60)),
        )
        stamped["calibrator_window_years"] = effective_years
        return stamped

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lookahead_days = int(row.get("lookahead_days", 60))
    data_start, data_end = _date_window(cutoff, effective_years, lookahead_days)
    # Dispatch by artifact type (2026-05-30 Bug C fix):
    #   .json  → GBDT panel-LTR fitter
    #   .pt    → HF PatchTST sequence fitter (same CLI surface)
    # Pre-fix: hardcoded to GBDT script, which crashed
    # UnicodeDecodeError on torch pickles.
    if scorer_path.suffix == ".pt":
        fitter_script = "fit_hf_patchtst_calibrator.py"
    else:
        fitter_script = "fit_calibrator_alpha158_fund.py"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / fitter_script),
        "--scorer-artifact",
        str(scorer_path),
        "--out",
        str(out_path),
        "--data-end",
        data_end,
        "--method",
        method,
    ]
    if data_start:
        cmd.extend(["--data-start", data_start])
    if panel:
        cmd.extend(["--panel", panel])
    if raw_label_panel:
        cmd.extend(["--raw-label-panel", raw_label_panel])
    proc = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"calibrator fit failed for cutoff={cutoff.date()} "
            f"rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # Stamp the chosen window into the artifact so reuse gates can audit it.
    _stamp_window_into_artifact(out_path, effective_years)

    stamped = copy.deepcopy(row)
    stamped["calibrator_uri"] = str(out_path)
    stamped["calibrator_data_start"] = data_start
    stamped["calibrator_data_end"] = data_end
    stamped["calibrator_method"] = method
    stamped["calibrator_cutoff_contract"] = "date < cutoff_date - lookahead_days BDay"
    stamped["calibrator_window_years"] = effective_years
    return stamped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument(
        "--calibrator-root",
        default="artifacts/sim/walkforward_calibrators",
        help="Relative paths resolve under backtesting/renquant_104.",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--method", default="platt", choices=["platt", "isotonic"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--raw-label-panel", default=None)
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="Don't abort the manifest run if a single cutoff's "
                             "calibrator fit fails (e.g. degenerate scorer output). "
                             "Stamps partial coverage; failed cutoffs are logged. "
                             "Useful when per-cut models have weak signal (Bug G).")
    parser.add_argument("--calibrator-window-years", type=float, default=None,
                        help="Bug G fix (2026-05-31): decouple the calibrator-fit "
                             "data window from the model-training window. The "
                             "model-training window (manifest's training_window_years) "
                             "stays narrow to avoid over-fitting ancient regimes, "
                             "but the calibrator needs a wider score-vs-label "
                             "distribution to keep its output non-compressed. "
                             "0.0 = full history up to per-cut "
                             "cutoff-minus-lookahead. Omitted → legacy behavior "
                             "(calibrator window = training_window_years).")
    parser.add_argument("--min-coverage", type=float, default=0.0,
                        help="Minimum fraction of planned cutoffs that must fit "
                             "successfully for main() to return 0. ``0.0`` (default) "
                             "means \"any non-zero coverage is OK\" but still treats "
                             "zero-fits as a hard failure under --continue-on-failure. "
                             "Strict research runs should pass ``--min-coverage 1.0`` "
                             "to require every cutoff to fit.")
    args = parser.parse_args()
    if not 0.0 <= float(args.min_coverage) <= 1.0:
        parser.error(
            f"--min-coverage must be in [0.0, 1.0], got {args.min_coverage}"
        )

    manifest_path = _resolve_strategy_path(args.manifest)
    out_manifest = _resolve_strategy_path(args.out_manifest)
    calibrator_root = _resolve_strategy_path(args.calibrator_root)
    payload = _load_manifest(manifest_path)
    rows = list(payload["retrains"])
    if args.limit is not None:
        rows = rows[: args.limit]
    training_window_years = float(payload.get("training_window_years", 3.0))

    fitted: list[dict[str, Any]] = []
    failed_cutoffs: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
        futures = {
            ex.submit(
                _fit_one,
                row,
                calibrator_root=calibrator_root,
                training_window_years=training_window_years,
                calibrator_window_years=args.calibrator_window_years,
                method=args.method,
                panel=args.panel,
                raw_label_panel=args.raw_label_panel,
                overwrite=args.overwrite,
            ): row
            for row in rows
        }
        for fut in cf.as_completed(futures):
            row = futures[fut]
            try:
                fitted.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                # Bug C v6 (2026-05-30): degenerate per-cut scorers (Bug G
                # symptom) raise here. Record + continue so the manifest gets
                # partial coverage instead of zero coverage.
                if not bool(getattr(args, "continue_on_failure", False)):
                    raise
                failed_cutoffs.append({
                    "cutoff_date": str(row.get("cutoff_date")),
                    "artifact_uri": str(row.get("artifact_uri")),
                    "error": repr(exc)[:300],
                })
                print(f"  (skipped cutoff={row.get('cutoff_date')}: {exc!s})",
                      flush=True)

    by_cutoff = {str(r["cutoff_date"]): r for r in fitted}
    stamped_rows = []
    for row in payload["retrains"]:
        stamped_rows.append(by_cutoff.get(str(row["cutoff_date"]), row))

    out_payload = copy.deepcopy(payload)
    out_payload["retrains"] = stamped_rows
    out_payload["calibrator_manifest_version"] = 1
    out_payload["calibrator_stamped_at_utc"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    out_payload["calibrator_policy"] = {
        "method": args.method,
        "fit_window": (
            "calibrator_window_through_effective_cutoff"
            if args.calibrator_window_years is not None
            else "training_window_through_effective_cutoff"
        ),
        "data_end": "cutoff_date_minus_lookahead_bday_exclusive",
        # Bug G stamp: decoupled calibrator window. ``training_window_years``
        # is the model window from the manifest; ``calibrator_window_years``
        # is the override (None = legacy, == training_window_years).
        "training_window_years": training_window_years,
        "calibrator_window_years": (
            float(args.calibrator_window_years)
            if args.calibrator_window_years is not None
            else training_window_years
        ),
        "min_coverage": float(args.min_coverage),
    }
    # Persist per-cutoff failures into the manifest so downstream audit /
    # research tooling can attribute coverage gaps without re-running the
    # fit job. Empty list when no failures.
    out_payload["calibrator_failures"] = failed_cutoffs

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(out_payload, indent=2, sort_keys=False) + "\n")
    total_planned = len(payload["retrains"])
    fitted_count = len(fitted)
    coverage = (fitted_count / total_planned) if total_planned > 0 else 0.0
    print(
        f"stamped {fitted_count}/{total_planned} calibrators "
        f"(coverage={coverage:.1%}, failures={len(failed_cutoffs)}) -> {out_manifest}"
    )

    # Exit-code policy:
    #   * zero successful fits  → rc=2 (silent zero-coverage is a process bug)
    #   * coverage < --min-coverage → rc=2 (strict research-run guard)
    #   * else                        → rc=0
    # --continue-on-failure relaxes the per-row exception behavior but does
    # NOT relax these aggregate gates.
    if fitted_count == 0 and total_planned > 0:
        print(
            "ERROR: zero successful calibrator fits — refusing to declare "
            "success on an empty manifest"
        )
        return 2
    if coverage + 1e-9 < float(args.min_coverage):
        print(
            f"ERROR: coverage {coverage:.1%} < required {float(args.min_coverage):.1%}"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
