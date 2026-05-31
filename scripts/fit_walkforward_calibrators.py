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
    args = parser.parse_args()

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
        futures = [
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
            )
            for row in rows
        ]
        for fut in cf.as_completed(futures):
            try:
                fitted.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                # Bug C v6 (2026-05-30): degenerate per-cut scorers (Bug G
                # symptom) raise here. Record + continue so the manifest gets
                # partial coverage instead of zero coverage.
                if not bool(getattr(args, "continue_on_failure", False)):
                    raise
                failed_cutoffs.append({"error": repr(exc)[:300]})
                print(f"  (skipped: {exc!s})", flush=True)

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
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(out_payload, indent=2, sort_keys=False) + "\n")
    print(f"stamped {len(fitted)}/{len(payload['retrains'])} calibrators -> {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
