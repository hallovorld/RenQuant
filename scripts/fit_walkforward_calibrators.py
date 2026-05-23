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
    method: str,
    panel: str | None,
    raw_label_panel: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    cutoff = pd.Timestamp(row["cutoff_date"])
    scorer_path = Path(str(row["artifact_uri"]))
    out_path = _calibrator_path(calibrator_root, cutoff)
    if out_path.exists() and not overwrite:
        stamped = copy.deepcopy(row)
        stamped["calibrator_uri"] = str(out_path)
        stamped["calibrator_data_start"], stamped["calibrator_data_end"] = _date_window(
            cutoff,
            training_window_years,
            int(row.get("lookahead_days", 60)),
        )
        return stamped

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lookahead_days = int(row.get("lookahead_days", 60))
    data_start, data_end = _date_window(cutoff, training_window_years, lookahead_days)
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "fit_calibrator_alpha158_fund.py"),
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
    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
        futures = [
            ex.submit(
                _fit_one,
                row,
                calibrator_root=calibrator_root,
                training_window_years=training_window_years,
                method=args.method,
                panel=args.panel,
                raw_label_panel=args.raw_label_panel,
                overwrite=args.overwrite,
            )
            for row in rows
        ]
        for fut in cf.as_completed(futures):
            fitted.append(fut.result())

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
        "fit_window": "training_window_through_effective_cutoff",
        "data_end": "cutoff_date_minus_lookahead_bday_exclusive",
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(out_payload, indent=2, sort_keys=False) + "\n")
    print(f"stamped {len(fitted)}/{len(payload['retrains'])} calibrators -> {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
