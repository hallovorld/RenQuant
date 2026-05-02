#!/usr/bin/env python
"""Side-by-side IC comparison across panel-LTR experimental retrains.

Reads the per-experiment ``artifacts/panel-ltr.<label>.json`` files
written by the training pipeline + the structured logs at
``/tmp/<label>.log`` and prints an apples-to-apples table::

    label                train_ic    cpcv_mean_ic    cpcv_std    n_splits    status
    wl178_baseline       +0.116      +0.0004         0.020       15          completed
    wl178_layer1         +0.069      -0.0008         0.020       15          guard_fired
    wl178_layer1_2       …           …               …           …           …

Usage::

    python scripts/compare_panel_experiments.py
    python scripts/compare_panel_experiments.py --logs /tmp/wl178_layer1.log /tmp/wl178_layer1_2.log
    python scripts/compare_panel_experiments.py --json out.json   # write machine-readable

Goal: every architecture experiment lands a row here so the operator
can read the full A/B series at a glance instead of grepping logs.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("compare-experiments")


# Regex for the structured CPCV summary line emitted by CrossValidateTask.
_RE_CPCV = re.compile(
    r"CrossValidateTask\[(?P<method>\w+)\]:\s*"
    r"mean=(?P<mean>[+-]?\d*\.\d+)\s+"
    r"std=(?P<std>[+-]?\d*\.\d+)\s+"
    r"q05=(?P<q05>[+-]?\d*\.\d+)\s+"
    r"q50=(?P<q50>[+-]?\d*\.\d+)\s+"
    r"q95=(?P<q95>[+-]?\d*\.\d+)\s+"
    r"n_splits=(?P<n_splits>\d+)"
)
_RE_FINAL_FIT = re.compile(
    r"FinalFitTask:\s+backend=\S+\s+train_ic=(?P<train_ic>[+-]?\d*\.\d+)"
)
_RE_NGBOOST_VAL = re.compile(
    r"val_mu_ic=(?P<val_mu_ic>[+-]?\d*\.\d+)"
)
_RE_PANEL_MEAN_IC = re.compile(
    r"RunPanelTrainingTask:\s*mean_ic=(?P<mean_ic>[+-]?\d*\.\d+)"
)
_RE_GUARD_FIRED = re.compile(
    r"early_stopping fired at round (\d+)\s*\(< min_best_iter=\d+\)"
)
_RE_SR_TASK = re.compile(
    r"SectorRankNormalizeTask: added (\d+) \(ticker × _sr column\) entries "
    r"across (\d+) tickers, (\d+) feature cols"
)
_RE_OH_TASK = re.compile(
    r"SectorOneHotTask: added (\d+) sector indicator entries across (\d+) "
    r"tickers \((\d+) distinct sectors\)"
)


def parse_log(path: Path) -> dict:
    """Extract structured metrics from a training log."""
    if not path.exists():
        return {"label": path.stem, "status": "missing", "log_path": str(path)}
    text = path.read_text(errors="replace")

    out: dict = {
        "label":            path.stem,
        "log_path":         str(path),
        "log_size_bytes":   path.stat().st_size,
        "log_mtime":        _dt.datetime.fromtimestamp(
            path.stat().st_mtime,
        ).isoformat(),
    }

    # CPCV is the OOS metric we care about (train_ic alone is in-sample).
    m = _RE_CPCV.search(text)
    if m:
        out["cpcv_method"]    = m.group("method")
        out["cpcv_mean_ic"]   = float(m.group("mean"))
        out["cpcv_std"]       = float(m.group("std"))
        out["cpcv_q05"]       = float(m.group("q05"))
        out["cpcv_q50"]       = float(m.group("q50"))
        out["cpcv_q95"]       = float(m.group("q95"))
        out["cpcv_n_splits"]  = int(m.group("n_splits"))

    m = _RE_FINAL_FIT.search(text)
    if m:
        out["train_ic"] = float(m.group("train_ic"))

    m = _RE_NGBOOST_VAL.search(text)
    if m:
        out["ngboost_val_mu_ic"] = float(m.group("val_mu_ic"))

    m = _RE_PANEL_MEAN_IC.search(text)
    if m:
        out["panel_mean_ic"] = float(m.group("mean_ic"))

    # Architecture flags actually fired
    m = _RE_SR_TASK.search(text)
    if m:
        out["layer1_sr_entries"]    = int(m.group(1))
        out["layer1_sr_tickers"]    = int(m.group(2))
        out["layer1_sr_feat_cols"]  = int(m.group(3))

    m = _RE_OH_TASK.search(text)
    if m:
        out["layer2_oh_entries"]    = int(m.group(1))
        out["layer2_oh_tickers"]    = int(m.group(2))
        out["layer2_oh_sectors"]    = int(m.group(3))

    # Status — guard fired? completed cleanly?
    if _RE_GUARD_FIRED.search(text):
        out["status"] = "min_best_iter_guard_fired"
    elif "panel-ltr" in text and "promote(" in text:
        out["status"] = "promoted"
    elif "panel-ltr" in text and "RunPanelTrainingTask: mean_ic" in text:
        out["status"] = "completed_no_promote"
    elif "FetchPanelDataTask: loaded" in text:
        out["status"] = "in_progress_or_crashed"
    else:
        out["status"] = "unknown"

    return out


def render_table(rows: list[dict]) -> str:
    """ASCII table of key metrics, sorted by label."""
    if not rows:
        return "(no experiments found)"
    headers = ["label", "layer1?", "layer2?", "train_ic",
               "cpcv_mean_ic", "cpcv_std", "n_splits", "status"]
    lines = []
    fmt_h = "{:<22} {:<8} {:<8} {:>10} {:>14} {:>10} {:>10} {:<28}"
    lines.append(fmt_h.format(*headers))
    lines.append("-" * 116)
    for r in sorted(rows, key=lambda x: x.get("label", "")):
        l1 = f"{r.get('layer1_sr_feat_cols','-')} cols" if "layer1_sr_feat_cols" in r else "off"
        l2 = f"{r.get('layer2_oh_sectors','-')} sec" if "layer2_oh_sectors" in r else "off"
        train_ic = (f"{r['train_ic']:+.4f}"
                    if "train_ic" in r else "—")
        cpcv = (f"{r['cpcv_mean_ic']:+.4f}"
                if "cpcv_mean_ic" in r else "—")
        std  = (f"{r['cpcv_std']:.4f}"
                if "cpcv_std" in r else "—")
        n    = (str(r["cpcv_n_splits"]) if "cpcv_n_splits" in r else "—")
        lines.append(fmt_h.format(
            r.get("label", "")[:22], l1, l2, train_ic, cpcv, std, n,
            r.get("status", "")[:28],
        ))
    return "\n".join(lines)


def discover_logs(roots: list[Path]) -> list[Path]:
    """Find candidate experiment logs in the conventional dirs."""
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            out.append(root)
        elif root.is_dir():
            out.extend(sorted(root.glob("*.log")))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", nargs="*", default=None,
                   help="Specific log files to compare. Default: glob "
                        "/tmp/wl*.log + /tmp/aa_half_*.log.")
    p.add_argument("--json", default=None,
                   help="Optional output JSON path for machine-readable.")
    args = p.parse_args()

    if args.logs:
        log_paths = [Path(x) for x in args.logs]
    else:
        # Conventional locations the dispatch scripts use.
        candidates = [
            "/tmp/aa_half_a.log",
            "/tmp/aa_half_b.log",
            "/tmp/wl178_layer1.log",
            "/tmp/wl178_layer1_2.log",
            "/tmp/b2_baseline.log",
        ]
        log_paths = [Path(p) for p in candidates if Path(p).exists()]

    if not log_paths:
        log.error("No experiment logs found. Pass --logs explicitly.")
        return 1

    rows = [parse_log(p) for p in log_paths]
    print()
    print(render_table(rows))
    print()

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        log.info("Wrote machine-readable to %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
