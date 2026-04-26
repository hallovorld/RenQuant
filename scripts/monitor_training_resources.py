#!/usr/bin/env python
"""Sample a running training PID's CPU + RSS at fixed intervals.

Writes one CSV row per sample to ``logs/retrain_panel/{date}.resources.csv``
so the post-training plotter can reconstruct the resource profile.

Designed to run as a background companion to a long training process.
The wrapper script (``scripts/retrain_panel.sh``) kicks this off after
launching ``sunday_panel_sweep.py`` and kills it when sweep completes.

CSV schema:
    ts            float   epoch seconds
    iso_ts        str     ISO-8601 with seconds resolution
    pid           int     sampled PID
    cmd_short     str     last 32 chars of cmdline (for dis-ambiguation)
    pcpu          float   percent of 1 core (>100 means multi-core)
    pmem          float   percent of system RAM
    rss_mb        float   resident set size in MB
    vsz_mb        float   virtual size in MB

Robust to:
  - Target PID disappearing mid-sample (logs a final NaN row + exits).
  - Subprocess fan-out: if --include-children given, sums rss across
    parent + descendants. Useful for sunday_panel_sweep which spawns
    one Python subprocess per backend.
  - Skipping silently if `psutil` not installed (writes a header-only
    CSV) — avoids forcing a hard dependency on the launchd path.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _has_psutil() -> bool:
    try:
        import psutil  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def _gather(pid: int, include_children: bool) -> dict | None:
    """Return one snapshot row dict or None if process gone."""
    import psutil  # noqa: PLC0415
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return None
    try:
        with p.oneshot():
            cpu = p.cpu_percent(interval=None)
            mem = p.memory_info()
            cmd = " ".join(p.cmdline())[-32:]
            rss = mem.rss
            vsz = mem.vms
            mem_pct = p.memory_percent()
        if include_children:
            for child in p.children(recursive=True):
                try:
                    with child.oneshot():
                        cpu += child.cpu_percent(interval=None)
                        rss += child.memory_info().rss
                        vsz += child.memory_info().vms
                        mem_pct += child.memory_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    return {
        "pcpu":   cpu,
        "pmem":   mem_pct,
        "rss_mb": rss / 1024 / 1024,
        "vsz_mb": vsz / 1024 / 1024,
        "cmd":    cmd,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", type=int, required=True,
                    help="PID to sample")
    ap.add_argument("--out",
                    default=f"logs/retrain_panel/"
                            f"{dt.date.today().isoformat()}.resources.csv",
                    help="output CSV path (relative to repo root)")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between samples (default 5)")
    ap.add_argument("--include-children", action="store_true", default=True,
                    help="sum across child processes too (default True)")
    ap.add_argument("--max-duration", type=float, default=14400,
                    help="hard cap in seconds (default 4h)")
    args = ap.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Header — always written, even if psutil missing.
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    f = out_path.open("a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "ts", "iso_ts", "pid", "cmd_short", "pcpu", "pmem",
            "rss_mb", "vsz_mb",
        ])
        f.flush()

    if not _has_psutil():
        # Silent degrade — header written, no rows. Plotter handles
        # empty CSV gracefully (renders log-derived gantt only).
        print(f"monitor: psutil not installed — wrote header to {out_path}")
        return 0

    # Reset cpu_percent baseline (first call returns 0)
    import psutil
    try:
        psutil.Process(args.pid).cpu_percent(interval=None)
    except psutil.NoSuchProcess:
        print(f"monitor: PID {args.pid} not found at start")
        return 1

    started = time.time()

    def _shutdown(*_):
        f.flush()
        f.close()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while time.time() - started < args.max_duration:
        snap = _gather(args.pid, args.include_children)
        now = dt.datetime.now()
        if snap is None:
            writer.writerow([
                f"{time.time():.3f}", now.isoformat(timespec="seconds"),
                args.pid, "", "", "", "", "",
            ])
            f.flush()
            print(f"monitor: PID {args.pid} ended at {now.isoformat()}")
            break
        writer.writerow([
            f"{time.time():.3f}", now.isoformat(timespec="seconds"),
            args.pid, snap["cmd"], f"{snap['pcpu']:.1f}",
            f"{snap['pmem']:.2f}", f"{snap['rss_mb']:.1f}",
            f"{snap['vsz_mb']:.1f}",
        ])
        f.flush()
        time.sleep(args.interval)

    f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
