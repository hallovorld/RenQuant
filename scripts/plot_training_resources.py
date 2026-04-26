#!/usr/bin/env python
"""Render a polished resource-usage chart for a Sunday sweep.

Combines two data sources:

1. ``logs/retrain_panel/{date}.resources.csv`` — per-sample CPU/memory
   captured by ``monitor_training_resources.py``.

2. ``logs/retrain_panel/{date}.log`` — phase markers parsed via regex
   (Job START / DONE lines) to build a gantt strip.

Output: ``logs/retrain_panel/{date}.resources.png`` — a single tall
figure with three vertically-stacked panels:

    ┌──────────────────────────────────────────────┐
    │ Gantt strip — colored bars per Job phase      │
    ├──────────────────────────────────────────────┤
    │ CPU% over time (with per-backend shading)     │
    ├──────────────────────────────────────────────┤
    │ Memory RSS in MB (line + filled area)         │
    └──────────────────────────────────────────────┘

Designed to be self-contained — no plotly, no seaborn, just matplotlib
with a clean light theme. Looks production-grade for a quick `open`
command on macOS.

Usage:
    python scripts/plot_training_resources.py
    python scripts/plot_training_resources.py --date 2026-04-26
    python scripts/plot_training_resources.py --date 2026-04-26 \
        --resources-csv logs/retrain_panel/2026-04-26.resources.csv \
        --log logs/retrain_panel/2026-04-26.log \
        --out logs/retrain_panel/2026-04-26.resources.png
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Regex per log structure observed in real Sunday sweep runs.
# Phase boundaries:
#   "─── training backend=xgboost ───"
#   "── PanelDataJob START / DONE 3.6s"
#   "── PanelFeatureJob START / DONE Xs"
#   "── PanelModelJob START / DONE Xs"
#   "── PanelNGBoostJob START / DONE Xs"
#   "── RefreshPanelCalibratorJob START / DONE Xs"
#   "backend=X train exit=0 elapsed=Xs"
RE_TS    = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
RE_BACKEND = re.compile(r"training backend=(\w+)")
RE_JOB_START = re.compile(r"── (\w+Job) START")
RE_JOB_DONE  = re.compile(r"── (\w+Job) DONE\s+([\d.]+)s")
RE_BACKEND_DONE = re.compile(r"backend=(\w+) train exit=-?\d+\s+elapsed=([\d.]+)s")


PALETTE = {
    "PanelDataJob":               "#5B8DEF",
    "PanelFeatureJob":            "#5BC0BE",
    "PanelModelJob":              "#FFC857",
    "PanelNGBoostJob":            "#E76F51",
    "RefreshPanelCalibratorJob":  "#A06CD5",
    "BaselineTournamentJob":      "#7F8C8D",
    "RecalibrationJob":           "#A06CD5",
    "PanelTrainingJob":           "#34495E",
}
DEFAULT_COLOR = "#95A5A6"

BACKEND_BG = {
    "xgboost":     "#FFEFD5",   # papaya
    "lightgbm":    "#E0F4FF",   # ice
    "transformer": "#F4E3FF",   # lavender
}


def _parse_ts(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")


def parse_log_phases(log_path: Path) -> list[dict]:
    """Return list of phase dicts {start, end, name, backend}."""
    open_jobs: dict[str, dt.datetime] = {}
    phases: list[dict] = []
    backend_curr = "?"

    if not log_path.exists():
        print(f"warn: log not found at {log_path}", file=sys.stderr)
        return phases

    for line in log_path.open():
        ts_match = RE_TS.match(line)
        if not ts_match:
            continue
        ts = _parse_ts(ts_match.group(1))

        m_back = RE_BACKEND.search(line)
        if m_back:
            backend_curr = m_back.group(1)
            continue
        m_back_end = RE_BACKEND_DONE.search(line)
        if m_back_end:
            phases.append({
                "name":    f"BACKEND:{m_back_end.group(1)}",
                "start":   ts - dt.timedelta(seconds=float(m_back_end.group(2))),
                "end":     ts,
                "backend": m_back_end.group(1),
                "kind":    "backend",
            })
            continue
        m_start = RE_JOB_START.search(line)
        if m_start:
            open_jobs[m_start.group(1)] = ts
            continue
        m_done = RE_JOB_DONE.search(line)
        if m_done:
            name = m_done.group(1)
            elapsed = float(m_done.group(2))
            start = open_jobs.pop(name, ts - dt.timedelta(seconds=elapsed))
            phases.append({
                "name":    name,
                "start":   start,
                "end":     ts,
                "backend": backend_curr,
                "kind":    "job",
            })
    return phases


def parse_resources(csv_path: Path) -> list[dict]:
    """Return list of sample dicts."""
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "ts":     dt.datetime.fromisoformat(r["iso_ts"]),
                    "pcpu":   float(r["pcpu"]) if r["pcpu"] else float("nan"),
                    "rss_mb": float(r["rss_mb"]) if r["rss_mb"] else float("nan"),
                })
            except (ValueError, KeyError):
                continue
    return rows


def render(phases: list[dict], samples: list[dict], out_path: Path,
           date_str: str) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.dates as mdates  # noqa: PLC0415
    import matplotlib.patches as mpatches  # noqa: PLC0415

    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":         10,
        "axes.labelsize":    10,
        "axes.titlesize":    11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linewidth":    0.5,
    })

    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    gs  = fig.add_gridspec(3, 1, height_ratios=[1.4, 2, 2])
    ax_g, ax_c, ax_m = gs.subplots()

    # Determine x-range
    xs = []
    for ph in phases:
        xs += [ph["start"], ph["end"]]
    for s in samples:
        xs.append(s["ts"])
    if not xs:
        ax_g.text(0.5, 0.5, "no data", ha="center", va="center")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        return
    x_min, x_max = min(xs), max(xs)

    # ── Gantt panel ───────────────────────────────────────────────────────
    ax_g.set_title(f"Sunday Sweep — Resource Profile  ·  {date_str}",
                   fontsize=13, fontweight="bold", loc="left", pad=14)
    ax_g.set_ylabel("Phase", labelpad=8)
    ax_g.set_yticks([])

    # backend bands behind everything
    for ph in [p for p in phases if p["kind"] == "backend"]:
        bg = BACKEND_BG.get(ph["backend"], "#F8F8F8")
        ax_g.axvspan(ph["start"], ph["end"], facecolor=bg,
                     alpha=0.65, zorder=0)
        ax_c.axvspan(ph["start"], ph["end"], facecolor=bg,
                     alpha=0.45, zorder=0)
        ax_m.axvspan(ph["start"], ph["end"], facecolor=bg,
                     alpha=0.45, zorder=0)
        # Backend label at top of gantt
        mid = ph["start"] + (ph["end"] - ph["start"]) / 2
        ax_g.text(mid, 0.93, ph["backend"].upper(),
                  ha="center", va="top", fontsize=11, fontweight="bold",
                  alpha=0.55, transform=ax_g.get_xaxis_transform())

    # Lay out job bars in rows so they don't overlap visually
    job_phases = [p for p in phases if p["kind"] == "job"]
    job_phases.sort(key=lambda p: p["start"])
    rows: list[list[dict]] = []
    for ph in job_phases:
        placed = False
        for row in rows:
            if all(p["end"] <= ph["start"] or p["start"] >= ph["end"]
                   for p in row):
                row.append(ph)
                placed = True
                break
        if not placed:
            rows.append([ph])

    for row_i, row in enumerate(rows):
        y = row_i + 0.2
        for ph in row:
            color = PALETTE.get(ph["name"], DEFAULT_COLOR)
            duration = (ph["end"] - ph["start"]).total_seconds()
            ax_g.barh(y, ph["end"] - ph["start"], height=0.6,
                      left=ph["start"], color=color, edgecolor="#222",
                      linewidth=0.4, zorder=2)
            mid = ph["start"] + (ph["end"] - ph["start"]) / 2
            label = ph["name"].replace("Job", "")
            if duration > 60:
                label += f"  {duration:.0f}s"
            if duration > 30:
                ax_g.text(mid, y, label, ha="center", va="center",
                          fontsize=8.5, color="#222",
                          fontweight="medium", zorder=3)

    ax_g.set_ylim(-0.2, max(1, len(rows)) + 0.3)
    ax_g.invert_yaxis()
    ax_g.set_xlim(x_min, x_max)
    ax_g.tick_params(axis="x", labelbottom=False)

    # ── CPU panel ─────────────────────────────────────────────────────────
    ts_arr  = [s["ts"]   for s in samples]
    cpu_arr = [s["pcpu"] for s in samples]
    rss_arr = [s["rss_mb"] for s in samples]

    ax_c.set_ylabel("CPU %  (per core)", labelpad=8)
    if ts_arr:
        ax_c.plot(ts_arr, cpu_arr, color="#2E86AB", linewidth=1.6,
                  zorder=3, label="CPU")
        ax_c.fill_between(ts_arr, 0, cpu_arr, color="#2E86AB",
                          alpha=0.18, zorder=2)
        # Reference lines
        ax_c.axhline(100, color="#888", linestyle="--", linewidth=0.8,
                     alpha=0.5, label="1 core saturated")
        ax_c.axhline(800, color="#bbb", linestyle=":", linewidth=0.7,
                     alpha=0.4)
        ax_c.legend(loc="upper right", frameon=False, fontsize=8.5)
        # Annotation: peak CPU
        if cpu_arr:
            import math
            peak = max((c for c in cpu_arr if not math.isnan(c)),
                       default=0)
            ax_c.text(0.01, 0.95, f"peak {peak:.0f}%", fontsize=9,
                      color="#2E86AB", fontweight="bold",
                      transform=ax_c.transAxes, va="top")
    else:
        ax_c.text(0.5, 0.5, "no resource samples — run "
                  "monitor_training_resources.py during training",
                  ha="center", va="center", color="#888",
                  transform=ax_c.transAxes)
    ax_c.set_xlim(x_min, x_max)
    ax_c.tick_params(axis="x", labelbottom=False)

    # ── Memory panel ──────────────────────────────────────────────────────
    ax_m.set_ylabel("Memory RSS  (MB)", labelpad=8)
    if ts_arr:
        ax_m.plot(ts_arr, rss_arr, color="#E07A5F", linewidth=1.6,
                  zorder=3, label="RSS")
        ax_m.fill_between(ts_arr, 0, rss_arr, color="#E07A5F",
                          alpha=0.20, zorder=2)
        ax_m.legend(loc="upper right", frameon=False, fontsize=8.5)
        if rss_arr:
            import math
            peak = max((r for r in rss_arr if not math.isnan(r)),
                       default=0)
            ax_m.text(0.01, 0.95, f"peak {peak:.0f} MB", fontsize=9,
                      color="#E07A5F", fontweight="bold",
                      transform=ax_m.transAxes, va="top")
    ax_m.set_xlim(x_min, x_max)
    ax_m.set_xlabel("Wall clock", labelpad=6)
    ax_m.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_m.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

    # ── Legend in gantt ───────────────────────────────────────────────────
    handles = [mpatches.Patch(color=c, label=n.replace("Job", ""))
               for n, c in PALETTE.items()
               if any(p["name"] == n for p in job_phases)]
    if handles:
        ax_g.legend(handles=handles, loc="lower right",
                    frameon=False, ncols=min(4, len(handles)),
                    fontsize=8.5, bbox_to_anchor=(1.0, -0.18))

    # Footer summary stats
    total_dur = (x_max - x_min).total_seconds()
    n_jobs = len(job_phases)
    n_backends = len(set(p["backend"] for p in phases if p["kind"] == "backend"))
    fig.text(0.013, 0.005,
             f"{n_backends} backend(s) · {n_jobs} job phase(s) · total {total_dur/60:.1f} min",
             fontsize=8, color="#666", ha="left")

    fig.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor="white")
    print(f"plot → {out_path}")


def main() -> int:
    today = dt.date.today().isoformat()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=today,
                    help="date string YYYY-MM-DD; default today")
    ap.add_argument("--log",
                    default=None, help="log path override")
    ap.add_argument("--resources-csv",
                    default=None, help="resources CSV override")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    log_path = (Path(args.log) if args.log
                else REPO_ROOT / "logs/retrain_panel" / f"{args.date}.log")
    csv_path = (Path(args.resources_csv) if args.resources_csv
                else REPO_ROOT / "logs/retrain_panel" / f"{args.date}.resources.csv")
    out_path = (Path(args.out) if args.out
                else REPO_ROOT / "logs/retrain_panel" / f"{args.date}.resources.png")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    phases  = parse_log_phases(log_path)
    samples = parse_resources(csv_path)
    print(f"parsed {len(phases)} phases, {len(samples)} resource samples")
    render(phases, samples, out_path, args.date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
