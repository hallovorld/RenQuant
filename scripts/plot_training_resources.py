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
RE_BACKEND_DONE = re.compile(r"backend=(\w+) train exit=(-?\d+)\s+elapsed=([\d.]+)s")
# Sweep diagnostics
RE_CPCV  = re.compile(r"CrossValidateTask\[cpcv\]: mean=([+-][\d.]+) std=([\d.]+) "
                      r"q05=([+-][\d.]+) q50=([+-][\d.]+) q95=([+-][\d.]+) "
                      r"n_splits=(\d+)")
RE_FINAL = re.compile(r"FinalFitTask: backend=(\w+)\s+train_ic=([+-][\d.]+)\s+"
                      r"elapsed=([\d.]+)s\s+device=(\w+)")
RE_FEAT  = re.compile(r"^\s+([\w_]+)\s+std=\s*([\d.]+)\s+IC=([+-][\d.]+)")
RE_FEAT_HDR = re.compile(r"FeatureDiagnosticTask: per-feature within-date std")


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


def parse_log_phases(log_path: Path) -> tuple[list[dict], dict]:
    """Return (phases, diagnostics).

    diagnostics keys:
      backend_metrics  : dict[backend, {cpcv_mean, cpcv_std, q05, q95,
                                        n_splits, train_ic, elapsed_s,
                                        device, exit_code}]
      feature_ic       : dict[feature_name, IC_value] from latest
                         FeatureDiagnosticTask (highest-IC backend wins)
    """
    open_jobs: dict[str, dt.datetime] = {}
    phases: list[dict] = []
    backend_curr = "?"
    diag: dict = {"backend_metrics": {}, "feature_ic": {}}

    if not log_path.exists():
        print(f"warn: log not found at {log_path}", file=sys.stderr)
        return phases, diag

    in_feat_section = False
    feat_buf: dict[str, float] = {}

    for line in log_path.open():
        # Feature-diagnostic capture (multi-line block)
        if RE_FEAT_HDR.search(line):
            in_feat_section = True
            feat_buf = {}
            continue
        if in_feat_section:
            m_feat = RE_FEAT.match(line)
            if m_feat:
                feat_buf[m_feat.group(1)] = float(m_feat.group(3))
                continue
            # End of section when we hit a non-matching indented line
            if line.strip() and not line.startswith(" "):
                in_feat_section = False
                if feat_buf:
                    # Keep latest one per backend (overwrite OK — last
                    # backend's features mirror the prior backends).
                    diag["feature_ic"] = feat_buf

        ts_match = RE_TS.match(line)
        if not ts_match:
            continue
        ts = _parse_ts(ts_match.group(1))

        # CPCV mean/std/percentiles
        m_cpcv = RE_CPCV.search(line)
        if m_cpcv:
            be = backend_curr
            diag["backend_metrics"].setdefault(be, {})
            diag["backend_metrics"][be].update({
                "cpcv_mean": float(m_cpcv.group(1)),
                "cpcv_std":  float(m_cpcv.group(2)),
                "q05":       float(m_cpcv.group(3)),
                "q50":       float(m_cpcv.group(4)),
                "q95":       float(m_cpcv.group(5)),
                "n_splits":  int(m_cpcv.group(6)),
            })

        m_final = RE_FINAL.search(line)
        if m_final:
            be = m_final.group(1)
            diag["backend_metrics"].setdefault(be, {})
            diag["backend_metrics"][be].update({
                "train_ic":  float(m_final.group(2)),
                "fit_secs":  float(m_final.group(3)),
                "device":    m_final.group(4),
            })

        m_back = RE_BACKEND.search(line)
        if m_back:
            backend_curr = m_back.group(1)
            continue
        m_back_end = RE_BACKEND_DONE.search(line)
        if m_back_end:
            be = m_back_end.group(1)
            elapsed_s = float(m_back_end.group(3))
            phases.append({
                "name":    f"BACKEND:{be}",
                "start":   ts - dt.timedelta(seconds=elapsed_s),
                "end":     ts,
                "backend": be,
                "kind":    "backend",
            })
            diag["backend_metrics"].setdefault(be, {})
            diag["backend_metrics"][be].update({
                "elapsed_s":  elapsed_s,
                "exit_code":  int(m_back_end.group(2)),
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
    if feat_buf and not diag["feature_ic"]:
        diag["feature_ic"] = feat_buf
    return phases, diag


def _calibrator_health(strategy_dir: Path) -> dict:
    """Read per-backend calibrator artifacts and surface unique_y / pool_ic."""
    out: dict[str, dict] = {}
    art = strategy_dir / "artifacts"
    if not art.exists():
        return out
    for backend in ("xgboost", "lightgbm", "transformer"):
        p = art / f"panel-rank-calibration.{backend}.bak.json"
        if not p.exists():
            continue
        try:
            import json as _json
            d = _json.loads(p.read_text())
            ys = d.get("probability", {}).get("y", [])
            md = d.get("metadata", {})
            out[backend] = {
                "unique_y":  len(set(ys)) if ys else 0,
                "n_pts":     len(ys) if ys else 0,
                "pool_ic":   md.get("pool_ic"),
                "scorer_oos_ic": md.get("scorer_oos_mean_ic"),
                "y_min":     min(ys) if ys else None,
                "y_max":     max(ys) if ys else None,
            }
        except Exception as exc:
            out[backend] = {"error": str(exc)[:60]}
    return out


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
           date_str: str, diag: dict | None = None,
           calib_health: dict | None = None) -> None:
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

    diag = diag or {}
    calib_health = calib_health or {}

    # Two-column outer layout: left = timeline (5 rows), right = side
    # diagnostics (3 stacked panels). Heights tuned for legibility.
    fig = plt.figure(figsize=(17, 12), constrained_layout=False)
    outer = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0],
                             wspace=0.10, left=0.05, right=0.97,
                             top=0.97, bottom=0.05)
    left  = outer[0, 0].subgridspec(5, 1,
                                    height_ratios=[0.55, 1.4, 1.8, 1.8, 1.6],
                                    hspace=0.42)
    right = outer[0, 1].subgridspec(3, 1,
                                    height_ratios=[1.4, 1.0, 1.1],
                                    hspace=0.30)

    ax_h    = fig.add_subplot(left[0])      # title + summary table
    ax_g    = fig.add_subplot(left[1])      # gantt
    ax_c    = fig.add_subplot(left[2])      # cpu
    ax_m    = fig.add_subplot(left[3])      # memory
    ax_b    = fig.add_subplot(left[4])      # backend IC compare
    ax_feat = fig.add_subplot(right[0])     # per-feature IC bars
    ax_cal  = fig.add_subplot(right[1])     # calibrator health
    ax_meta = fig.add_subplot(right[2])     # metadata text

    # Header + summary table
    ax_h.axis("off")
    bm = diag.get("backend_metrics", {})
    if bm:
        rows = [["backend", "CPCV mean", "± std", "train IC",
                 "elapsed", "status"]]
        for be in ("xgboost", "lightgbm", "transformer"):
            d = bm.get(be, {})
            if not d:
                continue
            cpcv = d.get("cpcv_mean")
            std  = d.get("cpcv_std")
            tr   = d.get("train_ic")
            el   = d.get("elapsed_s")
            ec   = d.get("exit_code")
            status = ("✓ OK" if ec == 0
                      else f"✗ exit={ec}" if ec is not None else "?")
            rows.append([
                be,
                f"{cpcv:+.4f}" if cpcv is not None else "—",
                f"{std:.4f}" if std is not None else "—",
                f"{tr:+.4f}" if tr is not None else "—",
                f"{el:.0f}s" if el else "—",
                status,
            ])
        n_cols = len(rows[0])
        col_w  = [0.16, 0.16, 0.14, 0.16, 0.16, 0.22]
        x_off  = 0.0
        for j, h in enumerate(rows[0]):
            ax_h.text(x_off, 0.85, h, fontsize=10, fontweight="bold",
                      transform=ax_h.transAxes, color="#222")
            x_off += col_w[j]
        for i, row in enumerate(rows[1:]):
            x_off = 0.0
            y = 0.45 - i * 0.30
            be  = row[0]
            row_color = BACKEND_BG.get(be, "#FFFFFF")
            ax_h.add_patch(mpatches.Rectangle(
                (-0.005, y - 0.10), 1.01, 0.24, transform=ax_h.transAxes,
                facecolor=row_color, alpha=0.55, edgecolor="none",
                clip_on=False))
            for j, cell in enumerate(row):
                weight = "bold" if j == 0 else "normal"
                color = "#C0392B" if (j == 5 and "✗" in cell) else "#222"
                ax_h.text(x_off, y, cell, fontsize=10, fontweight=weight,
                          color=color, transform=ax_h.transAxes)
                x_off += col_w[j]

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

    # ── Backend IC comparison (left, bottom) ──────────────────────────────
    ax_b.set_title("OOS IC comparison — CPCV mean ± std vs train IC",
                   fontsize=10, loc="left")
    ax_b.set_ylabel("IC", labelpad=8)
    ax_b.axhline(0, color="#888", linewidth=0.6, alpha=0.6, zorder=1)
    ax_b.axhline(0.05, color="#3A9D5D", linestyle=":", linewidth=0.7,
                 alpha=0.5, label="ship gate")

    bm = diag.get("backend_metrics", {})
    backends_present = [b for b in ("xgboost", "lightgbm", "transformer")
                        if b in bm]
    width = 0.32
    for i, be in enumerate(backends_present):
        d = bm[be]
        x = i + 0.5
        cpcv = d.get("cpcv_mean")
        std = d.get("cpcv_std")
        tr = d.get("train_ic")
        color = BACKEND_BG.get(be, "#CCC").replace("D5", "B5")
        # darker for foreground
        edge_color = "#222"
        if cpcv is not None:
            ax_b.bar(x - width / 2, cpcv, width=width,
                     color=color, edgecolor=edge_color, linewidth=0.6,
                     zorder=2, label=f"{be} CPCV" if i == 0 else None)
            if std is not None:
                ax_b.errorbar(x - width / 2, cpcv, yerr=std,
                              fmt="none", color="#222", linewidth=1.0,
                              capsize=4, capthick=1, zorder=3)
            ax_b.text(x - width / 2, cpcv + (std or 0) * 1.05 + 0.003,
                      f"{cpcv:+.3f}", ha="center", fontsize=9,
                      fontweight="medium", color="#222")
        if tr is not None:
            ax_b.bar(x + width / 2, tr, width=width,
                     color=color, edgecolor="#888", linewidth=0.6,
                     hatch="//", alpha=0.6, zorder=2,
                     label=f"{be} train" if i == 0 else None)
            ax_b.text(x + width / 2, tr + 0.003, f"{tr:+.3f}",
                      ha="center", fontsize=9, color="#222")
    ax_b.set_xticks([i + 0.5 for i in range(len(backends_present))])
    ax_b.set_xticklabels([b.upper() for b in backends_present])
    ax_b.set_xlim(0, max(1, len(backends_present)))
    if backends_present:
        ymin = min(bm[b].get("cpcv_mean", 0) - bm[b].get("cpcv_std", 0)
                   for b in backends_present)
        ymax = max(bm[b].get("train_ic", 0)
                   for b in backends_present if bm[b].get("train_ic"))
        ax_b.set_ylim(min(ymin, -0.02) * 1.3, max(ymax, 0.10) * 1.2)
    ax_b.legend(loc="upper right", frameon=False, fontsize=8)

    # ── Right: per-feature IC (top) ───────────────────────────────────────
    feat_ic = diag.get("feature_ic", {})
    ax_feat.set_title("Top features by |IC|", fontsize=10, loc="left")
    if feat_ic:
        # Sort by absolute IC desc, take top 15
        items = sorted(feat_ic.items(), key=lambda x: abs(x[1]),
                       reverse=True)[:15]
        names = [n for n, _ in items]
        ics   = [v for _, v in items]
        colors = ["#5BC0BE" if v > 0 else "#E07A5F" for v in ics]
        y = list(range(len(names)))
        ax_feat.barh(y, ics, color=colors, edgecolor="#222",
                     linewidth=0.4)
        for yi, (n, v) in zip(y, items):
            ax_feat.text(v + (0.001 if v >= 0 else -0.001), yi,
                         f"{v:+.3f}", fontsize=8,
                         ha="left" if v >= 0 else "right",
                         va="center", color="#222")
        ax_feat.set_yticks(y)
        ax_feat.set_yticklabels([n[:24] for n in names], fontsize=8)
        ax_feat.invert_yaxis()
        ax_feat.axvline(0, color="#888", linewidth=0.6)
        ax_feat.set_xlabel("Spearman IC")
    else:
        ax_feat.text(0.5, 0.5, "no feature diagnostics in log",
                     ha="center", va="center", color="#888",
                     transform=ax_feat.transAxes)
        ax_feat.axis("off")

    # ── Right: calibrator health ──────────────────────────────────────────
    ax_cal.set_title("Calibrator health", fontsize=10, loc="left")
    ax_cal.axis("off")
    if calib_health:
        rows = [["backend", "unique y", "pool IC", "scorer OOS",
                 "ratio"]]
        for be in ("xgboost", "lightgbm", "transformer"):
            d = calib_health.get(be, {})
            if not d or "error" in d:
                continue
            uy   = d.get("unique_y", 0)
            pool = d.get("pool_ic")
            scor = d.get("scorer_oos_ic")
            ratio = ((scor / pool) if pool and pool > 1e-6 and scor
                     else None)
            rows.append([
                be,
                str(uy),
                f"{pool:.4f}" if pool is not None else "—",
                f"{scor:.4f}" if scor is not None else "—",
                f"{ratio:.0f}×" if ratio else "—",
            ])
        if len(rows) > 1:
            tbl = ax_cal.table(cellText=rows, loc="upper left",
                               cellLoc="left", colWidths=[0.22] * 5)
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            tbl.scale(1.0, 1.55)
            for j in range(len(rows[0])):
                tbl[(0, j)].set_facecolor("#EEE")
                tbl[(0, j)].set_text_props(weight="bold")
            for i in range(1, len(rows)):
                be = rows[i][0]
                color = BACKEND_BG.get(be, "#FFF")
                for j in range(len(rows[0])):
                    tbl[(i, j)].set_facecolor(color)
                # red flag — collapsed calibrator (unique y < 5 or
                # pool_ic / scorer ratio > 5×)
                uy = int(rows[i][1]) if rows[i][1].isdigit() else 999
                ratio_str = rows[i][4]
                ratio_val = float(ratio_str.rstrip("×")) if "×" in ratio_str else 1
                if uy < 5 or ratio_val > 5:
                    tbl[(i, 0)].set_text_props(weight="bold",
                                               color="#C0392B")
        else:
            ax_cal.text(0.5, 0.5, "no calibrator artifacts found",
                        ha="center", va="center", color="#888",
                        transform=ax_cal.transAxes)
    else:
        ax_cal.text(0.5, 0.5, "no calibrator artifacts found",
                    ha="center", va="center", color="#888",
                    transform=ax_cal.transAxes)

    # ── Right: metadata + footer ──────────────────────────────────────────
    ax_meta.axis("off")
    total_dur = (x_max - x_min).total_seconds()
    n_jobs = len(job_phases)
    n_backends = len(set(p["backend"] for p in phases
                         if p["kind"] == "backend"))
    n_ok = sum(1 for be, d in bm.items() if d.get("exit_code") == 0)
    info_lines = [
        f"Date              {date_str}",
        f"Total wall time   {total_dur / 60:.1f} min",
        f"Backends run      {n_backends}",
        f"Backends OK       {n_ok}/{n_backends}",
        f"Job phases        {n_jobs}",
    ]
    if samples:
        import math
        peak_cpu = max((s["pcpu"] for s in samples
                        if not math.isnan(s["pcpu"])), default=0)
        peak_rss = max((s["rss_mb"] for s in samples
                        if not math.isnan(s["rss_mb"])), default=0)
        info_lines += [
            f"Peak CPU          {peak_cpu:.0f}%",
            f"Peak RSS          {peak_rss:.0f} MB",
            f"Samples           {len(samples)}",
        ]
    info_lines.append(f"Generated         {dt.datetime.now():%Y-%m-%d %H:%M}")
    ax_meta.set_title("Run metadata", fontsize=10, loc="left")
    for i, line in enumerate(info_lines):
        ax_meta.text(0.02, 0.92 - i * 0.10, line, fontsize=9,
                     family="monospace", transform=ax_meta.transAxes,
                     color="#333")

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

    phases, diag = parse_log_phases(log_path)
    samples = parse_resources(csv_path)
    calib = _calibrator_health(REPO_ROOT / "backtesting/renquant_104")
    print(f"parsed {len(phases)} phases, {len(samples)} resource samples, "
          f"{len(diag.get('backend_metrics', {}))} backend metrics, "
          f"{len(diag.get('feature_ic', {}))} features, "
          f"{len(calib)} calibrators")
    render(phases, samples, out_path, args.date,
           diag=diag, calib_health=calib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
