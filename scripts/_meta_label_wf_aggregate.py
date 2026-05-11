#!/usr/bin/env python
"""Aggregate walk-forward 3-window meta-label OOS results into a single
table and write a summary JSON. Companion to _meta_label_walkforward.sh.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "data" / "logs"

def parse(path: Path) -> dict:
    text = path.read_text()
    out: dict = {}
    m = re.search(r"APY: ([+\-]?[\d.]+)%", text)
    out["apy"] = float(m.group(1)) / 100 if m else float("nan")
    m = re.search(r"Risk: Sharpe=([+\-]?[\d.]+)\s+Sortino=([+\-]?[\d.]+)\s+"
                  r"Calmar=([+\-]?[\d.]+)\s+MaxDD=([+\-]?[\d.]+)%", text)
    if m:
        out["sharpe"]  = float(m.group(1))
        out["sortino"] = float(m.group(2))
        out["calmar"]  = float(m.group(3))
        out["max_dd"]  = float(m.group(4)) / 100
    m = re.search(r"single_day_loss': (\d+)", text)
    out["sdl_exits"] = int(m.group(1)) if m else None
    m = re.search(r"Trades: (\d+) buys, (\d+) sells", text)
    if m:
        out["buys"]  = int(m.group(1))
        out["sells"] = int(m.group(2))
    return out

windows = ["W1", "W2", "W3"]
variants = ["sim_baseline", "sim_wf_W1_deploy_bbopt", "sim_wf_W1_deploy_meta",
            "sim_wf_W2_deploy_bbopt", "sim_wf_W2_deploy_meta",
            "sim_wf_W3_deploy_bbopt", "sim_wf_W3_deploy_meta"]

all_results: dict = {}
for W in windows:
    wdir = LOGS / f"wf_meta_{W}"
    if not wdir.exists():
        print(f"  WARNING: {wdir} missing")
        continue
    window_results = {}
    for f in sorted(wdir.glob("oos_*.log")):
        label = f.stem.replace("oos_", "")
        window_results[label] = parse(f)
    all_results[W] = window_results

# Print 3-way per window
print("=" * 80)
print("WALK-FORWARD 3-WINDOW META-LABEL VALIDATION")
print("=" * 80)
for W, results in all_results.items():
    print(f"\n── Window {W} ──")
    keys = sorted(results.keys())
    for k in keys:
        r = results[k]
        print(f"  {k:35s}  APY={r.get('apy',float('nan'))*100:+.2f}%  "
              f"MaxDD={r.get('max_dd',float('nan'))*100:.1f}%  "
              f"Sharpe={r.get('sharpe',float('nan')):+.2f}  "
              f"SDLs={r.get('sdl_exits','?')}")

# Compute meta-vs-bbopt delta per window
print("\n── Meta-label DELTA vs BB_14-alone per window ──")
print(f"{'window':<8}  {'ΔAPY pp':>10}  {'ΔMaxDD pp':>12}  {'ΔSharpe':>10}  {'ΔSDLs':>8}")
deltas = []
for W, results in all_results.items():
    bb = next((r for k, r in results.items() if "bbopt" in k), None)
    mt = next((r for k, r in results.items() if "meta"  in k), None)
    if bb is None or mt is None:
        continue
    d_apy   = (mt["apy"]    - bb["apy"])    * 100
    d_dd    = (mt["max_dd"] - bb["max_dd"]) * 100
    d_sh    = mt["sharpe"]  - bb["sharpe"]
    d_sdl   = (mt.get("sdl_exits") or 0) - (bb.get("sdl_exits") or 0)
    deltas.append({"window": W, "d_apy_pp": d_apy, "d_maxdd_pp": d_dd,
                   "d_sharpe": d_sh, "d_sdl": d_sdl})
    print(f"  {W:<8}  {d_apy:+10.2f}  {d_dd:+12.2f}  {d_sh:+10.2f}  {d_sdl:+8d}")

if deltas:
    mean_d_apy   = np.mean([d["d_apy_pp"]   for d in deltas])
    mean_d_dd    = np.mean([d["d_maxdd_pp"] for d in deltas])
    mean_d_sh    = np.mean([d["d_sharpe"]   for d in deltas])
    print(f"  {'MEAN':<8}  {mean_d_apy:+10.2f}  {mean_d_dd:+12.2f}  {mean_d_sh:+10.2f}")
    # Sign-test for robustness — how many of N windows have +APY delta
    pos_apy = sum(1 for d in deltas if d["d_apy_pp"] > 0)
    print(f"\n  Sign-count APY:    {pos_apy}/{len(deltas)} windows with meta-label > BB_14")
    pos_dd  = sum(1 for d in deltas if d["d_maxdd_pp"] < 0)
    print(f"  Sign-count MaxDD:  {pos_dd}/{len(deltas)} windows with meta-label < BB_14")

out = LOGS / "wf_meta_summary.json"
out.write_text(json.dumps({"windows": all_results, "deltas": deltas}, indent=2, default=str))
print(f"\nWrote → {out}")
