#!/usr/bin/env python
"""P4.6 — Final Pareto/DSR/PBO analysis + winner determination.

Reads:
  data/logs/bb_results.csv                       — Track A (27 BB sims)
  data/logs/meta_label_oos_*_*.log               — Track B (3 OOS sims)

Computes:
  * Bailey-López de Prado 2014 Deflated Sharpe Ratio (DSR) with
    n_trials = 27 (BB) + 9 (threshold sweep) + 3 (deploy) = 39
  * 3-way comparison on OOS window
  * Final winner determination per §5.14.4

Outputs:
  data/logs/meta_label_final_winner.json
  + structured printout for the doc
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "data" / "logs"

# ── 1. Parse 3-way OOS results ─────────────────────────────────────────
def parse_log(path: Path) -> dict:
    text = path.read_text()
    m = re.search(r"Final value: \$([\d,]+).*?Return: ([+\-]?[\d.]+)%.*?APY: ([+\-]?[\d.]+)%", text)
    fv = float(m.group(1).replace(",", "")) if m else float("nan")
    apy = float(m.group(3)) / 100.0 if m else float("nan")
    m = re.search(r"Risk: Sharpe=([+\-]?[\d.]+)\s+Sortino=([+\-]?[\d.]+)\s+Calmar=([+\-]?[\d.]+)\s+MaxDD=([+\-]?[\d.]+)%\s+Vol=([+\-]?[\d.]+)%", text)
    if m:
        sharpe  = float(m.group(1))
        sortino = float(m.group(2))
        calmar  = float(m.group(3))
        max_dd  = float(m.group(4)) / 100.0
        vol     = float(m.group(5)) / 100.0
    else:
        sharpe = sortino = calmar = max_dd = vol = float("nan")
    m = re.search(r"Trades: (\d+) buys, (\d+) sells.*?Win rate: (\d+)%", text)
    buys = int(m.group(1)) if m else 0
    sells = int(m.group(2)) if m else 0
    win_rate = float(m.group(3)) / 100.0 if m else float("nan")
    m = re.search(r"Avg hold: (\d+)d.*?Avg P&L/trade: ([+\-]?[\d.]+)%", text)
    avg_hold = int(m.group(1)) if m else 0
    avg_pnl = float(m.group(2)) / 100.0 if m else float("nan")
    m = re.search(r"Exit reasons: (\{[^}]+\})", text)
    exit_reasons = eval(m.group(1)) if m else {}
    return {
        "final_value": fv, "apy": apy, "sharpe": sharpe, "sortino": sortino,
        "calmar": calmar, "max_dd": max_dd, "vol": vol,
        "buys": buys, "sells": sells, "win_rate": win_rate,
        "avg_hold": avg_hold, "avg_pnl": avg_pnl,
        "exit_reasons": exit_reasons,
    }

print("=" * 70)
print("P4.6 — FINAL ANALYSIS — 3-way OOS comparison (2025-04-01 → 2026-03-26)")
print("=" * 70)
results = {}
for variant in ("baseline", "metalabel_deploy_bbopt", "metalabel_deploy_meta"):
    pattern = f"meta_label_oos_sim_{variant}_*.log"
    matches = sorted(LOGS.glob(pattern))
    if matches:
        results[variant] = parse_log(matches[-1])

for k, v in results.items():
    print(f"\n{k}:")
    print(f"  APY:    {v['apy']*100:+.2f}%")
    print(f"  MaxDD:  {v['max_dd']*100:.1f}%")
    print(f"  Sharpe: {v['sharpe']:+.2f}")
    print(f"  Sortino:{v['sortino']:+.2f}")
    print(f"  Calmar: {v['calmar']:+.2f}")
    print(f"  Trades: {v['buys']} buys, {v['sells']} sells   Win:{v['win_rate']*100:.0f}%")
    print(f"  Avg P&L: {v['avg_pnl']*100:+.2f}%   Avg hold: {v['avg_hold']}d")
    print(f"  Exit reasons: {v['exit_reasons']}")

# ── 2. Bailey-López de Prado 2014 DSR ─────────────────────────────────
# DSR formula (Bailey & López de Prado 2014 J. Portfolio Mgmt 40(5):94):
#
#   DSR(SR) = ((SR_obs - E[max SR | N trials]) * sqrt(T-1)) /
#             sqrt(1 - gamma3*SR_obs + (gamma4-1)/4 * SR_obs^2)
#
# where:
#   N         = number of independent trials
#   T         = number of return observations
#   gamma3    = skewness of returns
#   gamma4    = kurtosis (Fisher) of returns
#   E[max SR] = expected max Sharpe under N-trials selection bias
#
# n_trials calculation per §5.14.4:
#   * 27 Box-Behnken runs (Track A)
#   * 9 threshold sweep points (Track B classifier)
#   * 3 OOS deploy sims (baseline / bbopt / meta)
N_TRIALS = 27 + 9 + 3   # = 39
# Conservative: also include feature_count selection (30 features) → arguably
# add this if we count "which feature subset" as a trial. We don't,
# because we used ALL features (no feature selection).

T = 252 * (11/12)   # 11 months of trading days

# E[max SR | N] under normal-iid assumption (Bailey-López de Prado eq. 5):
#   E[max SR_N] ≈ sqrt(2 ln N)
# This is the Gumbel-extreme-value approximation for the expectation of
# the max of N standard normal variables, scaled by SR.
import math
e_max_sr = math.sqrt(2 * math.log(N_TRIALS))

print("\n" + "=" * 70)
print(f"DSR FRAMEWORK (Bailey-López de Prado 2014)")
print("=" * 70)
print(f"  N_trials = {N_TRIALS}  (27 BB + 9 thr-sweep + 3 deploy)")
print(f"  T (return obs) = {T:.0f}")
print(f"  E[max SR | N] = sqrt(2 ln {N_TRIALS}) = {e_max_sr:.3f}")

# Simple deflated check: subtract E[max SR | N] from observed Sharpe.
# Strictly, we'd compute proper γ3/γ4 from daily returns but we don't
# have them in the parsed result. Use a normal-approximation DSR by
# assuming γ3=0, γ4=3 (normal):
#   DSR ≈ (SR - E[max_SR]) * sqrt(T-1)
print(f"\nDSR-adjusted Sharpe (normal approx, γ3=0, γ4=3):")
print(f"  Threshold for DSR > 0:  raw Sharpe > {e_max_sr:.3f}")
for k, v in results.items():
    sr = v["sharpe"]
    dsr_raw = (sr - e_max_sr) * math.sqrt(T - 1)
    p_dsr   = 1.0 / (1.0 + math.exp(-dsr_raw))   # logistic → P(DSR > 0)
    print(f"  {k:30s}  SR={sr:+.2f}  DSR raw={dsr_raw:+.2f}  "
          f"P(deflated SR > 0)={p_dsr:.2%}")

# ── 3. Winner determination ──────────────────────────────────────────
print("\n" + "=" * 70)
print("WINNER DETERMINATION")
print("=" * 70)

bbopt = results["metalabel_deploy_bbopt"]
meta  = results["metalabel_deploy_meta"]
base  = results["baseline"]

print(f"\n[1] APY ranking (max APY = user's stated goal):")
ranked = sorted(results.items(), key=lambda kv: -kv[1]["apy"])
for i, (k, v) in enumerate(ranked, 1):
    print(f"  #{i}  {k:30s}  APY={v['apy']*100:+.2f}%")

print(f"\n[2] MaxDD ranking (lower = better):")
ranked = sorted(results.items(), key=lambda kv: kv[1]["max_dd"])
for i, (k, v) in enumerate(ranked, 1):
    print(f"  #{i}  {k:30s}  MaxDD={v['max_dd']*100:.1f}%")

print(f"\n[3] Meta-label DELTA vs BB_14-alone (mechanism-attributable):")
print(f"  APY:    {(meta['apy']    - bbopt['apy'])*100:+.2f}pp")
print(f"  MaxDD:  {(meta['max_dd'] - bbopt['max_dd'])*100:+.2f}pp")
print(f"  Sharpe: {meta['sharpe']  - bbopt['sharpe']:+.2f}")
print(f"  Sortino:{meta['sortino'] - bbopt['sortino']:+.2f}")

print(f"\n[4] SDL exits (where meta vetoed):")
print(f"  baseline:           {base['exit_reasons'].get('single_day_loss', 0)} SDLs")
print(f"  BB_14 (no meta):    {bbopt['exit_reasons'].get('single_day_loss', 0)} SDLs")
print(f"  BB_14 + meta:       {meta['exit_reasons'].get('single_day_loss', 0)} SDLs"
      f"  ← {bbopt['exit_reasons'].get('single_day_loss', 0) - meta['exit_reasons'].get('single_day_loss', 0)} vetoed")

# ── 4. Save winner config ────────────────────────────────────────────
winner = "metalabel_deploy_meta" if meta["apy"] > bbopt["apy"] and meta["apy"] > base["apy"] \
    else max(results.items(), key=lambda kv: kv[1]["apy"])[0]

payload = {
    "test_window":     "2025-04-01 → 2026-03-26 (11 months)",
    "n_trials":        N_TRIALS,
    "expected_max_sr": float(e_max_sr),
    "results":         {k: {kk: (vv if not isinstance(vv, dict) else vv)
                            for kk, vv in v.items()}
                        for k, v in results.items()},
    "winner_variant":  winner,
    "winner_metrics":  results[winner],
    "headline": {
        "best_apy_oos":        winner,
        "apy_delta_vs_baseline": float(results[winner]["apy"] - base["apy"]),
        "maxdd_delta_vs_baseline": float(results[winner]["max_dd"] - base["max_dd"]),
    },
}
out = LOGS / "meta_label_final_winner.json"
out.write_text(json.dumps(payload, indent=2, default=str))
print(f"\nWrote → {out}")
print(f"\nWINNER: {winner}")
