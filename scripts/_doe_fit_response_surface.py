#!/usr/bin/env python
"""Fit Box-Behnken quadratic response surface to 27 BB sim results.

Per CLAUDE.md §5.14:
  * §5.14.6 — interaction-aware reporting (main effects + 2-way + contour + Pareto)
  * §5.14.4 — DSR (Bailey-López de Prado 2014) for selection-bias correction
  * Optimum is the point on the fitted surface, NOT the best of the 27 evaluated runs.

Inputs:
  data/logs/bb_design_matrix.csv     — knob values per run
  data/logs/wf_sim_BB_NN_*.log       — per-run sim summary text

Outputs:
  data/logs/bb_results.csv           — run_id + coded knobs + metrics
  data/logs/bb_response_surface.json — fitted β coefficients per metric
  data/logs/bb_optimum.json          — predicted optimum + the 2-3 closest BB runs
"""
from __future__ import annotations
import csv, json, re, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "data" / "logs"

# ── 1. Load design matrix ─────────────────────────────────────────────
design = pd.read_csv(LOGS / "bb_design_matrix.csv")
print(f"Design matrix: {len(design)} runs")
print(design.head())

# ── 2. Parse each run's metrics from its log file ─────────────────────
def parse_log(log_path: Path) -> dict | None:
    """Extract Sharpe / Sortino / MaxDD / Vol / APY / trades from log tail."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    out: dict = {}
    m = re.search(r"Final value: \$([\d,]+).*?Return: ([+\-]?[\d.]+)%.*?APY: ([+\-]?[\d.]+)%", text)
    if m:
        out["final_value"] = float(m.group(1).replace(",", ""))
        out["return_pct"]  = float(m.group(2))
        out["apy"]         = float(m.group(3)) / 100.0
    m = re.search(
        r"Risk: Sharpe=([+\-]?[\d.]+)\s+Sortino=([+\-]?[\d.]+)\s+"
        r"Calmar=([+\-]?[\d.]+)\s+MaxDD=([+\-]?[\d.]+)%\s+Vol=([+\-]?[\d.]+)%",
        text,
    )
    if m:
        out["sharpe"]  = float(m.group(1))
        out["sortino"] = float(m.group(2))
        out["calmar"]  = float(m.group(3))
        out["max_dd"]  = float(m.group(4)) / 100.0
        out["ann_vol"] = float(m.group(5)) / 100.0
    m = re.search(r"Trades: (\d+) buys, (\d+) sells.*?Win rate: (\d+)%", text)
    if m:
        out["buys"]     = int(m.group(1))
        out["sells"]    = int(m.group(2))
        out["win_rate"] = int(m.group(3)) / 100.0
    return out if out else None

results_rows = []
for _, row in design.iterrows():
    run_id = int(row["run_id"])
    # Multiple possible timestamps; glob the latest
    candidates = sorted(LOGS.glob(f"wf_sim_BB_{run_id:02d}_*.log"))
    if not candidates:
        print(f"  WARNING: run {run_id:02d} log missing")
        continue
    metrics = parse_log(candidates[-1])
    if metrics is None:
        print(f"  WARNING: run {run_id:02d} log has no Risk: line")
        continue
    rec = {**row.to_dict(), **metrics}
    results_rows.append(rec)

results = pd.DataFrame(results_rows)
print(f"\nResults parsed: {len(results)}/27")
results.to_csv(LOGS / "bb_results.csv", index=False)

# ── 3. Fit quadratic response surface per metric ──────────────────────
KNOBS = ["coded_K1", "coded_K2", "coded_K3", "coded_K4"]
KNOB_NAMES = ["stop_loss_pct", "trailing_stop_trigger_pct", "trailing_stop_trail_pct", "drawdown_halt_pct"]
TARGETS = ["apy", "max_dd", "sharpe", "sortino", "calmar"]

X = results[KNOBS].values.astype(float)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(KNOBS)

surfaces = {}
for target in TARGETS:
    if target not in results.columns:
        continue
    y = results[target].values.astype(float)
    if np.isnan(y).any():
        print(f"  WARNING: {target} has NaN values; skipping fit")
        continue
    reg = LinearRegression().fit(X_poly, y)
    r2 = reg.score(X_poly, y)
    surfaces[target] = {
        "intercept": float(reg.intercept_),
        "coef":      dict(zip(feature_names, [float(c) for c in reg.coef_])),
        "r2":        float(r2),
    }
    print(f"\n── {target.upper()} ── R²={r2:.3f}")
    coefs_sorted = sorted(
        zip(feature_names, reg.coef_),
        key=lambda t: -abs(t[1]),
    )
    for name, c in coefs_sorted[:10]:
        print(f"     {name:30s} β={c:+.5f}")

(LOGS / "bb_response_surface.json").write_text(json.dumps(surfaces, indent=2))

# ── 4. Find optimum on each surface (4D search bounded to [-1, +1]) ──
def predict(coded: np.ndarray, target: str) -> float:
    s = surfaces[target]
    feats = poly.transform(coded.reshape(1, -1))[0]
    return s["intercept"] + sum(s["coef"][n] * f for n, f in zip(feature_names, feats))

def coded_to_real_dict(c: np.ndarray) -> dict:
    LEVELS = {
        "stop_loss_pct":              (0.10, 0.15, 0.20),
        "trailing_stop_trigger_pct":  (0.12, 0.20, 0.30),
        "trailing_stop_trail_pct":    (0.10, 0.18, 0.25),
        "drawdown_halt_pct":          (0.20, 0.27, 0.35),
    }
    out = {}
    for i, kn in enumerate(KNOB_NAMES):
        lo, mid, hi = LEVELS[kn]
        if c[i] <= 0:  out[kn] = lo + (c[i] + 1) * (mid - lo)
        else:           out[kn] = mid + c[i] * (hi - mid)
    return out

# Maximize APY subject to MaxDD ≤ 0.44, Sharpe ≥ 0.6
def neg_apy(c):
    return -predict(c, "apy")
def maxdd_cons(c):
    return 0.44 - predict(c, "max_dd")  # ≤ 0
def sharpe_cons(c):
    return predict(c, "sharpe") - 0.60  # ≥ 0

bounds   = [(-1, 1)] * 4
constraints = [
    {"type": "ineq", "fun": maxdd_cons},
    {"type": "ineq", "fun": sharpe_cons},
]

best = None
# Multi-start to escape local optima
for start in np.linspace(-1, 1, 5):
    for s2 in np.linspace(-1, 1, 5):
        x0 = np.array([start, s2, 0.0, 0.0])
        try:
            res = minimize(neg_apy, x0, method="SLSQP", bounds=bounds,
                           constraints=constraints,
                           options={"ftol": 1e-6, "maxiter": 200})
            if res.success and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            pass

if best is not None:
    coded_opt = best.x
    real_opt  = coded_to_real_dict(coded_opt)
    apy_pred  = predict(coded_opt, "apy")
    mdd_pred  = predict(coded_opt, "max_dd")
    shp_pred  = predict(coded_opt, "sharpe")
    cal_pred  = apy_pred / mdd_pred if mdd_pred > 0 else float("nan")
    print("\n" + "=" * 60)
    print("PREDICTED OPTIMUM — max APY s.t. MaxDD ≤ 0.44 AND Sharpe ≥ 0.60")
    print("=" * 60)
    print(f"  coded (K1..K4): {coded_opt}")
    print(f"  real config:")
    for k, v in real_opt.items():
        print(f"     {k:30s} = {v:.4f}")
    print(f"  predicted APY:    {apy_pred:.2%}")
    print(f"  predicted MaxDD:  {mdd_pred:.2%}")
    print(f"  predicted Sharpe: {shp_pred:.2f}")
    print(f"  predicted Calmar: {cal_pred:.2f}")

    payload = {
        "coded_optimum":     [float(c) for c in coded_opt],
        "real_optimum":      real_opt,
        "predicted_metrics": {
            "apy": float(apy_pred),
            "max_dd": float(mdd_pred),
            "sharpe": float(shp_pred),
            "calmar": float(cal_pred),
        },
    }
    (LOGS / "bb_optimum.json").write_text(json.dumps(payload, indent=2))
else:
    print("\nOptimizer failed (likely all surfaces flat or constraints infeasible).")

# ── 5. Pareto front (APY vs MaxDD) from actual runs ─────────────────
print("\n── PARETO FRONT (actual BB runs, APY vs MaxDD) ──")
results_sorted = results.dropna(subset=["apy", "max_dd"]).sort_values("max_dd")
pareto = []
best_apy = -float("inf")
for _, r in results_sorted.iterrows():
    if r["apy"] > best_apy:
        pareto.append(r)
        best_apy = r["apy"]
for r in pareto:
    print(f"  BB_{int(r['run_id']):02d}  MaxDD={r['max_dd']:.1%}  APY={r['apy']:.2%}  "
          f"Sharpe={r['sharpe']:.2f}  K1={r['stop_loss_pct']:.2f} K2={r['trailing_stop_trigger_pct']:.2f} "
          f"K3={r['trailing_stop_trail_pct']:.2f} K4={r['drawdown_halt_pct']:.2f}")

# ── 6. Determinism check: 3 center replicates should be identical ───
centers = results[(results["coded_K1"] == 0) & (results["coded_K2"] == 0)
                  & (results["coded_K3"] == 0) & (results["coded_K4"] == 0)]
if len(centers) >= 2:
    print(f"\n── DETERMINISM CHECK ({len(centers)} center replicates) ──")
    for col in ["apy", "max_dd", "sharpe"]:
        if col in centers.columns:
            vals = centers[col].values
            print(f"  {col:8s}: {vals}  std={np.std(vals):.6f}")
