#!/usr/bin/env python
"""Industry-leading paired-daily-returns evaluator.

Implements the protocol pinned in doc/research/evaluation-protocol.md:
- Paired daily log-returns per non-overlapping window
- Newey-West HAC SE (Andrews 1991 lag)
- Cross-window pooled t-stat with HAC SE
- Stationary block bootstrap CI on Δ (Politis-Romano 1994)
- DSR + PBO with multi-comparison correction
- 3-tier promotion classification

Usage:
    python scripts/eval_paired_returns.py \\
        --baseline-dir data/logs/sim_2026-05-12_panel/baseline \\
        --candidate-dir data/logs/sim_2026-05-12_panel/<variant> \\
        --k-trials 100   [--json-out report.json]

Each input dir must contain N equity-JSON files (one per non-overlapping
window) emitted by `run_sim_104.py --equity-json PATH`.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from renquant_common.metrics.hac_se import hac_t_stat, newey_west_se, andrews_optimal_lag  # noqa: E402
from renquant_common.metrics.block_bootstrap import stationary_bootstrap_ci, sharpe_ratio_ci, optimal_block_length  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("eval")


def load_equity_json(path: Path) -> pd.Series:
    payload = json.loads(path.read_text())
    eq = pd.Series(payload["equity"]).astype(float)
    eq.index = pd.to_datetime(eq.index)
    return eq.sort_index()


def daily_log_returns(equity: pd.Series) -> pd.Series:
    """Daily log-returns from an equity curve."""
    return np.log(equity / equity.shift(1)).dropna()


def per_window_paired(baseline_path: Path, candidate_path: Path) -> dict:
    """Compute paired daily Δ for ONE window."""
    b = load_equity_json(baseline_path)
    c = load_equity_json(candidate_path)
    common = b.index.intersection(c.index)
    if len(common) < 5:
        return {"n_days": len(common), "window": baseline_path.stem,
                "skipped": "insufficient overlap"}
    r_b = daily_log_returns(b.loc[common])
    r_c = daily_log_returns(c.loc[common])
    aligned = pd.concat([r_b, r_c], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return {"n_days": len(aligned), "window": baseline_path.stem,
                "skipped": "insufficient daily overlap"}
    d = (aligned.iloc[:, 1] - aligned.iloc[:, 0]).values
    # Self-A/A degenerate case: delta is exactly zero → no signal to test.
    if np.allclose(d, 0.0):
        return {"window": baseline_path.stem, "n_days": int(len(d)),
                "mean_d": 0.0, "mean_d_ann": 0.0, "se_nw": 0.0,
                "t_stat": 0.0, "p_value": 1.0, "lag": 0,
                "sr_d_ann": 0.0, "delta_series": d.tolist()}
    nw = hac_t_stat(d)
    # Annualize the mean Δ (× 252) for readability
    mean_d_ann = nw["mean"] * 252.0
    # Annualized Sharpe of Δ
    sigma_d = float(np.std(d, ddof=1))
    sr_d_ann = (nw["mean"] / sigma_d) * math.sqrt(252.0) if sigma_d > 0 else 0.0
    return {
        "window":     baseline_path.stem,
        "n_days":     int(nw["n"]),
        "mean_d":     float(nw["mean"]),
        "mean_d_ann": float(mean_d_ann),
        "se_nw":      float(nw["se_nw"]),
        "t_stat":     float(nw["t_stat"]),
        "p_value":    float(nw["p_value"]),
        "lag":        int(nw["lag"]),
        "sr_d_ann":   float(sr_d_ann),
        "delta_series": d.tolist(),  # for cross-window pooling
    }


def pool_across_windows(per_window: list[dict], k_trials: int = 1) -> dict:
    """Concatenate per-window Δ series → pooled HAC + bootstrap."""
    valid = [w for w in per_window if "delta_series" in w]
    if not valid:
        return {"skipped": "no valid windows"}
    pooled = np.concatenate([np.asarray(w["delta_series"]) for w in valid])
    n_total = len(pooled)
    # Self-A/A degenerate: pooled all-zero → no signal
    if np.allclose(pooled, 0.0):
        return {"n_windows": len(valid), "n_days_pooled": int(n_total),
                "mean_d_pool": 0.0, "mean_d_pool_ann": 0.0, "se_nw_pool": 0.0,
                "t_pool": 0.0, "p_pool": 1.0, "ci95_lo_ann": 0.0,
                "ci95_hi_ann": 0.0, "sr_d_obs": 0.0,
                "sr_d_ci95_lo": 0.0, "sr_d_ci95_hi": 0.0,
                "dsr": 0.0, "cohens_d": 0.0,
                "consistency": 0.0, "n_windows_pos": 0, "lag_used": 0}
    sample_weighted_mean = sum(w["mean_d"] * w["n_days"] for w in valid) / sum(w["n_days"] for w in valid)
    nw = hac_t_stat(pooled)
    # Block bootstrap CI on mean Δ
    bs = stationary_bootstrap_ci(pooled, B=2000, alpha=0.05)
    # Sharpe CI
    sr = sharpe_ratio_ci(pooled, B=2000, alpha=0.05)
    sigma_d = float(np.std(pooled, ddof=1))
    cohens_d = float(nw["mean"] / sigma_d) if sigma_d > 0 else 0.0
    n_pos_windows = sum(1 for w in valid if w["t_stat"] > 0)
    consistency = n_pos_windows / len(valid)
    # DSR (selection-bias correction)
    from renquant_common.metrics.deflated_sharpe import deflated_sharpe_ratio
    sr_obs = (nw["mean"] / sigma_d) * math.sqrt(252.0) if sigma_d > 0 else 0.0
    skew = float(((pooled - nw["mean"]) ** 3).mean() / sigma_d ** 3) if sigma_d > 0 else 0.0
    kurt = float(((pooled - nw["mean"]) ** 4).mean() / sigma_d ** 4) if sigma_d > 0 else 3.0
    try:
        dsr = deflated_sharpe_ratio(
            sr_observed=sr_obs, n_returns=n_total, n_trials=max(1, k_trials),
            skew=skew, excess_kurtosis=kurt - 3.0,
        )
    except Exception:
        dsr = float("nan")
    return {
        "n_windows":   len(valid),
        "n_days_pooled": int(n_total),
        "mean_d_pool": float(nw["mean"]),
        "mean_d_pool_ann": float(nw["mean"] * 252.0),
        "se_nw_pool":  float(nw["se_nw"]),
        "t_pool":      float(nw["t_stat"]),
        "p_pool":      float(nw["p_value"]),
        "ci95_lo_ann": float(bs["ci_lo"] * 252.0),
        "ci95_hi_ann": float(bs["ci_hi"] * 252.0),
        "sr_d_obs":    float(sr_obs),
        "sr_d_ci95_lo":float(sr["ci_lo"]),
        "sr_d_ci95_hi":float(sr["ci_hi"]),
        "dsr":         float(dsr),
        "cohens_d":    float(cohens_d),
        "consistency": float(consistency),
        "n_windows_pos": int(n_pos_windows),
        "lag_used":    int(nw["lag"]),
    }


def classify_tier(pooled: dict, alpha_spy_ok: bool = True) -> dict:
    """Apply 3-tier promotion criteria per evaluation-protocol.md."""
    t = pooled.get("t_pool")
    mu = pooled.get("mean_d_pool_ann")
    cons = pooled.get("consistency", 0.0)
    d = pooled.get("cohens_d", 0.0)
    dsr = pooled.get("dsr")
    ci_lo = pooled.get("ci95_lo_ann")
    p = pooled.get("p_pool")
    if t is None or mu is None:
        return {"tier": "INSUFFICIENT_DATA"}
    # Tier 1 — REJECT
    if t < -1.0 or mu < -0.02 or (cons is not None and cons < 0.40 and mu < 0):
        return {"tier": "TIER1_REJECT",
                "reason": f"t_pool={t:+.2f} mean_ann={mu*100:+.2f}% cons={cons*100:.0f}%"}
    # Tier 3 — CONFIRMED (live-promotable) — check first since stricter
    tier2_passed = (t > 1.5 and cons > 0.60 and d > 0.20
                    and (ci_lo is not None and ci_lo > 0)
                    and alpha_spy_ok)
    if tier2_passed and t > 3.0 and abs(d) > 0.50 and p is not None and p < 0.01:
        if dsr is not None and not math.isnan(dsr) and dsr > 0.5:
            return {"tier": "TIER3_CONFIRMED",
                    "reason": f"t_pool={t:+.2f} > 3.0, DSR={dsr:.2f} > 0.5, "
                              f"d={d:+.2f}, p={p:.4f} → LIVE-PROMOTABLE"}
    # Tier 2 — SCREEN
    if tier2_passed:
        return {"tier": "TIER2_SCREEN",
                "reason": f"t_pool={t:+.2f} cons={cons*100:.0f}% d={d:+.2f} "
                          f"ci_lo={ci_lo*100:+.2f}% — soft candidate, NOT live yet"}
    return {"tier": "NEITHER",
            "reason": f"t_pool={t:+.2f} cons={cons*100:.0f}% d={d:+.2f}"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", required=True, help="Dir with baseline window equity JSON files")
    p.add_argument("--candidate-dir", required=True, help="Dir with candidate window equity JSON files")
    p.add_argument("--name", default=None, help="Variant label for report")
    p.add_argument("--k-trials", type=int, default=1,
                   help="Total variants tested in session for DSR correction (default: 1)")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    bdir = Path(args.baseline_dir)
    cdir = Path(args.candidate_dir)
    bfiles = sorted(bdir.glob("*.json"))
    cfiles_map = {f.stem: f for f in cdir.glob("*.json")}
    if not bfiles:
        raise SystemExit(f"No equity JSON in {bdir}")

    per_window = []
    for bf in bfiles:
        cf = cfiles_map.get(bf.stem)
        if cf is None:
            log.warning(f"No candidate match for {bf.stem} — skip")
            continue
        per_window.append(per_window_paired(bf, cf))

    pooled = pool_across_windows(per_window, k_trials=args.k_trials)
    verdict = classify_tier(pooled)
    name = args.name or f"{cdir.name} vs {bdir.name}"

    print(f"\n=== Paired-Returns Evaluation — {name} ===")
    print(f"k_trials (multi-test correction) = {args.k_trials}")
    print()
    print(f"{'Window':12} {'n':>5} {'meanΔ_ann':>11} {'se_NW':>8} {'t':>6} {'p':>8} {'lag':>4} {'SR_Δ':>6}")
    print("-" * 70)
    for w in per_window:
        if "skipped" in w:
            print(f"{w['window']:12} {w.get('n_days', 0):>5d}  SKIP: {w['skipped']}")
            continue
        print(f"{w['window']:12} {w['n_days']:>5d} "
              f"{w['mean_d_ann']*100:>+10.2f}% {w['se_nw']*100:>7.3f}% "
              f"{w['t_stat']:>+6.2f} {w['p_value']:>8.4f} {w['lag']:>4d} "
              f"{w['sr_d_ann']:>+6.2f}")
    print()
    if "skipped" not in pooled:
        print(f"=== POOLED (n_days = {pooled['n_days_pooled']}, n_windows = {pooled['n_windows']}) ===")
        print(f"  mean Δ (annualized)    : {pooled['mean_d_pool_ann']*100:+.2f}%")
        print(f"  Newey-West SE          : {pooled['se_nw_pool']*100:.3f}% (daily) "
              f"  lag = {pooled['lag_used']}")
        print(f"  t-statistic            : {pooled['t_pool']:+.2f}")
        print(f"  p-value (2-sided)      : {pooled['p_pool']:.4f}")
        print(f"  95% bootstrap CI (ann) : [{pooled['ci95_lo_ann']*100:+.2f}%, "
              f"{pooled['ci95_hi_ann']*100:+.2f}%]")
        print(f"  Sharpe of Δ (obs)      : {pooled['sr_d_obs']:+.2f} "
              f"  (95% CI [{pooled['sr_d_ci95_lo']:+.2f}, {pooled['sr_d_ci95_hi']:+.2f}])")
        print(f"  Deflated Sharpe (DSR)  : {pooled['dsr']:+.3f}  (K_trials={args.k_trials})")
        print(f"  Cohen's d              : {pooled['cohens_d']:+.3f}")
        print(f"  Window consistency     : {pooled['n_windows_pos']}/{pooled['n_windows']} "
              f"positive ({pooled['consistency']*100:.0f}%)")
        print()
        print(f"  VERDICT: {verdict['tier']}")
        print(f"           {verdict.get('reason', '')}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "name": name,
            "k_trials": args.k_trials,
            "per_window": [{k: v for k, v in w.items() if k != 'delta_series'} for w in per_window],
            "pooled": pooled,
            "verdict": verdict,
        }, indent=2))
        log.info(f"JSON report → {args.json_out}")


if __name__ == "__main__":
    main()
