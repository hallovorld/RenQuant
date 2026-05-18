#!/usr/bin/env python3
"""Rigorous statistics on a batch of regime-reeval panels.

Replaces `scripts/analyze_regime_stratified.py` for verdict-grade
analysis. The existing analyzer reports pooled mean + Wilcoxon p but
NEITHER deflates for multiple comparisons (multiple knobs tested)
NOR provides a bootstrap CI on the pooled Δ. Calling a -0.23pp
"WIN-CONDITIONAL" on n=1 BEAR window with a single-seed sim is not
a verdict — it is a noise label.

This script consumes the EXISTING per-window equity JSONs (zero new
sim compute) and produces:

  For each treatment panel (vs a chosen baseline panel):
    • Per-window paired Δ APY = treatment.apy − baseline.apy
    • Pooled mean Δ ± stationary block bootstrap 95% CI
      (Politis & Romano 1994 via `arch.bootstrap.StationaryBootstrap`)
    • Newey-West HAC t-stat (statsmodels OLS, maxlags = N**(1/4))
    • Deflated t-stat / probabilistic Sharpe (Bailey & López de Prado
      2014 §3) with n_trials = number of knobs tested in the batch

  For the BATCH (all treatments together):
    • PBO via CSCV (Bailey, Borwein, López de Prado & Zhu 2015) —
      single value across the batch, P(best-IS strategy ranks below
      median in OOS) using 16-choose-8 = 12,870 splits subsampled to
      1,000.

Verdicts:
  REAL_EFFECT   bootstrap CI excludes 0  AND  deflated t > 0
  SUSPECT       CI excludes 0 BUT deflated t ≤ 0 (passes raw but
                fails multiple-comparison correction)
  NULL          CI includes 0 (cannot reject H_0)

Usage::

    python scripts/analyze_panels_rigorous.py \\
      --baseline data/logs/sim_2026-05-16_re_kelly_t1_035 \\
      --treatments data/logs/sim_2026-05-16_re_stop007 \\
                   data/logs/sim_2026-05-16_re_sdl_n2 \\
                   data/logs/sim_2026-05-16_re_trail015 \\
                   data/logs/sim_2026-05-16_re_cvar025 \\
                   data/logs/sim_2026-05-16_re_cvar050 \\
      --out doc/research/2026-05-16-rigorous-verdicts.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from arch.bootstrap import StationaryBootstrap
except ImportError:
    StationaryBootstrap = None

try:
    import statsmodels.api as sm
except ImportError:
    sm = None


# ── helpers ──────────────────────────────────────────────────────────────

def load_panel_apys(panel_dir: Path) -> dict[str, float]:
    """Return {Q01: apy, Q02: apy, …} sorted by window label."""
    out = {}
    eq_dir = panel_dir / "equity"
    for f in sorted(eq_dir.glob("[QW]*.json")):
        d = json.loads(f.read_text())
        out[f.stem] = float(d["apy"])
    return out


def paired_deltas(base: dict[str, float], treat: dict[str, float]) -> np.ndarray:
    keys = sorted(set(base) & set(treat))
    return np.array([treat[k] - base[k] for k in keys])


def stationary_bootstrap_ci(
    deltas: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
    block_len: float | None = None,
) -> tuple[float, float]:
    """Politis-Romano 1994 stationary bootstrap CI on pooled mean.

    Block length defaults to N**(1/3) per Politis & White 2004 rule-of-thumb.
    """
    if StationaryBootstrap is None:
        # Fallback: simple iid percentile bootstrap
        rng = np.random.default_rng(42)
        means = [rng.choice(deltas, size=len(deltas), replace=True).mean()
                 for _ in range(n_boot)]
        return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))
    n = len(deltas)
    bl = block_len if block_len is not None else max(1.0, n ** (1 / 3))
    bs = StationaryBootstrap(bl, deltas, seed=42)
    boot_means = np.empty(n_boot)
    for i, b in enumerate(bs.bootstrap(n_boot)):
        boot_means[i] = b[0][0].mean()
    return float(np.percentile(boot_means, 100 * alpha / 2)), float(np.percentile(boot_means, 100 * (1 - alpha / 2)))


def newey_west_tstat(deltas: np.ndarray) -> tuple[float, float]:
    """Mean-of-deltas t-stat with Newey-West HAC SE. Returns (t, p_two_sided)."""
    if sm is None:
        # Fallback: vanilla t
        mean = deltas.mean()
        se = deltas.std(ddof=1) / math.sqrt(len(deltas))
        from scipy import stats as ss
        t = mean / se if se > 0 else 0.0
        p = 2 * (1 - ss.t.cdf(abs(t), len(deltas) - 1))
        return t, p
    n = len(deltas)
    X = np.ones(n)
    model = sm.OLS(deltas, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(n ** 0.25)})
    return float(model.tvalues[0]), float(model.pvalues[0])


def deflated_tstat(
    deltas: np.ndarray, n_trials: int, all_tstats: Sequence[float],
) -> tuple[float, float]:
    """Deflated Sharpe (Bailey & López de Prado 2014 §3), adapted to
    paired-Δ mean-zero test.

    Treats the per-window Δ APY series as the strategy return series.
    DSR = P(observed_SR > expected_max_SR_under_H0).

    expected_max_SR = E[max(SR_i)] over n_trials random strategies given
    the cross-trial variance of SRs.

    Returns (deflated_t, dsr_probability).
    """
    n = len(deltas)
    if n < 3:
        return 0.0, 0.0
    sr_obs = deltas.mean() / max(deltas.std(ddof=1), 1e-12)
    sigma_sr = np.std(np.array(all_tstats) / math.sqrt(n), ddof=1) if len(all_tstats) > 1 else 0.0
    # Expected max-SR under H_0 (Bailey-López de Prado 2014 eq. 7) using
    # the Hall et al. approximation
    e_max = 0.0
    if n_trials > 1 and sigma_sr > 0:
        gamma = 0.5772156649  # Euler-Mascheroni
        e_max = sigma_sr * ((1 - gamma) * stats_ppf(1 - 1 / n_trials) +
                            gamma * stats_ppf(1 - 1 / (n_trials * math.e)))
    # Series moments
    g3 = float(((deltas - deltas.mean()) ** 3).mean() / max(deltas.std(ddof=1) ** 3, 1e-12))
    g4 = float(((deltas - deltas.mean()) ** 4).mean() / max(deltas.std(ddof=1) ** 4, 1e-12))
    denom = math.sqrt(max(1 - g3 * sr_obs + (g4 - 1) / 4 * sr_obs ** 2, 1e-6))
    psr = sr_obs * math.sqrt(max(n - 1, 1)) / denom
    # Deflated t (subtract e_max for selection bias)
    deflated = (sr_obs - e_max) * math.sqrt(max(n - 1, 1)) / denom
    from scipy.stats import norm
    return float(deflated), float(norm.cdf(deflated))


def stats_ppf(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(p))


def pbo_via_cscv(
    deltas_by_strat: dict[str, np.ndarray], n_splits: int = 1000,
) -> float | None:
    """Probability of Backtest Overfitting (Bailey-Borwein-López de Prado-
    Zhu 2015 §2.4) via Combinatorial Symmetric CV.

    Treats each strategy's per-window Δ APY series as the "returns".
    Splits the N windows into halves ways, computes IS Sharpe and OOS
    Sharpe of each strategy on each half, asks: how often is the best-IS
    strategy ranked below median in OOS?

    Returns single PBO value for the batch (NOT per-strategy).
    """
    strats = list(deltas_by_strat.keys())
    S = len(strats)
    if S < 2:
        return None
    arr = np.array([deltas_by_strat[s] for s in strats])
    N = arr.shape[1]
    if N < 4:
        return None
    half = N // 2
    # Enumerate all C(N, half) splits; subsample if too large
    all_idx = list(itertools.combinations(range(N), half))
    if len(all_idx) > n_splits:
        rng = random.Random(42)
        all_idx = rng.sample(all_idx, n_splits)
    n_overfit = 0
    n_total = 0
    for idx_is in all_idx:
        idx_is = set(idx_is)
        idx_oos = [i for i in range(N) if i not in idx_is]
        idx_is_l = sorted(idx_is)
        is_sr = [arr[s, idx_is_l].mean() / max(arr[s, idx_is_l].std(ddof=1), 1e-12) for s in range(S)]
        oos_sr = [arr[s, idx_oos].mean() / max(arr[s, idx_oos].std(ddof=1), 1e-12) for s in range(S)]
        best_is = int(np.argmax(is_sr))
        oos_rank = np.argsort(np.argsort(oos_sr))[best_is]  # 0 = worst
        if oos_rank < S / 2:
            n_overfit += 1
        n_total += 1
    return n_overfit / n_total if n_total else None


def verdict(ci: tuple[float, float], deflated_t: float) -> str:
    if ci[0] <= 0 <= ci[1]:
        return "NULL"
    if deflated_t > 0:
        return "REAL_EFFECT"
    return "SUSPECT_MULTI_COMP"


# ── main ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--treatments", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="markdown report path; printed to stdout if absent")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--n-cscv-splits", type=int, default=1000)
    args = p.parse_args()

    base_apys = load_panel_apys(args.baseline)
    if not base_apys:
        print(f"ERROR: baseline {args.baseline} has no equity windows", file=__import__("sys").stderr)
        return 2

    n_trials = len(args.treatments)
    # First pass: collect per-strat deltas + raw t-stats for DSR's sigma_sr
    deltas_by_strat: dict[str, np.ndarray] = {}
    t_raw_by_strat: dict[str, float] = {}
    for tdir in args.treatments:
        treat_apys = load_panel_apys(tdir)
        deltas = paired_deltas(base_apys, treat_apys)
        deltas_by_strat[tdir.name] = deltas
        t, _ = newey_west_tstat(deltas)
        t_raw_by_strat[tdir.name] = t
    all_tstats = list(t_raw_by_strat.values())

    # Second pass: full per-strat report
    lines = []
    lines.append(f"# Rigorous verdicts: {len(args.treatments)} treatments vs baseline `{args.baseline.name}`\n")
    lines.append(f"- bootstrap iterations: {args.n_boot} (stationary block, block ≈ N^(1/3))")
    lines.append(f"- DSR n_trials: {n_trials} (= number of knobs tested in this batch)")
    lines.append(f"- CSCV splits: {args.n_cscv_splits} (or all C(N, N/2) if fewer)")
    lines.append(f"- t-stat: Newey-West HAC (maxlags = N^(1/4))\n")

    # Per-strategy table
    lines.append("| treatment | N windows | mean Δ APY | 95% CI (block bootstrap) | NW t | NW p | deflated t | DSR | VERDICT |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    per_strat = []
    for tdir in args.treatments:
        deltas = deltas_by_strat[tdir.name]
        n = len(deltas)
        mean = float(deltas.mean())
        ci_lo, ci_hi = stationary_bootstrap_ci(deltas, n_boot=args.n_boot)
        t_nw, p_nw = newey_west_tstat(deltas)
        deflated, dsr = deflated_tstat(deltas, n_trials=n_trials, all_tstats=all_tstats)
        v = verdict((ci_lo, ci_hi), deflated)
        per_strat.append({
            "treatment": tdir.name, "N": n,
            "mean_delta_apy": mean,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "nw_t": t_nw, "nw_p": p_nw,
            "deflated_t": deflated, "dsr": dsr,
            "verdict": v,
        })
        lines.append(
            f"| {tdir.name} | {n} | {mean:+.4f} | [{ci_lo:+.4f}, {ci_hi:+.4f}] | "
            f"{t_nw:+.2f} | {p_nw:.3f} | {deflated:+.2f} | {dsr:.2f} | **{v}** |"
        )

    # Batch-level PBO
    lines.append("\n## Batch-level overfit probability\n")
    pbo = pbo_via_cscv(deltas_by_strat, n_splits=args.n_cscv_splits)
    if pbo is None:
        lines.append("- PBO: not computable (need ≥2 strategies, ≥4 windows)")
    else:
        lines.append(f"- **PBO (CSCV) = {pbo:.3f}**  ({pbo*100:.1f}%)")
        if pbo > 0.5:
            lines.append("- Bailey-López de Prado threshold: PBO > 0.5 ⇒ batch is overfit; no strategy in this batch should be promoted as if it were a single-test discovery.")
        else:
            lines.append("- PBO ≤ 0.5: batch is not obviously overfit. Per-strategy verdicts above still need DSR > 0.5 to be Tier-3 promotable.")

    # Reasoning footer
    lines.append("\n## How to read these verdicts\n")
    lines.append("- **REAL_EFFECT** — 95% block-bootstrap CI on the per-window paired Δ APY excludes 0, AND deflated t-stat (selection-bias corrected for n_trials peers) is positive. This is the only verdict that warrants further confirmation work (multi-seed, larger panel).")
    lines.append("- **SUSPECT_MULTI_COMP** — CI excludes 0 but the deflated t-stat is ≤0. Effect is real in a single-test sense but disappears under multiple-comparison correction. Re-run as a single-hypothesis test (not as part of a sweep) before believing.")
    lines.append("- **NULL** — bootstrap CI contains 0. Cannot reject the null that the knob has no effect; pooled mean is within noise. Don't deploy. Don't run a follow-up; the next compute is better spent on a different hypothesis.\n")

    # Append JSON for tool consumption
    lines.append("```json")
    lines.append(json.dumps({
        "baseline": str(args.baseline),
        "n_trials": n_trials,
        "per_strategy": per_strat,
        "pbo": pbo,
    }, indent=2))
    lines.append("```")

    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
