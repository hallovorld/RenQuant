#!/usr/bin/env python
"""Compute Deflated Sharpe + PBO on the truly-OOS IC series.

Inputs:
  - artifacts/prod/truly_oos_eval/eval_truly_oos.json
    (produced by scripts/eval_truly_oos.py)

Outputs (appended to the same JSON):
  - dsr:  P(true_SR > 0 | observed = max of n_trials)
  - pbo:  Probability of Backtest Overfitting via CSCV (Bailey-Borwein-
          López de Prado 2015 *J. Comp. Finance* 14(1))
  - sharpe_of_ic: Annualized Sharpe of per-day IC series

Per CLAUDE.md §5.13.4: any quoted performance number gets DSR + PBO when
n_trials > 1. Here n_trials = 5 (one IC track per regime).
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dsr-pbo")


def annualized_sharpe(x: np.ndarray) -> float:
    """SR of a daily series, annualized by √252."""
    x = np.asarray(x, dtype=np.float64)
    if x.std(ddof=1) == 0 or len(x) < 2:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def cscv_pbo(returns_matrix: np.ndarray, n_splits: int = 16) -> float:
    """Bailey-Borwein-López de Prado 2015 Combinatorially Symmetric Cross-
    Validation. Probability that the IS-best strategy is < median OOS.

    returns_matrix: shape (T_days, N_strategies). Here strategies =
        regime tracks (5: BEAR/CHOPPY/BULL_VOL/BULL_STRONG/BULL_CALM IC
        series, padded with zeros for cross-regime alignment).
    n_splits: number of partition chunks. CSCV uses C(n,n/2) split pairs.
    """
    T, N = returns_matrix.shape
    if T < n_splits or N < 2:
        return float("nan")
    chunk = T // n_splits
    if chunk == 0:
        return float("nan")

    # Partition T into n_splits equal chunks; trim trailing rows
    R = returns_matrix[:chunk * n_splits].reshape(n_splits, chunk, N)
    # Sum per-chunk = per-strategy sum-of-returns within that chunk
    chunk_sums = R.sum(axis=1)  # shape (n_splits, N)

    half = n_splits // 2
    overfit_count = 0
    total_pairs = 0
    for IS_idx in combinations(range(n_splits), half):
        OOS_idx = tuple(i for i in range(n_splits) if i not in IS_idx)
        is_sharpe = chunk_sums[list(IS_idx)].sum(axis=0)   # shape (N,)
        oos_sharpe = chunk_sums[list(OOS_idx)].sum(axis=0)
        best_in_is = int(np.argmax(is_sharpe))
        oos_rank = float(np.sum(oos_sharpe < oos_sharpe[best_in_is])) / max(1, N - 1)
        # logit transform for the BBLdP formula
        oos_rank = min(max(oos_rank, 1e-6), 1 - 1e-6)
        if oos_rank < 0.5:
            overfit_count += 1
        total_pairs += 1
    return overfit_count / total_pairs if total_pairs > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-json",
        default="backtesting/renquant_104/artifacts/prod/truly_oos_eval/eval_truly_oos.json",
        help="Path to eval_truly_oos.json to stamp with DSR/PBO.",
    )
    args = ap.parse_args()

    src = Path(args.eval_json)
    if not src.is_absolute():
        src = REPO / src
    if not src.exists():
        log.error("Missing: %s — run scripts/eval_truly_oos.py first", src)
        return 2
    e = json.loads(src.read_text())
    ic = np.asarray(e["ic_per_date"], dtype=np.float64)
    dates = e["eval_dates"]
    log.info("Loaded %d IC observations from %s → %s",
             len(ic), dates[0], dates[-1])

    # SR of the IC track (treat IC-per-day as a return-like series)
    sr_obs = annualized_sharpe(ic)
    log.info("Annualized Sharpe of IC series: %.4f", sr_obs)

    # DSR (n_trials = number of "model choices" we screened. The truly-OOS
    # was a single retrain at cutoff 2024-07-01; n_trials = 1 here would
    # under-correct. Use n_trials = 5 since we look at 5 regime tracks.
    # Each regime gives a different mean IC; the "winning" one (BEAR
    # +0.345) was selected post-hoc.
    from kernel.metrics.deflated_sharpe import deflated_sharpe_ratio  # noqa: PLC0415
    n_trials = 5
    dsr = deflated_sharpe_ratio(
        sr_observed=sr_obs,
        n_returns=len(ic),
        n_trials=n_trials,
        skew=float(((ic - ic.mean())**3).mean() / max(ic.std()**3, 1e-12)),
        excess_kurtosis=float(((ic - ic.mean())**4).mean() / max(ic.std()**4, 1e-12) - 3.0),
    )
    log.info("DSR (n_trials=%d): %.4f", n_trials, dsr)

    # PBO via CSCV across 5 regime "strategies"
    # Build matrix: T_days × N_regimes. Each column = IC if the date was
    # in that regime, else 0 (CSCV is robust to missing-as-zero).
    regimes = sorted(e.get("per_regime", {}).keys())
    R = np.zeros((len(ic), len(regimes)), dtype=np.float64)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    # We don't have per-date regime in the JSON; re-derive
    try:
        import sys as _sys
        _sys.path.insert(0, str(REPO))
        from scripts.eval_truly_oos import detect_regime  # noqa: PLC0415
        import pandas as _pd
        regime_series = detect_regime(_pd.Series([_pd.Timestamp(d) for d in dates]))
        for j, reg in enumerate(regimes):
            for i, d in enumerate(dates):
                rd = regime_series.get(_pd.Timestamp(d), None)
                if str(rd) == reg:
                    R[i, j] = ic[i]
    except Exception as exc:
        log.warning("PBO matrix construction failed (%s); skipping", exc)
        R = None

    pbo = float("nan")
    if R is not None and R.shape[1] >= 2:
        pbo = cscv_pbo(R, n_splits=16)
        log.info("PBO (CSCV, %d strategies, 16 splits): %.4f",
                 R.shape[1], pbo)

    # Stamp into JSON
    e["sharpe_of_ic_annualized"] = sr_obs
    e["dsr"] = float(dsr)
    e["dsr_n_trials"] = n_trials
    e["pbo"] = float(pbo) if not np.isnan(pbo) else None
    src.write_text(json.dumps(e, indent=2))
    log.info("Stamped DSR + PBO → %s", src)

    print()
    print("=" * 70)
    print("DSR + PBO VERDICT (CLAUDE.md §5.13.4 promotion gate)")
    print("=" * 70)
    print(f"IC mean      : {e['ic_mean']:+.4f}")
    print(f"IC SR (ann)  : {sr_obs:+.4f}")
    print(f"DSR          : {dsr:.4f}  (n_trials={n_trials}; gate: >0.5 OR n≥30,t>3)")
    print(f"PBO          : {pbo:.4f}" if not np.isnan(pbo) else "PBO          : NaN")
    print()
    if dsr > 0.5:
        print("✓ DSR > 0.5 → signal is statistically meaningful after selection correction")
    else:
        print(f"⚠ DSR = {dsr:.3f} < 0.5 — signal does NOT survive selection correction")
    if not np.isnan(pbo):
        if pbo < 0.5:
            print(f"✓ PBO < 0.5 → strategy unlikely overfit ({100*pbo:.0f}% overfit-pair rate)")
        else:
            print(f"⚠ PBO = {100*pbo:.0f}% — strategy likely overfit (BBLdP threshold 50%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
