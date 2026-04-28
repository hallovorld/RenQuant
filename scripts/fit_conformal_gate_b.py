#!/usr/bin/env python
"""Fit regime-conditional Gate B threshold via conformal prediction (M3).

Replaces the hardcoded ``ranking.panel_scoring.quality_floor.edge_sharpe_floor.threshold``
(currently 0.10) with a per-regime threshold ``τ_r`` chosen so the
historical false-discovery rate (FDR) stays below a target.

Definition (per regime r)::

    candidates_at_τ = { c : edge_sharpe(c) ≥ τ AND regime(c) == r }
    FDR(τ_r)        = #{ c ∈ candidates_at_τ : fwd_5d_relative(c) ≤ 0 }
                       / #{ candidates_at_τ }
    τ_r             = min{ τ : FDR(τ) ≤ target_fdr }

Output: ``backtesting/renquant_104/artifacts/gate_b_thresholds.json``::

    {
      "fitted_at": "2026-04-28T00:18:00",
      "target_fdr": 0.30,
      "horizon_days": 5,
      "min_samples_per_regime": 100,
      "thresholds": {
        "BULL_CALM": 0.082,
        "BULL_VOLATILE": 0.137,
        "CHOPPY": 0.158,
        "BEAR": 0.250
      },
      "fit_stats": { ... }
    }

QualityFloorTask reads this file (broker-isolated path through state_paths
util) and uses the regime-keyed τ. Falls back to the static config
threshold when the file is missing or this regime has insufficient history.

Usage:
    python scripts/fit_conformal_gate_b.py
    python scripts/fit_conformal_gate_b.py --target-fdr 0.25 --horizon 5
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ART_DIR = REPO_ROOT / "backtesting" / "renquant_104" / "artifacts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fit-conformal-gate-b")


def load_calibration_rows(
    db_path: Path,
    horizon: int,
    benchmark: str = "SPY",
    min_run_date: str | None = None,
) -> list[dict]:
    """Pull (regime, edge_sharpe, label) tuples from the runs DB."""
    if not db_path.exists():
        log.error("DB not found: %s", db_path)
        return []
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Pull candidate rows joined with run-level regime + ticker forward returns.
    # The benchmark forward return is fetched once per date for relative-return computation.
    where = ""
    params: list = []
    if min_run_date:
        where = "WHERE pr.run_date >= ?"
        params.append(min_run_date)

    sql = f"""
    SELECT
        pr.regime              AS regime,
        cs.ticker              AS ticker,
        cs.mu                  AS mu,
        cs.sigma               AS sigma,
        tfr.fwd_{horizon}d     AS fwd_ticker,
        bfr.fwd_{horizon}d     AS fwd_bench,
        pr.run_date            AS run_date
    FROM candidate_scores cs
    INNER JOIN pipeline_runs pr ON cs.run_id = pr.run_id
    LEFT  JOIN ticker_forward_returns tfr
              ON tfr.ticker     = cs.ticker
             AND tfr.as_of_date = pr.run_date
    LEFT  JOIN ticker_forward_returns bfr
              ON bfr.ticker     = ?
             AND bfr.as_of_date = pr.run_date
    {where}
      AND cs.role IN ('cand', 'candidate')
    """
    full_params = [benchmark] + params
    rows: list[dict] = []
    for r in c.execute(sql, full_params).fetchall():
        regime, ticker, mu, sigma, fwd_t, fwd_b, run_date = r
        if mu is None or sigma is None or sigma <= 0:
            continue
        if fwd_t is None or fwd_b is None:
            continue
        edge = mu / sigma
        # Label = 1 if ticker BEAT benchmark over the horizon, else 0.
        # Conformal "false discovery" = candidate that the model said BUY
        # but underperformed SPY. This matches the panel-LTR objective.
        label = 1 if (fwd_t > fwd_b) else 0
        rows.append({
            "regime": regime,
            "edge": edge,
            "label": label,
            "ticker": ticker,
            "run_date": run_date,
        })
    conn.close()
    return rows


def fit_regime_threshold(
    rows: list[dict],
    target_fdr: float,
    tau_grid: list[float],
    min_samples: int,
) -> tuple[float | None, dict]:
    """Smallest τ such that empirical FDR ≤ target. Returns (τ, stats)."""
    n_total = len(rows)
    if n_total < min_samples:
        return None, {"reason": "insufficient_samples", "n": n_total}

    # Negative-class rate at the unconditional level (sanity baseline)
    base_fdr = sum(1 for r in rows if r["label"] == 0) / n_total

    fits = []
    for tau in tau_grid:
        admitted = [r for r in rows if r["edge"] >= tau]
        if not admitted:
            fits.append({"tau": tau, "n": 0, "fdr": None})
            continue
        fdr = sum(1 for r in admitted if r["label"] == 0) / len(admitted)
        fits.append({"tau": tau, "n": len(admitted), "fdr": fdr})

    chosen = None
    for f in fits:
        if f["fdr"] is None:
            continue
        if f["fdr"] <= target_fdr and f["n"] >= max(20, min_samples // 5):
            chosen = f["tau"]
            break

    return chosen, {
        "n_total": n_total,
        "base_fdr": round(base_fdr, 4),
        "fits": fits,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db",         default="data/runs.alpaca.db",
                   help="SQLite source (default: alpaca live db)")
    p.add_argument("--horizon",    type=int, default=5,
                   help="Forward-return horizon in days for FDR labels")
    p.add_argument("--target-fdr", type=float, default=0.30,
                   help="Maximum tolerable FDR per regime")
    p.add_argument("--benchmark",  default="SPY")
    p.add_argument("--min-run-date", default=None,
                   help="Optional ISO date floor — exclude runs before this "
                        "(useful to filter out pre-bug-fix runs)")
    p.add_argument("--min-samples", type=int, default=100,
                   help="Minimum candidates per regime to fit τ")
    p.add_argument("--out",        default=str(ART_DIR / "gate_b_thresholds.json"))
    args = p.parse_args()

    db_path = REPO_ROOT / args.db if not Path(args.db).is_absolute() else Path(args.db)
    log.info("Loading rows from %s (horizon=%dd, since=%s)",
             db_path, args.horizon, args.min_run_date or "all")

    rows = load_calibration_rows(
        db_path, horizon=args.horizon, benchmark=args.benchmark,
        min_run_date=args.min_run_date,
    )
    log.info("Loaded %d candidate rows with valid (mu/sigma, fwd) labels", len(rows))

    if not rows:
        log.error("No rows. Check db path + that ticker_forward_returns is populated.")
        return 2

    # Group by regime
    by_regime: dict[str, list[dict]] = {}
    for r in rows:
        by_regime.setdefault(r["regime"] or "UNKNOWN", []).append(r)

    log.info("Regime counts: %s", {k: len(v) for k, v in by_regime.items()})

    tau_grid = [round(0.02 + 0.005 * i, 4) for i in range(60)]   # 0.02 → 0.32 in 0.005 steps

    thresholds: dict[str, float] = {}
    fit_stats: dict[str, dict] = {}
    for regime, regime_rows in by_regime.items():
        chosen, stats = fit_regime_threshold(
            regime_rows, args.target_fdr, tau_grid, args.min_samples,
        )
        fit_stats[regime] = stats
        if chosen is None:
            log.warning("regime=%-14s — could not fit τ (%s)",
                        regime, stats.get("reason", "no τ achieves target FDR"))
            continue
        thresholds[regime] = chosen
        log.info("regime=%-14s — τ=%.3f  base_fdr=%.3f  n=%d",
                 regime, chosen, stats["base_fdr"], stats["n_total"])

    out = {
        "fitted_at":   datetime.datetime.utcnow().isoformat(),
        "horizon_days":          args.horizon,
        "target_fdr":            args.target_fdr,
        "min_samples_per_regime": args.min_samples,
        "min_run_date":          args.min_run_date,
        "thresholds":  thresholds,
        "fit_stats":   fit_stats,
        "source_db":   str(db_path),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
