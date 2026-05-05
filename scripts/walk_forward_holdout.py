#!/usr/bin/env python
"""Walk-forward eval: run B2-style hold-out at multiple training cuts,
report Sharpe / APY consistency. Discriminates "real alpha" from
"single-cut lucky" by spanning rolling 12-month OOS periods.

Usage::
    python scripts/walk_forward_holdout.py \\
        --strategy-config-name strategy_config.json \\
        --cuts 2024-05-04 2024-11-04 2025-05-04 \\
        --oos-months 12 \\
        --output data/walk_forward_results/

Each cut runs a B2 hold-out (--skip-train, reuse current artifact) for
the OOS period [cut, cut + oos_months]. SPY benchmark computed for
each window. Writes per-cut JSON + a roll-up summary.

Note: this assumes the artifact is fixed across all cuts (no
per-cut retraining — that would be true walk-forward training).
True walk-forward needs per-cut retrain (~hours per cut). This
"forward eval" with fixed artifact is a quick sanity check on
"is the alpha consistent across periods?".
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import fmean, stdev

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("walk-forward")


def _add_months(d: date, months: int) -> date:
    """Add N months to a date, clamping day-of-month."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    # day clamp for month-end edge (e.g. 1/31 → 2/28)
    import calendar
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))


def _spy_benchmark(start: str, end: str) -> dict:
    """Compute SPY buy-and-hold Sharpe + APY for the window."""
    sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))
    from kernel.data import fetch_ohlcv  # noqa: PLC0415
    from kernel.risk_metrics import (  # noqa: PLC0415
        sharpe_ratio, daily_returns_from_equity,
    )
    spy = fetch_ohlcv("SPY")
    window = spy.loc[start:end]
    if len(window) < 2:
        return {"sharpe": float("nan"), "apy": float("nan"),
                "total_return": float("nan"), "n_bars": int(len(window))}
    eq = window["close"]
    ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    n_yrs = len(eq) / 252
    apy = (1 + ret) ** (1 / n_yrs) - 1 if n_yrs > 0 else 0.0
    return {
        "sharpe": sharpe_ratio(daily_returns_from_equity(eq)),
        "apy": apy * 100,
        "total_return": ret * 100,
        "n_bars": int(len(window)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy-config-name", required=True)
    p.add_argument("--cuts", nargs="+", required=True,
                    help="train_end ISO dates; each cut runs an OOS "
                         "starting on cut+1day for oos_months months")
    p.add_argument("--oos-months", type=int, default=12)
    p.add_argument("--initial-cash", type=float, default=100_000)
    p.add_argument("--output", required=True,
                    help="output dir; per-cut JSONs + summary.json")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "strategy_config_name": args.strategy_config_name,
        "cuts":                  args.cuts,
        "oos_months":            args.oos_months,
        "results":               [],
        "consistency":           {},
    }

    for cut_str in args.cuts:
        cut = date.fromisoformat(cut_str)
        sim_start = (cut + timedelta(days=1)).isoformat()
        sim_end   = _add_months(cut, args.oos_months).isoformat()
        cut_out = out_dir / f"{cut_str}.json"

        log.info("══ cut=%s OOS [%s → %s] ══", cut_str, sim_start, sim_end)
        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "holdout_backtest.py"),
            "--skip-train",
            "--strategy-config-name", args.strategy_config_name,
            "--train-end",  cut_str,
            "--sim-start",  sim_start,
            "--sim-end",    sim_end,
            "--initial-cash", str(args.initial_cash),
            "--output",     str(cut_out),
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            log.error("cut %s failed (rc=%d) — skipping", cut_str, rc)
            continue

        with open(cut_out) as f:
            cut_data = json.load(f)
        spy = _spy_benchmark(sim_start, sim_end)
        cut_data["spy_benchmark"] = spy
        cut_data["alpha_vs_spy"] = (
            cut_data.get("apy_holdout", 0.0) - spy["apy"]
        )
        with open(cut_out, "w") as f:
            json.dump(cut_data, f, indent=2, default=str)
        summary["results"].append({
            "cut":             cut_str,
            "sim_start":       sim_start,
            "sim_end":         sim_end,
            "apy":             cut_data.get("apy_holdout"),
            "sharpe":          cut_data.get("sharpe_holdout"),
            "spy_apy":         spy["apy"],
            "spy_sharpe":      spy["sharpe"],
            "alpha_vs_spy":    cut_data["alpha_vs_spy"],
            "n_buys":          cut_data.get("n_buys"),
            "n_sells":         cut_data.get("n_sells"),
        })

    # Consistency stats across cuts.
    sharpes = [r["sharpe"] for r in summary["results"]
                if r.get("sharpe") is not None and isinstance(r["sharpe"], (int, float))
                and math.isfinite(r["sharpe"])]
    apys    = [r["apy"]    for r in summary["results"]
                if r.get("apy") is not None and isinstance(r["apy"], (int, float))
                and math.isfinite(r["apy"])]
    alphas  = [r["alpha_vs_spy"] for r in summary["results"]
                if isinstance(r.get("alpha_vs_spy"), (int, float))
                and math.isfinite(r["alpha_vs_spy"])]
    summary["consistency"] = {
        "n_cuts":             len(summary["results"]),
        "sharpe_mean":        fmean(sharpes) if sharpes else float("nan"),
        "sharpe_std":         stdev(sharpes) if len(sharpes) > 1 else 0.0,
        "apy_mean":           fmean(apys)    if apys    else float("nan"),
        "apy_std":            stdev(apys)    if len(apys) > 1 else 0.0,
        "alpha_vs_spy_mean":  fmean(alphas)  if alphas  else float("nan"),
        "alpha_vs_spy_std":   stdev(alphas)  if len(alphas) > 1 else 0.0,
        "alpha_consistent_sign": (
            all(a > 0 for a in alphas) if alphas else None
        ),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("══ summary written %s ══", out_dir / "summary.json")
    log.info("apy_mean=%.2f%% sharpe_mean=%.2f alpha_vs_spy_mean=%.2f%%",
              summary["consistency"]["apy_mean"],
              summary["consistency"]["sharpe_mean"],
              summary["consistency"]["alpha_vs_spy_mean"])


if __name__ == "__main__":
    main()
