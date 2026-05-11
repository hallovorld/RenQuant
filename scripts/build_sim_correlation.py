#!/usr/bin/env python
"""Build a leakage-free watchlist-correlation artifact for sim.

Computes pairwise 60-day return correlations using OHLCV data strictly
≤ ``--as-of YYYY-MM-DD`` and writes the v2-schema artifact (matrix +
as_of_date metadata) the correlation guard expects.

Per CLAUDE.md §5.13.5 / §5.13.13: this is a SIM-only helper. The
production correlation matrix lives in artifacts/prod/ and is built
weekly off the most-recent OHLCV (no cutoff). This script's --as-of
arg locks the data window so the resulting sim matrix has no forward-
looking information beyond the cutoff.

Usage::

    python scripts/build_sim_correlation.py --as-of 2023-12-01

Output: backtesting/renquant_104/artifacts/sim/watchlist-correlation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("build-sim-correlation")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--as-of", required=True,
        help="ISO date (YYYY-MM-DD). Last day of data included in the "
             "correlation window. For sim usage set ≤ sim_start to avoid "
             "forward leakage.",
    )
    p.add_argument(
        "--lookback-days", type=int, default=60,
        help="Rolling-correlation window in trading days (default 60).",
    )
    p.add_argument(
        "--strategy", default="renquant_104",
    )
    p.add_argument(
        "--strategy-config-name", default="strategy_config.sim_baseline.json",
        help="Config supplying the watchlist (default sim_baseline).",
    )
    p.add_argument(
        "--out", default=None,
        help="Output path. Defaults to "
             "backtesting/<strategy>/artifacts/sim/watchlist-correlation.json.",
    )
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    sys.path.insert(0, str(strategy_dir))
    config = json.loads((strategy_dir / args.strategy_config_name).read_text())
    watchlist = config["watchlist"]
    as_of = pd.Timestamp(args.as_of)

    from kernel.data import fetch_ohlcv  # noqa: PLC0415

    log.info(
        "Fetching OHLCV for %d tickers (cutoff ≤ %s, lookback %dd)",
        len(watchlist), as_of.date(), args.lookback_days,
    )

    # Collect aligned close-price series, dropping anything that breaches cutoff.
    closes: dict[str, pd.Series] = {}
    for t in watchlist:
        try:
            df = fetch_ohlcv(t)
        except Exception as exc:
            log.warning("  %s fetch failed: %s", t, exc)
            continue
        if df is None or df.empty or "close" not in df.columns:
            continue
        sub = df[df.index <= as_of]
        if len(sub) < args.lookback_days + 5:
            log.debug("  %s: only %d bars ≤ %s, skipping",
                      t, len(sub), as_of.date())
            continue
        closes[t] = sub["close"].astype(float).tail(args.lookback_days + 30)

    if len(closes) < 3:
        log.error("Too few tickers with sufficient history (%d); aborting",
                  len(closes))
        sys.exit(2)

    # Align to common date index, compute daily returns, take last `lookback`.
    px = pd.DataFrame(closes).sort_index()
    px = px[px.index <= as_of].tail(args.lookback_days + 1)
    rets = px.pct_change().dropna(how="all")
    rets = rets.tail(args.lookback_days)
    log.info("Aligned panel: %d returns × %d tickers", len(rets), rets.shape[1])

    # Pairwise correlation; non-overlapping ticker pairs (insufficient
    # joint history) drop to NaN — record as 0 to keep the matrix dense.
    corr = rets.corr(min_periods=max(args.lookback_days // 2, 20))
    corr = corr.fillna(0.0)

    matrix = {
        t: {u: float(corr.loc[t, u]) for u in corr.columns}
        for t in corr.index
    }

    out_path = (
        Path(args.out) if args.out else
        strategy_dir / "artifacts" / "sim" / "watchlist-correlation.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 2,
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "data_window_end": as_of.strftime("%Y-%m-%d"),
        "data_window_start": (
            rets.index[0].strftime("%Y-%m-%d") if not rets.empty else None
        ),
        "lookback_days": args.lookback_days,
        "n_tickers": len(matrix),
        "matrix": matrix,
    }
    out_path.write_text(json.dumps(artifact, indent=2))
    log.info("Saved → %s (n=%d tickers, lookback=%dd, as_of=%s)",
             out_path, len(matrix), args.lookback_days, as_of.date())


if __name__ == "__main__":
    main()
