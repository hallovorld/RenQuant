#!/usr/bin/env python
"""P4.2 driver — apply triple-barrier labels to a snapshot parquet.

Usage::

    python scripts/_meta_label_generate.py \
        --snapshots data/position_day_snapshots.parquet \
        --out       data/position_day_labels.parquet \
        --pt-mult 10 --sl-mult 10 --fwd-window 20

Reads close prices via :func:`kernel.data.fetch_ohlcv` for each ticker
in the snapshot frame, calls
:func:`kernel.meta_label.labeler.label_snapshots`, and writes the
result to ``--out``.

References
----------
* López de Prado 2018 AFML ch.3 (triple barrier) / ch.20 (meta-label)
* doc/research/meta-labeling-exit-policy.md §5 — RenQuant-specific
  barrier parameter choice (pt=10, sl=10, σ_daily=realized_20d/√252,
  fwd_window=20 business days)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO  = Path(__file__).resolve().parent.parent
STRAT = REPO / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("meta-label-generate")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshots", required=True,
                   help="Input parquet from SnapshotLogger.dump_to_parquet")
    p.add_argument("--out", required=True,
                   help="Output parquet (snapshot rows + meta_label + fwd_*_ret)")
    p.add_argument("--pt-mult", type=float, default=10.0,
                   help="Profit-take multiplier × σ_daily (default 10)")
    p.add_argument("--sl-mult", type=float, default=10.0,
                   help="Stop-loss multiplier × σ_daily (default 10)")
    p.add_argument("--default-sigma-daily", type=float, default=0.01,
                   help="Fallback when realized_vol_20d is 0/NaN (default 0.01)")
    p.add_argument("--fwd-window", type=int, default=20,
                   help="Vertical barrier (business days, default 20)")
    args = p.parse_args()

    log.info("Reading snapshots → %s", args.snapshots)
    df = pd.read_parquet(args.snapshots)
    log.info("  rows=%d unique tickers=%d any_trigger=%d",
             len(df), df["ticker"].nunique(),
             int(df["any_trigger"].fillna(0).sum()))

    # Fetch close paths via the same loader the sim uses
    from kernel.data import fetch_ohlcv  # noqa: PLC0415
    from kernel.meta_label.labeler import label_snapshots  # noqa: PLC0415

    close_paths: dict[str, pd.Series] = {}
    tickers = sorted(df["ticker"].dropna().unique())
    log.info("Fetching close paths for %d tickers …", len(tickers))
    for t in tickers:
        try:
            ohlc = fetch_ohlcv(t)
            if "close" in ohlc.columns:
                close_paths[t] = ohlc["close"].astype(float)
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s: fetch failed: %s", t, exc)

    log.info("Applying triple-barrier (pt=%.1f sl=%.1f σ=%.4f fwd=%dd) …",
             args.pt_mult, args.sl_mult, args.default_sigma_daily, args.fwd_window)
    labeled = label_snapshots(
        df, close_paths=close_paths,
        pt_mult=args.pt_mult, sl_mult=args.sl_mult,
        default_sigma_daily=args.default_sigma_daily,
        fwd_window=args.fwd_window,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out_path, index=False)

    n_triggered = int(labeled["any_trigger"].fillna(0).sum())
    n_labeled   = int(labeled["meta_label"].notna().sum())
    if n_labeled > 0:
        balance = labeled["meta_label"].dropna().mean()
        log.info("Wrote %s — triggered=%d  labeled=%d  class_balance(positive)=%.2f%%",
                 out_path, n_triggered, n_labeled, balance * 100)
    else:
        log.warning("Wrote %s — no labels generated (no triggers? no future bars?)", out_path)


if __name__ == "__main__":
    main()
