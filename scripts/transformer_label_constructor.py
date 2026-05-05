#!/usr/bin/env python
"""Phase 3 of Transformer data prep: multi-horizon label construction.

For each ticker × date in the clean universe, computes:
  - fwd_5d_excess  = (close[t+5]/close[t]) / (spy[t+5]/spy[t]) - 1
  - fwd_20d_excess = same with 20d horizon
  - fwd_60d_excess = same with 60d horizon

Excess returns relative to SPY (cross-sectional benchmark) so the
Transformer learns alpha, not market beta. Per Lim et al. (2021) "TFT"
§III.B "Multi-Horizon Forecasting" — multi-head loss on shared
representation lets the model balance short-term mean-reversion
against medium-term trend.

Output: a single Parquet file `data/transformer_panel_labels.parquet`
with columns: [ticker, date, fwd_5d_excess, fwd_20d_excess,
fwd_60d_excess]. Index = MultiIndex(ticker, date).

NB: forward returns require future data. Bars within `max(horizon)` of
the cache's last_date are NaN — they will be filtered out at training
time (no synthetic future). Per CLAUDE.md §5.6 + §5.2 (no leakage).
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("transformer-labels")


def _build_labels_for_ticker(ticker_close: pd.Series,
                              spy_close: pd.Series,
                              horizons: list[int]) -> pd.DataFrame:
    """Compute fwd_<H>d_excess for each horizon H in `horizons`.

    Aligns ticker.close to SPY.close via index intersection (avoids
    weekend / holiday mismatches). Forward return = price-relative
    over the horizon, expressed as relative-to-SPY excess.
    """
    idx = ticker_close.index.intersection(spy_close.index)
    if len(idx) < max(horizons) + 1:
        return pd.DataFrame()
    tc = ticker_close.loc[idx]
    sc = spy_close.loc[idx]
    out: dict[str, pd.Series] = {}
    for h in horizons:
        # Forward total return for ticker + SPY
        ticker_fwd = tc.shift(-h) / tc - 1.0
        spy_fwd    = sc.shift(-h) / sc - 1.0
        out[f"fwd_{h}d_excess"] = ticker_fwd - spy_fwd
    return pd.DataFrame(out, index=idx)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory",
                    default=str(REPO_ROOT / "data" / "transformer_universe_inventory.json"))
    p.add_argument("--integrity-report",
                    default=str(REPO_ROOT / "data" / "transformer_data_integrity_report.json"))
    p.add_argument("--ohlcv-dir",
                    default=str(REPO_ROOT / "data" / "ohlcv"))
    p.add_argument("--output",
                    default=str(REPO_ROOT / "data" / "transformer_panel_labels.parquet"))
    p.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60])
    args = p.parse_args()

    # Load inventory + integrity → universe = (Tier-A ∪ Tier-B) ∩ integrity_pass
    inv = json.loads(Path(args.inventory).read_text())
    integ = json.loads(Path(args.integrity_report).read_text())
    universe = set(inv["tier_A_tickers"]) | set(inv["tier_B_tickers"])
    # Drop tickers that failed integrity
    failed = set()
    for tier in ("A", "B"):
        for r in integ["per_ticker"][tier]:
            if not r["ok"]:
                failed.add(r["ticker"])
    universe -= failed
    universe = sorted(universe)
    log.info("Building labels for %d tickers (Tier-A+B minus %d failed integrity)",
              len(universe), len(failed))

    # Load SPY close
    spy_path = Path(args.ohlcv_dir) / "SPY" / "1d.parquet"
    if not spy_path.exists():
        log.error("SPY not found at %s — required as benchmark", spy_path)
        sys.exit(1)
    spy_close = pd.read_parquet(spy_path, columns=["close"])["close"]
    log.info("SPY: %d bars  %s → %s",
              len(spy_close), spy_close.index.min(), spy_close.index.max())

    rows: list[pd.DataFrame] = []
    skipped = 0
    for i, t in enumerate(universe):
        if i % 50 == 0 and i > 0:
            log.info("  ... %d/%d", i, len(universe))
        p = Path(args.ohlcv_dir) / t / "1d.parquet"
        try:
            tc = pd.read_parquet(p, columns=["close"])["close"]
        except Exception as exc:
            log.warning("  %s: read failed — %s", t, exc)
            skipped += 1
            continue
        labels = _build_labels_for_ticker(tc, spy_close, args.horizons)
        if labels.empty:
            skipped += 1
            continue
        labels.insert(0, "ticker", t)
        labels = labels.reset_index().rename(columns={"index": "date"})
        rows.append(labels)

    if not rows:
        log.error("No labels built — universe empty or all reads failed")
        sys.exit(1)
    panel = pd.concat(rows, ignore_index=True)
    log.info("Panel: %d rows × %d cols", len(panel), len(panel.columns))

    # NaN audit per horizon — the bars within `horizon` of cache end are NaN
    for h in args.horizons:
        col = f"fwd_{h}d_excess"
        nan_pct = float(panel[col].isna().mean())
        log.info("  %s: NaN rate = %.3f", col, nan_pct)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    log.info("══ labels written %s (%d rows, %d skipped) ══",
              out_path, len(panel), skipped)


if __name__ == "__main__":
    main()
