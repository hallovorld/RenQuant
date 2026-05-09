#!/usr/bin/env python
"""Track A — Build RAW (un-z-scored) fwd_60d_excess label for QuantileHead.

The production panel's fwd_60d_excess is cross-sectionally z-scored
PER DATE → median quantile estimation collapses to ≈ 0 for every
ticker (the conditional mean of a date-zero-mean target is ~0).
That's why QuantileHead val μ-IC = +0.021 (weak) — not a hyperparameter
issue, a label-construction issue.

Fix: rebuild fwd_60d_excess UN-NORMALIZED (raw return − SPY return
over same 60d window). This preserves absolute return-scale
information so quantile regression learns meaningful μ̂.

Output: data/alpha158_291_fundamental_dataset_rawlabel.parquet
  Same schema as production panel except `fwd_60d_excess_raw` replaces
  `fwd_60d_excess`. Train QuantileHead on this. The XGB rank-LTR
  production model is unchanged (it uses cross-sectionally z-scored
  label which is correct for cross-sectional ranking).

References:
- Lim et al. 2021 ICLR TFT §3 — quantile-based σ recovery requires
  un-normalized target for σ to be on return scale.
- Wakefield 2013 §3.4 — Gaussian-from-quantiles σ approximation.
"""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build-raw-label")

REPO = Path(__file__).resolve().parent.parent
HORIZON = 60   # match production fwd_60d


def main():
    panel_in  = REPO / "data" / "alpha158_291_fundamental_dataset.parquet"
    panel_out = REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet"
    ohlcv_dir = REPO / "data" / "ohlcv"

    log.info("Loading panel + computing raw fwd_60d_excess...")
    panel = pd.read_parquet(panel_in)
    panel["date"] = pd.to_datetime(panel["date"])
    log.info("Panel: rows=%d tickers=%d dates %s..%s",
             len(panel), panel["ticker"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    # Load SPY closes for benchmark
    spy_p = ohlcv_dir / "SPY" / "1d.parquet"
    spy = pd.read_parquet(spy_p)
    spy.index = pd.to_datetime(spy.index)
    spy = spy["close"].sort_index()
    log.info("SPY closes: %d days", len(spy))

    # Per-ticker compute raw fwd_60d return − SPY fwd_60d return
    log.info("Computing raw fwd_60d_excess per ticker...")
    out_blocks = []
    skipped = 0
    for tkr, g in panel.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True).copy()
        ohlcv_p = ohlcv_dir / tkr / "1d.parquet"
        if not ohlcv_p.exists():
            skipped += 1
            g["fwd_60d_excess_raw"] = np.nan
            out_blocks.append(g)
            continue
        ohlcv = pd.read_parquet(ohlcv_p)
        ohlcv.index = pd.to_datetime(ohlcv.index)
        close = ohlcv["close"].sort_index()
        # Compute fwd 60-trading-day return per panel-date
        # Use trading-day shift: close.shift(-HORIZON) gives close 60 trading days later
        fwd_close = close.shift(-HORIZON)
        ticker_fwd_ret = (fwd_close / close - 1.0).rename("ticker_fwd")
        spy_fwd_close = spy.shift(-HORIZON)
        spy_fwd_ret = (spy_fwd_close / spy - 1.0).rename("spy_fwd")
        # Align on g["date"]
        g_dates = g["date"].values
        ticker_fwd_aligned = ticker_fwd_ret.reindex(g_dates).values
        spy_fwd_aligned    = spy_fwd_ret.reindex(g_dates).values
        excess = ticker_fwd_aligned - spy_fwd_aligned
        g["fwd_60d_excess_raw"] = excess
        out_blocks.append(g)

    log.info("  skipped (no OHLCV): %d tickers", skipped)
    out = pd.concat(out_blocks, ignore_index=True)
    n_valid = out["fwd_60d_excess_raw"].notna().sum()
    log.info("Raw label: %d / %d rows valid (%.1f%%)",
             n_valid, len(out), 100*n_valid/len(out))
    log.info("  raw_label stats: mean=%+.4f std=%.4f min=%+.4f max=%+.4f",
             out["fwd_60d_excess_raw"].mean(),
             out["fwd_60d_excess_raw"].std(),
             out["fwd_60d_excess_raw"].min(),
             out["fwd_60d_excess_raw"].max())

    out.to_parquet(panel_out, index=False)
    log.info("Saved → %s  (%.1f MB)", panel_out, panel_out.stat().st_size/1e6)


if __name__ == "__main__":
    main()
