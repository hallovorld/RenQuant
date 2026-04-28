#!/usr/bin/env python
"""Watchlist 200 v2 candidate screener.

Per doc/research/watchlist-200-v2-plan.md, screens local OHLCV cache
for tickers passing four quality filters:

  1. Liquidity floor:    median 1y dollar-volume ≥ $50M
  2. History floor:      ≥ 504 trading days (~2y) of clean data
  3. Risk-distribution:  realized 1y annualized vol ∈ [0.15, 0.85]
                         (matches the bulk of current 103 panel)
  4. Sharpe quality:     1y realized Sharpe ≥ 0.5

Output: doc/research/watchlist-200-v2-candidates.json with the full
sorted candidate list, plus stats per ticker. Does NOT modify the
production watchlist — that's a separate step after paired CPCV.

Usage:
    python scripts/screen_watchlist_v2.py
    python scripts/screen_watchlist_v2.py --top 100  # take top 100 by Sharpe
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
log = logging.getLogger("screen-wl")


def _score_ticker(t: str, ohlcv_dir: Path) -> dict | None:
    p = ohlcv_dir / t / "1d.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if "close" not in df.columns or "volume" not in df.columns:
        return None
    if len(df) < 504:    # Filter 2: history floor
        return None
    recent = df.iloc[-252:]
    rets = recent["close"].pct_change().dropna()
    if len(rets) < 60:
        return None
    ann_ret = float(rets.mean() * 252)
    ann_vol = float(rets.std() * np.sqrt(252))
    if ann_vol <= 0:
        return None
    sharpe = ann_ret / ann_vol
    med_dv = float((recent["close"] * recent["volume"]).median())
    return {
        "ticker":          t,
        "sharpe":          sharpe,
        "ann_ret":         ann_ret,
        "ann_vol":         ann_vol,
        "med_dollar_vol":  med_dv,
        "history_days":    int(len(df)),
    }


def _passes_filters(row: dict, current_wl: set[str]) -> tuple[bool, str]:
    """Return (passes, reason). reason is the failing filter when False."""
    # Filter 1: liquidity
    if row["med_dollar_vol"] < 50e6:
        return False, "liquidity"
    # Filter 3: vol band
    if not (0.15 <= row["ann_vol"] <= 0.85):
        return False, "vol_band"
    # Filter 4: Sharpe quality
    if row["sharpe"] < 0.5:
        return False, "sharpe"
    return True, "pass"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=100,
                   help="Take top-N candidates by Sharpe (default 100)")
    p.add_argument("--ohlcv-dir", default=str(REPO_ROOT / "data" / "ohlcv"))
    p.add_argument("--out", default=str(REPO_ROOT / "doc" / "research"
                                      / "watchlist-200-v2-candidates.json"))
    args = p.parse_args()

    ohlcv_dir = Path(args.ohlcv_dir)
    if not ohlcv_dir.exists():
        log.error("OHLCV dir not found: %s", ohlcv_dir)
        return 1

    cur_wl_path = (REPO_ROOT / "backtesting" / "renquant_104"
                   / "strategy_config.json")
    cur_wl = set(json.loads(cur_wl_path.read_text())["watchlist"])
    log.info("Current watchlist: %d tickers", len(cur_wl))

    all_tickers = sorted(p.name for p in ohlcv_dir.iterdir() if p.is_dir())
    log.info("OHLCV cache: %d tickers", len(all_tickers))

    rows = []
    for t in all_tickers:
        r = _score_ticker(t, ohlcv_dir)
        if r is not None:
            r["in_current_wl"] = t in cur_wl
            ok, reason = _passes_filters(r, cur_wl)
            r["passes"] = ok
            r["reject_reason"] = reason
            rows.append(r)

    log.info("Scored: %d tickers (filtered out %d on history floor or data issues)",
             len(rows), len(all_tickers) - len(rows))

    df = pd.DataFrame(rows)
    passed = df[df["passes"]].copy().sort_values("sharpe", ascending=False)
    log.info("Passed all filters: %d tickers", len(passed))

    new_only = passed[~passed["in_current_wl"]].copy()
    log.info("Of those, NOT yet in watchlist: %d candidates", len(new_only))
    log.info("Top 30:\n%s", new_only.head(30).to_string(
        index=False,
        columns=["ticker", "sharpe", "ann_ret", "ann_vol", "med_dollar_vol", "history_days"],
        float_format=lambda x: f"{x:+.3f}" if abs(x) < 1 else f"{x:,.0f}",
    ))

    # Take top-N by Sharpe (out of new + current; final list is current 103 + top
    # new ones up to args.top total or ≥ 200)
    target = max(args.top, 100)
    new_take = new_only.head(target - len(cur_wl))
    final_wl = sorted(set(cur_wl) | set(new_take["ticker"].tolist()))
    log.info("Proposed v2 watchlist size: %d (current %d + new %d)",
             len(final_wl), len(cur_wl), len(new_take))

    out = {
        "generated_at":    pd.Timestamp.utcnow().isoformat(),
        "current_size":    len(cur_wl),
        "candidate_count": len(new_only),
        "proposed_top_n":  args.top,
        "proposed_wl_size": len(final_wl),
        "filters": {
            "liquidity_floor_dv":  50e6,
            "history_floor_days":  504,
            "vol_band":            [0.15, 0.85],
            "sharpe_floor":        0.5,
        },
        "new_candidates_top_30": new_only.head(30).to_dict(orient="records"),
        "proposed_watchlist":     final_wl,
        "reject_reasons_summary": (
            df[~df["passes"]]["reject_reason"].value_counts().to_dict()
        ),
        "validation_protocol_required_before_deploy": [
            "1. Per-ticker individual IC contribution test (single-ticker addition)",
            "2. Greedy forward selection by IC delta",
            "3. Final paired CPCV vs golden 103 baseline (paired t > +1.5)",
            "4. A/A sanity test (shuffled labels → IC ≈ 0)",
        ],
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log.info("Wrote %s", args.out)
    log.info("DO NOT deploy without running the 4-step validation protocol.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
