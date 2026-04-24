#!/usr/bin/env python
"""Check market-anomaly retrain triggers and emit which (if any) fired.

Runs daily at 13:10 PT (before daily_104.sh 13:55 PT) so an anomaly-
triggered retrain can land before the scheduled trading pass.

Triggers (default; all configurable via CLI):
  * SPY |daily change| > 2%   → "anomaly_spy_2pct"
  * VIX |daily change| > 5%   → "anomaly_vix_5pct"

Exit codes:
  0   — no trigger fired (no retrain needed)
  1   — one or more triggers fired; stdout prints trigger tag(s),
        one per line. Caller (shell wrapper) reads and fires
        `train_104.py --force --trigger=TAG`.

Data source: yfinance for ^SPY and ^VIX close. No local cache — network
request is cheap (2 tickers × ~5 days) and freshness is critical for
anomaly detection.

Usage::

    python scripts/check_retrain_triggers.py
    python scripts/check_retrain_triggers.py --spy-pct 0.015 --vix-pct 0.07
    python scripts/check_retrain_triggers.py --dry-run   # print but don't exit nonzero
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("check-retrain-triggers")


def _pct_change(symbol: str) -> "float | None":
    """Return (today_close / prior_close) - 1, or None on fetch failure."""
    import yfinance as yf  # noqa: PLC0415
    try:
        t = yf.Ticker(symbol)
        # 5d range gives us at least today + prior
        hist = t.history(period="5d", auto_adjust=False, actions=False)
    except Exception as exc:
        log.warning("yfinance fetch failed for %s: %s", symbol, exc)
        return None
    if hist is None or len(hist) < 2:
        log.warning("Insufficient history for %s (need ≥2 bars, got %s)",
                    symbol, 0 if hist is None else len(hist))
        return None
    closes = hist["Close"].dropna()
    if len(closes) < 2:
        return None
    latest = float(closes.iloc[-1])
    prior  = float(closes.iloc[-2])
    if prior <= 0:
        return None
    return (latest / prior) - 1.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spy-pct", type=float, default=0.02,
                   help="SPY |daily change| threshold. Default 0.02 (2%%).")
    p.add_argument("--vix-pct", type=float, default=0.05,
                   help="VIX |daily change| threshold. Default 0.05 (5%%).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print triggers but always exit 0 (for testing).")
    args = p.parse_args()

    spy_change = _pct_change("^SPY")   # ^GSPC is the S&P 500 index; SPY ETF
    if spy_change is None:
        spy_change = _pct_change("SPY")
    vix_change = _pct_change("^VIX")

    triggers: list[str] = []
    if spy_change is not None:
        log.info("SPY daily change: %+.2f%%  (threshold ±%.2f%%)",
                 spy_change * 100, args.spy_pct * 100)
        if abs(spy_change) > args.spy_pct:
            tag = "anomaly_spy_2pct" if args.spy_pct == 0.02 else \
                  f"anomaly_spy_{int(args.spy_pct*1000)}bp"
            triggers.append(tag)
    if vix_change is not None:
        log.info("VIX daily change: %+.2f%%  (threshold ±%.2f%%)",
                 vix_change * 100, args.vix_pct * 100)
        if abs(vix_change) > args.vix_pct:
            tag = "anomaly_vix_5pct" if args.vix_pct == 0.05 else \
                  f"anomaly_vix_{int(args.vix_pct*1000)}bp"
            triggers.append(tag)

    if triggers:
        for t in triggers:
            print(t)
        if args.dry_run:
            log.info("[dry-run] Would have exited 1 to fire retrain(s): %s",
                     triggers)
            return 0
        log.info("Firing retrain triggers: %s", triggers)
        return 1

    log.info("No anomaly triggers fired — no retrain needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
