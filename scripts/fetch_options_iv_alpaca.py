#!/usr/bin/env python3
"""Fetch IV features per watchlist ticker from Alpaca Options API.

Roadmap C1 (2026-05-18 user mandate, $0/mo). Alpaca's OptionHistoricalDataClient
returns OptionsSnapshot per contract, including implied_volatility and
greeks, for FREE on paper + live accounts. This script per-ticker:

  1. Pulls full option chain (~3000-5000 contracts per liquid name)
  2. Filters to monthly expirations near 30 and 60 days
  3. Identifies ATM call + put per expiration
  4. Computes:
       iv_30d_call_atm  : ATM call IV at ~30d expiry
       iv_30d_put_atm   : ATM put IV at ~30d expiry
       iv_skew_30d      : put_iv - call_iv (sentiment indicator;
                          Bali-Hovakimian 2009 MSci)
       iv_term_struct   : iv_60d_atm - iv_30d_atm (contango/backwardation;
                          Cremers-Weinbaum 2010 JFQA flag)

Output: data/options_iv_alpaca/{ticker}.parquet — one row per fetch
date with the 4 IV features above.

Rate-limit strategy:
  • Same TokenBucket pattern as fetch_news_alpaca.py
  • Default 180/min (90% of Free tier; OptionChainRequest is more
    expensive than news but still rate-limited per global key)
  • Exponential backoff on 429 (1s → 60s cap)

References:
  - Goyal-Saretto 2009 JFE "Cross-section of option-implied volatility
    and stock returns"
  - Bali-Hovakimian 2009 MSci "Volatility spreads and expected stock
    returns" (put-call IV skew)
  - Cremers-Weinbaum 2010 JFQA "Deviations from put-call parity"
  - An-Ang-Bali-Cakici 2014 RFS "Joint cross-section of stocks and
    options"

NOTE: Alpaca Free tier provides "indicative" IV (NBBO-snapshot Black-
Scholes), NOT the SPX-style mid-day vol surface in OptionMetrics IvyDB.
For a 60-day horizon strategy this is adequate; for HFT or vol-arb it
would not be.
"""
from __future__ import annotations
import argparse
import logging
import re
import sys
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "options_iv_alpaca"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch_options_iv")


class TokenBucket:
    """Sliding-window rate limiter (90% of Alpaca Free tier 200/min)."""

    def __init__(self, max_calls: int = 180, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        now = time.time()
        while self._timestamps and self._timestamps[0] <= now - self.window:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_calls:
            sleep_for = self.window - (now - self._timestamps[0]) + 0.05
            time.sleep(max(0.05, sleep_for))
            now = time.time()
            while self._timestamps and self._timestamps[0] <= now - self.window:
                self._timestamps.popleft()
        self._timestamps.append(now)


# Alpaca OCC symbol: AAPL_260529C00170000 → AAPL exp 2026-05-29 Call $170.00
_OCC_RE = re.compile(
    r"^([A-Z]+)(\d{6})([CP])(\d{8})$"
)


def parse_occ(occ: str) -> dict | None:
    """Parse OCC option symbol → dict with underlying, expiry, type, strike.

    Returns None if symbol doesn't match the standard format.
    Standard OCC: 6-char date (YYMMDD), 1-char type (C/P), 8-digit strike
    (cents × 1000).
    """
    m = _OCC_RE.match(occ)
    if not m:
        return None
    und, exp_s, typ, strike_s = m.groups()
    try:
        yy = 2000 + int(exp_s[:2])
        mm = int(exp_s[2:4])
        dd = int(exp_s[4:6])
        expiry = date(yy, mm, dd)
    except ValueError:
        return None
    try:
        strike = int(strike_s) / 1000.0
    except ValueError:
        return None
    return {
        "underlying": und,
        "expiry":     expiry,
        "option_type": typ,  # "C" or "P"
        "strike":     strike,
    }


def _nearest_atm_iv(
    contracts: list[dict],
    target_dte: int,
    option_type: str,
    spot: float,
    dte_tolerance: int = 10,
) -> tuple[float, int, float] | None:
    """Find ATM contract whose DTE is closest to target.

    Args:
        contracts: list of dicts with `expiry`, `strike`, `iv`, `option_type`
        target_dte: target days-to-expiry (e.g. 30 or 60)
        option_type: "C" or "P"
        spot: current underlying price (for ATM strike selection)
        dte_tolerance: only accept contracts within ±N days of target DTE

    Returns (iv, dte, strike) or None if no matching contract found.
    """
    today = date.today()
    candidates = [
        c for c in contracts
        if c["option_type"] == option_type
        and abs((c["expiry"] - today).days - target_dte) <= dte_tolerance
        and c.get("iv") is not None
        and c.get("iv") > 0
    ]
    if not candidates:
        return None
    # Pick the expiry CLOSEST to target_dte, then within that expiry pick
    # strike closest to spot.
    candidates.sort(key=lambda c: abs((c["expiry"] - today).days - target_dte))
    nearest_dte = (candidates[0]["expiry"] - today).days
    same_expiry = [c for c in candidates
                   if (c["expiry"] - today).days == nearest_dte]
    atm = min(same_expiry, key=lambda c: abs(c["strike"] - spot))
    return (atm["iv"], nearest_dte, atm["strike"])


def fetch_iv_features(client, symbol: str, spot: float,
                      bucket: TokenBucket) -> dict | None:
    """One-symbol IV feature extraction.

    Returns dict with keys: symbol, as_of, iv_30d_call_atm,
    iv_30d_put_atm, iv_60d_call_atm, iv_60d_put_atm, iv_skew_30d,
    iv_term_struct. Returns None if chain fetch failed.
    """
    from alpaca.data.requests import OptionChainRequest
    bucket.acquire()
    backoff = 1.0
    for attempt in range(5):
        try:
            chain = client.get_option_chain(OptionChainRequest(
                underlying_symbol=symbol
            ))
            break
        except Exception as exc:
            msg = str(exc)
            if "rate" in msg.lower() or "429" in msg:
                log.warning("  %s: rate-limited (try %d) — backoff %.1fs",
                            symbol, attempt, backoff)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)
                continue
            log.warning("  %s: chain fetch failed — %s", symbol, exc)
            return None
    else:
        log.warning("  %s: 5x rate-limit retries failed", symbol)
        return None

    if not chain:
        return None

    # Normalize each snapshot into a flat record
    contracts: list[dict] = []
    for occ, snap in chain.items():
        parsed = parse_occ(occ)
        if parsed is None:
            continue
        iv = getattr(snap, "implied_volatility", None)
        if iv is None or iv <= 0:
            continue
        contracts.append({
            **parsed,
            "iv": float(iv),
        })

    if not contracts:
        log.warning("  %s: chain returned %d contracts but no valid IV",
                    symbol, len(chain))
        return None

    c30 = _nearest_atm_iv(contracts, 30, "C", spot)
    p30 = _nearest_atm_iv(contracts, 30, "P", spot)
    c60 = _nearest_atm_iv(contracts, 60, "C", spot)
    p60 = _nearest_atm_iv(contracts, 60, "P", spot)

    iv_30d_call = c30[0] if c30 else np.nan
    iv_30d_put  = p30[0] if p30 else np.nan
    iv_60d_call = c60[0] if c60 else np.nan
    iv_60d_put  = p60[0] if p60 else np.nan

    # Bali-Hovakimian 2009 skew = put - call at near-term ATM
    iv_skew_30d = (iv_30d_put - iv_30d_call) if (
        not np.isnan(iv_30d_put) and not np.isnan(iv_30d_call)
    ) else np.nan
    # Term structure: 60d - 30d (positive = contango, negative = backwardation)
    iv_term_struct = np.nan
    if not np.isnan(iv_60d_call) and not np.isnan(iv_30d_call):
        iv_30d_atm = (iv_30d_call + iv_30d_put) / 2 if not np.isnan(iv_30d_put) else iv_30d_call
        iv_60d_atm = (iv_60d_call + iv_60d_put) / 2 if not np.isnan(iv_60d_put) else iv_60d_call
        iv_term_struct = iv_60d_atm - iv_30d_atm

    return {
        "symbol":           symbol,
        "as_of":            date.today().isoformat(),
        "spot":             float(spot),
        "iv_30d_call_atm":  iv_30d_call,
        "iv_30d_put_atm":   iv_30d_put,
        "iv_60d_call_atm":  iv_60d_call,
        "iv_60d_put_atm":   iv_60d_put,
        "iv_skew_30d":      iv_skew_30d,
        "iv_term_struct":   iv_term_struct,
        "dte_30":           c30[1] if c30 else None,
        "dte_60":           c60[1] if c60 else None,
        "n_valid_iv_contracts": len(contracts),
    }


def _load_watchlist(strategy_dir: Path) -> list[str]:
    import json
    cfg = json.loads((strategy_dir / "strategy_config.json").read_text())
    wl = cfg.get("watchlist", []) or cfg.get("data", {}).get("watchlist", [])
    if not wl:
        raise RuntimeError("watchlist empty in strategy_config.json")
    return list(wl)


def _fetch_spot(symbol: str) -> float | None:
    """Quick spot price via yfinance (free, no rate limit issue at our scale)."""
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period="1d", auto_adjust=True)
        if h.empty:
            return None
        return float(h["Close"].iloc[-1])
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy-dir", default="renquant_104")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="override watchlist with explicit symbols")
    p.add_argument("--rate-limit", type=int, default=180,
                   help="max calls / 60s (default 180 = 90% of Free tier 200/min)")
    args = p.parse_args()

    import os
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        log.error("ALPACA_API_KEY / SECRET not in env. "
                  "Run with `set -a; source .env; set +a; python ...`")
        return 3

    if args.symbols:
        symbols = args.symbols
    else:
        strategy_dir = REPO / "backtesting" / args.strategy_dir
        symbols = _load_watchlist(strategy_dir)

    from alpaca.data.historical.option import OptionHistoricalDataClient
    client = OptionHistoricalDataClient(api_key=key, secret_key=secret)
    bucket = TokenBucket(max_calls=args.rate_limit, window_seconds=60.0)

    log.info("fetching IV features for %d symbols (rate=%d/min)",
             len(symbols), args.rate_limit)

    all_rows: list[dict] = []
    for i, sym in enumerate(symbols):
        spot = _fetch_spot(sym)
        if spot is None:
            log.warning("  %s: spot not available, skipping", sym)
            continue
        feats = fetch_iv_features(client, sym, spot, bucket)
        if feats is None:
            continue
        all_rows.append(feats)
        # Persist per-symbol incrementally so a crash mid-run doesn't lose data
        out_p = OUT_DIR / f"{sym}.parquet"
        new_df = pd.DataFrame([feats])
        if out_p.exists():
            prior = pd.read_parquet(out_p)
            merged = pd.concat([prior, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["symbol", "as_of"], keep="last")
            merged = merged.sort_values("as_of").reset_index(drop=True)
            merged.to_parquet(out_p, index=False)
        else:
            new_df.to_parquet(out_p, index=False)
        if (i + 1) % 5 == 0:
            log.info("  %d/%d  (latest: %s skew=%.3f termstr=%.3f)",
                     i+1, len(symbols), sym,
                     feats["iv_skew_30d"], feats["iv_term_struct"])

    log.info("DONE. wrote %d IV snapshots → %s/",
             len(all_rows), OUT_DIR.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
