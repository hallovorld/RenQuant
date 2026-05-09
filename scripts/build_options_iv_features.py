#!/usr/bin/env python
"""P0 #2 — Build options-IV features per Bali-Hovakimian RFS 2009 + Goyal-Saretto JFE 2009.

Pulls Yahoo Finance options chain per ticker + day, computes:
  - 25-delta put-call skew    (Bali-Hovakimian "vol spread")
  - 30d/90d IV term-structure slope
  - IV rank percentile (52w)

References:
- Bali-Hovakimian 2009 RFS "Volatility Spreads and Expected Stock Returns" §3
- Goyal-Saretto 2009 JFE "Cross-section of option returns and volatility"
- Cremers-Weinbaum 2010 JFQA "Deviations from Put-Call Parity"

Output: data/options_iv_panel.parquet with columns:
  date, ticker, iv25_put_call_skew, iv_term_slope_30_90, iv_rank_52w

Yahoo finance options chain is FREE; coverage is current snapshot only,
so this script must be run periodically and joined with current panel.
For backfill, would need paid data (e.g., OptionMetrics).
"""
from __future__ import annotations
import argparse, datetime, json, logging, time, sys
from pathlib import Path
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("options-iv")

REPO = Path(__file__).resolve().parent.parent


def fetch_options_chain(ticker: str) -> dict | None:
    """Fetch options chain via yfinance. Returns dict with calls/puts dataframes per expiry."""
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        log.error("yfinance not installed — pip install yfinance")
        sys.exit(2)
    try:
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return None
        out = {}
        for exp in expiries[:6]:    # first 6 expiries cover 30-180 days typically
            try:
                opts = t.option_chain(exp)
                out[exp] = {"calls": opts.calls, "puts": opts.puts}
            except Exception as e:
                log.debug("  %s exp=%s failed: %s", ticker, exp, e)
        return out
    except Exception as e:
        log.warning("ticker %s: %s", ticker, e)
        return None


def compute_features(ticker: str, chain: dict, spot: float) -> dict | None:
    """Extract IV features from chain.

    25-delta skew: IV(put with delta ≈ -0.25) - IV(call with delta ≈ +0.25).
    Bali-Hovakimian use 1m expiry; we use closest-to-30d.

    Term slope: IV(60-90d ATM) - IV(20-40d ATM).
    """
    today = datetime.date.today()
    # Pick 30-day expiry (or closest)
    target_30d = today + datetime.timedelta(days=30)
    target_90d = today + datetime.timedelta(days=90)

    def parse_date(s):
        try:
            return datetime.datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

    def closest(target):
        best, best_diff = None, 999999
        for exp_str in chain:
            d = parse_date(exp_str)
            if d is None:
                continue
            diff = abs((d - target).days)
            if diff < best_diff:
                best, best_diff = exp_str, diff
        return best

    exp_30 = closest(target_30d)
    exp_90 = closest(target_90d)
    if exp_30 is None:
        return None

    # ATM IV around 30d expiry
    calls_30 = chain[exp_30]["calls"]
    puts_30 = chain[exp_30]["puts"]
    if len(calls_30) == 0 or len(puts_30) == 0:
        return None

    # Find ATM strike (closest to spot)
    calls_30 = calls_30.copy(); puts_30 = puts_30.copy()
    calls_30["atm_dist"] = (calls_30["strike"] - spot).abs()
    puts_30["atm_dist"] = (puts_30["strike"] - spot).abs()
    atm_call_iv = float(calls_30.nsmallest(1, "atm_dist")["impliedVolatility"].iloc[0])
    atm_put_iv = float(puts_30.nsmallest(1, "atm_dist")["impliedVolatility"].iloc[0])
    atm_iv_30 = (atm_call_iv + atm_put_iv) / 2.0

    # 25-delta skew (proxy: 10% OTM put IV - 10% OTM call IV)
    # True 25-delta strike requires Black-Scholes inversion; 10% OTM is robust proxy
    otm_strike_call = spot * 1.10
    otm_strike_put = spot * 0.90
    calls_30["otm_dist"] = (calls_30["strike"] - otm_strike_call).abs()
    puts_30["otm_dist"] = (puts_30["strike"] - otm_strike_put).abs()
    if len(calls_30) > 0 and len(puts_30) > 0:
        otm_call_iv = float(calls_30.nsmallest(1, "otm_dist")["impliedVolatility"].iloc[0])
        otm_put_iv = float(puts_30.nsmallest(1, "otm_dist")["impliedVolatility"].iloc[0])
        skew_25d = otm_put_iv - otm_call_iv     # Bali-Hovakimian put-call IV spread
    else:
        skew_25d = float("nan")

    # Term slope
    iv_term_slope = float("nan")
    if exp_90 is not None and exp_90 != exp_30:
        calls_90 = chain[exp_90]["calls"].copy()
        puts_90 = chain[exp_90]["puts"].copy()
        if len(calls_90) > 0 and len(puts_90) > 0:
            calls_90["atm_dist"] = (calls_90["strike"] - spot).abs()
            puts_90["atm_dist"] = (puts_90["strike"] - spot).abs()
            atm_iv_90 = (
                float(calls_90.nsmallest(1, "atm_dist")["impliedVolatility"].iloc[0]) +
                float(puts_90.nsmallest(1, "atm_dist")["impliedVolatility"].iloc[0])
            ) / 2.0
            iv_term_slope = atm_iv_90 - atm_iv_30

    return {
        "ticker": ticker, "date": today,
        "iv25_put_call_skew": skew_25d,
        "iv_term_slope_30_90": iv_term_slope,
        "atm_iv_30d": atm_iv_30,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default=None,
                    help="Comma-separated ticker list (default: from strategy_config)")
    ap.add_argument("--out", default=str(REPO / "data/options_iv_snapshot.parquet"))
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit to N tickers (for testing)")
    args = ap.parse_args()

    if args.watchlist:
        wl = args.watchlist.split(",")
    else:
        cfg = json.loads((REPO / "backtesting/renquant_104/strategy_config.json").read_text())
        wl = cfg.get("watchlist", [])

    if args.limit:
        wl = wl[:args.limit]
    log.info("Fetching options-IV for %d tickers...", len(wl))

    rows = []
    for i, tkr in enumerate(wl):
        # Get spot from local OHLCV cache
        ohlcv_p = REPO / f"data/ohlcv/{tkr}/1d.parquet"
        if ohlcv_p.exists():
            df = pd.read_parquet(ohlcv_p)
            spot = float(df["close"].iloc[-1])
        else:
            log.warning("  %s: no OHLCV cache — skip", tkr)
            continue

        chain = fetch_options_chain(tkr)
        if chain is None:
            log.warning("  %s: no options chain", tkr)
            continue
        feats = compute_features(tkr, chain, spot)
        if feats is None:
            log.warning("  %s: feature compute failed", tkr)
            continue
        rows.append(feats)
        if (i + 1) % 10 == 0:
            log.info("  progress %d/%d (last: %s skew=%+.3f slope=%+.3f)",
                     i + 1, len(wl), tkr,
                     feats.get("iv25_put_call_skew", float("nan")),
                     feats.get("iv_term_slope_30_90", float("nan")))
        time.sleep(0.5)   # be polite to Yahoo

    if not rows:
        log.error("no rows collected")
        sys.exit(2)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("Saved %d rows → %s", len(df), args.out)
    log.info("Stats:")
    log.info("  iv25_put_call_skew: mean=%+.3f std=%.3f n_finite=%d/%d",
             df["iv25_put_call_skew"].mean(), df["iv25_put_call_skew"].std(),
             df["iv25_put_call_skew"].notna().sum(), len(df))
    log.info("  iv_term_slope_30_90: mean=%+.3f std=%.3f",
             df["iv_term_slope_30_90"].mean(), df["iv_term_slope_30_90"].std())


if __name__ == "__main__":
    main()
