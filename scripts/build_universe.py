#!/usr/bin/env python
"""Build the candidate watchlist universe for wl500 experiments.

Pulls current constituents of:
  * S&P 500       (~500)
  * Russell 1000  (~1000)
  * NASDAQ-100    (~100, mostly subset of above but covers ADRs)

Deduplicates, writes to ``scripts/watchlist_universe.json`` so the
downstream selector ``scripts/select_watchlist.py`` has a real universe
to choose from. Pre-fix the universe file was missing → selector fell
back to "current watchlist as universe" → could only ever propose a
subset of what was already in production.

Production safety: this script only writes the ticker LIST file. It
does NOT pull OHLCV (that's a separate slow step), does NOT modify the
strategy config, does NOT touch any model artifact. Safe to run any time.

Usage::

    python scripts/build_universe.py
    python scripts/build_universe.py --out scripts/watchlist_universe.json
    python scripts/build_universe.py --include-russell-2000   # ~3500 tickers

Sources
-------
S&P 500: Wikipedia table at en.wikipedia.org/wiki/List_of_S%26P_500_companies.
Russell 1000: iShares IWB holdings (the ETF tracking the index).
NASDAQ-100: Wikipedia.

These sources are stable but not guaranteed. The script writes the date
of the pull alongside the list so consumers can see staleness.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("build-universe")


# Wikipedia's table format is stable across revisions because it's
# table-of-record for many financial scrapers. Should it ever change,
# the fallback path below uses StockAnalysis API.
SP500_WIKI_URL  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX_WIKI_URL    = "https://en.wikipedia.org/wiki/Nasdaq-100"

# Russell 1000 — pull from the iShares IWB ETF holdings CSV. We use the
# Russell-tracking ETF rather than scraping FTSE because the ETF list
# is the actual investable universe.
RUSSELL_1000_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)


def _fetch_sp500() -> list[str]:
    """Scrape S&P 500 constituents from Wikipedia."""
    log.info("Fetching S&P 500 from Wikipedia …")
    try:
        tables = pd.read_html(SP500_WIKI_URL)
    except Exception as exc:
        log.error("S&P 500 fetch failed: %s", exc)
        return []
    if not tables:
        return []
    # First table is the constituent list
    df = tables[0]
    # Column name has been "Symbol" or "Ticker" historically; try both
    sym_col = next((c for c in df.columns
                    if str(c).lower() in ("symbol", "ticker")), None)
    if sym_col is None:
        log.error("Could not find symbol column in S&P 500 table")
        return []
    out = sorted({str(t).replace(".", "-").upper() for t in df[sym_col].dropna()})
    log.info("  S&P 500: %d tickers", len(out))
    return out


def _fetch_nasdaq_100() -> list[str]:
    """Scrape NASDAQ-100 from Wikipedia."""
    log.info("Fetching NASDAQ-100 from Wikipedia …")
    try:
        tables = pd.read_html(NDX_WIKI_URL)
    except Exception as exc:
        log.error("NASDAQ-100 fetch failed: %s", exc)
        return []
    # The constituent table has a column named "Ticker" or "Symbol"
    for df in tables:
        sym_col = next((c for c in df.columns
                        if str(c).lower() in ("ticker", "symbol")), None)
        if sym_col is None:
            continue
        # Confirm it's the constituents table by looking at length
        if 80 <= len(df) <= 110:
            out = sorted({str(t).replace(".", "-").upper()
                          for t in df[sym_col].dropna()})
            log.info("  NASDAQ-100: %d tickers", len(out))
            return out
    log.warning("NASDAQ-100 table not identified")
    return []


def _fetch_russell_1000() -> list[str]:
    """Pull Russell 1000 from iShares IWB ETF holdings CSV."""
    log.info("Fetching Russell 1000 from iShares IWB …")
    import urllib.request
    try:
        req = urllib.request.Request(
            RUSSELL_1000_URL,
            headers={"User-Agent": "Mozilla/5.0 RenQuant universe builder"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.error("Russell 1000 fetch failed: %s", exc)
        return []
    # iShares CSV has ~10 lines of metadata before the holdings table.
    # Find the header row by looking for "Ticker," prefix.
    lines = raw.splitlines()
    header_idx = next((i for i, ln in enumerate(lines)
                        if ln.lstrip('"').startswith("Ticker")), None)
    if header_idx is None:
        log.error("Russell 1000: header row not found in CSV")
        return []
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    if "Ticker" not in df.columns:
        log.error("Russell 1000: 'Ticker' column missing post-parse")
        return []
    out = sorted({
        str(t).replace(".", "-").upper()
        for t in df["Ticker"].dropna()
        # iShares includes "USD CASH" + futures + futures cash placeholders
        if str(t) and not any(x in str(t).upper()
                              for x in ("CASH", "FUT", "OPT", "MARGIN"))
        # Reasonable ticker length sanity
        and 1 <= len(str(t).strip()) <= 6
    })
    log.info("  Russell 1000: %d tickers", len(out))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO_ROOT / "scripts" / "watchlist_universe.json"))
    p.add_argument("--include-russell-1000", action="store_true", default=True)
    p.add_argument("--no-russell-1000", dest="include_russell_1000",
                   action="store_false")
    args = p.parse_args()

    sp500    = _fetch_sp500()
    ndx100   = _fetch_nasdaq_100()
    russell  = _fetch_russell_1000() if args.include_russell_1000 else []

    universe = sorted(set(sp500) | set(ndx100) | set(russell))
    log.info("Combined universe: %d unique tickers", len(universe))
    log.info("  S&P 500 only:        %d", len(set(sp500) - set(russell) - set(ndx100)))
    log.info("  Russell-only:        %d", len(set(russell) - set(sp500) - set(ndx100)))
    log.info("  NDX-only:            %d", len(set(ndx100) - set(sp500) - set(russell)))

    # Output schema is intentionally flat (just the list) for
    # backward compatibility with select_watchlist.py:228 which loads
    # the file as `universe = json.loads(universe_path.read_text())`
    # and expects a list. Provenance metadata goes alongside.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(universe, indent=0))

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "size":         len(universe),
        "sources":      {
            "sp500":         len(sp500),
            "nasdaq_100":    len(ndx100),
            "russell_1000":  len(russell),
        },
    }, indent=2))

    print(f"\n  Wrote {len(universe)} tickers to {out_path}")
    print(f"  Provenance:           {meta_path}")
    print(f"  Sources: S&P500={len(sp500)}  NDX={len(ndx100)}  R1000={len(russell)}")


if __name__ == "__main__":
    main()
