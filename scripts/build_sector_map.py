#!/usr/bin/env python
"""Build a complete sector_map for a watchlist using IWB CSV sectors.

Pre-fix the wl178 strategy_config had sector_map covering only 104 of
178 tickers (58%). The Layer 1 sector-rank-norm fell back to global
percentile for 42% of the universe — defeating the architecture's
purpose. This script pulls GICS sectors from the iShares IWB
Russell-1000 ETF holdings CSV (already has the Sector column) and
emits an updated config.

Usage::

    python scripts/build_sector_map.py \\
        --in  backtesting/renquant_104/strategy_config.wl178.json \\
        --out backtesting/renquant_104/strategy_config.wl178_v2.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("build-sector-map")

IWB_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)

# Map IWB GICS-level-1 names to short snake_case labels — keeps existing
# sector_map shape consistent (was using snake_case before).
GICS_TO_SHORT: dict[str, str] = {
    "Information Technology":   "tech",
    "Financials":               "finance",
    "Health Care":              "healthcare",
    "Consumer Discretionary":   "consumer_disc",
    "Consumer Staples":         "consumer_staples",
    "Industrials":              "industrial",
    "Communication":            "comm",
    "Energy":                   "energy",
    "Materials":                "materials",
    "Real Estate":              "reit",
    "Utilities":                "utility",
    "Cash and/or Derivatives":  "cash",
}


def fetch_iwb_sectors() -> dict[str, str]:
    """Return ticker → short-sector dict from current IWB holdings."""
    log.info("Fetching IWB Russell 1000 holdings…")
    import pandas as pd  # noqa: PLC0415

    req = urllib.request.Request(
        IWB_URL,
        headers={"User-Agent": "Mozilla/5.0 RenQuant sector-map builder"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    lines = raw.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.lstrip('"').startswith("Ticker")),
        None,
    )
    if header_idx is None:
        log.error("IWB CSV header not found"); sys.exit(1)
    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    if "Sector" not in df.columns or "Ticker" not in df.columns:
        log.error("Required columns Ticker+Sector missing in IWB CSV"); sys.exit(1)

    out: dict[str, str] = {}
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip().upper().replace(".", "-")
        sector = str(row["Sector"]).strip()
        if not ticker or sector in ("", "nan", "Cash and/or Derivatives"):
            continue
        short = GICS_TO_SHORT.get(sector, sector.lower().replace(" ", "_"))
        out[ticker] = short
    log.info("IWB: %d ticker → sector mappings", len(out))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in",  dest="src",
                   default=str(REPO_ROOT / "backtesting" / "renquant_104"
                               / "strategy_config.wl178.json"))
    p.add_argument("--out", dest="out",
                   default=str(REPO_ROOT / "backtesting" / "renquant_104"
                               / "strategy_config.wl178_v2.json"))
    p.add_argument("--keep-existing", action="store_true",
                   help="Preserve hand-curated sector labels for tickers "
                        "already in the source config (default behavior).")
    p.add_argument("--audit-label", default="wl178_v2_full_sector_map",
                   help="String tag written into config['_audit_label'].")
    args = p.parse_args()

    cfg_path = Path(args.src)
    out_path = Path(args.out)
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path); return 1

    cfg = json.loads(cfg_path.read_text())
    wl = cfg.get("watchlist", [])
    existing_sm = dict(cfg.get("sector_map", {}))
    log.info("Source config: %d tickers, %d sector_map entries",
             len(wl), len(existing_sm))

    iwb_sectors = fetch_iwb_sectors()

    new_sm = dict(existing_sm) if args.keep_existing else {}
    n_added = 0
    n_unmapped = 0
    for t in wl:
        if t in new_sm:
            continue
        if t in iwb_sectors:
            new_sm[t] = iwb_sectors[t]
            n_added += 1
        else:
            n_unmapped += 1

    coverage_before = len(set(existing_sm) & set(wl))
    coverage_after  = len(set(new_sm) & set(wl))
    log.info("Coverage: %d → %d / %d  (+%d added, %d still unmapped)",
             coverage_before, coverage_after, len(wl), n_added, n_unmapped)

    # Defensive: if any ticker is still unmapped, log them so operator
    # can see what falls back. The fallback_global=True path handles
    # them gracefully but we want visibility.
    still_missing = [t for t in wl if t not in new_sm]
    if still_missing:
        log.warning("Still unmapped (%d): %s",
                    len(still_missing), still_missing[:20])

    cfg["sector_map"] = new_sm
    cfg["_audit_label"] = args.audit_label

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2))
    log.info("Wrote %s", out_path)

    # Print sector size distribution
    from collections import Counter
    counts = Counter(new_sm[t] for t in wl if t in new_sm)
    print()
    print(f"Sector distribution in updated config ({len(counts)} sectors):")
    for s, n in counts.most_common():
        print(f"  {s:<22} {n:>3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
