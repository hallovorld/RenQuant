#!/usr/bin/env python
"""Pull is_shortable + easy_to_borrow flags from Alpaca for the watchlist.

Free Alpaca asset metadata endpoint. Persisted as parquet for later use
by the short-side pipeline (see doc/research/short-side-design.md §4.2).

Output: data/shortable/{YYYY-MM-DD}.parquet  (one row per ticker per day)
Columns: ticker, is_shortable, easy_to_borrow, fractionable, marginable,
         tradable, asset_class, exchange, status

Usage::

    python scripts/fetch_alpaca_shortable.py
    python scripts/fetch_alpaca_shortable.py --watchlist /path/to/cfg.json
    python scripts/fetch_alpaca_shortable.py --tickers AAPL,MSFT,LITE
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fetch-shortable")


def _load_watchlist(path: Path) -> list[str]:
    cfg = json.loads(path.read_text())
    wl = cfg.get("watchlist") or []
    if not wl:
        raise ValueError(f"No 'watchlist' field in {path}")
    return sorted(set(wl))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--watchlist",
        default=str(REPO_ROOT / "backtesting" / "renquant_104" /
                    "strategy_config.wl_sweep_183.json"),
        help="Strategy config JSON whose 'watchlist' field is used. "
             "Default: wl_sweep_183 (current sweep peak).",
    )
    p.add_argument("--tickers", default=None,
                   help="Override: comma-separated ticker list, e.g. 'AAPL,MSFT'.")
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data" / "shortable"),
        help="Output directory for daily snapshots.",
    )
    args = p.parse_args()

    if args.tickers:
        tickers = sorted(set(t.strip().upper() for t in args.tickers.split(",") if t.strip()))
    else:
        tickers = _load_watchlist(Path(args.watchlist))
    log.info("Pulling shortable metadata for %d tickers", len(tickers))

    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not secret:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in env. "
                  "Source .env first.")
        return 2

    from alpaca.trading.client import TradingClient  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    client = TradingClient(api_key, secret, paper=False)

    rows = []
    n_shortable = 0
    n_etb = 0
    fail = 0
    for tic in tickers:
        try:
            asset = client.get_asset(tic)
        except Exception as exc:
            log.warning("  %s: lookup failed — %s", tic, exc)
            fail += 1
            continue
        rows.append({
            "ticker":         tic,
            "is_shortable":   bool(getattr(asset, "shortable", False)),
            "easy_to_borrow": bool(getattr(asset, "easy_to_borrow", False)),
            "fractionable":   bool(getattr(asset, "fractionable", False)),
            "marginable":     bool(getattr(asset, "marginable", False)),
            "tradable":       bool(getattr(asset, "tradable", False)),
            "asset_class":    str(getattr(asset, "asset_class", "")).split(".")[-1],
            "exchange":       str(getattr(asset, "exchange", "")).split(".")[-1],
            "status":         str(getattr(asset, "status", "")).split(".")[-1],
        })
        if rows[-1]["is_shortable"]:
            n_shortable += 1
        if rows[-1]["easy_to_borrow"]:
            n_etb += 1

    if not rows:
        log.error("No assets retrieved. Check creds / network.")
        return 1

    df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"{today}.parquet"
    df.to_parquet(out_path, index=False)

    log.info(
        "DONE  rows=%d  shortable=%d (%.1f%%)  easy_to_borrow=%d (%.1f%%)  "
        "failed=%d  → %s",
        len(df),
        n_shortable, 100 * n_shortable / max(len(df), 1),
        n_etb,       100 * n_etb       / max(len(df), 1),
        fail, out_path,
    )

    # Quick visibility into HARD-TO-BORROW names — these are the squeeze
    # candidates and the names a future short-side filter will reject first.
    htb = df[df["is_shortable"] & ~df["easy_to_borrow"]]
    if not htb.empty:
        log.info("Hard-to-borrow (shortable but NOT easy-to-borrow): %s",
                 ", ".join(htb["ticker"].tolist()))
    not_shortable = df[~df["is_shortable"]]
    if not not_shortable.empty:
        log.info("NOT shortable at all: %s",
                 ", ".join(not_shortable["ticker"].tolist()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
