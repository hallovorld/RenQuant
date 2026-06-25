#!/usr/bin/env python3
"""One-time FMP harvest — store everything for the universe while the paid month
is active (see doc/research/2026-06-25-fmp-harvest-plan.md).

Resumable: skips an endpoint whose output parquet already exists. Per-ticker
endpoints are pulled for the alpha158 training universe (291); a few are
universe-agnostic (macro). Every row stamped ticker + fetched_at + source.
Throttle ~0.2s (≈300/min Starter cap). Key from FMP_API_KEY (.env, never committed).

  fmp_harvest.py --out data/fmp_harvest --rate 0.2 [--only analyst]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = "https://financialmodelingprep.com/stable"

# (endpoint_key, path_template, per_ticker?) — path uses {sym} for per-ticker.
ENDPOINTS = [
    # A. analyst (high value — feeds the retrain)
    ("grades_historical", "grades-historical?symbol={sym}", True),
    ("grades_consensus", "grades-consensus?symbol={sym}", True),
    ("analyst_estimates", "analyst-estimates?symbol={sym}&period=annual", True),
    ("price_target_consensus", "price-target-consensus?symbol={sym}", True),
    ("price_target_summary", "price-target-summary?symbol={sym}", True),
    # B. fundamentals
    ("income_statement", "income-statement?symbol={sym}&period=annual&limit=20", True),
    ("balance_sheet", "balance-sheet-statement?symbol={sym}&period=annual&limit=20", True),
    ("cash_flow", "cash-flow-statement?symbol={sym}&period=annual&limit=20", True),
    ("ratios", "ratios?symbol={sym}&period=annual&limit=20", True),
    ("key_metrics", "key-metrics?symbol={sym}&period=annual&limit=20", True),
    ("financial_growth", "financial-growth?symbol={sym}&period=annual&limit=20", True),
    ("enterprise_values", "enterprise-values?symbol={sym}&limit=20", True),
    ("market_cap", "historical-market-capitalization?symbol={sym}", True),
    # C. earnings & events
    ("earnings", "earnings?symbol={sym}", True),
    ("dividends", "dividends?symbol={sym}", True),
    ("splits", "splits?symbol={sym}", True),
    # D. ownership & flow
    ("institutional_ownership", "institutional-ownership/symbol-ownership?symbol={sym}", True),
    ("insider_trading", "insider-trading/search?symbol={sym}", True),
    ("shares_float", "shares-float?symbol={sym}", True),
    # F. macro (universe-agnostic)
    ("treasury_rates", "treasury-rates", False),
]


def _universe(repo: Path) -> list[str]:
    import pandas as pd  # noqa: PLC0415
    d = pd.read_parquet(repo / "data" / "alpha158_291_fund_regime_dataset.parquet",
                        columns=["ticker"])
    return sorted(str(t).upper() for t in d["ticker"].unique())


def _get(path: str, key: str):
    sep = "&" if "?" in path else "?"
    url = f"{BASE}/{path}{sep}apikey={key}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        return {"_http": e.code}
    except Exception as e:  # noqa: BLE001
        return {"_err": type(e).__name__}


def harvest(out: Path, rate: float, only: str | None, key: str, repo: Path) -> None:
    import pandas as pd  # noqa: PLC0415
    out.mkdir(parents=True, exist_ok=True)
    uni = _universe(repo)
    fetched = pd.Timestamp("today").normalize()
    eps = [e for e in ENDPOINTS if (only is None or only in e[0])]
    print(f"universe={len(uni)} endpoints={len(eps)} → {out}", flush=True)
    for key_name, tmpl, per_ticker in eps:
        dst = out / f"{key_name}_291.parquet"
        if dst.exists():
            print(f"  skip {key_name} (exists)", flush=True)
            continue
        rows, miss, err = [], 0, 0
        targets = uni if per_ticker else ["_"]
        for sym in targets:
            d = _get(tmpl.format(sym=sym), key)
            if isinstance(d, list) and d:
                rows.extend({**r, "ticker": sym} for r in d if isinstance(r, dict))
            elif isinstance(d, dict) and not any(k.startswith("_") for k in d):
                rows.append({**d, "ticker": sym})
            elif isinstance(d, dict) and "_http" in d:
                err += 1
            else:
                miss += 1
            time.sleep(rate)
        df = pd.DataFrame(rows)
        if len(df):
            df["fetched_at"] = fetched
            df["source"] = f"fmp_{key_name}"
            df.to_parquet(dst, index=False)
        print(f"  {key_name}: tickers={df['ticker'].nunique() if len(df) else 0} "
              f"rows={len(df)} miss={miss} err={err} → {dst.name}", flush=True)
    print("HARVEST DONE", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/fmp_harvest")
    ap.add_argument("--rate", type=float, default=0.2, help="sleep between calls (≈300/min at 0.2)")
    ap.add_argument("--only", default=None, help="substring filter on endpoint key (e.g. 'analyst')")
    ap.add_argument("--repo", default="/Users/renhao/git/github/RenQuant")
    args = ap.parse_args(argv)
    key = os.environ.get("FMP_API_KEY")
    if not key:
        print("FMP_API_KEY not set"); return 1
    harvest(Path(args.out), args.rate, args.only, key, Path(args.repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
