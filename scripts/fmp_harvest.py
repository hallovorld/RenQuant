#!/usr/bin/env python3
"""One-time FMP harvest — store everything for the universe while the paid month
is active (see doc/research/2026-06-25-fmp-harvest-plan.md).

Auditable + fail-closed + resumable:
- Every endpoint writes a parquet AND a sidecar manifest (`<key>_291.manifest.json`)
  recording requested/with_data/no_data/http_error/fetch_error counts, error samples,
  endpoint URL template, started/finished, row+ticker counts, and the output sha256.
- Resumable on the MANIFEST, not just the parquet: an endpoint is skipped only when its
  manifest exists AND status == "ok". A partial/errored run re-pulls on the next invocation.
- Writes are atomic (tmp file -> os.replace) so an interruption never leaves a half file
  that a later run would trust.
- Fail-closed: any http_error/fetch_error makes the run exit non-zero unless --allow-errors.
  `no_data` (e.g. an ETF with no fundamentals, a name with no dividends) is EXPECTED and
  does not fail the run.
- Bounded retry/backoff on 429/5xx/timeouts.

Per-ticker endpoints are pulled for the alpha158 training universe (291); treasury is
universe-agnostic. Throttle ~0.2s (≈300/min Starter cap). Key from FMP_API_KEY (.env,
never committed).

  fmp_harvest.py --out data/fmp_harvest --rate 0.2 [--only analyst] [--allow-errors]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://financialmodelingprep.com/stable"
RETRY_CODES = {429, 500, 502, 503, 504}

# Macro indicators verified available on Starter 2026-06-25 (each a single call;
# rows already carry a `name` field, so we stamp ticker=<indicator>).
ECON_NAMES = ("GDP", "realGDP", "CPI", "inflationRate", "unemploymentRate",
              "federalFunds", "retailSales", "consumerSentiment")

# (endpoint_key, path_template, per_ticker?) — path uses {sym} for per-ticker.
# NOTE: institutional-ownership is intentionally omitted — verified 2026-06-25 to be
# plan-locked above Starter (402 "Restricted Endpoint"); keeping it would always fail-close.
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
    ("insider_trading", "insider-trading/search?symbol={sym}", True),
    ("shares_float", "shares-float?symbol={sym}", True),
    # F. macro (universe-agnostic). A tuple/list `targets` iterates those values as
    # {sym} instead of the ticker universe — economic_indicators is one call per name.
    ("treasury_rates", "treasury-rates", False),
    ("economic_indicators", "economic-indicators?name={sym}", ECON_NAMES),
]


def classify(payload):
    """Pure classifier of a parsed FMP response → (status, rows).

    status ∈ {with_data, no_data, http_error, fetch_error}. `_get` returns
    {"_http": code} / {"_err": name} sentinels for transport failures; a real
    list/dict payload is data. An empty list is legitimate no-coverage (no_data).
    """
    if isinstance(payload, dict):
        if "_http" in payload:
            return "http_error", []
        if "_err" in payload:
            return "fetch_error", []
        return "with_data", [payload]
    if isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]
        return ("with_data", rows) if rows else ("no_data", [])
    return "no_data", []


def _get(path: str, key: str, retries: int = 3, backoff: float = 0.5):
    """GET with bounded exponential backoff on 429/5xx/timeouts."""
    sep = "&" if "?" in path else "?"
    url = f"{BASE}/{path}{sep}apikey={key}"
    last = {"_err": "Unattempted"}
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as e:
            last = {"_http": e.code}
            if e.code in RETRY_CODES and attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            return last
        except Exception as e:  # noqa: BLE001
            last = {"_err": type(e).__name__}
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            return last
    return last


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_parquet(df, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dst)


def _atomic_write_json(obj, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, dst)


def _manifest_ok(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        return json.loads(manifest_path.read_text()).get("status") == "ok"
    except (ValueError, OSError):
        return False


def harvest_endpoint(key_name, tmpl, per_ticker, uni, out, rate, key, fetched,
                     retries, backoff, get=_get):
    """Pull one endpoint, write parquet + manifest atomically, return the manifest.

    `get` is injectable so tests can drive the classify/write/manifest path offline.
    """
    import pandas as pd  # noqa: PLC0415

    # per_ticker: True → the ticker universe; a tuple/list → those exact {sym} values
    # (e.g. macro indicator names); falsy → a single universe-agnostic call.
    if per_ticker is True:
        targets = uni
    elif isinstance(per_ticker, (list, tuple)):
        targets = list(per_ticker)
    else:
        targets = ["_"]
    rows, counts, errs = [], {"with_data": 0, "no_data": 0, "http_error": 0, "fetch_error": 0}, []
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    for sym in targets:
        status, recs = classify(get(tmpl.format(sym=sym), key, retries, backoff))
        counts[status] += 1
        if status == "with_data":
            rows.extend({**r, "ticker": sym} for r in recs)
        elif status in ("http_error", "fetch_error") and len(errs) < 10:
            errs.append({"ticker": sym})
        time.sleep(rate)

    df = pd.DataFrame(rows)
    dst = out / f"{key_name}_291.parquet"
    if len(df):
        df["fetched_at"] = fetched
        df["source"] = f"fmp_{key_name}"
        _atomic_write_parquet(df, dst)
    bad = counts["http_error"] + counts["fetch_error"]
    manifest = {
        "endpoint": key_name,
        "path_template": tmpl,
        "url_base": BASE,
        "requested": len(targets),
        **counts,
        "error_samples": errs,
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()) if len(df) else 0,
        "output": dst.name if len(df) else None,
        "sha256": _sha256(dst) if len(df) else None,
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "errors" if bad else "ok",
    }
    _atomic_write_json(manifest, out / f"{key_name}_291.manifest.json")
    return manifest


def _universe(repo: Path) -> list[str]:
    import pandas as pd  # noqa: PLC0415

    d = pd.read_parquet(repo / "data" / "alpha158_291_fund_regime_dataset.parquet",
                        columns=["ticker"])
    return sorted(str(t).upper() for t in d["ticker"].unique())


def harvest(out: Path, rate: float, only, key, repo, retries, backoff,
            allow_errors) -> int:
    import pandas as pd  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    uni = _universe(repo)
    fetched = pd.Timestamp("today").normalize()
    eps = [e for e in ENDPOINTS if (only is None or only in e[0])]
    print(f"universe={len(uni)} endpoints={len(eps)} → {out}", flush=True)
    failed = []
    for key_name, tmpl, per_ticker in eps:
        manifest_path = out / f"{key_name}_291.manifest.json"
        if (out / f"{key_name}_291.parquet").exists() and _manifest_ok(manifest_path):
            print(f"  skip {key_name} (manifest ok)", flush=True)
            continue
        m = harvest_endpoint(key_name, tmpl, per_ticker, uni, out, rate, key,
                             fetched, retries, backoff)
        print(f"  {key_name}: tickers={m['tickers']} rows={m['rows']} "
              f"no_data={m['no_data']} http_err={m['http_error']} "
              f"fetch_err={m['fetch_error']} → {m['status']}", flush=True)
        if m["status"] != "ok":
            failed.append(key_name)
    if failed and not allow_errors:
        print(f"HARVEST FAILED — endpoints with errors: {failed} "
              f"(rerun re-pulls them; --allow-errors to tolerate)", flush=True)
        return 1
    print(f"HARVEST DONE (errored={failed or 'none'})", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/fmp_harvest")
    ap.add_argument("--rate", type=float, default=0.2, help="sleep between calls (≈300/min at 0.2)")
    ap.add_argument("--only", default=None, help="substring filter on endpoint key (e.g. 'analyst')")
    ap.add_argument("--repo", default="/Users/renhao/git/github/RenQuant")
    ap.add_argument("--retries", type=int, default=3, help="retry attempts on 429/5xx/timeout")
    ap.add_argument("--backoff", type=float, default=0.5, help="base backoff seconds (exponential)")
    ap.add_argument("--allow-errors", action="store_true",
                    help="exit 0 even if some endpoints errored (default: fail closed)")
    args = ap.parse_args(argv)
    key = os.environ.get("FMP_API_KEY")
    if not key:
        print("FMP_API_KEY not set")
        return 1
    return harvest(Path(args.out), args.rate, args.only, key, Path(args.repo),
                   args.retries, args.backoff, args.allow_errors)


if __name__ == "__main__":
    raise SystemExit(main())
