#!/usr/bin/env python3
"""Fetch earnings calendar for a watchlist and save to a strategy artifact.

Usage:
    python scripts/fetch_earnings_calendar.py --strategy renquant_104
    python scripts/fetch_earnings_calendar.py --strategy renquant_104 \
        --config /path/to/pinned/strategy_config.json --min-horizon-days 5

Output (default): backtesting/{strategy}/artifacts/prod/earnings-calendar.json
    {
      "AAPL": ["2026-01-30", "2026-04-30", ...],
      ...
    }

2026-08-30 fix (data audit): this script was never scheduled — the prod
artifact froze at its last manual run (2026-04-24, last date 2026-07-24)
and the live pre/post-earnings buffer silently stopped firing for every
later print. Changes:
  * writes to artifacts/prod/ when it exists — the path the consumers
    actually read (main.py and adapters/runner_artifacts.py load
    `prod/earnings-calendar.json`). The pre-fix version wrote to
    artifacts/, one level off since the 2026-05-10 sim/prod isolation
    refactor (238359b), so even a manual re-run never refreshed the
    consumed artifact. --out overrides explicitly.
  * --config: fetch for the PINNED strategy config's watchlist (the
    config the live run actually trades) instead of only the umbrella
    copy; default stays the umbrella copy for back-compat.
  * merges with the previous calendar per ticker — a transient vendor
    failure for one ticker must not erase its known dates; dates older
    than --keep-past-days are dropped so the file stays bounded.
  * atomic write (tmp + rename), same convention as the parquet stores.
  * --min-horizon-days N self-check: exit 2 when the merged calendar's
    last date < today+N. The scheduled wrapper turns that into a loud
    ntfy instead of silently installing a still-stale calendar.
  * yfinance import is lazy so the pure helpers are importable in tests
    without the dependency.

Vendor: yfinance (free, no API key, no FMP/Finnhub quota consumed).
~142 tickers x 2 endpoints with a 0.3 s sleep ≈ a few minutes.
"""
import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_watchlist(strategy: str, config: str | None = None) -> list[str]:
    cfg_path = (
        Path(config) if config
        else ROOT / "backtesting" / strategy / "strategy_config.json"
    )
    with cfg_path.open() as f:
        return [str(t).upper() for t in json.load(f)["watchlist"]]


def resolve_output_path(strategy_dir: Path) -> Path:
    """Prefer artifacts/prod/ (the consumed path), then artifacts/, then
    the strategy dir — never invent a directory that isn't there."""
    prod_dir = strategy_dir / "artifacts" / "prod"
    if prod_dir.is_dir():
        return prod_dir / "earnings-calendar.json"
    artifacts_dir = strategy_dir / "artifacts"
    if artifacts_dir.is_dir():
        return artifacts_dir / "earnings-calendar.json"
    return strategy_dir / "earnings-calendar.json"


def load_previous_calendar(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        prev = json.loads(path.read_text())
        return prev if isinstance(prev, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _valid_dates(values) -> list[str]:
    out = []
    if not isinstance(values, (list, tuple)):
        return out
    for v in values:
        s = str(v)[:10]
        try:
            date.fromisoformat(s)
        except ValueError:
            continue
        out.append(s)
    return out


def merge_calendars(
    previous: dict,
    fetched: dict[str, list[str]],
    watchlist: list[str],
    today: date,
    keep_past_days: int = 45,
) -> dict[str, list[str]]:
    """Union previous + fetched dates per watchlist ticker.

    Keeps a ticker's previously-known dates when today's fetch returned
    nothing for it (vendor hiccup != the print vanished), drops dates
    older than `keep_past_days` (the post-earnings sell buffer only
    needs the recent past), and scopes the result to the watchlist."""
    cutoff = (today - timedelta(days=keep_past_days)).isoformat()
    merged: dict[str, list[str]] = {}
    for ticker in watchlist:
        dates = set(_valid_dates(previous.get(ticker, [])))
        dates.update(_valid_dates(fetched.get(ticker, [])))
        merged[ticker] = sorted(d for d in dates if d >= cutoff)
    return merged


def calendar_last_date(calendar: dict) -> str | None:
    best = None
    for dates in calendar.values():
        for d in _valid_dates(dates):
            if best is None or d > best:
                best = d
    return best


def fetch_earnings_dates(ticker: str, lookahead_days: int) -> list[str]:
    """Return upcoming + recent earnings dates for a ticker (best-effort)."""
    import yfinance as yf  # noqa: PLC0415 — lazy so helpers import without it

    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        dates = []

        # yfinance calendar dict may have 'Earnings Date' key
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date", [])
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            for d in raw:
                try:
                    ds = str(d)[:10]
                    dates.append(ds)
                except Exception:
                    pass

        # Also pull from earnings_dates for historical
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                cutoff = date.today() + timedelta(days=lookahead_days)
                for idx in ed.index:
                    try:
                        ds = str(idx)[:10]
                        d  = date.fromisoformat(ds)
                        if d >= date.today() - timedelta(days=30) and d <= cutoff:
                            dates.append(ds)
                    except Exception:
                        pass
        except Exception:
            pass

        return sorted(set(dates))
    except Exception as e:
        print(f"  WARNING: {ticker} earnings fetch failed: {e}", file=sys.stderr)
        return []


def write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch earnings calendar")
    parser.add_argument("--strategy",  required=True)
    parser.add_argument("--config", default=None,
                        help="Strategy config to read the watchlist from "
                             "(e.g. the PINNED subrepo config); default = "
                             "the umbrella backtesting/{strategy}/strategy_config.json")
    parser.add_argument("--lookahead", type=int, default=90,
                        help="Days ahead to fetch earnings (default: 90)")
    parser.add_argument("--keep-past-days", type=int, default=45,
                        help="Drop merged dates older than this (default: 45)")
    parser.add_argument("--min-horizon-days", type=int, default=0,
                        help="Exit 2 if the merged calendar's last date is "
                             "closer than today+N days (0 = no self-check)")
    parser.add_argument("--out", default=None,
                        help="Explicit output path (default: the consumed "
                             "artifacts/prod/earnings-calendar.json)")
    args = parser.parse_args()

    watchlist = load_watchlist(args.strategy, args.config)
    strategy_dir = ROOT / "backtesting" / args.strategy
    out_path = Path(args.out) if args.out else resolve_output_path(strategy_dir)

    print(f"Fetching earnings calendar for {len(watchlist)} symbols → {out_path}")
    fetched: dict[str, list[str]] = {}
    for ticker in watchlist:
        print(f"  {ticker}...", end=" ", flush=True)
        dates = fetch_earnings_dates(ticker, args.lookahead)
        fetched[ticker] = dates
        print(f"{len(dates)} dates")
        time.sleep(0.3)   # polite rate-limit

    today = date.today()
    previous = load_previous_calendar(out_path)
    calendar = merge_calendars(previous, fetched, watchlist,
                               today, args.keep_past_days)
    write_atomic(out_path, calendar)

    n_fetched = sum(1 for d in fetched.values() if d)
    last = calendar_last_date(calendar)
    print(f"\nSaved → {out_path}  ({n_fetched}/{len(watchlist)} tickers "
          f"returned dates; last date = {last})")

    if args.min_horizon_days > 0:
        required = (today + timedelta(days=args.min_horizon_days)).isoformat()
        if last is None or last < required:
            print(f"STALE-AFTER-FETCH: last date {last} < today+"
                  f"{args.min_horizon_days}d ({required}) — vendor returned "
                  f"no usable forward dates", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
