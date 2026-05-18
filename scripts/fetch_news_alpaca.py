#!/usr/bin/env python3
"""Fetch Alpaca News API headlines per watchlist ticker.

Roadmap C5 (2026-05-18 user mandate, $0/mo). Alpaca News API is
free for paper + live accounts (200 calls/min Free tier; Benzinga
partnership; history back to 2015).

Output: data/news_alpaca/{ticker}.parquet with columns:
  symbol, created_at, updated_at, headline, summary, author, url

Rate-limit strategy:
  • Token-bucket at 180 calls/min (90% of Free tier 200/min) to
    leave headroom for other processes hitting the same key.
  • Exponential backoff on 429 (1s → 2s → 4s ... cap 60s).
  • Per-symbol pagination via page_token; respects max_per_request.

Per CLAUDE.md §5.13.6: any new cron-cadence script docstring must
answer "what fresh info does this add per tick". This script:
  • Daily run: adds yesterday's headlines (1-day delta)
  • Backfill mode (`--since YYYY-MM-DD`): for first-run + retroactive
    label training. Backfill respects rate-limit.

References:
  - Alpaca docs: https://docs.alpaca.markets/us/docs/streaming-real-time-news
  - Tetlock 2007 *JF* "Giving Content to Investor Sentiment" — daily
    sentiment from newspaper text predicts S&P returns
  - Ke-Kelly-Xiu 2019 *NBER w26261* "Predicting Returns with Text Data"
    — supervised sentiment topics from per-ticker news

NOTE: this script ONLY fetches + persists headlines. FinBERT scoring
is a separate script (`scripts/score_news_finbert.py`, TBD).
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from collections import deque
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "news_alpaca"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch_news_alpaca")


class TokenBucket:
    """Simple sliding-window rate limiter.

    Allows up to `max_calls` in any rolling `window_seconds`.
    Sleeps the calling thread until a slot opens.

    Default: 180 calls / 60s (90% of Alpaca Free tier 200/min, leaving
    headroom for other concurrent users of the same API key).
    """

    def __init__(self, max_calls: int = 180, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        now = time.time()
        # Drop timestamps outside the window
        while self._timestamps and self._timestamps[0] <= now - self.window:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_calls:
            # Sleep until the oldest timestamp falls out of the window
            sleep_for = self.window - (now - self._timestamps[0]) + 0.05
            log.debug("rate-limit: sleeping %.2fs (queue=%d)",
                      sleep_for, len(self._timestamps))
            time.sleep(max(0.05, sleep_for))
            now = time.time()
            while self._timestamps and self._timestamps[0] <= now - self.window:
                self._timestamps.popleft()
        self._timestamps.append(now)


def _load_watchlist(strategy_dir: Path) -> list[str]:
    """Read the production watchlist (103 tickers as of 2026-05)."""
    import json
    cfg = json.loads((strategy_dir / "strategy_config.json").read_text())
    wl = cfg.get("watchlist", [])
    if not wl:
        # Fall back to whichever top-level field holds it
        wl = cfg.get("data", {}).get("watchlist", [])
    if not wl:
        raise RuntimeError("watchlist empty — check strategy_config.json")
    return list(wl)


def _fetch_one_symbol(client, bucket: TokenBucket, symbol: str,
                     start: datetime, end: datetime,
                     max_per_request: int = 50) -> pd.DataFrame:
    """Fetch ALL news for one symbol in [start, end), paginated.

    Returns: DataFrame with columns symbol, created_at, headline, summary,
    author, url, updated_at.
    """
    from alpaca.data.requests import NewsRequest
    rows: list[dict] = []
    page_token: str | None = None
    backoff = 1.0
    while True:
        bucket.acquire()
        req = NewsRequest(
            symbols=symbol,
            start=start,
            end=end,
            limit=max_per_request,
            page_token=page_token,
            include_content=False,
            sort="asc",
        )
        try:
            resp = client.get_news(req)
        except Exception as exc:
            msg = str(exc)
            if "rate" in msg.lower() or "429" in msg:
                log.warning("rate-limited on %s — backoff %.1fs", symbol, backoff)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)
                continue
            raise
        backoff = 1.0  # reset on success

        # 2026-05-18: Alpaca NewsSet returns `data["news"]` (NOT
        # `data[symbol]`) — single flat list of News objects regardless
        # of how many symbols requested. Each News object's `symbols`
        # field lists all tickers tagged in the article.
        if isinstance(resp, dict):
            news_list = resp.get("news", []) or resp.get("data", {}).get("news", [])
        else:
            data = getattr(resp, "data", {}) or {}
            news_list = data.get("news", [])
        if news_list is None:
            news_list = []
        for n in news_list:
            d = n.model_dump() if hasattr(n, "model_dump") else (
                n.dict() if hasattr(n, "dict") else (n if isinstance(n, dict) else {})
            )
            # The API returns articles where ANY of the requested symbols
            # is in `symbols`. Persist the canonical `symbol` column as
            # the per-row symbol so downstream FinBERT scoring fans out
            # correctly per ticker (one article → N ticker-rows if
            # tagged with N symbols).
            rows.append({
                "symbol":     symbol,
                "created_at": d.get("created_at"),
                "updated_at": d.get("updated_at"),
                "headline":   d.get("headline"),
                "summary":    d.get("summary"),
                "author":     d.get("author"),
                "url":        d.get("url"),
                "all_symbols": ",".join(d.get("symbols", []) or []),
            })
        # Pagination
        next_token = getattr(resp, "next_page_token", None) or (
            resp.get("next_page_token") if isinstance(resp, dict) else None
        )
        if not next_token:
            break
        page_token = next_token

    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
        df = df.drop_duplicates(subset=["symbol", "created_at", "headline"])
        df = df.sort_values("created_at").reset_index(drop=True)
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy-dir", default="renquant_104",
                   help="strategy under backtesting/ (for watchlist source)")
    p.add_argument("--since", type=lambda s: date.fromisoformat(s),
                   default=None,
                   help="backfill start date YYYY-MM-DD. Default: yesterday "
                        "(daily-delta mode).")
    p.add_argument("--until", type=lambda s: date.fromisoformat(s),
                   default=None,
                   help="end date (exclusive). Default: today.")
    p.add_argument("--max-per-request", type=int, default=50,
                   help="Alpaca page size (cap 50 per docs).")
    p.add_argument("--rate-limit", type=int, default=180,
                   help="max calls / 60s (default 180, 90% of free 200/min)")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="override watchlist with explicit symbols")
    args = p.parse_args()

    # Date range defaults
    if args.until is None:
        args.until = date.today() + timedelta(days=1)
    if args.since is None:
        args.since = args.until - timedelta(days=1)
    if args.since >= args.until:
        log.error("--since must be < --until")
        return 2

    start_dt = datetime.combine(args.since, datetime.min.time(), tzinfo=timezone.utc)
    end_dt   = datetime.combine(args.until, datetime.min.time(), tzinfo=timezone.utc)

    # Symbols
    if args.symbols:
        symbols = args.symbols
    else:
        strategy_dir = REPO / "backtesting" / args.strategy_dir
        symbols = _load_watchlist(strategy_dir)

    log.info("fetching news: %d symbols × [%s → %s)  rate=%d/min",
             len(symbols), args.since, args.until, args.rate_limit)

    # Auth
    import os
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        log.error("ALPACA_API_KEY / SECRET not in env. "
                  "Run with `set -a; source .env; set +a; python ...`")
        return 3

    from alpaca.data.historical.news import NewsClient
    client = NewsClient(api_key=key, secret_key=secret)
    bucket = TokenBucket(max_calls=args.rate_limit, window_seconds=60.0)

    n_total = 0
    for i, sym in enumerate(symbols):
        try:
            df = _fetch_one_symbol(client, bucket, sym, start_dt, end_dt,
                                    args.max_per_request)
        except Exception as exc:
            log.warning("  %s: fetch failed — %s", sym, exc)
            continue
        out_p = OUT_DIR / f"{sym}.parquet"
        if not df.empty:
            # Merge with prior file if exists (idempotent backfill)
            if out_p.exists():
                prior = pd.read_parquet(out_p)
                df = pd.concat([prior, df], ignore_index=True)
                df = df.drop_duplicates(subset=["symbol", "created_at", "headline"])
                df = df.sort_values("created_at").reset_index(drop=True)
            df.to_parquet(out_p, index=False)
        n_total += len(df)
        if (i + 1) % 10 == 0:
            log.info("  %d/%d symbols  cumulative news rows: %d",
                     i + 1, len(symbols), n_total)

    log.info("DONE. wrote %d total news rows across %d symbols → %s/",
             n_total, len(symbols), OUT_DIR.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
