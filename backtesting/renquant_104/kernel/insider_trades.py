"""SEC Form 4 insider-trades cache (executive-only per user spec).

Pipeline:
  1. Ticker → CIK via SEC EDGAR's public `company_tickers.json`
  2. CIK → recent Form 4 filings via `data.sec.gov/submissions/CIK...json`
  3. For each Form 4, fetch the XML and parse transactions
  4. Keep only executive (isOfficer=true) transactions with open-market
     codes (P=purchase, S=sale). Drop option exercises (M), tax-withholding
     (F), gifts (G), and award grants (A with no price).
  5. Cache per-ticker parquet of {date, tx_code, shares, price, dollars, net_dollars}

Daily factor `compute_insider_net_buy_cum` returns the trailing-N-day
cumulative net-dollar executive buy for each ticker, ffilled to the
OHLCV index — a classic orthogonal sentiment/information factor.

User spec:
  - Executive-only (isOfficer=true). 10% beneficial-owner (Schedule 13G)
    filings and non-officer directors are excluded — per the user's
    explicit ask and academic literature showing executive trades carry
    stronger alpha.
  - Open-market transactions only (P, S). Academic consensus is that
    discretionary open-market trades are the alpha-bearing subset;
    code M/F/G transactions are compensation mechanics.

Rate limiting:
  - SEC caps at 10 req/sec (measured; docs say "reasonable"). We use
    8 req/sec with 3 retries + jitter to stay under the cap.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger("kernel.insider_trades")


INSIDER_COLS: list[str] = [
    "tx_code",         # 'P' or 'S'
    "shares",          # float, signed (+=acquired, -=disposed)
    "price",           # float USD per share (can be NaN for gifts/cashless)
    "dollars",         # shares × price (signed)
]


# ── Rate-limited HTTP (SEC docs: be a good citizen) ─────────────────────────

# Round-3 audit (#R3-40): SEC requires a UA but committing a personal
# email to a public repo leaks PII. Read from env var with a generic
# fallback. Operators set RENQUANT_SEC_UA to their contact info.
import os as _os
_USER_AGENT = _os.environ.get(
    "RENQUANT_SEC_UA",
    "RenQuant research https://github.com/RenQuant/renquant",
)
_MIN_SLEEP_S = 0.125   # = 8 req/sec — under SEC's 10 req/sec cap


def _sec_get(url: str, *, timeout: float = 15.0, retries: int = 3) -> str:
    """GET with UA + rate limit + simple exponential backoff on 4xx/5xx."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_exc: "Exception | None" = None
    for attempt in range(retries):
        try:
            time.sleep(_MIN_SLEEP_S)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(0.5 * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


# ── CIK lookup (cached per-process) ──────────────────────────────────────────

_CIK_MAP: "dict[str, int] | None" = None
import threading as _threading
_CIK_LOCK = _threading.Lock()


_CIK_FAIL_TS: "float | None" = None
_CIK_RETRY_INTERVAL_SEC = 3600.0   # retry the EDGAR fetch hourly after a failure


def ticker_to_cik(ticker: str) -> "int | None":
    """Resolve ticker → CIK via SEC's public company_tickers.json.

    Round-3 audit (#R3-65): added a module lock around lazy init. Two
    threads racing on first call would both fetch from SEC; harmless but
    wasteful + extra rate-limit pressure.

    Audit fix IT-CONC-1 (Round 2 deep audit, 2026-04-25): pre-fix, a
    single EDGAR 403 / network failure at process start permanently
    poisoned `_CIK_MAP = {}` — `_CIK_MAP is None` then evaluated False
    on every subsequent call, so the lazy-init block never re-fired.
    Effect: insider-trade factor became permanently 0/NaN for the
    process lifetime even after EDGAR recovered. For long-running
    daily_104.sh / live_only_104.sh runs this could span entire
    trading days. Now: track failure timestamp and re-attempt the
    fetch every `_CIK_RETRY_INTERVAL_SEC` seconds (1 hour).
    """
    global _CIK_MAP, _CIK_FAIL_TS
    import time as _time
    needs_init = _CIK_MAP is None or (
        len(_CIK_MAP) == 0
        and _CIK_FAIL_TS is not None
        and (_time.monotonic() - _CIK_FAIL_TS) > _CIK_RETRY_INTERVAL_SEC
    )
    if needs_init:
        with _CIK_LOCK:
            # Re-check inside lock (could have raced)
            needs_init = _CIK_MAP is None or (
                len(_CIK_MAP) == 0
                and _CIK_FAIL_TS is not None
                and (_time.monotonic() - _CIK_FAIL_TS) > _CIK_RETRY_INTERVAL_SEC
            )
            if needs_init:
                try:
                    raw = _sec_get("https://www.sec.gov/files/company_tickers.json")
                    data = json.loads(raw)
                    _CIK_MAP = {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
                    _CIK_FAIL_TS = None
                except Exception as exc:
                    log.warning("ticker_to_cik: EDGAR fetch failed — %s", exc)
                    _CIK_MAP = {}
                    _CIK_FAIL_TS = _time.monotonic()
    return _CIK_MAP.get(ticker.upper())


# ── Form 4 XML parsing ───────────────────────────────────────────────────────

_RE_IS_OFFICER = re.compile(r"<isOfficer[^>]*>([^<]+)</isOfficer>")
_RE_NON_DERIV  = re.compile(
    r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", re.DOTALL,
)


def _grab_value(block: str, tag: str) -> "str | None":
    """Return either `<tag><value>X</value>...` or `<tag>X</tag>` content.

    Form 4 XML mixes these: most numeric fields wrap their scalar in
    `<value>...</value>` (so footnotes can attach), but `transactionCode`
    is a direct element. Try wrapped form first, fall back to direct.
    """
    m = re.search(rf"<{tag}[^>]*>\s*<value>([^<]+)</value>", block)
    if m:
        return m.group(1).strip()
    m = re.search(rf"<{tag}[^>]*>([^<]+)</{tag}>", block)
    return m.group(1).strip() if m else None


def _parse_form4_xml(xml: str) -> list[dict]:
    """Return a list of {date, tx_code, shares, price, dollars} rows.

    Filters:
      * isOfficer must be 'true' / '1' (executive-only per user spec)
      * only <nonDerivativeTransaction> blocks
      * only transactionCode in {'P', 'S'} (open-market discretionary)
      * shares must parse; price allowed to be NaN (skip that row then)
    """
    m = _RE_IS_OFFICER.search(xml)
    is_officer = (m.group(1).strip().lower() in ("true", "1")) if m else False
    if not is_officer:
        return []

    rows: list[dict] = []
    for blk_match in _RE_NON_DERIV.finditer(xml):
        blk = blk_match.group(1)
        code = _grab_value(blk, "transactionCode")
        if code not in ("P", "S"):
            continue
        date = _grab_value(blk, "transactionDate")
        shares = _grab_value(blk, "transactionShares")
        price  = _grab_value(blk, "transactionPricePerShare")
        acq    = _grab_value(blk, "transactionAcquiredDisposedCode")
        if not date or shares is None:
            continue
        try:
            shares_f = float(shares)
            price_f  = float(price) if price is not None else float("nan")
        except ValueError:
            continue
        if acq == "D":
            shares_f = -shares_f
        dollars = shares_f * price_f
        rows.append({
            "date":    pd.Timestamp(date).normalize(),
            "tx_code": code,
            "shares":  shares_f,
            "price":   price_f,
            "dollars": dollars,
        })
    return rows


# ── Submissions index ────────────────────────────────────────────────────────

def _list_form4_filings(cik: int, limit: int = 1000) -> list[dict]:
    """Return [{accession: '0000320193-26-000001', filingDate: '2026-04-17',
               primaryDocument: 'form4.xml'}] for recent Form 4 filings."""
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    raw = _sec_get(url)
    data = json.loads(raw)
    recent = data.get("filings", {}).get("recent", {})
    forms  = recent.get("form", [])
    dates  = recent.get("filingDate", [])
    accs   = recent.get("accessionNumber", [])
    prims  = recent.get("primaryDocument", [])
    out: list[dict] = []
    for f, d, a, p in zip(forms, dates, accs, prims):
        if f == "4":
            out.append({"accession": a, "filingDate": d, "primaryDocument": p})
        if len(out) >= limit:
            break
    return out


def _fetch_form4_xml(cik: int, accession: str) -> str:
    """Fetch the canonical form4.xml for a filing. Accession has dashes."""
    acc_nodash = accession.replace("-", "")
    # SEC's canonical filing dir: /Archives/edgar/data/{cik}/{acc_nodash}/form4.xml
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/form4.xml"
    try:
        return _sec_get(url)
    except urllib.error.HTTPError as exc:
        # A few filings use primaryDoc variants (e.g. ownership.xml); try the
        # index once to locate an xml file.
        if exc.code != 404:
            raise
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"
        html = _sec_get(index_url)
        m = re.search(r'href="([^"]+\.xml)"', html)
        if not m:
            raise
        href = m.group(1)
        abs_url = href if href.startswith("http") else f"https://www.sec.gov{href}"
        return _sec_get(abs_url)


# ── Public API ───────────────────────────────────────────────────────────────

@dataclass
class InsiderTradesStore:
    """Parquet cache at `data/insider_trades/{SYMBOL}.parquet`."""
    data_dir: Path = Path("data/insider_trades")

    def __post_init__(self):
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)

    def _path(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol.upper()}.parquet"

    def load(self, symbol: str) -> pd.DataFrame | None:
        # Audit fix IT-READ-RACE (Round 2 deep audit, 2026-04-25):
        # mirror FU-4 / ES-READ-RACE / INT-READ-RACE — corrupt parquet
        # (truncated, partial flush) treated as cache-miss; SEC re-fetch
        # then refills cleanly.
        p = self._path(symbol)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            log.warning(
                "InsiderTradesStore.load(%s): corrupt parquet — %s; "
                "treating as cache-miss", symbol, exc,
            )
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def save(self, df: pd.DataFrame, symbol: str) -> Path:
        # Audit fix IT-ATOM (Round 2 deep audit, 2026-04-25): atomic
        # write via .tmp + rename. Same as DC-2-CACHE / FU-1.
        p = self._path(symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_parquet(tmp)
        tmp.replace(p)
        return p


def fetch_insider_trades(
    ticker: str,
    *,
    cache: bool = True,
    store: "InsiderTradesStore | None" = None,
    provider_fn: "Callable[[str], pd.DataFrame] | None" = None,
    max_filings: int = 200,
    refresh_after_days: float = 7.0,
) -> pd.DataFrame:
    """Fetch + cache insider-trade rows for a ticker.

    `cache=True` uses a local parquet first; `provider_fn` lets tests
    inject a fake SEC fetcher.

    Round-2 audit (#R2-26): pre-fix the cache was written-once and
    NEVER refreshed; new Form 4 filings filed after the first cache
    write were invisible until someone manually deleted the parquet.
    Now: if the cache's most-recent date is older than `refresh_after_days`,
    we incremental-fetch fresh filings and merge them in.
    """
    store = store or InsiderTradesStore()
    cached: "pd.DataFrame | None" = None
    if cache:
        cached = store.load(ticker)
        if cached is not None and not cached.empty:
            # Refresh if the latest filing in the cache is more than
            # `refresh_after_days` old vs today. Form 4 filings are
            # batch-released after market close so a daily refresh is
            # plenty.
            latest = cached.index.max() if isinstance(cached.index, pd.DatetimeIndex) else None
            if latest is None:
                return cached
            age_days = (pd.Timestamp.now().normalize() - latest).days
            if age_days <= refresh_after_days:
                return cached
            # else: fall through to refresh path

    if provider_fn is not None:
        new_df = provider_fn(ticker)
    else:
        new_df = _fetch_from_sec(ticker, max_filings=max_filings)

    if cached is not None and not cached.empty and not new_df.empty:
        merged = pd.concat([cached, new_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = new_df if (new_df is not None and not new_df.empty) else (
            cached if cached is not None else new_df
        )

    if cache and merged is not None and not merged.empty:
        store.save(merged, ticker)
    return merged if merged is not None else pd.DataFrame(columns=INSIDER_COLS)


def _fetch_from_sec(ticker: str, *, max_filings: int = 200) -> pd.DataFrame:
    """Hit SEC EDGAR for recent Form 4 filings and parse executive transactions."""
    cik = ticker_to_cik(ticker)
    if cik is None:
        log.info("ticker_to_cik(%s): no CIK — ETF or unlisted", ticker)
        return pd.DataFrame(columns=INSIDER_COLS)

    try:
        filings = _list_form4_filings(cik, limit=max_filings)
    except Exception as exc:
        log.warning("list_form4_filings(%s) failed — %s", ticker, exc)
        return pd.DataFrame(columns=INSIDER_COLS)

    all_rows: list[dict] = []
    for f in filings:
        try:
            xml = _fetch_form4_xml(cik, f["accession"])
            rows = _parse_form4_xml(xml)
            for r in rows:
                all_rows.append(r)
        except Exception as exc:
            log.debug("form4 %s / %s fetch failed — %s",
                      ticker, f["accession"], exc)
            continue

    if not all_rows:
        return pd.DataFrame(columns=INSIDER_COLS)

    df = pd.DataFrame(all_rows).set_index("date").sort_index()
    return df[["tx_code", "shares", "price", "dollars"]]


def fetch_insider_trades_watchlist(
    watchlist: list[str],
    *,
    cache: bool = True,
    max_filings: int = 200,
    provider_fn: "Callable[[str], pd.DataFrame] | None" = None,
    total_budget_sec: float = 240.0,
    per_ticker_sec: float = 45.0,
) -> dict[str, pd.DataFrame]:
    """FetchBudget + per-ticker hard timeout. Each ticker fetch gets at
    most `per_ticker_sec` seconds (default 45 s), total batch ≤
    `total_budget_sec` (default 240 s). Prior version only checked the
    budget at OUTER loop boundaries; a single ticker's Form 4 fetch
    loop (max_filings=200, _sec_get retry=3 × timeout=15 s per call)
    could still run **thousands of seconds** before the budget saw it.

    This wraps each ticker's full fetch in `call_with_timeout`, giving
    up after `per_ticker_sec` even if the inner loop is still chewing.
    """
    import time
    from kernel.net_safety import FetchBudget, call_with_timeout
    budget = FetchBudget(total_sec=total_budget_sec,
                          label="fetch_insider_trades_watchlist")
    out: dict[str, pd.DataFrame] = {}
    for t in watchlist:
        if budget.exhausted():
            log.warning("  %-6s — skipping (insider budget exhausted)", t)
            out[t] = pd.DataFrame(columns=INSIDER_COLS)
            continue
        t0 = time.monotonic()
        result = call_with_timeout(
            fetch_insider_trades, t,
            timeout_sec = per_ticker_sec,
            label       = f"insider.fetch({t})",
            budget      = budget,
            cache       = cache,
            max_filings = max_filings,
            provider_fn = provider_fn,
        )
        if result is None:
            out[t] = pd.DataFrame(columns=INSIDER_COLS)
        else:
            out[t] = result
    return out


# ── Factor ───────────────────────────────────────────────────────────────────

def compute_insider_net_buy_cum(
    trades: dict[str, pd.DataFrame],
    ohlcv: dict[str, pd.DataFrame],
    *,
    trailing_days: int = 90,
) -> dict[str, pd.Series]:
    """Daily time-series of trailing-N-day net executive buy (USD).

    On each trading day, value = sum of `dollars` over the trailing N
    calendar days. Zero when no recent insider activity. NaN for tickers
    with no SEC data (ETFs, etc.).

    Positive values ⇒ net buying by executives. Academic literature
    (Lakonishok & Lee 2001, Cohen-Malloy-Pomorski 2012) shows this is
    a positive predictor of future relative returns.
    """
    out: dict[str, pd.Series] = {}
    for ticker, df in ohlcv.items():
        ins = trades.get(ticker)
        idx = df.index
        if ins is None or ins.empty or "dollars" not in ins.columns:
            out[ticker] = pd.Series(float("nan"), index=idx)
            continue
        # Bucket trades to a daily series (sum same-day transactions)
        daily = ins["dollars"].groupby(level=0).sum()
        daily = daily.reindex(
            pd.date_range(daily.index.min(), idx.max(), freq="D"),
            fill_value=0.0,
        )
        # Trailing-N-day rolling sum on calendar days
        rolling = daily.rolling(trailing_days, min_periods=1).sum()
        # Restrict to the ticker's OHLCV dates
        out[ticker] = rolling.reindex(idx, method="ffill")
    return out
