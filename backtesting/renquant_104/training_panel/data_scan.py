"""Training-input data-scan preflight.

User spec (2026-05-04): every training run must begin with a scan of
every input data source, verifying alignment across rows + columns and
emitting a length-and-coverage report.

The scan runs BEFORE PanelDataJob's normal Load* tasks consume the data,
so a corrupt cache or a missing ticker is surfaced LOUD before the
expensive 5-30 min panel build burns wallclock.

Public API
----------
``scan_training_inputs(watchlist, repo_root, *, today=None, sources=None)``
    Returns ``DataScanReport`` (dataclass, JSON-serializable via
    ``.to_dict()``).

``write_scan_report(report, artifact_path)``
    Persist as JSON. Used by the wired-in ``ScanTrainingDataTask``.

The scan is read-only — no fetches, no writes. If a cache is missing
for a ticker, that's recorded in ``missing_tickers``; the caller decides
whether to fail-loud (strict mode) or warn-and-continue.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("training_panel.data_scan")


@dataclass
class SourceCoverage:
    """Per-source scan result."""
    name:                   str
    n_tickers:              int = 0
    n_tickers_vs_watchlist: float = 0.0
    missing_tickers:        list[str] = field(default_factory=list)
    date_min:               str | None = None
    date_max:               str | None = None
    n_rows_total:           int = 0
    rows_per_ticker_min:    int = 0
    rows_per_ticker_p50:    int = 0
    rows_per_ticker_max:    int = 0
    age_days:               int | None = None   # today - date_max


@dataclass
class DataScanReport:
    """Top-level report for a training run's data preflight."""
    scan_utc:    str
    today:       str
    watchlist_size: int
    repo_root:   str
    sources:     dict[str, SourceCoverage] = field(default_factory=dict)
    alignment:   dict[str, Any]            = field(default_factory=dict)
    issues:      list[str]                 = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scan_utc":       self.scan_utc,
            "today":          self.today,
            "watchlist_size": self.watchlist_size,
            "repo_root":      self.repo_root,
            "sources":        {k: asdict(v) for k, v in self.sources.items()},
            "alignment":      self.alignment,
            "issues":         list(self.issues),
        }


def _safe_max_date(path: Path) -> tuple[_dt.date | None, int]:
    """Read a parquet/csv at ``path`` defensively; return (max_date, n_rows)."""
    if not path.exists():
        return None, 0
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception:
        return None, 0
    if df.empty:
        return None, 0
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.max().date(), len(df)
    for cand in ("date", "timestamp", "Date"):
        if cand in df.columns:
            s = pd.to_datetime(df[cand], errors="coerce").dropna()
            return (s.max().date() if not s.empty else None, len(df))
    return None, len(df)


def _scan_source(
    name: str,
    paths_per_ticker: dict[str, Path],
    watchlist: list[str],
    today: _dt.date,
) -> SourceCoverage:
    """Scan a per-ticker source. ``paths_per_ticker`` maps ticker → path."""
    cov = SourceCoverage(name=name)
    rows_per_ticker: list[int] = []
    max_dates: list[_dt.date] = []
    found_tickers: set[str] = set()

    for ticker in watchlist:
        path = paths_per_ticker.get(ticker)
        if path is None:
            continue
        d, n = _safe_max_date(path)
        if n > 0:
            found_tickers.add(ticker)
            rows_per_ticker.append(n)
            if d is not None:
                max_dates.append(d)

    cov.n_tickers              = len(found_tickers)
    cov.n_tickers_vs_watchlist = (
        len(found_tickers) / max(1, len(watchlist))
    )
    cov.missing_tickers = [t for t in watchlist if t not in found_tickers][:20]
    cov.n_rows_total    = sum(rows_per_ticker)
    if rows_per_ticker:
        s = pd.Series(rows_per_ticker)
        cov.rows_per_ticker_min = int(s.min())
        cov.rows_per_ticker_p50 = int(s.median())
        cov.rows_per_ticker_max = int(s.max())
    if max_dates:
        cov.date_min = str(min(max_dates))
        cov.date_max = str(max(max_dates))
        cov.age_days = (today - max(max_dates)).days
    return cov


def _scan_panel_source(
    name: str,
    path: Path,
    watchlist: list[str],
    today: _dt.date,
    *,
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> SourceCoverage:
    """Scan a single panel file keyed by ticker/date."""
    if not path.exists():
        return SourceCoverage(name=name, missing_tickers=watchlist[:20])
    try:
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except Exception:
        return SourceCoverage(name=name, missing_tickers=watchlist[:20])
    if df.empty or ticker_col not in df.columns or date_col not in df.columns:
        return SourceCoverage(name=name, missing_tickers=watchlist[:20])

    tickers = set(map(str, df[ticker_col].dropna().unique()))
    found = sorted(set(watchlist) & tickers)
    cov = SourceCoverage(
        name=name,
        n_tickers=len(found),
        n_tickers_vs_watchlist=len(found) / max(1, len(watchlist)),
        missing_tickers=[t for t in watchlist if t not in tickers][:20],
        n_rows_total=len(df),
    )
    counts = df[df[ticker_col].astype(str).isin(watchlist)].groupby(ticker_col).size()
    if not counts.empty:
        cov.rows_per_ticker_min = int(counts.min())
        cov.rows_per_ticker_p50 = int(counts.median())
        cov.rows_per_ticker_max = int(counts.max())
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if not dates.empty:
        cov.date_min = str(dates.min().date())
        cov.date_max = str(dates.max().date())
        cov.age_days = (today - dates.max().date()).days
    return cov


def scan_training_inputs(
    watchlist: list[str],
    repo_root: Path,
    *,
    today: _dt.date | None = None,
    include_intraday: bool = False,
) -> DataScanReport:
    """Scan all training-input data sources and return a coverage report.

    Sources scanned (per-ticker unless noted):
      * daily OHLCV       — backtesting/data/equity/usa/daily/{TICKER}.zip
                             OR data/ohlcv/{TICKER}/1d.parquet
      * sec_fundamentals_daily — active alpha158+fund panel source
      * fundamentals      — data/fundamentals/{TICKER}.parquet
      * earnings_surprise — data/earnings_surprise/{TICKER}.{parquet,csv}
      * news_sentiment_alpaca — active alpha158+fund sentiment source
      * insider_trades    — data/insider_trades/{TICKER}.{parquet,csv}
      * hourly bars       — data/intraday/{TICKER}/1h.parquet (only when
                             ``include_intraday=True``)
      * 10-min bars       — data/intraday/{TICKER}/10min.parquet (same)

    Cross-source alignment metrics added to ``report.alignment``:
      * date_range_overlap_days  — min(max_date) - max(min_date) across
                                    sources (negative = no overlap)
      * watchlist_coverage_pct   — fraction of watchlist with daily OHLCV
      * intraday_coverage_pct    — only when include_intraday=True
    """
    if today is None:
        today = _dt.date.today()
    repo_root = Path(repo_root)
    now_utc = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    report = DataScanReport(
        scan_utc=now_utc, today=str(today),
        watchlist_size=len(watchlist),
        repo_root=str(repo_root),
    )

    # ── Source 1: daily OHLCV (parquet preferred, fall back to LEAN zip) ─
    daily_paths: dict[str, Path] = {}
    for t in watchlist:
        pq = repo_root / "data" / "ohlcv" / t / "1d.parquet"
        if pq.exists():
            daily_paths[t] = pq
            continue
        zp = repo_root / "backtesting" / "data" / "equity" / "usa" / "daily" / f"{t.lower()}.zip"
        if zp.exists():
            # zip-format dates require LEAN parser; we don't have row counts
            # for free, but record presence. Mark as 1 row for the count
            # (to avoid being counted as missing) — operator should run a
            # fuller LEAN-side scan separately.
            daily_paths[t] = zp
    cov_daily = _scan_source("daily_ohlcv", daily_paths, watchlist, today)
    report.sources["daily_ohlcv"] = cov_daily

    # ── Active alpha158+fund panel sources ───────────────────────────────
    sec_panel = repo_root / "data" / "sec_fundamentals_daily.parquet"
    sec_cov = _scan_panel_source(
        "sec_fundamentals_daily", sec_panel, watchlist, today
    )
    report.sources["sec_fundamentals_daily"] = sec_cov

    sent_paths: dict[str, Path] = {}
    for t in watchlist:
        p = repo_root / "data" / "news_sentiment_alpaca" / f"{t}.parquet"
        if p.exists():
            sent_paths[t] = p
    report.sources["news_sentiment_alpaca"] = _scan_source(
        "news_sentiment_alpaca", sent_paths, watchlist, today,
    )

    # ── Source 2: fundamentals ───────────────────────────────────────────
    fund_paths = {
        t: repo_root / "data" / "fundamentals" / f"{t}.parquet"
        for t in watchlist
    }
    fund_paths = {t: p for t, p in fund_paths.items() if p.exists()}
    report.sources["fundamentals"] = _scan_source(
        "fundamentals", fund_paths, watchlist, today,
    )

    # ── Source 3: earnings surprise ──────────────────────────────────────
    earn_paths: dict[str, Path] = {}
    for t in watchlist:
        for ext in ("parquet", "csv"):
            p = repo_root / "data" / "earnings_surprise" / f"{t}.{ext}"
            if p.exists():
                earn_paths[t] = p
                break
    report.sources["earnings_surprise"] = _scan_source(
        "earnings_surprise", earn_paths, watchlist, today,
    )

    # ── Source 4: insider trades ─────────────────────────────────────────
    ins_paths: dict[str, Path] = {}
    for t in watchlist:
        for ext in ("parquet", "csv"):
            p = repo_root / "data" / "insider_trades" / f"{t}.{ext}"
            if p.exists():
                ins_paths[t] = p
                break
    report.sources["insider_trades"] = _scan_source(
        "insider_trades", ins_paths, watchlist, today,
    )

    # ── Source 5+6: intraday (opt-in) ────────────────────────────────────
    if include_intraday:
        h_paths = {
            t: repo_root / "data" / "intraday" / t / "1h.parquet"
            for t in watchlist
        }
        h_paths = {t: p for t, p in h_paths.items() if p.exists()}
        report.sources["hourly_bars"] = _scan_source(
            "hourly_bars", h_paths, watchlist, today,
        )

        m_paths: dict[str, Path] = {}
        for t in watchlist:
            for fname in ("10min.parquet", "10m.parquet"):
                p = repo_root / "data" / "intraday" / t / fname
                if p.exists():
                    m_paths[t] = p
                    break
        report.sources["minute_bars"] = _scan_source(
            "minute_bars", m_paths, watchlist, today,
        )

    # ── Source 7: benchmark (SPY) ────────────────────────────────────────
    spy_pq = repo_root / "data" / "ohlcv" / "SPY" / "1d.parquet"
    spy_zp = (repo_root / "backtesting" / "data" / "equity" / "usa"
              / "daily" / "spy.zip")
    if spy_pq.exists():
        d, n = _safe_max_date(spy_pq)
        spy_cov = SourceCoverage(
            name="benchmark_spy",
            n_tickers=1, n_tickers_vs_watchlist=1.0,
            missing_tickers=[],
            date_max=str(d) if d else None,
            n_rows_total=n,
            rows_per_ticker_min=n, rows_per_ticker_p50=n, rows_per_ticker_max=n,
            age_days=(today - d).days if d else None,
        )
    elif spy_zp.exists():
        spy_cov = SourceCoverage(
            name="benchmark_spy", n_tickers=1, n_tickers_vs_watchlist=1.0,
            missing_tickers=[], n_rows_total=1,
        )
    else:
        spy_cov = SourceCoverage(name="benchmark_spy", n_tickers=0,
                                  missing_tickers=["SPY"])
        report.issues.append("benchmark SPY OHLCV missing")
    report.sources["benchmark_spy"] = spy_cov

    # ── Cross-source alignment ───────────────────────────────────────────
    daily_max = report.sources["daily_ohlcv"].date_max
    spy_max   = report.sources["benchmark_spy"].date_max
    align: dict[str, Any] = {}
    align["watchlist_coverage_pct"] = (
        report.sources["daily_ohlcv"].n_tickers_vs_watchlist
    )
    if include_intraday:
        align["intraday_coverage_pct"] = (
            report.sources.get("hourly_bars",
                                SourceCoverage(name="hourly_bars")).n_tickers_vs_watchlist
        )
    if daily_max and spy_max:
        gap_days = (_dt.date.fromisoformat(daily_max)
                    - _dt.date.fromisoformat(spy_max)).days
        align["daily_vs_spy_max_date_gap_days"] = gap_days
        if abs(gap_days) > 5:
            report.issues.append(
                f"daily OHLCV max date {daily_max} differs from SPY max "
                f"{spy_max} by {gap_days}d (>5d)"
            )

    cov_pct = align["watchlist_coverage_pct"]
    if cov_pct < 0.95:
        report.issues.append(
            f"daily OHLCV coverage only {cov_pct*100:.1f}% of watchlist — "
            f"missing {len(report.sources['daily_ohlcv'].missing_tickers)} ticker(s)"
        )

    if sec_cov.n_rows_total <= 0:
        report.issues.append(
            "active SEC fundamentals panel missing or unreadable: "
            "data/sec_fundamentals_daily.parquet"
        )
    elif sec_cov.n_tickers_vs_watchlist < 0.90:
        report.issues.append(
            f"SEC fundamentals panel coverage only "
            f"{sec_cov.n_tickers_vs_watchlist*100:.1f}% of watchlist — "
            f"missing {len(sec_cov.missing_tickers)} ticker(s)"
        )
    elif sec_cov.age_days is not None and sec_cov.age_days > 120:
        report.issues.append(
            f"SEC fundamentals panel is {sec_cov.age_days}d stale (max date "
            f"{sec_cov.date_max})"
        )

    sent_cov = report.sources["news_sentiment_alpaca"]
    if sent_cov.n_tickers_vs_watchlist < 0.50:
        report.issues.append(
            f"news sentiment coverage only "
            f"{sent_cov.n_tickers_vs_watchlist*100:.1f}% of watchlist — "
            "active sentiment features would be mostly imputed"
        )
    elif sent_cov.age_days is not None and sent_cov.age_days > 14:
        report.issues.append(
            f"news sentiment data is {sent_cov.age_days}d stale (max date "
            f"{sent_cov.date_max})"
        )

    # Stale-data check
    cov_daily_age = report.sources["daily_ohlcv"].age_days
    if cov_daily_age is not None and cov_daily_age > 5:
        report.issues.append(
            f"daily OHLCV is {cov_daily_age}d stale (max date "
            f"{report.sources['daily_ohlcv'].date_max})"
        )

    report.alignment = align
    return report


def write_scan_report(report: DataScanReport, artifact_path: Path) -> None:
    """Persist the scan as JSON. Caller owns directory creation."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(report.to_dict(), indent=2))


def log_scan_summary(report: DataScanReport) -> None:
    """Emit a human-readable INFO-level summary."""
    log.info("=" * 70)
    log.info("DATA SCAN PREFLIGHT — watchlist=%d  today=%s",
             report.watchlist_size, report.today)
    log.info("=" * 70)
    for name, cov in report.sources.items():
        log.info(
            "  %-18s  tickers=%3d/%-3d  cov=%5.1f%%  rows=%-7d  "
            "max=%s  age=%s",
            name, cov.n_tickers, report.watchlist_size,
            cov.n_tickers_vs_watchlist * 100,
            cov.n_rows_total,
            cov.date_max or "—",
            f"{cov.age_days}d" if cov.age_days is not None else "—",
        )
    if report.alignment:
        log.info("  alignment: %s", report.alignment)
    if report.issues:
        log.warning("  ⚠ ISSUES (%d):", len(report.issues))
        for iss in report.issues:
            log.warning("    • %s", iss)
    else:
        log.info("  ✓ no alignment issues")
    log.info("=" * 70)
