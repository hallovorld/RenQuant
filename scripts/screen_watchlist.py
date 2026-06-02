#!/usr/bin/env python
"""Weekly watchlist screen — suggest adds/drops based on 6-month performance.

Goal: keep the watchlist anchored to high-quality names. Drops
chronically under-performing tickers that tie up training cycles;
surfaces high-Sharpe non-watchlist S&P 500 names that could lift the
panel.

Runs Sunday 12:05 PT (after weekly_apy_check.py). Writes markdown
report to `logs/watchlist_screen/{YYYY-MM-DD}.md` + ntfy summary.

Metrics per ticker (6-month window):
  * Total return
  * Annualized Sharpe ratio (raw, rf=0)
  * Realized volatility (annualized)
  * Max drawdown
  * Correlation to SPY

Recommendations:
  DROP candidates: Sharpe < 0.0 AND total_return < SPY's return
  ADD candidates: S&P 500 names with Sharpe > watchlist_median + 0.5σ

Usage::

    python scripts/screen_watchlist.py
    python scripts/screen_watchlist.py --strategy renquant_104
    python scripts/screen_watchlist.py --lookback-days 180
    python scripts/screen_watchlist.py --top-add-candidates 10
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from subrepo_paths import resolve_subrepo_root

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("screen-watchlist")

TRADING_DAYS_PER_YEAR = 252
NTFY_TOPIC = "renquant"


def _try_subrepo_screen(argv: list[str] | None = None) -> bool:
    """Delegate to renquant-base-data when the subrepo runtime is available."""
    if os.environ.get("RQ_SCREEN_WATCHLIST_RUNNER", "multirepo") != "multirepo":
        return False
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strategy", default="renquant_104")
    parser.add_argument("--strategy-dir-root", default=None)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--top-add-candidates", type=int, default=10)
    parser.add_argument("--cache-root", default="data/ohlcv")
    parser.add_argument("--spy-symbol", default="SPY")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        return False

    root = Path(args.strategy_dir_root) if args.strategy_dir_root else REPO_ROOT
    cache_root = Path(args.cache_root)
    if cache_root.is_absolute():
        # The subrepo CLI owns the canonical RenQuant data-dir contract
        # (data/ohlcv). Keep legacy behavior for one-off custom cache roots.
        return False
    if cache_root.as_posix().rstrip("/") != "data/ohlcv":
        return False

    subrepo_root = resolve_subrepo_root(REPO_ROOT)
    strategy_config = subrepo_root / "renquant-strategy-104" / "configs" / "strategy_config.json"
    if not strategy_config.exists():
        strict = (
            os.environ.get("RENQUANT_STRICT_SUBREPO_PATHS") == "1"
            or os.environ.get("RQ_SCREEN_WATCHLIST_STRICT") == "1"
        )
        if strict:
            raise RuntimeError(
                "pinned renquant-strategy-104 strategy_config.json unavailable"
            )
        strategy_config = root / "backtesting" / args.strategy / "strategy_config.json"
    for rel in ("renquant-base-data/src", "renquant-common/src"):
        path = str(subrepo_root / rel)
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from renquant_base_data.watchlist_screen import main as subrepo_main
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("RQ_SCREEN_WATCHLIST_STRICT") == "1":
            raise RuntimeError(
                "renquant_base_data.watchlist_screen unavailable and "
                "RQ_SCREEN_WATCHLIST_STRICT=1"
            ) from exc
        log.warning("base-data watchlist screen unavailable; using umbrella implementation: %s", exc)
        return False

    subrepo_main([
        "--strategy-config", str(strategy_config),
        "--data-dir", str(root / "data"),
        "--output-dir", str(root / "logs" / "watchlist_screen"),
        "--lookback-days", str(args.lookback_days),
        "--top-add-candidates", str(args.top_add_candidates),
        "--spy-symbol", args.spy_symbol,
    ])
    return True


def _perf_stats(closes: "pd.Series") -> dict:
    """Compute 6-month perf stats from a close-price series."""
    import numpy as np  # noqa: PLC0415
    rets = closes.pct_change().dropna()
    if len(rets) < 20:
        return {}
    total_return = (closes.iloc[-1] / closes.iloc[0]) - 1
    ann_vol = rets.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (rets.mean() * TRADING_DAYS_PER_YEAR) / ann_vol if ann_vol > 0 else 0
    # Max drawdown
    cummax = closes.cummax()
    drawdown = (closes - cummax) / cummax
    max_dd = float(drawdown.min())
    return {
        "total_return": float(total_return),
        "sharpe":       float(sharpe),
        "ann_vol":      float(ann_vol),
        "max_dd":       float(max_dd),
        "final_price":  float(closes.iloc[-1]),
        "n_days":       len(rets),
    }


def _load_ticker_series(ticker: str, cache_root: Path, lookback_days: int):
    """Return (close_series, True) or (None, False) if parquet missing."""
    import pandas as pd  # noqa: PLC0415
    path = cache_root / ticker / "1d.parquet"
    if not path.exists():
        return None, False
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
    closes = df.loc[df.index >= cutoff, "close"]
    if len(closes) < 20:
        return None, False
    return closes, True


def _correlation_to_spy(closes: "pd.Series", spy_closes: "pd.Series") -> float:
    rets = closes.pct_change().dropna()
    spy  = spy_closes.pct_change().dropna()
    common = rets.index.intersection(spy.index)
    if len(common) < 20:
        return float("nan")
    return float(rets.loc[common].corr(spy.loc[common]))


def _notify(title: str, body: str) -> None:
    # Suppress notifications in test runs: RENQUANT_NO_NOTIFY=1
    # (tests invoke the script end-to-end; without this guard every
    # pytest run fires a live ntfy and spams the user's phone).
    import os
    if os.environ.get("RENQUANT_NO_NOTIFY") == "1":
        log.info("[ntfy suppressed by RENQUANT_NO_NOTIFY] %s: %s", title, body)
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=body.encode("utf-8"),
            headers={"Title": title},
        )
        urllib.request.urlopen(req, timeout=5).close()
    except Exception as exc:
        log.warning("ntfy failed: %s", exc)


def main() -> None:
    if _try_subrepo_screen(sys.argv[1:]):
        return

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--strategy-dir-root", default=None,
                   help="Root dir containing backtesting/{strategy}/. Defaults "
                        "to repo root; overridable for testing.")
    p.add_argument("--lookback-days", type=int, default=180,
                   help="Days of history to evaluate (default 180 ≈ 6 months)")
    p.add_argument("--top-add-candidates", type=int, default=10,
                   help="How many non-watchlist high-Sharpe names to surface")
    p.add_argument("--cache-root", default="data/ohlcv")
    p.add_argument("--spy-symbol", default="SPY")
    args = p.parse_args()

    root = Path(args.strategy_dir_root) if args.strategy_dir_root else REPO_ROOT
    strategy_dir = root / "backtesting" / args.strategy
    cache_root   = (Path(args.cache_root) if Path(args.cache_root).is_absolute()
                     else root / args.cache_root)

    # Load watchlist
    cfg_path = strategy_dir / "strategy_config.json"
    cfg = json.loads(cfg_path.read_text())
    watchlist = list(cfg.get("watchlist", []))
    defensive = set(cfg.get("defensive_tickers", []))

    log.info("Screening watchlist: %d tickers, lookback %dd",
             len(watchlist), args.lookback_days)

    # SPY baseline
    spy_closes, _ = _load_ticker_series(args.spy_symbol, cache_root,
                                         args.lookback_days)
    if spy_closes is None:
        log.error("SPY parquet missing — cannot compute relative metrics")
        sys.exit(1)
    spy_stats = _perf_stats(spy_closes)

    # Watchlist stats
    watchlist_stats: dict[str, dict] = {}
    for ticker in watchlist:
        closes, ok = _load_ticker_series(ticker, cache_root, args.lookback_days)
        if not ok:
            log.warning("  %s — parquet missing or too short", ticker)
            continue
        stats = _perf_stats(closes)
        stats["corr_spy"] = _correlation_to_spy(closes, spy_closes)
        stats["is_defensive"] = ticker in defensive
        watchlist_stats[ticker] = stats

    if not watchlist_stats:
        log.error("No valid watchlist stats computed — abort")
        sys.exit(1)

    # DROP recommendations: sharpe < 0 AND total_return < SPY
    spy_ret = spy_stats["total_return"]
    drops = [
        (t, s) for t, s in watchlist_stats.items()
        if s["sharpe"] < 0 and s["total_return"] < spy_ret
        and not s["is_defensive"]   # don't drop defensives on short-term perf
    ]
    drops.sort(key=lambda x: x[1]["sharpe"])

    # Candidate adds — scan non-watchlist tickers in cache_root
    watchlist_median_sharpe = sorted(s["sharpe"] for s in watchlist_stats.values())
    median_idx = len(watchlist_median_sharpe) // 2
    median_sharpe = watchlist_median_sharpe[median_idx]
    # Standard deviation of sharpe in watchlist
    mean_s = sum(watchlist_median_sharpe) / len(watchlist_median_sharpe)
    sigma_s = math.sqrt(
        sum((x - mean_s) ** 2 for x in watchlist_median_sharpe)
        / max(1, len(watchlist_median_sharpe) - 1)
    )
    add_threshold = median_sharpe + 0.5 * sigma_s

    add_candidates = []
    if cache_root.exists():
        for ticker_dir in cache_root.iterdir():
            if not ticker_dir.is_dir():
                continue
            sym = ticker_dir.name
            if sym in watchlist or sym == args.spy_symbol:
                continue
            closes, ok = _load_ticker_series(sym, cache_root, args.lookback_days)
            if not ok:
                continue
            stats = _perf_stats(closes)
            if stats.get("sharpe", -99) > add_threshold:
                stats["corr_spy"] = _correlation_to_spy(closes, spy_closes)
                add_candidates.append((sym, stats))
    add_candidates.sort(key=lambda x: -x[1]["sharpe"])
    add_candidates = add_candidates[: args.top_add_candidates]

    # Build markdown report
    date = datetime.date.today().isoformat()
    out_dir = root / "logs" / "watchlist_screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.md"

    lines: list[str] = []
    lines.append(f"# Watchlist screen — {date}")
    lines.append("")
    lines.append(f"**Strategy:** `{args.strategy}`  ")
    lines.append(f"**Lookback:** {args.lookback_days} days  ")
    lines.append(f"**SPY baseline:** return={spy_ret:+.2%}, sharpe={spy_stats['sharpe']:.2f}")
    lines.append("")

    # Watchlist table
    lines.append("## Watchlist — 6-month metrics")
    lines.append("")
    lines.append("| Ticker | Return | Sharpe | Vol (ann) | Max DD | ρ(SPY) | Note |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    sorted_wl = sorted(watchlist_stats.items(), key=lambda x: -x[1]["sharpe"])
    for ticker, s in sorted_wl:
        note = []
        if s["is_defensive"]: note.append("defensive")
        if s["sharpe"] < 0: note.append("⚠️ negative Sharpe")
        if s["total_return"] < spy_ret and not s["is_defensive"]:
            note.append("🔻 vs SPY")
        lines.append(
            f"| {ticker} | {s['total_return']:+.1%} | {s['sharpe']:.2f} | "
            f"{s['ann_vol']:.1%} | {s['max_dd']:.1%} | {s['corr_spy']:.2f} | "
            f"{', '.join(note) or '-'} |"
        )
    lines.append("")

    # Drops
    lines.append("## 🔻 Drop candidates")
    lines.append("")
    if drops:
        lines.append("Names with Sharpe < 0 AND total return < SPY (defensives excluded):")
        lines.append("")
        for ticker, s in drops:
            lines.append(f"- **{ticker}** — Sharpe={s['sharpe']:.2f}  "
                          f"return={s['total_return']:+.1%}  "
                          f"(SPY {spy_ret:+.1%})")
    else:
        lines.append("*None — all watchlist names pass the floor.*")
    lines.append("")

    # Adds
    lines.append(f"## 🚀 Add candidates (Sharpe > {add_threshold:.2f})")
    lines.append("")
    if add_candidates:
        lines.append("Non-watchlist names with above-median Sharpe "
                      f"(threshold = median + 0.5σ = {add_threshold:.2f}):")
        lines.append("")
        lines.append("| Ticker | Return | Sharpe | Vol | ρ(SPY) |")
        lines.append("|---|---:|---:|---:|---:|")
        for ticker, s in add_candidates:
            lines.append(
                f"| {ticker} | {s['total_return']:+.1%} | {s['sharpe']:.2f} | "
                f"{s['ann_vol']:.1%} | {s['corr_spy']:.2f} |"
            )
    else:
        lines.append("*No non-watchlist names above threshold. Note: only "
                      "names with existing parquet cache are evaluated. "
                      "Expand cache (via `python -c \"import common; "
                      "common.fetch_ohlcv('TICKER')\"`) to broaden.*")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Watchlist size: {len(watchlist_stats)} / {len(watchlist)}")
    lines.append(f"- Median Sharpe: {median_sharpe:.2f}")
    lines.append(f"- Drop suggestions: {len(drops)}")
    lines.append(f"- Add suggestions: {len(add_candidates)}")
    lines.append("")
    lines.append(f"_Generated by `scripts/screen_watchlist.py` on {date}._")

    out_path.write_text("\n".join(lines))
    log.info("Report written → %s", out_path)

    # ntfy summary
    body = (f"drops={len(drops)} adds={len(add_candidates)} "
            f"median_sharpe={median_sharpe:.2f} SPY_ret={spy_ret:+.1%}")
    _notify(f"RenQuant watchlist screen {date}", body)
    print(f"\n{body}")
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
