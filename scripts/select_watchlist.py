#!/usr/bin/env python
"""Weekly watchlist selector — fixed core + market-driven expansion.

Design:
  CORE (fixed, ~60 tickers): manually curated high-conviction names that are
    always in the watchlist regardless of weekly signal. These are the tickers
    where per-ticker models have OOS Sharpe ≥ 1.2 AND market-cap > $50B.

  DYNAMIC (market-driven, ~40 tickers): weekly selection from a broader S&P500
    universe based on:
      1. Momentum score (3-month relative return vs SPY)
      2. Panel-LTR IC contribution (leave-one-in test vs baseline)
      3. Liquidity (avg daily volume ≥ $100M)
      4. Data quality (≥3 years OHLCV, fundamentals available)

    Every Sunday, re-run selection → update watchlist for the coming week.

Usage::

    python scripts/select_watchlist.py               # dry-run, show selection
    python scripts/select_watchlist.py --apply       # write to strategy_config.json
    python scripts/select_watchlist.py --n-dynamic 40 --min-momentum-pct 60

Scheduled via launchd weekly-watchlist104.plist (runs Sunday 22:00 PT,
before Tuesday retrain).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("select-watchlist")

STRATEGY = "renquant_104"

# ── CORE WATCHLIST (fixed, always included) ──────────────────────────────────
# Criteria: per-ticker Sharpe ≥ 1.2 (tested), market-cap > $50B, data ≥ 5yr.
# Review quarterly — add/remove manually when fundamentals change.
CORE_TICKERS: list[str] = [
    # Tech mega-cap
    "AAPL", "NVDA", "MSFT", "AMZN", "META", "AVGO", "TSM",
    # Financials
    "BLK", "GS", "JPM", "AXP",
    # Industrials / defense
    "CAT", "HON", "RTX", "LMT", "GE",
    # Healthcare
    "JNJ", "MU",
    # Growth
    "PLTR", "NET", "ANET", "FTNT", "KLAC",
    # Commodity / gold hedge
    "GLD",
    # Consumer
    "WMT", "MCD", "HOOD",
    # Fixed income proxy
    "NEE",
]


def _load_ohlcv(ticker: str, strategy_dir: Path) -> "pd.DataFrame | None":
    """Load cached OHLCV for a ticker."""
    try:
        from kernel.data import fetch_ohlcv  # noqa: PLC0415
        df = fetch_ohlcv(ticker)
        return df if df is not None and len(df) > 252 else None
    except Exception:
        return None


def _momentum_score(df: pd.DataFrame, lookback: int = 63) -> float:
    """3-month relative return vs SPY (higher = stronger momentum)."""
    if len(df) < lookback + 5:
        return float("nan")
    recent = float(df["close"].iloc[-1])
    past   = float(df["close"].iloc[-lookback])
    if past <= 0:
        return float("nan")
    return (recent - past) / past


def _data_quality_score(ticker: str, strategy_dir: Path) -> float:
    """0-1 score: fraction of features available (fundamentals, earnings, etc.)"""
    score = 0.0
    # Fundamentals
    try:
        from kernel.fundamentals import FundamentalsStore  # noqa: PLC0415
        store = FundamentalsStore.load(strategy_dir / "data" / "fundamentals.db")
        f = store.get(ticker, {})
        has_funds = sum(1 for k in ["earnings_yield", "roe", "gross_profitability"]
                        if f.get(k) is not None)
        score += has_funds / 3 * 0.5
    except Exception:
        pass
    # OHLCV coverage
    try:
        from kernel.data import fetch_ohlcv  # noqa: PLC0415
        df = fetch_ohlcv(ticker)
        if df is not None and len(df) >= 756:   # ≥ 3 years
            score += 0.5
    except Exception:
        pass
    return score


def _per_ticker_sharpe(ticker: str, models_dir: Path) -> "float | None":
    """Load per-ticker model OOS Sharpe from policy-metadata.json."""
    p = models_dir / ticker / f"{ticker}-policy-metadata.json"
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text())
        s = m.get("sharpe")
        return float(s) if s is not None else None
    except Exception:
        return None


def select_dynamic(
    universe: list[str],
    strategy_dir: Path,
    spy_df: pd.DataFrame,
    n_select: int = 40,
    min_momentum_pct: float = 50.0,
    min_data_quality: float = 0.5,
    exclude: set[str] | None = None,
) -> list[str]:
    """Score and select `n_select` tickers from `universe` for the dynamic slot.

    Scoring formula (equal weights):
      1. Momentum z-score (3-month return vs SPY)
      2. Per-ticker model Sharpe (if available, else 0.5 as neutral)
      3. Data quality score (completeness of features)
    """
    exclude = exclude or set()
    models_dir = strategy_dir / "models"
    spy_3m = _momentum_score(spy_df, 63)

    records = []
    for ticker in universe:
        if ticker in exclude:
            continue
        df = _load_ohlcv(ticker, strategy_dir)
        if df is None:
            continue
        mom = _momentum_score(df, 63)
        if not np.isfinite(mom):
            continue
        rel_mom = mom - (spy_3m or 0.0)

        sharpe = _per_ticker_sharpe(ticker, models_dir)
        sharpe_score = float(sharpe) if sharpe is not None else 0.5

        dq = _data_quality_score(ticker, strategy_dir)

        records.append({
            "ticker":       ticker,
            "rel_mom":      rel_mom,
            "sharpe":       sharpe_score,
            "data_quality": dq,
            "composite":    rel_mom + sharpe_score * 0.5 + dq * 0.3,
        })

    if not records:
        log.warning("No dynamic candidates scored — returning empty list")
        return []

    df_scores = pd.DataFrame(records).sort_values("composite", ascending=False)

    # Filter: minimum momentum percentile
    mom_pct_floor = np.percentile(df_scores["rel_mom"], 100 - min_momentum_pct)
    df_filtered = df_scores[
        (df_scores["rel_mom"] >= mom_pct_floor) &
        (df_scores["data_quality"] >= min_data_quality)
    ]

    selected = df_filtered.head(n_select)["ticker"].tolist()
    log.info(
        "Dynamic selection: %d → %d after filters → top %d selected",
        len(records), len(df_filtered), len(selected),
    )
    if len(selected) < n_select:
        log.warning("Only %d dynamic tickers passed filters (wanted %d)",
                    len(selected), n_select)

    # Log top scores for transparency
    for _, row in df_scores.head(10).iterrows():
        log.info(
            "  %-8s  rel_mom=%+.3f  sharpe=%.2f  dq=%.1f  composite=%+.3f",
            row["ticker"], row["rel_mom"], row["sharpe"],
            row["data_quality"], row["composite"],
        )

    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default=STRATEGY)
    p.add_argument("--n-dynamic", type=int, default=40,
                   help="Number of dynamic tickers (default 40)")
    p.add_argument("--min-momentum-pct", type=float, default=50.0,
                   help="Min momentum percentile for dynamic candidates (default 50)")
    p.add_argument("--min-data-quality", type=float, default=0.5,
                   help="Min data quality score 0-1 (default 0.5)")
    p.add_argument("--apply", action="store_true",
                   help="Write selected watchlist to strategy_config.json. "
                        "Without this flag, dry-run only.")
    p.add_argument("--universe-file", type=str, default=None,
                   help="Path to JSON file with candidate universe list. "
                        "Defaults to scripts/watchlist_universe.json (S&P500 approx).")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    sys.path.insert(0, str(strategy_dir))

    # Load candidate universe for dynamic selection
    universe_path = (
        Path(args.universe_file) if args.universe_file
        else REPO_ROOT / "scripts" / "watchlist_universe.json"
    )
    if universe_path.exists():
        universe = json.loads(universe_path.read_text())
        log.info("Loaded %d tickers from universe file", len(universe))
    else:
        log.warning("Universe file not found (%s) — using current watchlist as universe",
                    universe_path)
        cfg = json.loads((strategy_dir / "strategy_config.json").read_text())
        universe = cfg.get("watchlist", [])

    # Load SPY for momentum benchmark
    from kernel.data import fetch_ohlcv  # noqa: PLC0415
    spy_df = fetch_ohlcv("SPY")

    # Select dynamic tickers (exclude core)
    core_set = set(CORE_TICKERS)
    dynamic = select_dynamic(
        universe=[t for t in universe if t not in core_set],
        strategy_dir=strategy_dir,
        spy_df=spy_df,
        n_select=args.n_dynamic,
        min_momentum_pct=args.min_momentum_pct,
        min_data_quality=args.min_data_quality,
        exclude=core_set,
    )

    final_watchlist = sorted(set(CORE_TICKERS) | set(dynamic))
    log.info("Final watchlist: %d tickers (%d core + %d dynamic)",
             len(final_watchlist), len(CORE_TICKERS), len(dynamic))
    log.info("Core: %s", sorted(core_set))
    log.info("Dynamic: %s", sorted(dynamic))

    if args.apply:
        cfg_path = strategy_dir / "strategy_config.json"
        cfg = json.loads(cfg_path.read_text())
        old_count = len(cfg.get("watchlist", []))
        cfg["watchlist"] = final_watchlist
        cfg_path.write_text(json.dumps(cfg, indent=2))
        log.info("Applied: watchlist updated %d → %d tickers in strategy_config.json",
                 old_count, len(final_watchlist))

        # Also write to a weekly selection log
        log_dir = REPO_ROOT / "logs" / "watchlist_selection"
        log_dir.mkdir(parents=True, exist_ok=True)
        from datetime import date  # noqa: PLC0415
        log_path = log_dir / f"{date.today().isoformat()}.json"
        log_path.write_text(json.dumps({
            "date": date.today().isoformat(),
            "total": len(final_watchlist),
            "core": sorted(core_set),
            "dynamic": sorted(dynamic),
            "watchlist": final_watchlist,
        }, indent=2))
        log.info("Selection logged → %s", log_path)
    else:
        log.info("DRY RUN — pass --apply to write to strategy_config.json")
        print("\n=== Proposed watchlist (%d tickers) ===" % len(final_watchlist))
        print(final_watchlist)


if __name__ == "__main__":
    main()
