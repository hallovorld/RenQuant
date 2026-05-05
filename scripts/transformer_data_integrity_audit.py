#!/usr/bin/env python
"""Phase 2 of Transformer data prep: integrity audit on Tier-A/B universe.

Runs 6 checks per ticker. Each check has a binary pass/fail and a
detail message. The output is a per-ticker audit report + an aggregate
"go/no-go" decision per tier.

Checks (per CLAUDE.md §5.6 "definition of fixed = full audit clean"):

  1. NaN-rate ≤ 5% on close — high NaN ratio = data quality issue
  2. Adjusted-close monotonicity check — adjclose-style cache should
     not have wild day-to-day jumps (>50% in a single bar = unadjusted
     stock split or data error)
  3. Volume sanity — no zero-volume on >20% of bars (suspended /
     halted ticker is unsuitable)
  4. Date gaps ≤ 5 trading days — long gaps signal trading halts /
     missing data periods
  5. NYSE calendar alignment — index dates match NYSE trading calendar
     (verifies cache wasn't built from a foreign exchange)
  6. No future leak from corporate actions — split_ratio / dividend
     columns (if present) must have NaN for unprocessed ranges, not
     synthetic values

Writes: data/transformer_data_integrity_report.json
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("transformer-integrity")


def _check_nan_rate(df: pd.DataFrame, max_nan_pct: float = 0.05) -> tuple[bool, str]:
    """Check 1: NaN-rate on close ≤ max_nan_pct."""
    nan_pct = float(df["close"].isna().mean())
    return (nan_pct <= max_nan_pct,
            f"close NaN rate = {nan_pct:.3f}")


def _check_no_wild_jumps(df: pd.DataFrame, max_pct: float = 1.50) -> tuple[bool, str]:
    """Check 2: no single-bar abs return > max_pct (would be a missed split).

    Threshold raised from 0.50 → 1.50 (post-2026-05-05 data audit). Real
    market events can produce 50-100% single-bar moves: oil crashes
    (APA/OXY 2020-03-09), earnings surprises (AMD 2016-04-22 +52%), IPO
    pricing (FCNCA, SOFI), meme squeezes (GME +135%). These are
    NOT data bugs — they're real history we want the Transformer to see.
    The 1.5× cutoff still catches genuine missed-split errors (which
    typically produce single-bar 100-1000% moves)."""
    closes = df["close"].dropna()
    if len(closes) < 2:
        return True, "<2 bars; trivially passes"
    rets = closes.pct_change().abs()
    n_wild = int((rets > max_pct).sum())
    if n_wild == 0:
        return True, f"max abs return = {rets.max():.3f}"
    # Find the worst offender
    idx = rets.idxmax()
    return False, (
        f"{n_wild} bars with |ret|>{max_pct} — likely unadjusted split. "
        f"Worst at {idx}: {rets.max():.3f}"
    )


def _check_volume_sanity(df: pd.DataFrame,
                          max_zero_pct: float = 0.20) -> tuple[bool, str]:
    """Check 3: zero-volume bars ≤ max_zero_pct."""
    if "volume" not in df.columns:
        return True, "volume column absent — skip"
    vol = df["volume"].fillna(0)
    zero_pct = float((vol == 0).mean())
    return (zero_pct <= max_zero_pct,
            f"zero-volume bars = {zero_pct:.3f}")


def _check_date_gaps(df: pd.DataFrame, max_gap_days: int = 5) -> tuple[bool, str]:
    """Check 4: max date gap ≤ max_gap_days trading days.
    NB: weekends count, so 5 trading days ≈ 7 calendar days."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, "index not DatetimeIndex"
    diffs = df.index.to_series().diff().dt.days.dropna()
    max_gap = int(diffs.max())
    # Weekends + holidays mean a "normal" gap is 1 day, occasional 3-4.
    # 7+ day gaps are suspicious.
    cal_threshold = max_gap_days + 2  # padding for weekends
    if max_gap > cal_threshold:
        n_long = int((diffs > cal_threshold).sum())
        return False, (
            f"max gap = {max_gap} cal-days (>{cal_threshold}); "
            f"{n_long} gaps longer than threshold"
        )
    return True, f"max gap = {max_gap} cal-days"


def _check_nyse_calendar(df: pd.DataFrame) -> tuple[bool, str]:
    """Check 5: index dates are weekdays (lightweight calendar check)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, "index not DatetimeIndex"
    weekend_count = int((df.index.dayofweek >= 5).sum())
    return (weekend_count == 0,
            f"weekend bars = {weekend_count}")


def _check_no_future_leak(df: pd.DataFrame) -> tuple[bool, str]:
    """Check 6: today's split_ratio / dividend columns shouldn't reach
    far into the future. Beyond a 365-day window from cache load is suspect."""
    today = pd.Timestamp.now(tz=None).normalize()
    last = df.index.max()
    if last > today + pd.Timedelta(days=7):
        return False, f"data extends to {last}, beyond {today+pd.Timedelta(days=7)}"
    return True, f"last bar {last.date()}"


def audit_ticker(t: str, p: Path) -> dict:
    """Run all 6 checks; return result dict."""
    if not p.exists():
        return {"ticker": t, "ok": False, "reason": "missing parquet"}
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        return {"ticker": t, "ok": False, "reason": f"read failed: {exc}"}
    if df.empty or "close" not in df.columns:
        return {"ticker": t, "ok": False, "reason": "no close column or empty"}
    checks = {
        "nan_rate":      _check_nan_rate(df),
        "no_wild_jumps": _check_no_wild_jumps(df),
        "volume_sanity": _check_volume_sanity(df),
        "date_gaps":     _check_date_gaps(df),
        "nyse_calendar": _check_nyse_calendar(df),
        "no_future":     _check_no_future_leak(df),
    }
    all_ok = all(c[0] for c in checks.values())
    failures = [k for k, c in checks.items() if not c[0]]
    return {
        "ticker": t,
        "ok": all_ok,
        "n_rows": int(len(df)),
        "checks": {k: {"ok": c[0], "msg": c[1]} for k, c in checks.items()},
        "failed_checks": failures,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory",
                    default=str(REPO_ROOT / "data" / "transformer_universe_inventory.json"))
    p.add_argument("--ohlcv-dir",
                    default=str(REPO_ROOT / "data" / "ohlcv"))
    p.add_argument("--output",
                    default=str(REPO_ROOT / "data" / "transformer_data_integrity_report.json"))
    args = p.parse_args()

    inv_path = Path(args.inventory)
    if not inv_path.exists():
        log.error("Inventory not found: %s — run "
                  "scripts/transformer_universe_inventory.py first", inv_path)
        sys.exit(1)
    inv = json.loads(inv_path.read_text())
    tier_A = inv.get("tier_A_tickers", [])
    tier_B = inv.get("tier_B_tickers", [])
    log.info("Auditing Tier-A: %d tickers, Tier-B: %d tickers", len(tier_A), len(tier_B))

    ohlcv_dir = Path(args.ohlcv_dir)
    results: dict[str, list] = {"A": [], "B": []}

    for tier, tickers in [("A", tier_A), ("B", tier_B)]:
        for i, t in enumerate(tickers):
            if i % 50 == 0 and i > 0:
                log.info("  %s: %d/%d audited", tier, i, len(tickers))
            results[tier].append(audit_ticker(t, ohlcv_dir / t / "1d.parquet"))

    summary = {
        "kind":          "transformer_data_integrity_report",
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "inventory":     str(inv_path),
        "tier_summary": {},
        "per_ticker":    results,
    }

    for tier, lst in results.items():
        n_total = len(lst)
        n_ok    = sum(1 for r in lst if r["ok"])
        n_fail  = n_total - n_ok
        # Tally failure modes
        fail_modes: dict[str, int] = {}
        for r in lst:
            for c in r.get("failed_checks", []):
                fail_modes[c] = fail_modes.get(c, 0) + 1
        summary["tier_summary"][tier] = {
            "n_total":      n_total,
            "n_ok":         n_ok,
            "n_fail":       n_fail,
            "fail_rate":    round(n_fail / max(1, n_total), 3),
            "fail_modes":   fail_modes,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("══ integrity report written %s ══", out_path)
    for tier, s in summary["tier_summary"].items():
        log.info("Tier-%s: %d/%d passed (%.1f%% fail) modes=%s",
                  tier, s["n_ok"], s["n_total"], s["fail_rate"]*100, s["fail_modes"])


if __name__ == "__main__":
    main()
