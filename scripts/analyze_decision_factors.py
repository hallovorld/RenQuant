#!/usr/bin/env python
"""Plan AA — quantile IC, tier base-rate, and regime-conditional IC
from the decision-trace DB.

Inputs: `data/runs.db::candidate_scores` (what we *chose*) joined with
`ticker_forward_returns` (what *happened*). Assumes
`scripts/backfill_forward_returns.py` has run.

Prints four diagnostic tables:
  1. IC by rank_score quantile        — is the ranker ordering correctly?
  2. Tier base-rate realization       — does each `tiered_thresholds` cut
                                         correspond to a real edge?
  3. Regime-conditional IC            — does the ranker work in every regime?
  4. Block reason × fwd_Nd outcome    — did the blocks save us money?

Usage::

    python scripts/analyze_decision_factors.py
    python scripts/analyze_decision_factors.py --horizon 5   # use fwd_5d instead
    python scripts/analyze_decision_factors.py --quantiles 10
    python scripts/analyze_decision_factors.py --since 2026-01-01
    python scripts/analyze_decision_factors.py --source sim
"""
from __future__ import annotations

import argparse
import datetime
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


SOURCE_DB = {
    # Live decisions are broker-scoped. The old data/runs.db default is a
    # generic/local scratch DB and is often empty, which made diagnostics
    # falsely report "no rows" for real Alpaca decision traces.
    "live": "data/runs.alpaca.db",
    "alpaca": "data/runs.alpaca.db",
    "alpaca-shadow": "data/runs.alpaca_shadow.db",
    "paper": "data/runs.paper.db",
    "local": "data/runs.db",
    "sim": "data/sim_runs.db",
}


def resolve_db_path(source: str, override: str | None = None) -> Path:
    if override is not None:
        return REPO_ROOT / override
    rel = SOURCE_DB.get(source)
    if rel is None:
        raise ValueError(f"unknown source: {source}")
    return REPO_ROOT / rel


def _fetch_joined(
    conn: sqlite3.Connection,
    horizon: int,
    since: datetime.date | None,
) -> "pd.DataFrame":
    """Return the candidate × forward-return join as a DataFrame."""
    import pandas as pd  # noqa: PLC0415
    q = f"""
        SELECT ps.run_date,
               ps.run_id,
               ps.regime,
               ps.confidence,
               cs.ticker,
               cs.role,
               cs.rank_score,
               cs.panel_score,
               cs.mu,
               cs.sigma,
               cs.selected,
               cs.blocked_by,
               tfr.fwd_{horizon}d AS fwd,
               tfr.close_price
          FROM candidate_scores cs
          JOIN pipeline_runs ps   ON ps.run_id = cs.run_id
          JOIN ticker_forward_returns tfr
            ON tfr.as_of_date = ps.run_date AND tfr.ticker = cs.ticker
         WHERE cs.role = 'candidate'
           AND tfr.fwd_{horizon}d IS NOT NULL
    """
    params: list = []
    if since is not None:
        q += " AND ps.run_date >= ?"
        params.append(since.isoformat())
    df = pd.read_sql_query(q, conn, params=params)
    df["run_date"] = pd.to_datetime(df["run_date"])
    return df


def _print_quantile_ic(df: "pd.DataFrame", q: int) -> None:
    """Quantile table: rank_score bucket → mean fwd, hit rate, n."""
    import pandas as pd  # noqa: PLC0415
    if df.empty:
        print("  (no rows)")
        return
    # Quantile-bin per-bar to avoid date-aggregation bias
    df = df.copy()
    df["bucket"] = (df.groupby("run_date")["rank_score"]
                      .transform(lambda s: pd.qcut(s, q=q, labels=False, duplicates="drop")))
    grouped = df.groupby("bucket")["fwd"].agg(
        mean_fwd = "mean",
        median_fwd = "median",
        hit_rate = lambda s: float((s > 0).mean()),
        n = "count",
    ).round(4)
    print(grouped.to_string())


def _print_tier_realization(df: "pd.DataFrame", cuts: list[float]) -> None:
    """For each tier cut, show (n, base_rate P(fwd>0), mean fwd_N) of candidates
    AT OR ABOVE the cut. This is the empirical equivalent of the `base_rate=0.273`
    calibrator anchor used by the A-gate.
    """
    for cut in cuts:
        slice_ = df[df["rank_score"] >= cut]
        n = len(slice_)
        if n == 0:
            print(f"  rank_score ≥ {cut:.3f}  →  n=0  (skipped)")
            continue
        base = float((slice_["fwd"] > 0).mean())
        mean_fwd = float(slice_["fwd"].mean())
        print(f"  rank_score ≥ {cut:.3f}  →  n={n:5d}  P(fwd>0)={base:.3f}  "
              f"mean fwd={mean_fwd:+.4%}")


def _print_regime_ic(df: "pd.DataFrame") -> None:
    """Spearman corr(rank_score, fwd) per-regime."""
    import pandas as pd  # noqa: PLC0415
    from scipy.stats import spearmanr  # type: ignore  # noqa: PLC0415
    rows = []
    for regime, sub in df.groupby("regime"):
        rho, _ = spearmanr(sub["rank_score"], sub["fwd"], nan_policy="omit")
        n = len(sub)
        rows.append((regime or "(null)", n, rho))
    rows.sort(key=lambda r: -r[1])
    print(f"  {'regime':20} {'n':>8} {'spearman':>10}")
    for regime, n, rho in rows:
        print(f"  {regime:20} {n:>8} {rho:>+10.4f}")


def _print_selected_bucket(df: "pd.DataFrame", cuts: list[float]) -> None:
    """Realized hit rate + mean fwd for candidates that were actually SELECTED,
    bucketed by rank_score at select time. Answers: is each tier pulling its
    weight in realized outcomes, or are lower-score tiers dragging hit rate?"""
    sel = df[df["selected"] == 1]
    if sel.empty:
        print("  (no selected candidates)")
        return
    # Add an implicit 1.0 upper bound so the last bucket is closed
    edges = sorted(set([0.0] + list(cuts) + [1.0001]))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = sel[(sel["rank_score"] >= lo) & (sel["rank_score"] < hi)]
        if s.empty:
            continue
        hit = float((s["fwd"] > 0).mean())
        mean_fwd = float(s["fwd"].mean())
        rows.append((lo, hi, len(s), hit, mean_fwd))
    print(f"  {'rank_score':>18} {'n':>6} {'hit':>7} {'mean fwd':>10}")
    for lo, hi, n, hit, mean_fwd in rows:
        print(f"  [{lo:>5.3f}, {hi:>5.3f})  {n:>6} {hit:>7.3f} {mean_fwd:>+10.4%}")


def _print_slots_filled(df: "pd.DataFrame") -> None:
    """How many selected candidates per run (bar) — answers whether tier 2/3
    ever actually get used. Groups by run_id (not run_date — multiple runs
    can fire on one date during A/B sweeps; each is a distinct decision bar)."""
    import pandas as pd  # noqa: PLC0415
    sel = df[df["selected"] == 1]
    if sel.empty:
        print("  (no selected candidates)")
        return
    per_run = sel.groupby("run_id").size()
    total_runs = per_run.shape[0]
    print(f"  runs with ≥1 selected: {total_runs:,}    "
          f"max selections on one run: {per_run.max()}    "
          f"median: {int(per_run.median())}")
    print(f"  {'count':>8} {'runs':>8} {'fraction':>10}")
    dist = per_run.value_counts().sort_index()
    for n, cnt in dist.items():
        frac = cnt / total_runs
        print(f"  {n:>8} {cnt:>8} {frac:>10.1%}")


def _print_block_outcomes(df: "pd.DataFrame") -> None:
    """Did each block reason save us money? fwd_N on blocked-but-unselected
    candidates, grouped by reason."""
    rows = []
    # selected=1 is an executed outcome, not a block reason. Older DB rows can
    # contain stale blocked_by values from pre-selection Kelly diagnostics; keep
    # attribution semantics clean even when reading those historical rows.
    reason_series = df["blocked_by"].fillna("(passed/unselected)")
    if "selected" in df.columns:
        reason_series = reason_series.mask(df["selected"] == 1, "(selected)")
    for reason, sub in df.groupby(reason_series):
        n = len(sub)
        base = float((sub["fwd"] > 0).mean())
        mean_fwd = float(sub["fwd"].mean())
        rows.append((reason, n, base, mean_fwd))
    rows.sort(key=lambda r: -r[1])
    print(f"  {'reason':28} {'n':>6} {'P(fwd>0)':>10} {'mean fwd':>10}")
    for reason, n, base, mean_fwd in rows:
        print(f"  {reason:28} {n:>6} {base:>10.3f} {mean_fwd:>+10.4%}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=sorted(SOURCE_DB), default="live",
                   help="Read a decision-trace DB. Default live maps to "
                        "data/runs.alpaca.db; local maps to data/runs.db; "
                        "sim maps to data/sim_runs.db.")
    p.add_argument("--db", default=None,
                   help="Override path; bypasses --source mapping.")
    p.add_argument("--horizon", type=int, default=60, choices=[1, 5, 10, 20, 60])
    p.add_argument("--quantiles", type=int, default=5,
                   help="Number of quantile buckets for rank_score IC.")
    p.add_argument("--since", type=lambda s: datetime.date.fromisoformat(s), default=None)
    p.add_argument("--tier-cuts", nargs="+", type=float,
                   default=[0.10, 0.20, 0.27, 0.35, 0.45, 0.60],
                   help="rank_score cut points for tier realization table.")
    args = p.parse_args()

    db_path = resolve_db_path(args.source, args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    # Route through get_connection so ensure_schema runs (creates
    # ticker_forward_returns if the DB was frozen before Plan AA).
    sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))
    from kernel.persistence import get_connection  # noqa: PLC0415
    conn = get_connection({"persistence": {"enabled": True, "db_path": str(db_path)}})
    if conn is None:
        print(f"ERROR: failed to open {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n== Decision-factor analysis ({db_path.name}, horizon=fwd_{args.horizon}d) ==\n")

    df = _fetch_joined(conn, args.horizon, args.since)
    if df.empty:
        print("No (candidate × fwd return) rows found.")
        print("Run: python scripts/backfill_forward_returns.py")
        return
    print(f"Rows: {len(df):,}    date range: {df['run_date'].min().date()} → "
          f"{df['run_date'].max().date()}    tickers: {df['ticker'].nunique()}\n")

    print(f"─ 1. IC by rank_score quantile (q={args.quantiles}) ─")
    _print_quantile_ic(df, args.quantiles)
    print()

    print(f"─ 2. Tier realization: candidates with rank_score ≥ cut ─")
    _print_tier_realization(df, args.tier_cuts)
    print()

    print("─ 3. Regime-conditional Spearman IC ─")
    _print_regime_ic(df)
    print()

    print(f"─ 4. Block-reason outcomes: what happened on fwd_{args.horizon}d? ─")
    _print_block_outcomes(df)
    print()

    print(f"─ 5. Selected-candidate realization by rank_score bucket (Kelly-tier-tune input) ─")
    _print_selected_bucket(df, args.tier_cuts)
    print()

    print(f"─ 6. Per-bar selection count — how often does tier 2/3 even fire? ─")
    _print_slots_filled(df)


if __name__ == "__main__":
    main()
