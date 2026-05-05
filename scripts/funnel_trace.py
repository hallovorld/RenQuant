"""Funnel trace — turn a sim's per-bar log + decision-tree DB into a
single histogram showing where candidates leak, with skip-reason
breakdown.

Usage::

    python scripts/funnel_trace.py /tmp/sim_x/diag.log
    python scripts/funnel_trace.py --db data/sim_runs.db
    python scripts/funnel_trace.py /tmp/x.log --db data/sim_runs.db

Two complementary data sources:

  1. Log file — gives PER-BAR funnel counts (Phase 2b → vol gate →
     drift → kelly counts) plus the new ngb_skipped:* / kelly_zero:*
     summary lines my 2026-05-04 instrumentation emits.

  2. Sim DB — `candidate_scores` table has per-(bar, ticker) mu /
     sigma / kelly_target_pct / blocked_by columns. Aggregating by
     blocked_by gives a cross-sim histogram of skip reasons.

Why both: log is fast and human-readable for one run; DB queries
work across the whole run history without re-parsing logs. They
should AGREE — if they don't, the persistence layer dropped data.

This is the tool the user asked for after asking "are we lacking of
insightful logs?". Now we have insightful logs AND a one-shot way to
read them.
"""
from __future__ import annotations

import argparse
import collections
import re
import sqlite3
import statistics
import sys
from pathlib import Path


# ── log-side parsers ─────────────────────────────────────────────────────────

PHASE2B_RE  = re.compile(r"Phase 2b \(buy scan\): (\d+) candidates from (\d+) tickers")
VOLGATE_RE  = re.compile(r"RealizedVolGateTask: dropped (\d+)/(\d+)")
POSCON_RE   = re.compile(r"PositionConcentrationGateTask: dropped (\d+)/(\d+)")
POSTSTOP_RE = re.compile(r"PostStopCooldownFilterTask: dropped (\d+)/(\d+)")
DRIFT_RE    = re.compile(r"DriftGuardTask: \d+/\d+ \([\d.]+%\) STRUCTURALLY missing")
NGB_RE      = re.compile(
    r"ApplyNGBoostTask: mode=\w+ +λ=[\d.]+ +n_cands=(\d+) +n_holdings=\d+"
    r"(?: +\(set_μσ=(\d+) +not_in_idx=(\d+) +mu_nan=(\d+) +sigma_nan=(\d+)\))?"
)
KELLY_RE    = re.compile(
    r"ApplyKellySizingTask: .*cands=(\d+) non-zero \(avg=[\d.]+%\)"
    r" +holdings=\d+ non-zero \(avg=[\d.]+%\)"
    r"(?: +zero_reasons\[([^\]]*)\])?"
)
QPBUY_RE    = re.compile(r"QP_BUY ")
DONE_RE     = re.compile(r"InferencePipeline DONE")


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text()
    blocks = re.split(r"InferencePipeline START  date=(\d{4}-\d{2}-\d{2})", text)
    # blocks = [preamble, date1, body1, date2, body2, ...]
    n_bars = (len(blocks) - 1) // 2

    funnel = collections.defaultdict(list)
    skip_kelly = collections.Counter()
    skip_ngb   = collections.Counter()
    bars_drift = 0

    for i in range(n_bars):
        body = blocks[2 + 2 * i]
        m_p2b   = PHASE2B_RE.search(body);  p2b_n = int(m_p2b.group(1)) if m_p2b else 0
        m_vol   = VOLGATE_RE.search(body)
        vol_drop = int(m_vol.group(1)) if m_vol else 0
        m_pos   = POSCON_RE.search(body);   pos_drop = int(m_pos.group(1)) if m_pos else 0
        m_pst   = POSTSTOP_RE.search(body); pst_drop = int(m_pst.group(1)) if m_pst else 0
        is_drift = bool(DRIFT_RE.search(body))
        if is_drift:
            bars_drift += 1
        m_ngb   = NGB_RE.search(body)
        m_kelly = KELLY_RE.search(body)
        n_qp_buy = len(QPBUY_RE.findall(body))

        vol_kept = max(0, p2b_n - vol_drop)
        pos_kept = max(0, vol_kept - pos_drop)
        pst_kept = max(0, pos_kept - pst_drop)
        drift_clear = 0 if is_drift else pst_kept

        funnel["00_phase2b"].append(p2b_n)
        funnel["01_vol_kept"].append(vol_kept)
        funnel["02_poscon_kept"].append(pos_kept)
        funnel["03_poststop_kept"].append(pst_kept)
        funnel["04_drift_clear"].append(drift_clear)
        if m_ngb and m_ngb.group(2):
            funnel["05_ngb_setμσ"].append(int(m_ngb.group(2)))
            skip_ngb["not_in_idx"]  += int(m_ngb.group(3))
            skip_ngb["mu_nan"]       += int(m_ngb.group(4))
            skip_ngb["sigma_nan"]    += int(m_ngb.group(5))
        else:
            funnel["05_ngb_setμσ"].append(int(m_ngb.group(1)) if m_ngb else 0)

        funnel["06_kelly_nonzero"].append(int(m_kelly.group(1)) if m_kelly else 0)
        if m_kelly and m_kelly.group(2):
            for tok in m_kelly.group(2).split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    skip_kelly[k] += int(v)
        funnel["07_qp_buys"].append(n_qp_buy)

    return {
        "n_bars":     n_bars,
        "bars_drift": bars_drift,
        "funnel":     dict(funnel),
        "skip_ngb":   dict(skip_ngb),
        "skip_kelly": dict(skip_kelly),
    }


# ── DB-side aggregator ──────────────────────────────────────────────────────

def parse_db(db_path: Path, since_date: str | None = None) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    where = ""
    params: tuple = ()
    if since_date:
        where = "WHERE pr.run_date >= ?"
        params = (since_date,)
    rows = list(conn.execute(f"""
        SELECT pr.run_date, cs.ticker, cs.role, cs.mu, cs.sigma,
               cs.kelly_target_pct, cs.selected, cs.blocked_by
        FROM pipeline_runs pr
        JOIN candidate_scores cs ON cs.run_id = pr.run_id
        {where}
        ORDER BY pr.run_date
    """, params))
    by_block = collections.Counter()
    n_total = n_cand = n_kelly_pos = 0
    for r in rows:
        n_total += 1
        if r["role"] == "candidate":
            n_cand += 1
            if r["kelly_target_pct"] and r["kelly_target_pct"] > 0:
                n_kelly_pos += 1
            if r["blocked_by"]:
                by_block[r["blocked_by"]] += 1
            else:
                by_block["(none)"] += 1
    return {
        "n_rows": n_total, "n_cands": n_cand,
        "n_kelly_pos": n_kelly_pos, "by_block": dict(by_block),
    }


# ── per-ticker drill-down ────────────────────────────────────────────────────

def parse_db_ticker(db_path: Path, ticker: str,
                     since_date: str | None = None,
                     limit: int = 50) -> list[dict]:
    """Per-ticker journey: every bar, what role + blocked_by + mu / sigma /
    kelly_target_pct — sorted by run_date.

    Use case (the user's mandate "make it explainable"): given a ticker
    like AAPL, see exactly why it never traded across the 250-bar
    holdout. If `blocked_by` is consistently `kelly_zero:mu_le_min_edge`,
    the model's μ for this ticker is predominantly negative on this
    window and the strategy is correctly NOT betting.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    where = "cs.ticker = ?"
    params: tuple = (ticker,)
    if since_date:
        where += " AND pr.run_date >= ?"
        params += (since_date,)
    rows = list(conn.execute(f"""
        SELECT pr.run_date, cs.role, cs.rank_score, cs.mu, cs.sigma,
               cs.kelly_target_pct, cs.selected, cs.blocked_by
        FROM pipeline_runs pr
        JOIN candidate_scores cs ON cs.run_id = pr.run_id
        WHERE {where}
        ORDER BY pr.run_date DESC
        LIMIT ?
    """, params + (limit,)))
    return [dict(r) for r in rows]


def parse_db_mu_distribution(db_path: Path,
                              since_date: str | None = None) -> dict:
    """Histogram μ across all candidate-role rows. Reveals whether the
    model's μ predictions are biased toward negative (likely cause of
    Kelly returning 0 for nearly all candidates), centered at 0
    (50/50 sign split as expected), or biased positive (no Kelly leak,
    bug must be elsewhere).

    Buckets are chosen to highlight the thresholds Kelly cares about
    most: μ ≤ 0 (Kelly=0 for sure) vs μ > 0 (Kelly potentially > 0).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    where = "cs.role = 'candidate'"
    params: tuple = ()
    if since_date:
        where += " AND pr.run_date >= ?"
        params = (since_date,)
    rows = list(conn.execute(f"""
        SELECT cs.mu
        FROM pipeline_runs pr
        JOIN candidate_scores cs ON cs.run_id = pr.run_id
        WHERE {where}
    """, params))
    buckets = collections.OrderedDict([
        ("mu_null",           0),
        ("mu_<= -1pct",       0),
        ("mu_-1pct..-0.5pct", 0),
        ("mu_-0.5pct..0",     0),
        ("mu_=0",             0),
        ("mu_0..+0.5pct",     0),
        ("mu_+0.5pct..+1pct", 0),
        ("mu_>= +1pct",       0),
    ])
    for r in rows:
        mu = r["mu"]
        if mu is None:
            buckets["mu_null"] += 1
        elif mu <= -0.01:
            buckets["mu_<= -1pct"] += 1
        elif mu <= -0.005:
            buckets["mu_-1pct..-0.5pct"] += 1
        elif mu < 0:
            buckets["mu_-0.5pct..0"] += 1
        elif mu == 0:
            buckets["mu_=0"] += 1
        elif mu < 0.005:
            buckets["mu_0..+0.5pct"] += 1
        elif mu < 0.01:
            buckets["mu_+0.5pct..+1pct"] += 1
        else:
            buckets["mu_>= +1pct"] += 1
    return {"n": len(rows), "buckets": buckets}


def print_mu_distribution(db_path: Path, parsed: dict) -> None:
    print(f"\n[db] {db_path}  μ-distribution across {parsed['n']} candidate rows")
    print("-" * 78)
    n = max(parsed["n"], 1)
    cum_le_zero = 0
    for bucket, cnt in parsed["buckets"].items():
        bar = "█" * int(60 * cnt / n)
        print(f"  {bucket:<22}  {cnt:>7}  {100*cnt/n:5.1f}%  {bar}")
        if bucket in ("mu_null", "mu_<= -1pct", "mu_-1pct..-0.5pct",
                       "mu_-0.5pct..0", "mu_=0"):
            cum_le_zero += cnt
    print(f"\n  μ ≤ 0 (Kelly returns 0):  {cum_le_zero}  ({100*cum_le_zero/n:.1f}%)")
    print(f"  μ >  0 (Kelly may bet):   {n - cum_le_zero}  "
          f"({100*(n-cum_le_zero)/n:.1f}%)")


def print_ticker_report(db_path: Path, ticker: str, rows: list[dict]) -> None:
    print(f"\n[db] {db_path}  ticker={ticker}  ({len(rows)} most-recent rows)")
    print("-" * 78)
    if not rows:
        print(f"  no rows — ticker may not have appeared as candidate/holding")
        return
    print(f"  {'date':12s} {'role':10s} {'mu':>9} {'sigma':>9} {'kelly':>8} "
          f"{'sel':>4}  blocked_by")
    blocked_counts = collections.Counter()
    for r in rows:
        mu = r["mu"]
        sg = r["sigma"]
        ky = r["kelly_target_pct"]
        sel = r["selected"]
        blk = r["blocked_by"] or ""
        blocked_counts[blk or "(none)"] += 1
        mu_s = f"{mu:>+.5f}"  if mu is not None else "    None"
        sg_s = f"{sg:>.5f}"   if sg is not None else "    None"
        ky_s = f"{ky:>.4f}"   if ky is not None else "  None"
        print(f"  {r['run_date']:12s} {r['role']:10s} {mu_s:>9} {sg_s:>9} "
              f"{ky_s:>8} {(sel or 0):>4}  {blk}")
    print("\n  Per-ticker blocked_by histogram:")
    for k, v in sorted(blocked_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<48}  {v:>4}")


# ── pretty-print ────────────────────────────────────────────────────────────

def print_log_report(log_path: Path, parsed: dict) -> None:
    n_bars = parsed["n_bars"]
    print(f"\n[log] {log_path}  ({n_bars} bars)")
    print("-" * 78)
    f = parsed["funnel"]
    print(f"  {'stage':<24} {'mean':>6} {'med':>5} {'p25':>5} {'p75':>5} {'min':>4} {'max':>4} {'zero/n':>10}")
    for k in sorted(f):
        v = f[k]
        if not v:
            continue
        sorted_v = sorted(v)
        p25 = sorted_v[len(v)//4]
        p75 = sorted_v[3*len(v)//4]
        zeros = sum(1 for x in v if x == 0)
        print(f"  {k:<24} {statistics.fmean(v):>6.2f} {statistics.median(v):>5.0f} "
              f"{p25:>5} {p75:>5} {min(v):>4} {max(v):>4} {zeros:>4}/{n_bars:<5}")
    print(f"  bars w/ DriftGuard fail-safe: {parsed['bars_drift']}/{n_bars}")
    if parsed["skip_ngb"]:
        print(f"\n  NGBoost skip totals across all bars:")
        for k, v in sorted(parsed["skip_ngb"].items(), key=lambda x: -x[1]):
            print(f"    ngb_skipped:{k:<24}  {v}")
    if parsed["skip_kelly"]:
        print(f"\n  Kelly zero-reason totals across all bars:")
        for k, v in sorted(parsed["skip_kelly"].items(), key=lambda x: -x[1]):
            print(f"    kelly_zero:{k:<24}  {v}")


def print_db_report(db_path: Path, parsed: dict) -> None:
    print(f"\n[db] {db_path}  ({parsed['n_rows']} rows)")
    print("-" * 78)
    print(f"  candidate-role rows:       {parsed['n_cands']}")
    print(f"  kelly_target_pct > 0:      {parsed['n_kelly_pos']}")
    if parsed["by_block"]:
        print("\n  blocked_by histogram:")
        total = sum(parsed["by_block"].values())
        for k, v in sorted(parsed["by_block"].items(), key=lambda x: -x[1]):
            pct = 100 * v / max(total, 1)
            print(f"    {k:<48}  {v:>6}  ({pct:5.1f}%)")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", type=Path,
                     help="Sim log file (one InferencePipeline per bar)")
    ap.add_argument("--db", type=Path, default=None,
                     help="SQLite sim DB (data/sim_runs.db)")
    ap.add_argument("--since", default=None,
                     help="DB filter: only run_date >= this date (YYYY-MM-DD)")
    ap.add_argument("--ticker", default=None,
                     help="DB drill-down: per-bar journey for one ticker "
                          "(role / mu / sigma / kelly / blocked_by)")
    ap.add_argument("--limit", type=int, default=50,
                     help="--ticker max rows to display (default 50)")
    ap.add_argument("--mu-histogram", action="store_true",
                     help="DB μ-distribution histogram across all candidate rows")
    args = ap.parse_args()

    if args.log is None and args.db is None:
        ap.error("Provide a log file, --db, or both.")

    if args.log:
        if not args.log.exists():
            sys.exit(f"log not found: {args.log}")
        print_log_report(args.log, parse_log(args.log))

    if args.db:
        if not args.db.exists():
            sys.exit(f"db not found: {args.db}")
        if args.ticker:
            rows = parse_db_ticker(args.db, args.ticker,
                                    since_date=args.since,
                                    limit=args.limit)
            print_ticker_report(args.db, args.ticker, rows)
        elif args.mu_histogram:
            print_mu_distribution(args.db,
                parse_db_mu_distribution(args.db, since_date=args.since))
        else:
            print_db_report(args.db, parse_db(args.db, since_date=args.since))


if __name__ == "__main__":
    main()
