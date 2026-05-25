"""Live <-> Sim reconciliation: helpers split into <=50-line units.

Per CLAUDE.md §1c: every logical step is its own helper, single
responsibility, dedicated test. Per §5.13.1: at least one helper walks
through SimAdapter end-to-end (see ``replay_through_sim``).

The CLI (`scripts/reconcile_live_sim.py`) wires these together.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

log = logging.getLogger("kernel.reconciliation")


# ── Data carriers ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LiveFill:
    """A single fill as recorded by the live runner in runs.<broker>.db."""
    run_id: str
    run_date: str          # ISO date string
    ticker: str
    action: str            # 'buy' | 'sell'
    shares: float
    price: float


@dataclass(frozen=True)
class SimDecision:
    """The sim's decision for the same (date, ticker) pair."""
    run_date: str
    ticker: str
    action: str            # 'buy' | 'sell' | 'hold'
    shares: float
    price: float


# ── Loaders ────────────────────────────────────────────────────────────────


def _connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite DB in read-only mode (URI form). Caller must close."""
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"DB does not exist: {p}")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def load_live_fills(
    db_path: str | Path,
    start_date: str,
    end_date: str,
) -> list[LiveFill]:
    """Read live fills from ``runs.<broker>.db`` over an inclusive date range.

    Schema reference: ``trades`` joined to ``pipeline_runs`` on run_id, with
    ``run_type='live'``. Tickers/actions are uppercased / lowercased to a
    canonical form so downstream comparison is case-insensitive.
    """
    conn = _connect_readonly(db_path)
    try:
        rows = conn.execute(
            """
            SELECT t.run_id, p.run_date, t.ticker, t.action, t.shares, t.price
            FROM trades t
            JOIN pipeline_runs p ON t.run_id = p.run_id
            WHERE p.run_type = 'live'
              AND p.run_date >= ?
              AND p.run_date <= ?
            ORDER BY p.run_date, t.ticker, t.action
            """,
            (start_date, end_date),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_live_fill(r) for r in rows]


def _row_to_live_fill(row: tuple) -> LiveFill:
    run_id, run_date, ticker, action, shares, price = row
    return LiveFill(
        run_id=str(run_id),
        run_date=str(run_date),
        ticker=str(ticker or "").upper(),
        action=str(action or "").lower(),
        shares=float(shares) if shares is not None else 0.0,
        price=float(price) if price is not None else float("nan"),
    )


def load_sim_decisions(
    db_path: str | Path,
    start_date: str,
    end_date: str,
) -> list[SimDecision]:
    """Read sim's recorded trades from ``sim_runs.db`` for the same range.

    Treats every recorded sim trade as a 'buy' or 'sell' decision. Tickers
    that appear in live fills but NOT in sim trades are treated as 'hold'
    by ``replay_through_sim`` (they didn't trigger an action in sim).
    """
    conn = _connect_readonly(db_path)
    try:
        rows = conn.execute(
            """
            SELECT p.run_date, t.ticker, t.action, t.shares, t.price
            FROM trades t
            JOIN pipeline_runs p ON t.run_id = p.run_id
            WHERE p.run_type = 'sim'
              AND p.run_date >= ?
              AND p.run_date <= ?
            ORDER BY p.run_date, t.ticker, t.action
            """,
            (start_date, end_date),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_sim_decision(r) for r in rows]


def _row_to_sim_decision(row: tuple) -> SimDecision:
    run_date, ticker, action, shares, price = row
    return SimDecision(
        run_date=str(run_date),
        ticker=str(ticker or "").upper(),
        action=str(action or "").lower(),
        shares=float(shares) if shares is not None else 0.0,
        price=float(price) if price is not None else float("nan"),
    )


# ── Replay (the SimAdapter walk per §5.13.1) ──────────────────────────────


def replay_through_sim(
    fills: Sequence[LiveFill],
    sim_decisions: Sequence[SimDecision] | None = None,
    sim_adapter: Any | None = None,
) -> list[SimDecision]:
    """Match each live fill to sim's decision on the same (date, ticker).

    Two paths:

    * ``sim_adapter`` provided → call ``sim_adapter.lookup_decision(date,
      ticker)`` (used only by the end-to-end SimAdapter test). The adapter
      stub must return ``(action, shares, price)`` tuples.
    * ``sim_decisions`` provided → join from the prior sim DB read.

    If both are None, every fill is mapped to a ``hold`` (sim said nothing).
    """
    if sim_adapter is not None:
        return [_lookup_via_adapter(sim_adapter, f) for f in fills]
    sim_index = _index_sim_decisions(sim_decisions or [])
    return [_match_fill_to_sim(f, sim_index) for f in fills]


def _index_sim_decisions(
    decisions: Iterable[SimDecision],
) -> dict[tuple[str, str], SimDecision]:
    """(date, ticker) -> last decision wins (sim shouldn't double-trade)."""
    return {(d.run_date, d.ticker): d for d in decisions}


def _match_fill_to_sim(
    fill: LiveFill,
    sim_index: dict[tuple[str, str], SimDecision],
) -> SimDecision:
    key = (fill.run_date, fill.ticker)
    if key in sim_index:
        return sim_index[key]
    return SimDecision(
        run_date=fill.run_date, ticker=fill.ticker,
        action="hold", shares=0.0, price=float("nan"),
    )


def _lookup_via_adapter(sim_adapter: Any, fill: LiveFill) -> SimDecision:
    """Call sim_adapter.lookup_decision; defensive against missing methods."""
    try:
        action, shares, price = sim_adapter.lookup_decision(
            fill.run_date, fill.ticker
        )
    except Exception as exc:
        log.warning("sim_adapter.lookup_decision failed for %s/%s: %s",
                    fill.run_date, fill.ticker, exc)
        action, shares, price = "hold", 0.0, float("nan")
    return SimDecision(
        run_date=fill.run_date, ticker=fill.ticker,
        action=str(action).lower(),
        shares=float(shares) if shares is not None else 0.0,
        price=float(price) if price is not None else float("nan"),
    )


# ── Slippage ───────────────────────────────────────────────────────────────


def compute_slippage(
    live_fills: Sequence[LiveFill],
    sim_decisions: Sequence[SimDecision],
) -> dict[str, float]:
    """Return p50/p95 slippage in basis points.

    Slippage(bps) = 1e4 * (live_price - sim_price) / sim_price, signed by
    direction (positive = live paid more on buy / received less on sell).
    Pairs where sim said 'hold' or prices are non-finite are dropped.
    """
    bps: list[float] = []
    sim_index = _index_sim_decisions(sim_decisions)
    for f in live_fills:
        s = sim_index.get((f.run_date, f.ticker))
        if s is None or s.action == "hold":
            continue
        slip = _signed_slippage_bps(f, s)
        if slip is not None:
            bps.append(slip)
    return _slippage_quantiles(bps)


def _signed_slippage_bps(fill: LiveFill, sim: SimDecision) -> float | None:
    if not (math.isfinite(fill.price) and math.isfinite(sim.price)):
        return None
    if sim.price <= 0:
        return None
    raw = (fill.price - sim.price) / sim.price * 1e4
    # Flip sign on sells: live receiving LESS than sim is unfavorable, so
    # we want positive bps to mean "live was worse".
    if fill.action == "sell":
        raw = -raw
    return raw


def _slippage_quantiles(bps: list[float]) -> dict[str, float]:
    if not bps:
        return {"n": 0, "p50_bps": 0.0, "p95_bps": 0.0, "max_bps": 0.0}
    bps_sorted = sorted(bps)
    return {
        "n": len(bps_sorted),
        "p50_bps": _percentile(bps_sorted, 0.50),
        "p95_bps": _percentile(bps_sorted, 0.95),
        "max_bps": bps_sorted[-1],
    }


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1,
                     int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


# ── Decision divergence ────────────────────────────────────────────────────


def compute_decision_divergence(
    live_fills: Sequence[LiveFill],
    sim_decisions: Sequence[SimDecision],
    qty_tolerance_pct: float = 0.05,
) -> dict[str, Any]:
    """Fraction where live + sim disagreed on direction or qty (>5%).

    Returns ``{n, n_disagree, divergence_rate, cases}`` where ``cases`` is
    a list of dicts (date, ticker, live_action, sim_action, live_qty,
    sim_qty, reason) for the operator to inspect.
    """
    sim_index = _index_sim_decisions(sim_decisions)
    cases: list[dict[str, Any]] = []
    for f in live_fills:
        s = sim_index.get((f.run_date, f.ticker)) or _hold_for(f)
        reason = _disagreement_reason(f, s, qty_tolerance_pct)
        if reason is not None:
            cases.append(_disagreement_case(f, s, reason))
    n = len(live_fills)
    return {
        "n": n,
        "n_disagree": len(cases),
        "divergence_rate": (len(cases) / n) if n > 0 else 0.0,
        "cases": cases,
    }


def _hold_for(fill: LiveFill) -> SimDecision:
    return SimDecision(
        run_date=fill.run_date, ticker=fill.ticker,
        action="hold", shares=0.0, price=float("nan"),
    )


def _disagreement_reason(
    fill: LiveFill, sim: SimDecision, qty_tol_pct: float,
) -> str | None:
    if fill.action != sim.action:
        return f"direction:live={fill.action},sim={sim.action}"
    if sim.shares <= 0 and fill.shares > 0:
        return "qty:sim_zero"
    if sim.shares > 0:
        delta = abs(fill.shares - sim.shares) / sim.shares
        if delta > qty_tol_pct:
            return f"qty:{delta:.2%}_diff"
    return None


def _disagreement_case(
    fill: LiveFill, sim: SimDecision, reason: str,
) -> dict[str, Any]:
    return {
        "date": fill.run_date,
        "ticker": fill.ticker,
        "live_action": fill.action,
        "sim_action": sim.action,
        "live_qty": fill.shares,
        "sim_qty": sim.shares,
        "reason": reason,
    }


# ── Rolling IC ────────────────────────────────────────────────────────────


def compute_rolling_ic(
    db_path: str | Path,
    start_date: str,
    end_date: str,
    window_days: int = 30,
    horizon_days: int = 60,
) -> dict[str, Any]:
    """Rolling 30d Spearman IC of model predictions vs realized fwd returns.

    Reads ``candidate_scores`` (predictions) joined to ``ticker_forward_returns``
    (realized fwd_{horizon_days}d) from ``runs.<broker>.db``. If either table is empty
    or join is empty, returns ``{"ok": False, "warn": "..."}`` — does NOT
    crash (per spec).
    """
    pairs = _load_score_realized_pairs(db_path, start_date, end_date, horizon_days)
    if not pairs:
        return {"ok": False, "warn": "no_score_realized_pairs",
                "n": 0, "ic": 0.0, "window_days": window_days,
                "horizon_days": horizon_days}
    ic = _spearman_ic(
        [p[0] for p in pairs[-window_days * 50:]],
        [p[1] for p in pairs[-window_days * 50:]],
    )
    return {"ok": True, "n": len(pairs), "ic": ic, "window_days": window_days,
            "horizon_days": horizon_days}


def _load_score_realized_pairs(
    db_path: str | Path, start_date: str, end_date: str, horizon_days: int = 60,
) -> list[tuple[float, float]]:
    """Join candidate_scores.rank_score with ticker_forward_returns.fwd_Nd.

    Returns [] (with a logged warn) when either table is missing or empty.
    """
    if horizon_days not in {1, 5, 10, 20, 60}:
        raise ValueError(f"unsupported IC horizon_days={horizon_days}")
    fwd_col = f"fwd_{horizon_days}d"
    try:
        conn = _connect_readonly(db_path)
    except FileNotFoundError:
        return []
    try:
        try:
            rows = conn.execute(
                f"""
                SELECT cs.rank_score, tfr.{fwd_col}
                FROM candidate_scores cs
                JOIN pipeline_runs p ON cs.run_id = p.run_id
                JOIN ticker_forward_returns tfr
                  ON tfr.ticker = cs.ticker AND tfr.as_of_date = p.run_date
                WHERE p.run_date >= ? AND p.run_date <= ?
                  AND cs.rank_score IS NOT NULL
                  AND tfr.{fwd_col} IS NOT NULL
                """,
                (start_date, end_date),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            log.warning("rolling_ic: table missing — %s", exc)
            return []
    finally:
        conn.close()
    return [(float(r[0]), float(r[1])) for r in rows
            if r[0] is not None and r[1] is not None]


def _spearman_ic(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman = Pearson on ranks. Hand-rolled to avoid scipy dep."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _ranks(xs)
    ry = _ranks(ys)
    n = len(rx)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _ranks(vals: Sequence[float]) -> list[float]:
    """Average-rank tie-breaker."""
    indexed = sorted(enumerate(vals), key=lambda t: t[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0   # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


# ── Report emission ────────────────────────────────────────────────────────


def emit_report(metrics: dict[str, Any], output_path: str | Path) -> Path:
    """Write a markdown report. Returns the absolute path written.

    Sections: Summary / Per-Day Breakdown / Divergence Cases / Rolling IC.
    Caller is responsible for assembling ``metrics`` (see CLI).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.extend(_report_header(metrics))
    lines.extend(_report_summary(metrics))
    lines.extend(_report_per_day(metrics))
    lines.extend(_report_divergence_cases(metrics))
    lines.extend(_report_rolling_ic(metrics))
    out.write_text("\n".join(lines) + "\n")
    return out.resolve()


def _report_header(metrics: dict[str, Any]) -> list[str]:
    return [
        f"# Live<->Sim Reconciliation — {metrics.get('start_date', '?')} "
        f"to {metrics.get('end_date', '?')}",
        "",
        f"- Broker: `{metrics.get('broker', '?')}`",
        f"- Live fills: **{metrics.get('n_fills', 0)}**",
        f"- Generated at: {metrics.get('generated_at', '')}",
        "",
    ]


def _report_summary(metrics: dict[str, Any]) -> list[str]:
    div = metrics.get("divergence", {})
    slip = metrics.get("slippage", {})
    return [
        "## Summary",
        "",
        f"- Divergence rate: **{div.get('divergence_rate', 0.0):.2%}** "
        f"({div.get('n_disagree', 0)} / {div.get('n', 0)})",
        f"- Slippage p50: **{slip.get('p50_bps', 0.0):+.2f} bps**",
        f"- Slippage p95: **{slip.get('p95_bps', 0.0):+.2f} bps**",
        f"- Slippage max: **{slip.get('max_bps', 0.0):+.2f} bps**",
        "",
    ]


def _report_per_day(metrics: dict[str, Any]) -> list[str]:
    rows = metrics.get("per_day", [])
    if not rows:
        return ["## Per-Day Breakdown", "", "_(no fills in window)_", ""]
    out = ["## Per-Day Breakdown", "",
           "| Date | Fills | Disagree | Slip p50 (bps) |",
           "| --- | ---: | ---: | ---: |"]
    for r in rows:
        out.append(
            f"| {r['date']} | {r['n_fills']} | {r['n_disagree']} | "
            f"{r['slip_p50_bps']:+.2f} |"
        )
    out.append("")
    return out


def _report_divergence_cases(metrics: dict[str, Any]) -> list[str]:
    cases = metrics.get("divergence", {}).get("cases", [])
    if not cases:
        return ["## Divergence Cases", "", "_(none)_", ""]
    out = ["## Divergence Cases", "",
           "| Date | Ticker | Live | Sim | Live qty | Sim qty | Reason |",
           "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for c in cases[:50]:        # cap report at 50 to stay readable
        out.append(
            f"| {c['date']} | {c['ticker']} | {c['live_action']} | "
            f"{c['sim_action']} | {c['live_qty']:.0f} | "
            f"{c['sim_qty']:.0f} | {c['reason']} |"
        )
    if len(cases) > 50:
        out.append(f"_(... {len(cases) - 50} more cases truncated)_")
    out.append("")
    return out


def _report_rolling_ic(metrics: dict[str, Any]) -> list[str]:
    ic = metrics.get("rolling_ic", {})
    if not ic.get("ok"):
        return ["## Rolling IC",
                "",
                f"_(unavailable: {ic.get('warn', 'unknown')})_",
                ""]
    return ["## Rolling IC",
            "",
            f"- Window: {ic.get('window_days', 30)}d",
            f"- N pairs: {ic.get('n', 0)}",
            f"- Spearman IC: **{ic.get('ic', 0.0):+.4f}**",
            ""]


def build_per_day_breakdown(
    fills: Sequence[LiveFill],
    sim_decisions: Sequence[SimDecision],
) -> list[dict[str, Any]]:
    """Bucket the fills by date and compute per-day divergence + slip."""
    by_date: dict[str, list[LiveFill]] = {}
    for f in fills:
        by_date.setdefault(f.run_date, []).append(f)
    out: list[dict[str, Any]] = []
    for date in sorted(by_date.keys()):
        day_fills = by_date[date]
        day_div = compute_decision_divergence(day_fills, sim_decisions)
        day_slip = compute_slippage(day_fills, sim_decisions)
        out.append({
            "date": date,
            "n_fills": len(day_fills),
            "n_disagree": day_div["n_disagree"],
            "slip_p50_bps": day_slip["p50_bps"],
        })
    return out
