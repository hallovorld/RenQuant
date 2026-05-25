#!/usr/bin/env python
"""Trade-level decision attribution from RenQuant SQLite traces.

This is a diagnostic tool for APY / Sharpe root cause analysis. It pairs
executed buy/sell rows into closed long round trips, preserves the entry
decision payload, and reports which decision paths contributed or destroyed
net P&L.

The script is intentionally read-only. It accepts both the legacy `trades`
schema and the newer schema with `score_snapshot_json` / `decision_inputs_json`.

Usage:
    python scripts/analyze_trade_decision_attribution.py --db data/runs.alpaca.db
    python scripts/analyze_trade_decision_attribution.py --db data/sim_runs.db --run-type sim
    python scripts/analyze_trade_decision_attribution.py --since 2026-05-01 --output artifacts/trade_attr.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


TRADE_COLUMNS = [
    "run_id",
    "trade_date",
    "ticker",
    "action",
    "shares",
    "price",
    "invest",
    "target_pct",
    "exit_reason",
    "pnl_pct",
    "hold_days",
    "tax",
    "gross_pnl",
    "proceeds_basis",
    "net_pnl_after_tax",
    "rank_score",
    "conviction",
    "sigma_mult",
    "mu",
    "mu_horizon_days",
    "sigma",
    "panel_score",
    "rs_score",
    "expected_return",
    "expected_return_horizon_days",
    "kelly_target_pct",
    "model_type",
    "sector",
    "blocked_by",
    "qp_delta_w",
    "qp_target_w",
    "qp_status",
    "order_type",
    "source",
    "source_job",
    "source_task",
    "order_source",
    "attribution_version",
    "score_snapshot_json",
    "decision_inputs_json",
]

RUN_COLUMNS = [
    "run_id",
    "run_date",
    "run_type",
    "strategy",
    "regime",
    "confidence",
    "portfolio_value",
    "cash",
    "n_candidates",
    "n_exits",
    "n_rotations",
    "n_buys",
]


@dataclass
class AttributionResult:
    round_trips: pd.DataFrame
    summary: dict[str, Any]
    group_tables: dict[str, list[dict[str, Any]]]


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_cell(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, float) and math.isnan(value):
        return {}
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_trade_rows(
    db_path: Path,
    *,
    since: str | None = None,
    until: str | None = None,
    run_type: str | None = None,
) -> pd.DataFrame:
    """Load trades joined to pipeline_runs, tolerating old schemas."""
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = _connect_readonly(db_path)
    try:
        trade_cols = _table_columns(conn, "trades")
        run_cols = _table_columns(conn, "pipeline_runs")
        if not trade_cols:
            raise RuntimeError("missing trades table")
        if not run_cols:
            raise RuntimeError("missing pipeline_runs table")

        selected_trade_cols = [c for c in TRADE_COLUMNS if c in trade_cols]
        selected_run_cols = [c for c in RUN_COLUMNS if c in run_cols]
        trade_select = ["rowid AS trade_rowid"] + selected_trade_cols
        trades = pd.read_sql_query(
            f"SELECT {', '.join(trade_select)} FROM trades",
            conn,
        )
        runs = pd.read_sql_query(
            f"SELECT {', '.join(selected_run_cols)} FROM pipeline_runs",
            conn,
        )
    finally:
        conn.close()

    for col in TRADE_COLUMNS:
        if col not in trades.columns:
            trades[col] = None
    for col in RUN_COLUMNS:
        if col not in runs.columns:
            runs[col] = None

    runs = runs.rename(columns={
        "run_date": "run_date",
        "run_type": "run_type",
        "strategy": "run_strategy",
        "regime": "run_regime",
        "confidence": "run_confidence",
    })
    out = trades.merge(runs, on="run_id", how="left")
    out["date"] = out["trade_date"].where(
        out["trade_date"].notna() & (out["trade_date"].astype(str) != ""),
        out["run_date"],
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if since:
        out = out[out["date"] >= pd.Timestamp(since)]
    if until:
        out = out[out["date"] <= pd.Timestamp(until)]
    if run_type:
        out = out[out["run_type"].fillna("").astype(str) == run_type]
    out["action"] = out["action"].fillna("").astype(str).str.lower()
    return out.sort_values(["date", "trade_rowid"]).reset_index(drop=True)


def _entry_field(row: pd.Series, snapshot: dict[str, Any], key: str) -> Any:
    if key in snapshot and snapshot[key] is not None:
        return snapshot[key]
    return row.get(key)


def _exit_payload(row: pd.Series) -> dict[str, Any]:
    snapshot = _json_cell(row.get("score_snapshot_json"))
    decision = _json_cell(row.get("decision_inputs_json"))
    return {
        "exit_order_type": row.get("order_type"),
        "exit_source": row.get("source"),
        "exit_source_job": row.get("source_job"),
        "exit_source_task": row.get("source_task"),
        "exit_order_source": row.get("order_source"),
        "exit_attribution_version": row.get("attribution_version"),
        "exit_acceptance_reason": decision.get("acceptance_reason"),
        "exit_mu_horizon_days": _entry_field(row, snapshot, "mu_horizon_days"),
        "exit_expected_return_horizon_days": _entry_field(
            row, snapshot, "expected_return_horizon_days",
        ),
        "exit_model_type": _entry_field(row, snapshot, "model_type"),
        "exit_sector": _entry_field(row, snapshot, "sector"),
        "exit_blocked_by": _entry_field(row, snapshot, "blocked_by"),
        "exit_qp_delta_w": _finite_float(row.get("qp_delta_w")),
        "exit_qp_target_w": _finite_float(row.get("qp_target_w")),
        "exit_qp_status": row.get("qp_status"),
        "exit_decision_inputs": decision,
        "exit_score_snapshot": snapshot,
    }


def _hold_bucket(hold_days: float | None) -> str:
    if hold_days is None or not math.isfinite(hold_days):
        return "unknown"
    if hold_days <= 5:
        return "00-05d"
    if hold_days <= 20:
        return "06-20d"
    if hold_days <= 60:
        return "21-60d"
    if hold_days <= 180:
        return "061-180d"
    if hold_days <= 365:
        return "181-365d"
    return "366d+"


def build_round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """Pair buy and sell executions into FIFO long round trips."""
    open_lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    rows: list[dict[str, Any]] = []
    unmatched_sells = 0

    for _, row in trades.iterrows():
        action = str(row.get("action", "")).lower()
        ticker = str(row.get("ticker") or "")
        if not ticker:
            continue

        if action == "buy":
            shares = _finite_float(row.get("shares"))
            price = _finite_float(row.get("price"))
            if shares is None or price is None or shares <= 0 or price <= 0:
                continue
            invest = _finite_float(row.get("invest"))
            snapshot = _json_cell(row.get("score_snapshot_json"))
            decision = _json_cell(row.get("decision_inputs_json"))
            open_lots[ticker].append({
                "ticker": ticker,
                "shares_left": shares,
                "entry_shares": shares,
                "entry_date": row.get("date"),
                "entry_price": price,
                "entry_notional_per_share": (invest / shares) if invest and shares else price,
                "entry_order_type": row.get("order_type"),
                "entry_source": row.get("source"),
                "entry_source_job": row.get("source_job"),
                "entry_source_task": row.get("source_task"),
                "entry_order_source": row.get("order_source"),
                "entry_attribution_version": row.get("attribution_version"),
                "entry_rank_score": _finite_float(_entry_field(row, snapshot, "rank_score")),
                "entry_panel_score": _finite_float(_entry_field(row, snapshot, "panel_score")),
                "entry_rs_score": _finite_float(_entry_field(row, snapshot, "rs_score")),
                "entry_mu": _finite_float(_entry_field(row, snapshot, "mu")),
                "entry_mu_horizon_days": _entry_field(
                    row, snapshot, "mu_horizon_days",
                ),
                "entry_sigma": _finite_float(_entry_field(row, snapshot, "sigma")),
                "entry_kelly_target_pct": _finite_float(
                    _entry_field(row, snapshot, "kelly_target_pct")
                ),
                "entry_expected_return": _finite_float(
                    _entry_field(row, snapshot, "expected_return")
                ),
                "entry_expected_return_horizon_days": _entry_field(
                    row, snapshot, "expected_return_horizon_days",
                ),
                "entry_model_type": _entry_field(row, snapshot, "model_type"),
                "entry_sector": _entry_field(row, snapshot, "sector"),
                "entry_blocked_by": _entry_field(row, snapshot, "blocked_by"),
                "entry_qp_delta_w": _finite_float(row.get("qp_delta_w")),
                "entry_qp_target_w": _finite_float(row.get("qp_target_w")),
                "entry_qp_status": row.get("qp_status"),
                "entry_regime": snapshot.get("regime") or row.get("run_regime"),
                "entry_confidence": _finite_float(
                    snapshot.get("confidence", row.get("run_confidence"))
                ),
                "entry_acceptance_reason": decision.get("acceptance_reason"),
                "entry_decision_inputs": decision,
                "entry_score_snapshot": snapshot,
                "entry_run_id": row.get("run_id"),
            })
            continue

        if action != "sell":
            continue

        sell_price = _finite_float(row.get("price"))
        if sell_price is None or sell_price <= 0:
            continue
        total_open = sum(float(lot["shares_left"]) for lot in open_lots[ticker])
        sell_shares = _finite_float(row.get("shares"))
        if sell_shares is None or sell_shares <= 0:
            sell_shares = total_open
        if sell_shares <= 0 or total_open <= 0:
            unmatched_sells += 1
            continue

        pnl_pct = _finite_float(row.get("pnl_pct"))
        hold_days_from_sell = _finite_float(row.get("hold_days"))
        tax_total = _finite_float(row.get("tax")) or 0.0
        remaining = min(sell_shares, total_open)
        matched_total = remaining
        matched_rows: list[dict[str, Any]] = []
        exit_payload = _exit_payload(row)

        while remaining > 1e-9 and open_lots[ticker]:
            lot = open_lots[ticker][0]
            take = min(remaining, float(lot["shares_left"]))
            entry_price = float(lot["entry_price"])
            entry_notional = take * float(lot["entry_notional_per_share"])
            if entry_price > 0:
                gross_return = (sell_price - entry_price) / entry_price
                gross_pnl = entry_notional * gross_return
            elif pnl_pct is None:
                gross_return = np.nan
                gross_pnl = np.nan
            else:
                gross_return = pnl_pct
                gross_pnl = entry_notional * gross_return
            entry_date = pd.Timestamp(lot["entry_date"]) if lot["entry_date"] is not None else None
            exit_date = pd.Timestamp(row.get("date")) if row.get("date") is not None else None
            if hold_days_from_sell is not None:
                hold_days = hold_days_from_sell
            elif entry_date is not None and exit_date is not None:
                hold_days = float((exit_date - entry_date).days)
            else:
                hold_days = None

            matched_rows.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "shares": take,
                "entry_price": entry_price,
                "exit_price": sell_price,
                "entry_notional": entry_notional,
                "gross_return": gross_return,
                "gross_pnl": gross_pnl,
                "tax": 0.0,
                "net_pnl": gross_pnl,
                "net_return": gross_pnl / entry_notional if entry_notional else np.nan,
                "hold_days": hold_days,
                "hold_bucket": _hold_bucket(hold_days),
                "exit_reason": row.get("exit_reason"),
                "exit_run_id": row.get("run_id"),
                "exit_run_regime": row.get("run_regime"),
                **exit_payload,
                **{k: lot.get(k) for k in (
                    "entry_order_type",
                    "entry_source",
                    "entry_source_job",
                    "entry_source_task",
                    "entry_order_source",
                    "entry_attribution_version",
                    "entry_rank_score",
                    "entry_panel_score",
                    "entry_rs_score",
                    "entry_mu",
                    "entry_mu_horizon_days",
                    "entry_sigma",
                    "entry_kelly_target_pct",
                    "entry_expected_return",
                    "entry_expected_return_horizon_days",
                    "entry_model_type",
                    "entry_sector",
                    "entry_blocked_by",
                    "entry_qp_delta_w",
                    "entry_qp_target_w",
                    "entry_qp_status",
                    "entry_regime",
                    "entry_confidence",
                    "entry_acceptance_reason",
                    "entry_decision_inputs",
                    "entry_score_snapshot",
                    "entry_run_id",
                )},
            })

            lot["shares_left"] = float(lot["shares_left"]) - take
            remaining -= take
            if lot["shares_left"] <= 1e-9:
                open_lots[ticker].popleft()

        if matched_rows:
            positive_gross = sum(
                max(0.0, _finite_float(r.get("gross_pnl")) or 0.0)
                for r in matched_rows
            )
            if tax_total > 0 and positive_gross > 0:
                for r in matched_rows:
                    gp = max(0.0, _finite_float(r.get("gross_pnl")) or 0.0)
                    tax = tax_total * (gp / positive_gross)
                    r["tax"] = tax
                    gross = _finite_float(r.get("gross_pnl")) or 0.0
                    notional = _finite_float(r.get("entry_notional")) or 0.0
                    r["net_pnl"] = gross - tax
                    r["net_return"] = r["net_pnl"] / notional if notional else np.nan
            rows.extend(matched_rows)

    out = pd.DataFrame(rows)
    if not out.empty:
        out.attrs["unmatched_sells"] = unmatched_sells
        out.attrs["open_lots"] = sum(len(v) for v in open_lots.values())
    return out


def summarize_round_trips(round_trips: pd.DataFrame) -> dict[str, Any]:
    if round_trips.empty:
        return {
            "n_round_trips": 0,
            "win_rate": None,
            "profit_factor": None,
            "gross_pnl": 0.0,
            "tax": 0.0,
            "net_pnl": 0.0,
        }
    gross_pnl = round_trips["gross_pnl"].fillna(0.0)
    net_pnl = round_trips["net_pnl"].fillna(0.0)
    wins = net_pnl[net_pnl > 0]
    losses = net_pnl[net_pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "n_round_trips": int(len(round_trips)),
        "win_rate": float((net_pnl > 0).mean()),
        "avg_gross_return": float(round_trips["gross_return"].mean()),
        "avg_net_return": float(round_trips["net_return"].mean()),
        "median_net_return": float(round_trips["net_return"].median()),
        "avg_hold_days": float(round_trips["hold_days"].dropna().mean())
        if round_trips["hold_days"].notna().any() else None,
        "gross_pnl": float(gross_pnl.sum()),
        "tax": float(round_trips["tax"].fillna(0.0).sum()),
        "net_pnl": float(net_pnl.sum()),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "unmatched_sells": int(round_trips.attrs.get("unmatched_sells", 0)),
        "open_lots": int(round_trips.attrs.get("open_lots", 0)),
    }


def _group_metrics(round_trips: pd.DataFrame, group_col: str, min_n: int) -> pd.DataFrame:
    if round_trips.empty or group_col not in round_trips.columns:
        return pd.DataFrame()
    rows = []
    for key, g in round_trips.groupby(group_col, dropna=False, observed=False):
        if len(g) < min_n:
            continue
        summary = summarize_round_trips(g)
        rows.append({
            group_col: "NULL" if pd.isna(key) else str(key),
            "n": summary["n_round_trips"],
            "win_rate": summary["win_rate"],
            "avg_net_return": summary.get("avg_net_return"),
            "net_pnl": summary["net_pnl"],
            "tax": summary["tax"],
            "profit_factor": summary["profit_factor"],
            "avg_hold_days": summary.get("avg_hold_days"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=True)


def _rank_quantile_table(round_trips: pd.DataFrame) -> pd.DataFrame:
    if round_trips.empty or "entry_rank_score" not in round_trips.columns:
        return pd.DataFrame()
    df = round_trips.dropna(subset=["entry_rank_score"]).copy()
    if len(df) < 10 or df["entry_rank_score"].nunique() < 3:
        return pd.DataFrame()
    q = min(5, df["entry_rank_score"].nunique())
    try:
        df["rank_bucket"] = pd.qcut(
            df["entry_rank_score"],
            q,
            labels=[f"Q{i + 1}" for i in range(q)],
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()
    return _group_metrics(df, "rank_bucket", min_n=1).sort_values("rank_bucket")


def analyze(db_path: Path, *, since: str | None = None, until: str | None = None,
            run_type: str | None = None, min_n: int = 3) -> AttributionResult:
    trades = load_trade_rows(db_path, since=since, until=until, run_type=run_type)
    round_trips = build_round_trips(trades)
    summary = summarize_round_trips(round_trips)
    summary["db"] = str(db_path)
    summary["n_trade_events"] = int(len(trades))
    summary["since"] = since
    summary["until"] = until
    summary["run_type"] = run_type

    group_tables: dict[str, list[dict[str, Any]]] = {}
    for group_col in [
        "entry_order_type",
        "entry_source_job",
        "exit_source_job",
        "entry_regime",
        "exit_reason",
        "hold_bucket",
        "ticker",
    ]:
        table = _group_metrics(round_trips, group_col, min_n)
        group_tables[group_col] = table.to_dict(orient="records")

    rank_table = _rank_quantile_table(round_trips)
    group_tables["entry_rank_quantile"] = rank_table.to_dict(orient="records")
    return AttributionResult(round_trips=round_trips, summary=summary,
                             group_tables=group_tables)


def _fmt_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):+.2%}"


def _fmt_money(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"${float(value):+,.2f}"


def _print_group(name: str, rows: list[dict[str, Any]], *, top_loss: int = 8) -> None:
    print(f"\n{name}")
    if not rows:
        print("  no rows")
        return
    df = pd.DataFrame(rows).head(top_loss)
    for col in ["win_rate", "avg_net_return"]:
        if col in df:
            df[col] = df[col].map(_fmt_pct)
    for col in ["net_pnl", "tax"]:
        if col in df:
            df[col] = df[col].map(_fmt_money)
    if "profit_factor" in df:
        df["profit_factor"] = df["profit_factor"].map(
            lambda v: "NA" if v is None or pd.isna(v) else f"{float(v):.2f}"
        )
    print(df.to_string(index=False))


def print_report(result: AttributionResult) -> None:
    s = result.summary
    print("Trade Decision Attribution")
    print("=" * 78)
    print(f"DB              : {s['db']}")
    print(f"trade events    : {s['n_trade_events']}")
    print(f"round trips     : {s['n_round_trips']}")
    print(f"unmatched sells : {s.get('unmatched_sells', 0)}")
    print(f"open lots       : {s.get('open_lots', 0)}")
    print(f"win rate        : {_fmt_pct(s.get('win_rate'))}")
    print(f"avg net return  : {_fmt_pct(s.get('avg_net_return'))}")
    print(f"profit factor   : {s.get('profit_factor')}")
    print(f"gross P&L       : {_fmt_money(s.get('gross_pnl'))}")
    print(f"tax             : {_fmt_money(s.get('tax'))}")
    print(f"net P&L         : {_fmt_money(s.get('net_pnl'))}")

    print("\nWorst groups by net P&L. These are the direct APY/Sharpe levers:")
    for name, rows in result.group_tables.items():
        _print_group(name, rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/runs.alpaca.db",
                        help="SQLite trace DB. Default: data/runs.alpaca.db")
    parser.add_argument("--since", default=None, help="Inclusive start date")
    parser.add_argument("--until", default=None, help="Inclusive end date")
    parser.add_argument("--run-type", default=None,
                        help="Optional pipeline_runs.run_type filter, e.g. live or sim")
    parser.add_argument("--min-n", type=int, default=3,
                        help="Minimum round trips per group in group tables")
    parser.add_argument("--output", default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    result = analyze(db_path, since=args.since, until=args.until,
                     run_type=args.run_type, min_n=args.min_n)
    print_report(result)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": result.summary,
            "group_tables": result.group_tables,
            "round_trips_head": result.round_trips.head(100).to_dict(orient="records"),
        }
        out_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        print(f"\nWrote JSON report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
