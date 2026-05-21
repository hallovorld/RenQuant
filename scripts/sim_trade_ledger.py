#!/usr/bin/env python
"""Trade-ledger utilities for renquant_104 simulations.

The sim engine already returns ``SimResult.trade_log`` in memory. This module
turns that volatile list into durable audit artifacts: raw trade events,
FIFO-matched round trips, and a compact forensic report.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd


REGIME_THESES = {
    "BULL_CALM": (
        "Trend/momentum continuation: cross-sectional rank should select "
        "relative winners while market volatility is benign."
    ),
    "BULL_VOLATILE": (
        "Risk-managed upside: score must compensate for higher market "
        "volatility and faster drawdown risk."
    ),
    "CHOPPY": (
        "Relative-strength/divergence: entry needs stock-specific strength "
        "because broad beta is less reliable."
    ),
    "BEAR": (
        "Capital preservation: offensive buys are suspect unless the ticker "
        "is explicitly defensive."
    ),
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_date(value: Any) -> str:
    if value is None:
        return ""
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:  # noqa: BLE001
        return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def trade_log_frame(trade_log: list[dict]) -> pd.DataFrame:
    """Return a stable raw-event DataFrame sorted by event date."""
    rows = []
    for idx, row in enumerate(trade_log or []):
        r = dict(row)
        r["event_id"] = idx
        r["date"] = _as_date(r.get("date"))
        rows.append(r)
    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df = df.sort_values(["date", "event_id"]).reset_index(drop=True)
    return df


def _equity_regime_map(result: Any) -> dict[str, str]:
    equity = getattr(result, "equity_df", None)
    if equity is None or getattr(equity, "empty", True) or "regime" not in equity.columns:
        return {}
    out: dict[str, str] = {}
    for idx, regime in equity["regime"].items():
        if regime is not None and regime == regime:
            out[_as_date(idx)] = str(regime)
    return out


def _enrich_trade_log_from_result(result: Any) -> list[dict]:
    """Fill audit-only fields that older order emitters may omit."""
    regime_by_date = _equity_regime_map(result)
    enriched: list[dict] = []
    for event in list(getattr(result, "trade_log", []) or []):
        row = dict(event)
        if row.get("action") == "buy" and not row.get("regime"):
            row["regime"] = regime_by_date.get(_as_date(row.get("date")))
        enriched.append(row)
    return enriched


def round_trips_from_trade_log(
    trade_log: list[dict],
    *,
    end_prices: dict[str, float] | None = None,
) -> pd.DataFrame:
    """FIFO-match long buys to long sells.

    The simulator can top up and partially trim positions. Per-trade sell rows
    contain event-level realized P&L, but root-cause analysis needs entry-side
    fields (regime, rank_score, mu/sigma) joined to each realized exit. FIFO is
    deliberately conservative and transparent for this diagnostic ledger; tax
    lots may use HIFO internally, so tax attribution is allocated
    proportionally from the simulator's sell event.
    """
    lots: dict[str, deque[dict]] = defaultdict(deque)
    rows: list[dict] = []

    for event_id, event in enumerate(trade_log or []):
        action = event.get("action")
        ticker = str(event.get("ticker") or "")
        if not ticker:
            continue
        if action == "buy":
            shares = _as_float(event.get("shares"))
            if shares <= 0:
                continue
            lots[ticker].append({
                "event_id": event_id,
                "ticker": ticker,
                "entry_date": _as_date(event.get("date")),
                "entry_price": _as_float(event.get("price")),
                "remaining_shares": shares,
                "entry_invest": _as_float(event.get("invest")),
                "entry_regime": event.get("regime"),
                "entry_rank_score": event.get("rank_score"),
                "entry_rs_score": event.get("rs_score"),
                "entry_mu": event.get("mu"),
                "entry_sigma": event.get("sigma"),
                "entry_sigma_mult": event.get("sigma_mult"),
            })
            continue

        if action != "sell":
            continue
        sell_shares = _as_float(event.get("shares"))
        if sell_shares <= 0:
            continue
        remaining = sell_shares
        exit_price = _as_float(event.get("price"))
        event_tax = _as_float(event.get("tax"))
        while remaining > 1e-9 and lots[ticker]:
            lot = lots[ticker][0]
            take = min(remaining, lot["remaining_shares"])
            entry_price = _as_float(lot.get("entry_price"))
            gross_pnl = (exit_price - entry_price) * take
            entry_value = entry_price * take
            pnl_pct = gross_pnl / entry_value if entry_value > 0 else 0.0
            tax_alloc = event_tax * (take / sell_shares) if sell_shares > 0 else 0.0
            entry_date = lot.get("entry_date", "")
            exit_date = _as_date(event.get("date"))
            rows.append({
                "status": "closed",
                "ticker": ticker,
                "entry_event_id": lot.get("event_id"),
                "exit_event_id": event_id,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "hold_days": (
                    pd.Timestamp(exit_date) - pd.Timestamp(entry_date)
                ).days if entry_date and exit_date else event.get("hold_days"),
                "shares": take,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "tax": tax_alloc,
                "net_pnl_after_tax": gross_pnl - tax_alloc,
                "pnl_pct": pnl_pct,
                "sim_sell_pnl_pct": event.get("pnl_pct"),
                "exit_reason": event.get("exit_reason"),
                "partial_exit": bool(event.get("partial", False)),
                "entry_regime": lot.get("entry_regime"),
                "entry_rank_score": lot.get("entry_rank_score"),
                "entry_rs_score": lot.get("entry_rs_score"),
                "entry_mu": lot.get("entry_mu"),
                "entry_sigma": lot.get("entry_sigma"),
                "entry_sigma_mult": lot.get("entry_sigma_mult"),
            })
            lot["remaining_shares"] -= take
            remaining -= take
            if lot["remaining_shares"] <= 1e-9:
                lots[ticker].popleft()

        if remaining > 1e-9:
            rows.append({
                "status": "unmatched_sell",
                "ticker": ticker,
                "entry_event_id": None,
                "exit_event_id": event_id,
                "entry_date": "",
                "exit_date": _as_date(event.get("date")),
                "hold_days": event.get("hold_days"),
                "shares": remaining,
                "entry_price": None,
                "exit_price": exit_price,
                "gross_pnl": None,
                "tax": event_tax * (remaining / sell_shares),
                "net_pnl_after_tax": None,
                "pnl_pct": event.get("pnl_pct"),
                "sim_sell_pnl_pct": event.get("pnl_pct"),
                "exit_reason": event.get("exit_reason"),
                "partial_exit": bool(event.get("partial", False)),
            })

    end_prices = end_prices or {}
    for ticker, q in lots.items():
        mark = _as_float(end_prices.get(ticker), default=float("nan"))
        for lot in q:
            shares = _as_float(lot.get("remaining_shares"))
            entry_price = _as_float(lot.get("entry_price"))
            gross_pnl = (
                (mark - entry_price) * shares
                if math.isfinite(mark) and entry_price > 0 else None
            )
            rows.append({
                "status": "open",
                "ticker": ticker,
                "entry_event_id": lot.get("event_id"),
                "exit_event_id": None,
                "entry_date": lot.get("entry_date"),
                "exit_date": "",
                "hold_days": None,
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": mark if math.isfinite(mark) else None,
                "gross_pnl": gross_pnl,
                "tax": 0.0,
                "net_pnl_after_tax": gross_pnl,
                "pnl_pct": (
                    gross_pnl / (entry_price * shares)
                    if gross_pnl is not None and entry_price > 0 and shares > 0
                    else None
                ),
                "sim_sell_pnl_pct": None,
                "exit_reason": "open",
                "partial_exit": False,
                "entry_regime": lot.get("entry_regime"),
                "entry_rank_score": lot.get("entry_rank_score"),
                "entry_rs_score": lot.get("entry_rs_score"),
                "entry_mu": lot.get("entry_mu"),
                "entry_sigma": lot.get("entry_sigma"),
                "entry_sigma_mult": lot.get("entry_sigma_mult"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["entry_thesis"] = df["entry_regime"].map(REGIME_THESES).fillna(
        "No regime thesis recorded on the buy event."
    )
    df["outcome"] = df["gross_pnl"].apply(
        lambda x: "win" if _as_float(x) > 0 else ("loss" if _as_float(x) < 0 else "flat")
    )
    return df.sort_values(["entry_date", "exit_date", "ticker"]).reset_index(drop=True)


def build_forensic_report(
    *,
    raw_trades: pd.DataFrame,
    round_trips: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any] | None = None,
    title: str = "Sim Trade Forensics",
) -> str:
    """Build a Markdown report summarizing how the sim made or lost money."""
    lines: list[str] = [f"# {title}", ""]
    lines.append("## Run Metrics")
    for key in ("config", "start", "end", "final_value", "total_return",
                "apy", "sharpe", "max_dd", "win_rate", "n_buys", "n_sells"):
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    if config:
        ranking = (config.get("ranking") or {}).get("panel_scoring") or {}
        lines.append(f"- panel_buy_floor: {ranking.get('buy_floor')}")
        lines.append(f"- panel_artifact_path: {ranking.get('artifact_path')}")
        wf = config.get("walkforward") or {}
        if wf.get("enabled"):
            lines.append(f"- walkforward_manifest: {wf.get('manifest_path')}")
    lines.append("")

    lines.append("## Theoretical Frame")
    lines.append(
        "A long-only cross-sectional rank strategy should earn money only if "
        "entry scores have positive realized cross-sectional information "
        "coefficient, the regime label matches the regime-specific thesis, and "
        "sizing/exits do not convert alpha into tax or drawdown drag."
    )
    lines.append(
        "For each round trip below, the entry thesis is derived from the buy "
        "event's regime. A losing closed trade means the entry thesis failed, "
        "the exit came too late, sizing was too large, or the regime label was "
        "not economically correct for that bar."
    )
    lines.append("")

    if raw_trades.empty:
        lines.append("No trade events recorded.")
        return "\n".join(lines) + "\n"

    closed = round_trips[round_trips.get("status", "") == "closed"].copy() if not round_trips.empty else pd.DataFrame()
    open_rows = round_trips[round_trips.get("status", "") == "open"].copy() if not round_trips.empty else pd.DataFrame()

    lines.append("## Attribution")
    if closed.empty:
        lines.append("No closed round trips.")
    else:
        total_gross = float(closed["gross_pnl"].fillna(0).sum())
        total_tax = float(closed["tax"].fillna(0).sum())
        total_net = float(closed["net_pnl_after_tax"].fillna(0).sum())
        lines.append(f"- closed_round_trips: {len(closed)}")
        lines.append(f"- gross_pnl: {total_gross:+.2f}")
        lines.append(f"- tax: {total_tax:+.2f}")
        lines.append(f"- net_pnl_after_tax: {total_net:+.2f}")
        lines.append(
            f"- win_rate_closed: {closed['gross_pnl'].gt(0).mean():.2%}"
        )
        lines.append("")

        for group_col in ("entry_regime", "exit_reason", "ticker"):
            grp = (
                closed.groupby(group_col, dropna=False)
                .agg(
                    n=("ticker", "size"),
                    gross_pnl=("gross_pnl", "sum"),
                    net_pnl_after_tax=("net_pnl_after_tax", "sum"),
                    mean_pnl_pct=("pnl_pct", "mean"),
                    win_rate=("gross_pnl", lambda s: float((s > 0).mean())),
                    median_hold_days=("hold_days", "median"),
                )
                .sort_values("net_pnl_after_tax")
            )
            lines.append(f"### By {group_col}")
            lines.append(grp.to_markdown(floatfmt=".4f"))
            lines.append("")

        worst_cols = [
            "ticker", "entry_date", "exit_date", "entry_regime", "exit_reason",
            "shares", "entry_price", "exit_price", "gross_pnl",
            "net_pnl_after_tax", "pnl_pct", "hold_days", "entry_rank_score",
            "entry_mu", "entry_sigma",
        ]
        lines.append("### Worst 25 Closed Round Trips")
        lines.append(closed.sort_values("net_pnl_after_tax").head(25)[worst_cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        lines.append("### Best 15 Closed Round Trips")
        lines.append(closed.sort_values("net_pnl_after_tax", ascending=False).head(15)[worst_cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    if not open_rows.empty:
        cols = [
            "ticker", "entry_date", "entry_regime", "shares", "entry_price",
            "exit_price", "gross_pnl", "pnl_pct", "entry_rank_score",
            "entry_mu", "entry_sigma",
        ]
        lines.append("## Open Lots At End")
        lines.append(open_rows[cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("## Full Ledger Location")
    lines.append(
        "The CSV sidecars contain every raw event and every FIFO-matched "
        "round trip; this report only shows the most diagnostic slices."
    )
    return "\n".join(lines) + "\n"


def write_trade_outputs(
    *,
    result: Any,
    config: dict[str, Any] | None = None,
    trade_json: str | Path | None = None,
    trade_csv: str | Path | None = None,
    round_trips_csv: str | Path | None = None,
    report_md: str | Path | None = None,
    end_prices: dict[str, float] | None = None,
    title: str = "Sim Trade Forensics",
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write requested trade-ledger artifacts and return written paths."""
    trade_log = _enrich_trade_log_from_result(result)
    raw = trade_log_frame(trade_log)
    trips = round_trips_from_trade_log(trade_log, end_prices=end_prices)
    metrics = {
        "final_value": float(getattr(result, "final_value", 0.0)),
        "total_return": float(getattr(result, "total_return", 0.0)),
        "apy": float(getattr(result, "apy", 0.0)),
        "sharpe": float(getattr(result, "sharpe", float("nan"))),
        "max_dd": float(getattr(result, "max_dd", float("nan"))),
        "win_rate": float(getattr(result, "win_rate", 0.0)),
        "n_buys": len(getattr(result, "buys", []) or []),
        "n_sells": len(getattr(result, "sells", []) or []),
    }
    metrics.update(extra_metrics or {})

    written: dict[str, str] = {}
    if trade_json:
        p = Path(trade_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_json_safe(raw.to_dict(orient="records")), indent=2))
        written["trade_json"] = str(p)
    if trade_csv:
        p = Path(trade_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(p, index=False)
        written["trade_csv"] = str(p)
    if round_trips_csv:
        p = Path(round_trips_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        trips.to_csv(p, index=False)
        written["round_trips_csv"] = str(p)
    if report_md:
        p = Path(report_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            build_forensic_report(
                raw_trades=raw,
                round_trips=trips,
                metrics=metrics,
                config=config,
                title=title,
            )
        )
        written["report_md"] = str(p)
    return written
