#!/usr/bin/env python
"""Aggregate renquant_104 WF trade traces into decision-quality forensics.

The WF gate writes one cut directory containing ``*.trades.json``,
``*.round_trips.csv`` and ``*.equity.json`` sidecars. This script rebuilds
round trips from the raw trade events with the configured tax-lot method, then
summarizes the direct APY/Sharpe levers: exit buckets, entry sources, regimes,
score monotonicity, hold time, and tax integrity.

Use this instead of ad-hoc pandas snippets when answering why a WF/sim made or
lost money.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sim_trade_ledger import round_trips_from_trade_log


REPO_ROOT = Path(__file__).resolve().parent.parent


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _tax_lot_method(config: dict[str, Any] | None, override: str | None) -> str:
    if override:
        method = override.lower()
    else:
        ja_cfg = (
            ((config or {}).get("rotation") or {})
            .get("joint_actions", {})
            or {}
        )
        method = str(ja_cfg.get("qp_tax_lot_method", "fifo")).lower()
    return method if method in {"fifo", "hifo", "avg"} else "fifo"


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _closed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    return df[df["status"].astype(str).str.lower().eq("closed")].copy()


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n": 0,
            "gross_pnl": 0.0,
            "tax": 0.0,
            "net_pnl_after_tax": 0.0,
            "win_rate": None,
            "avg_hold_days": None,
            "median_hold_days": None,
        }
    gross = df["gross_pnl"].fillna(0.0)
    net = df["net_pnl_after_tax"].fillna(0.0)
    return {
        "n": int(len(df)),
        "gross_pnl": float(gross.sum()),
        "tax": float(df["tax"].fillna(0.0).sum()) if "tax" in df.columns else 0.0,
        "net_pnl_after_tax": float(net.sum()),
        "win_rate": float((gross > 0.0).mean()),
        "avg_hold_days": (
            float(df["hold_days"].dropna().mean())
            if "hold_days" in df.columns and df["hold_days"].notna().any()
            else None
        ),
        "median_hold_days": (
            float(df["hold_days"].dropna().median())
            if "hold_days" in df.columns and df["hold_days"].notna().any()
            else None
        ),
        "avg_pnl_pct": (
            float(df["pnl_pct"].dropna().mean())
            if "pnl_pct" in df.columns and df["pnl_pct"].notna().any()
            else None
        ),
    }


def _group_table(df: pd.DataFrame, group_col: str, *, min_n: int = 1) -> list[dict[str, Any]]:
    if df.empty or group_col not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_col, dropna=False, observed=False):
        if len(group) < min_n:
            continue
        item = _summary(group)
        item[group_col] = "NULL" if pd.isna(key) else str(key)
        rows.append(item)
    return sorted(rows, key=lambda r: float(r.get("net_pnl_after_tax") or 0.0))


def _rank_deciles(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "entry_rank_score" not in df.columns:
        return []
    work = df.dropna(subset=["entry_rank_score"]).copy()
    if len(work) < 10 or work["entry_rank_score"].nunique() < 3:
        return []
    q = min(10, int(work["entry_rank_score"].nunique()))
    try:
        work["entry_rank_decile"] = pd.qcut(
            work["entry_rank_score"],
            q,
            labels=[f"D{i + 1}" for i in range(q)],
            duplicates="drop",
        )
    except ValueError:
        return []
    return _group_table(work, "entry_rank_decile")


def _score_spearman(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for score_col in ("entry_rank_score", "entry_mu", "entry_panel_score"):
        if score_col not in df.columns:
            continue
        valid = df[[score_col, "pnl_pct", "gross_pnl", "net_pnl_after_tax"]].dropna(
            subset=[score_col]
        )
        if len(valid) < 10 or valid[score_col].nunique() < 3:
            continue
        out[score_col] = {
            "n": int(len(valid)),
            "vs_pnl_pct": _safe_corr(valid[score_col], valid["pnl_pct"]),
            "vs_gross_pnl": _safe_corr(valid[score_col], valid["gross_pnl"]),
            "vs_net_pnl_after_tax": _safe_corr(valid[score_col], valid["net_pnl_after_tax"]),
        }
    return out


def _safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    valid = pd.concat([a, b], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return None
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman"))


def _tax_integrity(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    gross = df["gross_pnl"].fillna(0.0)
    tax = df["tax"].fillna(0.0) if "tax" in df.columns else pd.Series(0.0, index=df.index)
    tax_cash = (
        df["tax_cash_debited"].fillna(0.0)
        if "tax_cash_debited" in df.columns else pd.Series(0.0, index=df.index)
    )
    positive_tax_gt_gross = (gross.gt(0.0) & tax.gt(gross + 1e-9))
    losing_tax = (gross.le(0.0) & tax.gt(1e-9))
    modes = (
        df["tax_cash_debit_mode"].fillna("NULL").astype(str).value_counts().to_dict()
        if "tax_cash_debit_mode" in df.columns else {}
    )
    return {
        "tax_cash_debited": float(tax_cash.sum()),
        "tax_cash_debit_modes": modes,
        "positive_rows_with_tax_gt_gross": int(positive_tax_gt_gross.sum()),
        "positive_tax_gt_gross_excess": float((tax[positive_tax_gt_gross] - gross[positive_tax_gt_gross]).sum()),
        "losing_rows_with_positive_tax": int(losing_tax.sum()),
        "losing_rows_tax": float(tax[losing_tax].sum()),
    }


def _benchmark_ticker(config: dict[str, Any] | None) -> str:
    sleeve = (((config or {}).get("portfolio") or {}).get("benchmark_sleeve") or {})
    ticker = str(sleeve.get("ticker", "SPY") or "SPY").upper()
    return ticker


def _load_close_series(ticker: str) -> pd.Series:
    path = REPO_ROOT / "data" / "ohlcv" / ticker.upper() / "1d.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    if "close" not in df.columns:
        return pd.Series(dtype=float)
    out = pd.to_numeric(df["close"], errors="coerce").dropna()
    out.index = pd.to_datetime(out.index).normalize()
    return out.sort_index()


def _price_on_or_before(prices: pd.Series, date: Any) -> float:
    if prices.empty:
        return float("nan")
    ts = pd.Timestamp(date).normalize()
    idx = prices.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return float("nan")
    return float(prices.iloc[idx])


def _alpha_trade_mask(df: pd.DataFrame, benchmark_ticker: str) -> pd.Series:
    ticker = df.get("ticker", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    source = df.get("entry_source_job", pd.Series("", index=df.index)).fillna("").astype(str)
    is_benchmark_job = source.eq("BenchmarkSleeveJob")
    return ticker.ne(benchmark_ticker.upper()) & ~is_benchmark_job


def _with_same_capital_benchmark(
    df: pd.DataFrame,
    *,
    benchmark_prices: pd.Series,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for col in ("benchmark_pnl_same_capital", "active_net_after_tax", "active_return"):
            out[col] = pd.Series(dtype=float)
        return out
    entry_px = out["entry_date"].map(lambda d: _price_on_or_before(benchmark_prices, d))
    exit_px = out["exit_date"].map(lambda d: _price_on_or_before(benchmark_prices, d))
    shares = _numeric_series(out, "shares")
    entry_trade_px = _numeric_series(out, "entry_price")
    entry_capital = (shares * entry_trade_px).abs()
    bench_ret = (exit_px / entry_px.replace(0.0, np.nan)) - 1.0
    out["benchmark_pnl_same_capital"] = entry_capital * bench_ret
    out["active_net_after_tax"] = (
        pd.to_numeric(out.get("net_pnl_after_tax", 0.0), errors="coerce").fillna(0.0)
        - out["benchmark_pnl_same_capital"].fillna(0.0)
    )
    out["active_return"] = out["active_net_after_tax"] / entry_capital.replace(0.0, np.nan)
    return out


def _active_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n": 0,
            "gross_pnl": 0.0,
            "tax": 0.0,
            "net_pnl_after_tax": 0.0,
            "benchmark_pnl_same_capital": 0.0,
            "active_net_after_tax": 0.0,
            "gross_win_rate": None,
            "active_win_rate": None,
            "median_hold_days": None,
        }
    gross = pd.to_numeric(df["gross_pnl"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(df["net_pnl_after_tax"], errors="coerce").fillna(0.0)
    bench = pd.to_numeric(df["benchmark_pnl_same_capital"], errors="coerce").fillna(0.0)
    active = pd.to_numeric(df["active_net_after_tax"], errors="coerce").fillna(0.0)
    return {
        "n": int(len(df)),
        "gross_pnl": float(gross.sum()),
        "tax": float(_numeric_series(df, "tax").sum()),
        "net_pnl_after_tax": float(net.sum()),
        "benchmark_pnl_same_capital": float(bench.sum()),
        "active_net_after_tax": float(active.sum()),
        "gross_win_rate": float((gross > 0.0).mean()),
        "active_win_rate": float((active > 0.0).mean()),
        "median_hold_days": (
            float(pd.to_numeric(df["hold_days"], errors="coerce").dropna().median())
            if "hold_days" in df.columns and pd.to_numeric(df["hold_days"], errors="coerce").notna().any()
            else None
        ),
    }


def _active_group_table(df: pd.DataFrame, group_col: str, *, min_n: int = 1) -> list[dict[str, Any]]:
    if df.empty or group_col not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_col, dropna=False, observed=False):
        if len(group) < min_n:
            continue
        row = _active_summary(group)
        row[group_col] = "NULL" if pd.isna(key) else str(key)
        rows.append(row)
    return sorted(rows, key=lambda r: float(r.get("active_net_after_tax") or 0.0))


def _alpha_vs_benchmark(
    closed: pd.DataFrame,
    *,
    benchmark_ticker: str,
    min_group_n: int,
) -> dict[str, Any]:
    alpha = closed[_alpha_trade_mask(closed, benchmark_ticker)].copy()
    prices = _load_close_series(benchmark_ticker)
    alpha = _with_same_capital_benchmark(alpha, benchmark_prices=prices)
    return {
        "benchmark_ticker": benchmark_ticker,
        "price_source": (
            f"data/ohlcv/{benchmark_ticker}/1d.parquet"
            if not prices.empty else "missing"
        ),
        "overall": _active_summary(alpha),
        "by_cut": _active_group_table(alpha, "cut", min_n=min_group_n),
        "by_exit_reason": _active_group_table(alpha, "exit_reason", min_n=min_group_n),
        "by_entry_regime": _active_group_table(alpha, "entry_regime", min_n=min_group_n),
        "by_ticker": _active_group_table(alpha, "ticker", min_n=min_group_n),
    }


def _trace_positions_exposure(
    trace_dir: Path,
    *,
    benchmark_ticker: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for equity_path in sorted(trace_dir.glob("*.equity.json")):
        cut = equity_path.name.replace(".equity.json", "")
        trade_path = trace_dir / f"{cut}.trades.json"
        if not trade_path.exists():
            continue
        equity_payload = _load_json(equity_path)
        equity = equity_payload.get("annual_net_equity") or equity_payload.get("equity")
        if not isinstance(equity, dict) or not equity:
            continue
        trades = _load_json(trade_path)
        if not isinstance(trades, list):
            continue
        rows.append(_cut_exposure_summary(
            cut=cut,
            equity=equity,
            trades=trades,
            benchmark_ticker=benchmark_ticker,
        ))
    return rows


def _cut_exposure_summary(
    *,
    cut: str,
    equity: dict[str, Any],
    trades: list[dict[str, Any]],
    benchmark_ticker: str,
) -> dict[str, Any]:
    dates = [pd.Timestamp(d).normalize() for d in equity.keys()]
    tickers = sorted({
        str(t.get("ticker", "")).upper()
        for t in trades
        if t.get("ticker")
    })
    closes = {ticker: _load_close_series(ticker) for ticker in tickers}
    by_date: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for trade in trades:
        try:
            day = pd.Timestamp(trade.get("date")).normalize()
        except (TypeError, ValueError):
            continue
        by_date.setdefault(day, []).append(trade)

    positions = {ticker: 0.0 for ticker in tickers}
    alpha_weights: list[float] = []
    benchmark_weights: list[float] = []
    gross_weights: list[float] = []
    alpha_counts: list[int] = []
    for day in sorted(dates):
        for trade in by_date.get(day, []):
            ticker = str(trade.get("ticker", "")).upper()
            shares = _as_float(trade.get("shares"), 0.0)
            if shares <= 0.0 or not ticker:
                continue
            action = str(trade.get("action", "")).lower()
            if action == "buy":
                positions[ticker] = positions.get(ticker, 0.0) + shares
            elif action == "sell":
                positions[ticker] = max(0.0, positions.get(ticker, 0.0) - shares)
        nav = _as_float(equity.get(day.strftime("%Y-%m-%d")), float("nan"))
        if not math.isfinite(nav) or nav <= 0.0:
            continue
        benchmark_value = 0.0
        alpha_value = 0.0
        alpha_n = 0
        for ticker, shares in positions.items():
            if shares <= 0.0:
                continue
            price = _price_on_or_before(closes.get(ticker, pd.Series(dtype=float)), day)
            if not math.isfinite(price):
                continue
            value = shares * price
            if ticker == benchmark_ticker.upper():
                benchmark_value += value
            else:
                alpha_value += value
                alpha_n += 1
        alpha_w = alpha_value / nav
        bench_w = benchmark_value / nav
        alpha_weights.append(alpha_w)
        benchmark_weights.append(bench_w)
        gross_weights.append(alpha_w + bench_w)
        alpha_counts.append(alpha_n)

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    avg_gross = mean_or_none(gross_weights)
    return {
        "cut": cut,
        "avg_alpha_weight": mean_or_none(alpha_weights),
        "avg_benchmark_weight": mean_or_none(benchmark_weights),
        "avg_gross_weight": avg_gross,
        "avg_cash_weight": (1.0 - avg_gross) if avg_gross is not None else None,
        "avg_alpha_positions": mean_or_none(alpha_counts),
        "max_alpha_weight": float(np.max(alpha_weights)) if alpha_weights else None,
    }


def _cut_metrics(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.equity.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        rows.append({
            "cut": path.name.replace(".equity.json", ""),
            "event_level_apy": _as_float(payload.get("event_level_apy", payload.get("apy"))),
            "event_level_sharpe": _as_float(payload.get("event_level_sharpe", payload.get("sharpe"))),
            "annual_net_apy": _as_float(payload.get("annual_net_apy")),
            "annual_net_sharpe": _as_float(payload.get("annual_net_sharpe")),
            "annual_net_tax_estimate": _as_float(payload.get("annual_net_tax_estimate")),
            "tax_cash_debited": _as_float(payload.get("tax_cash_debited"), 0.0),
            "tax_cash_debit_mode": payload.get("tax_cash_debit_mode"),
            "max_dd": _as_float(payload.get("max_dd")),
            "annual_net_max_dd": _as_float(payload.get("annual_net_max_dd")),
        })
    return rows


def _load_round_trips(trace_dir: Path, *, lot_method: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    trade_paths = sorted(trace_dir.glob("*.trades.json"))
    if trade_paths:
        for path in trade_paths:
            trips = round_trips_from_trade_log(
                _load_json(path),
                lot_method=lot_method,
            )
            trips["cut"] = path.name.replace(".trades.json", "")
            frames.append(trips)
    else:
        for path in sorted(trace_dir.glob("*.round_trips.csv")):
            trips = pd.read_csv(path)
            trips["cut"] = path.name.replace(".round_trips.csv", "")
            frames.append(trips)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return _coerce_numeric(
        out,
        [
            "shares",
            "gross_pnl",
            "tax",
            "tax_cash_debited",
            "net_pnl_after_tax",
            "pnl_pct",
            "hold_days",
            "entry_rank_score",
            "entry_mu",
            "entry_sigma",
            "entry_panel_score",
        ],
    )


def analyze_trace(
    trace_dir: Path,
    *,
    config: dict[str, Any] | None = None,
    lot_method_override: str | None = None,
    min_group_n: int = 1,
) -> dict[str, Any]:
    trace_dir = trace_dir.resolve()
    lot_method = _tax_lot_method(config, lot_method_override)
    benchmark_ticker = _benchmark_ticker(config)
    trips = _load_round_trips(trace_dir, lot_method=lot_method)
    closed = _closed(trips)
    payload = {
        "trace_dir": str(trace_dir),
        "tax_lot_method": lot_method,
        "benchmark_ticker": benchmark_ticker,
        "cut_metrics": _cut_metrics(trace_dir),
        "exposure": _trace_positions_exposure(
            trace_dir,
            benchmark_ticker=benchmark_ticker,
        ),
        "overall": _summary(closed),
        "alpha_vs_benchmark": _alpha_vs_benchmark(
            closed,
            benchmark_ticker=benchmark_ticker,
            min_group_n=min_group_n,
        ),
        "tax_integrity": _tax_integrity(closed),
        "score_spearman": _score_spearman(closed),
        "groups": {
            "by_cut": _group_table(closed, "cut", min_n=min_group_n),
            "by_exit_reason": _group_table(closed, "exit_reason", min_n=min_group_n),
            "by_entry_regime": _group_table(closed, "entry_regime", min_n=min_group_n),
            "by_exit_regime": _group_table(closed, "exit_regime", min_n=min_group_n),
            "by_entry_source_job": _group_table(closed, "entry_source_job", min_n=min_group_n),
            "by_exit_source_job": _group_table(closed, "exit_source_job", min_n=min_group_n),
            "by_ticker": _group_table(closed, "ticker", min_n=min_group_n),
            "by_entry_rank_decile": _rank_deciles(closed),
        },
        "worst_round_trips": _json_ready(
            closed.sort_values("net_pnl_after_tax").head(25).to_dict(orient="records")
        ),
        "best_round_trips": _json_ready(
            closed.sort_values("net_pnl_after_tax", ascending=False).head(15).to_dict(orient="records")
        ),
        "n_rows": {
            "round_trips": int(len(trips)),
            "closed": int(len(closed)),
            "open": int(
                trips["status"].astype(str).str.lower().eq("open").sum()
            ) if "status" in trips.columns else 0,
        },
    }
    return _json_ready(payload)


def _fmt_money(value: Any) -> str:
    if value is None:
        return "NA"
    return f"${float(value):+,.0f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.2%}"


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# WF Trade Forensics",
        "",
        f"- trace_dir: `{payload['trace_dir']}`",
        f"- tax_lot_method: `{payload['tax_lot_method']}`",
        f"- benchmark_ticker: `{payload['benchmark_ticker']}`",
        f"- rows: {payload['n_rows']}",
        "",
        "## Overall",
    ]
    overall = payload["overall"]
    lines.extend([
        f"- closed_round_trips: {overall['n']}",
        f"- gross_pnl: {_fmt_money(overall['gross_pnl'])}",
        f"- tax_estimate: {_fmt_money(overall['tax'])}",
        f"- net_pnl_after_tax: {_fmt_money(overall['net_pnl_after_tax'])}",
        f"- win_rate: {_fmt_pct(overall['win_rate'])}",
        f"- median_hold_days: {overall['median_hold_days']}",
        "",
        "## Cut Metrics",
    ])
    if payload["cut_metrics"]:
        lines.append(pd.DataFrame(payload["cut_metrics"]).to_markdown(index=False, floatfmt=".4f"))
    else:
        lines.append("No equity sidecars found.")
    lines.extend(["", "## Exposure"])
    if payload["exposure"]:
        lines.append(pd.DataFrame(payload["exposure"]).to_markdown(index=False, floatfmt=".4f"))
    else:
        lines.append("No exposure rows.")
    lines.extend(["", "## Alpha vs Benchmark"])
    avb = payload["alpha_vs_benchmark"]
    lines.extend([
        f"- benchmark: `{avb['benchmark_ticker']}`",
        f"- price_source: `{avb['price_source']}`",
    ])
    lines.append(pd.DataFrame([avb["overall"]]).to_markdown(index=False, floatfmt=".4f"))
    for key in ("by_cut", "by_exit_reason", "by_entry_regime", "by_ticker"):
        lines.extend(["", f"### alpha_vs_benchmark.{key}"])
        rows = avb.get(key, [])
        if rows:
            lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
        else:
            lines.append("No rows.")
    lines.extend(["", "## Tax Integrity"])
    lines.append(pd.DataFrame([payload["tax_integrity"]]).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Score Monotonicity"])
    if payload["score_spearman"]:
        lines.append(pd.DataFrame.from_dict(payload["score_spearman"], orient="index").to_markdown(floatfmt=".4f"))
    else:
        lines.append("Insufficient scored closed trades.")

    for key, rows in payload["groups"].items():
        lines.extend(["", f"## {key}"])
        if rows:
            lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
        else:
            lines.append("No rows.")

    lines.extend(["", "## Worst Round Trips"])
    worst = pd.DataFrame(payload["worst_round_trips"])
    if not worst.empty:
        cols = [
            "cut", "ticker", "entry_date", "exit_date", "entry_regime",
            "exit_regime", "exit_reason", "gross_pnl", "tax",
            "net_pnl_after_tax", "pnl_pct", "hold_days", "entry_rank_score",
            "entry_mu", "entry_sigma", "entry_source_job", "exit_source_job",
        ]
        lines.append(worst[[c for c in cols if c in worst.columns]].to_markdown(index=False, floatfmt=".4f"))
    else:
        lines.append("No rows.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", help="WF trace directory")
    parser.add_argument("--config", default=None, help="Strategy config JSON used by the sim")
    parser.add_argument("--lot-method", choices=["fifo", "hifo", "avg"], default=None)
    parser.add_argument("--min-group-n", type=int, default=1)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    if not trace_dir.is_absolute():
        trace_dir = REPO_ROOT / trace_dir
    config = _load_json(Path(args.config)) if args.config else None
    payload = analyze_trace(
        trace_dir,
        config=config,
        lot_method_override=args.lot_method,
        min_group_n=args.min_group_n,
    )

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if args.md_out:
        out = Path(args.md_out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown_report(payload))
    if not args.json_out and not args.md_out:
        print(markdown_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
