#!/usr/bin/env python
"""Exit counterfactual replay for RenQuant trade traces.

This script answers a narrow APY / Sharpe question:

    "When an exit fired, did it improve net P&L versus simply holding longer?"

It pairs trades using `analyze_trade_decision_attribution.py`, loads the
ticker OHLCV path, and compares the actual exit with two counterfactual
lenses:

    1. fixed entry-age barriers such as exit-at-entry+20d / entry+60d
    2. post-exit continuation such as hold 20d / 60d after the actual exit

The second lens is the direct answer for exit-policy false positives. It also
applies the existing AFML triple-barrier exit meta-label helper when possible:

    meta_label = 1 -> exit was correct; price kept falling
    meta_label = 0 -> exit was likely a false positive; price recovered

The tool is read-only. Generated JSON may contain trade history; keep it local.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from scripts.analyze_trade_decision_attribution import analyze as load_attribution  # noqa: E402
from kernel.meta_label.triple_barrier import meta_label_for_exit_event  # noqa: E402


@dataclass(frozen=True)
class TaxConfig:
    short_rate: float = 0.50
    long_rate: float = 0.32
    lt_days: int = 365


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def tax_on_gain(gross_pnl: float, hold_days: int, tax: TaxConfig) -> float:
    """Approximate capital-gains tax for a counterfactual closed trade."""
    if not math.isfinite(gross_pnl) or gross_pnl <= 0:
        return 0.0
    rate = tax.long_rate if hold_days >= tax.lt_days else tax.short_rate
    return gross_pnl * rate


def load_close_series(ticker: str, data_root: Path) -> pd.Series | None:
    path = data_root / ticker / "1d.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "close" not in df.columns:
        return None
    close = df["close"].dropna().astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close


def rolling_sigma_daily(close: pd.Series, event_idx: pd.Timestamp,
                        lookback: int = 20) -> float:
    """Daily realized volatility used by the AFML exit meta-label."""
    if event_idx not in close.index:
        pos_arr = close.index.searchsorted(event_idx)
        if pos_arr >= len(close):
            return 0.01
        event_idx = close.index[pos_arr]
    pos = int(close.index.get_loc(event_idx))
    start = max(0, pos - lookback)
    rets = close.iloc[start:pos + 1].pct_change().dropna()
    sigma = float(rets.std()) if len(rets) else 0.01
    return sigma if math.isfinite(sigma) and sigma > 0 else 0.01


def _index_on_or_after(index: pd.DatetimeIndex, date: pd.Timestamp) -> int | None:
    pos = int(index.searchsorted(pd.Timestamp(date)))
    return pos if pos < len(index) else None


def close_after_bars(close: pd.Series, date: pd.Timestamp, bars: int) -> tuple[pd.Timestamp, float] | None:
    """Return close `bars` trading bars after `date`, clipped at series end."""
    start = _index_on_or_after(close.index, pd.Timestamp(date))
    if start is None:
        return None
    target = min(start + max(int(bars), 0), len(close) - 1)
    return close.index[target], float(close.iloc[target])


def path_mae_mfe(close: pd.Series, entry_date: pd.Timestamp, entry_price: float,
                 horizon: int) -> tuple[float | None, float | None]:
    start = _index_on_or_after(close.index, pd.Timestamp(entry_date))
    if start is None or not math.isfinite(entry_price) or entry_price <= 0:
        return None, None
    end = min(start + max(int(horizon), 1), len(close) - 1)
    window = close.iloc[start:end + 1]
    if window.empty:
        return None, None
    returns = window / entry_price - 1.0
    return float(returns.min()), float(returns.max())


def counterfactual_rows(
    round_trips: pd.DataFrame,
    *,
    data_root: Path,
    horizons: list[int],
    tax: TaxConfig,
    barrier_window: int,
    pt_mult: float,
    sl_mult: float,
) -> pd.DataFrame:
    """Compute per-round-trip counterfactual exits."""
    close_cache: dict[str, pd.Series | None] = {}
    rows: list[dict[str, Any]] = []
    max_h = max(horizons) if horizons else barrier_window

    for _, trip in round_trips.iterrows():
        ticker = str(trip.get("ticker") or "")
        if not ticker:
            continue
        if ticker not in close_cache:
            close_cache[ticker] = load_close_series(ticker, data_root)
        close = close_cache[ticker]
        if close is None or close.empty:
            continue

        entry_date = pd.Timestamp(trip.get("entry_date"))
        exit_date = pd.Timestamp(trip.get("exit_date"))
        entry_price = _finite(trip.get("entry_price"))
        exit_price = _finite(trip.get("exit_price"))
        entry_notional = _finite(trip.get("entry_notional"))
        actual_net_pnl = _finite(trip.get("net_pnl"))
        actual_gross_pnl = _finite(trip.get("gross_pnl"))
        if (
            entry_price is None or exit_price is None or entry_notional is None
            or actual_net_pnl is None or entry_notional <= 0
        ):
            continue

        actual_hold = int(_finite(trip.get("hold_days")) or 0)
        mae, mfe = path_mae_mfe(close, entry_date, entry_price, max_h)
        post_exit = close_after_bars(close, exit_date, barrier_window)
        if post_exit is not None:
            post_exit_date, post_exit_price = post_exit
            post_exit_return = post_exit_price / exit_price - 1.0 if exit_price > 0 else None
        else:
            post_exit_date, post_exit_price, post_exit_return = None, None, None

        sigma_daily = rolling_sigma_daily(close, exit_date)
        if _index_on_or_after(close.index, exit_date) is not None:
            event_idx = close.index[_index_on_or_after(close.index, exit_date)]
            exit_meta_label = meta_label_for_exit_event(
                close,
                event_idx=event_idx,
                event_price=exit_price,
                sigma_daily=sigma_daily,
                fwd_window=barrier_window,
                pt_mult=pt_mult,
                sl_mult=sl_mult,
            )
        else:
            exit_meta_label = None

        row = {
            "ticker": ticker,
            "entry_date": entry_date.date().isoformat(),
            "exit_date": exit_date.date().isoformat(),
            "exit_reason": trip.get("exit_reason"),
            "entry_regime": trip.get("entry_regime"),
            "entry_order_type": trip.get("entry_order_type"),
            "hold_days": actual_hold,
            "entry_notional": entry_notional,
            "actual_gross_pnl": actual_gross_pnl,
            "actual_net_pnl": actual_net_pnl,
            "actual_net_return": actual_net_pnl / entry_notional,
            "mae_to_max_horizon": mae,
            "mfe_to_max_horizon": mfe,
            "post_exit_return_to_barrier_window": post_exit_return,
            "post_exit_price_date": None if post_exit_date is None else post_exit_date.date().isoformat(),
            "exit_meta_label_correct": exit_meta_label,
            "sigma_daily_at_exit": sigma_daily,
        }

        for h in horizons:
            cf = close_after_bars(close, entry_date, h)
            if cf is None:
                row[f"hold_{h}d_net_pnl"] = None
                row[f"hold_{h}d_delta_vs_actual"] = None
                row[f"hold_{h}d_return"] = None
                continue
            cf_date, cf_price = cf
            cf_hold_days = max(int((cf_date - entry_date).days), actual_hold)
            cf_return = cf_price / entry_price - 1.0
            cf_gross_pnl = entry_notional * cf_return
            cf_tax = tax_on_gain(cf_gross_pnl, cf_hold_days, tax)
            cf_net_pnl = cf_gross_pnl - cf_tax
            row[f"hold_{h}d_date"] = cf_date.date().isoformat()
            row[f"hold_{h}d_return"] = cf_return
            row[f"hold_{h}d_net_pnl"] = cf_net_pnl
            row[f"hold_{h}d_delta_vs_actual"] = cf_net_pnl - actual_net_pnl

            post_cf = close_after_bars(close, exit_date, h)
            if post_cf is None:
                row[f"post_exit_hold_{h}d_date"] = None
                row[f"post_exit_hold_{h}d_return"] = None
                row[f"post_exit_hold_{h}d_net_pnl"] = None
                row[f"post_exit_hold_{h}d_delta_vs_actual"] = None
                continue
            post_cf_date, post_cf_price = post_cf
            post_cf_hold_days = max(int((post_cf_date - entry_date).days), actual_hold)
            post_cf_return = post_cf_price / entry_price - 1.0
            post_cf_gross_pnl = entry_notional * post_cf_return
            post_cf_tax = tax_on_gain(post_cf_gross_pnl, post_cf_hold_days, tax)
            post_cf_net_pnl = post_cf_gross_pnl - post_cf_tax
            row[f"post_exit_hold_{h}d_date"] = post_cf_date.date().isoformat()
            row[f"post_exit_hold_{h}d_return"] = post_cf_return
            row[f"post_exit_hold_{h}d_net_pnl"] = post_cf_net_pnl
            row[f"post_exit_hold_{h}d_delta_vs_actual"] = (
                post_cf_net_pnl - actual_net_pnl
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_counterfactuals(cf: pd.DataFrame, horizons: list[int],
                              min_n: int) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    if cf.empty:
        return tables

    group_cols = ["exit_reason", "entry_regime", "entry_order_type", "ticker"]
    for group_col in group_cols:
        rows = []
        for key, g in cf.groupby(group_col, dropna=False):
            if len(g) < min_n:
                continue
            record: dict[str, Any] = {
                group_col: "NULL" if pd.isna(key) else str(key),
                "n": int(len(g)),
                "actual_net_pnl": float(g["actual_net_pnl"].sum()),
                "avg_actual_net_return": float(g["actual_net_return"].mean()),
                "exit_correct_rate": (
                    float(g["exit_meta_label_correct"].dropna().mean())
                    if g["exit_meta_label_correct"].notna().any() else None
                ),
                "avg_post_exit_return": (
                    float(g["post_exit_return_to_barrier_window"].dropna().mean())
                    if g["post_exit_return_to_barrier_window"].notna().any() else None
                ),
                "avg_mae_to_max_horizon": (
                    float(g["mae_to_max_horizon"].dropna().mean())
                    if g["mae_to_max_horizon"].notna().any() else None
                ),
                "avg_mfe_to_max_horizon": (
                    float(g["mfe_to_max_horizon"].dropna().mean())
                    if g["mfe_to_max_horizon"].notna().any() else None
                ),
            }
            for h in horizons:
                delta_col = f"hold_{h}d_delta_vs_actual"
                net_col = f"hold_{h}d_net_pnl"
                if delta_col in g:
                    delta = g[delta_col].dropna()
                    record[f"hold_{h}d_delta_sum"] = float(delta.sum()) if len(delta) else None
                    record[f"hold_{h}d_better_rate"] = float((delta > 0).mean()) if len(delta) else None
                if net_col in g:
                    vals = g[net_col].dropna()
                    record[f"hold_{h}d_net_pnl"] = float(vals.sum()) if len(vals) else None
                post_delta_col = f"post_exit_hold_{h}d_delta_vs_actual"
                post_net_col = f"post_exit_hold_{h}d_net_pnl"
                if post_delta_col in g:
                    post_delta = g[post_delta_col].dropna()
                    record[f"post_exit_hold_{h}d_delta_sum"] = (
                        float(post_delta.sum()) if len(post_delta) else None
                    )
                    record[f"post_exit_hold_{h}d_better_rate"] = (
                        float((post_delta > 0).mean()) if len(post_delta) else None
                    )
                if post_net_col in g:
                    post_vals = g[post_net_col].dropna()
                    record[f"post_exit_hold_{h}d_net_pnl"] = (
                        float(post_vals.sum()) if len(post_vals) else None
                    )
            rows.append(record)
        rows = sorted(rows, key=lambda r: r.get("actual_net_pnl", 0.0))
        tables[group_col] = rows
    return tables


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):+.2%}"


def _fmt_money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"${float(value):+,.2f}"


def print_report(summary: dict[str, list[dict[str, Any]]], horizons: list[int]) -> None:
    print("Exit Counterfactual Replay")
    print("=" * 78)
    for group_name, rows in summary.items():
        print(f"\n{group_name}")
        if not rows:
            print("  no rows")
            continue
        df = pd.DataFrame(rows).head(12)
        for col in df.columns:
            if col.endswith("_rate") or col.startswith("avg_"):
                df[col] = df[col].map(_fmt_pct)
            if col.endswith("_pnl") or col.endswith("_delta_sum"):
                df[col] = df[col].map(_fmt_money)
        keep = [group_name, "n", "actual_net_pnl", "avg_actual_net_return",
                "exit_correct_rate", "avg_post_exit_return"]
        for h in horizons:
            keep.extend([f"hold_{h}d_delta_sum", f"hold_{h}d_better_rate"])
            keep.extend([
                f"post_exit_hold_{h}d_delta_sum",
                f"post_exit_hold_{h}d_better_rate",
            ])
        keep = [c for c in keep if c in df.columns]
        print(df[keep].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/sim_runs.db")
    parser.add_argument("--run-type", default="sim")
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--data-root", default="data/ohlcv")
    parser.add_argument("--horizons", default="20,60,120",
                        help="Comma-separated trading-bar horizons from entry")
    parser.add_argument("--barrier-window", type=int, default=20)
    parser.add_argument("--pt-mult", type=float, default=10.0)
    parser.add_argument("--sl-mult", type=float, default=10.0)
    parser.add_argument("--short-tax-rate", type=float, default=0.50)
    parser.add_argument("--long-tax-rate", type=float, default=0.32)
    parser.add_argument("--lt-days", type=int, default=365)
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    tax = TaxConfig(args.short_tax_rate, args.long_tax_rate, args.lt_days)

    attr = load_attribution(
        db_path,
        since=args.since,
        until=args.until,
        run_type=args.run_type,
        min_n=1,
    )
    cf = counterfactual_rows(
        attr.round_trips,
        data_root=data_root,
        horizons=horizons,
        tax=tax,
        barrier_window=args.barrier_window,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
    )
    summary = summarize_counterfactuals(cf, horizons, args.min_n)
    print_report(summary, horizons)

    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": {
                "db": str(db_path),
                "run_type": args.run_type,
                "since": args.since,
                "until": args.until,
                "horizons": horizons,
                "barrier_window": args.barrier_window,
                "pt_mult": args.pt_mult,
                "sl_mult": args.sl_mult,
                "short_tax_rate": args.short_tax_rate,
                "long_tax_rate": args.long_tax_rate,
                "lt_days": args.lt_days,
            },
            "summary": summary,
            "rows_head": cf.head(200).to_dict(orient="records"),
        }
        out.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        print(f"\nWrote JSON report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
