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
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sim_trade_ledger import round_trips_from_trade_log


REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.meta_label.triple_barrier import apply_triple_barrier  # noqa: E402


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


def _entry_score_ladder(
    closed: pd.DataFrame,
    *,
    benchmark_ticker: str,
    min_group_n: int = 5,
    q: int = 5,
) -> list[dict[str, Any]]:
    """Regime-first score buckets measured against same-capital benchmark P&L.

    This is the direct alpha-conversion lens: within each regime, higher model
    score buckets should show better active outcomes versus the benchmark. If a
    ladder slopes the wrong way, the failure is in entry score semantics or a
    downstream decision rule that systematically harvests the wrong side.
    """
    if closed.empty or "entry_regime" not in closed.columns:
        return []
    alpha = closed[_alpha_trade_mask(closed, benchmark_ticker)].copy()
    if alpha.empty:
        return []
    alpha = _with_same_capital_benchmark(
        alpha,
        benchmark_prices=_load_close_series(benchmark_ticker),
    )
    rows: list[dict[str, Any]] = []
    for regime, regime_df in alpha.groupby("entry_regime", dropna=False, observed=False):
        regime_name = "NULL" if pd.isna(regime) else str(regime)
        for score_col in ("entry_rank_score", "entry_panel_score", "entry_mu"):
            if score_col not in regime_df.columns:
                continue
            work = regime_df.copy()
            work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
            work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col])
            if len(work) < min_group_n or work[score_col].nunique() < 2:
                continue
            n_bins = min(max(2, int(q)), int(work[score_col].nunique()), len(work))
            try:
                buckets = pd.qcut(
                    work[score_col],
                    q=n_bins,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                continue
            work = work.assign(_score_bucket=buckets)
            for code, group in work.groupby("_score_bucket", dropna=True, observed=False):
                if len(group) < min_group_n:
                    continue
                summary = _active_summary(group)
                score = pd.to_numeric(group[score_col], errors="coerce")
                exit_mix = (
                    group.get("exit_reason", pd.Series("NULL", index=group.index))
                    .fillna("NULL")
                    .astype(str)
                    .value_counts()
                    .head(3)
                    .to_dict()
                )
                rows.append({
                    "entry_regime": regime_name,
                    "score_col": score_col,
                    "score_bucket": f"Q{int(code) + 1}",
                    "min_score": float(score.min()),
                    "median_score": float(score.median()),
                    "max_score": float(score.max()),
                    "mean_active_return": (
                        float(pd.to_numeric(group["active_return"], errors="coerce").dropna().mean())
                        if "active_return" in group.columns
                        and pd.to_numeric(group["active_return"], errors="coerce").notna().any()
                        else None
                    ),
                    "top_exit_reasons": exit_mix,
                    **summary,
                })
    return rows


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


def _score_spearman_by_group(
    df: pd.DataFrame,
    group_col: str,
    *,
    min_n: int = 10,
) -> list[dict[str, Any]]:
    """Score/outcome monotonicity by regime or other audit bucket.

    A regime-conditional strategy cannot be promoted from a single pooled
    Spearman. Keep the grouped rows long-form so markdown/JSON consumers can
    sort and filter without parsing nested dicts.
    """
    if df.empty or group_col not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_col, dropna=False, observed=False):
        if len(group) < min_n:
            continue
        corr = _score_spearman(group)
        for score_col, values in corr.items():
            rows.append({
                group_col: "NULL" if pd.isna(key) else str(key),
                "score_col": score_col,
                **values,
            })
    return rows


def _with_entry_exit_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    entry = out.get("entry_regime", pd.Series("NULL", index=out.index))
    exit_ = out.get("exit_regime", pd.Series("NULL", index=out.index))
    out["entry_exit_regime"] = (
        entry.fillna("NULL").astype(str)
        + "->"
        + exit_.fillna("NULL").astype(str)
    )
    return out


def _entry_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    required = {"cut", "entry_event_id", "ticker", "entry_date"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    def _join_unique(values: pd.Series) -> str:
        uniq = sorted({str(v) for v in values.dropna().tolist()})
        return "|".join(uniq)

    first_cols = [
        "entry_regime",
        "entry_rank_score",
        "entry_panel_score",
        "entry_mu",
        "entry_sigma",
        "entry_expected_return",
        "entry_source_job",
    ]
    agg: dict[str, Any] = {
        "shares": "sum",
        "gross_pnl": "sum",
        "tax": "sum",
        "net_pnl_after_tax": "sum",
        "hold_days": "mean",
        "pnl_pct": "mean",
        "exit_reason": _join_unique,
    }
    for col in first_cols:
        if col in df.columns:
            agg[col] = "first"
    events = (
        df.groupby(["cut", "entry_event_id", "ticker", "entry_date"], dropna=False)
        .agg(agg)
        .reset_index()
    )
    events["entry_date"] = pd.to_datetime(events["entry_date"], errors="coerce")
    return events


def _load_close(root: Path, ticker: str) -> pd.Series | None:
    p = root / ticker / "1d.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=["close"])
    except Exception:
        return None
    if "close" not in df.columns:
        return None
    s = pd.to_numeric(df["close"], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _forward_return(close: pd.Series | None, date: Any, horizon: int) -> float:
    if close is None or close.empty:
        return float("nan")
    ts = pd.Timestamp(date)
    idx = close.index.searchsorted(ts, side="left")
    end = idx + int(horizon)
    if idx >= len(close) or end >= len(close):
        return float("nan")
    base = float(close.iloc[idx])
    future = float(close.iloc[end])
    if base <= 0 or not math.isfinite(base) or not math.isfinite(future):
        return float("nan")
    return future / base - 1.0


def _forward_return_alignment(
    closed: pd.DataFrame,
    *,
    ohlcv_root: Path | None,
    benchmark_ticker: str,
    horizons: tuple[int, ...] = (20, 60, 120),
    min_n: int = 10,
) -> dict[str, Any]:
    if ohlcv_root is None:
        return {"enabled": False, "reason": "ohlcv_root_not_set"}
    events = _entry_events(closed)
    if events.empty:
        return {"enabled": True, "reason": "no_entry_events", "overall": []}
    benchmark = _load_close(ohlcv_root, benchmark_ticker)
    close_cache: dict[str, pd.Series | None] = {benchmark_ticker: benchmark}
    for ticker in events["ticker"].dropna().astype(str).unique():
        close_cache.setdefault(ticker, _load_close(ohlcv_root, ticker))

    enriched = events.copy()
    for h in horizons:
        raw_vals: list[float] = []
        bench_vals: list[float] = []
        excess_vals: list[float] = []
        for row in enriched.itertuples(index=False):
            stock_ret = _forward_return(
                close_cache.get(str(row.ticker)),
                row.entry_date,
                h,
            )
            bench_ret = _forward_return(benchmark, row.entry_date, h)
            raw_vals.append(stock_ret)
            bench_vals.append(bench_ret)
            excess_vals.append(
                stock_ret - bench_ret
                if math.isfinite(stock_ret) and math.isfinite(bench_ret)
                else float("nan")
            )
        enriched[f"fwd_{h}d_return"] = raw_vals
        enriched[f"fwd_{h}d_{benchmark_ticker.lower()}"] = bench_vals
        enriched[f"fwd_{h}d_excess_{benchmark_ticker.lower()}"] = excess_vals

    def _rows(group: pd.DataFrame, group_name: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for score_col in ("entry_rank_score", "entry_panel_score", "entry_mu"):
            if score_col not in group.columns:
                continue
            for h in horizons:
                y_col = f"fwd_{h}d_excess_{benchmark_ticker.lower()}"
                d = group[[score_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(d) < min_n or d[score_col].nunique() < 3 or d[y_col].nunique() < 3:
                    continue
                from scipy.stats import spearmanr  # noqa: PLC0415
                rho, _ = spearmanr(d[score_col], d[y_col])
                item = {
                    "score_col": score_col,
                    "horizon_days": int(h),
                    "n": int(len(d)),
                    "spearman_vs_forward_excess": float(rho),
                    "mean_forward_excess": float(d[y_col].mean()),
                }
                if group_name is not None:
                    item["entry_regime"] = group_name
                out.append(item)
        return out

    by_regime: list[dict[str, Any]] = []
    if "entry_regime" in enriched.columns:
        for regime, group in enriched.groupby("entry_regime", dropna=False, observed=False):
            by_regime.extend(
                _rows(group, "NULL" if pd.isna(regime) else str(regime))
            )

    return {
        "enabled": True,
        "benchmark_ticker": benchmark_ticker,
        "ohlcv_root": str(ohlcv_root),
        "n_entry_events": int(len(enriched)),
        "overall": _rows(enriched),
        "by_entry_regime": by_regime,
    }


def _realized_sigma_daily(
    close: pd.Series | None,
    date: Any,
    *,
    window: int = 20,
    default: float = 0.01,
) -> float:
    """Daily realized volatility known at ``date``.

    This is a diagnostic helper for exit-path labeling. It intentionally uses
    only returns up to the exit bar, so the triple-barrier width never sees the
    future path it is labeling.
    """
    if close is None or close.empty:
        return float(default)
    idx = close.index.searchsorted(pd.Timestamp(date), side="right")
    if idx <= 1:
        return float(default)
    rets = close.iloc[:idx].pct_change().dropna().tail(int(window))
    if len(rets) < 5:
        return float(default)
    sigma = float(rets.std())
    return sigma if math.isfinite(sigma) and sigma > 0 else float(default)


def _exit_path_audit(
    closed: pd.DataFrame,
    *,
    ohlcv_root: Path | None,
    benchmark_ticker: str,
    horizons: tuple[int, ...] = (20, 60, 120),
    barrier_window_days: int = 60,
    pt_mult: float = 10.0,
    sl_mult: float = 10.0,
    min_n: int = 1,
) -> dict[str, Any]:
    """Audit whether realized exits were path-correct after the sell.

    The triple-barrier lens follows AFML's path-dependent labeling: from the
    exit bar, did price hit a lower volatility-scaled barrier first
    (exit was correct), or recover / finish positive first (exit was likely a
    false-positive path exit)?  This is diagnostic evidence, not a trading
    rule by itself.
    """
    if ohlcv_root is None:
        return {"enabled": False, "reason": "ohlcv_root_not_set"}
    if closed.empty:
        return {"enabled": True, "reason": "no_closed_trades", "overall": {}}
    closed = closed[_alpha_trade_mask(closed, benchmark_ticker)].copy()
    if closed.empty:
        return {"enabled": True, "reason": "no_alpha_closed_trades", "overall": {}}

    benchmark = _load_close(ohlcv_root, benchmark_ticker)
    close_cache: dict[str, pd.Series | None] = {benchmark_ticker: benchmark}
    for ticker in closed["ticker"].dropna().astype(str).unique():
        close_cache.setdefault(ticker, _load_close(ohlcv_root, ticker))

    rows: list[dict[str, Any]] = []
    for row in closed.itertuples(index=False):
        ticker = str(getattr(row, "ticker", "")).upper()
        exit_date = getattr(row, "exit_date", None)
        exit_price = _as_float(getattr(row, "exit_price", None), float("nan"))
        close = close_cache.get(ticker)
        barrier_label = None
        barrier_date = None
        barrier_price = None
        exit_correct = None
        sigma_daily = _realized_sigma_daily(close, exit_date)
        if close is not None and math.isfinite(exit_price):
            tb = apply_triple_barrier(
                close,
                entry_idx=pd.Timestamp(exit_date),
                entry_price=exit_price,
                pt_mult=float(pt_mult),
                sl_mult=float(sl_mult),
                sigma_daily=sigma_daily,
                max_horizon_days=int(barrier_window_days),
                return_terminal_sign=True,
            )
            if tb is not None:
                barrier_label, barrier_date, barrier_price = tb
                # After a sell, lower-first means the exit avoided further loss.
                exit_correct = 1 if int(barrier_label) == -1 else 0

        item = {
            "cut": getattr(row, "cut", None),
            "ticker": ticker,
            "entry_date": getattr(row, "entry_date", None),
            "exit_date": exit_date,
            "entry_regime": getattr(row, "entry_regime", None),
            "exit_regime": getattr(row, "exit_regime", None),
            "entry_exit_regime": getattr(row, "entry_exit_regime", None),
            "exit_reason": getattr(row, "exit_reason", None),
            "gross_pnl": _as_float(getattr(row, "gross_pnl", None), 0.0),
            "net_pnl_after_tax": _as_float(getattr(row, "net_pnl_after_tax", None), 0.0),
            "pnl_pct": _as_float(getattr(row, "pnl_pct", None), float("nan")),
            "hold_days": _as_float(getattr(row, "hold_days", None), float("nan")),
            "entry_rank_score": _as_float(getattr(row, "entry_rank_score", None), float("nan")),
            "entry_mu": _as_float(getattr(row, "entry_mu", None), float("nan")),
            "entry_sigma": _as_float(getattr(row, "entry_sigma", None), float("nan")),
            "exit_price": exit_price,
            "path_sigma_daily": sigma_daily,
            "barrier_window_days": int(barrier_window_days),
            "barrier_label": barrier_label,
            "barrier_date": barrier_date,
            "barrier_price": barrier_price,
            "exit_correct_by_barrier": exit_correct,
        }
        for h in horizons:
            stock_ret = _forward_return(close, exit_date, h)
            bench_ret = _forward_return(benchmark, exit_date, h)
            item[f"post_exit_{h}d_return"] = stock_ret
            item[f"post_exit_{h}d_{benchmark_ticker.lower()}"] = bench_ret
            item[f"post_exit_{h}d_excess_{benchmark_ticker.lower()}"] = (
                stock_ret - bench_ret
                if math.isfinite(stock_ret) and math.isfinite(bench_ret)
                else float("nan")
            )
        rows.append(item)

    audit = pd.DataFrame(rows)
    if audit.empty:
        return {"enabled": True, "reason": "no_auditable_exits", "overall": {}}

    def _path_summary(group: pd.DataFrame) -> dict[str, Any]:
        correct = pd.to_numeric(group["exit_correct_by_barrier"], errors="coerce")
        out: dict[str, Any] = {
            "n": int(len(group)),
            "labeled_n": int(correct.notna().sum()),
            "barrier_correct_exit_rate": (
                float(correct.dropna().mean()) if correct.notna().any() else None
            ),
            "barrier_false_positive_rate": (
                float((1.0 - correct.dropna()).mean()) if correct.notna().any() else None
            ),
            "gross_pnl": float(pd.to_numeric(group["gross_pnl"], errors="coerce").fillna(0.0).sum()),
            "net_pnl_after_tax": float(
                pd.to_numeric(group["net_pnl_after_tax"], errors="coerce").fillna(0.0).sum()
            ),
            "median_hold_days": (
                float(pd.to_numeric(group["hold_days"], errors="coerce").dropna().median())
                if pd.to_numeric(group["hold_days"], errors="coerce").notna().any()
                else None
            ),
        }
        for h in horizons:
            raw_col = f"post_exit_{h}d_return"
            excess_col = f"post_exit_{h}d_excess_{benchmark_ticker.lower()}"
            raw = pd.to_numeric(group[raw_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            excess = pd.to_numeric(group[excess_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out[f"mean_post_exit_{h}d_return"] = (
                float(raw.dropna().mean()) if raw.notna().any() else None
            )
            out[f"mean_post_exit_{h}d_excess"] = (
                float(excess.dropna().mean()) if excess.notna().any() else None
            )
        return out

    def _group_rows(group_col: str) -> list[dict[str, Any]]:
        if group_col not in audit.columns:
            return []
        out: list[dict[str, Any]] = []
        for key, group in audit.groupby(group_col, dropna=False, observed=False):
            if len(group) < min_n:
                continue
            out.append({
                group_col: "NULL" if pd.isna(key) else str(key),
                **_path_summary(group),
            })
        return sorted(out, key=lambda r: float(r.get("net_pnl_after_tax") or 0.0))

    bad_exits = audit[
        pd.to_numeric(audit["exit_correct_by_barrier"], errors="coerce").eq(0)
    ].sort_values("net_pnl_after_tax").head(25)
    cols = [
        "cut",
        "ticker",
        "entry_date",
        "exit_date",
        "entry_regime",
        "exit_regime",
        "exit_reason",
        "gross_pnl",
        "net_pnl_after_tax",
        "pnl_pct",
        "hold_days",
        "entry_rank_score",
        "entry_mu",
        "path_sigma_daily",
        "barrier_label",
        "barrier_date",
        "barrier_price",
    ]
    for h in horizons:
        cols.append(f"post_exit_{h}d_excess_{benchmark_ticker.lower()}")

    return {
        "enabled": True,
        "benchmark_ticker": benchmark_ticker,
        "ohlcv_root": str(ohlcv_root),
        "barrier_window_days": int(barrier_window_days),
        "pt_mult": float(pt_mult),
        "sl_mult": float(sl_mult),
        "n_exits": int(len(audit)),
        "overall": _path_summary(audit),
        "by_exit_reason": _group_rows("exit_reason"),
        "by_entry_regime": _group_rows("entry_regime"),
        "by_exit_regime": _group_rows("exit_regime"),
        "by_entry_exit_regime": _group_rows("entry_exit_regime"),
        "barrier_false_positive_examples": _json_ready(
            bad_exits[[c for c in cols if c in bad_exits.columns]].to_dict(orient="records")
        ),
    }


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
    alpha = _with_entry_exit_regime(alpha)
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
        "by_entry_exit_regime": _active_group_table(alpha, "entry_exit_regime", min_n=min_group_n),
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
    ohlcv_root: Path | None = None,
) -> dict[str, Any]:
    trace_dir = trace_dir.resolve()
    lot_method = _tax_lot_method(config, lot_method_override)
    benchmark_ticker = _benchmark_ticker(config)
    trips = _load_round_trips(trace_dir, lot_method=lot_method)
    closed = _closed(trips)
    closed = _with_entry_exit_regime(closed) if not closed.empty else closed
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
        "forward_return_alignment": _forward_return_alignment(
            closed,
            ohlcv_root=ohlcv_root,
            benchmark_ticker=benchmark_ticker,
            min_n=max(10, min_group_n),
        ),
        "exit_path_audit": _exit_path_audit(
            closed,
            ohlcv_root=ohlcv_root,
            benchmark_ticker=benchmark_ticker,
            min_n=min_group_n,
        ),
        "score_spearman_by_entry_regime": _score_spearman_by_group(
            closed,
            "entry_regime",
            min_n=max(10, min_group_n),
        ),
        "score_spearman_by_exit_regime": _score_spearman_by_group(
            closed,
            "exit_regime",
            min_n=max(10, min_group_n),
        ),
        "entry_score_ladder": _entry_score_ladder(
            closed,
            benchmark_ticker=benchmark_ticker,
            min_group_n=max(3, min_group_n),
        ),
        "groups": {
            "by_cut": _group_table(closed, "cut", min_n=min_group_n),
            "by_exit_reason": _group_table(closed, "exit_reason", min_n=min_group_n),
            "by_entry_regime": _group_table(closed, "entry_regime", min_n=min_group_n),
            "by_exit_regime": _group_table(closed, "exit_regime", min_n=min_group_n),
            "by_entry_exit_regime": _group_table(closed, "entry_exit_regime", min_n=min_group_n),
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
    for key in ("by_cut", "by_exit_reason", "by_entry_regime", "by_entry_exit_regime", "by_ticker"):
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
    for key in ("score_spearman_by_entry_regime", "score_spearman_by_exit_regime"):
        lines.extend(["", f"### {key}"])
        rows = payload.get(key, [])
        if rows:
            lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
        else:
            lines.append("Insufficient scored closed trades by regime.")

    lines.extend(["", "## Entry Score Ladder"])
    ladder = payload.get("entry_score_ladder", [])
    if ladder:
        lines.append(pd.DataFrame(ladder).to_markdown(index=False, floatfmt=".4f"))
    else:
        lines.append("Insufficient scored alpha trades by regime.")

    fra = payload.get("forward_return_alignment", {})
    lines.extend(["", "## Forward Return Alignment"])
    if fra.get("enabled"):
        lines.append(
            f"- entry_events: {fra.get('n_entry_events')}"
            f"  benchmark: `{fra.get('benchmark_ticker')}`"
        )
        for key in ("overall", "by_entry_regime"):
            rows = fra.get(key, [])
            lines.extend(["", f"### forward_return_alignment.{key}"])
            if rows:
                lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
            else:
                lines.append("Insufficient rows.")
    else:
        lines.append(f"Disabled: {fra.get('reason')}")

    epa = payload.get("exit_path_audit", {})
    lines.extend(["", "## Exit Path Audit"])
    if epa.get("enabled"):
        lines.extend([
            f"- exits: {epa.get('n_exits')}",
            f"- barrier_window_days: {epa.get('barrier_window_days')}",
            f"- barrier: ±{epa.get('pt_mult')} / {epa.get('sl_mult')} daily-sigma multiples",
            f"- benchmark: `{epa.get('benchmark_ticker')}`",
        ])
        if epa.get("overall"):
            lines.append(pd.DataFrame([epa["overall"]]).to_markdown(index=False, floatfmt=".4f"))
        for key in ("by_exit_reason", "by_entry_regime", "by_exit_regime", "by_entry_exit_regime"):
            rows = epa.get(key, [])
            lines.extend(["", f"### exit_path_audit.{key}"])
            if rows:
                lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
            else:
                lines.append("No rows.")
        examples = pd.DataFrame(epa.get("barrier_false_positive_examples", []))
        lines.extend(["", "### exit_path_audit.false_positive_examples"])
        if not examples.empty:
            lines.append(examples.to_markdown(index=False, floatfmt=".4f"))
        else:
            lines.append("No false-positive examples.")
    else:
        lines.append(f"Disabled: {epa.get('reason')}")

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
    parser.add_argument(
        "--ohlcv-root",
        default=None,
        help="Optional OHLCV parquet root for entry-score vs forward-return checks. "
             "Default: disabled. Typical value: data/ohlcv.",
    )
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
        ohlcv_root=(REPO_ROOT / args.ohlcv_root if args.ohlcv_root else None),
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
