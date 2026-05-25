#!/usr/bin/env python
"""Trade-level score monotonicity gates for WF acceptance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeMonotonicityReport:
    passed: bool
    reason: str
    regimes: list[dict[str, Any]]
    pooled: dict[str, Any]


def evaluate_trade_monotonicity(
    round_trips: pd.DataFrame,
    *,
    score_col: str = "entry_rank_score",
    return_col: str = "pnl_pct",
    net_col: str = "net_pnl_after_tax",
    regime_col: str = "entry_regime",
    min_n_per_regime: int = 30,
    min_spearman: float = 0.02,
    min_top_bottom_spread: float = 0.0,
    small_n_inversion_min_n: int = 10,
    allow_pass_open: bool = False,
) -> TradeMonotonicityReport:
    """Require entry scores to be economically monotone per active regime."""
    df = _clean_round_trips(round_trips, score_col, return_col, net_col, regime_col)
    pooled = _summarize_group(df, score_col, return_col, net_col)
    if df.empty:
        return TradeMonotonicityReport(
            passed=False,
            reason="no closed round trips with finite score and return",
            regimes=[],
            pooled=pooled,
        )

    regime_reports: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for regime, g in df.groupby(regime_col, dropna=False):
        row = _summarize_group(g, score_col, return_col, net_col)
        row["regime"] = str(regime)
        row["eligible"] = int(row["n"]) >= int(min_n_per_regime)
        row["small_n_inversion"] = _is_small_n_inversion(
            row,
            min_n=small_n_inversion_min_n,
        )
        if row["eligible"]:
            row["passed"] = _passes_row(
                row,
                min_spearman=min_spearman,
                min_top_bottom_spread=min_top_bottom_spread,
            )
            eligible.append(row)
        elif row["small_n_inversion"]:
            row["passed"] = False
            row["detail"] = (
                "failed_small_n_inversion: "
                f"n={row['n']} >= {small_n_inversion_min_n} and "
                "score ordering is economically inverted"
            )
        else:
            row["passed"] = bool(allow_pass_open)
            row["detail"] = (
                f"{'pass-open' if allow_pass_open else 'fail-closed'}: "
                f"n={row['n']} < min_n_per_regime={min_n_per_regime}"
            )
        regime_reports.append(row)

    if not eligible:
        inverted = [r for r in regime_reports if r.get("small_n_inversion")]
        if inverted:
            labels = ", ".join(r["regime"] for r in inverted)
            return TradeMonotonicityReport(
                passed=False,
                reason=f"small-sample score inversion in regime(s): {labels}",
                regimes=regime_reports,
                pooled=pooled,
            )
        if not allow_pass_open:
            return TradeMonotonicityReport(
                passed=False,
                reason=(
                    "insufficient per-regime trade evidence: no regime has "
                    f"n >= {min_n_per_regime}"
                ),
                regimes=regime_reports,
                pooled=pooled,
            )
        return TradeMonotonicityReport(
            passed=True,
            reason=f"pass-open: no regime has n >= {min_n_per_regime}",
            regimes=regime_reports,
            pooled=pooled,
        )

    failed = [r for r in eligible if not r["passed"]]
    if failed:
        labels = ", ".join(r["regime"] for r in failed)
        return TradeMonotonicityReport(
            passed=False,
            reason=f"score monotonicity failed in active regime(s): {labels}",
            regimes=regime_reports,
            pooled=pooled,
        )
    return TradeMonotonicityReport(
        passed=True,
        reason=f"score monotonicity passed in {len(eligible)} active regime(s)",
        regimes=regime_reports,
        pooled=pooled,
    )


def _clean_round_trips(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    net_col: str,
    regime_col: str,
) -> pd.DataFrame:
    needed = [score_col, return_col, regime_col]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame(columns=[score_col, return_col, net_col, regime_col])
    out = df.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower() == "closed"]
    if net_col not in out.columns:
        out[net_col] = np.nan
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out[return_col] = pd.to_numeric(out[return_col], errors="coerce")
    out[net_col] = pd.to_numeric(out[net_col], errors="coerce")
    out[regime_col] = out[regime_col].fillna("UNKNOWN").astype(str)
    return out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[score_col, return_col]
    )


def _summarize_group(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    net_col: str,
) -> dict[str, Any]:
    n = int(len(df))
    if n == 0:
        return {"n": 0, "spearman": None, "top_bottom_return_spread": None}
    spearman = (
        float(df[score_col].corr(df[return_col], method="spearman"))
        if n >= 5 and df[score_col].nunique() > 1 else None
    )
    q = _quintile_summary(df, score_col, return_col, net_col)
    row: dict[str, Any] = {"n": n, "spearman": spearman}
    row.update(q)
    return row


def _quintile_summary(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    net_col: str,
) -> dict[str, Any]:
    if len(df) < 5 or df[score_col].nunique() < 2:
        return {
            "top_return_mean": None,
            "bottom_return_mean": None,
            "top_bottom_return_spread": None,
            "top_net_pnl": None,
            "bottom_net_pnl": None,
        }
    tmp = df.copy()
    tmp["__q"] = pd.qcut(
        tmp[score_col].rank(method="first"),
        q=min(5, len(tmp)),
        labels=False,
        duplicates="drop",
    )
    lo = tmp[tmp["__q"] == tmp["__q"].min()]
    hi = tmp[tmp["__q"] == tmp["__q"].max()]
    top_ret = float(hi[return_col].mean())
    bot_ret = float(lo[return_col].mean())
    return {
        "top_return_mean": top_ret,
        "bottom_return_mean": bot_ret,
        "top_bottom_return_spread": top_ret - bot_ret,
        "top_net_pnl": float(hi[net_col].sum()),
        "bottom_net_pnl": float(lo[net_col].sum()),
    }


def _passes_row(
    row: dict[str, Any],
    *,
    min_spearman: float,
    min_top_bottom_spread: float,
) -> bool:
    spearman = row.get("spearman")
    spread = row.get("top_bottom_return_spread")
    if spearman is None or not np.isfinite(float(spearman)):
        return False
    if spread is None or not np.isfinite(float(spread)):
        return False
    return float(spearman) >= min_spearman and float(spread) > min_top_bottom_spread


def _is_small_n_inversion(row: dict[str, Any], *, min_n: int) -> bool:
    if int(row.get("n") or 0) < int(min_n):
        return False
    spearman = row.get("spearman")
    spread = row.get("top_bottom_return_spread")
    spearman_bad = (
        spearman is not None
        and np.isfinite(float(spearman))
        and float(spearman) < 0.0
    )
    spread_bad = (
        spread is not None
        and np.isfinite(float(spread))
        and float(spread) < 0.0
    )
    return bool(spearman_bad or spread_bad)
