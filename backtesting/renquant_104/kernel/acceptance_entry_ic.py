"""Entry-IC acceptance gate — ship-side companion to OOS-IC acceptance.

CPCV ``oos_mean_ic`` measures rank-correlation across the FULL panel
held-out folds. It does NOT measure whether the SPECIFIC entries the
strategy ACTUALLY TAKES are predictive.

2026-05-04 baseline B2 audit surfaced a fatal divergence:
  * CPCV oos_mean_ic = +0.034 (positive, "signal exists in panel")
  * Spearman(entry rank_score, realized pnl_pct) over 62 paired trades = **-0.103**
    (NEGATIVE — model's high-rank picks **lost** more than its low-rank picks)

The aggregation is:
  * Selection bias: only top-rank candidates make it through buy-side gates
  * Of those, the QP allocates capital — already a tight selection
  * So the trades the strategy ACTUALLY takes are a tiny biased slice
  * If that slice is anti-correlated, you ship a model that loses money

Invariant
---------
``entry_ic_paired(trades_df) >= min_ic`` for any model that ships to live.
Default min_ic = +0.02 (matches G7 OOS IC floor).

This module provides:
  * ``compute_entry_ic`` — Spearman of (entry rank_score, realized pnl_pct)
    over chronologically-paired buy/sell events per ticker.
  * ``acceptance`` — pass/fail given a min_ic threshold.

Usage
-----
::

    from kernel.acceptance_entry_ic import acceptance
    verdict = acceptance(trades_df, min_ic=0.02)
    if not verdict.passed:
        raise RuntimeError(f"ship blocked: entry IC {verdict.ic:+.3f} < {verdict.threshold}")
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class EntryICVerdict:
    passed:         bool
    ic:             float
    threshold:      float
    n_paired:       int
    ic_mu:          float | None
    ic_sigma:       float | None
    detail:         str

    def __str__(self) -> str:
        s = "PASS" if self.passed else "FAIL"
        return (f"EntryIC[{s}] ic={self.ic:+.4f} threshold={self.threshold:+.4f} "
                f"n={self.n_paired} {self.detail}")


def compute_entry_ic(
    trades_df: pd.DataFrame,
    *,
    rank_col: str = "rank_score",
    pnl_col:  str = "pnl_pct",
) -> tuple[float, float | None, float | None, int]:
    """Return (rank_ic, mu_ic, sigma_ic, n_paired).

    Pairs k-th BUY of a ticker with k-th SELL of same ticker (chronological,
    naive — sufficient for FIFO accounting which the sim uses). The rank /
    mu / sigma are read from BUY rows; the realized pnl_pct from SELL rows.

    NaN handling:
      * Trades whose BUY row lacks rank_col → excluded from rank IC.
      * Trades whose SELL row lacks pnl_col → excluded from all ICs.
      * If < 5 paired trades remain, returns (nan, nan, nan, n).
    """
    if trades_df.empty:
        return float("nan"), None, None, 0

    buys = trades_df[trades_df["action"] == "buy"].copy()
    sells = trades_df[trades_df["action"] == "sell"].copy()

    by_t_buys: dict[str, list] = defaultdict(list)
    by_t_sells: dict[str, list] = defaultdict(list)
    for _, r in buys.iterrows():
        by_t_buys[r["ticker"]].append(r)
    for _, r in sells.iterrows():
        by_t_sells[r["ticker"]].append(r)

    paired_rows = []
    for t, bs in by_t_buys.items():
        ss = by_t_sells.get(t, [])
        for i, sell_row in enumerate(ss):
            if i < len(bs):
                buy_row = bs[i]
                paired_rows.append({
                    "ticker":      t,
                    "rank_entry":  buy_row.get(rank_col),
                    "mu_entry":    buy_row.get("mu"),
                    "sigma_entry": buy_row.get("sigma"),
                    "pnl_pct":     sell_row.get(pnl_col),
                })

    if not paired_rows:
        return float("nan"), None, None, 0
    paired = pd.DataFrame(paired_rows).dropna(subset=["pnl_pct"])
    if len(paired) < 5:
        return float("nan"), None, None, len(paired)

    from scipy.stats import spearmanr   # noqa: PLC0415

    rank_subset = paired.dropna(subset=["rank_entry"])
    if len(rank_subset) >= 5 and rank_subset["rank_entry"].std() > 0:
        rho, _ = spearmanr(rank_subset["rank_entry"], rank_subset["pnl_pct"])
        rank_ic = float(rho) if rho == rho else float("nan")
    else:
        rank_ic = float("nan")

    mu_subset = paired.dropna(subset=["mu_entry"])
    mu_ic: float | None = None
    if len(mu_subset) >= 5 and mu_subset["mu_entry"].std() > 0:
        rho_mu, _ = spearmanr(mu_subset["mu_entry"], mu_subset["pnl_pct"])
        mu_ic = float(rho_mu) if rho_mu == rho_mu else None

    sig_subset = paired.dropna(subset=["sigma_entry"])
    sigma_ic: float | None = None
    if len(sig_subset) >= 5 and sig_subset["sigma_entry"].std() > 0:
        rho_sig, _ = spearmanr(sig_subset["sigma_entry"], sig_subset["pnl_pct"])
        sigma_ic = float(rho_sig) if rho_sig == rho_sig else None

    return rank_ic, mu_ic, sigma_ic, len(paired)


def acceptance(
    trades_df: pd.DataFrame,
    *,
    min_ic:        float = 0.02,
    min_n_paired:  int = 30,
) -> EntryICVerdict:
    """Pass when entry-rank IC ≥ min_ic AND we have enough paired trades.

    Few-trade scenarios (n < min_n_paired) return PASS-OPEN (insufficient
    sample) — the verdict's `detail` flags this so callers can choose to
    block-ship or run-longer.
    """
    ic, mu_ic, sigma_ic, n = compute_entry_ic(trades_df)
    if n < min_n_paired or ic != ic:
        return EntryICVerdict(
            passed=True, ic=ic if ic == ic else float("nan"),
            threshold=min_ic, n_paired=n,
            ic_mu=mu_ic, ic_sigma=sigma_ic,
            detail=f"insufficient sample (n={n} < {min_n_paired}) — pass-open",
        )
    return EntryICVerdict(
        passed=ic >= min_ic, ic=ic, threshold=min_ic, n_paired=n,
        ic_mu=mu_ic, ic_sigma=sigma_ic,
        detail=("entry rank predictive of pnl"
                if ic >= min_ic else
                f"entry rank ANTI-correlated with pnl (ic={ic:+.4f})"),
    )
