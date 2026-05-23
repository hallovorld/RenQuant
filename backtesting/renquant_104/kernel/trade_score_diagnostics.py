"""Trade-level score diagnostics for executed round trips.

Panel IC tells us whether a model has cross-sectional signal in the full
candidate universe. This module answers the stricter execution question:
among the trades the decision tree actually bought, did entry-time scores
separate winners from losers?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCORE_COLS = (
    "entry_rank_score",
    "entry_mu",
    "entry_sigma",
    "entry_mu_over_sigma",
    "entry_panel_score",
    "entry_kelly_target_pct",
)


@dataclass(frozen=True)
class ScoreDiagnostic:
    score_col: str
    n: int
    spearman: float | None
    top_mean: float | None
    bottom_mean: float | None
    top_bottom_spread: float | None
    winner_mean: float | None
    loser_mean: float | None
    winner_loser_spread: float | None
    higher_is_better: bool


def _finite_pair(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    pair = df[[x_col, y_col]].copy()
    pair[x_col] = pd.to_numeric(pair[x_col], errors="coerce")
    pair[y_col] = pd.to_numeric(pair[y_col], errors="coerce")
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    return pair


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _score_direction(score_col: str) -> bool:
    """Return True when a larger score should imply larger future P&L."""
    # Volatility/uncertainty is a risk variable, not an alpha score. A
    # successful risk filter should have negative raw correlation with P&L.
    return score_col not in {"entry_sigma"}


def _prepare_round_trips(
    round_trips: pd.DataFrame,
    *,
    outcome_col: str,
    closed_only: bool = True,
) -> pd.DataFrame:
    df = round_trips.copy()
    if closed_only and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "closed"].copy()
    if "entry_mu_over_sigma" not in df.columns:
        mu_src = df["entry_mu"] if "entry_mu" in df.columns else pd.Series(np.nan, index=df.index)
        sig_src = (
            df["entry_sigma"]
            if "entry_sigma" in df.columns else pd.Series(np.nan, index=df.index)
        )
        mu = pd.to_numeric(mu_src, errors="coerce")
        sig = pd.to_numeric(sig_src, errors="coerce")
        df["entry_mu_over_sigma"] = mu / sig.replace(0, np.nan)
    if outcome_col not in df.columns:
        raise KeyError(f"outcome column missing: {outcome_col}")
    return df


def compute_score_diagnostics(
    round_trips: pd.DataFrame,
    *,
    outcome_col: str = "pnl_pct",
    score_cols: tuple[str, ...] = DEFAULT_SCORE_COLS,
    closed_only: bool = True,
    min_n: int = 5,
) -> dict[str, Any]:
    """Compute execution-level predictive diagnostics.

    Returns a JSON-serialisable dict. Spearman and bucket spreads are computed
    on closed trades by default because open lots have not realized an exit
    decision yet.
    """
    df = _prepare_round_trips(
        round_trips, outcome_col=outcome_col, closed_only=closed_only,
    )
    outcome = pd.to_numeric(df[outcome_col], errors="coerce")
    df = df.assign(_outcome=outcome).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["_outcome"])
    winners = df[df["_outcome"] > 0]
    losers = df[df["_outcome"] <= 0]

    metrics: list[ScoreDiagnostic] = []
    for col in score_cols:
        if col not in df.columns:
            continue
        pair = _finite_pair(df, col, "_outcome")
        higher_is_better = _score_direction(col)
        if len(pair) < min_n or pair[col].nunique(dropna=True) < 2:
            metrics.append(ScoreDiagnostic(
                score_col=col,
                n=len(pair),
                spearman=None,
                top_mean=None,
                bottom_mean=None,
                top_bottom_spread=None,
                winner_mean=None,
                loser_mean=None,
                winner_loser_spread=None,
                higher_is_better=higher_is_better,
            ))
            continue

        spearman = _maybe_float(pair[col].corr(pair["_outcome"], method="spearman"))
        q = max(int(len(pair) * 0.25), 1)
        ordered = pair.sort_values(col, ascending=True)
        bottom = ordered.head(q)
        top = ordered.tail(q)
        top_mean = _maybe_float(top["_outcome"].mean())
        bottom_mean = _maybe_float(bottom["_outcome"].mean())

        win_mean = None
        lose_mean = None
        if not winners.empty:
            win_mean = _maybe_float(pd.to_numeric(winners[col], errors="coerce").mean())
        if not losers.empty:
            lose_mean = _maybe_float(pd.to_numeric(losers[col], errors="coerce").mean())

        metrics.append(ScoreDiagnostic(
            score_col=col,
            n=len(pair),
            spearman=spearman,
            top_mean=top_mean,
            bottom_mean=bottom_mean,
            top_bottom_spread=(
                _maybe_float(top_mean - bottom_mean)
                if top_mean is not None and bottom_mean is not None else None
            ),
            winner_mean=win_mean,
            loser_mean=lose_mean,
            winner_loser_spread=(
                _maybe_float(win_mean - lose_mean)
                if win_mean is not None and lose_mean is not None else None
            ),
            higher_is_better=higher_is_better,
        ))

    payload = {
        "n_trades": int(len(df)),
        "n_winners": int(len(winners)),
        "n_losers": int(len(losers)),
        "win_rate": _maybe_float(len(winners) / len(df)) if len(df) else None,
        "outcome_col": outcome_col,
        "closed_only": closed_only,
        "outcome_mean": _maybe_float(df["_outcome"].mean()) if len(df) else None,
        "outcome_median": _maybe_float(df["_outcome"].median()) if len(df) else None,
        "metrics": [m.__dict__ for m in metrics],
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Trade-Level Score Diagnostics",
        "",
        f"- closed_only: `{payload.get('closed_only')}`",
        f"- outcome_col: `{payload.get('outcome_col')}`",
        f"- n_trades: `{payload.get('n_trades')}`",
        f"- win_rate: `{_fmt_pct(payload.get('win_rate'))}`",
        f"- outcome_mean: `{_fmt_pct(payload.get('outcome_mean'))}`",
        f"- outcome_median: `{_fmt_pct(payload.get('outcome_median'))}`",
        "",
        "| score | n | Spearman vs outcome | top-bottom outcome spread | winner mean | loser mean | winner-loser score spread | expected direction |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in payload.get("metrics", []):
        direction = "higher better" if m.get("higher_is_better") else "lower risk better"
        lines.append(
            "| {score} | {n} | {spearman} | {spread} | {win} | {lose} | {wl} | {direction} |".format(
                score=m.get("score_col"),
                n=m.get("n"),
                spearman=_fmt_num(m.get("spearman")),
                spread=_fmt_pct(m.get("top_bottom_spread")),
                win=_fmt_num(m.get("winner_mean")),
                lose=_fmt_num(m.get("loser_mean")),
                wl=_fmt_num(m.get("winner_loser_spread")),
                direction=direction,
            )
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- For alpha scores, positive Spearman and positive top-bottom spread are required.",
        "- For sigma, negative Spearman is desirable because lower risk should realize better P&L.",
        "- If winners and losers have nearly identical rank/μ, the execution slice is not using a discriminative alpha score.",
        "",
    ])
    return "\n".join(lines)


def _fmt_num(v: Any) -> str:
    f = _maybe_float(v)
    return "NA" if f is None else f"{f:+.4f}"


def _fmt_pct(v: Any) -> str:
    f = _maybe_float(v)
    return "NA" if f is None else f"{f:+.2%}"
