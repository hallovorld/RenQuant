"""Candidate scoring, guards, and tiered selection loop.

Self-contained: only datetime, dataclasses, math.  No common/ imports.

Public API:
  compute_relative_strength(stock_ret, etf_ret)  → float
  score_candidates(candidates, w_rank, w_rs)      → ranked list
  run_selection_loop(ranked, ctx)                 → (selected, blocks)
"""
from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class CandidateResult:
    ticker:     str
    raw_score:  float
    rank_score: float
    rs_score:   float
    detail:     str = ""


# ── Guard helpers ──────────────────────────────────────────────────────────────

def is_wash_sale_blocked(
    ticker: str,
    today: datetime.date,
    last_sell_dates: dict[str, datetime.date | None],
    wash_sale_days: int,
) -> bool:
    """Return True if ticker sold within wash_sale_days of today."""
    if wash_sale_days <= 0:
        return False
    last = last_sell_dates.get(ticker)
    if last is None:
        return False
    return (today - last).days < wash_sale_days


def is_earnings_blocked(
    ticker: str,
    today: datetime.date,
    earnings_calendar: dict[str, list[str]],
    buffer_days: int,
) -> bool:
    """Return True if ticker has earnings within ±buffer_days of today."""
    if not earnings_calendar:
        return False
    for d_str in earnings_calendar.get(ticker, []):
        try:
            d = datetime.date.fromisoformat(d_str)
            if abs((d - today).days) <= buffer_days:
                return True
        except ValueError:
            continue
    return False


def passes_sector_guard(
    ticker: str,
    held_tickers: list[str],
    sector_map: dict[str, str],
    max_per_sector: int,
    defensive_set: set[str],
) -> bool:
    """Return True if adding ticker would not exceed max_per_sector."""
    if max_per_sector <= 0:
        return True
    if ticker in defensive_set:
        return True   # defensives bypass sector guard
    sector = sector_map.get(ticker, "other")
    count  = sum(1 for t in held_tickers if sector_map.get(t, "other") == sector)
    return count < max_per_sector


def passes_correlation_guard(
    ticker: str,
    held_tickers: list[str],
    corr_matrix: dict[str, dict[str, float]] | None,
    threshold: float,
) -> bool:
    """Return True if ticker is not too correlated with any held position."""
    if corr_matrix is None or not held_tickers:
        return True
    for held in held_tickers:
        corr = (corr_matrix.get(ticker, {}).get(held)
                or corr_matrix.get(held, {}).get(ticker))
        if corr is not None and abs(corr) >= threshold:
            return False
    return True


# ── Ranking ────────────────────────────────────────────────────────────────────

def _norm(v: float, lo: float, hi: float) -> float:
    return (v - lo) / (hi - lo) if hi > lo else 0.5


def score_candidates(
    candidates: list[CandidateResult],
    w_rank: float,
    w_rs: float,
) -> list[CandidateResult]:
    """Return candidates sorted by blended rank (descending)."""
    if not candidates:
        return []

    rank_scores = [c.rank_score for c in candidates]
    rs_scores   = [c.rs_score   for c in candidates]
    rank_min, rank_max = min(rank_scores), max(rank_scores)
    rs_min,   rs_max   = min(rs_scores),   max(rs_scores)

    def blend(c: CandidateResult) -> float:
        return (w_rank * _norm(c.rank_score, rank_min, rank_max)
                + w_rs   * _norm(c.rs_score,   rs_min,   rs_max))

    return sorted(candidates, key=blend, reverse=True)


# ── Selection loop ─────────────────────────────────────────────────────────────

@dataclass
class SelectionContext:
    """All guards + state needed by the selection loop.

    Callers build one per bar and pass to run_selection_loop.
    """
    today:              datetime.date
    held_tickers:       list[str]
    last_sell_dates:    dict[str, datetime.date | None]
    earnings_calendar:  dict[str, list[str]]
    corr_matrix:        dict[str, dict[str, float]] | None
    sector_map:         dict[str, str]
    defensive_set:      set[str]
    wash_sale_days:     int
    earnings_buffer:    int
    corr_threshold:     float
    max_per_sector:     int
    tiered_thresholds:  list[dict]   # [{min_model_score: 0.10}, ...]
    open_slots:         int


def run_selection_loop(
    ranked: list[CandidateResult],
    ctx: SelectionContext,
) -> tuple[list[str], dict[str, int]]:
    """Greedy slot-filling with tiered thresholds and all guards.

    Returns (selected_tickers, block_counts).
    block_counts keys: "wash_sale", "sector", "correlation", "tier".
    """
    selected: list[str] = []
    blocks = {"wash_sale": 0, "sector": 0, "correlation": 0, "tier": 0}
    slots_filled = 0

    for c in ranked:
        if slots_filled >= ctx.open_slots:
            break

        # Tiered threshold — escalating conviction requirement per slot
        if ctx.tiered_thresholds:
            tier_idx = min(slots_filled, len(ctx.tiered_thresholds) - 1)
            tier_min = float(ctx.tiered_thresholds[tier_idx].get("min_model_score", 0.0))
            if c.rank_score < tier_min:
                blocks["tier"] += 1
                continue

        if is_wash_sale_blocked(c.ticker, ctx.today, ctx.last_sell_dates, ctx.wash_sale_days):
            blocks["wash_sale"] += 1
            continue

        if not passes_sector_guard(
            c.ticker, ctx.held_tickers + selected,
            ctx.sector_map, ctx.max_per_sector, ctx.defensive_set,
        ):
            blocks["sector"] += 1
            continue

        if not passes_correlation_guard(
            c.ticker, ctx.held_tickers + selected,
            ctx.corr_matrix, ctx.corr_threshold,
        ):
            blocks["correlation"] += 1
            continue

        selected.append(c.ticker)
        slots_filled += 1

    return selected, blocks


# ── Relative-strength helper ───────────────────────────────────────────────────

def compute_relative_strength(stock_ret_20d: float, etf_ret_20d: float) -> float:
    """Return stock outperformance vs its sector ETF over a 20-day window.

    Args:
        stock_ret_20d: 20-day return of the stock  (pct_change(20)).
        etf_ret_20d:   20-day return of its sector ETF.

    Returns 0.0 when either input is NaN.
    """
    if math.isnan(stock_ret_20d) or math.isnan(etf_ret_20d):
        return 0.0
    return stock_ret_20d - etf_ret_20d
