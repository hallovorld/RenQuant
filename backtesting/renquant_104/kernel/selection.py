"""Candidate scoring, guards, and tiered selection loop.

Self-contained: only datetime, dataclasses, math.  No common/ imports.

Public API:
  compute_relative_strength(stock_ret, etf_ret)  → float
  score_candidates(candidates, w_rank, w_rs)      → ranked list
  run_selection_loop(ranked, ctx)                 → (selected, blocks)
"""
from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field

log = logging.getLogger("pipeline.execution")


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class CandidateResult:
    ticker:          str
    raw_score:       float
    rank_score:      float
    rs_score:        float
    detail:          str = ""
    expected_return: float = 0.0   # E[R - SPY] over rotation.target_horizon_days
    panel_score:     float | None = None   # cross-sectional panel-LTR score; None when disabled
    mu:              float | None = None   # NGBoost μ (residual return forecast)
    sigma:           float | None = None   # NGBoost σ (predictive stdev)


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
    # Plan O (2026-04-23): defensive tickers are only eligible in the BEAR
    # branch. When `bear_only=False`, the selection loop rejects any
    # candidate in `defensive_set` with `blocks["defensive_non_bear"]`.
    # The pre-Plan-O behavior was a latent design bug: defensives could
    # compete as regular candidates in BULL_*/CHOPPY regimes AND bypass
    # the sector guard — e.g. XLU bought on 2026-04-20 at regime=BULL_VOLATILE.
    bear_only:          bool = False


def run_selection_loop(
    ranked: list[CandidateResult],
    ctx: SelectionContext,
    blocked_by_ticker: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Greedy slot-filling with tiered thresholds and all guards.

    Returns (selected_tickers, block_counts).
    block_counts keys: "wash_sale", "sector", "correlation", "tier",
                       "defensive_non_bear".

    If `blocked_by_ticker` is passed (must be an empty dict), it is
    populated in-place with per-ticker rejection reasons (ticker →
    one of the block_counts keys). Used by `RunSelectionTask` to
    feed `candidate_scores.blocked_by` in the decision-trace DB.
    """
    selected: list[str] = []
    blocks = {"wash_sale": 0, "sector": 0, "correlation": 0, "tier": 0,
              "defensive_non_bear": 0}
    slots_filled = 0

    def _reject(ticker: str, reason: str) -> None:
        blocks[reason] += 1
        if blocked_by_ticker is not None:
            blocked_by_ticker[ticker] = reason

    for c in ranked:
        if slots_filled >= ctx.open_slots:
            log.info("  %-6s  SKIP   [slots full]", c.ticker)
            break

        # Plan O — defensive tickers only admissible in the BEAR branch.
        # Non-BEAR regimes: filter them out early so they can't occupy
        # offensive slots. This also sidesteps the sector_guard bypass
        # (passes_sector_guard returns True for defensives) — the bypass
        # was safe in BEAR but a loophole in BULL_*/CHOPPY regimes.
        if c.ticker in ctx.defensive_set and not ctx.bear_only:
            _reject(c.ticker, "defensive_non_bear")
            log.info("  %-6s  SKIP   [defensive — not BEAR regime]", c.ticker)
            continue

        # Tiered threshold — escalating conviction requirement per slot
        if ctx.tiered_thresholds:
            tier_idx = min(slots_filled, len(ctx.tiered_thresholds) - 1)
            tier_min = float(ctx.tiered_thresholds[tier_idx].get("min_model_score", 0.0))
            if c.rank_score < tier_min:
                _reject(c.ticker, "tier")
                log.info("  %-6s  SKIP   [tier %d needs %.2f, got %.4f]",
                         c.ticker, tier_idx + 1, tier_min, c.rank_score)
                continue

        if is_wash_sale_blocked(c.ticker, ctx.today, ctx.last_sell_dates, ctx.wash_sale_days):
            _reject(c.ticker, "wash_sale")
            last = ctx.last_sell_dates.get(c.ticker)
            log.info("  %-6s  SKIP   [wash sale — sold %s]", c.ticker, last)
            continue

        if not passes_sector_guard(
            c.ticker, ctx.held_tickers + selected,
            ctx.sector_map, ctx.max_per_sector, ctx.defensive_set,
        ):
            _reject(c.ticker, "sector")
            sector = ctx.sector_map.get(c.ticker, "other")
            log.info("  %-6s  SKIP   [sector cap — %s at max %d]",
                     c.ticker, sector, ctx.max_per_sector)
            continue

        if not passes_correlation_guard(
            c.ticker, ctx.held_tickers + selected,
            ctx.corr_matrix, ctx.corr_threshold,
        ):
            _reject(c.ticker, "correlation")
            # find which held ticker caused the block
            corr_culprit = ""
            if ctx.corr_matrix:
                for held in ctx.held_tickers + selected:
                    corr = (ctx.corr_matrix.get(c.ticker, {}).get(held)
                            or ctx.corr_matrix.get(held, {}).get(c.ticker))
                    if corr is not None and abs(corr) >= ctx.corr_threshold:
                        corr_culprit = f" (corr with {held}: {corr:.2f})"
                        break
            log.info("  %-6s  SKIP   [correlation guard%s]", c.ticker, corr_culprit)
            continue

        slots_filled += 1
        log.info("  %-6s  SELECT [slot %d  calibrated=%+.4f  rs=%+.4f]",
                 c.ticker, slots_filled, c.rank_score, c.rs_score)
        selected.append(c.ticker)

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
