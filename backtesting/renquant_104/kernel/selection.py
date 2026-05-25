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
    expected_return_horizon_days: int | None = None
    panel_score:     float | None = None   # cross-sectional panel-LTR score; None when disabled
    mu:              float | None = None   # NGBoost μ (residual return forecast)
    mu_horizon_days: int | None = None
    sigma:           float | None = None   # NGBoost σ (predictive stdev)


# ── Guard helpers ──────────────────────────────────────────────────────────────

def is_wash_sale_blocked(
    ticker: str,
    today: datetime.date,
    last_sell_dates: dict[str, datetime.date | None],
    wash_sale_days: int,
) -> bool:
    """Return True if ticker sold within wash_sale_days of today.

    DEPRECATED for production buy-side filtering — this is a binary block
    that ignores the actual economic cost. Use is_wash_sale_blocked_with_cost
    to factor in (a) gain sales have ZERO wash-sale cost (rule does not
    apply per IRC §1091) and (b) loss sales have only an NPV time-value
    cost (deferred deduction recovered on eventual sale of replacement).

    Kept for back-compat callers that don't have realized-pnl data.
    """
    if wash_sale_days <= 0:
        return False
    last = last_sell_dates.get(ticker)
    if last is None:
        return False
    return (today - last).days < wash_sale_days


def wash_sale_npv_cost(
    realized_loss: float,
    *,
    tax_rate: float = 0.30,
    discount_rate: float = 0.05,
    estimated_hold_years: float = 2.0,
) -> float:
    """NPV economic cost of a §1091-disallowed loss deduction.

    The disallowed loss is added to the basis of the replacement security
    (§1091(d)) and recovered when the replacement is eventually sold.
    Per §1223(3), the holding period of the original carries forward
    (no LT/ST treatment penalty).

    Real cost = lost present value of the deferred tax savings:
        cost = |loss| × tax_rate × (1 − 1/(1+r)^t)
    where:
        r = discount rate (cost of capital)
        t = expected years until the replacement is sold and the
            disallowed loss flows back into a deduction

    Defaults: 30% combined federal+state tax, 5% discount, 2-year hold.
    For a $100 loss this gives ~$2.78 NPV cost — a small fraction of the
    raw loss.

    Reference: IRC §1091, §1091(d), §1223(3); IRS Publication 550.
    """
    if realized_loss >= 0:
        return 0.0   # gains have NO wash-sale cost (rule does not apply)
    loss_abs = abs(realized_loss)
    deferred_savings_now = loss_abs * tax_rate
    nav_factor = 1.0 - 1.0 / ((1.0 + discount_rate) ** estimated_hold_years)
    return deferred_savings_now * nav_factor


def is_wash_sale_blocked_with_cost(
    ticker: str,
    today: datetime.date,
    last_sell_dates: dict[str, datetime.date | None],
    last_sell_pls: dict[str, float | None] | None,
    wash_sale_days: int,
    *,
    tax_rate: float = 0.30,
    discount_rate: float = 0.05,
    estimated_hold_years: float = 2.0,
    expected_dollar_return: float | None = None,
    safety_margin: float = 1.5,
) -> tuple[bool, str, float]:
    """Cost-aware wash-sale decision per IRC §1091.

    Logic:
      1. If sale is outside the 30-day window → no rule applies → not blocked
      2. If prior sale was a GAIN (or unknown but assume gain in absence
         of data) → §1091 does not apply → not blocked
      3. If prior sale was a LOSS:
           cost_npv = wash_sale_npv_cost(loss, tax_rate, ...)
         (a) if expected_dollar_return is known → block if expected_return
             < safety_margin × cost_npv
         (b) else (no μ̂ at this stage) → soft-block: keep blocking on
             losses but log the cost so caller can route to a later
             economic-aware gate

    Returns: (blocked: bool, reason: str, cost_npv: float)

    The (b) branch is the conservative default at the per-ticker
    candidate-filter stage where μ̂ isn't available yet. Callers that
    have μ̂ (e.g. the post-NGB economic gate) should pass
    expected_dollar_return.
    """
    if wash_sale_days <= 0:
        return (False, "wash_sale_days=0 (disabled)", 0.0)
    last = last_sell_dates.get(ticker)
    if last is None:
        return (False, "no recent sale", 0.0)
    # 2026-05-09 audit Phase 2.2 fix: state files persist dates as ISO
    # strings; live runner had its own coercion at the call site, but the
    # QP wash-sale path passed raw last_sell_dates straight in. Coerce
    # here once so all callers share the same path. Returns "no recent
    # sale" sentinel on bad/unparseable strings.
    if isinstance(last, str):
        try:
            last = datetime.date.fromisoformat(last[:10])
        except (ValueError, TypeError):
            return (False, "no recent sale (unparseable date)", 0.0)
    days_since = (today - last).days
    if days_since >= wash_sale_days:
        return (False, f"{days_since}d since sale (≥ {wash_sale_days}d window)", 0.0)
    pl = (last_sell_pls or {}).get(ticker)
    if pl is None:
        # P/L data not available (broker doesn't expose history, or sim
        # adapter didn't compute it). Fall back to binary block — cannot
        # safely allow without knowing if it was a loss.
        return (
            True,
            f"sold {days_since}d ago (P/L unknown — binary block)",
            0.0,
        )
    if pl >= 0.0:
        # GAIN sale → §1091 does not apply (rule applies only to losses)
        return (False, f"§1091 N/A (gain sale ${pl:+.2f})", 0.0)
    # LOSS sale within window — compute economic cost
    cost_npv = wash_sale_npv_cost(
        pl, tax_rate=tax_rate, discount_rate=discount_rate,
        estimated_hold_years=estimated_hold_years,
    )
    if expected_dollar_return is None:
        # Can't run cost-vs-return test at this stage — keep block.
        return (
            True,
            f"loss sale ${pl:.2f} {days_since}d ago, NPV cost ≈${cost_npv:.2f}",
            cost_npv,
        )
    if expected_dollar_return >= safety_margin * cost_npv:
        return (
            False,
            f"expected ${expected_dollar_return:+.2f} > {safety_margin}×NPV cost ${cost_npv:.2f}",
            cost_npv,
        )
    return (
        True,
        f"expected ${expected_dollar_return:+.2f} < {safety_margin}×NPV cost ${cost_npv:.2f}",
        cost_npv,
    )


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
    sector = sector_map.get(ticker)
    if not isinstance(sector, str) or not sector:
        return False
    for held in held_tickers:
        if held in defensive_set:
            continue
        held_sector = sector_map.get(held)
        if not isinstance(held_sector, str) or not held_sector:
            return False
    count = sum(1 for t in held_tickers if sector_map.get(t) == sector)
    return count < max_per_sector


def passes_correlation_guard(
    ticker: str,
    held_tickers: list[str],
    corr_matrix: dict[str, dict[str, float]] | None,
    threshold: float,
) -> bool:
    """Return True if ticker is not too correlated with any held position.

    2026-04-24 (#28): explicit None check instead of `a or b` — `0.0 or X`
    short-circuits to X, so a real zero correlation was discarded in favour
    of the reverse-direction lookup (which might be missing / stale).
    """
    # Audit fix SL-2 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
    # correlation slipped past `abs(corr) >= threshold` (NaN comparisons
    # are False) → guard returned True → highly-correlated buy allowed
    # silently when correlation matrix has NaN cells.
    import math as _math
    if not held_tickers:
        return True
    if corr_matrix is None:
        return False
    for held in held_tickers:
        # Audit fix SELF-CORR (2026-04-25): pre-fix, when a candidate
        # was being considered for slot N AFTER it was already added to
        # held_tickers + selected via a prior rotation pair (e.g. iter3:
        # JNJ entered via rotation, then iterated as selection candidate
        # → corr(JNJ, JNJ) = 1.0 → self-rejected). Skip self-match — a
        # ticker is by definition perfectly correlated with itself, but
        # that's irrelevant for the diversification check we're doing.
        if held == ticker:
            continue
        corr = corr_matrix.get(ticker, {}).get(held)
        if corr is None:
            corr = corr_matrix.get(held, {}).get(ticker)
        if corr is None:
            return False
        if not _math.isfinite(corr):
            # Treat NaN/inf correlation as MAX correlation — fail-SAFE
            # to block the buy when we can't verify independence.
            return False
        if abs(corr) >= threshold:
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

    # 2026-05-04 audit Issue 20 fix: drop NaN/inf entries before computing
    # min/max. Pre-fix, a single NaN rank_score made `min(...)` return
    # NaN → `_norm()` returned NaN for every candidate → `blend()` returned
    # NaN → `sorted()` is non-deterministic on NaN keys (Python's Timsort
    # comparisons return False both ways for NaN). Different runs of the
    # same data could produce different rankings. Same RA-1 pattern as
    # SortCandidatesTask but at the upstream entry point.
    import math as _math
    finite_rank = [c.rank_score for c in candidates
                   if c.rank_score is not None
                   and _math.isfinite(float(c.rank_score))]
    finite_rs   = [c.rs_score for c in candidates
                   if c.rs_score is not None
                   and _math.isfinite(float(c.rs_score))]
    if finite_rank:
        rank_min, rank_max = min(finite_rank), max(finite_rank)
    else:
        rank_min, rank_max = 0.0, 0.0
    if finite_rs:
        rs_min, rs_max = min(finite_rs), max(finite_rs)
    else:
        rs_min, rs_max = 0.0, 0.0

    def _safe_norm(v: float | None, lo: float, hi: float) -> float:
        if v is None or not _math.isfinite(float(v)):
            return 0.0   # NaN/None contributes nothing — same effect as ranking last
        return _norm(float(v), lo, hi)

    scored: list[tuple[CandidateResult, float]] = []
    for c in candidates:
        rank_component = _safe_norm(c.rank_score, rank_min, rank_max)
        rs_component = _safe_norm(c.rs_score, rs_min, rs_max)
        composite = w_rank * rank_component + w_rs * rs_component
        setattr(c, "_ranking_composite", composite)
        setattr(c, "_ranking_norm_rank", rank_component)
        setattr(c, "_ranking_norm_rs", rs_component)
        scored.append((c, composite))

    ranked = [c for c, _ in sorted(scored, key=lambda item: item[1], reverse=True)]
    for idx, c in enumerate(ranked):
        setattr(c, "_ranking_order_index", idx)
    return ranked


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
    # 2026-05-09 audit FIX-A (Phase 2.3): per-ticker realized $ P/L for the
    # most-recent FULL liquidation. Enables cost-aware wash-sale (§1091
    # N/A on gain sales) instead of binary 30d block. Default empty dict —
    # fail-conservative when caller doesn't supply (mirrors WashSaleFilterTask).
    last_sell_pls:      dict[str, float | None] = field(default_factory=dict)


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
        # Audit fix SL-1 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
        # rank_score made `c.rank_score < tier_min` evaluate False → the
        # tier filter let NaN through. Defense in depth (TC-1 should
        # have already filtered upstream, but selection guards must
        # also be NaN-safe).
        import math as _math
        if ctx.tiered_thresholds:
            tier_idx = min(slots_filled, len(ctx.tiered_thresholds) - 1)
            tier_min = float(ctx.tiered_thresholds[tier_idx].get("min_model_score", 0.0))
            rs = c.rank_score
            if rs is None or not _math.isfinite(rs) or rs < tier_min:
                _reject(c.ticker, "tier")
                log.info("  %-6s  SKIP   [tier %d needs %.2f, got %s]",
                         c.ticker, tier_idx + 1, tier_min, rs)
                continue

        # 2026-05-09 audit FIX-A: cost-aware wash-sale (§1091).
        # Pre-fix used binary 30d block ignoring P/L. WashSaleFilterTask in
        # candidate path was already cost-aware; this selection loop
        # (greedy non-prod path) was the last binary holdout. Now uses the
        # same single-source-of-truth helper.
        ws_blocked, ws_reason, _ = is_wash_sale_blocked_with_cost(
            ticker=c.ticker,
            today=ctx.today,
            last_sell_dates=ctx.last_sell_dates,
            last_sell_pls=ctx.last_sell_pls,
            wash_sale_days=ctx.wash_sale_days,
        )
        if ws_blocked:
            _reject(c.ticker, "wash_sale")
            log.info("  %-6s  SKIP   [wash sale — %s]", c.ticker, ws_reason)
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

    Returns 0.0 when either input is NaN OR inf.
    """
    # 2026-05-04 audit Issue 21 fix: also guard against inf. Pre-fix,
    # an inf return (theoretically possible from an upstream divide-by-
    # zero) propagated through subtraction and downstream rs ranking.
    if not (math.isfinite(stock_ret_20d) and math.isfinite(etf_ret_20d)):
        return 0.0
    return stock_ret_20d - etf_ret_20d
