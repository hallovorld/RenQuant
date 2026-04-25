"""Exit-check pure functions — all 5 exit types + tax-aware hold gate.

Self-contained: only datetime, dataclasses.  No common/ imports.
Priority order (highest → lowest):
  1. trailing_stop   (BULL_CALM only, peak-gain armed)
  2. stop_loss       (cumulative from entry)
  3. single_day_loss (drop from previous close — BULL_CALM only)
  4. max_hold        (forced time exit)
  5. [tax_hold_gate] (suppresses model-sell near 1-year mark with unrealized gain)
  6. model_sell      (consecutive sell-signal streak)
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field


# ── Per-position mutable state ─────────────────────────────────────────────────

@dataclass
class HoldingState:
    """Mutable state for a single held position.

    Callers own instances and update them each bar.
    """
    entry_price:    float
    entry_date:     datetime.date
    high_watermark: float   # max close seen since entry (trailing stop trigger)
    sell_streak:     int = 0
    prev_close:      float | None = None
    rank_score:      float | None = None   # latest calibrated probability (set by ScoreModelTask)
    expected_return: float | None = None   # latest E[R-SPY] over rotation horizon
    panel_score:     float | None = None   # latest cross-sectional panel-LTR score (set by PanelScoringJob)
    mu:              float | None = None   # latest NGBoost μ (set by PanelScoringJob)
    sigma:           float | None = None   # latest NGBoost σ (set by PanelScoringJob)
    # Shares actually held at broker — populated by adapters from
    # broker positions cache. Needed to compute current-pct vs
    # kelly_target_pct for top-up decisions.
    shares:              float = 0.0
    kelly_target_pct:    float | None = None   # set by ApplyScoresTask when kelly_sizing.enabled

    # Thesis-degradation rotation (Approach A, 2026-04-24): snapshot of
    # the decision signals AT ENTRY, stamped by adapters when a fresh
    # position is opened. These are FIXED baselines — not recomputed each
    # bar — so rotation decisions compare "Y today vs Y when we bought"
    # instead of two noisy Kelly targets. See kernel.rotation for the
    # decision rule.
    entry_rank_score:    float | None = None
    entry_panel_score:   float | None = None
    entry_kelly_target_pct: float | None = None


# ── Exit result ────────────────────────────────────────────────────────────────

@dataclass
class ExitSignal:
    should_exit: bool
    reason:      str
    exit_type:   str   # "trailing_stop" | "stop_loss" | "single_day_loss" | "max_hold" | "model_sell" | "rotation" | "kelly_trim" | ""
    # Partial-sell infra (Plan: prereq for AB-trim).
    # None = full liquidation (default, current behaviour).
    # float < current_shares = partial sell, keep the position open.
    # float ≥ current_shares = full liquidation (same as None).
    quantity:    float | None = None
    # Diagnostic: when ScoreModel said "sell" but min_hold_days / streak
    # rule blocked the exit, EvaluateExitsTask flips this so pp_inference
    # can increment the blocked_streak counter without resorting to
    # untyped attribute writes (audit #17).
    blocked_streak: bool = False


_NO_EXIT = ExitSignal(should_exit=False, reason="", exit_type="")


# ── Individual exit checks ─────────────────────────────────────────────────────

def check_trailing_stop(
    current_price: float,
    state: HoldingState,
    ts_trigger: float,   # e.g. 0.20 (20% gain threshold)
    ts_trail: float,     # e.g. 0.18 (18% below HWM)
) -> ExitSignal:
    """BULL_CALM trailing stop — armed once peak gain crosses trigger.

    Uses peak gain (HWM-based) not current gain — stays armed after pullbacks.
    """
    if ts_trigger <= 0 or ts_trail <= 0 or state.entry_price <= 0:
        return _NO_EXIT
    peak_gain = (state.high_watermark - state.entry_price) / state.entry_price
    if peak_gain < ts_trigger:
        return _NO_EXIT
    trail_floor = state.high_watermark * (1 - ts_trail)
    if current_price <= trail_floor:
        return ExitSignal(
            should_exit=True,
            reason=f"trailing_stop trail_floor={trail_floor:.2f}",
            exit_type="trailing_stop",
        )
    return _NO_EXIT


def check_stop_loss(
    current_price: float,
    state: HoldingState,
    stop_pct: float,   # e.g. 0.15 (15% cumulative loss)
) -> ExitSignal:
    """Fixed cumulative stop-loss from entry price."""
    if stop_pct <= 0 or state.entry_price <= 0:
        return _NO_EXIT
    loss_pct = (state.entry_price - current_price) / state.entry_price
    if loss_pct >= stop_pct:
        return ExitSignal(
            should_exit=True,
            reason=f"stop_loss loss={loss_pct:.1%}",
            exit_type="stop_loss",
        )
    return _NO_EXIT


def check_single_day_loss(
    current_price: float,
    state: HoldingState,
    sdl_pct: float,   # e.g. 0.10 (10% single-day drop)
) -> ExitSignal:
    """Single-day loss gate — fires on intraday gap-downs vs previous close.

    Only meaningful in BULL_CALM (wide 15% cumulative stop).  Other regimes
    use a tight 5% cumulative stop so sdl_pct should be 0 there.
    """
    # Audit fix EX-LE-5 (Round 2 deep audit, 2026-04-25): defense in
    # depth — even though PH-1/PH-2 now coerces NaN prev_close to None,
    # this function should also fail-safe locally on non-finite inputs.
    import math
    if sdl_pct <= 0 or state.prev_close is None or state.prev_close <= 0:
        return _NO_EXIT
    if not math.isfinite(state.prev_close):
        return _NO_EXIT
    daily_drop = (state.prev_close - current_price) / state.prev_close
    if daily_drop >= sdl_pct:
        return ExitSignal(
            should_exit=True,
            reason=f"single_day_loss drop={daily_drop:.1%}",
            exit_type="single_day_loss",
        )
    return _NO_EXIT


def check_max_hold(
    today: datetime.date,
    state: HoldingState,
    max_hold: int,   # calendar days; 0 = disabled
) -> ExitSignal:
    """Forced exit after max_hold calendar days."""
    if max_hold <= 0:
        return _NO_EXIT
    days_held = (today - state.entry_date).days
    if days_held >= max_hold:
        return ExitSignal(
            should_exit=True,
            reason=f"max_hold days={days_held}",
            exit_type="max_hold",
        )
    return _NO_EXIT


def check_model_sell(
    model_action: str,    # "buy" | "hold" | "sell"
    state: HoldingState,
    consecutive_required: int,  # e.g. 3
    min_hold_days: int,         # model-sell blocked before this many days
    today: datetime.date,
) -> tuple[HoldingState, ExitSignal]:
    """Accumulate consecutive sell signals; exit when streak meets required.

    Streak only counts after min_hold_days.  Returns updated state and exit.
    """
    if min_hold_days > 0:
        days_held = (today - state.entry_date).days
        if days_held < min_hold_days:
            # Don't touch streak — can't have earned streak yet
            return state, _NO_EXIT

    if model_action == "sell":
        state.sell_streak += 1
    else:
        state.sell_streak = 0

    if state.sell_streak >= consecutive_required:
        return state, ExitSignal(
            should_exit=True,
            reason=f"model_sell streak={state.sell_streak}",
            exit_type="model_sell",
        )
    return state, _NO_EXIT


# ── Orchestrator ───────────────────────────────────────────────────────────────

def compute_exits(
    current_price: float,
    today: datetime.date,
    model_action: str,
    state: HoldingState,
    params: dict,
) -> tuple[ExitSignal, HoldingState]:
    """Run all exits in priority order; return first triggered signal.

    params keys (all optional, default disabled if absent or zero):
      trailing_stop_trigger_pct, trailing_stop_trail_pct  — trailing stop (BULL_CALM)
      stop_loss_pct                   — cumulative stop
      max_single_day_loss_pct         — single-day gate (BULL_CALM)
      max_hold_days                   — time exit
      lt_hold_gate_days               — suppress model-sell when approaching 1-year (tax)
      lt_hold_min_gain                — min unrealized gain required for tax gate (default 0.10)
      consecutive_sell_signals        — model sell streak threshold
      min_hold_days                   — model-sell blocked before N days
    """
    # Audit fix E-5 (Round 5, 2026-04-25): pre-fix, a NaN/inf current_price
    # silently corrupted high_watermark via `max(HWM, NaN) = NaN`. Once HWM
    # was NaN, every subsequent trailing-stop computation propagated NaN
    # → no exit ever fires for that position. Now: skip HWM update and
    # all other exit checks on non-finite price (caller's responsibility
    # to retry next bar with a valid price). Returning _NO_EXIT is the
    # safe choice — caller sees no signal vs corrupted state.
    import math
    if not math.isfinite(current_price):
        return _NO_EXIT, state
    # Audit fix EX-HWM (Round 2 deep audit, 2026-04-25): defense in
    # depth on the OTHER side of the HWM update. E-5 protected against
    # NaN propagating INTO HWM via `max(HWM, NaN_price)`. But HWM could
    # already be non-finite when we enter this function — e.g. read
    # back from a corrupted live_state.json that predates E-5, or a
    # legacy snapshot created when prev_close validation wasn't there.
    # Once HWM was NaN, peak_gain stayed NaN forever and trailing-stop
    # silently disabled itself for the lifetime of the position.
    # Now: when stored HWM is non-finite, reset it to current_price so
    # tracking restarts cleanly from this bar onward.
    if not math.isfinite(state.high_watermark):
        state.high_watermark = current_price
    state.high_watermark = max(state.high_watermark, current_price)

    # 1. Trailing stop
    sig = check_trailing_stop(
        current_price, state,
        float(params.get("trailing_stop_trigger_pct", 0)),
        float(params.get("trailing_stop_trail_pct",   0)),
    )
    if sig.should_exit:
        return sig, state

    # 2. Cumulative stop-loss
    sig = check_stop_loss(
        current_price, state,
        float(params.get("stop_loss_pct", 0)),
    )
    if sig.should_exit:
        return sig, state

    # 3. Single-day loss gate
    sig = check_single_day_loss(
        current_price, state,
        float(params.get("max_single_day_loss_pct", 0)),
    )
    if sig.should_exit:
        return sig, state

    # 4. Max hold
    sig = check_max_hold(
        today, state,
        int(params.get("max_hold_days", 0)),
    )
    if sig.should_exit:
        return sig, state

    # 5. Tax-aware hold gate — suppress model-sell near the 1-year LT threshold
    #    when the position has a meaningful unrealized gain worth protecting.
    #    Hard stops (trailing, cumulative, single-day) above still fire normally.
    lt_gate = int(params.get("lt_hold_gate_days", 0))
    if lt_gate > 0 and state.entry_price > 0:
        days_held      = (today - state.entry_date).days
        unrealized_gain = (current_price - state.entry_price) / state.entry_price
        lt_min_gain    = float(params.get("lt_hold_min_gain", 0.10))
        # Use config'd LT threshold, not hardcoded 365 (#18 in audit).
        lt_thresh_days = int(params.get("lt_hold_threshold_days", 365))
        if lt_gate <= days_held < lt_thresh_days and unrealized_gain >= lt_min_gain:
            # Still update sell streak so it's ready when the window passes
            state, _ = check_model_sell(
                model_action, state,
                int(params.get("consecutive_sell_signals", 3)),
                int(params.get("min_hold_days", 0)),
                today,
            )
            return _NO_EXIT, state

    # 6. Model sell streak
    state, sig = check_model_sell(
        model_action, state,
        int(params.get("consecutive_sell_signals", 3)),
        int(params.get("min_hold_days", 0)),
        today,
    )
    return sig, state
