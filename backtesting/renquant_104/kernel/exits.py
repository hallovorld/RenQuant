"""Exit-check pure functions — all 5 exit types + tax-aware hold gate.

Self-contained: only datetime, dataclasses.  No common/ imports.
Priority order (highest → lowest):
  1. trailing_stop   (regime-configured, peak-gain armed)
  2. stop_loss       (regime-configured cumulative loss from entry)
  3. single_day_loss (regime-configured drop from previous close)
  4. max_hold        (forced time exit)
  5. [tax_hold_gate] (suppresses model-sell near 1-year mark with unrealized gain)
  6. model_sell      (consecutive sell-signal streak)
"""
from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from functools import lru_cache


# ── G7: tax-lot tracking ──────────────────────────────────────────────────────

@dataclass
class TaxLot:
    """Individual tax lot for a position.

    Real-world brokerages track each buy as a separate lot with its own
    cost basis and acquisition date. The QP solver's tax-cost vector should
    mirror that: on a partial sell, only the actually-disposed lots'
    gain/loss matters, not a single average.

    Fields:
      shares: float  — remaining shares in this lot (post any partial
                       consumption from upstream sells).
      price:  float  — fill price per share for this lot.
      date:   date   — acquisition date (used for ST/LT classification).
    """
    shares: float
    price:  float
    date:   datetime.date


@dataclass(frozen=True)
class DisposedTaxLot:
    """A consumed slice of a tax lot from a sell event."""

    shares: float
    price: float
    date: datetime.date


@lru_cache(maxsize=4096)
def _is_nyse_trading_day(d: datetime.date) -> bool:
    """Return True iff d is a regular NYSE trading day (Mon-Fri, not a US
    market holiday).

    Uses pandas_market_calendars (already in requirements.lock.txt) for
    holiday calendar — same source as scripts/daily_104.sh's NYSE guard.
    Lazy-imported + LRU-cached so the cold-start cost is paid once.

    Audit fix STREAK-TRADING-DAY (2026-04-26 round-7): per user spec,
    sell streak should ONLY count trading days. Sundays / Saturdays /
    market holidays do NOT increment the streak.
    """
    # Cheap weekday check first
    if d.weekday() >= 5:   # 5=Sat, 6=Sun
        return False
    try:
        import pandas_market_calendars as mcal  # noqa: PLC0415
        nyse = mcal.get_calendar("XNYS")
        sched = nyse.schedule(start_date=d, end_date=d)
        return not sched.empty
    except Exception:
        # Defensive — if pmc unavailable, fall back to weekday-only check.
        return True   # weekday already passed; assume trading day


@lru_cache(maxsize=4096)
def nyse_trading_days_between(start: datetime.date, end: datetime.date) -> int:
    """Count NYSE sessions after ``start`` through and including ``end``.

    Entry day is day zero. This matches forward-label horizons such as
    `fwd_60d_excess`: a position bought on T has completed one thesis day on
    the next NYSE session, not on the next calendar day.
    """
    if not isinstance(start, datetime.date) or not isinstance(end, datetime.date):
        return 0
    if end <= start:
        return 0
    try:
        import pandas_market_calendars as mcal  # noqa: PLC0415

        nyse = mcal.get_calendar("XNYS")
        sessions = nyse.valid_days(
            start_date=start + datetime.timedelta(days=1),
            end_date=end,
        )
        return int(len(sessions))
    except Exception:
        pass
    cur = start + datetime.timedelta(days=1)
    count = 0
    while cur <= end:
        if _is_nyse_trading_day(cur):
            count += 1
        cur += datetime.timedelta(days=1)
    return count


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
    # Audit fix STREAK-DAY-DEDUP (2026-04-26 round-5): track date of
    # last streak increment so multiple bars per calendar day (testing,
    # intraday sell-only runs) don't double-count. None means never
    # incremented yet.
    last_streak_inc_date: datetime.date | None = None
    prev_close:      float | None = None
    rank_score:      float | None = None   # latest calibrated probability (set by ScoreModelTask)
    expected_return: float | None = None   # latest E[R-SPY] over rotation horizon
    expected_return_horizon_days: int | None = None
    panel_score:     float | None = None   # latest cross-sectional panel-LTR score (set by PanelScoringJob)
    mu:              float | None = None   # latest NGBoost μ (set by PanelScoringJob)
    mu_horizon_days: int | None = None
    sigma:           float | None = None   # latest NGBoost σ (set by PanelScoringJob, fwd-5d)
    # 2026-05-10: realized daily-return std (20d rolling) — fallback when
    # NGBoost is OFF in prod so σ-aware exits (stop_loss / single_day_loss)
    # remain active. Written by PrepareHoldingTask each bar. Resolves to
    # daily vol directly; NGB sigma needs /√5 conversion (see
    # `_resolve_daily_sigma`). Revives Fix #0a (was dead-code per
    # AUDIT_2026-05-09 #1) by removing the NGB dependency.
    realized_sigma_daily: float | None = None
    # 2026-05-11 L5 experiment: Wilder-smoothed ATR(14) for ATR-based trailing
    # (Wilder 1978 "New Concepts in Technical Trading Systems" + Le Beau 1993
    # "Chandelier exit"). Used by check_trailing_stop when atr_n_multiplier > 0.
    # Written by PrepareHoldingTask each bar from last 14+ bars of high/low/close.
    realized_atr_daily: float | None = None
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
    # Regime thesis at fresh entry. Tenure rules such as max_hold_days should
    # remain anchored to the entry thesis; current-regime risk rules can still
    # adapt each bar.
    entry_regime:        str | None = None

    # G7 (2026-05-04): explicit tax-lot list. Each buy appends a TaxLot;
    # sells consume lots per FIFO/HIFO. Default empty for back-compat;
    # `ensure_lots()` synthesizes a single lot from the legacy
    # entry_price/entry_date/shares fields when the list is empty.
    lots: list = field(default_factory=list)

    # Lot-level helpers ────────────────────────────────────────────────
    def total_shares(self) -> float:
        """Sum of shares across all lots. Falls back to `self.shares`
        when lots are empty (legacy / un-migrated holdings)."""
        if self.lots:
            return float(sum(L.shares for L in self.lots))
        return float(self.shares or 0.0)

    def weighted_avg_entry_price(self) -> float:
        """Cost-basis-weighted average entry price across all lots.
        Falls back to `self.entry_price` when lots are empty."""
        if not self.lots:
            return float(self.entry_price or 0.0)
        total_sh = sum(L.shares for L in self.lots)
        if total_sh <= 0:
            return float(self.entry_price or 0.0)
        cost = sum(L.shares * L.price for L in self.lots)
        return float(cost / total_sh)


def ensure_lots(hs) -> None:
    """Migrate a legacy HoldingState to the lot model in-place.

    If `hs.lots` is already populated, no-op. Otherwise, when the legacy
    fields describe a real position (shares > 0 AND entry_price > 0),
    synthesize a single `TaxLot` from them. This keeps the QP HIFO path
    correct on holdings that haven't been touched by a lot-aware buy yet
    (e.g. positions hydrated from broker on adapter startup).

    Idempotent and cheap — call freely at the head of any consumer.

    Defensive: accepts any HoldingState-like object. If the object is
    missing a `lots` attribute (test stubs / legacy snapshot), we
    auto-attach an empty list before populating.
    """
    if not hasattr(hs, "lots") or hs.lots is None:
        try:
            hs.lots = []
        except (AttributeError, TypeError):
            return   # frozen / unsettable — caller's stub, skip silently
    if hs.lots:
        return
    sh = float(getattr(hs, "shares", 0.0) or 0.0)
    px = float(getattr(hs, "entry_price", 0.0) or 0.0)
    ed = getattr(hs, "entry_date", None)
    if sh > 0 and px > 0 and ed is not None:
        hs.lots.append(TaxLot(shares=sh, price=px, date=ed))


def apply_buy_lot(hs: HoldingState, shares: float, price: float,
                   date: datetime.date) -> None:
    """Append a new TaxLot to `hs.lots` and refresh the legacy fields.

    `entry_price` is recomputed as the weighted average across all lots
    (back-compat for code paths that read it). `entry_date` is left at
    the FIRST lot's date so tenure-based rules (max_hold, lt_hold_gate)
    track the original acquisition. This mirrors broker convention:
    "first acquired" anchors hold-period reporting.
    """
    if shares <= 0 or price <= 0:
        return
    ensure_lots(hs)
    hs.lots.append(TaxLot(shares=float(shares), price=float(price), date=date))
    hs.entry_price = hs.weighted_avg_entry_price()
    if not hs.lots[:-1]:   # this was the first lot
        hs.entry_date = date


def apply_sell_lots_detailed(
    hs: HoldingState,
    shares_to_sell: float,
    method: str = "fifo",
) -> tuple[float, float, list[DisposedTaxLot]]:
    """Consume lots and return cost basis plus disposed lot slices.

    Return tuple: ``(proceeds_basis, realized_gain_dollar, disposed_lots)``.
    ``realized_gain_dollar`` remains a legacy placeholder because the caller
    owns the sell price. ``disposed_lots`` carries the acquisition date for
    each consumed lot slice so tax age can match the same lots used for basis.
    """
    if shares_to_sell <= 0 or not hs.lots:
        return 0.0, 0.0, []
    method_norm = (method or "fifo").lower()
    disposed: list[DisposedTaxLot] = []
    if method_norm == "hifo":
        # sort copy so we don't mutate the user-visible order until
        # we actually consume; pop highest-price lot first.
        order = sorted(range(len(hs.lots)), key=lambda i: -hs.lots[i].price)
    elif method_norm == "avg":
        # avg method: trim each lot proportionally to its share weight.
        total = sum(L.shares for L in hs.lots)
        if total <= 0:
            return 0.0, 0.0, []
        take_frac = min(1.0, shares_to_sell / total)
        basis = 0.0
        for L in hs.lots:
            t = L.shares * take_frac
            if t <= 0:
                continue
            basis += t * L.price
            disposed.append(DisposedTaxLot(shares=t, price=L.price, date=L.date))
            L.shares -= t
        hs.lots = [L for L in hs.lots if L.shares > 1e-9]
        return basis, 0.0, disposed
    else:   # FIFO (default)
        order = list(range(len(hs.lots)))

    remaining = float(shares_to_sell)
    basis = 0.0
    for idx in order:
        if remaining <= 1e-12:
            break
        L = hs.lots[idx]
        take = min(L.shares, remaining)
        if take <= 0:
            continue
        basis += take * L.price
        disposed.append(DisposedTaxLot(shares=take, price=L.price, date=L.date))
        L.shares -= take
        remaining -= take
    # Drop any lot whose remaining shares are below a numerical floor.
    hs.lots = [L for L in hs.lots if L.shares > 1e-9]
    return basis, 0.0, disposed


def apply_sell_lots(hs: HoldingState, shares_to_sell: float,
                     method: str = "fifo") -> tuple[float, float]:
    """Consume lots from `hs.lots` per FIFO/HIFO; return (proceeds_basis,
    realized_gain_dollar) where:
      - proceeds_basis is the cost-basis $ disposed (sum lot.price*take)
      - realized_gain_dollar requires caller to add (sell_price * shares)
        and subtract proceeds_basis. We return cost basis here so the
        caller can compute gain at its own sell_price.

    Modifies `hs.lots` in place. If the request exceeds total lot shares,
    consumes everything available and returns whatever was matched.

    Caller is responsible for updating `hs.shares` / legacy `entry_price`
    after this call (or rely on the helpers in HoldingState). When
    `method == 'avg'` we still consume FIFO (legacy avg-cost mutation
    ignored lots entirely; here we keep books consistent by trimming
    proportionally).
    """
    basis, realized, _disposed = apply_sell_lots_detailed(
        hs, shares_to_sell, method
    )
    return basis, realized


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
    # Diagnostic contract: the exact exit-parameter snapshot used by the
    # sell decision. Adapters persist this into trade logs so decision trees
    # show applied rules, not merely the current regime defaults.
    exit_params: dict | None = None


_NO_EXIT = ExitSignal(should_exit=False, reason="", exit_type="")


# ── Individual exit checks ─────────────────────────────────────────────────────

def check_take_profit(
    current_price: float,
    state: HoldingState,
    take_profit_pct: float,  # e.g. 0.25 (exit at +25% gain)
) -> ExitSignal:
    """Hard take-profit — exits unconditionally once cumulative gain ≥ threshold.

    Audit 2026-04-29: the exit chain had no take-profit rule. Positions with
    +30%+ gain held indefinitely unless the model reversed or max_hold fired,
    leaving them exposed to mean reversion. Hard take-profit locks in gains
    at a configurable threshold.

    Configured via regime_params.take_profit_pct (default 0 = disabled).
    Runs BEFORE trailing stop so it fires on the way up, not only on pullback.
    """
    import math
    # 2026-05-04 audit Issue 18 fix: NaN entry_price slipped past `<= 0`
    # (NaN comparisons all False) → gain = (px - NaN)/NaN = NaN → no
    # exit. Same NaN-slip class as SE-1/EX-LE-5/SL-2. Defense in depth.
    if (take_profit_pct <= 0
            or not math.isfinite(state.entry_price)
            or state.entry_price <= 0):
        return _NO_EXIT
    gain = (current_price - state.entry_price) / state.entry_price
    if gain >= take_profit_pct:
        return ExitSignal(
            should_exit=True,
            reason=f"take_profit gain={gain:.1%} >= threshold={take_profit_pct:.1%}",
            exit_type="take_profit",
        )
    return _NO_EXIT


def check_trailing_stop(
    current_price: float,
    state: HoldingState,
    ts_trigger: float,        # e.g. 0.20 (20% gain threshold)
    ts_trail: float,          # e.g. 0.18 (18% below HWM)
    atr_n_multiplier: float = 0.0,  # L5: Chandelier multiplier (Le Beau k≈3)
) -> ExitSignal:
    """BULL_CALM trailing stop — armed once peak gain crosses trigger.

    Uses peak gain (HWM-based) not current gain — stays armed after pullbacks.

    L5 (Wilder 1978 + Le Beau 1993): when ``atr_n_multiplier > 0`` and the
    holding has a finite ``realized_atr_daily``, the effective trail-pct
    becomes ``max(ts_trail, k × ATR / HWM)`` — the Chandelier exit. This
    adapts the trail width to per-ticker realized range instead of a fixed
    %, so high-volatility names (NVDA σ_daily ≈ 5%) don't whipsaw on noise.
    """
    # 2026-05-11 audit (A-2): mirror check_take_profit:288 + check_stop_loss:398
    # NaN/inf entry_price guard. Pre-fix, a corrupted entry_price silently
    # bypassed `<= 0` (NaN comparisons all False) and propagated NaN through
    # peak_gain → `< ts_trigger` False → trailing never armed → exit dead.
    import math  # noqa: PLC0415 — match check_take_profit / check_stop_loss pattern
    if (ts_trigger <= 0
            or ts_trail <= 0
            or not math.isfinite(state.entry_price)
            or state.entry_price <= 0):
        return _NO_EXIT
    peak_gain = (state.high_watermark - state.entry_price) / state.entry_price
    if peak_gain < ts_trigger:
        return _NO_EXIT

    # L5 ATR-based widening (Wilder 1978 §9, Le Beau 1993 Chandelier exit).
    # Effective trail = max(legacy_pct, k × ATR / HWM). When ATR is missing
    # or atr_n_multiplier is 0, legacy fixed pct governs (backward compat).
    trail_pct = float(ts_trail)
    if atr_n_multiplier and atr_n_multiplier > 0:
        atr = getattr(state, "realized_atr_daily", None)
        if (atr is not None
                and math.isfinite(atr) and atr > 0
                and math.isfinite(state.high_watermark)
                and state.high_watermark > 0):
            atr_trail = float(atr_n_multiplier) * atr / state.high_watermark
            if math.isfinite(atr_trail) and atr_trail > trail_pct:
                trail_pct = atr_trail

    trail_floor = state.high_watermark * (1 - trail_pct)
    if current_price <= trail_floor:
        return ExitSignal(
            should_exit=True,
            reason=(f"trailing_stop trail_floor={trail_floor:.2f} "
                    f"(trail_pct={trail_pct:.1%})"),
            exit_type="trailing_stop",
        )
    return _NO_EXIT


def _resolve_daily_sigma(state: "HoldingState") -> float | None:
    """Resolve per-position daily volatility from NGBoost σ (preferred) or
    realized-vol fallback.

    Priority:
      1. ``state.sigma`` (NGBoost 5-day σ) → daily = σ / √5
      2. ``state.realized_sigma_daily`` (20-day realized) → use as-is
      3. ``None`` when neither available

    Reviving Fix #0a (σ-aware stops, was dead-code per AUDIT_2026-05-09 #1):
    NGB is OFF in production so state.sigma is always None; the realized-vol
    fallback (computed in PrepareHoldingTask from last 20d daily returns)
    makes σ-aware exits actually fire. Industry-standard volatility-scaled
    risk control (Almgren-Chriss 2000; Edwards-Magee 1948; RiskMetrics
    1996 — daily-σ as the canonical risk unit).
    """
    import math  # noqa: PLC0415 — local import keeps helper self-contained
    sg = getattr(state, "sigma", None)
    if sg is not None:
        try:
            sgf = float(sg)
            if math.isfinite(sgf) and sgf > 0:
                return sgf / math.sqrt(5.0)
        except (TypeError, ValueError):
            pass
    rs = getattr(state, "realized_sigma_daily", None)
    if rs is not None:
        try:
            rsf = float(rs)
            if math.isfinite(rsf) and rsf > 0:
                return rsf
        except (TypeError, ValueError):
            pass
    return None


def check_stop_loss(
    current_price: float,
    state: HoldingState,
    stop_pct: float,   # e.g. 0.15 (15% cumulative loss) — legacy absolute floor
    stop_n_sigma: float = 0.0,   # N × daily σ × √hold_days (σ-adaptive ceiling)
    today: datetime.date | None = None,
    stop_decay_days: int = 0,    # B1: after N days held, linearly tighten stop_pct (0 = off)
    stop_decay_floor: float = 0.5,  # B1: minimum multiplier (0.5 = stop tightens to 50% of original)
) -> ExitSignal:
    """Cumulative stop-loss from entry price — absolute and/or σ-adaptive.

    Two threshold modes (effective = max of both):
      stop_pct (legacy)    absolute fraction of entry (e.g. 0.15 = 15%)
      stop_n_sigma (new)   N × daily_σ × √hold_days — scales with the
                           ticker's own daily vol AND the cumulative drift
                           over the holding window. High-σ stocks get
                           wider stops (no noise-day exits); long-held
                           positions get wider stops as the noise band
                           accumulates with sqrt(t).

    Industry refs:
      Almgren-Chriss 2000 (Optimal execution): risk in σ-units, scales √t.
      Edwards-Magee 1948 (Technical Analysis of Stock Trends, Ch. 28):
        stop placement at "N-σ-day move" beyond entry.
      RiskMetrics 1996 (J.P. Morgan): daily-σ is the canonical risk unit;
        N-σ band = N × σ_daily × √horizon for cumulative drift.

    Revives Fix #0a (σ-aware stop_loss) which was rolled back as dead code
    per AUDIT_2026-05-09 #1. Root cause was σ source (NGB OFF in prod);
    `_resolve_daily_sigma` adds realized-vol fallback so σ-aware works
    independently of NGB.

    2026-05-04 audit Issue 19 fix: NaN entry_price slipped past `<= 0`
    → loss_pct = NaN → no stop ever fires. Same NaN-slip class as
    Issue 18 (check_take_profit). Defense in depth.
    """
    import math  # noqa: PLC0415
    if not math.isfinite(state.entry_price) or state.entry_price <= 0:
        return _NO_EXIT

    abs_thresh = float(stop_pct) if stop_pct and stop_pct > 0 else 0.0

    # B1 (2026-05-12 revival) — time-decay tightening
    # After `stop_decay_days` days, linearly reduce abs_thresh toward
    # `stop_decay_floor × abs_thresh`. Catches bleeders that have been
    # held too long without the model deciding to exit.
    # Disabled when stop_decay_days <= 0.
    if (stop_decay_days and stop_decay_days > 0 and abs_thresh > 0
            and today is not None and state.entry_date is not None):
        try:
            held = max(0, (today - state.entry_date).days)
            if held > stop_decay_days:
                # Linear decay from 1.0 at decay_days → floor at 2×decay_days
                excess = (held - stop_decay_days) / float(stop_decay_days)
                multiplier = max(float(stop_decay_floor), 1.0 - excess * (1.0 - float(stop_decay_floor)))
                abs_thresh = abs_thresh * multiplier
        except (TypeError, ValueError):
            pass

    sigma_thresh = 0.0
    if stop_n_sigma and stop_n_sigma > 0:
        sg_daily = _resolve_daily_sigma(state)
        if sg_daily is not None:
            if today is not None:
                days_held = max(1, (today - state.entry_date).days)
            else:
                days_held = 1
            # 2026-05-11 audit (A-9): cap √t at √20 (~4.47×) so σ-band doesn't
            # grow unbounded over hold time. Almgren-Chriss σ × √t is correct
            # for cumulative Brownian drift, but at t=250d the band becomes
            # ~50% and effectively disables the stop. Capping at 20d aligns
            # with the realized-σ window (PrepareHoldingTask 20d rolling)
            # and keeps σ-aware meaningful through the full hold horizon.
            days_capped = min(days_held, 20)
            sigma_thresh = float(stop_n_sigma) * sg_daily * math.sqrt(float(days_capped))

    threshold = max(abs_thresh, sigma_thresh)
    if threshold <= 0:
        return _NO_EXIT

    # 2026-05-13 Long-Short Phase 2A: detect short position via state.shares<0.
    # For longs: loss when price < entry. For shorts: loss when price > entry.
    # Flipping the sign in loss_pct gives the same |loss| metric for either side.
    is_short = float(getattr(state, "total_shares", lambda: state.shares)() if callable(getattr(state, "total_shares", None)) else state.shares) < 0
    if is_short:
        loss_pct = (current_price - state.entry_price) / state.entry_price
    else:
        loss_pct = (state.entry_price - current_price) / state.entry_price
    if loss_pct >= threshold:
        return ExitSignal(
            should_exit=True,
            reason=(f"stop_loss{' [SHORT]' if is_short else ''} "
                    f"loss={loss_pct:.1%} ≥ {threshold:.1%} "
                    f"(abs={abs_thresh:.1%} / σN={sigma_thresh:.1%})"),
            exit_type="stop_loss",
        )
    return _NO_EXIT


def check_single_day_loss(
    current_price: float,
    state: HoldingState,
    sdl_pct: float,    # absolute %: e.g. 0.06 (6% single-day drop)
    sdl_n_sigma: float = 0.0,   # N × daily realized vol (preferred when set)
    sdl_skip_if_unrealized_above: float = 0.0,  # B2: skip SDL if position is up X% (default 0 = off)
) -> ExitSignal:
    """Single-day loss gate — fires on gap-downs vs previous close.

    Two threshold modes:
      sdl_pct (legacy)    absolute % of prev_close (e.g. 0.06 = 6%)
      sdl_n_sigma (new)   N × per-ticker daily realized vol, derived from
                          state.sigma (NGBoost's predicted 5-day σ);
                          daily_vol = sigma / sqrt(5)
    When both are configured, the EFFECTIVE threshold is
    max(absolute, N×σ_daily) — i.e. we use whichever is more permissive,
    so high-vol names don't panic on a normal noise day. Set sdl_pct=0
    AND sdl_n_sigma>0 for fully σ-adaptive behaviour.

    2026-05-04 motivation: B2 holdout showed `single_day_loss` had
    win_rate 40% / median pnl −5.2% — high-vol stocks (NVDA/RBLX/etc.
    daily σ ≈ 4-5%) tripped the absolute 6% threshold on noise days
    and crystallized losses on positions that would have recovered.
    σ-scaled threshold aligns the gate with each ticker's own
    volatility.
    """
    import math
    if state.prev_close is None or state.prev_close <= 0:
        return _NO_EXIT
    if not math.isfinite(state.prev_close):
        return _NO_EXIT

    # B2 (2026-05-12 revival) — skip SDL if position is currently a winner
    # by more than `sdl_skip_if_unrealized_above`. Industry rationale: a
    # 6%-down day on a stock that's up 20% from entry is noise, not signal;
    # the position has cushion and SDL would prematurely realize the win.
    # Disabled when threshold ≤ 0 (default).
    if (sdl_skip_if_unrealized_above and sdl_skip_if_unrealized_above > 0
            and math.isfinite(state.entry_price) and state.entry_price > 0):
        unrealized = (current_price - state.entry_price) / state.entry_price
        if math.isfinite(unrealized) and unrealized >= float(sdl_skip_if_unrealized_above):
            return _NO_EXIT

    abs_thresh = float(sdl_pct) if sdl_pct and sdl_pct > 0 else 0.0
    sigma_thresh = 0.0
    if sdl_n_sigma and sdl_n_sigma > 0:
        # 2026-05-10: route through _resolve_daily_sigma so realized-vol
        # fallback is used when NGB is OFF (NGB σ unavailable). Industry-
        # standard daily-σ resolution per Almgren-Chriss / RiskMetrics.
        daily_vol = _resolve_daily_sigma(state)
        if daily_vol is not None:
            sigma_thresh = float(sdl_n_sigma) * daily_vol

    threshold = max(abs_thresh, sigma_thresh)
    if threshold <= 0:
        return _NO_EXIT

    # 2026-05-13 Long-Short Phase 2A: short positions take losses on UP moves.
    is_short = float(getattr(state, "shares", 0.0) or 0.0) < 0
    if is_short:
        daily_drop = (current_price - state.prev_close) / state.prev_close
    else:
        daily_drop = (state.prev_close - current_price) / state.prev_close
    if daily_drop >= threshold:
        return ExitSignal(
            should_exit=True,
            reason=f"single_day_loss{' [SHORT]' if is_short else ''} drop={daily_drop:.1%} ≥ "
                    f"{threshold:.1%} (abs={abs_thresh:.1%} / "
                    f"σN={sigma_thresh:.1%})",
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
        days_held = nyse_trading_days_between(state.entry_date, today)
        if days_held < min_hold_days:
            # Don't touch streak — can't have earned streak yet
            return state, _NO_EXIT

    # Audit fix STREAK-DAY-DEDUP (2026-04-26 round-5): increment AT MOST
    # ONCE per calendar day.
    #
    # Audit fix STREAK-TRADING-DAY (2026-04-26 round-7, after user spec):
    # ALSO require `today` to be an NYSE TRADING day before incrementing.
    # Pre-fix, e2e on Sunday 2026-04-26 (calendar day, market closed)
    # incremented streak from 2 → 3 → triggered model_sell on GOOG/AMZN/BA
    # within 24 hours. User: "today is not a trading day, streak shouldn't
    # be > 1!" Fix: skip increment + skip reset on non-trading days. The
    # streak should reflect TRADING-day signals only.
    is_trading_day = _is_nyse_trading_day(today)
    if not is_trading_day:
        # Sunday / market holiday — leave streak unchanged. Don't reset
        # either (otherwise a Sun e2e would clear a legitimate streak).
        pass
    elif model_action == "sell":
        if state.last_streak_inc_date != today:
            state.sell_streak += 1
            state.last_streak_inc_date = today
        # If same trading day, leave streak unchanged (idempotent)
    else:
        state.sell_streak = 0
        # Don't touch last_streak_inc_date on reset — it's only for INC dedup

    # Audit fix STREAK-TRADING-DAY ROUND 2 (2026-04-26 round-7, after user
    # spec round 2: "怎么他妈的还有streak sell！"). Strengthening: model_sell
    # must NOT FIRE on a non-trading day either. The original fix prevented
    # INCREMENT on Sunday; today's e2e showed that doesn't help when the
    # streak was already at threshold from a buggy prior run — fire still
    # happened. This guard makes the rule symmetric: no streak movement
    # AND no streak fire on non-trading days.
    #
    # Path-dependent rules (stop_loss, trailing, SDL, max_hold) are NOT
    # affected — those go through compute_exits's other branches and
    # represent risk management that must always fire.
    if not is_trading_day:
        return state, _NO_EXIT

    if state.sell_streak >= consecutive_required:
        return state, ExitSignal(
            should_exit=True,
            reason=f"model_sell streak={state.sell_streak}",
            exit_type="model_sell",
        )
    return state, _NO_EXIT


def effective_model_sell_min_hold_days(
    params: dict,
    state: HoldingState,
    current_price: float,
) -> int:
    """Resolve the configured model-driven sell hold floor.

    ``min_hold_days`` is the base floor. Optional profit/loss floors are
    stricter soft-exit guards: they only affect model-driven sells and never
    block path-risk exits that already fired earlier in ``compute_exits``.
    """
    base = int(params.get("min_hold_days", 0))
    try:
        entry_price = float(state.entry_price)
        price = float(current_price)
    except (TypeError, ValueError):
        return base
    if entry_price <= 0 or not math.isfinite(price):
        return base
    key = "min_hold_profit_days" if price > entry_price else "min_hold_loss_days"
    return max(base, int(params.get(key, 0) or 0))


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
      min_hold_profit_days/loss_days  — optional stricter model-sell floors
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

    # 0. Take-profit (hard ceiling on gain — runs before trailing stop)
    sig = check_take_profit(
        current_price, state,
        float(params.get("take_profit_pct", 0)),
    )
    if sig.should_exit:
        return sig, state

    # 1. Trailing stop
    sig = check_trailing_stop(
        current_price, state,
        float(params.get("trailing_stop_trigger_pct", 0)),
        float(params.get("trailing_stop_trail_pct",   0)),
        float(params.get("atr_n_multiplier",          0)),
    )
    if sig.should_exit:
        return sig, state

    # 2. Cumulative stop-loss (legacy absolute % AND/OR σ-adaptive)
    sig = check_stop_loss(
        current_price, state,
        float(params.get("stop_loss_pct", 0)),
        float(params.get("stop_n_sigma", 0)),
        today,
        stop_decay_days=int(params.get("stop_decay_days", 0)),
        stop_decay_floor=float(params.get("stop_decay_floor", 0.5)),
    )
    if sig.should_exit:
        return sig, state

    # 3. Single-day loss gate (absolute % AND/OR σ-scaled threshold).
    sig = check_single_day_loss(
        current_price, state,
        float(params.get("max_single_day_loss_pct", 0)),
        float(params.get("sdl_n_sigma", 0)),
        sdl_skip_if_unrealized_above=float(params.get("sdl_skip_if_unrealized_above", 0)),
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
                effective_model_sell_min_hold_days(params, state, current_price),
                today,
            )
            return _NO_EXIT, state

    # 6. Model sell streak
    state, sig = check_model_sell(
        model_action, state,
        int(params.get("consecutive_sell_signals", 3)),
        effective_model_sell_min_hold_days(params, state, current_price),
        today,
    )
    return sig, state
