"""Per-ticker sell evaluation tasks."""
from __future__ import annotations

import datetime
import logging

from .context import TickerInferenceContext
from .pipeline import Task
from .soft_exit_guards import (
    lt_gate_suppression,
    resolve_current_price,
    soft_exit_horizon_suppression,
    tax_adjusted_soft_exit_suppression,
)

log = logging.getLogger("kernel.pipeline.sell")


# Module-level taxonomy of exit types — single source of truth so the
# earnings-blackout veto, the SellGateB μ/σ guard, and any future
# sector-conditional or regime-conditional sell gate all classify exits
# the same way. Adding a new exit_type? Add it here and decide whether
# it's MODEL- or PATH-driven.
#
# MODEL_DRIVEN_EXIT_TYPES — exits derived from a model output (panel
# rank, NGBoost μ/σ, streak counter on per-ticker tournament). These
# are SUPPRESSED inside the earnings blackout window because the model
# can't see the upcoming/just-released fundamental information.
#
# PATH_DRIVEN_EXIT_TYPES — exits derived from price-action invariants
# (stop_loss, trailing, single-day-loss, max-hold) or portfolio-level
# rebalance (kelly_trim, rotation). These ALWAYS fire even inside the
# earnings window — the price has already moved against us, the
# blackout doesn't change that.
# Canonical exit-type taxonomy (CLAUDE.md §5.13.5).
# Refactored 2026-05-11 — kernel/exit_types owns the lookup.
from kernel.exit_types import MODEL_DRIVEN as MODEL_DRIVEN_EXIT_TYPES  # noqa: E402
from kernel.exit_types import PATH_DRIVEN_LEGACY as PATH_DRIVEN_EXIT_TYPES  # noqa: E402


class PrepareHoldingTask(Task):
    """Validate holding + price; attach prev_close.

    Audit fix PH-1/PH-2 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
    tc.price slipped past `<= 0` (NaN comparisons False) → downstream
    exit checks ran on NaN prices and silently failed. Same with
    prev_close from `iloc[-2]` — could be NaN if data has gaps.
    Now: explicit isfinite + > 0 guard on price; coerce NaN prev_close
    to None so check_single_day_loss can short-circuit cleanly.
    """

    def run(self, tc: TickerInferenceContext) -> bool | None:
        import math
        if tc.holding is None:
            return False

        if not math.isfinite(tc.price) or tc.price <= 0:
            log.warning(
                "PrepareHoldingTask: price=%s for %s — skipping",
                tc.price, tc.ticker,
            )
            return False

        stock_df = tc.ohlcv.get(tc.ticker)
        if stock_df is None:
            return False

        if len(stock_df) >= 2:
            pc = float(stock_df["close"].iloc[-2])
            tc.holding.prev_close = pc if math.isfinite(pc) else None
        else:
            tc.holding.prev_close = None

        # 2026-05-10: realized daily-return std (20d rolling) — fallback σ
        # source for σ-aware exits when NGBoost is OFF in production
        # (state.sigma is None). Revives Fix #0a per AUDIT_2026-05-09 #1.
        # Window: 20 trading days (≈ 1 month), matches RiskMetrics 1996
        # canonical daily-σ horizon and most Qlib volatility indicators.
        # Defensive: needs ≥10 valid returns to emit; else None.
        if len(stock_df) >= 21:
            recent_close = stock_df["close"].iloc[-21:]
            rets = recent_close.pct_change().dropna()
            if len(rets) >= 10:
                rv = float(rets.std(ddof=1))
                if math.isfinite(rv) and rv > 0:
                    tc.holding.realized_sigma_daily = rv
                else:
                    tc.holding.realized_sigma_daily = None
            else:
                tc.holding.realized_sigma_daily = None
        else:
            tc.holding.realized_sigma_daily = None

        # 2026-05-11 L5: Wilder-smoothed ATR(14) for ATR-based trailing stop
        # (Wilder 1978 "New Concepts in Technical Trading Systems" §9, Le Beau
        # 1993 Chandelier exit). Delegated to kernel.indicators.compute_atr —
        # the project's single source of truth for ATR, matching the existing
        # ADX implementation in kernel/regime.py.
        from kernel.indicators import compute_atr  # noqa: PLC0415
        period = 14
        if len(stock_df) >= period + 1:
            atr_series = compute_atr(
                stock_df["high"].astype(float),
                stock_df["low"].astype(float),
                stock_df["close"].astype(float),
                period=period,
            ).dropna()
            if not atr_series.empty:
                atr_val = float(atr_series.iloc[-1])
                tc.holding.realized_atr_daily = (
                    atr_val if math.isfinite(atr_val) and atr_val > 0 else None
                )
            else:
                tc.holding.realized_atr_daily = None
        else:
            tc.holding.realized_atr_daily = None


class ScoreModelTask(Task):
    """Build feature frame and score model → tc.model_action."""

    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.models     import score_artifact       # noqa: PLC0415
        from kernel.indicators import build_feature_frame  # noqa: PLC0415

        spy_df   = tc.ohlcv.get("SPY")
        stock_df = tc.ohlcv.get(tc.ticker)

        if tc.model is None or spy_df is None or stock_df is None:
            tc.model_action = "hold"
            return

        # Feature cache optimization (2026-04-24): use pre-built frame
        # if available (SimAdapter populates via make_context), otherwise
        # fall back to per-bar rebuild (live runner path).
        cached = getattr(tc, "feature_cache_frame", None)
        if cached is not None and not cached.empty:
            tc.features = cached.loc[:tc.today]
        else:
            spec    = tc.config.get("indicator_spec", {})
            vol_win = int(tc.config.get("regime", {}).get("vol_realized_window", 20))
            tc.features = build_feature_frame(stock_df, spy_df, spec, vol_win)

        if tc.features is not None and not tc.features.empty:
            rotation_horizon = int(tc.config.get("rotation", {}).get("target_horizon_days", 20))
            sr = score_artifact(
                tc.model, tc.features.iloc[-1],
                holdings=1, horizon_days=rotation_horizon,
            )
            tc.model_action = sr.signal
            if tc.holding is not None:
                tc.holding.rank_score      = float(sr.rank_score)
                tc.holding.expected_return = float(sr.expected_return)
        else:
            tc.model_action = "hold"

        log.debug("ScoreModelTask [%s]: action=%s", tc.ticker, tc.model_action)


class EvaluateExitsTask(Task):
    """Run the 5-exit priority chain; update tc.holding and tc.exit_signal."""

    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.exits import compute_exits  # noqa: PLC0415

        sig, updated_hs = compute_exits(
            tc.price, tc.today, tc.model_action, tc.holding, tc.exit_params
        )
        tc.holding = updated_hs

        if sig.should_exit:
            sig.exit_params = dict(tc.exit_params or {})
            tc.exit_signal = sig
        elif tc.model_action == "sell" and updated_hs.sell_streak > 0:
            # Use the typed field on ExitSignal (was a dynamic attribute write
            # before audit #17 — easier for static analysis + tests now).
            sig.blocked_streak = True
            tc.exit_signal = sig

        log.debug("EvaluateExitsTask [%s]: should_exit=%s  type=%s",
                  tc.ticker, sig.should_exit, getattr(sig, "exit_type", None))


class SellGateBTask(Task):
    """Sell-side Gate B (NGBoost edge-Sharpe guard) — mirror of buy-side Gate B.

    References:
      Lo, A.W. (2002). "The Statistics of Sharpe Ratios", Financial
        Analysts Journal 58(4): 36-52. — instantaneous-edge Sharpe
        criterion μ/σ used by buy-side Gate B; this is the symmetric
        sell-side analog.
      Grinold, R.C. & Kahn, R.N. (1999). Active Portfolio Management
        (2nd ed.), McGraw-Hill. Ch. 5: information ratio = α/ω as the
        signal-strength threshold for action.
      Kahneman & Tversky (1979). "Prospect Theory: An Analysis of
        Decision under Risk", Econometrica 47(2): 263-291. — loss
        aversion / disposition-effect motivation: sell-side gate
        balances asymmetric pain of forced exit vs. holding cost.


    Blocks `model_sell` exit signals when the latest NGBoost edge-Sharpe
    (μ/σ) is NOT sufficiently negative. Path-dependent rules
    (stop_loss, trailing_stop, single_day_loss, max_hold) are EXEMPT —
    they always fire. Only the streak-based model exit goes through
    Gate B, mirroring the asymmetric rule on the buy side where path
    rules don't see Gate B either.

    User spec 2026-04-26 round-7: "你的 portfolio manager不管卖吗？"
    Pre-fix the sell path had only per-ticker model + path rules. Buy
    path has Gate A/B/C as a quality floor; sell side had no analog —
    so a single-day model spike could exit a holding the panel/μ still
    likes. This task adds the symmetric guard.

    Semantics:
      * Reads `hs.mu`, `hs.sigma` (set by PanelScoringJob in the previous
        bar; persists on HoldingState).
      * If `μ/σ > -threshold`, blocks model_sell. (Buy gate uses
        `μ/σ ≥ +threshold`; sell mirror requires `μ/σ ≤ -threshold` to
        proceed.)
      * Doesn't touch the streak — model can keep accumulating sell
        signals; once edge-Sharpe drops below the floor, the existing
        streak fires immediately.
      * On block, clears tc.exit_signal so PanelConvictionExitTask gets
        a chance to run (panel_conviction has its own μ check inside).

    Pre-conditions:
      * `ranking.panel_scoring.sell_gate_b.enabled = true`
      * NGBoost μ/σ available on the holding (from prior PanelScoringJob)
      * exit_signal is `model_sell` (NOT a path rule)

    Falls through gracefully (no block) when:
      * Flag off (default)
      * No exit_signal, or signal is not should_exit
      * exit_type is not "model_sell"
      * μ or σ unavailable (panel disabled / first bar after entry / warmup)
      * σ <= 0 or NaN (defensive — same as buy-side)
    """

    name = "SellGateBTask"

    def run(self, tc: TickerInferenceContext) -> bool | None:
        cfg = (tc.config.get("ranking", {})
                          .get("panel_scoring", {})
                          .get("sell_gate_b", {}))
        if not bool(cfg.get("enabled", False)):
            return

        sig = getattr(tc, "exit_signal", None)
        if sig is None or not sig.should_exit:
            return

        if sig.exit_type != "model_sell":
            return  # path rules exempt

        hs = tc.holding
        if hs is None:
            return

        mu    = getattr(hs, "mu", None)
        sigma = getattr(hs, "sigma", None)
        if mu is None or sigma is None:
            return  # panel scores unavailable → don't block

        try:
            mu_f    = float(mu)
            sigma_f = float(sigma)
        except (TypeError, ValueError):
            return

        if sigma_f <= 0.0 or mu_f != mu_f or sigma_f != sigma_f:
            return  # defensive — bad μ/σ → don't block

        threshold = float(cfg.get("threshold", 0.10))   # symmetric to buy default
        edge_sharpe = mu_f / sigma_f

        # Block if μ/σ is NOT sufficiently negative.
        # (To proceed with sell, we need edge_sharpe ≤ -threshold.)
        if edge_sharpe > -threshold:
            log.info(
                "SellGateBTask [%s]: BLOCKED model_sell  μ=%+.4f σ=%.4f "
                "edge_sharpe=%+.3f > %+.3f",
                tc.ticker, mu_f, sigma_f, edge_sharpe, -threshold,
            )
            # Clear so PanelConvictionExit can still consider firing.
            # Don't touch streak — once μ/σ drops below floor, the
            # accumulated streak fires immediately on the next bar.
            tc.exit_signal = None
            # Note: visibility comes from the log.info above; no per-task
            # counter needed here (round-7 audit removed dead diagnostic
            # write of `_sell_gate_b_blocked` that nothing read).


class PanelConvictionExitTask(Task):
    """Exit criterion: panel conviction has degraded (panel/NGBoost agreement).

    User spec 2026-04-24: "买卖换加减仓都要是 model+policy" — sell was
    the only surface using only per-ticker tournament model + price rules.
    This task adds a panel-based exit that consults the calibrated
    panel score + NGBoost μ/σ (persisted on HoldingState from the
    previous bar's PanelScoringJob).

    Fires only when the current priority chain did NOT already fire
    (checked via tc.exit_signal). That way stop-loss / trailing / max-hold
    always win first, and this is the tiebreaker for "nothing else said
    exit but the model has turned bearish".

    Trigger conditions (when `risk.panel_exit.enabled=true`):
      * hs.rank_score < panel_sell_floor (default 0.20 — below tier 1
        A-gate threshold, so the calibrated probability now disagrees
        with the original entry conviction)
      * hs.mu <= mu_sell_ceiling (default 0.0 — NGBoost says no edge)

    Audit (2026-04-24): the comparison is against `rank_score`, NOT
    `panel_score`. After PanelScoringJob, `rank_score` is the calibrated
    probability (0..1 range, matching the tier-gate scale that the
    `panel_sell_floor=0.20` default targets). `panel_score` is the raw
    LTR output (~N(0,1)) or μ−λσ (~±0.05 in NGBoost mode); comparing
    those to a probability-scale floor would fire on ~58% of holdings
    (raw mode) or ALL holdings (μ−λσ mode). Requires
    `ranking.panel_scoring.global_calibration.enabled=true` for the
    rank_score field to carry probability-scale values from the panel
    pipeline; tournament-only (panel disabled) holdings already get
    probability-scale rank_score from ScoreModelTask.

    Legacy task default remains ON for backward compatibility when
    `risk.panel_exit.enabled=true`. Production configs that use
    CrossSectionalPanelExitTask should set `risk.panel_exit.legacy_enabled`
    to false so one panel-exit policy owns the decision.
    """

    def run(self, tc: TickerInferenceContext) -> bool | None:
        # Already exiting via higher-priority rule (stop/trailing/max_hold/
        # model-streak) → don't override with panel exit
        if getattr(tc, "exit_signal", None) is not None:
            return

        cfg = tc.config.get("risk", {}).get("panel_exit", {})
        if not bool(cfg.get("enabled", False)):
            return
        if not bool(cfg.get("legacy_enabled", True)):
            return

        hs = tc.holding
        if hs is None:
            return

        # Use rank_score (calibrated probability, 0..1) — NOT panel_score
        # which is raw LTR (~N(0,1)) or μ−λσ.
        prob_score = getattr(hs, "rank_score", None)
        mu         = getattr(hs, "mu", None)

        # Fallback: no panel scores on this holding yet (first bar after
        # purchase, or panel disabled for this run) — don't fire
        if prob_score is None or mu is None:
            return

        panel_floor = float(cfg.get("panel_sell_floor", 0.20))
        mu_ceiling  = float(cfg.get("mu_sell_ceiling", 0.0))
        # V2 (2026-04-24): trigger mode. Default "and" (both conditions)
        # preserves V1 behaviour. "or" fires when EITHER condition is
        # true — useful when panel and μ disagree (e.g. panel still
        # says okay but μ flipped negative, or vice versa).
        trigger_mode = str(cfg.get("trigger_mode", "and")).lower()

        if trigger_mode == "or":
            fires = (prob_score < panel_floor) or (mu <= mu_ceiling)
        else:
            fires = (prob_score < panel_floor) and (mu <= mu_ceiling)

        if fires:
            suppress, why = soft_exit_horizon_suppression(
                panel_cfg=cfg,
                regime=getattr(tc, "regime", None),
                today=getattr(tc, "today", None),
                holding=hs,
            )
            if suppress:
                log.info(
                    "PanelConvictionExitTask [%s]: SUPPRESSED by horizon gate (%s)",
                    tc.ticker, why,
                )
                return

            current_price = resolve_current_price(tc, hs, getattr(tc, "ticker", None))

            # Audit fix 2026-04-29: respect the LT-hold tax gate.
            # compute_exits() suppresses model_sell when a position is in the
            # last N days before the 1-year LT capital-gain threshold AND has
            # an unrealized gain ≥ lt_hold_min_gain. PanelConvictionExit was
            # bypassing this — could trigger a forced ST exit on a position
            # 30 days from LT. Skip the exit if we're in the LT-protected window.
            suppress, why = lt_gate_suppression(
                config=tc.config,
                today=getattr(tc, "today", None),
                holding=hs,
                current_price=current_price,
            )
            if suppress:
                log.info(
                    "PanelConvictionExitTask [%s]: SUPPRESSED by LT tax gate (%s)",
                    tc.ticker, why,
                )
                return  # skip exit — let LT threshold pass

            suppress, why = tax_adjusted_soft_exit_suppression(
                panel_cfg=cfg,
                tax_cfg=tc.config.get("tax") or {},
                today=getattr(tc, "today", None),
                holding=hs,
                current_price=current_price,
                mu=mu,
            )
            if suppress:
                log.info(
                    "PanelConvictionExitTask [%s]: SUPPRESSED by tax-adjusted gate (%s)",
                    tc.ticker, why,
                )
                return

            # Build signal via existing ExitSignal dataclass
            from kernel.exits import ExitSignal  # noqa: PLC0415
            tc.exit_signal = ExitSignal(
                should_exit = True,
                reason      = (f"panel conviction lost rank={prob_score:.3f} "
                                f"μ={mu:+.4f} (floor={panel_floor}, "
                                f"ceiling={mu_ceiling}, mode={trigger_mode})"),
                exit_type   = "panel_conviction",
                exit_params = dict(getattr(tc, "exit_params", {}) or {}),
            )
            log.info("PanelConvictionExitTask [%s]: EXIT rank=%.3f μ=%+.4f (%s)",
                     tc.ticker, prob_score, mu, trigger_mode)


class EarningsBlackoutSellTask(Task):
    """Veto model-driven exits inside the earnings event-blackout window.

    Invariant
    ---------
    Model-driven exits (`model_sell`, `panel_conviction`) are SUPPRESSED when
    the holding sits inside its earnings event-blackout window. Path-action
    exits (`stop_loss`, `trailing_stop`, `single_day_loss`, `max_hold`,
    `kelly_trim`, `rotation`) ALWAYS fire — they are price-action signals not
    affected by event-driven information asymmetries.

    Window is asymmetric:
      * pre_days  (default 2): tighter window before a known print — operator
        is expected to size down voluntarily, not panic-exit on a model spike
        in the run-up.
      * post_days (default 5): wider window after the print to respect
        Post-Earnings Announcement Drift (Bernard & Thomas 1989; Sadka 2006;
        Hirshleifer-Lim-Teoh 2009). Pure-momentum signals systematically
        whipsaw against the drift — give it room to play out.

    References
    ----------
    Bernard, V.L. & Thomas, J.K. (1989). "Post-Earnings-Announcement Drift:
        Delayed Price Response or Risk Premium?" Journal of Accounting
        Research 27(Supp): 1-36.
    Sadka, R. (2006). "Momentum and post-earnings-announcement drift
        anomalies: The role of liquidity risk." Journal of Financial
        Economics 80(2): 309-349.
    Hirshleifer, D., Lim, S.S. & Teoh, S.H. (2009). "Driven to Distraction:
        Extraneous Events and Underreaction to Earnings News." Journal of
        Finance 64(5): 2289-2325.

    Motivating incident
    -------------------
    2026-05-01 trade audit: CAT printed Q1 EPS $5.54 vs $4.62 est (+19.9%
    beat) on 2026-04-30, single-day +9.88%. The per-ticker tournament model
    accumulated `streak=3` model_sell signals through the print, and on
    2026-05-01 the system sold CAT into the post-beat strength — a textbook
    momentum-vs-PEAD whipsaw. This Task is the structural fix.

    Pre-conditions
    --------------
    * `regime.earnings_sell_buffer_pre_days  >= 0` (default 2)
    * `regime.earnings_sell_buffer_post_days >= 0` (default 5)
    * `tc.earnings_calendar` is dict[ticker → list[ISO date strings]]
      (populated by `_make_sell_tctx` from `ctx.earnings_calendar`)

    Falls through gracefully (no veto) when:
      * No exit_signal, or signal is not should_exit
      * exit_type is a path rule (stop/trailing/SDL/max_hold/kelly_trim/rotation)
      * earnings_calendar missing or no dates for this ticker
      * No earnings date within the asymmetric window
      * Both pre and post buffer = 0 (operator-disabled)
    """

    name = "EarningsBlackoutSellTask"

    def run(self, tc: TickerInferenceContext) -> bool | None:
        sig = getattr(tc, "exit_signal", None)
        if sig is None or not sig.should_exit:
            return

        # Use module-level taxonomy (single source of truth — see header).
        # Adding a new exit_type elsewhere requires updating the taxonomy
        # too, otherwise the new type silently bypasses this gate.
        if sig.exit_type not in MODEL_DRIVEN_EXIT_TYPES:
            return  # path-action rule — exempt

        regime_cfg = tc.config.get("regime", {})
        pre_days  = int(regime_cfg.get("earnings_sell_buffer_pre_days",  2))
        post_days = int(regime_cfg.get("earnings_sell_buffer_post_days", 5))
        if pre_days <= 0 and post_days <= 0:
            return  # operator-disabled

        calendar = tc.earnings_calendar or {}
        dates = calendar.get(tc.ticker, []) if calendar else []
        if not dates:
            return  # no calendar entries — can't veto, fail open

        today = tc.today
        if not isinstance(today, datetime.date):
            return  # malformed bar date — defensive

        for d_str in dates:
            try:
                d = datetime.date.fromisoformat(d_str)
            except (TypeError, ValueError):
                continue
            offset = (d - today).days
            # offset > 0  → earnings is N days in the future (pre-window)
            # offset == 0 → today is earnings day
            # offset < 0  → earnings was N days in the past (post-window)
            inside_pre  = (0 <  offset <= pre_days)
            inside_day  = (offset == 0)
            inside_post = (-post_days <= offset < 0)
            if inside_pre or inside_day or inside_post:
                log.info(
                    "EarningsBlackoutSellTask [%s]: VETO %s exit  "
                    "earnings=%s  offset=%+d  window=(-%d, +%d)",
                    tc.ticker, sig.exit_type, d_str, offset,
                    post_days, pre_days,
                )
                # Clear so downstream portfolio-level tasks (LimitSellsPerBar)
                # see no exit. Streak preserved — once the window closes the
                # accumulated streak fires immediately on the next bar.
                tc.exit_signal = None
                return
