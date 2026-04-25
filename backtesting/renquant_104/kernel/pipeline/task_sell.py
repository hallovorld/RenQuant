"""Per-ticker sell evaluation tasks."""
from __future__ import annotations

import logging

from .context import TickerInferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.sell")


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
            tc.exit_signal = sig
        elif tc.model_action == "sell" and updated_hs.sell_streak > 0:
            # Use the typed field on ExitSignal (was a dynamic attribute write
            # before audit #17 — easier for static analysis + tests now).
            sig.blocked_streak = True
            tc.exit_signal = sig

        log.debug("EvaluateExitsTask [%s]: should_exit=%s  type=%s",
                  tc.ticker, sig.should_exit, getattr(sig, "exit_type", None))


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

    Flag default OFF — user can A/B before flipping.
    """

    def run(self, tc: TickerInferenceContext) -> bool | None:
        # Already exiting via higher-priority rule (stop/trailing/max_hold/
        # model-streak) → don't override with panel exit
        if getattr(tc, "exit_signal", None) is not None:
            return

        cfg = tc.config.get("risk", {}).get("panel_exit", {})
        if not bool(cfg.get("enabled", False)):
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
            # Build signal via existing ExitSignal dataclass
            from kernel.exits import ExitSignal  # noqa: PLC0415
            tc.exit_signal = ExitSignal(
                should_exit = True,
                reason      = (f"panel conviction lost rank={prob_score:.3f} "
                                f"μ={mu:+.4f} (floor={panel_floor}, "
                                f"ceiling={mu_ceiling}, mode={trigger_mode})"),
                exit_type   = "panel_conviction",
            )
            log.info("PanelConvictionExitTask [%s]: EXIT rank=%.3f μ=%+.4f (%s)",
                     tc.ticker, prob_score, mu, trigger_mode)
