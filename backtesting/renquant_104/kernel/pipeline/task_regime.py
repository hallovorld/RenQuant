"""Regime detection tasks: Hurst → CUSUM → GMM → BEAR override → finalize."""
from __future__ import annotations

import datetime
import logging
import math

import numpy as np

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.regime")


class HurstTask(Task):
    """Layer 1: compute Hurst exponent → state.hurst, state.hurst_regime."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.regime import compute_hurst  # noqa: PLC0415

        cfg = ctx.config.get("regime", {})
        hurst_window = int(cfg.get("hurst_window", 63))
        hurst_trend  = float(cfg.get("hurst_trending_threshold",  0.65))
        hurst_rev    = float(cfg.get("hurst_reversion_threshold", 0.52))

        spy_returns = np.array(ctx.spy_returns)
        if len(spy_returns) < 30:
            return None

        state = ctx.regime_state
        state.hurst = compute_hurst(spy_returns, window=hurst_window)

        if state.hurst > hurst_trend:
            state.hurst_regime = "MOMENTUM"
        elif state.hurst < hurst_rev:
            state.hurst_regime = "REVERSION"
        else:
            state.hurst_regime = "AMBIGUOUS"

        log.debug("HurstTask: H=%.3f  regime=%s", state.hurst, state.hurst_regime)


class CUSUMTask(Task):
    """Layer 2: CUSUM changepoint detection → `state.cusum_triggered` (flag).

    Plan B (2026-04-23): this task NO LONGER sets `state.countdown`
    directly. The cooldown is only armed when `RegimeFinalizeTask`
    determines the *resolved* regime has actually switched
    (`prev_regime != new_regime`). CUSUM firing inside a stable
    regime (e.g. SPY 20d window rolling over during a bull recovery)
    no longer perpetually blocks buys. The raw trigger is kept on
    `state.cusum_triggered` for downstream diagnostics.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.regime import compute_cusum  # noqa: PLC0415

        cfg = ctx.config.get("regime", {})
        cusum_lookback = int(cfg.get("cusum_lookback", 20))
        cusum_thresh   = float(cfg.get("cusum_threshold", 5.5))
        cusum_drift    = float(cfg.get("cusum_drift", 0.5))

        spy_returns = np.array(ctx.spy_returns)
        state = ctx.regime_state

        triggered = compute_cusum(spy_returns, cusum_lookback, cusum_thresh, cusum_drift)
        # Stash the raw signal; the countdown arm/decrement happens in
        # RegimeFinalizeTask once prev_regime / new_regime are known.
        state.cusum_triggered = bool(triggered)

        log.debug("CUSUMTask: triggered=%s (cooldown arming deferred to finalize)",
                  triggered)


class GMMTask(Task):
    """Layer 3: GMM posterior probabilities → state.gmm_probs."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.regime import gmm_predict  # noqa: PLC0415

        cfg = ctx.config.get("regime", {})
        vol_window = int(cfg.get("vol_realized_window", 20))

        spy_df      = ctx.ohlcv.get("SPY")
        spy_returns = np.array(ctx.spy_returns)

        # 2026-05-04 audit Issue 01 fix: null-guard for ctx.gmm. Pre-fix,
        # if GMM artifact loading failed upstream, gmm_predict(None, …)
        # raised inside the call and crashed the daily104 cron hard.
        # Fail-SAFE: empty probs dict → RegimeFinalizeTask defaults to
        # AMBIGUOUS regime via the existing empty-dict path (Issue 02
        # documents that default; the ambiguity is now intentional rather
        # than silent). Also guard SPY df.
        if ctx.gmm is None:
            log.warning(
                "GMMTask: ctx.gmm is None — GMM artifact missing/failed "
                "to load. Setting empty gmm_probs (RegimeFinalizeTask "
                "will fall back to default regime).",
            )
            ctx.regime_state.gmm_probs = {}
            return
        if spy_df is None or spy_df.empty:
            log.warning(
                "GMMTask: SPY OHLCV missing/empty — skipping GMM predict.",
            )
            ctx.regime_state.gmm_probs = {}
            return

        # 2026-05-14 P0/HMM upgrade (Hamilton 1989 Markov-switching):
        # Route to hmm_predict when the loaded artifact carries
        # `model_type=GaussianHMM`. Per-bar GMM is retained as
        # fallback for legacy artifacts. This is the ONLY decision
        # site — kernel/regime_hmm.py owns the forward algorithm.
        from kernel.regime_hmm import is_hmm_artifact, hmm_predict  # noqa: PLC0415
        if is_hmm_artifact(ctx.gmm):
            ctx.regime_state.gmm_probs = hmm_predict(
                ctx.gmm, spy_returns, spy_df, vol_window=vol_window,
            )
        else:
            ctx.regime_state.gmm_probs = gmm_predict(
                ctx.gmm, spy_returns, spy_df, vol_window=vol_window,
            )
        if not ctx.regime_state.gmm_probs:
            log.warning("GMMTask: prediction returned empty probs.")
            return
        dominant = max(ctx.regime_state.gmm_probs, key=ctx.regime_state.gmm_probs.get)
        log.debug("GMMTask: probs=%s  dominant=%s", ctx.regime_state.gmm_probs, dominant)


class BEAROverrideTask(Task):
    """Hard BEAR override: fire if realized vol or cumulative return cross thresholds."""

    def run(self, ctx: InferenceContext) -> bool | None:
        cfg = ctx.config.get("regime", {})
        vol_window   = int(cfg.get("vol_realized_window", 20))
        bear_vol_thr = float(cfg.get("bear_vol_threshold",    0.35))
        bear_ret_thr = float(cfg.get("bear_return_threshold", -0.08))

        spy_returns = np.array(ctx.spy_returns)
        state = ctx.regime_state

        if len(spy_returns) >= vol_window:
            window = spy_returns[-vol_window:]
            # Audit fix RG-1/RG-2 (Round 2 deep audit, 2026-04-25):
            # pre-fix, NaN in `window` propagated through np.std/np.prod
            # to NaN. Then `vol > bear_vol_thr` and `ret < bear_ret_thr`
            # both evaluated False → state.hard_bear = False → BEAR
            # override silently disabled on bad SPY data. This is a
            # safety gate; failing silent is unacceptable. Now: on
            # non-finite vol/ret, fail-SAFE to True (assume BEAR).
            if np.isnan(window).any() or np.isinf(window).any():
                state.hard_bear = True
                log.warning(
                    "BEAROverrideTask: SPY returns contain NaN/inf — "
                    "fail-SAFE forcing hard_bear=True (block buys)",
                )
            else:
                spy_20d_vol = float(np.std(window, ddof=1) * math.sqrt(252))
                # Audit #11: prior implementation used np.sum() which approximates
                # cumulative return only for tiny daily moves. For a real selloff
                # (e.g. -3% × 5 days), arithmetic sum overestimates the cumulative
                # drop vs prod(1+r)-1. Use the strict cumulative product so the
                # threshold matches what "cumulative return < -8%" actually means.
                spy_20d_ret = float(np.prod(1.0 + window) - 1.0)
                state.hard_bear = spy_20d_vol > bear_vol_thr or spy_20d_ret < bear_ret_thr
        else:
            state.hard_bear = False

        if state.hard_bear:
            log.info("BEAROverrideTask: hard BEAR override triggered")


class RegimeFinalizeTask(Task):
    """Resolve final regime from all layer outputs → ctx.regime, ctx.confidence.

    Plan B owns the cooldown here. After new_regime is resolved:
      - If `new_regime != prev_regime` AND `countdown == 0`, ARM the
        cooldown to `transition_uncertainty_bars`.
      - Compute `in_transition = countdown > 0`.
      - Decrement `countdown` (so the last bar of the cooldown window
        still signals `in_transition=True`).

    CUSUM fires (state.cusum_triggered) no longer re-arm the cooldown
    inside a stable regime — previously that produced the 2026-04-22
    → 04-23 3-day zero-trade streak.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.regime import compute_regime_confidence  # noqa: PLC0415
        from kernel.config import BEAR, BULL_VOLATILE        # noqa: PLC0415

        state = ctx.regime_state
        gmm_probs    = state.gmm_probs
        dominant_gmm = max(gmm_probs, key=gmm_probs.get) if gmm_probs else "BULL_CALM"

        prev_regime = state.regime   # snapshot BEFORE mutating

        # 2026-05-14 Direction-aware Hurst with MA200 confirmation:
        # Hurst > 0.65 tells us the market is TRENDING; pair with TWO SPY
        # direction signals (MA50 + MA200) to distinguish bull rally from
        # bear decline.
        #
        # Why BOTH MAs are required (2026-05-14 audit):
        # Initial fix (commit 3925c0d) used MA50 alone. Empirically, in
        # bull markets (Q11 BULL_STRONG) SPY < MA50 happens on 8% of days
        # (normal corrections). Those days mis-labeled BEAR → transition
        # cooldown fired (3 bars at conf=0.5) → confidence_to_size_multiplier
        # halved positions for 8 days → Q11 Panel A −27pt vs original baseline.
        # MA200 is much stickier — 0/64 bars below MA200 in Q11 BULL_STRONG.
        # Real bears (Q01 2022-Q2) have 94% bars below MA200 too — gate
        # preserves true-BEAR detection while eliminating bull noise.
        spy_below_ma50 = False
        spy_below_ma200 = False
        spy_df = (ctx.ohlcv or {}).get("SPY") if hasattr(ctx, "ohlcv") else None
        if spy_df is not None and len(spy_df) >= 200:
            try:
                import math as _math
                spy_close = float(spy_df["close"].iloc[-1])
                spy_ma50 = float(spy_df["close"].rolling(50).mean().iloc[-1])
                spy_ma200 = float(spy_df["close"].rolling(200).mean().iloc[-1])
                if _math.isfinite(spy_close) and _math.isfinite(spy_ma50) \
                   and _math.isfinite(spy_ma200):
                    spy_below_ma50 = spy_close < spy_ma50
                    spy_below_ma200 = spy_close < spy_ma200
            except Exception:
                pass
        spy_bearish_trend = spy_below_ma50 and spy_below_ma200

        if state.hard_bear or gmm_probs.get(BEAR, 0) > 0.5:
            new_regime = BEAR
        elif state.hurst_regime == "MOMENTUM":
            # Direction-aware (both MA50 AND MA200 must be below):
            #   trending up OR mixed (MA50/MA200 disagree)  → BULL_CALM
            #   trending down (both MAs below)              → BEAR
            new_regime = BEAR if spy_bearish_trend else "BULL_CALM"
        elif state.hurst_regime == "REVERSION":
            new_regime = "CHOPPY"
        else:
            new_regime = dominant_gmm if dominant_gmm != BEAR else BULL_VOLATILE

        # Plan B: cooldown only on actual regime switch.
        # CUSUM-v2 Design C (user-locked 2026-04-24): also stamp wall-clock
        # `cooldown_start` so intraday runners can read elapsed time instead
        # of relying on bar-count alone. Both fields persist in live_state.
        trans_bars = int(ctx.config.get("regime", {})
                         .get("transition_uncertainty_bars", 3))
        if new_regime != prev_regime and state.countdown == 0:
            state.countdown = trans_bars
            # Record the wall-clock start. Use today's calendar date (sim)
            # or datetime.now() (live); both are convertible by
            # cusum_cooldown_progress(). InferenceContext.today is a date
            # in the sim path and a datetime in live.
            now = getattr(ctx, "today", None)
            if isinstance(now, datetime.date) and not isinstance(now, datetime.datetime):
                state.cooldown_start = datetime.datetime(
                    now.year, now.month, now.day,
                )
            elif isinstance(now, datetime.datetime):
                state.cooldown_start = now
            else:
                state.cooldown_start = datetime.datetime.utcnow()
        state.in_transition = state.countdown > 0
        if state.countdown > 0:
            state.countdown -= 1
        # Clear cooldown_start once the bar-count window fully elapses (so
        # wall-clock progress reads 1.0 after recovery even if nobody
        # retrains the regime). Guard: only clear when we're past the
        # full cooldown window.
        if state.countdown == 0 and state.cooldown_start is not None:
            cd_days = float(ctx.config.get("regime", {})
                            .get("cusum_cooldown_days", 3.0))
            now = getattr(ctx, "today", None)
            if now is not None and cd_days > 0:
                from kernel.regime import cusum_cooldown_progress  # noqa: PLC0415
                if cusum_cooldown_progress(now, state.cooldown_start, cd_days) >= 1.0:
                    state.cooldown_start = None

        confidence = compute_regime_confidence(
            new_regime, state.hurst, gmm_probs, state.in_transition, ctx.config,
            hurst_regime=state.hurst_regime,
            hard_bear=state.hard_bear,
        )

        state.regime     = new_regime
        state.confidence = confidence
        ctx.regime       = new_regime
        ctx.confidence   = confidence
        ctx.regime_counts[new_regime] = ctx.regime_counts.get(new_regime, 0) + 1

        log.info("RegimeFinalizeTask: regime=%s  conf=%.2f  transition=%s",
                 new_regime, confidence, state.in_transition)
