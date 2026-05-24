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
    """Hard BEAR override + vol-cluster CHOPPY detection.

    Fires `state.hard_bear` if any of:
      • 20-day vol > bear_vol_threshold        (default 0.35 = GFC-level)
      • 20-day cum-ret < bear_return_threshold (default -0.08)
      • 5-day vol > bear_vol_threshold_5d      (default 0.25)  ← 2026-05-17 fix A
      • 5-day cum-ret < bear_return_threshold_5d (default -0.04)  ← 2026-05-17 fix A

    Also computes `state.vol_cluster_choppy` (2026-05-17 fix C):
      vol_5d > vol_60d × choppy_vol_ratio_threshold (1.5)
        AND |cum_ret_20d| < choppy_drift_threshold (0.02)

    Rationale: 2026-05-17 dense panel + detector audit. The 20-day vol
    threshold (35%) is GFC-calibrated; SVB / DeepSeek+tariff / Aug-2024
    crises never crossed it (max 19% / 18% / 22% respectively). The
    Hurst-REVERSION CHOPPY route is dead because SPY rarely has Hurst <
    0.52 on a 63-bar window. The vol-cluster gate provides an
    orthogonal CHOPPY signal that doesn't rely on Hurst at all.

    References for short-horizon distress detection:
      - Andersen-Bollerslev-Diebold-Labys (ABDL) 1999 "Realized Volatility
        and Correlation" (NYU wp 99061) — establishes 5-day realized vol
        as canonical short-horizon measurement window for regime detection.
      - Industry practice (VolatilityBox.com, dozendiamonds.com): "If
        realized vol > X for Y days, switch regime" — `first 3-5 days of
        a regime transition often contain the largest price moves`.
      - Bollerslev 1986 (Econometrica) GARCH(1,1) — theoretical
        foundation for vol clustering / vol-spike detection.
      - Engle 2002 (J. Bus. Econ. Stat.) "Dynamic Conditional Correlation"
        — vol-regime threshold methodology; typical vol-spike trigger is
        1.5-2.0 × baseline (motivates 1.5× choppy_vol_ratio).

    Hysteresis (in RegimeFinalizeTask): persists σ-wire activation for
    N=10 bars after last trigger.
      - Hamilton 1989 (Econometrica) "A new approach to nonstationary
        time series" — Markov-switching expected duration; in his GNP
        2-state model, high-state persistence prob ≈ 0.90 → expected
        duration ≈ 10 quarters. We use 10 trading days = ~2 weeks as
        conservative analogue for vol-regime persistence.

    Threshold-specific numbers (5d_vol=0.25, 5d_ret=-0.04, ratio=1.5,
    drift=0.02): empirically derived from the 5-window audit at
    `scripts/audit_regime_detector.py` + 5/17 dense panel. Per
    CLAUDE.md §5.12: these are exploratory; will tune via A/B once
    the per-regime σ-wire infrastructure is exercised on real bars.

    State writes: state.hard_bear, state.vol_5d, state.ret_5d,
    state.vol_cluster_choppy.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        cfg = ctx.config.get("regime", {})
        vol_window      = int(cfg.get("vol_realized_window", 20))
        bear_vol_thr    = float(cfg.get("bear_vol_threshold",    0.35))
        bear_ret_thr    = float(cfg.get("bear_return_threshold", -0.08))
        vol_window_5d   = int(cfg.get("vol_realized_window_5d",  5))
        bear_vol_thr_5d = float(cfg.get("bear_vol_threshold_5d",    0.25))
        bear_ret_thr_5d = float(cfg.get("bear_return_threshold_5d", -0.04))
        choppy_baseline = int(cfg.get("choppy_vol_baseline_window", 60))
        choppy_vol_rat  = float(cfg.get("choppy_vol_ratio_threshold", 1.5))
        choppy_drift_th = float(cfg.get("choppy_drift_threshold",     0.02))

        spy_returns = np.array(ctx.spy_returns)
        state = ctx.regime_state

        # ── NaN/inf guard (Audit RG-1/RG-2) — fail-SAFE to BEAR ──
        if len(spy_returns) >= vol_window and (
            np.isnan(spy_returns[-max(vol_window, choppy_baseline):]).any()
            or np.isinf(spy_returns[-max(vol_window, choppy_baseline):]).any()
        ):
            state.hard_bear = True
            state.vol_cluster_choppy = False
            log.warning(
                "BEAROverrideTask: SPY returns contain NaN/inf — "
                "fail-SAFE forcing hard_bear=True (block buys)",
            )
            return

        def _vol_ret(arr: np.ndarray) -> tuple[float, float]:
            """Annualized vol + strict cumulative product return."""
            if len(arr) < 2:
                return 0.0, 0.0
            v = float(np.std(arr, ddof=1) * math.sqrt(252))
            r = float(np.prod(1.0 + arr) - 1.0)
            return v, r

        # ── 20-day check (existing) ──
        hard_bear_20d = False
        if len(spy_returns) >= vol_window:
            spy_20d_vol, spy_20d_ret = _vol_ret(spy_returns[-vol_window:])
            hard_bear_20d = spy_20d_vol > bear_vol_thr or spy_20d_ret < bear_ret_thr
        else:
            spy_20d_vol = spy_20d_ret = 0.0

        # ── 5-day check (2026-05-17 fix A) ──
        hard_bear_5d = False
        if len(spy_returns) >= vol_window_5d:
            vol_5d, ret_5d = _vol_ret(spy_returns[-vol_window_5d:])
            hard_bear_5d = vol_5d > bear_vol_thr_5d or ret_5d < bear_ret_thr_5d
        else:
            vol_5d = ret_5d = 0.0
        state.vol_5d = vol_5d
        state.ret_5d = ret_5d

        state.hard_bear = hard_bear_20d or hard_bear_5d

        # ── vol-cluster CHOPPY (2026-05-17 fix C) ──
        # Elevated 5-day vol relative to 60-day baseline, AND market not trending.
        # Requires enough history for both windows.
        vol_cluster = False
        if len(spy_returns) >= max(vol_window_5d, choppy_baseline):
            vol_60d, _ = _vol_ret(spy_returns[-choppy_baseline:])
            if math.isfinite(vol_5d) and math.isfinite(vol_60d) and vol_60d > 0:
                vol_elevated = vol_5d > vol_60d * choppy_vol_rat
                no_trend     = abs(spy_20d_ret) < choppy_drift_th
                vol_cluster  = vol_elevated and no_trend
        state.vol_cluster_choppy = vol_cluster

        if state.hard_bear:
            which = []
            if hard_bear_20d: which.append(f"20d_vol={spy_20d_vol:.2f},ret={spy_20d_ret:+.2%}")
            if hard_bear_5d:  which.append(f"5d_vol={vol_5d:.2f},ret={ret_5d:+.2%}")
            log.info("BEAROverrideTask: hard BEAR triggered (%s)", "; ".join(which))
        elif state.vol_cluster_choppy:
            log.info("BEAROverrideTask: vol-cluster CHOPPY (vol5d=%.2f vol60d=%.2f drift20d=%+.2f%%)",
                     vol_5d, vol_60d if 'vol_60d' in locals() else 0.0, spy_20d_ret*100)


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
        spy_close = spy_ma50 = spy_ma200 = None
        spy_df = (ctx.ohlcv or {}).get("SPY") if hasattr(ctx, "ohlcv") else None
        if spy_df is not None and len(spy_df) >= 200:
            try:
                import math as _math
                _spy_close = float(spy_df["close"].iloc[-1])
                _spy_ma50 = float(spy_df["close"].rolling(50).mean().iloc[-1])
                _spy_ma200 = float(spy_df["close"].rolling(200).mean().iloc[-1])
                if _math.isfinite(_spy_close) and _math.isfinite(_spy_ma50) \
                   and _math.isfinite(_spy_ma200):
                    spy_close = _spy_close
                    spy_ma50 = _spy_ma50
                    spy_ma200 = _spy_ma200
                    spy_below_ma50 = spy_close < spy_ma50
                    spy_below_ma200 = spy_close < spy_ma200
            except Exception:
                pass
        spy_bearish_trend = spy_below_ma50 and spy_below_ma200

        # 2026-05-17 fix C: vol-cluster CHOPPY signal (set by BEAROverrideTask)
        # is an orthogonal CHOPPY trigger that doesn't rely on Hurst<0.52
        # (which essentially never fires on SPY's 63-bar window). BEAR routes
        # still take precedence — vol_cluster only fires if hard_bear is False.
        vol_cluster_choppy = bool(getattr(state, "vol_cluster_choppy", False))

        if state.hard_bear or gmm_probs.get(BEAR, 0) > 0.5:
            new_regime = BEAR
            decision_source = "hard_bear" if state.hard_bear else "gmm_bear"
        elif state.hurst_regime == "MOMENTUM":
            # Direction-aware (both MA50 AND MA200 must be below):
            #   trending up OR mixed (MA50/MA200 disagree)  → BULL_CALM
            #   trending down (both MAs below)              → BEAR
            if spy_bearish_trend:
                new_regime = BEAR
                decision_source = "hurst_momentum_spy_bearish"
            elif vol_cluster_choppy:
                new_regime = "CHOPPY"
                decision_source = "hurst_momentum_vol_cluster_choppy"
            else:
                new_regime = "BULL_CALM"
                decision_source = "hurst_momentum_bull"
        elif state.hurst_regime == "REVERSION" or vol_cluster_choppy:
            new_regime = "CHOPPY"
            decision_source = (
                "hurst_reversion"
                if state.hurst_regime == "REVERSION"
                else "vol_cluster_choppy"
            )
        else:
            new_regime = dominant_gmm if dominant_gmm != BEAR else BULL_VOLATILE
            decision_source = "dominant_gmm"

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
        ctx._regime_evidence = {  # noqa: SLF001
            "source": decision_source,
            "final_regime": new_regime,
            "prev_regime": prev_regime,
            "confidence": confidence,
            "hurst": state.hurst,
            "hurst_regime": state.hurst_regime,
            "gmm_probs": dict(gmm_probs or {}),
            "dominant_gmm": dominant_gmm,
            "hard_bear": bool(state.hard_bear),
            "vol_5d": getattr(state, "vol_5d", None),
            "ret_5d": getattr(state, "ret_5d", None),
            "vol_cluster_choppy": vol_cluster_choppy,
            "in_transition": bool(state.in_transition),
            "countdown": getattr(state, "countdown", None),
            "cooldown_start": getattr(state, "cooldown_start", None),
            "spy_close": spy_close,
            "spy_ma50": spy_ma50,
            "spy_ma200": spy_ma200,
            "spy_below_ma50": spy_below_ma50,
            "spy_below_ma200": spy_below_ma200,
            "spy_bearish_trend": spy_bearish_trend,
        }

        # 2026-05-17 σ-wire hysteresis update.
        # If the newly-resolved regime has a per-regime ngboost overlay
        # that activates σ wire, memo the overlay and arm the hysteresis
        # counter. Decrement (without re-arming) on bars that don't
        # activate. _ngb_cfg in job_panel_scoring.py reads these state
        # fields to keep σ-wire sticky across bar-to-bar regime flicker.
        # Without this, the 5-day BEAR detector catching SVB/Aug-2024/
        # DeepSeek 1-5 BEAR bars caused σ-wire ON↔OFF churn → 5/17 A/B
        # per-regime version lost -4.7pp pooled (W7 -21pp single
        # window) where uniform σ-on won +3pp pooled.
        #
        # Default N=10 bars rationale: Hamilton 1989 (Econometrica)
        # Markov-switching expected duration ≈ 1/(1−p) where p is the
        # state persistence probability. His 2-state GNP model fit p≈0.90
        # → duration ≈ 10 quarters. For DAILY equity vol regimes we use
        # 10 bars (~2 trading weeks) as a conservative analogue. Exact
        # value is exploratory (CLAUDE.md §5.12); to tune properly we'd
        # need to fit a Hamilton-style 2-state model on SPY vol history.
        _hysteresis_bars = int(ctx.config.get("regime", {})
                                          .get("sigma_wire_hysteresis_bars", 10))
        _NGB_REGIME_KEYS = ("enabled", "score_mode", "lambda_sigma")
        regime_p = (ctx.config.get("regime_params", {}) or {}).get(new_regime, {}) or {}
        regime_ngb = (regime_p.get("ngboost") or {}) if isinstance(regime_p, dict) else {}
        live_overlay = {k: regime_ngb[k] for k in _NGB_REGIME_KEYS if k in regime_ngb}
        if live_overlay.get("enabled") is True:
            state.sigma_wire_overlay_memo = dict(live_overlay)
            state.sigma_wire_hysteresis_remaining = _hysteresis_bars
        elif getattr(state, "sigma_wire_hysteresis_remaining", 0) > 0:
            state.sigma_wire_hysteresis_remaining -= 1
        # else: counter at 0, memo untouched (don't clear so introspection
        # can see what was last memorized, but counter==0 means it's not
        # actively applied).

        log.info("RegimeFinalizeTask: regime=%s  conf=%.2f  transition=%s",
                 new_regime, confidence, state.in_transition)
