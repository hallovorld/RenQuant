"""Buy-side quality gates (2026-05-15 Upgrades A + B).

Two complementary filters that catch META-style "buy a beaten mega-cap
in an otherwise momentum-friendly regime" trades:

  A. RegimeMomentumAlignmentTask — when the SPY regime is momentum-
     dominated (BULL_CALM/BULL_VOLATILE with hurst > 0.65), shrink the
     score of any buy candidate whose own 60-day return is negative.
     Catches strategy-vs-individual-momentum contradictions.
     Inspired by Asness-Moskowitz-Pedersen 2013 momentum-everywhere:
     momentum signals are strongest at the BOTH cross-sectional AND
     time-series levels; betting against a stock's own momentum in a
     momentum regime is theoretically backwards.

  B. DeepDrawdownVetoTask — veto buy candidates trading > 20% below
     their 52-week high UNLESS a positive fundamental confirmation
     (sue_signal > 0 OR pead_signal > 0) is also present. Catches
     "falling knife" mega-caps. Inspired by Hong-Stein 2003 underreaction
     and Daniel-Hirshleifer-Subrahmanyam 1998 overconfidence: stocks
     20%+ off highs without earnings-driven revisions tend to keep
     falling.

Both DEFAULT OFF. Opt-in via:
  ranking.buy_quality_gates.regime_momentum.enabled
  ranking.buy_quality_gates.deep_drawdown_veto.enabled

Wired in pp_inference between PanelScoringJob and RankingJob — so the
upstream panel score has settled but RankingJob hasn't picked top-K
yet. Tasks act on `ctx.candidates` in place.

Today's META trade (2026-05-15) would have been:
  * A: META r60d = -4.66% in MOM regime → score × 0.5
  * B: META dd_from_52w_high = -22.1%, no SUE confirmation → VETO

This file ships the architecture only — promotion to the golden config
is a separate operator decision after A/B sim verification.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from kernel.pipeline.pipeline import Task

log = logging.getLogger("kernel.pipeline.buy_quality_gates")

# ── Regime-vs-individual-momentum (Upgrade A) ───────────────────────────────


class RegimeMomentumAlignmentTask(Task):
    """Shrink candidate scores when regime momentum and individual
    momentum disagree.

    Config (read from `ranking.buy_quality_gates.regime_momentum`):
      enabled: bool                 (default False)
      momentum_regimes: list[str]   (default ["BULL_CALM", "BULL_VOLATILE"])
      hurst_floor: float            (default 0.65 — Hurst exponent threshold
                                    above which the regime is "trending")
      r60d_floor: float             (default 0.0 — individual 60d return must
                                    exceed this to be momentum-aligned)
      mismatch_scale: float         (default 0.5 — multiply rank_score by
                                    this factor when mismatched)
      attr: str                     (default "rank_score" — which score
                                    attribute to shrink; "panel_score" also valid)
    """
    name = "RegimeMomentumAlignmentTask"

    def run(self, ctx) -> bool | None:
        cfg = (ctx.config or {}).get("ranking", {}) \
                                 .get("buy_quality_gates", {}) \
                                 .get("regime_momentum", {})
        if not cfg.get("enabled", False):
            return
        momentum_regimes = set(cfg.get("momentum_regimes",
                                         ["BULL_CALM", "BULL_VOLATILE"]))
        # 2026-05-15 (post p0activated 16-window): empirical regime-conditional
        # opt-out. The p0activated panel showed gates HELP in BEAR/CHOPPY/VOL
        # (+8 to +30pp) but HURT in BULL_CALM/BULL_STRONG rallies (-10 to
        # -25pp) where mean-revert mega-caps with negative r60 are exactly
        # the bounce winners we want to keep. `disabled_in_regimes` is the
        # operator's regime-conditional skip list. CLAUDE.md PRIME DIRECTIVE.
        disabled_in_regimes = set(cfg.get("disabled_in_regimes", []))
        hurst_floor    = float(cfg.get("hurst_floor",    0.65))
        r60d_floor     = float(cfg.get("r60d_floor",     0.0))
        mismatch_scale = float(cfg.get("mismatch_scale", 0.5))
        attr = str(cfg.get("attr", "rank_score"))

        regime = getattr(ctx, "regime", None)
        hurst = getattr(ctx, "hurst", None) or getattr(ctx, "_hurst", None)
        # No momentum regime → no-op
        if regime not in momentum_regimes:
            return
        # Empirical opt-out (per p0activated 16-window evidence)
        if regime in disabled_in_regimes:
            log.info("RegimeMomentumAlignment: regime=%s in disabled_in_regimes "
                     "→ skip (operator opt-out per regime-conditional empirics)",
                     regime)
            return
        if hurst is None or not math.isfinite(hurst) or hurst < hurst_floor:
            return

        candidates = list(getattr(ctx, "candidates", []) or [])
        ohlcv = getattr(ctx, "ohlcv", None) or {}
        if not candidates:
            return

        shrunk = []
        for cand in candidates:
            r60 = _trailing_return(ohlcv.get(cand.ticker), days=60)
            if r60 is None:
                continue
            if r60 >= r60d_floor:
                continue
            # Mismatch — shrink the chosen score attribute
            old = getattr(cand, attr, None)
            if old is None or not math.isfinite(old):
                continue
            new = float(old) * mismatch_scale
            setattr(cand, attr, new)
            shrunk.append((cand.ticker, r60, old, new))

        if shrunk:
            log.info(
                "RegimeMomentumAlignment: regime=%s hurst=%.2f → shrunk "
                "%d/%d candidate %s by ×%.2f (mismatched 60d returns)",
                regime, hurst, len(shrunk), len(candidates), attr, mismatch_scale,
            )
            for t, r60, old, new in shrunk[:5]:
                log.info("  %s r60=%+.2f%% %s %.4f → %.4f",
                          t, r60 * 100, attr, old, new)
            if len(shrunk) > 5:
                log.info("  (+%d more shrunk)", len(shrunk) - 5)
            ctx.counters = getattr(ctx, "counters", None) or {}
            ctx.counters["regime_momentum_shrunk"] = (
                ctx.counters.get("regime_momentum_shrunk", 0) + len(shrunk)
            )


# ── Deep-drawdown veto (Upgrade B) ──────────────────────────────────────────


class DeepDrawdownVetoTask(Task):
    """Veto buy candidates > X% below their 52-week high unless a
    positive fundamental signal confirms the entry.

    Config (read from `ranking.buy_quality_gates.deep_drawdown_veto`):
      enabled: bool                  (default False)
      dd_threshold: float            (default 0.20 — fraction below 52w high)
      window_days: int               (default 252)
      sue_floor: float               (default 0.0 — SUE > this is "positive")
      pead_floor: float              (default 0.0 — PEAD > this is "positive")
      require_either: bool           (default True — need SUE OR PEAD > floor
                                     to bypass; if False, both required)

    A candidate is vetoed iff:
      dd_from_52w_high < -dd_threshold AND
      neither (sue_signal > sue_floor) NOR (pead_signal > pead_floor)
    """
    name = "DeepDrawdownVetoTask"

    def run(self, ctx) -> bool | None:
        cfg = (ctx.config or {}).get("ranking", {}) \
                                 .get("buy_quality_gates", {}) \
                                 .get("deep_drawdown_veto", {})
        if not cfg.get("enabled", False):
            return
        dd_threshold   = float(cfg.get("dd_threshold",   0.20))
        window_days    = int  (cfg.get("window_days",    252))
        sue_floor      = float(cfg.get("sue_floor",      0.0))
        pead_floor     = float(cfg.get("pead_floor",     0.0))
        require_either = bool (cfg.get("require_either", True))
        # 2026-05-15 regime-conditional opt-out (see RegimeMomentumAlignmentTask
        # docstring for evidence). In BULL_CALM/BULL_STRONG rallies the deep-dd
        # cohort IS the bounce winners (META, etc.); vetoing them costs +25pp.
        disabled_in_regimes = set(cfg.get("disabled_in_regimes", []))
        regime = getattr(ctx, "regime", None)
        if regime in disabled_in_regimes:
            log.info("DeepDrawdownVeto: regime=%s in disabled_in_regimes → skip",
                     regime)
            return

        candidates = list(getattr(ctx, "candidates", []) or [])
        if not candidates:
            return
        ohlcv = getattr(ctx, "ohlcv", None) or {}

        kept, vetoed = [], []
        for cand in candidates:
            dd = _dd_from_high(ohlcv.get(cand.ticker), window_days)
            if dd is None or dd > -dd_threshold:
                kept.append(cand)
                continue
            # Deep drawdown — check fundamental confirmation.
            # FIX 2026-05-20 audit P0-13: attr names are sue_signal /
            # pead_signal in ApplyScoresTask (job_panel_scoring.py:441,372),
            # NOT sue_score / pead_score. Pre-fix _feature() returned None
            # 100% of time → confirmed=False → all deep-DD candidates vetoed
            # unconditionally. Masked while DDV disabled globally 2026-05-17,
            # would silent-misfire on regime-conditional re-enable.
            sue  = _feature(cand, "sue_signal")
            pead = _feature(cand, "pead_signal")
            sue_ok  = sue  is not None and math.isfinite(sue)  and sue  > sue_floor
            pead_ok = pead is not None and math.isfinite(pead) and pead > pead_floor
            confirmed = (sue_ok or pead_ok) if require_either else (sue_ok and pead_ok)
            if confirmed:
                kept.append(cand)
            else:
                vetoed.append((cand.ticker, dd, sue, pead))

        if vetoed:
            log.info(
                "DeepDrawdownVeto: dropped %d/%d candidates (dd > -%.0f%% "
                "without fund confirmation)",
                len(vetoed), len(candidates), dd_threshold * 100,
            )
            for t, dd, sue, pead in vetoed[:5]:
                log.info("  %s dd=%+.1f%% sue=%s pead=%s",
                          t, dd * 100,
                          f"{sue:+.3f}"  if sue  is not None else "—",
                          f"{pead:+.3f}" if pead is not None else "—")
            if len(vetoed) > 5:
                log.info("  (+%d more vetoed)", len(vetoed) - 5)
            ctx.counters = getattr(ctx, "counters", None) or {}
            ctx.counters["deep_dd_vetoed"] = (
                ctx.counters.get("deep_dd_vetoed", 0) + len(vetoed)
            )
        ctx.candidates = kept


# ── Pure helpers ────────────────────────────────────────────────────────────


def _trailing_return(df, days: int) -> float | None:
    """Return ret_t = close[-1] / close[-days-1] - 1, or None if insufficient
    history. Pure function so the task remains easily testable."""
    if df is None:
        return None
    try:
        close = df["close"]
    except (KeyError, TypeError):
        return None
    if len(close) < days + 1:
        return None
    try:
        prior = float(close.iloc[-(days + 1)])
        latest = float(close.iloc[-1])
    except (IndexError, ValueError, TypeError):
        return None
    if not (math.isfinite(prior) and math.isfinite(latest)) or prior == 0:
        return None
    return latest / prior - 1.0


def _dd_from_high(df, window_days: int) -> float | None:
    """Return (latest_close - max_close_in_window) / max_close_in_window.
    Negative ≡ drawdown. None if insufficient history.
    """
    if df is None:
        return None
    try:
        close = df["close"]
    except (KeyError, TypeError):
        return None
    if len(close) < min(window_days, 20):
        return None
    try:
        hi = float(close.tail(window_days).max())
        cur = float(close.iloc[-1])
    except (ValueError, TypeError):
        return None
    if not (math.isfinite(hi) and math.isfinite(cur)) or hi == 0:
        return None
    return cur / hi - 1.0


def _feature(cand, name: str) -> float | None:
    """Try several locations where a fundamental feature might live on
    a CandidateResult. Returns None if not found / non-finite."""
    val: Any = getattr(cand, name, None)
    if val is None:
        # Try `cand.features` dict if it exists
        feats = getattr(cand, "features", None)
        if isinstance(feats, dict):
            val = feats.get(name)
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
