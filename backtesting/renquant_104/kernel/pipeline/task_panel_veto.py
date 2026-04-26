"""PanelRankVetoTask — block per-ticker `model_sell` exits when the held
position still ranks strong in the cross-sectional panel.

**Theoretical motivation** (per user audit 2026-04-26):

Per-ticker XGBoost / Classification / QLearning / Manual models are
trained in isolation — they only see the ticker's own indicators
(RSI, MACD, momentum…). They have no awareness of:
  * the ticker's panel rank vs other watchlist tickers
  * the current regime (BULL_CALM / CHOPPY / BEAR)
  * relative strength vs sector ETF or SPY

Consequence: a ticker that ranks #1 in the panel (rank_score ~ 0.85)
can still trigger a per-ticker `model_sell` because its OWN MACD
crossed down, even though it's the strongest holding in the universe.
Round-3 e2e showed exactly this — GOOG and AMZN both sold despite
panel-LTR not flagging them as weak (joint_sell menu was empty).

Fix: when a per-ticker model_sell would fire, check the panel-LTR's
calibrated rank_score for the held. If it's above `min_rank_score`
(default 0.5 = >50% probability of forward outperformance), VETO
the sell and let the position stay another bar.

**What we DON'T veto** (these are risk-driven, must fire regardless
of panel rank):
  * stop_loss     - drawdown breach
  * trailing_stop - HWM-based protection
  * single_day_loss / gap_down - intraday catastrophe
  * max_hold      - tax / holding-period gate
  * rotation      - already a panel-aware swap
  * joint_sell    - panel-driven exit (already conscious of panel)
  * kelly_trim    - panel-aware rebalance (μ/σ² target)

**Streak cap** (audit fix PV-NEW-7): when the per-ticker model has been
signalling SELL for ``max_streak_to_veto`` consecutive **trading days**
(default 5 — about 1 week of trading), the veto STOPS firing for that
held — persistent weakness override. Prevents "stuck" scenarios where
panel ranks the held strong forever but the per-ticker model keeps
insisting on exit. Note: each bar = one trading day; weekends excluded.

**Dependency** (audit fix PV-NEW-9): this task assumes
``ranking.panel_scoring.global_calibration.enabled = true`` so that
`held.rank_score` is the calibrated panel-LTR probability. Without
calibration, `held.rank_score` would be the per-ticker model score
(which is what we're trying to veto on) — using it as the veto signal
would be self-contradictory. Run-time check below logs a warning.

**Telemetry**: every veto logged with ticker / score / threshold +
streak. ctx.counters["model_sell_vetoed"] tracks count. ctx.exits_vetoed
list lets ntfy surface "would have sold X but panel says strong".

Pipeline ordering: this Task runs at the START of Phase 3 (after
PanelScoringJob populates rank_score on holdings, but BEFORE
RankingJob / JointActionJob / SelectionJob act on the exits).
"""
from __future__ import annotations

import logging
import math
from typing import Any

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.panel_veto")


# Exit types that are RISK-DRIVEN and must fire regardless of panel rank.
# These are user-protective and should never be deferred just because
# the model thinks the ticker is otherwise strong.
# Audit fix PV-NEW-8 (2026-04-26 round-3): canonical exit_type names
# from kernel/exits.py + JointActionTask. Hardcoded for explicitness;
# the test_panel_rank_veto.TestRiskExitsNeverVetoed parametrizes over
# this set so a name drift would surface immediately.
RISK_EXIT_TYPES: frozenset[str] = frozenset({
    "stop_loss",
    "trailing_stop",
    "single_day_loss",
    "max_hold",
    "rotation",          # already a panel-aware swap; don't undo it
    "kelly_trim",        # Kelly-target rebalance (panel-aware via μ/σ²)
    "joint_sell",        # panel-driven exit (already considers panel)
    "joint_rotation",
    # Defensive aliases — variants seen in older configs / docs:
    "trailing_stop_loss",
    "gap_down",
    "max_hold_days",
})


class PanelRankVetoTask(Task):
    """Veto `model_sell` exits when the held's rank_score is strong.

    Reads:
      ctx.holdings (must have rank_score populated by PanelScoringJob)
      ctx.exits    (list of (ticker, ExitSignal) tuples)
      ctx.config["model_sell"]["panel_veto"]

    Mutates:
      ctx.exits         — vetoed entries removed
      ctx.exits_vetoed  — list of veto records {ticker, exit_type, reason, rank_score, threshold}
      ctx.counters["model_sell_vetoed"]
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        cfg = ((ctx.config.get("model_sell") or {})
                       .get("panel_veto") or {})
        if not cfg.get("enabled", False):
            return False
        if not ctx.exits:
            return False

        # Audit fix PV-NEW-9 (2026-04-26 round-3): defensive check that
        # global_calibration is enabled. Without it, held.rank_score is
        # the per-ticker model's score (the very thing we're trying to
        # veto on) — self-contradictory. Log a warning but proceed —
        # operator may have intentional reasons (e.g., debugging).
        cal_enabled = bool(
            (ctx.config.get("ranking", {})
                       .get("panel_scoring", {})
                       .get("global_calibration", {})
                       .get("enabled", False))
        )
        if not cal_enabled:
            log.warning(
                "PanelRankVetoTask: ranking.panel_scoring.global_calibration "
                "is DISABLED. held.rank_score will be the per-ticker model "
                "score (not panel-calibrated). Veto logic may be misleading. "
                "Enable global_calibration for correct semantics."
            )

        min_rank_score = float(cfg.get("min_rank_score", 0.50))
        # Audit fix PV-NEW-7 (2026-04-26 round-3): streak cap so a held
        # whose per-ticker model has been screaming SELL for N consecutive
        # days is allowed to exit despite panel saying strong. Default 5.
        max_streak_to_veto = int(cfg.get("max_streak_to_veto", 5))
        # Allow override of which exit_types are vetoable. Default: only
        # model_sell. Audit fix PV-NEW-3 (2026-04-26 round-3): defend
        # against operator passing a STRING rather than a list (set("xyz")
        # would split into characters). Wrap single-string to single-element list.
        _vetoable_raw = cfg.get("vetoable_exit_types", ["model_sell"])
        if isinstance(_vetoable_raw, str):
            _vetoable_raw = [_vetoable_raw]
        vetoable_set: set[str] = set(_vetoable_raw)
        # Operator can also add to RISK_EXIT_TYPES via override.
        _extra_raw = cfg.get("extra_risk_exit_types", [])
        if isinstance(_extra_raw, str):
            _extra_raw = [_extra_raw]
        extra_risk = set(_extra_raw)
        risk_set = RISK_EXIT_TYPES | extra_risk

        if not hasattr(ctx, "exits_vetoed"):
            ctx.exits_vetoed = []

        kept: list = []
        n_vetoed = 0
        for ticker, sig in ctx.exits:
            exit_type = str(getattr(sig, "exit_type", "") or "")
            # Skip risk-driven exits — they must fire.
            if exit_type in risk_set:
                kept.append((ticker, sig))
                continue
            # Only veto exit types in the vetoable set.
            if exit_type not in vetoable_set:
                kept.append((ticker, sig))
                continue
            # Get held's panel-LTR rank_score (calibrated probability).
            held = ctx.holdings.get(ticker)
            if held is None:
                # Held already gone — no decision to veto.
                kept.append((ticker, sig))
                continue
            rank_score = getattr(held, "rank_score", None)
            # Defensive: NaN / None rank_score → don't veto (let exit fire).
            # Pre-fix risk: silent veto on missing data would BLOCK risk
            # exit fallback. Post-fix: safe-default to NOT veto.
            if rank_score is None or not math.isfinite(float(rank_score)):
                kept.append((ticker, sig))
                continue
            score_f = float(rank_score)
            if score_f <= min_rank_score:
                # Panel says weak too — exit fires.
                kept.append((ticker, sig))
                continue

            # Audit fix PV-NEW-7 (streak cap): if the per-ticker model
            # has been signalling SELL for >= max_streak_to_veto days,
            # let it through. Persistent weakness overrides panel veto.
            sell_streak = int(getattr(held, "sell_streak", 0) or 0)
            if max_streak_to_veto > 0 and sell_streak >= max_streak_to_veto:
                kept.append((ticker, sig))
                log.info(
                    "PANEL_VETO_BYPASS  %-6s  sell_streak=%d >= cap=%d "
                    "→ allow exit (rank_score=%.3f, persistent weakness override)",
                    ticker, sell_streak, max_streak_to_veto, score_f,
                )
                ctx.counters["model_sell_veto_bypassed"] = (
                    ctx.counters.get("model_sell_veto_bypassed", 0) + 1
                )
                continue

            # Veto: panel says held is strong. Let it stay.
            ctx.exits_vetoed.append({
                "ticker":      ticker,
                "exit_type":   exit_type,
                "reason":      getattr(sig, "reason", ""),
                "rank_score":  score_f,
                "threshold":   min_rank_score,
                "sell_streak": sell_streak,
            })
            n_vetoed += 1
            log.info(
                "PANEL_VETO  %-6s  exit_type=%s  rank_score=%.3f > "
                "threshold=%.3f  streak=%d/%d → keep position (panel says strong)",
                ticker, exit_type, score_f, min_rank_score,
                sell_streak, max_streak_to_veto,
            )

        if n_vetoed > 0:
            ctx.counters["model_sell_vetoed"] = (
                ctx.counters.get("model_sell_vetoed", 0) + n_vetoed
            )
            log.info(
                "PanelRankVetoTask: vetoed %d %s exit(s) (kept %d)",
                n_vetoed, ", ".join(sorted(vetoable_set)), len(kept),
            )
        ctx.exits = kept
