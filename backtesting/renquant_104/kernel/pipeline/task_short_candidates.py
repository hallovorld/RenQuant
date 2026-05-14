"""ShortCandidateSelectionTask — Long-Short Phase 2B.

Populates ``ctx.short_candidates`` with the bottom-decile of panel-LTR
scores when ``long_short.enabled = true``. No-op when disabled.

The QP solver (``BuildSourceMapTask``) already reads from
``ctx.short_candidates`` if present (Phase 2A wiring). This task fills
that source with bottom-of-rank names.

Selection logic
---------------
1. Read full-universe panel scores from ``ctx._panel_scores_all`` (stashed
   by ``ApplyScoresTask``).
2. Exclude tickers already in ``ctx.candidates`` (top-K longs) and
   ``ctx.holdings`` (already long-held — can't be short).
3. Exclude ETFs / non-shortable names from a static blacklist.
4. Take the bottom ``short_decile`` fraction (default 0.10 → bottom 10%).
5. Cap at ``max_shorts`` (default 5) by lowest score.
6. Wrap each in a ``CandidateResult`` with the panel score as both
   ``raw_score`` and ``panel_score``. Negative score signals "short".

References
----------
* Grinold-Kahn 1999 §5 — IR = IC × √breadth. Adding shorts doubles
  breadth → IR×√2 ≈ 40% improvement (theoretical).
* López de Prado, AFML 2018 §10 — meta-labeling for bet sizing.

Config (in side config, OFF by default):

    "long_short": {
      "enabled": true,
      "short_decile": 0.10,
      "max_shorts": 5,
      "etf_blacklist": ["SPY", "GLD", "QQQ", "IWM", "DIA"],
      "max_short_pct": 0.05,
      "max_gross_exposure": 1.30
    }

`max_short_pct` and `max_gross_exposure` are consumed by
``ComputeQPConstraintsTask`` — already wired in Phase 2A.
"""
from __future__ import annotations

import logging

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Task

log = logging.getLogger("kernel.pipeline.task_short_candidates")


# Default static ETF blacklist (non-individual-equity instruments).
# Override via `long_short.etf_blacklist` in config.
DEFAULT_ETF_BLACKLIST = frozenset({
    "SPY", "GLD", "QQQ", "IWM", "DIA", "TLT", "HYG", "LQD", "VXX",
    "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
})


class ShortCandidateSelectionTask(Task):
    """Phase 2B: pick bottom-decile of panel scores as short candidates.

    No-op when ``long_short.enabled`` is false (default).
    """

    name = "ShortCandidateSelectionTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        ls_cfg = (getattr(ctx, "config", None) or {}).get("long_short", {}) or {}
        if not ls_cfg.get("enabled", False):
            ctx.short_candidates = []  # ensure attribute exists
            return None

        # BEAR regime — no shorts (defensive). Phase 2A also skips shorts
        # at the QP constraint layer for BEAR, but we short-circuit here
        # to save the work too.
        if getattr(ctx, "bear_only", False):
            ctx.short_candidates = []
            log.info("ShortCandidateSelectionTask: BEAR mode → no shorts")
            return None

        scores = getattr(ctx, "_panel_scores_all", None)
        if scores is None or len(scores) == 0:
            log.warning(
                "ShortCandidateSelectionTask: ctx._panel_scores_all "
                "missing — was ApplyScoresTask run? Skipping shorts."
            )
            ctx.short_candidates = []
            return None

        # Exclusion sets — only ACTIVELY HELD tickers (overlap with long
        # candidate pool is OK; `BuildSourceMapTask` precedence guarantees
        # the long candidate wins when a ticker is in both sets).
        # Pre-fix this also excluded ctx.candidates (the broad admission
        # pool of 60-70 tickers), leaving the eligible set empty in 2026-05-14
        # smoke. The intent was "don't short tickers we're about to buy long" —
        # but ctx.candidates ⊃ {actual longs}, so excluding the whole pool
        # over-filters. The QP's source-map merging already handles overlap.
        held_tickers = set((ctx.holdings or {}).keys())
        blacklist = frozenset(
            ls_cfg.get("etf_blacklist", DEFAULT_ETF_BLACKLIST)
        )

        # Filter and sort ascending (lowest panel score first)
        eligible = scores.dropna().sort_values(ascending=True)
        eligible = eligible[~eligible.index.isin(held_tickers)]
        eligible = eligible[~eligible.index.isin(blacklist)]

        if len(eligible) == 0:
            ctx.short_candidates = []
            log.info("ShortCandidateSelectionTask: 0 eligible after filters")
            return None

        # Take bottom decile, capped at max_shorts
        short_decile = float(ls_cfg.get("short_decile", 0.10))
        max_shorts = int(ls_cfg.get("max_shorts", 5))
        n_bottom = max(1, int(len(eligible) * short_decile))
        n_picked = min(n_bottom, max_shorts)
        picks = eligible.head(n_picked)

        # Wrap in CandidateResult objects
        from kernel.selection import CandidateResult  # noqa: PLC0415
        ctx.short_candidates = [
            CandidateResult(
                ticker=t,
                raw_score=float(v),
                rank_score=float(v),
                rs_score=0.0,
                detail=f"short_candidate panel_score={v:.4f}",
                expected_return=-float(v),  # negative — short profits on decline
                panel_score=float(v),
            )
            for t, v in picks.items()
        ]
        log.info(
            "ShortCandidateSelectionTask: %d short candidates picked "
            "(bottom %.0f%%, max_shorts=%d) — tickers=%s",
            len(ctx.short_candidates), short_decile * 100, max_shorts,
            [c.ticker for c in ctx.short_candidates],
        )


__all__ = ["ShortCandidateSelectionTask", "DEFAULT_ETF_BLACKLIST"]
