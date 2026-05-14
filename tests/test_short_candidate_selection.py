"""Phase 2B regression: ShortCandidateSelectionTask invariants.

Pins the contract that ``ctx.short_candidates`` is populated only when
``long_short.enabled = true`` and that selection logic correctly:
- Excludes long candidates and holdings
- Excludes ETF blacklist
- Returns bottom-decile of panel scores capped at ``max_shorts``
- No-op when BEAR regime or disabled
- Defaults to [] (never None) so QP source-map can iterate safely
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _mk_ctx(scores_dict, *, long_tickers=(), held_tickers=(),
            ls_cfg=None, bear_only=False):
    """Minimal ctx with panel_scores_all + filter inputs."""
    scores = pd.Series(scores_dict, name="panel_score")
    cands = [SimpleNamespace(ticker=t) for t in long_tickers]
    holdings = {t: SimpleNamespace(ticker=t) for t in held_tickers}
    cfg = {}
    if ls_cfg is not None:
        cfg["long_short"] = ls_cfg
    ctx = SimpleNamespace(
        config=cfg,
        candidates=cands,
        holdings=holdings,
        bear_only=bear_only,
    )
    ctx._panel_scores_all = scores
    return ctx


class TestShortCandidateSelectionTask:

    def test_disabled_returns_empty_no_op(self):
        """No long_short config → short_candidates set to []."""
        from kernel.pipeline.task_short_candidates import ShortCandidateSelectionTask
        ctx = _mk_ctx({"AAPL": 0.5, "TSLA": -0.5})
        ShortCandidateSelectionTask().run(ctx)
        assert ctx.short_candidates == []

    def test_enabled_picks_bottom_decile(self):
        """enabled=true with 20 tickers → bottom 2 (decile=0.10)."""
        from kernel.pipeline.task_short_candidates import ShortCandidateSelectionTask
        scores = {f"T{i:02d}": float(i) for i in range(20)}  # T00 lowest
        ctx = _mk_ctx(scores, ls_cfg={"enabled": True, "short_decile": 0.10})
        ShortCandidateSelectionTask().run(ctx)
        tickers = [c.ticker for c in ctx.short_candidates]
        assert tickers == ["T00", "T01"], (
            f"Expected bottom 2 (T00, T01); got {tickers}"
        )

    def test_long_candidate_overlap_allowed(self):
        """Tickers in ctx.candidates CAN be short candidates. The QP source
        map (BuildSourceMapTask) handles overlap by long-precedence. This
        replaces the prior over-aggressive exclusion that left the
        eligible set empty in 2026-05-14 smoke (60-of-70 candidates).
        """
        from kernel.pipeline.task_short_candidates import ShortCandidateSelectionTask
        scores = {"TSLA": -1.0, "NFLX": -0.9, "AAPL": 0.5}
        # TSLA is a long candidate but also has the lowest score
        ctx = _mk_ctx(scores, long_tickers=["TSLA"],
                      ls_cfg={"enabled": True, "short_decile": 0.5})
        ShortCandidateSelectionTask().run(ctx)
        tickers = [c.ticker for c in ctx.short_candidates]
        # TSLA SHOULD be picked (lowest score) — QP resolves the long/short
        # collision at source-map merge time, not here
        assert "TSLA" in tickers

    def test_excludes_holdings(self):
        """Tickers already held long are not shorted."""
        from kernel.pipeline.task_short_candidates import ShortCandidateSelectionTask
        scores = {"AAPL": -1.0, "NFLX": -0.9, "MSFT": 0.5}
        ctx = _mk_ctx(scores, held_tickers=["AAPL"],
                      ls_cfg={"enabled": True, "short_decile": 0.5})
        ShortCandidateSelectionTask().run(ctx)
        tickers = [c.ticker for c in ctx.short_candidates]
        assert "AAPL" not in tickers
        assert "NFLX" in tickers

    def test_excludes_etf_blacklist(self):
        """ETFs (SPY, GLD, etc.) are never short candidates."""
        from kernel.pipeline.task_short_candidates import ShortCandidateSelectionTask
        scores = {"SPY": -1.0, "GLD": -0.9, "NFLX": -0.8}
        ctx = _mk_ctx(scores, ls_cfg={"enabled": True, "short_decile": 0.5})
        ShortCandidateSelectionTask().run(ctx)
        tickers = [c.ticker for c in ctx.short_candidates]
        assert "SPY" not in tickers
        assert "GLD" not in tickers
        assert "NFLX" in tickers

    def test_capped_by_max_shorts(self):
        """max_shorts caps the count even if decile would allow more."""
        from kernel.pipeline.task_short_candidates import ShortCandidateSelectionTask
        scores = {f"T{i:02d}": float(i) for i in range(50)}
        ctx = _mk_ctx(scores, ls_cfg={
            "enabled": True, "short_decile": 0.5, "max_shorts": 3,
        })
        ShortCandidateSelectionTask().run(ctx)
        assert len(ctx.short_candidates) == 3

    def test_bear_only_skips(self):
        """BEAR regime → no shorts (defensive)."""
        from kernel.pipeline.task_short_candidates import ShortCandidateSelectionTask
        ctx = _mk_ctx(
            {f"T{i:02d}": -float(i) for i in range(10)},
            ls_cfg={"enabled": True},
            bear_only=True,
        )
        ShortCandidateSelectionTask().run(ctx)
        assert ctx.short_candidates == []

    def test_missing_scores_skips_gracefully(self):
        """If ApplyScoresTask didn't run, no crash, empty result."""
        from kernel.pipeline.task_short_candidates import ShortCandidateSelectionTask
        ctx = SimpleNamespace(
            config={"long_short": {"enabled": True}},
            candidates=[],
            holdings={},
            bear_only=False,
        )
        # _panel_scores_all intentionally missing
        ShortCandidateSelectionTask().run(ctx)
        assert ctx.short_candidates == []

    def test_short_candidate_panel_score_is_negative_or_low(self):
        """Short candidates store the original (low) panel_score; the
        QP_BUY/SELL emission interprets shares<0 to mean short."""
        from kernel.pipeline.task_short_candidates import ShortCandidateSelectionTask
        scores = {"T01": -2.0, "T02": -1.5, "T03": 0.5}
        ctx = _mk_ctx(scores, ls_cfg={"enabled": True, "short_decile": 0.5})
        ShortCandidateSelectionTask().run(ctx)
        for c in ctx.short_candidates:
            assert c.panel_score < 0, (
                f"{c.ticker} panel_score={c.panel_score} should be < 0"
            )
