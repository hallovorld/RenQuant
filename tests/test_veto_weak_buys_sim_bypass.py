"""Test the RQ_SIM_BYPASS_BUY_FLOOR env-flag escape hatch in VetoWeakBuysTask.

Background (2026-05-30 audit):
  WF gate sim's `buy_floor: adaptive_mean_std` rejects all PatchTST candidates
  in per-cut sim because per-cut calibrators output narrow probability ranges
  (0.07-0.13) compared to daily shadow's 15-mo calibrator (0.49). The `mean+1σ`
  rule clears only the very top of the narrow range, leaving most cuts with
  zero trades. This isn't a model-quality failure; it's a methodology mismatch.

Bypass contract:
  - RQ_SIM_BYPASS_BUY_FLOOR=1 → VetoWeakBuysTask skips floor; all candidates pass
  - env unset / other value → original behavior preserved (floor applied)
  - Bypass is sim-only — production live/cron NEVER set this env var
  - Logs an INFO line so operators can see the bypass fired
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask  # noqa: E402


@dataclass
class _Cand:
    """Minimal candidate fixture — only the fields VetoWeakBuysTask reads."""
    ticker: str
    rank_score: float | None
    panel_score: float | None = None


class _Ctx:
    """Minimal InferenceContext fixture."""
    def __init__(self, candidates, config):
        self.candidates = candidates
        self.config = config
        # VetoWeakBuysTask writes ctx.counters["panel_vetoed"] when floor fires
        self.counters: dict = {}
        # ApplyScoresTask normally sets _blocked_by_ticker; needed for veto path
        self._blocked_by_ticker: dict = {}


def _make_ctx(scores: list[float], buy_floor: float | str | None = 0.50):
    return _Ctx(
        candidates=[_Cand(f"T{i}", s) for i, s in enumerate(scores)],
        config={
            "ranking": {
                "panel_scoring": {
                    "buy_floor": buy_floor,
                    "buy_floor_min": 0.20,
                    "buy_floor_adaptive_cap": 0.30,
                    "buy_floor_std_mult": 1.0,
                },
            },
        },
    )


class TestSimBypassBuyFloor:

    def test_env_unset_preserves_floor_behavior(self, monkeypatch):
        """Env not set → buy_floor=0.50 strict; candidates below dropped."""
        monkeypatch.delenv("RQ_SIM_BYPASS_BUY_FLOOR", raising=False)
        ctx = _make_ctx([0.10, 0.30, 0.55, 0.80])  # 4 cands
        VetoWeakBuysTask().run(ctx)
        # buy_floor=0.50 absolute → keeps 0.55 + 0.80 only
        kept = {c.ticker for c in ctx.candidates}
        assert kept == {"T2", "T3"}

    def test_env_zero_preserves_floor_behavior(self, monkeypatch):
        """Env explicitly '0' → strict (only '1' bypasses)."""
        monkeypatch.setenv("RQ_SIM_BYPASS_BUY_FLOOR", "0")
        ctx = _make_ctx([0.10, 0.30, 0.55, 0.80])
        VetoWeakBuysTask().run(ctx)
        kept = {c.ticker for c in ctx.candidates}
        assert kept == {"T2", "T3"}

    def test_env_set_to_one_bypasses_floor(self, monkeypatch):
        """RQ_SIM_BYPASS_BUY_FLOOR=1 → ALL candidates pass."""
        monkeypatch.setenv("RQ_SIM_BYPASS_BUY_FLOOR", "1")
        ctx = _make_ctx([0.10, 0.30, 0.55, 0.80])
        VetoWeakBuysTask().run(ctx)
        # All 4 kept — even those below the configured 0.50 floor
        kept = {c.ticker for c in ctx.candidates}
        assert kept == {"T0", "T1", "T2", "T3"}

    def test_bypass_works_with_adaptive_floor(self, monkeypatch):
        """When floor is the adaptive_mean_std rule (the actual prod-derived
        config flavour), bypass still skips entirely."""
        monkeypatch.setenv("RQ_SIM_BYPASS_BUY_FLOOR", "1")
        # PatchTST-like narrow distribution (mean≈0.50, std≈0.02)
        narrow_scores = [0.48, 0.49, 0.50, 0.51, 0.52, 0.53]
        ctx = _make_ctx(narrow_scores, buy_floor="adaptive_mean_std")
        VetoWeakBuysTask().run(ctx)
        # mean+1σ ≈ 0.52 would normally keep only [0.53] — but bypass kept all
        kept = {c.ticker for c in ctx.candidates}
        assert len(kept) == 6

    def test_bypass_logs_info_line(self, monkeypatch, caplog):
        """Operators must see when bypass fires (audit trail)."""
        import logging
        monkeypatch.setenv("RQ_SIM_BYPASS_BUY_FLOOR", "1")
        caplog.set_level(logging.INFO, logger="kernel.panel_pipeline.scoring")
        ctx = _make_ctx([0.10, 0.30])
        VetoWeakBuysTask().run(ctx)
        assert any(
            "RQ_SIM_BYPASS_BUY_FLOOR=1" in rec.message
            and "distribution-fair sim mode" in rec.message
            for rec in caplog.records
        )

    def test_empty_candidates_unaffected(self, monkeypatch):
        """No candidates → no-op regardless of env."""
        monkeypatch.setenv("RQ_SIM_BYPASS_BUY_FLOOR", "1")
        ctx = _make_ctx([], buy_floor=0.50)
        result = VetoWeakBuysTask().run(ctx)
        assert result is None
        assert ctx.candidates == []

    def test_buy_floor_unset_unaffected(self, monkeypatch):
        """If config has no buy_floor at all, bypass is moot — task returns
        early before reaching the env check (preserves no-op semantics)."""
        monkeypatch.setenv("RQ_SIM_BYPASS_BUY_FLOOR", "1")
        ctx = _make_ctx([0.10, 0.50], buy_floor=None)
        VetoWeakBuysTask().run(ctx)
        # Both kept (no floor anyway, with or without bypass)
        assert len(ctx.candidates) == 2

    def test_env_set_to_arbitrary_other_value_preserves_strict(self, monkeypatch):
        """Only the literal string '1' triggers bypass (safety: typos like
        'true' or 'yes' should NOT silently turn off the floor in prod)."""
        for val in ("true", "TRUE", "yes", "on", "2"):
            monkeypatch.setenv("RQ_SIM_BYPASS_BUY_FLOOR", val)
            ctx = _make_ctx([0.10, 0.30, 0.55, 0.80])
            VetoWeakBuysTask().run(ctx)
            kept = {c.ticker for c in ctx.candidates}
            assert kept == {"T2", "T3"}, (
                f"env={val!r} should NOT bypass — only '1' does"
            )
