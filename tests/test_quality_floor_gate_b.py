"""Tests for QualityFloorTask Gate B (Edge-Sharpe floor).

Design: ``doc/buy_logic_redesign_2026-04-26.md`` §2 / Gate B (Lo 2002).

Gate B: edge_sharpe = μ / σ; reject when below τ_S threshold or
σ ≤ 0 or μ NaN. Defaults: enabled=false, threshold=0.20.

Stage 0 contract: with all gates disabled, ctx.candidates is left
untouched (bit-for-bit parity with current behaviour).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.panel_pipeline.task_quality_floor import (  # noqa: E402
    QualityFloorTask,
    _gate_b_edge_sharpe,
)


@dataclass
class _Cand:
    ticker: str
    mu:    float | None = None
    sigma: float | None = None
    panel_score: float | None = None
    rank_score:  float | None = None


@dataclass
class _Ctx:
    config: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    holdings:   dict = field(default_factory=dict)
    regime: str | None = None


def _on_b(threshold: float = 0.20) -> dict:
    """Strategy-config dict with Gate B turned on at given threshold."""
    return {
        "ranking": {
            "panel_scoring": {
                "quality_floor": {
                    "enabled": True,
                    "edge_sharpe_floor": {
                        "enabled": True,
                        "threshold": threshold,
                    },
                },
            },
        },
    }


# ── Pure-function gate B ──────────────────────────────────────────────────────

class TestGateBPure:
    def test_passes_when_above_threshold(self):
        c = _Cand("A", mu=0.10, sigma=0.20)   # edge_sharpe = 0.5
        ok, reason = _gate_b_edge_sharpe(c, threshold=0.20)
        assert ok is True
        assert reason is None

    def test_rejects_when_below_threshold(self):
        c = _Cand("A", mu=0.008, sigma=0.08)  # 0.10 — today's R7 case
        ok, reason = _gate_b_edge_sharpe(c, threshold=0.20)
        assert ok is False
        assert "edge_sharpe" in reason
        assert "<0.200" in reason

    def test_rejects_negative_mu(self):
        c = _Cand("A", mu=-0.05, sigma=0.10)
        ok, _ = _gate_b_edge_sharpe(c, threshold=0.20)
        assert ok is False

    def test_rejects_zero_sigma(self):
        c = _Cand("A", mu=0.10, sigma=0.0)
        ok, reason = _gate_b_edge_sharpe(c, threshold=0.20)
        assert ok is False
        assert reason == "sigma_nonpositive"

    def test_rejects_nan_mu(self):
        """2026-05-04 audit Issue 24: NaN sigma OR mu now hits an
        isfinite guard FIRST; reason string changed to
        `sigma_or_mu_nonfinite`. Both outcomes correctly reject."""
        c = _Cand("A", mu=float("nan"), sigma=0.10)
        ok, reason = _gate_b_edge_sharpe(c, threshold=0.20)
        assert ok is False
        assert reason in {"mu_nan", "sigma_or_mu_nonfinite"}

    def test_passes_when_no_ngboost_attached(self):
        """Cand without μ/σ (no NGBoost ran) — gate should pass through."""
        c = _Cand("A", mu=None, sigma=None)
        ok, _ = _gate_b_edge_sharpe(c, threshold=0.20)
        assert ok is True


# ── Task integration ──────────────────────────────────────────────────────────

class TestQualityFloorTaskIntegration:
    def test_disabled_default_preserves_candidates(self):
        """No config block → all candidates pass through (Stage 0 contract)."""
        ctx = _Ctx(config={})
        ctx.candidates = [
            _Cand("A", mu=0.001, sigma=0.10),  # very weak; would fail Gate B
            _Cand("B", mu=0.002, sigma=0.10),
        ]
        QualityFloorTask().run(ctx)
        assert len(ctx.candidates) == 2

    def test_quality_floor_enabled_but_gate_b_off_preserves(self):
        ctx = _Ctx(config={
            "ranking": {"panel_scoring": {"quality_floor": {
                "enabled": True,
                "edge_sharpe_floor": {"enabled": False},
            }}}
        })
        ctx.candidates = [_Cand("A", mu=0.001, sigma=0.10)]
        QualityFloorTask().run(ctx)
        assert len(ctx.candidates) == 1

    def test_gate_b_filters_weak_signal(self):
        """R7 reproduction: μ=0.008 σ=0.08 → edge_sharpe=0.1 → reject at τ=0.2."""
        ctx = _Ctx(config=_on_b(0.20))
        strong = _Cand("STRONG", mu=0.05, sigma=0.10)   # 0.5
        weak   = _Cand("WEAK",   mu=0.008, sigma=0.08)  # 0.1
        ctx.candidates = [strong, weak]
        QualityFloorTask().run(ctx)
        kept = {c.ticker for c in ctx.candidates}
        assert kept == {"STRONG"}

    def test_blocked_by_ticker_populated(self):
        ctx = _Ctx(config=_on_b(0.20))
        ctx.candidates = [_Cand("WEAK", mu=0.008, sigma=0.08)]
        QualityFloorTask().run(ctx)
        blocked = getattr(ctx, "_blocked_by_ticker", {})
        assert "WEAK" in blocked
        assert blocked["WEAK"].startswith("quality_floor:gate_b:")

    def test_holdings_untouched(self):
        """Quality floor is buy-side; holdings shouldn't be filtered."""
        ctx = _Ctx(config=_on_b(0.20))
        ctx.candidates = [_Cand("WEAK", mu=0.008, sigma=0.08)]
        ctx.holdings = {
            "HELD": _Cand("HELD", mu=0.002, sigma=0.10),
        }
        QualityFloorTask().run(ctx)
        assert "HELD" in ctx.holdings   # untouched

    def test_no_candidates_short_circuits(self):
        ctx = _Ctx(config=_on_b(0.20))
        ctx.candidates = []
        QualityFloorTask().run(ctx)   # must not raise
        assert ctx.candidates == []

    def test_threshold_15_keeps_borderline_ngboost(self):
        """At τ=0.15 a μ=0.012 σ=0.08 (edge=0.15) borderline-passes."""
        ctx = _Ctx(config=_on_b(0.149))
        ctx.candidates = [_Cand("BORDERLINE", mu=0.012, sigma=0.08)]
        QualityFloorTask().run(ctx)
        assert len(ctx.candidates) == 1

    def test_threshold_25_rejects_borderline(self):
        ctx = _Ctx(config=_on_b(0.25))
        ctx.candidates = [_Cand("BORDERLINE", mu=0.012, sigma=0.08)]
        QualityFloorTask().run(ctx)
        assert ctx.candidates == []

    def test_cand_without_ngboost_passes_when_gate_b_on(self):
        """If μ/σ are None on the candidate, gate has no input → pass."""
        ctx = _Ctx(config=_on_b(0.20))
        ctx.candidates = [_Cand("NOMU", mu=None, sigma=None)]
        QualityFloorTask().run(ctx)
        assert len(ctx.candidates) == 1


class TestGateBRegimeConditionalRegressionGuard:
    """AUDIT REGRESSION GUARD: Gate B threshold is a regime knob.

    CLAUDE.md's prime directive requires numeric decision knobs to resolve
    through ``regime_params.<REGIME>`` first. Pre-fix, QualityFloorTask only
    read the global ``quality_floor.edge_sharpe_floor.threshold`` so a
    BULL_CALM-specific admission threshold had no effect.
    """

    def test_regime_threshold_overrides_global_threshold(self):
        config = _on_b(0.20)
        config["regime_params"] = {
            "BULL_CALM": {"edge_sharpe_floor_threshold": 0.10},
        }
        ctx = _Ctx(config=config, regime="BULL_CALM")
        ctx.candidates = [_Cand("BORDERLINE", mu=0.012, sigma=0.08)]  # edge=0.15

        QualityFloorTask().run(ctx)

        assert [c.ticker for c in ctx.candidates] == ["BORDERLINE"]
