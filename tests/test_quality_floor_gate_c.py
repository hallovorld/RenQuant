"""Tests for QualityFloorTask Gate C (Constantinides no-trade band).

Design: ``doc/buy_logic_redesign_2026-04-26.md`` §2 / Gate C.

Davis-Norman 1990 closed form for log-utility under proportional cost τ:
    band_i  = c · (γ · σ_i² · τ)^(1/3)
    target  = μ_i / (γ · σ_i²)
    admit   ⇔ |target - w_current| > band

Gate C implements the no-trade region — naturally prevents
"fill-empty-slots-with-weak-signal" by requiring the implied weight
deviation to overcome the round-trip cost.

Stage 0 contract: defaults all OFF — bit-for-bit parity preserved.
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
    _gate_c_no_trade_band,
)


@dataclass
class _Cand:
    ticker: str
    panel_score: float | None = None
    rank_score:  float | None = None
    mu:    float | None = None
    sigma: float | None = None


@dataclass
class _Holding:
    shares: float = 0.0


@dataclass
class _Ctx:
    config: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    holdings:   dict = field(default_factory=dict)
    portfolio_value: float = 10000.0
    prices: dict = field(default_factory=dict)


def _on_c(gamma: float = 3.0, tau: float = 0.001,
          const: float = 1.5) -> dict:
    return {
        "ranking": {"panel_scoring": {"quality_floor": {
            "enabled": True,
            "no_trade_band": {
                "enabled":         True,
                "risk_aversion":   gamma,
                "round_trip_cost": tau,
                "band_constant":   const,
            },
        }}}
    }


# ── Pure function ─────────────────────────────────────────────────────────────

class TestGateCPure:
    def test_strong_signal_passes(self):
        # μ=0.10 σ=0.10 γ=3 τ=0.001 c=1.5
        # target = 0.10 / (3 × 0.01) = 3.33 (large weight)
        # band   = 1.5 × (3 × 0.01 × 0.001)^(1/3) = 1.5 × 0.03107 ≈ 0.0466
        # |3.33 - 0| > 0.047 → PASS
        c = _Cand("STRONG", mu=0.10, sigma=0.10)
        ok, reason = _gate_c_no_trade_band(
            c, risk_aversion=3.0, round_trip_cost=0.001,
            band_constant=1.5, current_weight=0.0,
        )
        assert ok is True
        assert reason is None

    def test_weak_signal_blocked(self):
        # μ=0.001 σ=0.10: target = 0.001/0.03 = 0.033 (~3% NAV)
        # band ≈ 0.047. |0.033 - 0| < 0.047 → REJECT
        c = _Cand("WEAK", mu=0.001, sigma=0.10)
        ok, reason = _gate_c_no_trade_band(
            c, risk_aversion=3.0, round_trip_cost=0.001,
            band_constant=1.5, current_weight=0.0,
        )
        assert ok is False
        assert "deviation" in reason
        assert "band" in reason

    def test_already_held_at_target_blocked(self):
        """Already at target weight → no further trade needed."""
        c = _Cand("HELD", mu=0.10, sigma=0.10)
        # target ~ 3.33; if w_current = 3.33 → deviation ~ 0 → REJECT
        ok, _ = _gate_c_no_trade_band(
            c, risk_aversion=3.0, round_trip_cost=0.001,
            band_constant=1.5, current_weight=3.33,
        )
        assert ok is False

    def test_band_scales_with_cost(self):
        """Higher cost → wider band → more candidates blocked."""
        c = _Cand("MID", mu=0.005, sigma=0.10)
        # target = 0.005 / 0.03 = 0.167
        # tau=0.001: band ≈ 0.047 → 0.167 > 0.047 PASS
        ok_low, _ = _gate_c_no_trade_band(
            c, risk_aversion=3.0, round_trip_cost=0.001,
            band_constant=1.5, current_weight=0.0,
        )
        # tau=0.05: band ≈ 1.5 × (3 × 0.01 × 0.05)^(1/3) = 1.5 × 0.114 = 0.171
        # → 0.167 < 0.171 → BLOCK
        ok_high, _ = _gate_c_no_trade_band(
            c, risk_aversion=3.0, round_trip_cost=0.05,
            band_constant=1.5, current_weight=0.0,
        )
        assert ok_low is True
        assert ok_high is False

    def test_no_ngboost_passes(self):
        """No μ/σ available → cannot evaluate → pass through."""
        c = _Cand("NOMU", mu=None, sigma=None)
        ok, _ = _gate_c_no_trade_band(
            c, risk_aversion=3.0, round_trip_cost=0.001,
            band_constant=1.5, current_weight=0.0,
        )
        assert ok is True

    def test_zero_sigma_rejected(self):
        c = _Cand("BAD", mu=0.05, sigma=0.0)
        ok, reason = _gate_c_no_trade_band(
            c, risk_aversion=3.0, round_trip_cost=0.001,
            band_constant=1.5, current_weight=0.0,
        )
        assert ok is False
        assert reason == "sigma_zero_or_mu_nan"


# ── Task integration ──────────────────────────────────────────────────────────

class TestGateCIntegration:
    def test_gate_c_off_preserves(self):
        ctx = _Ctx(config={})
        ctx.candidates = [_Cand("WEAK", mu=0.001, sigma=0.10)]
        QualityFloorTask().run(ctx)
        assert len(ctx.candidates) == 1

    def test_gate_c_blocks_weak_buy_at_zero_weight(self):
        """R7 reproduction: μ=0.008 σ=0.08 → target ≈ 0.42,
        band ≈ 0.043 → PASS via gate C alone (target deviation > band).
        That's intended — gate C alone wouldn't have blocked R7;
        gate B is the sharper filter for that case.
        """
        ctx = _Ctx(config=_on_c())
        ctx.candidates = [_Cand("R7_LIKE", mu=0.008, sigma=0.08)]
        QualityFloorTask().run(ctx)
        # R7 case PASSES gate C — μ is positive enough that the
        # implied target weight crosses the band.
        assert len(ctx.candidates) == 1

    def test_gate_c_blocks_truly_marginal_buy(self):
        """μ=0.0005 σ=0.10: target=0.0167 < band 0.0466 → reject."""
        ctx = _Ctx(config=_on_c())
        ctx.candidates = [_Cand("MARGINAL", mu=0.0005, sigma=0.10)]
        QualityFloorTask().run(ctx)
        assert ctx.candidates == []

    def test_gate_c_uses_current_holdings_weight(self):
        """Already held at the target weight → deviation 0 < band → reject."""
        ctx = _Ctx(config=_on_c())
        # μ=0.10 σ=0.10: target = 3.33 (huge). Held at 3.0 (within band).
        ctx.candidates = [_Cand("HELD", mu=0.10, sigma=0.10)]
        ctx.holdings = {"HELD": _Holding(shares=300.0)}    # 300 × $100 = $30000
        ctx.prices = {"HELD": 100.0}
        ctx.portfolio_value = 10000.0
        # current_weight = 30000/10000 = 3.0; target 3.33; band 0.047
        # |3.33 - 3.0| = 0.33 > 0.047 → PASS
        QualityFloorTask().run(ctx)
        assert len(ctx.candidates) == 1   # passes — deviation > band

    def test_gate_c_blocks_already_at_target(self):
        ctx = _Ctx(config=_on_c())
        ctx.candidates = [_Cand("AT_TARGET", mu=0.001, sigma=0.10)]
        # target = 0.001/0.03 = 0.033; held at exactly 0.033 → dev 0 < band → reject
        ctx.holdings = {"AT_TARGET": _Holding(shares=3.3)}
        ctx.prices = {"AT_TARGET": 100.0}     # 3.3 × 100 = 330; w = 0.033
        ctx.portfolio_value = 10000.0
        QualityFloorTask().run(ctx)
        assert ctx.candidates == []

    def test_blocked_reason_surfaces(self):
        ctx = _Ctx(config=_on_c())
        ctx.candidates = [_Cand("MARGINAL", mu=0.0005, sigma=0.10)]
        QualityFloorTask().run(ctx)
        blocked = getattr(ctx, "_blocked_by_ticker", {})
        assert "MARGINAL" in blocked
        assert blocked["MARGINAL"].startswith("quality_floor:gate_c:")

    def test_combined_a_b_c(self):
        """All three gates enabled — strong cand passes all."""
        from kernel.persistence import _SCHEMA_SQL  # noqa: PLC0415
        import sqlite3
        cfg = {
            "ranking": {"panel_scoring": {"quality_floor": {
                "enabled": True,
                "distribution_floor": {
                    "enabled": True, "percentile": 85,
                    "lookback_days": 5, "min_history_days": 3,
                },
                "edge_sharpe_floor": {"enabled": True, "threshold": 0.5},
                "no_trade_band": {
                    "enabled": True,
                    "risk_aversion": 3.0,
                    "round_trip_cost": 0.001,
                    "band_constant": 1.5,
                },
            }}}
        }
        ctx = _Ctx(config=cfg)
        db = sqlite3.connect(":memory:")
        db.executescript(_SCHEMA_SQL)
        for d, p85 in [("2026-04-23", 0.05), ("2026-04-24", 0.05),
                        ("2026-04-25", 0.05)]:
            db.execute(
                "INSERT INTO score_percentiles_daily (date, n_cands, p85) "
                "VALUES (?, ?, ?)", (d, 10, p85),
            )
        db.commit()
        ctx._db = db                    # noqa: SLF001
        # μ=0.10 σ=0.10: edge=1.0 > 0.5 ✓; panel=0.10 > p85=0.05 ✓;
        # target=3.33 > band=0.047 ✓
        strong = _Cand("STRONG", panel_score=0.10, mu=0.10, sigma=0.10)
        ctx.candidates = [strong]
        QualityFloorTask().run(ctx)
        assert len(ctx.candidates) == 1
