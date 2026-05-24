"""Regression tests for 2026-05-17 detector fix (A: 5-day BEAR, C: vol-cluster CHOPPY).

Invariants pinned:
  • A: hard_bear fires if EITHER 20-day OR 5-day threshold trips
        (5-day defaults: vol > 0.25 ann, ret < -0.04)
  • A: 20-day-only baseline behavior preserved when 5-day clean
  • C: vol_cluster_choppy fires when vol_5d > vol_60d × 1.5 AND |ret_20d| < 0.02
  • C: RegimeFinalizeTask routes vol_cluster_choppy → CHOPPY when hard_bear is False
  • Regression: BULL_STRONG-like calm bull (low vol, positive drift)
        does NOT spuriously fire BEAR or CHOPPY

Both task path (task_regime.py::BEAROverrideTask + RegimeFinalizeTask)
and standalone path (regime.py::detect_regime) must agree (PRIME
DIRECTIVE: regime detector is the foundation of regime-conditional
strategy).
"""
from __future__ import annotations
import sys
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.regime import RegimeState, detect_regime  # noqa: E402
from kernel.pipeline.task_regime import (  # noqa: E402
    BEAROverrideTask, RegimeFinalizeTask,
)


def _ctx(returns: np.ndarray, spy_df: pd.DataFrame | None = None,
         regime_state: RegimeState | None = None,
         config: dict | None = None):
    """Build a minimal InferenceContext-like SimpleNamespace.

    Includes the attributes RegimeFinalizeTask mutates: regime,
    confidence, regime_counts. Without these, the finalize task raises
    AttributeError when it tries to write back.
    """
    state = regime_state or RegimeState()
    return SimpleNamespace(
        spy_returns=returns,
        regime_state=state,
        config=config or {},
        ohlcv={"SPY": spy_df} if spy_df is not None else {},
        today=None,
        regime=None,
        confidence=0.0,
        regime_counts={},
    )


def _make_spy_df(closes: list[float]) -> pd.DataFrame:
    """Build a SPY OHLCV-like frame with a real DatetimeIndex (200+ bars)."""
    idx = pd.date_range("2023-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"close": closes}, index=idx)


# ── A: 5-day BEAR trigger ─────────────────────────────────────────────────────

class TestFiveDayBearTrigger:
    """Fix A: hard_bear fires on a SVB-like 1-week crash even though
    20-day windows look benign."""

    def test_5day_drop_triggers_hard_bear(self):
        """SVB-like: flat for 20 days, then -1.5% × 4 days. 5-day cumret
        drops below -4% → hard_bear=True even though 20-day cumret is
        only ~-6% (above default -8% threshold)."""
        # 16 benign days + 4 sharp down days  → 5-day window = last 5
        # = 1 benign + 4 × -0.015 → cum_ret ≈ -5.9%, vol5d ≈ 0.25
        returns = np.concatenate([
            np.array([0.001] * 16),
            np.array([-0.015, -0.015, -0.015, -0.015]),
        ])
        ctx = _ctx(returns, config={})
        BEAROverrideTask().run(ctx)
        assert ctx.regime_state.hard_bear is True, \
            f"5-day -6% drop should fire hard_bear (vol_5d={ctx.regime_state.vol_5d:.3f}, ret_5d={ctx.regime_state.ret_5d:.3f})"
        assert ctx.regime_state.ret_5d < -0.04

    def test_20day_only_check_still_works(self):
        """Pre-fix behavior: 20-day cumret < -8% fires hard_bear, even
        if 5-day is benign. Regression guard."""
        # 20 days of -0.005 = -9.6% cumret (crosses -8%)
        returns = np.array([-0.005] * 20)
        ctx = _ctx(returns, config={})
        BEAROverrideTask().run(ctx)
        assert ctx.regime_state.hard_bear is True

    def test_5day_high_vol_only_triggers(self):
        """High realized vol over 5 days (no big drawdown) triggers
        5-day vol gate."""
        # alternating ±2% for 4 days, last day flat → vol_5d ≈ 32%, ret_5d ≈ +0%
        returns = np.concatenate([
            np.array([0.001] * 16),
            np.array([+0.02, -0.02, +0.02, -0.02]),
        ])
        ctx = _ctx(returns, config={})
        BEAROverrideTask().run(ctx)
        # vol_5d should cross 0.25 (annualized)
        assert ctx.regime_state.vol_5d > 0.25
        assert ctx.regime_state.hard_bear is True

    def test_calm_bull_does_not_trigger(self):
        """REGRESSION GUARD: calm bull (low vol, mildly positive returns)
        must NOT fire hard_bear. Without this, fix A would over-trigger."""
        np.random.seed(42)
        # 100 days of N(+0.0006, 0.008) — calm bull, ~15% ann vol, +15% drift
        returns = np.random.normal(0.0006, 0.008, 100)
        ctx = _ctx(returns, config={})
        BEAROverrideTask().run(ctx)
        assert ctx.regime_state.hard_bear is False, \
            f"calm bull spuriously fired hard_bear " \
            f"(vol_5d={ctx.regime_state.vol_5d:.3f} ret_5d={ctx.regime_state.ret_5d:.3f})"


# ── C: vol-cluster CHOPPY ─────────────────────────────────────────────────────

class TestVolClusterChoppy:
    """Fix C: vol_cluster_choppy = (vol_5d > vol_60d × 1.5) AND
    (|ret_20d| < 0.02). Resurrects CHOPPY without depending on dead
    Hurst<0.52 test."""

    def test_vol_spike_with_no_drift_fires_choppy(self):
        """60 days of N(0, 0.005) then 5 days of N(0, 0.015) — no
        drift, 3× vol spike. Should fire vol_cluster_choppy."""
        np.random.seed(7)
        base = np.random.normal(0.0, 0.005, 60)  # vol_60d ~ 8%
        spike = np.array([+0.015, -0.015, +0.015, -0.015, +0.001])  # vol_5d ~ 24%
        # Mean-zero spike → cumret ≈ +0.001 (close to 0)
        returns = np.concatenate([base, spike])
        ctx = _ctx(returns, config={})
        BEAROverrideTask().run(ctx)
        assert ctx.regime_state.vol_cluster_choppy is True, \
            f"vol spike with no drift should fire CHOPPY " \
            f"(vol_5d={ctx.regime_state.vol_5d:.3f}, " \
            f"ret_5d={ctx.regime_state.ret_5d:.3f}, " \
            f"hard_bear={ctx.regime_state.hard_bear})"

    def test_vol_cluster_log_format_is_valid(self, monkeypatch):
        """Audit logs are part of the decision trace; formatting must not
        throw when the CHOPPY vol-cluster branch fires."""
        np.random.seed(7)
        base = np.random.normal(0.0, 0.005, 60)
        spike = np.array([+0.015, -0.015, +0.015, -0.015, +0.001])
        ctx = _ctx(np.concatenate([base, spike]), config={})

        messages: list[str] = []

        def strict_info(msg, *args, **kwargs):
            messages.append(msg % args if args else msg)

        monkeypatch.setattr("kernel.pipeline.task_regime.log.info", strict_info)
        BEAROverrideTask().run(ctx)

        assert ctx.regime_state.vol_cluster_choppy is True
        assert any("vol-cluster CHOPPY" in m and "drift20d=" in m for m in messages)

    def test_trending_market_does_not_fire_choppy(self):
        """REGRESSION GUARD: a strong trend (|ret_20d| > 0.02) MUST NOT
        fire CHOPPY even if 5-day vol elevated."""
        np.random.seed(11)
        base = np.random.normal(0.0, 0.005, 60)
        # 20 days uptrend +0.5%/day = +10% cumret
        uptrend = np.array([0.005] * 20)
        returns = np.concatenate([base, uptrend])
        ctx = _ctx(returns, config={})
        BEAROverrideTask().run(ctx)
        # 20-day cumret ≈ +10%, far above 2% threshold
        assert ctx.regime_state.vol_cluster_choppy is False

    def test_hard_bear_takes_precedence_over_choppy(self):
        """If both hard_bear AND vol_cluster_choppy fire, regime routes
        to BEAR (precedence rule in RegimeFinalizeTask)."""
        # Build a series that triggers both: low-vol baseline + sharp
        # 5-day drop. cum_ret_20d drops past -8% via the 4 down days,
        # so hard_bear fires. vol_cluster might also fire (drift now
        # too large to satisfy CHOPPY no-trend gate though).
        returns = np.concatenate([
            np.array([0.0] * 16),
            np.array([-0.025, -0.025, -0.025, -0.025]),
        ])
        ctx = _ctx(returns, regime_state=RegimeState(hurst_regime="MOMENTUM"),
                   spy_df=_make_spy_df([100.0] * 200), config={})
        BEAROverrideTask().run(ctx)
        assert ctx.regime_state.hard_bear is True
        # Now route through finalize: hard_bear → BEAR (not CHOPPY)
        RegimeFinalizeTask().run(ctx)
        assert ctx.regime_state.regime == "BEAR"


# ── Routing: vol_cluster_choppy → CHOPPY through RegimeFinalizeTask ──────────

class TestRegimeRouting:
    """RegimeFinalizeTask wires state.vol_cluster_choppy → CHOPPY when
    hard_bear is False."""

    def test_vol_cluster_choppy_routes_to_choppy_via_momentum(self):
        """Hurst=MOMENTUM, no bearish trend, vol_cluster=True → CHOPPY."""
        state = RegimeState(hurst_regime="MOMENTUM", vol_cluster_choppy=True,
                            hard_bear=False)
        # Build SPY df with close > both MAs (not bearish trend)
        closes = [100.0 + i * 0.01 for i in range(200)]
        ctx = _ctx(np.array([0.001] * 30), spy_df=_make_spy_df(closes),
                   regime_state=state, config={})
        RegimeFinalizeTask().run(ctx)
        assert state.regime == "CHOPPY"
        assert ctx._regime_evidence["source"] == "hurst_momentum_vol_cluster_choppy"
        assert ctx._regime_evidence["spy_bearish_trend"] is False

    def test_vol_cluster_choppy_routes_to_choppy_via_ambiguous(self):
        """Hurst=AMBIGUOUS, vol_cluster=True → CHOPPY (not GMM fallback)."""
        state = RegimeState(hurst_regime="AMBIGUOUS", vol_cluster_choppy=True,
                            hard_bear=False, gmm_probs={"BULL_CALM": 0.6})
        ctx = _ctx(np.array([0.001] * 30), regime_state=state, config={})
        RegimeFinalizeTask().run(ctx)
        assert state.regime == "CHOPPY"
        assert ctx._regime_evidence["source"] == "vol_cluster_choppy"

    def test_no_vol_cluster_no_hurst_reversion_keeps_bull(self):
        """REGRESSION GUARD: without vol_cluster AND without Hurst<0.52,
        we get BULL_CALM (or AMBIGUOUS→GMM dominant). Must NOT route
        to CHOPPY spuriously."""
        state = RegimeState(hurst_regime="MOMENTUM", vol_cluster_choppy=False,
                            hard_bear=False)
        closes = [100.0 + i * 0.05 for i in range(200)]  # uptrend, above MAs
        ctx = _ctx(np.array([0.001] * 30), spy_df=_make_spy_df(closes),
                   regime_state=state, config={})
        RegimeFinalizeTask().run(ctx)
        assert state.regime == "BULL_CALM"
        assert ctx._regime_evidence["source"] == "hurst_momentum_bull"


# ── Standalone path agreement ────────────────────────────────────────────────

class TestStandaloneAgrees:
    """kernel/regime.py::detect_regime must produce same regime as the
    task pipeline. PRIME DIRECTIVE: both paths must stay in sync."""

    def test_standalone_5day_bear_matches_task(self):
        returns = np.concatenate([
            np.array([0.001] * 30),  # need ≥30 for detect_regime guard
            np.array([-0.025, -0.025, -0.025, -0.025]),
        ])
        # Standalone path
        state_s = RegimeState()
        # Need SPY df with at least 200 bars for MA200
        closes = [100.0] * 200
        spy_df = _make_spy_df(closes)
        result = detect_regime(returns, spy_df, None, state_s, {})
        # Task path
        state_t = RegimeState()
        ctx = _ctx(returns, spy_df=spy_df, regime_state=state_t, config={})
        BEAROverrideTask().run(ctx)
        # Manually set hurst_regime for the task path (Hurst task not invoked here)
        from kernel.regime import compute_hurst
        h = compute_hurst(returns, window=63)
        state_t.hurst = h
        state_t.hurst_regime = (
            "MOMENTUM" if h > 0.65 else "REVERSION" if h < 0.52 else "AMBIGUOUS"
        )
        RegimeFinalizeTask().run(ctx)
        assert state_s.regime == state_t.regime, \
            f"standalone={state_s.regime} task={state_t.regime}"
        assert state_s.regime == "BEAR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
