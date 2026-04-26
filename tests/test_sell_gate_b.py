"""Tests for SellGateBTask (audit fix SELL-GATE-B, 2026-04-26 round-7).

User spec: "你的 portfolio manager不管卖吗？1，3，4都要修！"
Pre-fix the sell path had only per-ticker model + path rules. Buy
path has Gate A/B/C as a quality floor; sell side had no analog —
so a single-day model spike could exit a holding the panel/μ still
likes. SellGateBTask is the symmetric guard.

Behavioral contract:
  * Default OFF (flag preserves existing behaviour).
  * Path rules (stop_loss / trailing_stop / single_day_loss / max_hold)
    are EXEMPT — only model_sell can be blocked.
  * Block triggers on `μ/σ > -threshold` (mirror of buy gate's
    `μ/σ ≥ +threshold`).
  * On block, exit_signal is CLEARED (not just should_exit=False) so
    PanelConvictionExitTask still gets a chance to evaluate.
  * Streak is NOT touched — model can keep accumulating; on the next
    bar where μ/σ drops below the floor, the streak fires immediately.
  * Defensive on NaN / σ≤0 / missing fields → no block.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.exits import ExitSignal, HoldingState   # noqa: E402
from kernel.pipeline.context import TickerInferenceContext   # noqa: E402
from kernel.pipeline.task_sell import SellGateBTask  # noqa: E402
from kernel.pipeline.job_sell import TickerSellJob   # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _holding(*, mu: float | None, sigma: float | None,
              sell_streak: int = 3,
              shares: float = 10.0) -> HoldingState:
    today = datetime.date(2026, 4, 27)
    h = HoldingState(
        entry_price=100.0,
        entry_date=today - datetime.timedelta(days=60),
        high_watermark=110.0,
        sell_streak=sell_streak,
        last_streak_inc_date=today,
        shares=shares,
    )
    h.mu    = mu
    h.sigma = sigma
    return h


def _exit_signal(exit_type: str, should_exit: bool = True) -> ExitSignal:
    return ExitSignal(
        should_exit=should_exit,
        reason=f"{exit_type} test",
        exit_type=exit_type,
    )


def _ctx(*, holding: HoldingState | None,
          exit_signal: ExitSignal | None,
          gate_b_enabled: bool = True,
          threshold: float = 0.10) -> TickerInferenceContext:
    cfg = {
        "ranking": {
            "panel_scoring": {
                "sell_gate_b": {
                    "enabled":   gate_b_enabled,
                    "threshold": threshold,
                },
            },
        },
    }
    tc = TickerInferenceContext(
        ticker="AAPL",
        ohlcv={},
        model=None,
        config=cfg,
        today=datetime.date(2026, 4, 27),
        regime="BULL_CALM",
        regime_params={},
        exit_params={},
        holding=holding,
        price=100.0,
    )
    tc.exit_signal = exit_signal
    return tc


# ── Default OFF (flag-disabled = bit-for-bit no-op) ───────────────────────────

class TestDefaultOff:
    def test_disabled_does_not_clear_model_sell(self):
        tc = _ctx(
            holding=_holding(mu=0.05, sigma=0.05),  # would block if enabled
            exit_signal=_exit_signal("model_sell"),
            gate_b_enabled=False,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is not None
        assert tc.exit_signal.should_exit is True

    def test_missing_config_block_does_not_crash(self):
        """No ranking.panel_scoring.sell_gate_b key at all."""
        tc = TickerInferenceContext(
            ticker="X", ohlcv={}, model=None, config={},
            today=datetime.date(2026, 4, 27),
            regime="BULL_CALM", regime_params={}, exit_params={},
            holding=_holding(mu=0.05, sigma=0.05),
            price=100.0,
        )
        tc.exit_signal = _exit_signal("model_sell")
        SellGateBTask().run(tc)   # should be a no-op, not raise
        assert tc.exit_signal is not None


# ── Path rules are EXEMPT ─────────────────────────────────────────────────────

class TestPathRulesExempt:
    """Stop_loss / trailing / SDL / max_hold MUST always pass through.

    User contract: "Stop_loss/trailing/SDL preserved as path-dependent rules"
    """

    @pytest.mark.parametrize("exit_type", [
        "stop_loss", "trailing_stop", "single_day_loss", "max_hold",
        "panel_conviction", "rotation", "kelly_trim",
    ])
    def test_path_rule_not_blocked_even_with_positive_mu(self, exit_type):
        tc = _ctx(
            holding=_holding(mu=0.05, sigma=0.05),  # would block model_sell
            exit_signal=_exit_signal(exit_type),
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is not None
        assert tc.exit_signal.should_exit is True
        assert tc.exit_signal.exit_type == exit_type

    def test_no_exit_signal_does_nothing(self):
        tc = _ctx(
            holding=_holding(mu=0.05, sigma=0.05),
            exit_signal=None,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is None

    def test_should_not_exit_signal_does_nothing(self):
        """sig.should_exit=False (e.g. blocked_streak diagnostic) is left alone."""
        sig = _exit_signal("model_sell", should_exit=False)
        tc = _ctx(
            holding=_holding(mu=0.05, sigma=0.05),
            exit_signal=sig,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is sig
        assert tc.exit_signal.should_exit is False


# ── Block / pass behaviour ────────────────────────────────────────────────────

class TestBlockOrPass:
    def test_blocks_model_sell_when_edge_sharpe_above_negative_threshold(self):
        """μ=+0.005, σ=0.05 → edge_sharpe=+0.10 > -0.10 → BLOCK."""
        tc = _ctx(
            holding=_holding(mu=0.005, sigma=0.05),
            exit_signal=_exit_signal("model_sell"),
            threshold=0.10,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is None, "model_sell should be cleared"

    def test_blocks_when_mu_neutral(self):
        """μ=0.0 → edge_sharpe=0.0 > -0.10 → BLOCK."""
        tc = _ctx(
            holding=_holding(mu=0.0, sigma=0.05),
            exit_signal=_exit_signal("model_sell"),
            threshold=0.10,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is None

    def test_blocks_when_mu_slightly_negative_but_above_floor(self):
        """μ=-0.004, σ=0.05 → edge_sharpe=-0.08 > -0.10 → BLOCK."""
        tc = _ctx(
            holding=_holding(mu=-0.004, sigma=0.05),
            exit_signal=_exit_signal("model_sell"),
            threshold=0.10,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is None

    def test_passes_when_edge_sharpe_below_threshold(self):
        """μ=-0.01, σ=0.05 → edge_sharpe=-0.20 ≤ -0.10 → PASS."""
        sig = _exit_signal("model_sell")
        tc = _ctx(
            holding=_holding(mu=-0.01, sigma=0.05),
            exit_signal=sig,
            threshold=0.10,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is sig
        assert tc.exit_signal.should_exit is True

    def test_passes_at_exact_boundary(self):
        """μ=-0.10, σ=1.0 → edge_sharpe=-0.10 == -0.10 → PASS (>, not ≥).

        Use σ=1.0 so the division is fp-exact; -0.005/0.05 has rounding
        artifacts that break the boundary semantics.
        """
        sig = _exit_signal("model_sell")
        tc = _ctx(
            holding=_holding(mu=-0.10, sigma=1.0),
            exit_signal=sig,
            threshold=0.10,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is sig

    def test_threshold_zero_means_only_negative_mu_passes(self):
        """threshold=0 → require μ/σ ≤ 0 → any non-negative μ blocks."""
        tc = _ctx(
            holding=_holding(mu=0.001, sigma=0.05),
            exit_signal=_exit_signal("model_sell"),
            threshold=0.0,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is None


# ── Streak preservation ───────────────────────────────────────────────────────

class TestStreakPreservation:
    """User contract: streak NOT touched on block — once μ flips, the
    accumulated streak fires immediately on the next bar."""

    def test_blocked_model_sell_does_not_reset_streak(self):
        h = _holding(mu=0.05, sigma=0.05, sell_streak=3)
        tc = _ctx(holding=h, exit_signal=_exit_signal("model_sell"))
        SellGateBTask().run(tc)
        assert h.sell_streak == 3, "streak preserved across block"


# ── Defensive paths (no block on bad inputs) ──────────────────────────────────

class TestDefensive:
    def test_missing_mu_does_not_block(self):
        sig = _exit_signal("model_sell")
        tc = _ctx(
            holding=_holding(mu=None, sigma=0.05),
            exit_signal=sig,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is sig

    def test_missing_sigma_does_not_block(self):
        sig = _exit_signal("model_sell")
        tc = _ctx(
            holding=_holding(mu=0.05, sigma=None),
            exit_signal=sig,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is sig

    def test_zero_sigma_does_not_block(self):
        """σ=0 would be a /0 → defensive: don't block."""
        sig = _exit_signal("model_sell")
        tc = _ctx(
            holding=_holding(mu=0.05, sigma=0.0),
            exit_signal=sig,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is sig

    def test_negative_sigma_does_not_block(self):
        sig = _exit_signal("model_sell")
        tc = _ctx(
            holding=_holding(mu=0.05, sigma=-0.01),
            exit_signal=sig,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is sig

    def test_nan_mu_does_not_block(self):
        sig = _exit_signal("model_sell")
        tc = _ctx(
            holding=_holding(mu=float("nan"), sigma=0.05),
            exit_signal=sig,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is sig

    def test_nan_sigma_does_not_block(self):
        sig = _exit_signal("model_sell")
        tc = _ctx(
            holding=_holding(mu=0.05, sigma=float("nan")),
            exit_signal=sig,
        )
        SellGateBTask().run(tc)
        assert tc.exit_signal is sig

    def test_no_holding_does_not_crash(self):
        tc = _ctx(holding=None, exit_signal=_exit_signal("model_sell"))
        SellGateBTask().run(tc)   # no-op, no crash


# ── Wiring (TickerSellJob includes SellGateBTask) ─────────────────────────────

class TestWiring:
    def test_sellgateb_in_ticker_sell_job(self):
        """Both InferencePipeline and SellOnlyPipeline use TickerSellJob,
        so wiring it here covers both surfaces.
        """
        tasks = TickerSellJob().tasks
        names = [type(t).__name__ for t in tasks]
        assert "SellGateBTask" in names

    def test_sellgateb_after_evaluate_before_panel_conviction(self):
        """Order: PrepareHolding → ScoreModel → EvaluateExits → SellGateB →
        PanelConvictionExit. SellGateB MUST come after EvaluateExits (so
        exit_signal is populated) and BEFORE PanelConvictionExit (so
        clearing model_sell lets panel_conviction still consider firing).
        """
        names = [type(t).__name__ for t in TickerSellJob().tasks]
        i_eval = names.index("EvaluateExitsTask")
        i_gate = names.index("SellGateBTask")
        i_pc   = names.index("PanelConvictionExitTask")
        assert i_eval < i_gate < i_pc
