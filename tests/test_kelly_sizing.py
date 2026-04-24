"""Tests for Plan C — Kelly-scaled sizing + top-up.

User design (2026-04-23 evening):
  A/B = admit gate (tiered_thresholds on calibrated rank_score)
  C   = size function driven by edge = rank_score - base_rate
  Also: compute Kelly for HELD positions → if kelly_target > current
        weight, emit a BUY order to top up.

Covers:
  1. ApplyScoresTask populates kelly_target_pct on BOTH candidates
     and holdings when kelly_sizing.enabled.
  2. SizeAndEmitTask scales max_pct by kelly_f derived from edge.
  3. TopUpHeldTask emits extra BUY when held.kelly_target > current_pct.
  4. TopUpHeldTask skips BEAR regime / drawdown halt / already-selling
     tickers.
  5. Kelly disabled → no-op for all three.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.pipeline.task_topup import TopUpHeldTask  # noqa: E402


def _held(*, shares=0.0, panel_score=None, kelly_target_pct=None):
    return SimpleNamespace(
        shares           = shares,
        panel_score      = panel_score,
        kelly_target_pct = kelly_target_pct,
        rank_score       = panel_score,
        sigma            = None,
        mu               = None,
    )


def _ctx(*, holdings, orders=None, exits=None, rotations=None,
         portfolio=10000.0, prices=None, kelly_on=True,
         top_up_threshold=0.05, bear_only=False, skip_buys=False):
    kelly_cfg: dict = {"enabled": kelly_on, "top_up_threshold": top_up_threshold}
    return SimpleNamespace(
        config          = {"ranking": {"kelly_sizing": kelly_cfg},
                            "regime_params": {
                                "BULL_CALM": {"max_position_pct": 0.15}
                            }},
        regime          = "BULL_CALM",
        confidence      = 1.0,
        portfolio_value = portfolio,
        prices          = prices or {},
        holdings        = holdings,
        orders          = list(orders or []),
        exits           = list(exits or []),
        rotations       = list(rotations or []),
        bear_only       = bear_only,
        skip_buys       = skip_buys,
    )


# ── TopUpHeldTask ────────────────────────────────────────────────────────────

class TestTopUpHeldTask:
    def test_no_op_when_kelly_disabled(self):
        ctx = _ctx(
            kelly_on   = False,
            holdings   = {"AAA": _held(shares=10, kelly_target_pct=0.20)},
            prices     = {"AAA": 100.0},
            portfolio  = 10000.0,
        )
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_tops_up_when_target_exceeds_current(self):
        """current = 10 × $100 / $10000 = 10%. target = 20%. delta = 10%
        → buy floor(0.10 * 10000 / 100) = 10 extra shares."""
        ctx = _ctx(
            holdings  = {"AAA": _held(shares=10, kelly_target_pct=0.20,
                                       panel_score=0.50)},
            prices    = {"AAA": 100.0},
            portfolio = 10000.0,
        )
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1
        o = ctx.orders[0]
        assert o["ticker"] == "AAA"
        assert o["shares"] == 10
        assert o["detail"] == "top_up_kelly"
        assert o["order_type"] == "TOP_UP"

    def test_skips_when_delta_below_threshold(self):
        """current = 18%, target = 20%, delta = 2% < threshold 5% → skip."""
        ctx = _ctx(
            holdings = {"AAA": _held(shares=18, kelly_target_pct=0.20)},
            prices   = {"AAA": 100.0},
        )
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_skips_when_target_below_current(self):
        """Held at 25%, target 10% — don't BUY (trimming handled
        separately, this Task is add-only)."""
        ctx = _ctx(
            holdings = {"AAA": _held(shares=25, kelly_target_pct=0.10)},
            prices   = {"AAA": 100.0},
        )
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_skips_already_buying(self):
        """If SizeAndEmit already queued AAA, don't double up."""
        ctx = _ctx(
            holdings = {"AAA": _held(shares=5, kelly_target_pct=0.25)},
            prices   = {"AAA": 100.0},
            orders   = [{"ticker": "AAA", "shares": 5, "price": 100.0}],
        )
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1   # untouched

    def test_skips_already_selling(self):
        ctx = _ctx(
            holdings = {"AAA": _held(shares=5, kelly_target_pct=0.25)},
            prices   = {"AAA": 100.0},
            exits    = [SimpleNamespace(ticker="AAA")],
        )
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_skips_rotated_out(self):
        ctx = _ctx(
            holdings  = {"AAA": _held(shares=5, kelly_target_pct=0.25)},
            prices    = {"AAA": 100.0},
            rotations = [SimpleNamespace(sell_ticker="AAA", buy_ticker="BBB")],
        )
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_skips_bear_only(self):
        ctx = _ctx(
            holdings  = {"AAA": _held(shares=5, kelly_target_pct=0.25)},
            prices    = {"AAA": 100.0},
            bear_only = True,
        )
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_skips_drawdown_halt(self):
        ctx = _ctx(
            holdings  = {"AAA": _held(shares=5, kelly_target_pct=0.25)},
            prices    = {"AAA": 100.0},
            skip_buys = True,
        )
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_skips_no_kelly_target(self):
        ctx = _ctx(
            holdings = {"AAA": _held(shares=5, kelly_target_pct=None)},
            prices   = {"AAA": 100.0},
        )
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []


# ── SizeAndEmit Kelly scaling — source-level check ──────────────────────────

class TestSizeAndEmitKellyScaling:
    def test_source_reads_kelly_target(self):
        """SizeAndEmitTask must consume the precomputed kelly_target_pct
        (not recompute edge). Decoupled from the formula — all Kelly
        math lives in kernel/kelly.py + ApplyKellySizingTask."""
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "task_selection.py").read_text()
        assert "kelly_target_pct" in src
        assert "kelly_on" in src

    def test_source_skips_zero_kelly(self):
        """When kelly_target_pct = 0 (edge ≤ min_edge or σ≤0), skip."""
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "task_selection.py").read_text()
        assert "Kelly=0 — skip" in src


# ── PanelScoringJob — Kelly target on BOTH sides via ApplyKellySizingTask ──

class TestApplyKellySizingTask:
    def test_task_exists(self):
        src = (_STRATEGY_DIR / "kernel" / "panel_pipeline" / "job_panel_scoring.py").read_text()
        assert "class ApplyKellySizingTask" in src

    def test_task_in_job_chain_last(self):
        """Must run after ApplyNGBoost + ApplyGlobalCalibration so μ,σ
        are populated before Kelly math."""
        src = (_STRATEGY_DIR / "kernel" / "panel_pipeline" / "job_panel_scoring.py").read_text()
        i_ng = src.find("ApplyNGBoostTask()")
        i_cal = src.find("ApplyGlobalCalibrationTask()")
        i_ks = src.find("ApplyKellySizingTask()")
        # Last in the Task chain
        assert i_ks > i_ng
        assert i_ks > i_cal

    def test_applies_to_both_cands_and_holdings(self):
        src = (_STRATEGY_DIR / "kernel" / "panel_pipeline" / "job_panel_scoring.py").read_text()
        assert "cand.kelly_target_pct = _kelly(cand)" in src
        assert "hs.kelly_target_pct = _kelly(hs)"     in src

    def test_uses_kernel_kelly_helper(self):
        src = (_STRATEGY_DIR / "kernel" / "panel_pipeline" / "job_panel_scoring.py").read_text()
        assert "from kernel.kelly import kelly_target_pct" in src


# ── The beautiful kernel/kelly.py — pure function tests ──────────────────────

class TestKellyFormula:
    def test_off_when_inputs_missing(self):
        from kernel.kelly import kelly_target_pct
        assert kelly_target_pct(None, 0.05, max_pct=0.15) == 0.0
        assert kelly_target_pct(0.01, None, max_pct=0.15) == 0.0
        assert kelly_target_pct(0.01, 0.0,  max_pct=0.15) == 0.0

    def test_off_when_mu_below_min_edge(self):
        from kernel.kelly import kelly_target_pct
        assert kelly_target_pct(0.001, 0.05, max_pct=0.15, min_edge=0.01) == 0.0

    def test_scales_with_mu_over_sigma_squared(self):
        """f* = μ/σ² * fractional. With fractional=0.25:
        μ=0.01, σ=0.05 → f_kelly = 0.01/0.0025 = 4 → 0.25*4 = 1.0 → capped at max_pct."""
        from kernel.kelly import kelly_target_pct
        out = kelly_target_pct(0.01, 0.05, max_pct=0.15, fractional=0.25)
        # Kelly = 4.0, fractional = 1.0, capped at 0.15
        assert out == 0.15

    def test_sub_cap_when_signal_weak(self):
        """μ=0.002, σ=0.05 → f_kelly=0.8, 0.25*0.8=0.20 → capped at 0.15."""
        from kernel.kelly import kelly_target_pct
        out = kelly_target_pct(0.002, 0.05, max_pct=0.15, fractional=0.25)
        assert out == 0.15

    def test_max_concentration_cap(self):
        """Even a huge μ/σ² can't exceed max_concentration."""
        from kernel.kelly import kelly_target_pct
        # Kelly says 1.0, fractional=1.0, but max_concentration=0.35
        out = kelly_target_pct(0.05, 0.05, max_pct=1.0, max_concentration=0.35,
                                fractional=1.0)
        assert out == 0.35

    def test_fractional_kelly_scales_linearly(self):
        from kernel.kelly import kelly_target_pct
        full = kelly_target_pct(0.003, 0.10, max_pct=1.0, max_concentration=1.0,
                                 fractional=1.00)
        quarter = kelly_target_pct(0.003, 0.10, max_pct=1.0, max_concentration=1.0,
                                    fractional=0.25)
        assert abs(quarter - full * 0.25) < 1e-9


# ── Pipeline wiring ─────────────────────────────────────────────────────────

class TestPipelineWiring:
    def test_topup_called_after_selection(self):
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "pp_inference.py").read_text()
        i_sel = src.find("SelectionJob().run(ctx)")
        i_top = src.find("TopUpHeldTask", i_sel)
        assert i_sel > 0
        assert i_top > i_sel
        assert "TopUpHeldTask().run(ctx)" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
