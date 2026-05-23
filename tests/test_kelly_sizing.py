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
         top_up_threshold=0.05, bear_only=False, skip_buys=False,
         cash=None):
    kelly_cfg: dict = {"enabled": kelly_on, "top_up_threshold": top_up_threshold}
    return SimpleNamespace(
        config          = {"ranking": {"kelly_sizing": kelly_cfg},
                            "regime_params": {
                                "BULL_CALM": {"max_position_pct": 0.15}
                            }},
        regime          = "BULL_CALM",
        confidence      = 1.0,
        portfolio_value = portfolio,
        # Cash defaults to portfolio value when not specified — TopUpHeldTask's
        # cash-cap guard (Bug 26) requires this to be > 0 for the order to fire.
        cash            = portfolio if cash is None else cash,
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

    def test_topup_uses_cash_after_pending_buy_orders(self):
        """QP/selection buys already queued this bar reserve cash before TopUp.

        cash=$10k, pending QP buy=$9.5k, desired top-up=$2k
        → TopUp may buy only 5 shares ($500), not the full Kelly delta.
        """
        ctx = _ctx(
            holdings={"AAA": _held(shares=10, kelly_target_pct=0.30,
                                   panel_score=0.50)},
            prices={"AAA": 100.0},
            portfolio=10000.0,
            cash=10000.0,
            orders=[{"ticker": "BBB", "shares": 95, "price": 100.0,
                     "invest": 9500.0, "order_type": "QP_BUY"}],
        )
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 2
        assert ctx.orders[-1]["ticker"] == "AAA"
        assert ctx.orders[-1]["shares"] == 5
        assert ctx.orders[-1]["invest"] == 500.0
        assert ctx.orders[-1]["decision_inputs"]["pending_buy_cash"] == 9500.0
        assert ctx.orders[-1]["decision_inputs"]["available_cash_before"] == 500.0

    def test_topup_skips_when_pending_buys_exhaust_cash(self):
        ctx = _ctx(
            holdings={"AAA": _held(shares=10, kelly_target_pct=0.30,
                                   panel_score=0.50)},
            prices={"AAA": 100.0},
            portfolio=10000.0,
            cash=10000.0,
            orders=[{"ticker": "BBB", "shares": 100, "price": 100.0,
                     "invest": 10000.0, "order_type": "QP_BUY"}],
        )
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1
        assert ctx.orders[0]["ticker"] == "BBB"

    def test_topup_decrements_cash_budget_across_multiple_holdings(self):
        ctx = _ctx(
            holdings={
                "AAA": _held(shares=0, kelly_target_pct=0.10,
                             panel_score=0.50),
                "BBB": _held(shares=0, kelly_target_pct=0.10,
                             panel_score=0.50),
            },
            prices={"AAA": 100.0, "BBB": 100.0},
            portfolio=10000.0,
            cash=1000.0,
        )
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1
        assert ctx.orders[0]["ticker"] == "AAA"
        assert ctx.orders[0]["invest"] == 1000.0

    def test_topup_respects_regime_cash_reserve(self):
        ctx = _ctx(
            holdings={"AAA": _held(shares=0, kelly_target_pct=0.60,
                                   panel_score=0.50)},
            prices={"AAA": 100.0},
            portfolio=10000.0,
            cash=10000.0,
        )
        ctx.config["regime_params"]["BULL_CALM"]["cash_reserve_pct"] = 0.50
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1
        assert ctx.orders[0]["shares"] == 50
        assert ctx.orders[0]["invest"] == 5000.0
        assert ctx.orders[0]["decision_inputs"]["reserve_cash"] == 5000.0


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
        """Kelly target is written onto every candidate AND every holding.

        Pre-2026-05-04: pinned the literal `_kelly(cand)` helper call.
        Post-2026-05-04: the helper was renamed `_kelly_with_reason`
        because Kelly now returns (target, skip_reason) so the Task can
        write a per-ticker reason into ctx._blocked_by_ticker for the
        decision-tree DB. The semantic invariant is unchanged: every
        cand and every holding gets a kelly_target_pct assignment.
        """
        src = (_STRATEGY_DIR / "kernel" / "panel_pipeline" / "job_panel_scoring.py").read_text()
        assert "cand.kelly_target_pct = target" in src
        assert "hs.kelly_target_pct = target"   in src
        # And the helper that produces (target, reason) exists exactly
        # once in this Task body.
        assert src.count("_kelly_with_reason") >= 3   # def + 2 callsites
        # Decision-tree DB plumbing: skip-reason MUST be written into
        # ctx._blocked_by_ticker so record_candidate_scores persists it.
        assert "ctx._blocked_by_ticker = blocked" in src
        assert "kelly_zero:" in src   # at least one prefix in the source

    def test_uses_kernel_kelly_helper(self):
        src = (_STRATEGY_DIR / "kernel" / "panel_pipeline" / "job_panel_scoring.py").read_text()
        assert "from kernel.kelly import kelly_target_pct" in src


# ── Skip-reason instrumentation (2026-05-04 user mandate: explainable funnel) ──
#
# When candidates flow through ApplyKellySizingTask, every cand whose
# kelly_target lands at zero MUST be tagged with one of:
#
#   kelly_zero:mu_none          — NGBoost left mu=None (skipped at predict)
#   kelly_zero:mu_nonfinite     — NaN / inf
#   kelly_zero:sigma_none       — sigma not populated
#   kelly_zero:sigma_nonfinite  — NaN / inf
#   kelly_zero:sigma_nonpos     — sigma ≤ 0
#   kelly_zero:mu_le_min_edge   — μ ≤ configured min_edge
#   kelly_zero:capped_zero      — formula returned 0 after caps
#
# That tag goes into ctx._blocked_by_ticker so record_candidate_scores
# persists it to the candidate_scores.blocked_by column. Without that,
# "Kelly is returning 0 for all 50 candidates" is opaque on a SQL query.

class TestKellySkipReasonsInstrumentation:
    """Behavioral tests for the per-ticker kelly_zero:* skip reasons
    written into ctx._blocked_by_ticker by ApplyKellySizingTask.
    """

    def _make_ctx(self, candidates, holdings=(), kelly_min_edge=0.0):
        cfg = {
            "ranking": {"kelly_sizing": {
                "enabled": True,
                "fractional": 0.5,
                "max_concentration": 0.35,
                "min_edge": kelly_min_edge,
            }},
            "regime_params": {"BULL_CALM": {"max_position_pct": 0.20}},
        }
        return SimpleNamespace(
            candidates=list(candidates),
            holdings={h.ticker: h for h in holdings},
            confidence=1.0,
            regime="BULL_CALM",
            config=cfg,
            counters={},
        )

    def _cand(self, ticker, *, mu=None, sigma=None):
        return SimpleNamespace(
            ticker=ticker, mu=mu, sigma=sigma,
            kelly_target_pct=None,
        )

    def test_mu_none_tagged(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyKellySizingTask
        ctx = self._make_ctx([self._cand("AAPL", mu=None, sigma=0.05)])
        ApplyKellySizingTask().run(ctx)
        assert ctx._blocked_by_ticker["AAPL"] == "kelly_zero:mu_none"
        assert ctx.candidates[0].kelly_target_pct == 0.0

    def test_sigma_none_tagged(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyKellySizingTask
        ctx = self._make_ctx([self._cand("MSFT", mu=0.02, sigma=None)])
        ApplyKellySizingTask().run(ctx)
        assert ctx._blocked_by_ticker["MSFT"] == "kelly_zero:sigma_none"

    def test_sigma_nonpos_tagged(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyKellySizingTask
        ctx = self._make_ctx([self._cand("NVDA", mu=0.02, sigma=0.0)])
        ApplyKellySizingTask().run(ctx)
        assert ctx._blocked_by_ticker["NVDA"] == "kelly_zero:sigma_nonpos"

    def test_mu_nonfinite_tagged(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyKellySizingTask
        ctx = self._make_ctx([self._cand("AMD", mu=float("nan"), sigma=0.05)])
        ApplyKellySizingTask().run(ctx)
        assert ctx._blocked_by_ticker["AMD"] == "kelly_zero:mu_nonfinite"

    def test_mu_le_min_edge_tagged(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyKellySizingTask
        ctx = self._make_ctx(
            [self._cand("ZM", mu=0.001, sigma=0.05)],
            kelly_min_edge=0.005,   # 50 bps floor
        )
        ApplyKellySizingTask().run(ctx)
        assert ctx._blocked_by_ticker["ZM"] == "kelly_zero:mu_le_min_edge"

    def test_positive_kelly_does_not_tag(self):
        """When kelly is healthy, no entry in _blocked_by_ticker for that ticker."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyKellySizingTask
        # μ=0.005, σ=0.05 → f* = 0.005/0.0025 = 2.0 → fractional 0.5 × 2.0 = 1.0
        # → capped by max_pct (0.20) → kelly = 0.20
        ctx = self._make_ctx([self._cand("META", mu=0.005, sigma=0.05)])
        ApplyKellySizingTask().run(ctx)
        assert "META" not in ctx._blocked_by_ticker
        assert ctx.candidates[0].kelly_target_pct == pytest.approx(0.20)

    def test_does_not_clobber_upstream_block(self):
        """If an UPSTREAM Task (e.g. NGBoost) already wrote a reason for
        this ticker, Kelly's setdefault must not overwrite it."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyKellySizingTask
        ctx = self._make_ctx([self._cand("RBLX", mu=None, sigma=None)])
        ctx._blocked_by_ticker = {"RBLX": "ngb_skipped:not_in_predict_index"}
        ApplyKellySizingTask().run(ctx)
        # The NGB reason should still be there, NOT the kelly reason
        assert ctx._blocked_by_ticker["RBLX"] == "ngb_skipped:not_in_predict_index"


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
        # The Phase-3 jobs (PanelScoring → Ranking → Rotation → Selection)
        # are dispatched from a tuple loop in pp_inference.py (2026-04-24
        # rewrite to honour Job.should_skip). The check now confirms that
        # SelectionJob is registered in that tuple AND TopUpHeldTask runs
        # after the loop closes.
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "pp_inference.py").read_text()
        i_sel = src.find("SelectionJob()")
        i_top = src.find("TopUpHeldTask", i_sel)
        assert i_sel > 0
        assert i_top > i_sel
        assert "TopUpHeldTask().run(ctx)" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
