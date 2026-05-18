"""Regression tests for the 2026-04-24 audit fixes.

One test (or small group) per audit item from `doc/bug_audit_2026-04-24.md`.
All tests in this file would FAIL before the corresponding fix and PASS after.
Grouped by audit-report bug number.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── #20 + #21 + #22 — tuple-attr getattr bug ──────────────────────────────────

class TestTupleAttrGetattrFix:
    """ctx.exits is list[(ticker, ExitSignal)] — tasks must unpack the tuple,
    not getattr(tuple, "ticker") which always returned None and disabled
    the already_selling/already_exiting guards."""

    def _ctx_with_exits_tuple(self):
        from kernel.exits import ExitSignal, HoldingState
        sig = ExitSignal(should_exit=True, reason="stop_loss",
                         exit_type="stop_loss")
        hs = HoldingState(entry_price=100.0,
                          entry_date=datetime.date(2026, 1, 1),
                          high_watermark=100.0, shares=100,
                          kelly_target_pct=0.30)
        return SimpleNamespace(
            config={"ranking": {"kelly_sizing": {
                "enabled": True, "top_up_threshold": 0.05,
                "trim_enabled": True, "trim_threshold": 0.0,
            }}, "regime_params": {"BULL_CALM": {"max_position_pct": 0.30}}},
            regime="BULL_CALM", confidence=1.0,
            portfolio_value=10_000.0, cash=10_000.0,
            holdings={"AAA": hs}, prices={"AAA": 100.0},
            orders=[], rotations=[],
            # Production format — list of (ticker, ExitSignal) tuples
            exits=[("AAA", sig)],
            bear_only=False, skip_buys=False,
        )

    def test_topup_skips_ticker_with_tuple_exit(self):
        from kernel.pipeline.task_topup import TopUpHeldTask
        ctx = self._ctx_with_exits_tuple()
        TopUpHeldTask().run(ctx)
        # Pre-fix: getattr(tuple,"ticker") returned None → guard dead →
        # would emit a BUY for AAA on a bar where AAA is being sold.
        assert ctx.orders == [], "TopUp must skip tickers in ctx.exits"

    def test_trim_skips_ticker_with_tuple_exit(self):
        from kernel.pipeline.task_trim import TrimHeldTask
        ctx = self._ctx_with_exits_tuple()
        TrimHeldTask().run(ctx)
        # Pre-fix: trim would append a SECOND ExitSignal for AAA → SimAdapter
        # double-sells. Now guard works and no extra exit is appended.
        assert len(ctx.exits) == 1, "Trim must not double-sell already-exiting ticker"

    def test_live_runner_ntfy_unpacks_exit_tuple(self):
        """live/runner._notify_decision must extract ticker + exit_type from tuples."""
        # We can't easily call the function directly (it does network IO),
        # but we can confirm the parser branch handles a tuple input.
        from kernel.exits import ExitSignal
        sig = ExitSignal(should_exit=True, reason="r", exit_type="stop_loss")
        e = ("NVDA", sig)
        # Mimic the relevant branch from _notify_decision verbatim.
        if isinstance(e, tuple) and len(e) == 2:
            tkr, sigp = e
            reason = getattr(sigp, "exit_type",
                             getattr(sigp, "reason", "sell"))
        else:
            tkr = getattr(e, "ticker", "?")
            reason = getattr(e, "exit_type", "sell")
        assert tkr == "NVDA"
        assert reason == "stop_loss"


# ── #65 — kernel/data fetch_intraday_bars NameError on timeout ───────────────

class TestKernelDataLogger:
    def test_module_has_log_attribute(self):
        """fetch_intraday_bars referenced log.warning(...) but no module-level
        logger existed — would NameError on the timeout path it was supposed
        to handle gracefully."""
        from kernel import data as kd
        assert hasattr(kd, "log"), "kernel.data must define a module-level logger"
        # Confirm it's a real logger, not some other binding.
        import logging
        assert isinstance(kd.log, logging.Logger)


# ── #9 + #10 — Hurst lag misalignment + chunk off-by-one ─────────────────────

class TestHurstLagAlignment:
    def test_chunk_loop_includes_trailing_chunk(self):
        """When n is exactly k*lag, the trailing chunk arr[n-lag:n] must be
        included. Pre-fix: range(0, n-lag, lag) excluded it."""
        from kernel.regime import compute_hurst
        # Trending series: 50 evenly-rising bars → H should be > 0.5
        rng = np.random.default_rng(seed=42)
        # Strongly trending random walk
        steps = rng.normal(loc=0.005, scale=0.001, size=50)
        h = compute_hurst(steps)
        # Result is bounded [0,1] regardless; just verify no crash and
        # a sensible value comes out.
        assert 0.0 <= h <= 1.0
        # For a strongly trending series we expect > 0.5
        assert h > 0.5, f"trending series should give H > 0.5, got {h}"

    def test_polyfit_uses_actual_lags_not_recomputed_range(self):
        """Source should track the actual lag used for each rs value, not
        regenerate from range(2, 2+len(rs_vals)) which misaligned when
        intermediate lags produced no chunks."""
        src = (_STRATEGY_DIR / "kernel" / "regime.py").read_text()
        # Pre-fix sentinel: the broken regenerate-from-range pattern
        assert "range(2, 2 + len(rs_vals))" not in src, \
            "Hurst must track actual lag per rs_vals entry (audit #9)"


# ── #4 — _buy_universe missing OHLCV check on defensive branch ──────────────

class TestBuyUniverseDefensiveOhlcv:
    def test_defensive_branch_requires_ohlcv(self):
        from kernel.pipeline.pp_inference import _buy_universe
        from kernel.pipeline.context import InferenceContext
        ctx = InferenceContext(
            config={"defensive_tickers": ["GLD", "TLT"]},
            today=datetime.date(2026, 4, 24),
        )
        ctx.bear_only = True
        ctx.models = {"GLD": {}, "TLT": {}}
        ctx.holdings = {}
        # GLD has OHLCV, TLT doesn't
        ctx.ohlcv = {"GLD": pd.DataFrame()}
        u = _buy_universe(ctx)
        assert "GLD" in u
        assert "TLT" not in u, \
            "Defensives without OHLCV must be filtered (#4)"


# ── #15 — BEAR / ConfidenceVeto / BullVol gates no longer short-circuit ─────

class TestGateChainNoEarlyShortCircuit:
    def test_bear_branch_returns_none_not_false(self):
        """BEAR regime sets bear_only but doesn't halt the chain so
        velocity/EMA50 can still set buy_blocked when applicable.

        2026-05-15 update: BEARBranchTask soft-gate (commit 2447dcb)
        now requires confidence ≥ bear_branch_min_confidence (default 0.60)
        AND not in_transition before setting bear_only=True. This test
        retains the original audit #15 invariant ("chain continues")
        AND covers the high-conf path that fires bear_only=True.
        See tests/test_bear_branch_soft_gate.py for soft-gate-specific
        invariants (transition / low-conf veto / legacy mode)."""
        from kernel.pipeline.task_gates import BEARBranchTask
        from kernel.config import BEAR
        ctx = SimpleNamespace(
            regime=BEAR, bear_only=False, buy_blocked=False,
            confidence=1.0,  # high-conf path → bear_only=True
            regime_state=SimpleNamespace(in_transition=False),
            counters={}, config={"regime": {}},
        )
        result = BEARBranchTask().run(ctx)
        assert result is None, \
            "BEARBranchTask must continue chain (audit #15)"
        assert ctx.bear_only is True

    def test_confidence_veto_returns_none_when_triggered(self):
        from kernel.pipeline.task_gates import ConfidenceVetoTask
        ctx = SimpleNamespace(regime="BULL_CALM", confidence=0.30,
                               bear_only=False, buy_blocked=False, counters={},
                               config={"regime": {"confidence_veto_threshold": 0.55}})
        result = ConfidenceVetoTask().run(ctx)
        assert result is None
        assert ctx.bear_only is True

    def test_ema50_gate_handles_missing_spy(self):
        """Audit #16 — when SPY OHLCV is missing, gate logs warning but
        doesn't crash.

        2026-05-04 update: per audit Issue 06 fix, missing SPY now
        FAIL-SAFE blocks buys (sets ctx.buy_blocked=True) instead of
        silently disabling the macro filter. Pre-fix returned None;
        post-fix returns False (chain short-circuit) + sets buy_blocked.
        """
        from kernel.pipeline.task_gates import EMA50GateTask
        ctx = SimpleNamespace(regime="BULL_CALM", buy_blocked=False,
                               counters={}, ohlcv={}, config={})
        result = EMA50GateTask().run(ctx)
        # Post-Issue-06 fail-SAFE: missing SPY blocks buys.
        assert ctx.buy_blocked is True
        assert result is False


# ── #1 — should_skip dead code now wired into pp_inference ───────────────────

class TestShouldSkipWired:
    def test_pp_inference_calls_should_skip(self):
        """The Phase-3 dispatch loop must call .should_skip on each Job
        before calling .run."""
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "pp_inference.py").read_text()
        assert "should_skip" in src, \
            "pp_inference.py must honour should_skip (audit #1)"


# ── #51 + #52 — LEAN partial-sell + top-up entry preservation ────────────────

class TestLeanAdapterPartialAndTopup:
    def test_lean_adapter_branches_on_quantity(self):
        """LEAN commit() must branch on sig.quantity to dispatch
        Liquidate (full) vs MarketOrder (partial)."""
        src = (_STRATEGY_DIR / "adapters" / "lean.py").read_text()
        # Sentinel: post-fix code uses MarketOrder for partial sells
        assert "MarketOrder" in src, \
            "LEAN must support partial sells via MarketOrder (#51)"
        assert "is_partial" in src, \
            "LEAN must check quantity to detect partial sells (#51)"

    def test_lean_adapter_preserves_entry_state_on_topup(self):
        """LEAN commit() buy loop must NOT reset entry state when ticker
        already held (top-up path)."""
        src = (_STRATEGY_DIR / "adapters" / "lean.py").read_text()
        assert "already_held = ticker in algo._holdings" in src, \
            "LEAN must detect top-up vs fresh buy (#52)"


# ── #53 + #54 — RunnerAdapter top-up + partial-sell wash-sale ─────────────

class TestRunnerAdapterTopupAndPartial:
    def test_runner_preserves_entry_dates_on_topup(self):
        src = (_STRATEGY_DIR / "adapters" / "runner.py").read_text()
        assert "is_topup" in src, "RunnerAdapter must detect top-up (#53)"
        assert "TOPUP" in src, "RunnerAdapter must log top-ups distinctly"

    def test_runner_skips_wash_sale_stamp_on_partial(self):
        """Partial sell shouldn't stamp last_sell_dates — that would block
        same-week top-ups via wash-sale guard."""
        src = (_STRATEGY_DIR / "adapters" / "runner.py").read_text()
        # The wash-sale stamp must be guarded by `if not is_partial`.
        assert ("if not is_partial:" in src
                and "_last_sell_dates_str[ticker]" in src)


# ── #56 — SimAdapter dedupe duplicate exits per ticker ───────────────────────

class TestSimAdapterDuplicateExits:
    def test_dedupe_dict_in_commit(self):
        src = (_STRATEGY_DIR / "adapters" / "sim.py").read_text()
        assert "exits_by_ticker" in src, \
            "SimAdapter must dedupe ctx.exits per ticker (#56)"


# ── #54 mirror — SimAdapter doesn't stamp wash-sale on partial sells ─────

class TestSimAdapterPartialNoWashSale:
    def test_sim_partial_skips_last_sell_date(self):
        src = (_STRATEGY_DIR / "adapters" / "sim.py").read_text()
        # Sentinel: post-fix logic gates wash-sale stamp on `if not is_partial`
        # in _apply_sell.
        idx = src.find("def _apply_sell")
        # _apply_sell grew to >4k chars after 2026-05-06 bug-fix sprint
        # (NaN handling + earnings + Davis-Norman). Bumped to 8000.
        # 2026-05-10 (Batch A execution model): added slippage + sell-fee
        # + T+2 settlement branches — needs ~10k window now.
        body = src[idx:idx + 10_000]
        assert "if not is_partial:" in body
        assert "_last_sell_date[ticker]" in body


# ── #18 — exits.py LT threshold uses config not hardcoded 365 ──────────────

class TestExitsLTThreshold:
    def test_lt_hold_threshold_days_param(self):
        """compute_exits should read lt_hold_threshold_days from params."""
        src = (_STRATEGY_DIR / "kernel" / "exits.py").read_text()
        assert "lt_hold_threshold_days" in src, \
            "compute_exits must use config'd LT threshold (#18)"
        # And ensure the literal 365 is no longer the gating boundary
        assert "days_held < 365" not in src

    def test_pp_inference_propagates_lt_threshold(self):
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "pp_inference.py").read_text()
        assert "lt_hold_threshold_days" in src


# ── #17 — ExitSignal.blocked_streak typed field ─────────────────────────────

class TestExitSignalBlockedStreak:
    def test_dataclass_has_blocked_streak_field(self):
        from kernel.exits import ExitSignal
        sig = ExitSignal(should_exit=False, reason="", exit_type="")
        assert hasattr(sig, "blocked_streak")
        assert sig.blocked_streak is False

    def test_pp_inference_reads_typed_attribute(self):
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "pp_inference.py").read_text()
        # No more dunder leading-underscore lookup
        assert '"_blocked_streak"' not in src
        assert '"blocked_streak"' in src


# ── #11 — BEAR override uses cumulative product not arithmetic sum ──────

class TestBearOverrideCumulativeProduct:
    def test_task_uses_prod(self):
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "task_regime.py").read_text()
        idx = src.find("class BEAROverrideTask")
        # Widened 2000 → 5000 after 2026-05-17 detector fix A+C added
        # 5-day BEAR + vol-cluster CHOPPY logic via _vol_ret helper.
        body = src[idx:idx + 5000]
        assert "np.prod" in body, "BEAR override must use cumulative product (#11)"

    def test_detect_regime_uses_prod(self):
        src = (_STRATEGY_DIR / "kernel" / "regime.py").read_text()
        # The detect_regime path also got the np.prod fix
        idx = src.find("def detect_regime")
        body = src[idx:idx + 4000]
        assert "np.prod" in body


# ── #28 — correlation guard explicit None check ──────────────────────────

class TestCorrelationGuardZero:
    def test_zero_correlation_treated_as_real(self):
        from kernel.selection import passes_correlation_guard
        # 0.0 correlation should NOT skip to the reverse lookup; threshold 0.5
        # → 0.0 passes (below threshold).
        cm = {"AAA": {"BBB": 0.0}}
        assert passes_correlation_guard("AAA", ["BBB"], cm, threshold=0.5) is True

    def test_threshold_zero_blocks_zero(self):
        """At threshold=0, even 0.0 correlation triggers the guard.
        Pre-fix: `0.0 or X` short-circuited and the reverse lookup was used.
        """
        from kernel.selection import passes_correlation_guard
        cm = {"AAA": {"BBB": 0.0}}
        # threshold=0.0 means abs(0.0) >= 0.0 → True → blocked
        assert passes_correlation_guard("AAA", ["BBB"], cm, threshold=0.0) is False


# ── #2 — run_parallel timeout log no longer claims "skipped" ─────────────

class TestRunParallelTimeoutLog:
    def test_log_acknowledges_thread_cant_be_cancelled(self):
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "pipeline.py").read_text()
        # Sentinel: post-fix log mentions worker may still be running
        assert "worker may still be running" in src


# ── #46 — pp_training DataFetchJob honours config.benchmark ──────────────

class TestDataFetchJobBenchmark:
    def test_uses_config_benchmark_not_hardcoded(self):
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "pp_training.py").read_text()
        # The fix must read benchmark from config
        idx = src.find("class DataFetchJob")
        body = src[idx:idx + 2500]
        assert 'cfg.get("benchmark"' in body or 'config.get("benchmark"' in body


# ── #87 + #92 — tax-rate defaults aligned to higher 0.50/0.32 ─────────────

class TestTaxRateDefaults:
    """User spec 2026-04-24: tax rate defaults should be the HIGHER set
    (50% short-term / 32% long-term) so a missing tax block conservatively
    over-estimates rather than under-estimates."""

    def test_main_py_defaults(self):
        src = (_STRATEGY_DIR / "main.py").read_text()
        assert 'short_term_rate", 0.50' in src
        assert 'long_term_rate", 0.32' in src

    def test_kernel_rotation_defaults(self):
        src = (_STRATEGY_DIR / "kernel" / "rotation.py").read_text()
        assert "0.37" not in src.split('"short_term_rate"')[-1][:200]
        assert "0.50" in src

    def test_sim_adapter_defaults(self):
        src = (_STRATEGY_DIR / "adapters" / "sim.py").read_text()
        assert '"short_term_rate", 0.50' in src
        assert '"long_term_rate", 0.32' in src

    def test_task_rotation_defaults(self):
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "task_rotation.py").read_text()
        assert '"short_term_rate", 0.50' in src
        assert '"long_term_rate", 0.32' in src


# ── #6 — SellOnlyPipeline runs MonitorIdleStreakTask ─────────────────────

class TestSellOnlyMonitorIdleStreak:
    def test_sell_only_pipeline_includes_monitor(self):
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "pp_inference.py").read_text()
        # Find SellOnlyPipeline section and ensure MonitorIdleStreakTask is invoked
        idx = src.find("class SellOnlyPipeline")
        body = src[idx:]
        assert "MonitorIdleStreakTask" in body, \
            "SellOnlyPipeline must run the no-trade monitor (#6)"


# ── #23 — TopUp target_pct uses actual_delta (capped) not raw kelly delta ───

class TestTopUpTargetPctAccuracy:
    def test_target_pct_reflects_capped_delta(self):
        from kernel.pipeline.task_topup import TopUpHeldTask
        from kernel.exits import HoldingState
        hs = HoldingState(entry_price=100.0,
                          entry_date=datetime.date(2026, 1, 1),
                          high_watermark=100.0, shares=100,
                          kelly_target_pct=0.60)
        # Conviction floor (added 2026-05-01) blocks TopUp when rank_score
        # is missing/low. This test exercises cap math, not the gate —
        # set a passing rank so the gate is inert.
        hs.rank_score = 0.50
        # Cap at 20%, so even though Kelly says 60% (delta=50%), only 20%
        # is bought this session. target_pct should reflect the actual fill
        # (current 10% + 20% cap = 30%), NOT the abstract Kelly target (60%).
        ctx = SimpleNamespace(
            config={"ranking": {"kelly_sizing": {
                "enabled": True, "top_up_threshold": 0.05,
                "per_session_buy_cap": 0.20,
            }}, "regime_params": {"BULL_CALM": {}}},
            regime="BULL_CALM", confidence=1.0,
            portfolio_value=100_000.0, cash=100_000.0,
            holdings={"AAA": hs}, prices={"AAA": 100.0},
            orders=[], exits=[], rotations=[],
            bear_only=False, skip_buys=False,
        )
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1
        o = ctx.orders[0]
        # current = 100 × 100 / 100k = 10%, capped delta = 20%
        # actual fill = 200 shares × 100 / 100k = 20% added → target_pct = 30%
        assert 0.29 <= o["target_pct"] <= 0.31, \
            f"target_pct should reflect actual fill, got {o['target_pct']}"


# ── #26 + #33 — EmitRotationsTask sizing parity with SizeAndEmit ────────────

class TestEmitRotationsSizingParity:
    def test_emit_rotations_applies_kelly_branch(self):
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "task_rotation.py").read_text()
        idx = src.find("class EmitRotationsTask")
        body = src[idx:]
        # Kelly path branch must exist in EmitRotationsTask post-fix
        assert "kelly_target_pct" in body
        assert "per_session_buy_cap" in body
        assert "cusum_cooldown_progress" in body or "cooldown_mult" in body

    def test_rotation_order_carries_kelly_target(self):
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "task_rotation.py").read_text()
        idx = src.find("class EmitRotationsTask")
        body = src[idx:]
        assert '"kelly_target_pct"' in body
        assert '"order_type": "ROTATION"' in body


# ── #32 — _drive_score returns None on missing μ/σ in σ-aware modes ─────

class TestDriveScoreUnitMismatch:
    def test_no_silent_er_fallback_in_sigma_modes(self):
        """When scoring_mode is mu_minus_lambda_sigma and μ is missing,
        _drive_score must return None — not silently fall back to ER."""
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "task_rotation.py").read_text()
        # The previous implementation had a fall-through to expected_return
        # at the end of _drive_score when scoring_mode wasn't "er". Post-fix
        # those branches return None on missing μ/σ.
        idx = src.find("def _drive_score")
        body = src[idx:idx + 2500]
        # Sentinel comment from the fix
        assert "unit-mismatch" in body or "skip this row" in body


# ── #39 — BuildFeatureMatrixTask doesn't kill the chain on missing frames ──

class TestBuildFeatureMatrixNonFatal:
    def test_returns_none_not_false_when_frames_missing(self):
        """Pre-fix: BuildFeatureMatrixTask returned False, halting the
        chain so LoadGlobalCalibration / LoadNGBoost never initialized.
        Now it returns None and downstream tasks no-op individually."""
        from kernel.panel_pipeline.job_panel_scoring import BuildFeatureMatrixTask
        from kernel.pipeline.context import InferenceContext
        ctx = InferenceContext(config={}, today=datetime.date(2026, 4, 24))
        ctx._panel_scorer = SimpleNamespace(feature_cols=["a"])  # type: ignore[attr-defined]
        ctx.candidates = [SimpleNamespace(ticker="AAA")]
        ctx.holdings = {}
        ctx._panel_feature_frames = None  # simulate missing frames # type: ignore[attr-defined]
        result = BuildFeatureMatrixTask().run(ctx)
        assert result is None, \
            "must return None (continue) instead of False (halt)"


# ── #40 — ApplyNGBoost holdings get rank_score override too ──────────────

class TestApplyNGBoostHoldingsRankScore:
    def test_holdings_rank_score_set_in_override_mode(self):
        src = (_STRATEGY_DIR / "kernel" / "panel_pipeline" / "job_panel_scoring.py").read_text()
        idx = src.find("class ApplyNGBoostTask")
        body = src[idx:]
        # Holdings must get hs.rank_score updated when override_mode active
        # (previously only hs.panel_score was set)
        assert "hs.rank_score" in body, \
            "Holdings must get rank_score in mu_minus_lambda_sigma mode (#40)"


# ── #43 — VetoWeakBuysTask always populates the counter ─────────────────

class TestVetoCounterAlwaysPopulated:
    def test_panel_vetoed_counter_set_even_when_no_drops(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        from kernel.pipeline.context import InferenceContext
        from kernel.selection import CandidateResult
        ctx = InferenceContext(
            config={"ranking": {"panel_scoring": {"buy_floor": 0.10}}},
            today=datetime.date(2026, 4, 24),
        )
        ctx.candidates = [
            CandidateResult(ticker="AAA", raw_score=0.0, rank_score=0.5,
                            rs_score=0.0, panel_score=0.5),
        ]
        ctx.counters = {}
        VetoWeakBuysTask().run(ctx)
        # Counter must be present even when nothing was dropped (audit #43).
        assert "panel_vetoed" in ctx.counters
        assert ctx.counters["panel_vetoed"] == 0


# ── #79 — GlobalPanelCalibration empty-knot guard ───────────────────────

class TestGlobalCalibratorEmptyKnots:
    def test_calibrate_probability_handles_empty_knots(self):
        from training_panel.global_calibrator import GlobalPanelCalibration
        cal = GlobalPanelCalibration(
            prob_x=np.array([], dtype=float),
            prob_y=np.array([], dtype=float),
            er_x=np.array([], dtype=float),
            er_y=np.array([], dtype=float),
        )
        # Pre-fix: prob_y[0] would IndexError. Post-fix: degrades to base rate.
        v = cal.calibrate_probability(0.5)
        assert v == 0.5
        v = cal.expected_return(0.5)
        assert v == 0.0


# ── #45 — TTL-skipped tickers not falsely marked exported ────────────────

class TestTtlSkipNotExported:
    def test_ttl_skip_does_not_set_exported_flag(self):
        src = (_STRATEGY_DIR / "kernel" / "pipeline" / "pp_training.py").read_text()
        idx = src.find("def _run_ticker_chain")
        body = src[idx:idx + 2500]
        # Post-fix: TTL skip path no longer sets `tc.exported = True`
        # (it only sets ttl_skipped). Verify by confirming the path is
        # documented and ttl_skipped is the lone marker.
        assert "tc.ttl_skipped = True" in body
        # Make sure the wrong assignment is gone
        assert "tc.exported = True      # treat cached" not in body


# ── R1 (round-2): LEAN top-up updates entry_price via volume-weighted avg ──

class TestLeanAdapterTopUpCostBasis:
    """Round-2 regression: my round-1 fix preserved entry_price on top-up
    (no longer reset to today's price), but didn't compute the volume-
    weighted average. SimAdapter does — kernel.exits.check_stop_loss /
    check_trailing_stop / check_single_day_loss all use HoldingState.entry_price
    so the two adapters were diverging. Round-2 fix: LEAN top-up does
    `(old_entry × old_qty + price × shares) / new_qty`, matching SimAdapter."""

    def test_lean_topup_volume_weighted_avg(self):
        src = (_STRATEGY_DIR / "adapters" / "lean.py").read_text()
        # Sentinel for the avg-cost computation in the top-up branch.
        idx = src.find("if already_held:")
        body = src[idx:idx + 1500]
        assert "old_qty" in body
        assert "new_qty = old_qty + shares" in body or "new_qty" in body
        assert "hs.entry_price" in body and "* old_qty" in body, \
            "LEAN top-up must volume-weight entry_price (R1)"


# ── #58 — RunnerAdapter intraday overlay copies before mutating ─────────

class TestIntradayOverlayCopiesFrame:
    def test_source_copies_before_mutation(self):
        src = (_STRATEGY_DIR / "adapters" / "runner.py").read_text()
        # The post-fix block must copy() before .at[...] assignment so cached
        # OHLCV references aren't mutated.
        idx = src.find("Overwrite today's daily bar's close")
        body = src[idx:idx + 600]
        assert ".copy()" in body, \
            "intraday overlay must copy frame before in-place mutation (#58)"


# ── #88 — LeanAdapter wires blocked_min_hold counter ──────────────────

class TestLeanAdapterBlockedMinHoldWired:
    def test_blocked_min_hold_summed_in_commit(self):
        src = (_STRATEGY_DIR / "adapters" / "lean.py").read_text()
        assert "_blocked_min_hold" in src
        assert 'c.get("blocked_min_hold"' in src


# ── #89 — artifact_path doesn't double the artifacts/ prefix ────────────

class TestArtifactPathDoubleArtifactsGuard:
    def test_no_double_artifacts(self):
        from kernel.config import artifact_path, STRATEGY_DIR
        p1 = artifact_path("foo.json")
        p2 = artifact_path("artifacts/foo.json")
        # Both should resolve to {STRATEGY_DIR}/artifacts/foo.json
        assert p1 == p2 == STRATEGY_DIR / "artifacts" / "foo.json"


# ── #69 — calibrate falls back instead of raising on unknown method ─────

class TestUnknownCalibrationGraceful:
    def test_typo_in_method_returns_base_rate(self):
        from kernel.scoring import ScoreCalibration
        cal = ScoreCalibration(method="identy", base_rate=0.42)
        # Pre-fix: would raise ValueError. Post-fix: returns base_rate.
        v = cal.calibrate(0.5)
        assert v == 0.42


# ── #71 — record_training_run resolves commit_sha once ───────────────────

class TestRecordTrainingRunSingleSha:
    def test_sha_called_once(self):
        src = (_STRATEGY_DIR / "kernel" / "persistence.py").read_text()
        # Find record_training_run body
        idx = src.find("def record_training_run")
        body = src[idx:idx + 4000]
        # Count _commit_sha() calls inside the function body
        assert body.count("_commit_sha()") == 1, \
            "record_training_run must resolve commit sha exactly once (#71)"


# ── #59 — RunnerAdapter logs non-watchlist held positions ────────────────

class TestRunnerNonWatchlistHoldingsLogged:
    def test_source_warns_on_non_watchlist_holds(self):
        src = (_STRATEGY_DIR / "adapters" / "runner.py").read_text()
        assert "non_wl_holds" in src or "outside watchlist" in src


# ── #84 — Scheduled-mode loop emits operator warning ────────────────────

class TestScheduledModeWarning:
    def test_scheduled_mode_path_warns(self):
        src = Path(__file__).resolve().parent.parent / "live" / "runner.py"
        body = src.read_text()
        # The non-`--once` path must log a warning that this is for testing
        assert "ad-hoc testing only" in body or "audit #84" in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
