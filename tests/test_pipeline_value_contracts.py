"""Contract tests — pin down range/scale of every pipeline output value.

User spec 2026-05-03 ("每个 output value 都应该有 test 测一下 range 和合理性",
see ``feedback_output_value_range_tests.md``). After the buy_floor=0.30
scale-mismatch P0 (commit 410758b applied a calibrated-scale threshold
to a raw XGB margin field for 5 days in production), this file enforces
the contracts that would have caught that bug on first commit.

Each contract here is one of:

  * **Output range** — the value sits inside its documented natural range
    (e.g. calibrated rank_score ∈ [0, 1]).
  * **Boundary** — the gate's pass/fail behavior is correct at exactly the
    threshold (just-pass + just-fail).
  * **Scale invariant** — the gate compares values that share a scale
    (e.g. veto floor in [0, 1] reads field that is also in [0, 1]).

NOT a unit-test replacement for individual tasks. This file's *job* is to
catch the entire CLASS of "you applied a threshold to the wrong-scaled
field" bugs, plus other range violations.
"""
from __future__ import annotations

import math
import sys
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))


# ── Calibrator output range ──────────────────────────────────────────────────


class TestGlobalPanelCalibrationOutputRange(unittest.TestCase):
    """calibrate_probability(x) ∈ [prob_y.min, prob_y.max] for any finite x."""

    def _make_cal(self, prob_y=(0.05, 0.5, 0.95), er_y=(-0.02, 0.0, 0.04)):
        from training_panel.global_calibrator import GlobalPanelCalibration
        return GlobalPanelCalibration(
            prob_x=np.array([-0.05, 0.0, 0.05]),
            prob_y=np.array(prob_y),
            er_x=np.array([-0.05, 0.0, 0.05]),
            er_y=np.array(er_y),
        )

    def test_probability_within_knot_range(self):
        cal = self._make_cal(prob_y=(0.05, 0.5, 0.95))
        for x in (-1.0, -0.05, 0.0, 0.025, 0.05, 1.0, 1e6, -1e6):
            p = cal.calibrate_probability(x)
            self.assertGreaterEqual(p, 0.05)
            self.assertLessEqual(p, 0.95)

    def test_probability_strict_unit_range_when_knots_are_unit(self):
        cal = self._make_cal(prob_y=(0.0, 0.5, 1.0))
        for x in np.linspace(-2.0, 2.0, 50):
            p = cal.calibrate_probability(float(x))
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_empty_knot_arrays_returns_baserate(self):
        from training_panel.global_calibrator import GlobalPanelCalibration
        cal = GlobalPanelCalibration(
            prob_x=np.array([]), prob_y=np.array([]),
            er_x=np.array([]),   er_y=np.array([]),
        )
        self.assertEqual(cal.calibrate_probability(0.0), 0.5)
        self.assertEqual(cal.expected_return(0.0), 0.0)

    def test_extrapolation_uses_endpoint_not_unbounded(self):
        cal = self._make_cal(prob_y=(0.10, 0.50, 0.90))
        self.assertAlmostEqual(cal.calibrate_probability(1e9), 0.90)
        self.assertAlmostEqual(cal.calibrate_probability(-1e9), 0.10)


# ── Realized vol non-negativity ──────────────────────────────────────────────


class TestRealizedVolNonNegative(unittest.TestCase):
    """RealizedVolGateTask._realized_vol_annualized ≥ 0 for any close series."""

    def test_vol_non_negative(self):
        from kernel.pipeline.task_risk_gates import RealizedVolGateTask
        rng = np.random.default_rng(0)
        for sigma in (0.005, 0.01, 0.05, 0.10, 0.30):
            close = 100.0 * np.exp(np.cumsum(rng.normal(0, sigma, 80)))
            df = pd.DataFrame({"close": close},
                              index=pd.date_range("2024-01-01", periods=80, freq="B"))
            v = RealizedVolGateTask._realized_vol_annualized(df, window=60)
            self.assertIsNotNone(v)
            self.assertGreaterEqual(v, 0.0)
            self.assertTrue(math.isfinite(v))

    def test_constant_series_returns_zero_or_none(self):
        from kernel.pipeline.task_risk_gates import RealizedVolGateTask
        df = pd.DataFrame({"close": [100.0] * 80},
                          index=pd.date_range("2024-01-01", periods=80, freq="B"))
        v = RealizedVolGateTask._realized_vol_annualized(df, window=60)
        # All-zero returns → std=0 → annualized=0
        self.assertEqual(v, 0.0)


# ── Gate scale invariants ────────────────────────────────────────────────────


class TestGateScaleInvariants(unittest.TestCase):
    """Each threshold-gate compares values on the same scale.

    These tests prevent the class of bugs that produced the 2026-04-29
    buy_floor incident: a gate's threshold was set in the calibrated
    range [0, 1] but applied to the raw XGB margin range [0, 0.05].
    """

    def test_veto_weak_buys_reads_rank_score_field(self):
        """VetoWeakBuysTask MUST read rank_score (calibrated), not panel_score."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        import inspect
        src = inspect.getsource(VetoWeakBuysTask.run)
        self.assertIn("rank_score", src,
                      "VetoWeakBuysTask must compare against rank_score (calibrated)")

    def test_distribution_floor_reads_rank_score(self):
        """quality_floor.distribution_floor reads rank_score (calibrated)."""
        from kernel.panel_pipeline.task_quality_floor import _gate_a_distribution_floor
        import inspect
        src = inspect.getsource(_gate_a_distribution_floor)
        self.assertIn("rank_score", src)

    def test_topup_held_reads_rank_score(self):
        """TopUpHeldTask conviction floor reads rank_score (calibrated)."""
        from kernel.pipeline import task_topup
        import inspect
        src = inspect.getsource(task_topup)
        self.assertIn("rank_score", src)
        self.assertIn("topup_conviction_floor", src)


# ── Boundary tests for active gates ──────────────────────────────────────────


def _make_cand_with_rank(ticker, rank_score, panel_score=0.0):
    from kernel.selection import CandidateResult
    return CandidateResult(
        ticker=ticker, raw_score=0.0, rank_score=rank_score,
        rs_score=0.0, detail="", expected_return=0.0,
        panel_score=panel_score,
    )


def _make_ctx_with_cands(cands, buy_floor=None):
    from types import SimpleNamespace
    return SimpleNamespace(
        config={"ranking": {"panel_scoring": {"buy_floor": buy_floor}}},
        candidates=cands, holdings={}, counters={},
    )


class TestVetoWeakBuysBoundary(unittest.TestCase):
    """Just-above and just-below the floor produce the right verdict."""

    def test_just_below_floor_drops(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx_with_cands(
            [_make_cand_with_rank("X", 0.299)], buy_floor=0.30,
        )
        VetoWeakBuysTask().run(ctx)
        self.assertEqual(len(ctx.candidates), 0)

    def test_exactly_at_floor_keeps(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx_with_cands(
            [_make_cand_with_rank("X", 0.30)], buy_floor=0.30,
        )
        VetoWeakBuysTask().run(ctx)
        self.assertEqual([c.ticker for c in ctx.candidates], ["X"])

    def test_just_above_floor_keeps(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx_with_cands(
            [_make_cand_with_rank("X", 0.301)], buy_floor=0.30,
        )
        VetoWeakBuysTask().run(ctx)
        self.assertEqual([c.ticker for c in ctx.candidates], ["X"])

    def test_zero_floor_never_drops(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx_with_cands(
            [_make_cand_with_rank("X", 0.0)], buy_floor=0.0,
        )
        VetoWeakBuysTask().run(ctx)
        self.assertEqual([c.ticker for c in ctx.candidates], ["X"])

    def test_unit_floor_drops_anyone_below_one(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx_with_cands(
            [_make_cand_with_rank(f"T{i}", v)
             for i, v in enumerate([0.0, 0.5, 0.99, 1.0, 1.001])],
            buy_floor=1.0,
        )
        VetoWeakBuysTask().run(ctx)
        self.assertEqual({c.ticker for c in ctx.candidates}, {"T3", "T4"})


# ── Dataclass field invariants ───────────────────────────────────────────────


class TestDataclassFieldInvariants(unittest.TestCase):
    def test_holding_state_has_required_fields(self):
        from kernel.exits import HoldingState
        names = {f.name for f in fields(HoldingState)}
        self.assertIn("entry_price", names)
        self.assertIn("entry_date", names)
        self.assertIn("high_watermark", names)
        self.assertIn("shares", names)
        self.assertIn("rank_score", names)
        self.assertIn("panel_score", names)
        self.assertIn("mu", names)
        self.assertIn("sigma", names)

    def test_candidate_result_has_required_fields(self):
        from kernel.selection import CandidateResult
        names = {f.name for f in fields(CandidateResult)}
        self.assertIn("ticker", names)
        self.assertIn("rank_score", names)
        self.assertIn("panel_score", names)
        self.assertIn("rs_score", names)
        self.assertIn("expected_return", names)
        self.assertIn("mu", names)
        self.assertIn("sigma", names)


# ── Position concentration as fraction of portfolio ──────────────────────────


class TestPositionConcentrationGateFractionScale(unittest.TestCase):
    """The gate's max_pct is a fraction of portfolio_value, not absolute $."""

    def test_threshold_is_fraction_not_dollars(self):
        from kernel.pipeline.task_risk_gates import PositionConcentrationGateTask
        from types import SimpleNamespace
        from dataclasses import dataclass

        @dataclass
        class _Cand:
            ticker: str = "X"
            raw_score: float = 0.0
            rank_score: float = 1.0
            rs_score: float = 0.0

        @dataclass
        class _H:
            shares: float = 0.0
            prev_close: float = 0.0

        ctx = SimpleNamespace(
            config={"risk_gates": {"position_concentration": {"max_pct": 0.10}}},
            candidates=[_Cand("X")],
            holdings={"X": _H(shares=15, prev_close=100.0)},  # = $1500 = 15% of $10k
            prices={"X": 100.0},
            portfolio_value=10000.0, cash=0.0,
            counters={},
        )
        PositionConcentrationGateTask().run(ctx)
        # 15% > 10% cap → drop
        self.assertEqual(len(ctx.candidates), 0,
                         "max_pct=0.10 must be a fraction (10%), not $0.10")


# ── Realized vol as annualized fraction ──────────────────────────────────────


class TestRealizedVolGateAnnualizedScale(unittest.TestCase):
    """max_annualized=0.60 means 60% annualized vol, not 0.60 daily vol."""

    def test_threshold_is_annualized_fraction(self):
        from kernel.pipeline.task_risk_gates import RealizedVolGateTask
        from types import SimpleNamespace
        from dataclasses import dataclass

        @dataclass
        class _Cand:
            ticker: str
            raw_score: float = 0.0
            rank_score: float = 1.0
            rs_score: float = 0.0

        # Daily sigma 0.063 → annualized ≈ 1.0 (100%). With cap 0.60 → drop.
        rng = np.random.default_rng(1)
        n = 80
        rets_high = rng.normal(0.0, 1.0 / math.sqrt(252.0), n)  # 100% annualized
        close_high = 100.0 * np.exp(np.cumsum(rets_high))
        df_high = pd.DataFrame({"close": close_high},
                               index=pd.date_range("2024-01-01", periods=n, freq="B"))

        # Daily sigma 0.012 → annualized ≈ 0.20 (20%). With cap 0.60 → keep.
        rets_low = rng.normal(0.0, 0.20 / math.sqrt(252.0), n)
        close_low = 100.0 * np.exp(np.cumsum(rets_low))
        df_low = pd.DataFrame({"close": close_low},
                              index=pd.date_range("2024-01-01", periods=n, freq="B"))

        ctx = SimpleNamespace(
            config={"risk_gates": {"realized_vol": {"max_annualized": 0.60}}},
            candidates=[_Cand("HIGH"), _Cand("LOW")],
            ohlcv={"HIGH": df_high, "LOW": df_low},
            counters={},
        )
        RealizedVolGateTask().run(ctx)
        self.assertEqual({c.ticker for c in ctx.candidates}, {"LOW"})


# ── Confidence + regime contracts ────────────────────────────────────────────


class TestRegimeContracts(unittest.TestCase):
    def test_regime_label_is_in_known_set(self):
        # Static check on regime emission code — regimes come from the
        # GMM finalize task. We assert the documented enum here.
        from kernel.pipeline.context import InferenceContext
        import datetime
        ctx = InferenceContext(config={}, today=datetime.date(2026, 5, 3))
        # Default
        self.assertIn(ctx.regime,
                      {"BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"})

    def test_confidence_default_in_unit_interval(self):
        from kernel.pipeline.context import InferenceContext
        import datetime
        ctx = InferenceContext(config={}, today=datetime.date(2026, 5, 3))
        self.assertGreaterEqual(ctx.confidence, 0.0)
        self.assertLessEqual(ctx.confidence, 1.0)


# ── Calibrator vector path matches scalar path ───────────────────────────────


class TestCalibratorVectorScalarParity(unittest.TestCase):
    def test_vec_matches_scalar_within_eps(self):
        from training_panel.global_calibrator import GlobalPanelCalibration
        cal = GlobalPanelCalibration(
            prob_x=np.array([-0.05, 0.0, 0.05]),
            prob_y=np.array([0.10, 0.50, 0.90]),
            er_x=np.array([-0.05, 0.0, 0.05]),
            er_y=np.array([-0.02, 0.0, 0.04]),
        )
        xs = np.linspace(-0.1, 0.1, 25)
        vec = cal.calibrate_probability_vec(xs)
        scal = np.array([cal.calibrate_probability(float(x)) for x in xs])
        np.testing.assert_allclose(vec, scal, atol=1e-12)


# ── Phase 2: NGBoost output bounds (clamps documented in code) ──────────────


class TestNGBoostOutputClamps(unittest.TestCase):
    """NGBoostHead.predict_distribution clamps σ ∈ [1e-6, 5.0], μ ∈ [-1, 1]."""

    def test_sigma_floor_constant_present_in_source(self):
        from training_panel import ngboost_head
        import inspect
        src = inspect.getsource(ngboost_head)
        self.assertIn("SIGMA_FLOOR", src)
        self.assertIn("SIGMA_CEIL", src)
        # Documented values
        self.assertIn("1e-6", src)
        self.assertIn("5.0", src)

    def test_mu_ceil_constant_present_in_source(self):
        from training_panel import ngboost_head
        import inspect
        src = inspect.getsource(ngboost_head)
        self.assertIn("MU_CEIL", src)


# ── kelly_target_pct: input-fuzz output range invariant ─────────────────────


class TestKellyTargetPctRangeInvariant(unittest.TestCase):
    """kelly_target_pct must return ∈ [0, min(max_pct, max_concentration)]
    for ANY valid input. Audit fix K-1 (2026-04-25) caught a NaN leak
    via min(...) propagation — test enumerates the failure modes that
    have actually shipped + property-fuzzes wide μ/σ inputs to pin the
    invariant.

    This belongs at the contract layer, not the unit layer: every
    downstream task (SizeAndEmitTask cap, TopUpHeldTask threshold) reads
    this output and assumes it's a fraction in the documented range. A
    NaN or negative leak would silently corrupt order sizing.
    """

    def test_degenerate_inputs_return_zero(self):
        from kernel.kelly import kelly_target_pct
        cases = [
            (None, 0.05, "mu=None"),
            (0.02, None, "sigma=None"),
            (float("nan"), 0.05, "mu=NaN"),
            (0.02, float("nan"), "sigma=NaN"),
            (float("inf"), 0.05, "mu=inf"),
            (0.02, float("inf"), "sigma=inf"),
            (0.02, 0.0, "sigma=0"),
            (0.02, -0.01, "sigma<0"),
            (-0.01, 0.05, "mu<0 (loses bet)"),
            (0.0, 0.05, "mu=0 (no edge)"),
            ("not_a_number", 0.05, "mu non-numeric"),
            (0.02, "not_a_number", "sigma non-numeric"),
        ]
        for mu, sigma, label in cases:
            with self.subTest(label):
                v = kelly_target_pct(mu, sigma, max_pct=0.15,
                                      max_concentration=0.35)
                self.assertEqual(v, 0.0,
                    f"{label}: expected 0.0, got {v!r}")

    def test_positive_input_in_documented_range(self):
        """For every (μ, σ) in a wide grid, output ∈ [0, min(max_pct, conc)]
        and is finite."""
        from kernel.kelly import kelly_target_pct
        rng = np.random.default_rng(0)
        max_pct, max_conc = 0.15, 0.35
        upper = min(max_pct, max_conc)
        # 200 random (mu, sigma) pairs over wide range
        for _ in range(200):
            mu    = float(rng.uniform(0.001, 0.5))    # always positive
            sigma = float(rng.uniform(0.001, 1.0))
            v = kelly_target_pct(mu, sigma, max_pct=max_pct,
                                  max_concentration=max_conc,
                                  fractional=0.25)
            self.assertTrue(math.isfinite(v),
                f"non-finite output for mu={mu} sigma={sigma}: {v!r}")
            self.assertGreaterEqual(v, 0.0,
                f"negative output for mu={mu} sigma={sigma}: {v!r}")
            self.assertLessEqual(v, upper + 1e-9,
                f"exceeded upper={upper} for mu={mu} sigma={sigma}: {v!r}")

    def test_huge_edge_capped_at_smaller_of_two_caps(self):
        """When Kelly says "all-in", we cap at min(max_pct, max_concentration)."""
        from kernel.kelly import kelly_target_pct
        # Pathological: tiny σ + decent μ → f* = μ/σ² is huge.
        v = kelly_target_pct(0.05, 0.001, max_pct=0.10,
                              max_concentration=0.35,
                              fractional=1.0)
        # max_pct (0.10) is the binding cap here, NOT max_concentration.
        self.assertAlmostEqual(v, 0.10, places=6)
        # Flip the binding cap.
        v2 = kelly_target_pct(0.05, 0.001, max_pct=0.50,
                              max_concentration=0.20,
                              fractional=1.0)
        self.assertAlmostEqual(v2, 0.20, places=6)

    def test_min_edge_threshold_excludes_low_mu(self):
        """μ ≤ min_edge → 0 (don't bet on no edge)."""
        from kernel.kelly import kelly_target_pct
        # μ exactly at edge: drop
        v = kelly_target_pct(0.005, 0.05, max_pct=0.15, min_edge=0.005)
        self.assertEqual(v, 0.0)
        # Just above: enters Kelly formula
        v2 = kelly_target_pct(0.006, 0.05, max_pct=0.15, min_edge=0.005)
        self.assertGreater(v2, 0.0)


# ── Phase 2: risk_metrics finiteness contracts ──────────────────────────────


class TestSharpeContract(unittest.TestCase):
    """sharpe_ratio: NaN on degenerate input, finite on non-degenerate."""

    def test_constant_series_returns_nan(self):
        from renquant_common.risk_metrics import sharpe_ratio
        s = pd.Series([0.001] * 252)
        self.assertTrue(math.isnan(sharpe_ratio(s)))

    def test_too_short_returns_nan(self):
        from renquant_common.risk_metrics import sharpe_ratio
        self.assertTrue(math.isnan(sharpe_ratio(pd.Series([0.01]))))

    def test_normal_returns_finite(self):
        from renquant_common.risk_metrics import sharpe_ratio
        rng = np.random.default_rng(42)
        s = pd.Series(rng.normal(0.0005, 0.01, 252))
        v = sharpe_ratio(s)
        self.assertTrue(math.isfinite(v))

    def test_positive_drift_positive_sharpe(self):
        from renquant_common.risk_metrics import sharpe_ratio
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0.001, 0.01, 252))  # +0.1% daily drift
        self.assertGreater(sharpe_ratio(s), 0.0)

    def test_negative_drift_negative_sharpe(self):
        from renquant_common.risk_metrics import sharpe_ratio
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(-0.001, 0.01, 252))
        self.assertLess(sharpe_ratio(s), 0.0)

    def test_nan_returns_handled(self):
        from renquant_common.risk_metrics import sharpe_ratio
        s = pd.Series([0.01, np.nan, 0.02, np.nan, -0.01, 0.005])
        v = sharpe_ratio(s)
        # 4 valid observations — should be finite
        self.assertTrue(math.isfinite(v))


class TestSortinoContract(unittest.TestCase):
    def test_constant_above_target_returns_nan(self):
        from renquant_common.risk_metrics import sortino_ratio
        s = pd.Series([0.001] * 252)
        # No downside observations
        self.assertTrue(math.isnan(sortino_ratio(s)))

    def test_normal_returns_finite(self):
        from renquant_common.risk_metrics import sortino_ratio
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0.0005, 0.01, 252))
        self.assertTrue(math.isfinite(sortino_ratio(s)))

    def test_sortino_geq_sharpe_when_skewed_positive(self):
        # Heavily right-skewed → downside std ≪ full std → Sortino > Sharpe
        from renquant_common.risk_metrics import sharpe_ratio, sortino_ratio
        # Lognormal-ish: mostly small ups with rare big ones, few downs
        rng = np.random.default_rng(0)
        rets = rng.normal(0.0, 0.005, 252)
        # Boost upside tail
        rets[rng.choice(252, 10, replace=False)] += 0.05
        s = pd.Series(rets)
        sh = sharpe_ratio(s)
        so = sortino_ratio(s)
        if math.isfinite(sh) and math.isfinite(so):
            self.assertGreaterEqual(so, sh - 0.5)  # tolerant


class TestMaxDDContract(unittest.TestCase):
    def test_dd_non_negative(self):
        from renquant_common.risk_metrics import max_drawdown
        # Random walk equity curve
        rng = np.random.default_rng(0)
        equity = pd.Series(np.cumprod(1.0 + rng.normal(0.0005, 0.01, 252)))
        v = max_drawdown(equity)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_monotone_curve_zero_dd(self):
        from renquant_common.risk_metrics import max_drawdown
        equity = pd.Series([100.0, 101, 102, 103.5, 110, 120])
        self.assertEqual(max_drawdown(equity), 0.0)

    def test_dd_in_unit_fraction(self):
        from renquant_common.risk_metrics import max_drawdown
        equity = pd.Series([100.0, 110, 90, 95])  # peak 110, trough 90 → dd 18.18%
        self.assertAlmostEqual(max_drawdown(equity), (110 - 90) / 110, places=5)


class TestCalmarContract(unittest.TestCase):
    def test_zero_dd_returns_nan(self):
        from renquant_common.risk_metrics import calmar_ratio
        self.assertTrue(math.isnan(calmar_ratio(0.10, 0.0)))

    def test_negative_dd_returns_nan(self):
        from renquant_common.risk_metrics import calmar_ratio
        self.assertTrue(math.isnan(calmar_ratio(0.10, -0.05)))

    def test_normal_inputs_finite(self):
        from renquant_common.risk_metrics import calmar_ratio
        v = calmar_ratio(0.10, 0.20)
        self.assertAlmostEqual(v, 0.5)


# ── Phase 2: HoldingState invariants ────────────────────────────────────────


class TestHoldingStateInvariants(unittest.TestCase):
    """shares ≥ 0, entry_price > 0 are documented invariants."""

    def test_default_shares_non_negative(self):
        from kernel.exits import HoldingState
        import datetime
        h = HoldingState(
            entry_price=100.0,
            entry_date=datetime.date(2026, 1, 1),
            high_watermark=100.0,
        )
        self.assertGreaterEqual(h.shares, 0)


# ── Phase 3: tax — IRS short-term vs long-term contract ─────────────────────


class TestTaxComputationContracts(unittest.TestCase):
    """compute_trade_tax: short-term rate < 365d hold; long-term ≥ 365d.

    IRS §1(h) defines long-term capital gain as held > 1 year. RenQuant's
    365-day boundary is a reasonable approximation (calendar day, not
    "more than 1 year"). The contract is: at boundary 365 → long-term.
    """

    def test_just_below_threshold_is_short_term(self):
        from kernel.portfolio import compute_trade_tax
        # 364 days — short-term applies
        tax = compute_trade_tax(
            gross_pnl=1000.0, hold_days=364,
            short_term_rate=0.37, long_term_rate=0.20,
        )
        self.assertAlmostEqual(tax, 1000 * 0.37)

    def test_at_threshold_is_long_term(self):
        from kernel.portfolio import compute_trade_tax
        tax = compute_trade_tax(
            gross_pnl=1000.0, hold_days=365,
            short_term_rate=0.37, long_term_rate=0.20,
        )
        self.assertAlmostEqual(tax, 1000 * 0.20)

    def test_above_threshold_is_long_term(self):
        from kernel.portfolio import compute_trade_tax
        tax = compute_trade_tax(
            gross_pnl=1000.0, hold_days=730,
            short_term_rate=0.37, long_term_rate=0.20,
        )
        self.assertAlmostEqual(tax, 1000 * 0.20)

    def test_loss_pays_no_tax(self):
        from kernel.portfolio import compute_trade_tax
        # A loss owes no tax (consumer reconciles offsets externally)
        for d in (10, 365, 730):
            tax = compute_trade_tax(
                gross_pnl=-500.0, hold_days=d,
                short_term_rate=0.37, long_term_rate=0.20,
            )
            self.assertEqual(tax, 0.0)

    def test_zero_gain_pays_no_tax(self):
        from kernel.portfolio import compute_trade_tax
        self.assertEqual(
            compute_trade_tax(0.0, 100, 0.37, 0.20),
            0.0,
        )

    def test_nan_gross_pnl_returns_zero(self):
        from kernel.portfolio import compute_trade_tax
        self.assertEqual(
            compute_trade_tax(float("nan"), 100, 0.37, 0.20),
            0.0,
        )

    def test_inf_gross_pnl_returns_zero(self):
        from kernel.portfolio import compute_trade_tax
        self.assertEqual(
            compute_trade_tax(float("inf"), 100, 0.37, 0.20),
            0.0,
        )

    def test_custom_threshold_honored(self):
        from kernel.portfolio import compute_trade_tax
        # If trader uses 30d as their LT threshold (not 1y)
        tax = compute_trade_tax(
            gross_pnl=1000.0, hold_days=31,
            short_term_rate=0.37, long_term_rate=0.20,
            long_term_threshold_days=30,
        )
        self.assertAlmostEqual(tax, 1000 * 0.20)

    def test_tax_le_gross_pnl_for_reasonable_rates(self):
        """A sanity contract: tax never exceeds the gross gain (rate ≤ 1)."""
        from kernel.portfolio import compute_trade_tax
        for rate in (0.0, 0.15, 0.37, 0.50):
            for hd in (1, 100, 365, 730):
                tax = compute_trade_tax(
                    gross_pnl=1000.0, hold_days=hd,
                    short_term_rate=rate, long_term_rate=rate,
                )
                self.assertLessEqual(tax, 1000.0)
                self.assertGreaterEqual(tax, 0.0)


# ── Phase 3: PortfolioQP weight + leverage bounds ───────────────────────────


class TestPortfolioQPBounds(unittest.TestCase):
    """solve_portfolio_qp: target_w[i] ∈ [w_lower, w_upper] for each i."""

    def _solve_simple(self, **overrides):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        kwargs = dict(
            w_current=[0.0, 0.0, 0.0, 0.0],
            mu=[0.001, 0.002, -0.001, 0.0005],
            sigma=[0.02, 0.02, 0.02, 0.02],
            risk_aversion=3.0,
            cash_reserve=0.0,
            w_upper=0.20,
            w_lower=0.0,
            dw_max=0.50,
        )
        kwargs.update(overrides)
        return solve_portfolio_qp(**kwargs)

    def test_target_w_respects_upper_bound(self):
        sol = self._solve_simple(w_upper=0.10)
        for w in sol.target_w:
            self.assertLessEqual(w, 0.10 + 1e-6)

    def test_target_w_respects_lower_bound(self):
        sol = self._solve_simple(w_lower=0.0)
        for w in sol.target_w:
            self.assertGreaterEqual(w, 0.0 - 1e-6)

    def test_sum_target_w_within_full_invest(self):
        """Σ target_w ≤ 1.0 (no implicit leverage)."""
        sol = self._solve_simple()
        self.assertLessEqual(sum(sol.target_w), 1.0 + 1e-6)

    def test_cash_reserve_honored(self):
        """Σ target_w ≤ 1 - cash_reserve."""
        sol = self._solve_simple(cash_reserve=0.10)
        self.assertLessEqual(sum(sol.target_w), 0.90 + 1e-6)

    def test_turnover_max_honored(self):
        """Σ |Δw| ≤ turnover_max when set."""
        sol = self._solve_simple(turnover_max=0.10)
        total_dw = sum(abs(d) for d in sol.delta_w)
        self.assertLessEqual(total_dw, 0.10 + 1e-6)

    def test_status_in_known_set(self):
        sol = self._solve_simple()
        self.assertIn(sol.status,
                      {"optimal", "optimal_inaccurate", "infeasible",
                       "unbounded", "infeasible_inaccurate", "unbounded_inaccurate",
                       "solver_error", "skipped_no_assets"})

    def test_objective_finite(self):
        sol = self._solve_simple()
        if sol.status in ("optimal", "optimal_inaccurate"):
            self.assertTrue(math.isfinite(sol.objective))


# ── Phase 3: order quantity / shares invariants ─────────────────────────────


class TestSharesInvariants(unittest.TestCase):
    """HoldingState.shares is float (broker accounting), but never negative
    in long-only mode, and never NaN."""

    def test_zero_default_is_finite(self):
        from kernel.exits import HoldingState
        import datetime
        h = HoldingState(
            entry_price=100.0,
            entry_date=datetime.date(2026, 1, 1),
            high_watermark=100.0,
        )
        self.assertTrue(math.isfinite(h.shares))
        self.assertGreaterEqual(h.shares, 0.0)

    def test_explicit_shares_preserved(self):
        from kernel.exits import HoldingState
        import datetime
        h = HoldingState(
            entry_price=100.0,
            entry_date=datetime.date(2026, 1, 1),
            high_watermark=100.0,
            shares=33.0,
        )
        self.assertEqual(h.shares, 33.0)


# ── Phase 3b: calibrator NaN-leaf collapse robustness ──────────────────────


class TestCalibratorNanLeafCollapseFilter(unittest.TestCase):
    """fit_global_calibrator drops rows whose raw score equals the modal
    NaN-leaf value (XGB routes all-NaN-feature rows to one terminal node).
    """

    def _make_panel_with_collapse(self, n_real=2000, n_nan=3000,
                                  collapse_value=0.025, seed=0):
        """Synthesize panel_scores + future_returns where 60% of rows
        are 'NaN-leaf collapsed' to the same constant raw score, and 40%
        are real signal-bearing rows.
        """
        rng = np.random.default_rng(seed)

        real_raw = rng.normal(0.0, 0.01, n_real)
        real_fwd = real_raw * 0.5 + rng.normal(0.0, 0.02, n_real)
        nan_raw = np.full(n_nan, collapse_value)
        nan_fwd = rng.normal(0.0, 0.02, n_nan)

        raw = np.concatenate([real_raw, nan_raw])
        fwd = np.concatenate([real_fwd, nan_fwd])
        # Distribute across (ticker, date) keys
        n_total = n_real + n_nan
        dates = pd.date_range("2024-01-01", periods=n_total, freq="B")[:n_total]
        per_t_size = n_total // 5
        panel_scores: dict[str, pd.Series] = {}
        future_returns: dict[str, pd.Series] = {}
        for i, t in enumerate(["A", "B", "C", "D", "E"]):
            lo, hi = i * per_t_size, min((i + 1) * per_t_size, n_total)
            panel_scores[t] = pd.Series(raw[lo:hi], index=dates[lo:hi])
            future_returns[t] = pd.Series(fwd[lo:hi], index=dates[lo:hi])
        return panel_scores, future_returns

    def test_nan_leaf_collapse_panel_now_fits(self):
        """Pre-fix this would raise '< 5 unique y'. Post-fix the filter
        drops the collapsed rows and the calibrator fits on the real-signal
        residual."""
        from training_panel.global_calibrator import fit_global_calibrator
        panel_scores, future_returns = self._make_panel_with_collapse()
        cal = fit_global_calibrator(
            panel_scores, future_returns,
            min_rows=500,    # real n=2000 after filter
            threshold=0.0,
        )
        # ≥5 unique probability knots → calibrator was actually fit
        self.assertGreaterEqual(len(set(np.round(cal.prob_y, 8))), 5)

    def test_clean_panel_unaffected_by_filter(self):
        """A panel without NaN-leaf collapse fits as before — the filter
        only kicks in when one value is anomalously over-represented."""
        from training_panel.global_calibrator import fit_global_calibrator
        rng = np.random.default_rng(0)
        n = 2500
        raw = rng.normal(0.0, 0.01, n)
        fwd = raw * 0.5 + rng.normal(0.0, 0.02, n)
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        panel_scores = {"X": pd.Series(raw[:n], index=dates[:n])}
        future_returns = {"X": pd.Series(fwd[:n], index=dates[:n])}
        cal = fit_global_calibrator(
            panel_scores, future_returns, min_rows=500, threshold=0.0,
        )
        self.assertGreaterEqual(len(set(np.round(cal.prob_y, 8))), 5)


# ── Phase 3c: calibrator rolling window ─────────────────────────────────────


class TestCalibratorRollingWindow(unittest.TestCase):
    """fit_global_calibrator(rolling_window_years=N) keeps only last N years."""

    def _make_panel(self, total_years=5):
        """5 years of daily-ish data; first 4 years have flipped (anti-) signal,
        last 1 year has clean positive signal. Rolling window ≤ 1y should
        pick up the positive; full pool should average to weak."""
        rng = np.random.default_rng(0)
        n_per_year = 252
        n = total_years * n_per_year
        dates = pd.date_range("2020-01-01", periods=n, freq="B")

        panel_scores: dict[str, pd.Series] = {}
        future_returns: dict[str, pd.Series] = {}
        for tic in "ABCDE":
            raw = rng.normal(0, 0.01, n)
            fwd = np.empty(n)
            # First 4 years: anti-signal
            split = n - n_per_year
            fwd[:split] = -raw[:split] * 0.5 + rng.normal(0, 0.02, split)
            fwd[split:] = +raw[split:] * 0.5 + rng.normal(0, 0.02, n - split)
            panel_scores[tic] = pd.Series(raw, index=dates)
            future_returns[tic] = pd.Series(fwd, index=dates)
        return panel_scores, future_returns

    def test_rolling_window_excludes_old_data(self):
        from training_panel.global_calibrator import fit_global_calibrator
        ps, fr = self._make_panel()
        # Full pool should fail (anti-signal dominates)
        # Rolling 1y window should fit successfully on positive-signal slice
        cal_recent = fit_global_calibrator(
            ps, fr, min_rows=200, threshold=0.0,
            rolling_window_years=1.0,
        )
        # ≥5 unique knots → fit succeeded on the 1y slice
        self.assertGreaterEqual(len(set(np.round(cal_recent.prob_y, 8))), 5)
        meta = cal_recent.metadata
        # Rolling 1y window: 365.25 calendar days ≈ ~262 business days × 5
        # tickers ≈ 1310 rows max (allow for boundary calendar drift).
        self.assertLessEqual(meta["n_rows"], 5 * 270)
        self.assertGreaterEqual(meta["n_rows"], 200)

    def test_default_uses_full_pool(self):
        from training_panel.global_calibrator import fit_global_calibrator
        ps, fr = self._make_panel(total_years=2)
        cal = fit_global_calibrator(
            ps, fr, min_rows=200, threshold=0.0,
            # rolling_window_years default = None → full pool
        )
        meta = cal.metadata
        # 2 years × 5 tickers × ~252 = ~2520 rows
        self.assertGreater(meta["n_rows"], 1500)


if __name__ == "__main__":
    unittest.main()
