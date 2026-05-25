"""Phase 3 of 2026-05-15 calibrator P0 — μ/σ wiring tests.

Phase 3 wires the calibrator's expected_return head into c.mu (so Kelly
has a real μ when NGBoost is off) and adds a realized-vol fallback for
c.sigma. Both opt-in via config flags so prod behavior is unchanged
until the operator flips them.

Tests pin:
  A. use_calibrator_mu=False (default) → c.mu stays None
  B. use_calibrator_mu=True → c.mu = c.expected_return (finite values only)
  C. use_realized_vol_fallback=False (default) → c.sigma stays None
  D. use_realized_vol_fallback=True → c.sigma = trailing 60d annualized
     stdev, clipped to [floor, ceiling]
  E. Both flags on + Kelly enabled → kelly_target_pct non-zero and
     non-uniform across candidates (the real prod-readiness check)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _make_calibrator(prob_y, er_y):
    """Return a mock calibrator. prob_y / er_y indexed by candidate id."""
    return SimpleNamespace(
        calibrate_probability=lambda s: prob_y[int(s)],
        expected_return     =lambda s: er_y[int(s)],
        metadata            ={"expected_return_label_contract": "raw_return_units_required"},
    )


def _make_ohlcv_for(ticker, sigma_target):
    """Synthesize a 100-day OHLCV df whose pct_change std produces
    a target annualized vol."""
    daily_sigma = sigma_target / math.sqrt(252.0)
    # Deterministic pattern with sample std exactly equal to daily_sigma.
    # Do not use Python hash/random here: xdist workers randomize hash seeds,
    # making "identical sigma" tests flaky and scientifically false.
    pattern = np.tile([-1.0, 1.0], 50)
    pattern = (pattern - pattern.mean()) / pattern.std(ddof=1)
    rets = pattern * daily_sigma
    closes = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": closes})


def _make_ctx(*, n=5, use_cal_mu=False, use_vol_fallback=False,
              er_values=None, sigma_targets=None):
    if er_values is None:
        er_values = [0.02 * (i - 2) for i in range(n)]  # -0.04 to +0.04
    if sigma_targets is None:
        sigma_targets = [0.20 + 0.05 * i for i in range(n)]  # 20-40%

    prob_y = [0.30 + 0.10 * i for i in range(n)]  # 0.30 .. 0.70

    cands = []
    ohlcv = {}
    for i in range(n):
        ticker = f"T{i:02d}"
        c = SimpleNamespace(
            ticker=ticker,
            panel_score=float(i),  # encode index for mock calibrator dispatch
            rank_score=None,
            expected_return=None,
            mu=None, sigma=None,
            kelly_target_pct=0.0,
        )
        cands.append(c)
        ohlcv[ticker] = _make_ohlcv_for(ticker, sigma_targets[i])

    cfg = {
        "ranking": {
            "panel_scoring": {
                "global_calibration": {"enabled": True},
            },
            "kelly_sizing": {
                "enabled": True,
                "use_calibrator_mu": use_cal_mu,
                "use_realized_vol_fallback": use_vol_fallback,
                "fractional": 0.50,
                "max_concentration": 0.35,
            },
        },
        "regime_params": {"BULL_CALM": {"max_position_pct": 0.20}},
    }
    return SimpleNamespace(
        config=cfg,
        candidates=cands,
        holdings={},
        regime="BULL_CALM",
        confidence=1.0,
        ohlcv=ohlcv,
        _global_calibrator=_make_calibrator(prob_y, er_values),
        _regime_calibrators=None,
    )


class TestCalibratorMuWiring:
    """Phase 3-A,B: c.mu = c.expected_return wiring."""

    def test_default_off_leaves_mu_none(self):
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
        )
        ctx = _make_ctx(use_cal_mu=False, er_values=[0.05] * 5)
        ApplyGlobalCalibrationTask().run(ctx)
        for c in ctx.candidates:
            assert c.expected_return == 0.05, (
                f"expected_return should be set; got {c.expected_return}"
            )
            assert c.mu is None, (
                f"with use_calibrator_mu=False, c.mu must remain None "
                f"(got {c.mu} for {c.ticker})"
            )

    def test_flag_on_wires_mu_to_expected_return(self):
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
        )
        ers = [-0.04, -0.02, 0.0, +0.02, +0.04]
        ctx = _make_ctx(use_cal_mu=True, er_values=ers)
        ApplyGlobalCalibrationTask().run(ctx)
        for c, expected_er in zip(ctx.candidates, ers):
            assert c.mu == expected_er, (
                f"{c.ticker}: c.mu={c.mu} ≠ expected_return={expected_er}"
            )
            assert c.expected_return == expected_er

    def test_flag_on_blocks_non_raw_return_calibrator(self):
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
        )
        ctx = _make_ctx(use_cal_mu=True, er_values=[0.05] * 5)
        ctx._global_calibrator.metadata = {
            "expected_return_label_contract": "gaussianized_rank_label",
        }

        ok = ApplyGlobalCalibrationTask().run(ctx)

        assert ok is False
        assert ctx.candidates == []
        assert ctx.buy_blocked is True
        assert ctx.skip_buys is True
        assert ctx._calibrator_contract_failed is True

    def test_nan_expected_return_does_not_set_mu(self):
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
        )
        # 3 candidates: one NaN er, two finite
        ers = [float("nan"), 0.03, 0.05]
        ctx = _make_ctx(use_cal_mu=True, n=3, er_values=ers)
        ApplyGlobalCalibrationTask().run(ctx)
        # Finite ers wire to mu; NaN er should leave mu None
        assert math.isnan(ctx.candidates[0].expected_return)
        assert ctx.candidates[0].mu is None
        assert ctx.candidates[1].mu == 0.03
        assert ctx.candidates[2].mu == 0.05


class TestRealizedVolFallback:
    """Phase 3-C,D: σ fallback when NGBoost off."""

    def test_default_off_leaves_sigma_none(self):
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyRealizedVolFallbackTask,
        )
        ctx = _make_ctx(use_vol_fallback=False)
        ApplyRealizedVolFallbackTask().run(ctx)
        for c in ctx.candidates:
            assert c.sigma is None

    def test_flag_on_fills_sigma_from_ohlcv(self):
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyRealizedVolFallbackTask,
        )
        sigma_targets = [0.15, 0.25, 0.40]
        ctx = _make_ctx(n=3, use_vol_fallback=True,
                         sigma_targets=sigma_targets)
        ApplyRealizedVolFallbackTask().run(ctx)
        for c, target in zip(ctx.candidates, sigma_targets):
            assert c.sigma is not None, f"{c.ticker}: sigma still None"
            # Allow generous tolerance — random sampling adds noise
            assert 0.5 * target <= c.sigma <= 2.0 * target, (
                f"{c.ticker}: sigma={c.sigma} ≉ target={target}"
            )

    def test_clip_floor_and_ceiling(self):
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyRealizedVolFallbackTask,
        )
        ctx = _make_ctx(n=2, use_vol_fallback=True,
                         sigma_targets=[0.01, 5.0])  # below floor + above ceiling
        ctx.config["ranking"]["kelly_sizing"]["realized_vol_floor"] = 0.05
        ctx.config["ranking"]["kelly_sizing"]["realized_vol_ceiling"] = 1.50
        ApplyRealizedVolFallbackTask().run(ctx)
        for c in ctx.candidates:
            assert 0.05 - 1e-9 <= c.sigma <= 1.50 + 1e-9, (
                f"sigma {c.sigma} outside [0.05, 1.50]"
            )

    def test_ngboost_sigma_not_overwritten(self):
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyRealizedVolFallbackTask,
        )
        ctx = _make_ctx(n=2, use_vol_fallback=True)
        # Pretend NGBoost already populated sigma
        ctx.candidates[0].sigma = 0.111
        ctx.candidates[1].sigma = None
        ApplyRealizedVolFallbackTask().run(ctx)
        # NGBoost's sigma preserved
        assert ctx.candidates[0].sigma == 0.111
        # Fallback fired on the None one
        assert ctx.candidates[1].sigma is not None


class TestKellyEnabledEndToEnd:
    """Phase 3-E: with both flags on + Kelly enabled, kelly_target_pct
    becomes non-zero AND non-uniform — the prod-readiness check."""

    def test_both_flags_on_kelly_produces_non_uniform_targets(self):
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
            ApplyRealizedVolFallbackTask,
            ApplyKellySizingTask,
        )
        # 5 candidates: small μ values + high σ so Kelly raw f*=μ/σ² stays
        # well below the regime max_pct cap (0.20) and per-candidate μ
        # differences are visible in the final target.
        ers    = [-0.01, 0.0, 0.005, 0.01, 0.02]   # ≤ ±2% expected return
        sigmas = [0.60, 0.60, 0.60, 0.60, 0.60]    # 60% σ → Kelly raw small
        # Kelly raw = 0.02 / 0.36 ≈ 0.056; fractional 0.5 ≈ 0.028
        # → no cap binding → ranking by μ visible
        ctx = _make_ctx(n=5, use_cal_mu=True, use_vol_fallback=True,
                         er_values=ers, sigma_targets=sigmas)

        ApplyGlobalCalibrationTask().run(ctx)
        ApplyRealizedVolFallbackTask().run(ctx)
        ApplyKellySizingTask().run(ctx)

        targets = [c.kelly_target_pct for c in ctx.candidates]
        # Pre-fix: all targets would be 0 (mu_none). Post-fix: positive
        # μ → positive kelly_target.
        positive_targets = [t for t in targets if t > 0]
        assert len(positive_targets) >= 3, (
            f"Phase 3 wiring should produce ≥3 non-zero kelly targets "
            f"with positive μ; got targets={targets}"
        )
        # Non-uniformity: top-3 (er=0.005, 0.01, 0.02 with identical σ)
        # should produce strictly increasing kelly_target_pct under
        # f*=μ/σ² + identical fractional + identical caps.
        t3, t4, t5 = targets[2], targets[3], targets[4]
        assert t3 < t4 < t5, (
            f"Kelly with identical σ should rank by μ: "
            f"er=0.005→{t3}, er=0.01→{t4}, er=0.02→{t5}"
        )
        # All capped by max_position_pct (0.20) AND max_concentration (0.35)
        for t in targets:
            assert t <= 0.20 + 1e-9, f"kelly_target {t} exceeds max_pct 0.20"

    def test_flags_off_kelly_returns_zero_everywhere(self):
        """Regression-pin the pre-fix behavior."""
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
            ApplyRealizedVolFallbackTask,
            ApplyKellySizingTask,
        )
        ers = [-0.02, 0.00, 0.05, 0.10, 0.15]
        ctx = _make_ctx(n=5, use_cal_mu=False, use_vol_fallback=False,
                         er_values=ers)
        ApplyGlobalCalibrationTask().run(ctx)
        ApplyRealizedVolFallbackTask().run(ctx)
        ApplyKellySizingTask().run(ctx)
        # Without wiring, mu=None for all → Kelly returns 0 everywhere
        for c in ctx.candidates:
            assert c.kelly_target_pct == 0.0, (
                f"Kelly should return 0 with use_calibrator_mu=False; "
                f"got {c.kelly_target_pct} for {c.ticker}"
            )
