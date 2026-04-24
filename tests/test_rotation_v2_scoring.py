"""Rotation V2 — μ−λσ scoring mode.

Flag `rotation.scoring_mode = "mu_minus_lambda_sigma"` replaces the
calibrated ER with a direct NGBoost-derived driver:

    score = μ − λσ     (λ defaults to 1.0, override via rotation.lambda_)

Bypasses the isotonic calibrator (which may flatten the tails on
conviction). Falls back to expected_return for any holding / candidate
missing μ or σ — mixed panels still work.

These tests exercise the task-level override (task_rotation.py) plus
the fallback. The kernel primitive itself stays untouched.
"""
from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _ctx_with_one_held(held_attrs, cand_attrs, **config_overrides):
    """Build a minimal InferenceContext-ish fixture for BuildPairsTask.

    BuildPairsTask reads many fields; we stub the ones it needs.
    """
    from kernel.exits import HoldingState

    entry_price = held_attrs.get("entry_price", 100.0)
    hs = HoldingState(
        entry_price    = entry_price,
        entry_date     = held_attrs.get("entry_date",
                                         datetime.date(2024, 1, 1)),
        high_watermark = entry_price,
        shares         = 100,
    )
    hs.rank_score       = held_attrs.get("rank_score", 0.30)
    hs.expected_return  = held_attrs.get("expected_return", -0.02)
    hs.mu               = held_attrs.get("mu", None)
    hs.sigma            = held_attrs.get("sigma", None)
    hs.kelly_target_pct = held_attrs.get("kelly_target", 0.10)
    held_ticker = held_attrs["ticker"]

    cand = SimpleNamespace(
        ticker          = cand_attrs["ticker"],
        rank_score      = cand_attrs.get("rank_score", 0.50),
        expected_return = cand_attrs.get("expected_return", 0.03),
        mu              = cand_attrs.get("mu", None),
        sigma           = cand_attrs.get("sigma", None),
        panel_score     = 0.5,
        kelly_target_pct = cand_attrs.get("kelly_target", 0.10),
    )

    default_config = {
        "rotation": {
            "enabled":                     True,
            "min_expected_advantage_pct":  0.01,
            "target_horizon_days":         20,
            "transaction_cost_pct":        0.0,
            "min_rotation_hold_days":      0,
            "lt_protection_days":          0,
            "max_rotations_per_bar":       2,
        },
        "tax": {},
        "ranking": {"kelly_sizing": {}},
    }
    for k, v in config_overrides.items():
        # Merge top-level only
        if k in default_config and isinstance(default_config[k], dict):
            default_config[k].update(v)
        else:
            default_config[k] = v

    ctx = SimpleNamespace(
        config   = default_config,
        today    = datetime.date(2025, 6, 1),
        ranked   = [cand],
        holdings = {held_ticker: hs},
        bear_only = False,
        exits    = [],
        prices   = {held_ticker: held_attrs.get("entry_price", 100.0)},
        counters = {},
        rotations = [],
        prior_rotation_proposals = [],
    )
    return ctx


class TestV2ScoringMode:
    def test_mu_minus_sigma_prefers_lower_sigma(self):
        """With ER equal but μ/σ different, μ−λσ should pick the higher-μ."""
        from kernel.pipeline.task_rotation import BuildPairsTask

        # Held: μ=0.00, σ=0.05 → drive_score = -0.05
        # Cand: μ=0.08, σ=0.02 → drive_score = +0.06
        # raw_adv_v2 = 0.06 - (-0.05) = 0.11  (huge gap, gets proposed)
        ctx = _ctx_with_one_held(
            held_attrs = {"ticker": "NVDA", "mu": 0.00, "sigma": 0.05,
                          "expected_return": 0.0,
                          "entry_price": 100.0},
            cand_attrs = {"ticker": "AMD",  "mu": 0.08, "sigma": 0.02,
                          "expected_return": 0.0},
            rotation  = {"scoring_mode": "mu_minus_lambda_sigma",
                         "lambda_": 1.0},
        )
        BuildPairsTask().run(ctx)
        assert len(ctx.rotations) == 1
        assert ctx.rotations[0].sell_ticker == "NVDA"
        assert ctx.rotations[0].buy_ticker  == "AMD"

    def test_falls_back_to_er_when_mu_missing(self):
        """Missing μ → fall back to expected_return; behave like default."""
        from kernel.pipeline.task_rotation import BuildPairsTask

        # μ/σ NOT set on either side → fallback path
        ctx = _ctx_with_one_held(
            held_attrs = {"ticker": "NVDA", "expected_return": 0.0,
                          "entry_price": 100.0},
            cand_attrs = {"ticker": "AMD",  "expected_return": 0.05},
            rotation  = {"scoring_mode": "mu_minus_lambda_sigma",
                         "lambda_": 1.0,
                         "min_expected_advantage_pct": 0.03},
        )
        BuildPairsTask().run(ctx)
        # raw_adv = 0.05 - 0.0 = 0.05 ≥ 0.03 → fires via fallback
        assert len(ctx.rotations) == 1

    def test_default_er_mode_unchanged(self):
        """Default scoring_mode='er' → no μ−λσ computed; uses ER as before."""
        from kernel.pipeline.task_rotation import BuildPairsTask

        # Cand has great μ−λσ edge but weak ER → ER-mode shouldn't fire.
        ctx = _ctx_with_one_held(
            held_attrs = {"ticker": "NVDA", "mu": 0.0, "sigma": 0.0,
                          "expected_return": 0.05,
                          "entry_price": 100.0},
            cand_attrs = {"ticker": "AMD",  "mu": 0.5, "sigma": 0.01,
                          "expected_return": 0.05},
            # default scoring_mode not overridden → stays "er"
        )
        BuildPairsTask().run(ctx)
        # raw_adv_er = 0.05 - 0.05 = 0.0 < 0.01 threshold → no pair
        assert ctx.rotations == []

    def test_lambda_scaling_changes_outcome(self):
        """High λ penalises σ; with large σ the cand should LOSE the comparison."""
        from kernel.pipeline.task_rotation import BuildPairsTask

        # Cand μ=0.06 but σ=0.15 → with λ=1: μ−λσ = -0.09 (bad)
        #                      with λ=0.1: μ−λσ = +0.045 (good)
        # Held: μ=0.02, σ=0.02 → λ=1: 0.0, λ=0.1: +0.018
        # At λ=1.0: raw = -0.09 - 0.0 = -0.09 (blocked)
        ctx_high_lambda = _ctx_with_one_held(
            held_attrs = {"ticker": "NVDA", "mu": 0.02, "sigma": 0.02,
                          "entry_price": 100.0, "expected_return": 0.0},
            cand_attrs = {"ticker": "AMD",  "mu": 0.06, "sigma": 0.15,
                          "expected_return": 0.0},
            rotation  = {"scoring_mode": "mu_minus_lambda_sigma",
                         "lambda_": 1.0,
                         "min_expected_advantage_pct": 0.01},
        )
        BuildPairsTask().run(ctx_high_lambda)
        assert ctx_high_lambda.rotations == []

        # At λ=0.1: raw_adv = 0.045 - 0.018 = 0.027 ≥ 0.01 → fires
        ctx_low_lambda = _ctx_with_one_held(
            held_attrs = {"ticker": "NVDA", "mu": 0.02, "sigma": 0.02,
                          "entry_price": 100.0, "expected_return": 0.0},
            cand_attrs = {"ticker": "AMD",  "mu": 0.06, "sigma": 0.15,
                          "expected_return": 0.0},
            rotation  = {"scoring_mode": "mu_minus_lambda_sigma",
                         "lambda_": 0.1,
                         "min_expected_advantage_pct": 0.01},
        )
        BuildPairsTask().run(ctx_low_lambda)
        assert len(ctx_low_lambda.rotations) == 1
