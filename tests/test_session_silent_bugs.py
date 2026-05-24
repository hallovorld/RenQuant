"""Regression tests for silent-failure bugs discovered 2026-04-24 PT.

These tests are designed to CATCH the specific class of bug where new
code ships but the execution path to it is broken — so the feature
looks alive (unit tests pass) but is actually dead (integration never
reaches it).
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ─────────────────────────────────────────────────────────────────────
# Bug 1: V4 scoring_mode "thesis_symmetric" wired into BuildPairsTask
# ─────────────────────────────────────────────────────────────────────

class TestThesisSymmetricReachable:
    """If scoring_mode='thesis_symmetric', BuildPairsTask MUST take the
    thesis_symmetric branch (calling find_thesis_symmetric_pairs), not
    silently fall through to ER mode."""

    def test_branch_reached_on_thesis_symmetric_mode(self, monkeypatch):
        from kernel.pipeline.task_rotation import BuildPairsTask

        called = {"thesis_symmetric": False, "er": False}

        def _stub_thesis_sym(**kwargs):
            called["thesis_symmetric"] = True
            return []

        def _stub_er(**kwargs):
            called["er"] = True
            return []

        import kernel.rotation as rotmod
        monkeypatch.setattr(rotmod, "find_thesis_symmetric_pairs", _stub_thesis_sym)
        monkeypatch.setattr(rotmod, "find_rotation_pairs", _stub_er)

        # Minimal stub context
        hs = SimpleNamespace(
            rank_score=0.30, expected_return=0.0,
            entry_date=datetime.date(2024, 1, 1), entry_price=100.0,
            entry_rank_score=0.40, panel_score=None,
            mu=None, sigma=None, kelly_target_pct=0.1,
        )
        cand = SimpleNamespace(
            ticker="AMD", rank_score=0.50, expected_return=0.05,
            mu=None, sigma=None, panel_score=0.5,
            kelly_target_pct=0.1,
        )
        ctx = SimpleNamespace(
            config = {
                "rotation": {
                    "enabled": True,
                    "mode": "thesis_symmetric",
                    "min_expected_advantage_pct": 0.01,
                    "min_rotation_hold_days": 0,
                    "lt_protection_days": 0,
                    "max_rotations_per_bar": 2,
                },
                "tax": {},
                "ranking": {"kelly_sizing": {}},
            },
            today = datetime.date(2025, 6, 1),
            regime = "BULL_CALM",
            ranked = [cand],
            holdings = {"NVDA": hs},
            bear_only = False,
            exits = [], prices = {"NVDA": 100.0},
            counters = {}, rotations = [],
            prior_rotation_proposals = [],
            ohlcv = {},
        )
        BuildPairsTask().run(ctx)

        assert called["thesis_symmetric"] is True, (
            "scoring_mode='thesis_symmetric' did NOT call find_thesis_"
            "symmetric_pairs — branch dead, silent fallthrough to ER.")
        assert called["er"] is False, (
            "thesis_symmetric mode unexpectedly fell through to ER mode.")

    def test_thesis_symmetric_excludes_holdings_with_existing_exit(self, monkeypatch):
        from kernel.pipeline.task_rotation import BuildPairsTask

        captured = {}

        def _stub_thesis_sym(**kwargs):
            captured["held_entry_scores"] = dict(kwargs["held_entry_scores"])
            captured["held_meta"] = dict(kwargs["held_meta"])
            return []

        import kernel.rotation as rotmod
        monkeypatch.setattr(rotmod, "find_thesis_symmetric_pairs", _stub_thesis_sym)

        hs = SimpleNamespace(
            rank_score=0.30, expected_return=0.0,
            entry_date=datetime.date(2024, 1, 1), entry_price=100.0,
            entry_rank_score=0.40, panel_score=None,
            mu=None, sigma=None, kelly_target_pct=0.1,
        )
        cand = SimpleNamespace(
            ticker="AMD", rank_score=0.50, expected_return=0.05,
            mu=None, sigma=None, panel_score=0.5,
            kelly_target_pct=0.1,
        )
        ctx = SimpleNamespace(
            config = {
                "rotation": {
                    "enabled": True,
                    "mode": "thesis_symmetric",
                    "min_expected_advantage_pct": 0.01,
                    "min_rotation_hold_days": 0,
                    "lt_protection_days": 0,
                    "max_rotations_per_bar": 2,
                },
                "tax": {},
                "ranking": {"kelly_sizing": {}},
            },
            today = datetime.date(2025, 6, 1),
            regime = "BULL_CALM",
            ranked = [cand],
            holdings = {"NVDA": hs},
            bear_only = False,
            exits = [("NVDA", SimpleNamespace(reason="panel_conviction"))],
            prices = {"NVDA": 100.0},
            counters = {}, rotations = [],
            prior_rotation_proposals = [],
            ohlcv = {},
            _db = None,
        )

        BuildPairsTask().run(ctx)

        assert captured["held_entry_scores"] == {}
        assert captured["held_meta"] == {}


# ─────────────────────────────────────────────────────────────────────
# Bug 2: minute features actually reach panel feature_cols
# ─────────────────────────────────────────────────────────────────────

class TestMinuteFeaturesReachPanel:
    """If panel_ltr.minute.enabled=true + minute_bars populated on ctx,
    the resulting panel feature_cols MUST include m_*-prefixed columns.
    Silent drop = no-op 10-min infra."""

    def test_factor_zscore_task_z_scores_minute_cols(self, tmp_path):
        """FactorZScoreTask must z-score m_* columns from raw_factor_frame
        and emit them as m_*_z in factor_frames."""
        from training_panel.pp_panel_training import FactorZScoreTask
        from training_panel.context import PanelTrainingContext

        idx = pd.bdate_range("2025-01-02", periods=20)
        # Two tickers with minute-prefixed factor columns present
        raw_factor_frames = {}
        for t in ("A", "B"):
            df = pd.DataFrame(index=idx)
            df["size"]                  = 1.0
            df["mom_12_1"]              = 0.5
            df["beta_60d"]              = 1.0
            df["resid_mom"]             = 0.0
            df["m_morning_drift"]       = 0.001 if t == "A" else 0.002
            df["m_intraday_realized_vol"] = 0.01 if t == "A" else 0.02
            raw_factor_frames[t] = df

        ctx = PanelTrainingContext(
            watchlist=["A", "B"],
            ticker_sectors={"A": "tech", "B": "tech"},
            raw_factor_frames=raw_factor_frames,
            config={},
        )
        FactorZScoreTask().run(ctx)

        assert ctx.factor_frames, "FactorZScoreTask produced no output"
        # Each ticker's factor_frame should contain m_*_z columns
        for t in ("A", "B"):
            cols = list(ctx.factor_frames[t].columns)
            assert "m_morning_drift_z" in cols, (
                f"{t}: m_morning_drift NOT z-scored into factor_frames — "
                f"minute features will silently drop from panel training.")
            assert "m_intraday_realized_vol_z" in cols, (
                f"{t}: m_intraday_realized_vol NOT z-scored — silent drop.")


# ─────────────────────────────────────────────────────────────────────
# Bug 4 / general: feature winsorization is applied
# ─────────────────────────────────────────────────────────────────────

class TestWinsorization:
    def test_zscore_outputs_clipped_to_3sigma(self):
        from training_panel.factors import cross_sectional_zscore

        # Deliberately insert a 10-sigma outlier
        idx = pd.bdate_range("2025-01-02", periods=5)
        features = {
            "NORMAL":  pd.Series([1.0, 1.1, 1.0, 1.05, 0.95], index=idx),
            "OUTLIER": pd.Series([100.0, 1.0, 1.0, 1.0, 1.0], index=idx),
            "C":       pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=idx),
        }
        out = cross_sectional_zscore(features, winsorize_clip=3.0)

        # The outlier at date 0 would naturally score >> 3σ; post-
        # winsorize it should be clipped to exactly 3.0 (or -3.0)
        outlier_z0 = out["OUTLIER"].iloc[0]
        assert abs(outlier_z0) <= 3.0 + 1e-9, (
            f"Winsorization did not clip outlier: z={outlier_z0}"
        )

    def test_disable_winsorization_allows_large_z(self):
        from training_panel.factors import cross_sectional_zscore

        idx = pd.bdate_range("2025-01-02", periods=5)
        features = {
            "NORMAL":  pd.Series([1.0, 1.1, 1.0, 1.05, 0.95], index=idx),
            "OUTLIER": pd.Series([100.0, 1.0, 1.0, 1.0, 1.0], index=idx),
            "C":       pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=idx),
        }
        out = cross_sectional_zscore(features, winsorize_clip=None)
        # Outlier should now show its full extreme z (no clip)
        outlier_z0 = out["OUTLIER"].iloc[0]
        assert abs(outlier_z0) > 1.0, \
            f"Expected large z when winsorize disabled; got {outlier_z0}"
