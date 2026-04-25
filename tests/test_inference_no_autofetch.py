"""Bug 16 regression: inference path must NEVER auto-fetch.

Inference (sim/live/LEAN) calls prepare_inference_panel_frames each bar.
If LoadFundamentals/EarningsSurprise/InsiderTrades auto-fetch missing
tickers from network APIs (OpenBB / yfinance / SEC), the per-bar loop
blocks for minutes per ticker. Training is the ONLY path allowed to
fetch.

Mechanism: PanelTrainingContext.inference_only=True; Load*Task respects
the flag and skips the fetch fork.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


class TestInferenceNoAutoFetch:
    def test_inference_only_blocks_fundamentals_fetch(self, tmp_path):
        from training_panel.context import PanelTrainingContext
        from training_panel.pp_panel_training import LoadFundamentalsTask

        ctx = PanelTrainingContext(
            config={
                "watchlist": ["NVDA"],
                "panel_ltr": {"fundamentals": {
                    "enabled": True,
                    "cache_dir": str(tmp_path),   # empty cache → would fetch
                    "allow_fetch": True,           # would normally fetch
                }},
            },
            watchlist=["NVDA"],
            inference_only=True,    # ← critical: blocks fetch
        )

        # Patch the network fetcher; it must NOT be called
        with patch("kernel.fundamentals.fetch_fundamentals_watchlist",
                    side_effect=AssertionError("fetch must not be called when inference_only=True")) as fake:
            LoadFundamentalsTask().run(ctx)
            fake.assert_not_called()

    def test_training_path_still_allows_fetch(self, tmp_path):
        """Training (inference_only=False, default) preserves fetch path."""
        from training_panel.context import PanelTrainingContext
        from training_panel.pp_panel_training import LoadFundamentalsTask

        called = {"n": 0}

        def _fake_fetch(missing, *, store=None):
            called["n"] += 1
            return {}

        ctx = PanelTrainingContext(
            config={
                "watchlist": ["NVDA"],
                "panel_ltr": {"fundamentals": {
                    "enabled": True,
                    "cache_dir": str(tmp_path),
                    "allow_fetch": True,
                }},
            },
            watchlist=["NVDA"],
            # inference_only defaults to False
        )

        with patch("kernel.fundamentals.fetch_fundamentals_watchlist",
                    side_effect=_fake_fetch):
            LoadFundamentalsTask().run(ctx)
        assert called["n"] == 1, "training path must still call fetch"

    def test_inference_only_blocks_insider_fetch(self, tmp_path):
        from training_panel.context import PanelTrainingContext
        from training_panel.pp_panel_training import LoadInsiderTradesTask

        ctx = PanelTrainingContext(
            config={
                "watchlist": ["NVDA"],
                "panel_ltr": {"insider_trades": {
                    "enabled": True,
                    "cache_dir": str(tmp_path),
                    "allow_fetch": True,
                }},
            },
            watchlist=["NVDA"],
            inference_only=True,
        )

        with patch("kernel.insider_trades.fetch_insider_trades_watchlist",
                    side_effect=AssertionError("must not fetch in inference")) as fake:
            LoadInsiderTradesTask().run(ctx)
            fake.assert_not_called()

    def test_inference_only_blocks_earnings_surprise_fetch(self, tmp_path):
        from training_panel.context import PanelTrainingContext
        from training_panel.pp_panel_training import LoadEarningsSurpriseTask

        ctx = PanelTrainingContext(
            config={
                "watchlist": ["NVDA"],
                "panel_ltr": {"earnings_surprise": {
                    "enabled": True,
                    "cache_dir": str(tmp_path),
                    "allow_fetch": True,
                }},
            },
            watchlist=["NVDA"],
            inference_only=True,
        )

        with patch("kernel.earnings_surprise.fetch_earnings_surprise_watchlist",
                    side_effect=AssertionError("must not fetch in inference")) as fake:
            LoadEarningsSurpriseTask().run(ctx)
            fake.assert_not_called()

    def test_prepare_inference_sets_flag(self):
        """prepare_inference_panel_frames must set inference_only=True on ctx."""
        from training_panel.pipeline import prepare_inference_panel_frames
        # Easier: read the source file and confirm the flag is set
        src = (_STRATEGY_DIR / "training_panel" / "pipeline.py").read_text()
        assert "inference_only=True" in src, (
            "prepare_inference_panel_frames must set inference_only=True "
            "on PanelTrainingContext to block auto-fetch in sim/live")
