"""Invariant test — the current 104 config must NOT produce a systemic no-trade period.

User contract: "it's ok not to make trade, but systematically not making
trade is not acceptable". This test runs a full OOS sim with the current
strategy_config.json and asserts that no idle streak exceeds a hard
threshold. If any config change (panel retrain, universe tweak, tier
threshold bump) creates a 20+ day no-trade window, CI fails.

The test is gated behind an env var so it only runs in nightly /
pre-merge CI (not on every unit-test pass) — the full sim takes ~60s.

    RENQUANT_FULL_SIM=1 pytest tests/test_no_trade_invariant.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# Skip by default — opt in via env var for nightly runs
_RUN_FULL = os.environ.get("RENQUANT_FULL_SIM") == "1"
_MAX_NO_TRADE_DAYS = 20          # ~1 trading month = hard fail threshold


@pytest.mark.skipif(not _RUN_FULL, reason="Set RENQUANT_FULL_SIM=1 to run")
class TestNoTradeInvariant:
    """Must produce at least one trade every 20 consecutive trading days."""

    def test_baseline_sim_no_long_idle(self):
        from kernel.config import load_strategy_config
        from kernel.data import fetch_ohlcv
        from sim.runner import run_backtest

        cfg = load_strategy_config(_STRATEGY_DIR / "strategy_config.json")
        cfg["_strategy_dir"] = str(_STRATEGY_DIR)
        cfg.setdefault("initial_cash", 100_000)
        # Force panel off so we isolate the per-ticker path
        cfg["ranking"] = dict(cfg["ranking"])
        cfg["ranking"]["panel_scoring"] = dict(cfg["ranking"]["panel_scoring"])
        cfg["ranking"]["panel_scoring"]["enabled"] = False

        symbols = set(cfg["watchlist"]) | set(cfg.get("sector_etf_map", {}).values()) | {"SPY"}
        ohlcv = {s: fetch_ohlcv(s) for s in symbols}
        ohlcv = {s: df for s, df in ohlcv.items() if df is not None and not df.empty}

        result = run_backtest(
            config=cfg, strategy_dir=_STRATEGY_DIR,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"],
            sector_etf_map=cfg.get("sector_etf_map", {}),
        )
        assert result.longest_no_trade_streak < _MAX_NO_TRADE_DAYS, (
            f"Baseline sim produced a {result.longest_no_trade_streak}-day "
            f"no-trade streak (threshold {_MAX_NO_TRADE_DAYS}). "
            f"first_trade_date={result.first_trade_date}"
        )

    def test_panel_sim_no_long_idle(self):
        from kernel.config import load_strategy_config
        from kernel.data import fetch_ohlcv
        from sim.runner import run_backtest
        from training_panel.pipeline import prepare_inference_panel_frames

        cfg = load_strategy_config(_STRATEGY_DIR / "strategy_config.json")
        cfg["_strategy_dir"] = str(_STRATEGY_DIR)
        cfg.setdefault("initial_cash", 100_000)

        symbols = set(cfg["watchlist"]) | set(cfg.get("sector_etf_map", {}).values()) | {"SPY"}
        ohlcv = {s: fetch_ohlcv(s) for s in symbols}
        ohlcv = {s: df for s, df in ohlcv.items() if df is not None and not df.empty}

        ff, fac = prepare_inference_panel_frames(
            watchlist=cfg["watchlist"], ohlcv=ohlcv,
            ticker_sectors={t: cfg["sector_map"][t] for t in cfg["watchlist"]
                            if t in cfg.get("sector_map", {})},
            config=cfg,
        )
        result = run_backtest(
            config=cfg, strategy_dir=_STRATEGY_DIR,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"],
            sector_etf_map=cfg.get("sector_etf_map", {}),
            panel_feature_frames=ff, panel_factor_frames=fac,
        )
        assert result.longest_no_trade_streak < _MAX_NO_TRADE_DAYS, (
            f"Panel sim produced a {result.longest_no_trade_streak}-day "
            f"no-trade streak (threshold {_MAX_NO_TRADE_DAYS}). "
            f"first_trade_date={result.first_trade_date}. "
            f"Root cause is likely calibrator mis-alignment or tier thresholds "
            f"too tight for the current panel's score distribution."
        )
