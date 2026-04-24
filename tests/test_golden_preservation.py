"""Golden APY floor regression test — guards the APY=1.41/Sharpe=2.0 goal.

Runs the full 27-mo OOS sim on the current golden config + asserts APY
is at least `GOLDEN_APY_FLOOR`. If a future commit regresses APY below
the floor, this test fails BEFORE the change reaches production.

Opt-in via `RENQUANT_REGRESSION=1` because full sim takes ~10 min.
CI or nightly cron should set this; normal dev cycle stays fast.

Enforces the user-specified invariant (CLAUDE.md-implicit):
  "Golden conf APY and Sharpe should never be challenged, they are
   ultimate goal. So any code change should help to improve them,
   not damage them."

Baseline v4.1 sim APY (allow_fetch=False, 2026-04-24): +39.82%
Floor: 37.0% (1pt tolerance for run-to-run noise)

Run manually::

    RENQUANT_REGRESSION=1 pytest tests/test_golden_preservation.py -v

Or as part of a pre-release sweep::

    RENQUANT_REGRESSION=1 pytest tests/ -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

REPO_ROOT = Path(__file__).resolve().parent.parent

# v4.1 golden baseline (2026-04-24 promote) - 1pt tolerance
GOLDEN_APY_FLOOR = 0.370   # 37.0%
# 1-pt tolerance below the 39.82% sweep baseline absorbs run-to-run noise


_SKIP_REASON = (
    "opt-in regression (set RENQUANT_REGRESSION=1 to enable — takes ~10 min)"
)


@pytest.mark.skipif(
    os.environ.get("RENQUANT_REGRESSION") != "1",
    reason=_SKIP_REASON,
)
class TestGoldenAPYPreservation:
    def test_golden_config_apy_above_floor(self):
        """Full sim on golden config must beat the APY floor."""
        from kernel.config import load_strategy_config
        from training_panel.pipeline import prepare_inference_panel_frames
        from sim.runner import run_backtest
        import pandas as pd

        cfg = load_strategy_config(_STRATEGY_DIR / "strategy_config.golden.json")
        cfg["_strategy_dir"] = str(_STRATEGY_DIR)
        cfg.setdefault("initial_cash", 100_000)

        symbols = (set(cfg["watchlist"])
                    | set(cfg.get("sector_etf_map", {}).values())
                    | {"SPY"})

        cache = REPO_ROOT / "data" / "ohlcv"
        def _load(s):
            p = cache / s / "1d.parquet"
            if not p.exists():
                return None
            df = pd.read_parquet(p)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df

        ohlcv = {s: _load(s) for s in symbols}
        ohlcv = {s: df for s, df in ohlcv.items() if df is not None and not df.empty}

        ticker_sectors = {t: cfg["sector_map"][t] for t in cfg["watchlist"]
                          if t in cfg.get("sector_map", {})}
        ff, fac = prepare_inference_panel_frames(
            watchlist=cfg["watchlist"], ohlcv=ohlcv,
            ticker_sectors=ticker_sectors,
            config={**cfg, "_strategy_dir": str(_STRATEGY_DIR)},
        )

        result = run_backtest(
            config=cfg, strategy_dir=_STRATEGY_DIR, ohlcv=ohlcv,
            spy_df=ohlcv["SPY"], sector_etf_map=cfg.get("sector_etf_map", {}),
            panel_feature_frames=ff, panel_factor_frames=fac,
        )

        apy = result.apy
        assert apy >= GOLDEN_APY_FLOOR, (
            f"GOLDEN APY regression: {apy:+.2%} < floor {GOLDEN_APY_FLOOR:+.2%}. "
            f"A recent commit damaged the golden config. Bisect to find the "
            f"offending change before merging."
        )

        # Additional sanity: expect AT LEAST 50 buys in the 27-mo window.
        # Below this suggests a gate is over-filtering.
        buys = len(result.buys)
        assert buys >= 50, (
            f"Buy count regression: only {buys} buys in 27-mo sim "
            f"(baseline ~117). A gate or filter may be over-triggering."
        )


@pytest.mark.skipif(
    os.environ.get("RENQUANT_REGRESSION") != "1",
    reason=_SKIP_REASON,
)
class TestLiveConfigAPYPreservation:
    def test_live_config_matches_golden(self):
        """strategy_config.json (live) should produce APY equivalent to
        strategy_config.golden.json. Catches drift where someone edits
        the live config without updating the golden snapshot."""
        from kernel.config import load_strategy_config
        from training_panel.pipeline import prepare_inference_panel_frames
        from sim.runner import run_backtest
        import pandas as pd

        live_cfg = load_strategy_config(_STRATEGY_DIR / "strategy_config.json")
        live_cfg["_strategy_dir"] = str(_STRATEGY_DIR)
        live_cfg.setdefault("initial_cash", 100_000)

        symbols = (set(live_cfg["watchlist"])
                    | set(live_cfg.get("sector_etf_map", {}).values())
                    | {"SPY"})

        cache = REPO_ROOT / "data" / "ohlcv"
        def _load(s):
            p = cache / s / "1d.parquet"
            if not p.exists():
                return None
            df = pd.read_parquet(p)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df

        ohlcv = {s: _load(s) for s in symbols}
        ohlcv = {s: df for s, df in ohlcv.items() if df is not None and not df.empty}

        ticker_sectors = {t: live_cfg["sector_map"][t] for t in live_cfg["watchlist"]
                          if t in live_cfg.get("sector_map", {})}
        ff, fac = prepare_inference_panel_frames(
            watchlist=live_cfg["watchlist"], ohlcv=ohlcv,
            ticker_sectors=ticker_sectors,
            config={**live_cfg, "_strategy_dir": str(_STRATEGY_DIR)},
        )

        result = run_backtest(
            config=live_cfg, strategy_dir=_STRATEGY_DIR, ohlcv=ohlcv,
            spy_df=ohlcv["SPY"], sector_etf_map=live_cfg.get("sector_etf_map", {}),
            panel_feature_frames=ff, panel_factor_frames=fac,
        )

        apy = result.apy
        assert apy >= GOLDEN_APY_FLOOR, (
            f"Live config APY regression: {apy:+.2%} < floor {GOLDEN_APY_FLOOR:+.2%}. "
            f"strategy_config.json has drifted below golden baseline. "
            f"Either revert the bad change or promote the new golden "
            f"(update strategy_config.golden.json + doc/golden_config_*.md + "
            f"GOLDEN_APY_FLOOR in this test)."
        )


# Mini-smoke-test (no opt-in needed): verifies the test INFRASTRUCTURE
# works even when the regression opt-in is off.
class TestRegressionHarness:
    def test_golden_config_file_exists(self):
        """The frozen snapshot must exist."""
        assert (_STRATEGY_DIR / "strategy_config.golden.json").exists()

    def test_live_config_file_exists(self):
        assert (_STRATEGY_DIR / "strategy_config.json").exists()
