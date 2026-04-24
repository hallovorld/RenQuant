"""Tests for scripts/screen_watchlist.py.

Can't cleanly test the network-fetch path; mock parquet cache with
pytest tmp_path instead.
"""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _seed_parquet(cache_root: Path, ticker: str, closes: list[float],
                  start: datetime.date) -> None:
    import pandas as pd
    dates = pd.bdate_range(start=start, periods=len(closes))
    df = pd.DataFrame({
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1_000_000] * len(closes),
    }, index=dates)
    df.index.name = "Date"
    out = cache_root / ticker / "1d.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)


def _seed_strategy_config(strategy_dir: Path, watchlist: list[str]) -> None:
    import json
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "strategy_config.json").write_text(json.dumps({
        "watchlist": watchlist,
        "defensive_tickers": ["GLD"],
    }))


class TestEndToEnd:
    """Run the script with a mocked cache and verify it produces a report."""

    def test_produces_markdown_report(self, tmp_path, monkeypatch):
        # Build a fake data/ohlcv/ + backtesting/renquant_test/strategy_config.json
        cache_root = tmp_path / "ohlcv"
        # SPY — modest return
        _seed_parquet(cache_root, "SPY",
                      [100 + i * 0.3 for i in range(200)],
                      datetime.date(2025, 10, 1))
        # Watchlist — one good, one bad
        _seed_parquet(cache_root, "NVDA",   # strong uptrend
                      [100 + i * 1.5 for i in range(200)],
                      datetime.date(2025, 10, 1))
        _seed_parquet(cache_root, "WEAK",   # flat/negative
                      [100 - i * 0.2 for i in range(200)],
                      datetime.date(2025, 10, 1))
        # Non-watchlist strong add candidate
        _seed_parquet(cache_root, "ADDTKR",
                      [100 + i * 1.0 for i in range(200)],
                      datetime.date(2025, 10, 1))

        strategy_dir = tmp_path / "backtesting" / "renquant_test"
        _seed_strategy_config(strategy_dir, ["NVDA", "WEAK"])

        # Run the script with cache-root + strategy-dir-root overrides
        script = REPO_ROOT / "scripts" / "screen_watchlist.py"
        result = subprocess.run(
            [sys.executable, str(script),
             "--strategy", "renquant_test",
             "--strategy-dir-root", str(tmp_path),
             "--cache-root", str(cache_root),
             "--lookback-days", "180"],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        assert result.returncode == 0, result.stderr

        # Output should contain drop/add language
        output = result.stdout + result.stderr
        assert "drops=" in output or "adds=" in output
        # Report file should exist
        report_dir = tmp_path / "logs" / "watchlist_screen"
        reports = list(report_dir.glob("*.md"))
        assert len(reports) == 1, f"expected 1 report, found {reports}"
        text = reports[0].read_text()
        assert "Watchlist screen" in text
        assert "Drop candidates" in text
        assert "Add candidates" in text


class TestPerfStats:
    def test_sharpe_positive_on_uptrend(self, tmp_path):
        """Inject the script as a module to hit _perf_stats directly."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_screen", REPO_ROOT / "scripts" / "screen_watchlist.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import pandas as pd
        # Strong uptrend → positive Sharpe
        idx = pd.bdate_range(start="2025-10-01", periods=100)
        closes = pd.Series([100 + i for i in range(100)], index=idx)
        stats = mod._perf_stats(closes)
        assert stats["total_return"] > 0
        assert stats["sharpe"] > 0

    def test_sharpe_negative_on_downtrend(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_screen", REPO_ROOT / "scripts" / "screen_watchlist.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import pandas as pd
        idx = pd.bdate_range(start="2025-10-01", periods=100)
        closes = pd.Series([100 - i for i in range(100)], index=idx)
        stats = mod._perf_stats(closes)
        assert stats["total_return"] < 0
        assert stats["sharpe"] < 0
