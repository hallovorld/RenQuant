"""Regression test for scripts/recalibrate_scores.py config-write safety.

Bug history (2026-04-22): recalibrate_scores.py read strategy_config.json at
start, did ~30 s of per-ticker work, then wrote the whole in-memory config
back — silently wiping any edit that landed in that window. The
defensive_tickers / confidence_veto_threshold fixes from commit 3c366b6
disappeared this way.

Fix: re-read the file immediately before writing and merge ONLY the two
fields this script owns (ranking.blend_updated, ranking.blend_n_symbols).

This test simulates a concurrent edit that lands during the work window
and asserts the edit survives. It would FAIL before the fix.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_minimal_config(config_path: Path) -> dict:
    config = {
        "watchlist":       [],
        "benchmark":       "SPY",
        "data_src":        "yfinance",
        "indicator_spec":  {},
        "model_params": {
            "feature_columns": [],
            "lookahead":       5,
            "threshold":       0.03,
        },
        "defensive_tickers": ["GLD", "TLT", "XLV", "XLU"],
        "ranking":           {"blend_updated": "2020-01-01"},
        "regime":            {"confidence_veto_threshold": 0.0},
    }
    config_path.write_text(json.dumps(config, indent=2))
    return config


class TestRecalibrateScoresConcurrentEdit:
    """Simulate: (1) script reads config, (2) another process edits config,
    (3) script writes — the other process's edit must NOT be lost.
    """

    def test_concurrent_edit_survives_write(self, tmp_path: Path, monkeypatch):
        # Build a fake strategy dir layout matching recalibrate_scores.REPO_ROOT.
        repo_root = tmp_path
        backtesting_dir = repo_root / "backtesting"
        strategy_dir = backtesting_dir / "renquant_test"
        (strategy_dir / "models").mkdir(parents=True)
        config_path = strategy_dir / "strategy_config.json"
        _write_minimal_config(config_path)

        import scripts.recalibrate_scores as rs  # noqa: PLC0415
        monkeypatch.setattr(rs, "REPO_ROOT", repo_root)

        # Inject a concurrent edit AFTER the script's initial config read
        # (line 165) and BEFORE its final write (line 275). The script
        # fetches benchmark OHLCV first thing after config load, so we
        # piggyback on a fake fetch_ohlcv: when it's first called, the
        # read has just happened, so we land the concurrent edit right
        # now and the script's final write path has to cope with it.
        import pandas as pd

        concurrent_done: list[bool] = []

        def fake_fetch(sym, provider="yfinance"):
            if not concurrent_done:
                concurrent_done.append(True)
                # "Another process" lands these edits on disk.
                latest = json.loads(config_path.read_text())
                latest["defensive_tickers"] = ["GLD", "NEW_TICKER"]
                latest["regime"]["confidence_veto_threshold"] = 0.99
                config_path.write_text(json.dumps(latest, indent=2))
            idx = pd.date_range("2024-01-01", periods=10, freq="B")
            return pd.DataFrame({
                "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                "volume": 1_000_000,
            }, index=idx)

        fake_kernel_data = type(sys)("kernel.data")
        fake_kernel_data.fetch_ohlcv = fake_fetch
        monkeypatch.setitem(sys.modules, "kernel.data", fake_kernel_data)

        fake_kernel_scoring = type(sys)("kernel.scoring")
        fake_kernel_scoring.ScoreCalibration = object
        fake_kernel_scoring.extract_raw_scores_bulk = lambda *a, **k: pd.Series([])
        monkeypatch.setitem(sys.modules, "kernel.scoring", fake_kernel_scoring)

        fake_training_scoring = type(sys)("training.scoring")
        fake_training_scoring.fit_probability_calibration = lambda *a, **k: None
        fake_training_scoring.fit_expected_return_calibration = lambda *a, **k: {}
        fake_training_scoring.raw_score_kind_for_model = lambda m: "raw"
        monkeypatch.setitem(sys.modules, "training.scoring", fake_training_scoring)

        # Run it. Empty watchlist → no per-ticker loop, jumps straight from
        # the benchmark fetch (where we inject the concurrent edit) to the
        # write path.
        rs.recalibrate("renquant_test", dry_run=False)

        assert concurrent_done, "concurrent edit helper didn't fire — test setup broken"

        final = json.loads(config_path.read_text())

        # The concurrent edit MUST survive:
        assert final["defensive_tickers"] == ["GLD", "NEW_TICKER"], (
            "concurrent defensive_tickers edit was wiped — race-condition fix is regressed"
        )
        assert final["regime"]["confidence_veto_threshold"] == 0.99, (
            "concurrent regime edit was wiped — race-condition fix is regressed"
        )

        # And the script's own two fields were still applied:
        assert final["ranking"]["blend_updated"] == str(date.today())
        assert final["ranking"]["blend_n_symbols"] == 0
        assert "blend_weights" not in final["ranking"]

    def test_dry_run_never_writes(self, tmp_path: Path, monkeypatch):
        repo_root = tmp_path
        (repo_root / "backtesting" / "renquant_test" / "models").mkdir(parents=True)
        config_path = repo_root / "backtesting" / "renquant_test" / "strategy_config.json"
        original = _write_minimal_config(config_path)
        mtime_before = config_path.stat().st_mtime

        import scripts.recalibrate_scores as rs  # noqa: PLC0415
        import pandas as pd
        monkeypatch.setattr(rs, "REPO_ROOT", repo_root)

        fake_kernel_data = type(sys)("kernel.data")
        fake_kernel_data.fetch_ohlcv = lambda *a, **k: pd.DataFrame({
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1],
        }, index=pd.date_range("2024-01-01", periods=1, freq="B"))
        monkeypatch.setitem(sys.modules, "kernel.data", fake_kernel_data)

        fake_kernel_scoring = type(sys)("kernel.scoring")
        fake_kernel_scoring.ScoreCalibration = object
        fake_kernel_scoring.extract_raw_scores_bulk = lambda *a, **k: pd.Series([])
        monkeypatch.setitem(sys.modules, "kernel.scoring", fake_kernel_scoring)

        fake_training_scoring = type(sys)("training.scoring")
        fake_training_scoring.fit_probability_calibration = lambda *a, **k: None
        fake_training_scoring.fit_expected_return_calibration = lambda *a, **k: {}
        fake_training_scoring.raw_score_kind_for_model = lambda m: "raw"
        monkeypatch.setitem(sys.modules, "training.scoring", fake_training_scoring)

        rs.recalibrate("renquant_test", dry_run=True)

        assert config_path.stat().st_mtime == mtime_before, \
            "dry_run must not touch the config file"
        assert json.loads(config_path.read_text()) == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
