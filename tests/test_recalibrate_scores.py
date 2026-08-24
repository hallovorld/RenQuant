"""Regression test for scripts/recalibrate_scores.py config-write safety.

Bug history (2026-04-22): recalibrate_scores.py read strategy_config.json at
start, did ~30 s of per-ticker work, then wrote the whole in-memory config
back — silently wiping any edit that landed in that window. The
defensive_tickers / confidence_veto_threshold fixes from commit 3c366b6
disappeared this way.

Fix (2026-04-22): re-read the file immediately before writing and merge ONLY
the two fields this script owns (ranking.blend_updated, ranking.blend_n_symbols).

Superseded (2026-08-24, #1024): the script no longer writes strategy_config.json
AT ALL. Those two fields are telemetry, the config is a git-TRACKED reviewed
input, and stamping runtime state into it left the live umbrella tree
permanently dirty — blocking every deploy that touched the path (it blocked
#602) or inviting a `git checkout --` that destroys the state silently. The
telemetry now goes to a gitignored runtime sidecar under logs/.

The 2026-04-22 guarantee therefore holds in a stronger form: a concurrent edit
cannot be wiped by a write that does not happen. The test below still drives the
identical race and still asserts the edit survives — the guarantee is what
matters, not the mechanism — and now also asserts the config is byte-identical
across the run.
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


def _prepare(tmp_path: Path, monkeypatch):
    """Fake strategy tree + the module stubs recalibrate() needs to run offline.

    Returns (strategy_dir, config_path, module). No concurrent-edit injection —
    the tests that need the race build their own fetch stub on top.
    """
    import pandas as pd  # noqa: PLC0415

    strategy_dir = tmp_path / "backtesting" / "renquant_test"
    (strategy_dir / "models").mkdir(parents=True)
    config_path = strategy_dir / "strategy_config.json"
    _write_minimal_config(config_path)

    import scripts.recalibrate_scores as rs  # noqa: PLC0415
    monkeypatch.setattr(rs, "REPO_ROOT", tmp_path)

    def fake_fetch(sym, provider="yfinance"):
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

    return strategy_dir, config_path, rs


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

        # And the script's own two fields are NOT applied to the config any
        # more (#1024) — they went to the sidecar, which is untracked.
        assert final["ranking"]["blend_updated"] == "2020-01-01", (
            "recalibrate_scores wrote telemetry into the git-tracked config again"
        )
        assert "blend_n_symbols" not in final["ranking"]

        state = json.loads(
            (strategy_dir / rs.BLEND_STATE_RELPATH).read_text()
        )
        assert state["blend_updated"] == str(date.today())
        assert state["blend_n_symbols"] == 0
        assert state["previous"]["blend_updated"] == "2020-01-01", (
            "first run must SEED from the config — otherwise the migration loses "
            "the only copy of the live values"
        )
        assert state["previous"]["seeded_from_config"] is True

    def test_the_config_is_byte_identical_across_a_run(self, tmp_path: Path, monkeypatch):
        """The actual #1024 requirement, stated directly.

        Not "the right keys are preserved" — *nothing* changes, so a live
        checkout stays clean and a deploy touching this path cannot abort.
        """
        strategy_dir, config_path, rs = _prepare(tmp_path, monkeypatch)
        before = config_path.read_bytes()
        rs.recalibrate("renquant_test", dry_run=False)
        assert config_path.read_bytes() == before, (
            "strategy_config.json changed — it is a reviewed, git-tracked input"
        )
        assert (strategy_dir / rs.BLEND_STATE_RELPATH).exists()

    def test_the_sidecar_lives_under_the_gitignored_logs_dir(self):
        """Placement IS the fix. Anywhere tracked reintroduces the defect."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rs_place", REPO_ROOT / "scripts" / "recalibrate_scores.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        assert mod.BLEND_STATE_RELPATH.parts[0] == "logs", mod.BLEND_STATE_RELPATH
        gitignore = (REPO_ROOT / ".gitignore").read_text().splitlines()
        assert any(line.strip() in ("logs/", "/logs/", "logs") for line in gitignore), (
            "logs/ is not gitignored — the sidecar would dirty a tracked path"
        )

    def test_a_second_run_does_not_reseed_from_the_config(self, tmp_path: Path, monkeypatch):
        """Seeding is a one-time migration. If it repeated, the sidecar's
        `previous` would keep resurrecting a config value that is by then
        stale, and the record of the real prior run would be lost."""
        strategy_dir, config_path, rs = _prepare(tmp_path, monkeypatch)
        rs.recalibrate("renquant_test", dry_run=False)
        rs.recalibrate("renquant_test", dry_run=False)
        state = json.loads((strategy_dir / rs.BLEND_STATE_RELPATH).read_text())
        assert state["previous"].get("seeded_from_config") is not True, state
        assert state["previous"]["blend_updated"] == str(date.today())

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
