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
        assert state["previous"]["source"] == "config-seed"

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
        assert state["previous"]["source"] == "prior-run", state
        assert state["previous"]["blend_updated"] == str(date.today())

    def test_the_sidecar_does_not_grow_or_nest_across_many_runs(self, tmp_path: Path, monkeypatch):
        """The first version of this file nested the whole prior PAYLOAD into
        `previous`, so `previous` contained its own `previous` and depth grew by
        one every run — 51 levels and ~10KB of duplicated `note` after a year of
        weekly calibrations (codex on #606).

        It survived review because the second-run test above checked a top-level
        VALUE and never the SHAPE. So this asserts the shape directly, over
        enough runs that any per-run growth is unmistakable.
        """
        strategy_dir, config_path, rs = _prepare(tmp_path, monkeypatch)
        sidecar = strategy_dir / rs.BLEND_STATE_RELPATH

        def depth(obj, d: int = 0) -> int:
            nxt = obj.get("previous") if isinstance(obj, dict) else None
            return depth(nxt, d + 1) if isinstance(nxt, dict) else d

        shapes, depths, sizes = set(), set(), []
        for _ in range(8):
            rs.recalibrate("renquant_test", dry_run=False)
            raw = sidecar.read_text()
            doc = json.loads(raw)
            shapes.add(tuple(sorted(doc)))
            depths.add(depth(doc))
            sizes.append(len(raw))

        assert len(shapes) == 1, f"the document's key set changed across runs: {shapes}"
        assert depths == {1}, f"`previous` nested beyond one level: {depths}"
        # Run 1's `previous` is the config seed, which here carries one fewer
        # field than a prior-run record, so size legitimately changes ONCE. From
        # run 2 on it must be constant — that is the difference between a
        # one-time transition and accumulation.
        assert len(set(sizes[1:])) == 1, (
            f"the sidecar kept changing size across runs with identical inputs — "
            f"something accumulates: {sizes}"
        )
        assert max(sizes) < 2 * min(sizes), f"size is not bounded: {sizes}"
        # And `previous` itself must stay flat, not just shallow.
        doc = json.loads(sidecar.read_text())
        assert set(doc["previous"]) <= set(rs.STATE_FIELDS) | {"source"}, doc["previous"]
        assert not any(isinstance(v, (dict, list)) for v in doc["previous"].values())

    def test_an_existing_blend_weights_is_left_alone(self, tmp_path: Path, monkeypatch):
        """`blend_weights` is a DECISION INPUT — legacy and zero-weighted at the
        current 104 seam, but an input. The 2026-04-22 write path deleted it in
        passing; under the input-only rule a calibration run must not, because a
        rule with one silent exception is not a rule. Removing the key is a
        reviewed config change, not a side effect."""
        strategy_dir, config_path, rs = _prepare(tmp_path, monkeypatch)
        cfg = json.loads(config_path.read_text())
        cfg["ranking"]["blend_weights"] = {"rank": 0.7, "rs": 0.3}
        config_path.write_text(json.dumps(cfg, indent=2))

        rs.recalibrate("renquant_test", dry_run=False)

        after = json.loads(config_path.read_text())
        assert after["ranking"]["blend_weights"] == {"rank": 0.7, "rs": 0.3}, (
            "runtime calibration silently edited a decision input"
        )

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
