"""LoadUniverseJob alignment tests for renquant_104.

Verifies that universe admission is a single pipeline decision applied
identically by every surface that loads models:

  1. LeanAdapter, RunnerAdapter, SimAdapter — they all call
     kernel.pipeline.job_universe.LoadUniverseJob rather than each
     re-implementing the filter loop.
  2. FilterUniverseFloorTask dispatches by ranking.universe_floor.type
     — the three built-in types (none / sharpe / ic) must produce the
     expected rejections for the same inputs.
  3. Extensibility: adding a new entry to FLOOR_EVALUATORS makes the
     Task aware of the new type without touching any adapter.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.config import universe_floor_spec  # noqa: E402
from kernel.pipeline.job_universe import (  # noqa: E402
    FLOOR_EVALUATORS,
    FilterStalenessTask,
    FilterUniverseFloorTask,
    LoadArtifactsTask,
    LoadUniverseJob,
    UniverseContext,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _write_ticker(models_dir: Path, ticker: str, meta: dict) -> None:
    """Write a minimal policy-metadata.json + rf-trees.json so load_artifact succeeds."""
    d = models_dir / ticker
    d.mkdir(parents=True, exist_ok=True)
    full_meta = {
        "policy_type":     "classification",
        "feature_columns": ["rsi"],
        "buy_threshold":   0.1,
        "sell_threshold":  -0.1,
        **meta,
    }
    (d / f"{ticker}-policy-metadata.json").write_text(json.dumps(full_meta))
    (d / f"{ticker}-rf-trees.json").write_text(json.dumps([]))


@pytest.fixture
def fake_models_dir(tmp_path: Path) -> tuple[Path, list[str]]:
    models_dir = tmp_path / "models"
    tickers = ["AAA", "BBB", "CCC"]
    today = date.today().isoformat()
    _write_ticker(models_dir, "AAA", {
        "trained_date": today, "sharpe": 1.5,
        "live_holdout_sharpe": 1.5, "panel_oos_ic": 0.05,
    })
    _write_ticker(models_dir, "BBB", {
        "trained_date": today, "sharpe": 0.3,
        "live_holdout_sharpe": 0.3, "panel_oos_ic": 0.02,
    })
    _write_ticker(models_dir, "CCC", {
        "trained_date": today, "sharpe": 1.1,
        "live_holdout_sharpe": 1.1,
        # no panel_oos_ic — tests "admit_on_missing"
    })
    return tmp_path, tickers


# ── universe_floor_spec ──────────────────────────────────────────────────────

class TestUniverseFloorSpec:
    def test_default_is_none(self):
        assert universe_floor_spec({}) == ("none", 0.0)

    def test_reads_type_and_threshold(self):
        cfg = {"ranking": {"universe_floor": {"type": "sharpe", "threshold": 0.8}}}
        assert universe_floor_spec(cfg) == ("sharpe", 0.8)

    def test_lowercases_type(self):
        cfg = {"ranking": {"universe_floor": {"type": "SHARPE", "threshold": 1.0}}}
        assert universe_floor_spec(cfg) == ("sharpe", 1.0)


# ── LoadUniverseJob end-to-end ───────────────────────────────────────────────

def _run_job(strategy_dir: Path, watchlist: list[str], floor_type: str,
             threshold: float) -> UniverseContext:
    config = {
        "watchlist":           watchlist,
        "model_staleness_days": 0,
        "ranking": {"universe_floor": {"type": floor_type, "threshold": threshold}},
    }
    uctx = UniverseContext(config=config, strategy_dir=strategy_dir)
    LoadUniverseJob().run(uctx)
    return uctx


class TestLoadUniverseJob:
    def test_none_admits_all(self, fake_models_dir):
        strategy_dir, tickers = fake_models_dir
        uctx = _run_job(strategy_dir, tickers, floor_type="none", threshold=0.0)
        assert set(uctx.loaded_models.keys()) == set(tickers)
        # No filter-stage rejections.
        filter_reasons = [r for _, r in uctx.rejections
                          if r.startswith(("sharpe", "ic", "stale"))]
        assert filter_reasons == []

    def test_sharpe_drops_below_threshold(self, fake_models_dir):
        strategy_dir, tickers = fake_models_dir
        uctx = _run_job(strategy_dir, tickers, floor_type="sharpe", threshold=1.0)
        # AAA=1.5 and CCC=1.1 pass; BBB=0.3 fails.
        assert "AAA" in uctx.loaded_models
        assert "CCC" in uctx.loaded_models
        assert "BBB" not in uctx.loaded_models
        reasons = dict(uctx.rejections)
        assert "sharpe_0.300_below_1.0" in reasons["BBB"]

    def test_ic_drops_below_threshold_and_admits_missing(self, fake_models_dir):
        strategy_dir, tickers = fake_models_dir
        uctx = _run_job(strategy_dir, tickers, floor_type="ic", threshold=0.04)
        # AAA=0.05 passes; BBB=0.02 fails; CCC missing → admitted with warning.
        assert "AAA" in uctx.loaded_models
        assert "CCC" in uctx.loaded_models  # admit-on-missing
        assert "BBB" not in uctx.loaded_models

    def test_unknown_floor_type_admits_all_with_warning(self, fake_models_dir, caplog):
        strategy_dir, tickers = fake_models_dir
        uctx = _run_job(strategy_dir, tickers, floor_type="nonsense", threshold=1.0)
        assert set(uctx.loaded_models.keys()) == set(tickers)


# ── Filter task short-circuits ───────────────────────────────────────────────

class TestFilterShortCircuits:
    def test_filter_task_skips_when_type_is_none(self):
        cfg = {"ranking": {"universe_floor": {"type": "none"}}}
        uctx = UniverseContext(config=cfg, strategy_dir=Path("."))
        assert FilterUniverseFloorTask().should_skip(uctx) is True

    def test_filter_task_runs_when_type_is_sharpe(self):
        cfg = {"ranking": {"universe_floor": {"type": "sharpe"}}}
        uctx = UniverseContext(config=cfg, strategy_dir=Path("."))
        assert FilterUniverseFloorTask().should_skip(uctx) is False

    def test_filter_task_runs_when_type_is_ic(self):
        cfg = {"ranking": {"universe_floor": {"type": "ic"}}}
        uctx = UniverseContext(config=cfg, strategy_dir=Path("."))
        assert FilterUniverseFloorTask().should_skip(uctx) is False


# ── Extensibility — the Task picks up new types from FLOOR_EVALUATORS ───────

class TestExtensibility:
    def test_registering_new_evaluator_is_wired(self, fake_models_dir, monkeypatch):
        strategy_dir, tickers = fake_models_dir
        # Register a synthetic "hit_rate" type that just reads meta["hit_rate"].
        # Write a matching field onto AAA and BBB.
        aaa_meta_path = strategy_dir / "models" / "AAA" / "AAA-policy-metadata.json"
        bbb_meta_path = strategy_dir / "models" / "BBB" / "BBB-policy-metadata.json"
        aaa_meta = json.loads(aaa_meta_path.read_text()); aaa_meta["hit_rate"] = 0.6
        bbb_meta = json.loads(bbb_meta_path.read_text()); bbb_meta["hit_rate"] = 0.4
        aaa_meta_path.write_text(json.dumps(aaa_meta))
        bbb_meta_path.write_text(json.dumps(bbb_meta))

        monkeypatch.setitem(FLOOR_EVALUATORS, "hit_rate",
                            lambda m: m.get("hit_rate"))
        uctx = _run_job(strategy_dir, tickers, floor_type="hit_rate", threshold=0.5)
        assert "AAA" in uctx.loaded_models
        assert "BBB" not in uctx.loaded_models


# ── Staleness ────────────────────────────────────────────────────────────────

class TestStaleness:
    def test_stale_models_dropped(self, fake_models_dir):
        strategy_dir, tickers = fake_models_dir
        # Rewrite BBB's metadata with an ancient date.
        p = strategy_dir / "models" / "BBB" / "BBB-policy-metadata.json"
        m = json.loads(p.read_text())
        m["trained_date"] = "2000-01-01"
        p.write_text(json.dumps(m))

        cfg = {
            "watchlist":           tickers,
            "model_staleness_days": 30,
            "ranking": {"universe_floor": {"type": "none"}},
        }
        uctx = UniverseContext(config=cfg, strategy_dir=strategy_dir)
        LoadUniverseJob().run(uctx)
        assert "AAA" in uctx.loaded_models
        assert "BBB" not in uctx.loaded_models
        reasons = dict(uctx.rejections)
        assert "stale" in reasons["BBB"]


# ── Adapter surface parity ───────────────────────────────────────────────────

class TestAdapterParity:
    """All three adapters use LoadUniverseJob — verify via source inspection."""

    def _read(self, rel: str) -> str:
        return (Path(__file__).resolve().parent.parent / rel).read_text()

    def test_lean_adapter_calls_job(self):
        src = self._read("backtesting/renquant_104/main.py")
        assert "LoadUniverseJob" in src

    def test_sim_adapter_calls_job(self):
        src = self._read("backtesting/renquant_104/adapters/sim.py")
        assert "LoadUniverseJob" in src

    def test_live_runner_calls_job(self):
        src = self._read("live/runner.py")
        assert "LoadUniverseJob" in src

    def test_no_hand_written_filter_loops_remain(self):
        """Enforce 'every logical unit is a Task/Job/Pipeline' for admission."""
        for rel in [
            "backtesting/renquant_104/main.py",
            "backtesting/renquant_104/adapters/sim.py",
            "live/runner.py",
        ]:
            src = self._read(rel)
            # The hand-written pattern we just removed:
            assert "sharpe_floor > 0 and" not in src, \
                f"{rel} still contains a hand-written Sharpe filter loop"
