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

    def test_ic_drops_below_threshold_and_rejects_missing(self, fake_models_dir):
        strategy_dir, tickers = fake_models_dir
        uctx = _run_job(strategy_dir, tickers, floor_type="ic", threshold=0.04)
        # AAA=0.05 passes; BBB=0.02 fails; CCC missing → rejected fail-closed.
        assert "AAA" in uctx.loaded_models
        assert "BBB" not in uctx.loaded_models
        assert "CCC" not in uctx.loaded_models
        reasons = dict(uctx.rejections)
        assert reasons["CCC"] == "ic_missing"

    def test_unknown_floor_type_raises(self, fake_models_dir):
        strategy_dir, tickers = fake_models_dir
        with pytest.raises(ValueError, match="unknown universe_floor.type"):
            _run_job(strategy_dir, tickers, floor_type="nonsense", threshold=1.0)


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


# ── Sharpe evaluator prefers tournament sharpe over live_holdout_sharpe ─────

class TestSharpeEvaluatorPrefersTournament:
    """Regression: _eval_sharpe must prefer tournament `sharpe` (full
    walk-forward OOS, statistically stable) over `live_holdout_sharpe`
    (126-day tail, noisy).

    Prior to 2026-04-23 the order was reversed: a single volatile 6-month
    window pushed ~30 healthy models below the 0.5 floor, gutting the
    universe (21/52 admitted vs 50/52 by tournament sharpe). APY collapsed
    from ~20% to ~2% because the per-ticker buy signals never arrived.
    """

    def test_prefers_sharpe_when_both_present(self, tmp_path: Path):
        from datetime import date as _date
        from kernel.pipeline.job_universe import _eval_sharpe
        meta = {"sharpe": 1.5, "live_holdout_sharpe": -0.5}
        assert _eval_sharpe(meta) == 1.5, \
            "tournament sharpe should win when both are present"

    def test_falls_back_to_holdout_when_sharpe_missing(self):
        from kernel.pipeline.job_universe import _eval_sharpe
        meta = {"live_holdout_sharpe": 0.8}
        assert _eval_sharpe(meta) == 0.8

    def test_returns_none_when_neither_present(self):
        from kernel.pipeline.job_universe import _eval_sharpe
        assert _eval_sharpe({}) is None

    def test_full_job_admits_ticker_with_good_sharpe_but_bad_holdout(
        self, tmp_path: Path,
    ):
        """End-to-end: AAPL-like ticker (sharpe=1.5, live_holdout_sharpe=-0.5)
        must pass a sharpe>=1.0 floor — the noisy holdout should not exclude it.
        """
        from datetime import date as _date
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _write_ticker(models_dir, "AAPL", {
            "trained_date": _date.today().isoformat(),
            "sharpe": 1.5,
            "live_holdout_sharpe": -0.5,    # noisy 6-month tail — should be ignored
        })
        _write_ticker(models_dir, "BAD", {
            "trained_date": _date.today().isoformat(),
            "sharpe": 0.3,                  # genuine tournament failure
            "live_holdout_sharpe": 1.5,     # would have passed the old order
        })
        config = {
            "watchlist":           ["AAPL", "BAD"],
            "model_staleness_days": 0,
            "ranking": {"universe_floor": {"type": "sharpe", "threshold": 1.0}},
        }
        uctx = UniverseContext(config=config, strategy_dir=tmp_path)
        LoadUniverseJob().run(uctx)
        assert "AAPL" in uctx.loaded_models, \
            "tournament sharpe 1.5 must admit AAPL despite negative holdout"
        assert "BAD" not in uctx.loaded_models, \
            "tournament sharpe 0.3 must drop BAD despite positive holdout"


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

class TestDefensivesExemptFromFloor:
    """Regression: universe_floor must not filter out defensive_tickers.

    If defensives (the only universe allowed during bear_only / ConfidenceVeto)
    are filtered out by the Sharpe floor, the strategy sits idle for weeks
    whenever regime confidence dips below the veto threshold. User explicitly
    flagged this class of systemic no-trade as unacceptable.
    """

    def test_defensive_below_sharpe_floor_still_admitted(self, tmp_path: Path):
        # Build 2 tickers: one defensive with sharpe=0.3 (below floor), one
        # offensive with sharpe=0.3 (below floor). Only the offensive should drop.
        from datetime import date as _date
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        for ticker, sharpe in [("DEF", 0.3), ("OFF", 0.3)]:
            _write_ticker(models_dir, ticker, {
                "trained_date": _date.today().isoformat(),
                "sharpe": sharpe, "live_holdout_sharpe": sharpe,
            })

        config = {
            "watchlist":           ["DEF", "OFF"],
            "defensive_tickers":   ["DEF"],
            "model_staleness_days": 0,
            "ranking": {"universe_floor": {"type": "sharpe", "threshold": 1.0}},
        }
        uctx = UniverseContext(config=config, strategy_dir=tmp_path)
        LoadUniverseJob().run(uctx)
        assert "DEF" in uctx.loaded_models, \
            "defensives must be exempt from the Sharpe floor"
        assert "OFF" not in uctx.loaded_models, \
            "non-defensives below floor should still be dropped"


class TestResilience:
    """A single malformed artifact must not crash the entire LoadArtifactsTask
    (regression — this used to kill daily_104.sh in production)."""

    def test_empty_metadata_rejects_only_that_ticker(self, fake_models_dir):
        strategy_dir, tickers = fake_models_dir
        # Zero out one ticker's metadata (empty file → JSONDecodeError)
        p = strategy_dir / "models" / "BBB" / "BBB-policy-metadata.json"
        p.write_text("")
        uctx = _run_job(strategy_dir, tickers, floor_type="none", threshold=0.0)
        # BBB rejected with a load_error reason; AAA + CCC still admitted.
        assert "AAA" in uctx.loaded_models
        assert "CCC" in uctx.loaded_models
        assert "BBB" not in uctx.loaded_models
        assert any(r[0] == "BBB" and r[1].startswith("load_error_")
                   for r in uctx.rejections)

    def test_truncated_metadata_rejects_only_that_ticker(self, fake_models_dir):
        strategy_dir, tickers = fake_models_dir
        p = strategy_dir / "models" / "BBB" / "BBB-policy-metadata.json"
        p.write_text("{\"policy_type\": \"classifi")  # truncated mid-write
        uctx = _run_job(strategy_dir, tickers, floor_type="none", threshold=0.0)
        assert "BBB" not in uctx.loaded_models
        assert "AAA" in uctx.loaded_models
        assert "CCC" in uctx.loaded_models


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

    def test_missing_trained_date_rejected_when_staleness_enabled(self, fake_models_dir):
        strategy_dir, tickers = fake_models_dir
        p = strategy_dir / "models" / "BBB" / "BBB-policy-metadata.json"
        m = json.loads(p.read_text())
        m.pop("trained_date", None)
        p.write_text(json.dumps(m))

        cfg = {
            "watchlist": tickers,
            "model_staleness_days": 30,
            "ranking": {"universe_floor": {"type": "none"}},
        }
        uctx = UniverseContext(config=cfg, strategy_dir=strategy_dir)
        LoadUniverseJob().run(uctx)

        assert "BBB" not in uctx.loaded_models
        assert dict(uctx.rejections)["BBB"] == "trained_date_missing"

    def test_invalid_trained_date_rejected_when_staleness_enabled(self, fake_models_dir):
        strategy_dir, tickers = fake_models_dir
        p = strategy_dir / "models" / "BBB" / "BBB-policy-metadata.json"
        m = json.loads(p.read_text())
        m["trained_date"] = "not-a-date"
        p.write_text(json.dumps(m))

        cfg = {
            "watchlist": tickers,
            "model_staleness_days": 30,
            "ranking": {"universe_floor": {"type": "none"}},
        }
        uctx = UniverseContext(config=cfg, strategy_dir=strategy_dir)
        LoadUniverseJob().run(uctx)

        assert "BBB" not in uctx.loaded_models
        assert dict(uctx.rejections)["BBB"] == "trained_date_invalid"


# ── Adapter surface parity ───────────────────────────────────────────────────

class TestAdapterParity:
    """All three adapters use LoadUniverseJob — verify via source inspection."""

    def _read(self, rel: str) -> str:
        return (Path(__file__).resolve().parent.parent / rel).read_text()

    def test_lean_adapter_calls_job(self):
        src = self._read("backtesting/renquant_104/main.py")
        assert "LoadUniverseJob" in src

    def test_lean_universe_passes_finite_nonzero_held_tickers(self):
        src = self._read("backtesting/renquant_104/main.py")
        assert "held_tickers=self._current_held_tickers()" in src
        assert "def _current_held_tickers" in src
        assert "math.isfinite(qty)" in src
        assert "abs(qty) > 1e-9" in src

    def test_sim_adapter_calls_job(self):
        src = self._read("backtesting/renquant_104/adapters/sim.py")
        assert "LoadUniverseJob" in src

    def test_live_runner_calls_job(self):
        src = self._read("live/runner.py")
        assert "LoadUniverseJob" in src

    def test_live_runner_preserves_rejection_reasons_for_db_trace(self):
        src = self._read("live/runner.py")
        assert 'config["_universe_rejections"] = dict(uctx.rejections)' in src

    def test_no_hand_written_filter_loops_remain(self):
        """Enforce 'every logical unit is a Task/Job/Pipeline' for admission.

        live/runner.py retains a legacy fallback for renquant_103 (which
        predates job_universe.py). 104 invocations always take the
        LoadUniverseJob branch; the legacy loop is unreachable for 104.
        We allow the legacy loop in live/runner.py when (a) it's gated
        by a job_universe_path.exists() check, and (b) the file otherwise
        wires LoadUniverseJob — both already verified by other tests in
        this class.
        """
        for rel in [
            "backtesting/renquant_104/main.py",
            "backtesting/renquant_104/adapters/sim.py",
        ]:
            src = self._read(rel)
            assert "sharpe_floor > 0 and" not in src, \
                f"{rel} still contains a hand-written Sharpe filter loop"
        # live/runner.py: ensure the legacy loop is gated by the
        # job_universe.py existence check (renquant_103 fallback only).
        live_src = self._read("live/runner.py")
        if "sharpe_floor > 0 and" in live_src:
            assert "job_universe_path.exists()" in live_src, (
                "live/runner.py contains a Sharpe filter but the legacy "
                "fallback is not gated by job_universe_path.exists() — "
                "it would now fire for 104 invocations too."
            )
