"""WF sim-time provenance wiring through the sim adapter (pipeline#215/#216).

Umbrella half of the ``wf_sim_provenance.v1`` contract:

    1. WalkForwardModelLoader(provenance_sink=...) emits ONE fold_resolved
       per bar at the entry_as_of seam; default (no sink) is byte-identical.
    2. SimAdapter stamps ctx._wf_provenance_sink / _wf_active_fold /
       _wf_input_watermark; ctx.run_timestamp stays None (bar-date-only sim).
    3. RecordScoreDistributionTask emits score_committed post-INSERT whose
       payload digest matches a recompute over what the DB reads back.
    4. Persistence-off leg: SimAdapter.commit emits persisted:false.
    5. pipeline_runs mirror columns accept + persist the fold provenance.
    6. The daily/live path never constructs a sink.

Emit tests importorskip the pipeline provenance module: while the PINNED
renquant-pipeline predates pipeline#216 these tests SKIP (the sim then runs
without emit — by design); they light up when the pin advances. Run with
the pipeline main export on PYTHONPATH to exercise them now.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

_STRATEGY_DIR = (
    Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
)
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

_PROV_MOD = "renquant_pipeline.kernel.walk_forward.provenance"


# ─── Fixtures (same synthetic-panel pattern as test_sim_walkforward) ────────


def _tiny_ohlcv(days: int = 400, seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=days)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, days)))
    return pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": np.ones(days) * 1e6,
    }, index=idx)


def _write_synthetic_panel_artifact(
    path: Path, *, trained_date: str, tag: str = "synthetic",
) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)).astype("float32")
    y = rng.normal(size=50).astype("float32")
    booster = xgb.train(
        params={"objective": "reg:squarederror", "max_depth": 2,
                "tree_method": "hist", "verbosity": 0},
        dtrain=xgb.DMatrix(X, label=y), num_boost_round=2,
    )
    raw = bytes(booster.save_raw(raw_format="json")).decode("utf-8")
    path.write_text(json.dumps({
        "version": 2,
        "kind": "panel_ltr_xgboost",
        "trained_date": trained_date,
        "feature_cols": ["f0", "f1", "f2"],
        "params": {},
        "best_iter": 1,
        "booster_raw_json": raw,
        "tag": tag,
    }, default=str))


@pytest.fixture
def manifest_with_two_retrains(tmp_path: Path) -> Path:
    art_a = tmp_path / "wf_modelA.json"
    art_b = tmp_path / "wf_modelB.json"
    _write_synthetic_panel_artifact(art_a, trained_date="2024-03-15", tag="A")
    _write_synthetic_panel_artifact(art_b, trained_date="2024-06-15", tag="B")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "retrains": [
            {"cutoff_date": "2024-03-01", "trained_date": "2024-03-15",
             "artifact_uri": str(art_a)},
            {"cutoff_date": "2024-06-01", "trained_date": "2024-06-15",
             "artifact_uri": str(art_b)},
        ]
    }))
    return manifest


def _sink_for(tmp_path: Path, *, seed: "int | None" = 7):
    provenance = pytest.importorskip(_PROV_MOD)
    return provenance.JsonlProvenanceSink(
        "wfsim-test", tmp_path / "data" / "wf_provenance",
        seed=seed, revision_pins={"umbrella": "deadbeef"},
    )


def _read_jsonl(sink) -> list[dict]:
    if not Path(sink.path).exists():
        return []
    return [json.loads(line)
            for line in Path(sink.path).read_text().splitlines() if line]


def _wf_adapter(tmp_path, manifest, *, sink, extra_cfg=None, days=400):
    from adapters.sim import SimAdapter
    ohlcv = {"SPY": _tiny_ohlcv(days=days)}
    cfg = {
        "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
        "walkforward": {"enabled": True, "manifest_path": str(manifest)},
    }
    cfg.update(extra_cfg or {})
    adapter = SimAdapter(
        config=cfg, strategy_dir=tmp_path,
        ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
        initial_cash=100_000, provenance_sink=sink,
    )
    return adapter, ohlcv


def _first_bar_on_or_after(ohlcv, date: str) -> pd.Timestamp:
    return next(d for d in ohlcv["SPY"].index if d >= pd.Timestamp(date))


# ─── 1. Loader boundary: fold_resolved ──────────────────────────────────────


class TestLoaderFoldResolvedEmit:
    def test_model_as_of_emits_once_per_bar(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        from kernel.walk_forward.loader import WalkForwardModelLoader
        sink = _sink_for(tmp_path)
        loader = WalkForwardModelLoader(
            manifest_with_two_retrains, provenance_sink=sink,
        )
        bar = pd.Timestamp("2024-04-15")
        loader.model_as_of(bar)
        # Re-entrant resolutions of the same bar (calibrator leg funnels
        # through entry_as_of too) must not duplicate the record.
        loader.entry_as_of(bar)
        loader.model_as_of(bar)

        records = _read_jsonl(sink)
        assert len(records) == 1
        rec = records[0]
        assert rec["record_kind"] == "fold_resolved"
        assert rec["schema_version"] == "wf_sim_provenance.v1"
        assert rec["prediction_date"] == "2024-04-15"
        assert rec["cutoff_date"] == "2024-03-01"
        # Sink completes the run identity (design §2.3).
        assert rec["sim_run_id"] == "wfsim-test"
        assert rec["seed"] == 7
        assert rec["revision_pins"] == {"umbrella": "deadbeef"}
        # Full digest grammar over the REAL fold artifact bytes.
        provenance = pytest.importorskip(_PROV_MOD)
        art = tmp_path / "wf_modelA.json"
        assert rec["artifact_digest"] == provenance.file_digest(art)
        assert rec["is_real_content_digest"] is True
        assert rec["manifest_digest"] == provenance.file_digest(
            manifest_with_two_retrains)
        assert rec["family"] == "json"
        # Loader exposes the record for the adapter's ctx stamp.
        assert loader.fold_record_for(bar)["cutoff_date"] == "2024-03-01"

    def test_two_bars_two_records_fold_switch(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        from kernel.walk_forward.loader import WalkForwardModelLoader
        sink = _sink_for(tmp_path)
        loader = WalkForwardModelLoader(
            manifest_with_two_retrains, provenance_sink=sink,
        )
        loader.model_as_of(pd.Timestamp("2024-04-15"))
        loader.model_as_of(pd.Timestamp("2024-07-15"))
        records = _read_jsonl(sink)
        assert [r["prediction_date"] for r in records] == [
            "2024-04-15", "2024-07-15"]
        assert [r["cutoff_date"] for r in records] == [
            "2024-03-01", "2024-06-01"]

    def test_no_sink_default_no_records_no_attr_stamps(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        from kernel.walk_forward.loader import WalkForwardModelLoader
        loader = WalkForwardModelLoader(manifest_with_two_retrains)
        bar = pd.Timestamp("2024-04-15")
        loader.model_as_of(bar)
        assert loader.fold_record_for(bar) is None
        assert not (tmp_path / "data").exists()


# ─── 2. Adapter ctx stamps ──────────────────────────────────────────────────


class TestAdapterCtxStamps:
    def test_make_context_stamps_sink_fold_watermark(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        sink = _sink_for(tmp_path)
        adapter, ohlcv = _wf_adapter(
            tmp_path, manifest_with_two_retrains, sink=sink)
        bar = _first_bar_on_or_after(ohlcv, "2024-04-15")
        ctx = adapter.make_context(bar)

        assert getattr(ctx, "_wf_provenance_sink") is sink
        fold = getattr(ctx, "_wf_active_fold")
        assert isinstance(fold, dict)
        assert fold["cutoff_date"] == "2024-03-01"
        assert fold["prediction_date"] == bar.date().isoformat()
        # Bar-date-only sim: the real decision instant is unknowable, so
        # run_timestamp stays None and the emitter's documented 16:00 ET
        # close fallback owns score_timestamp.
        assert ctx.run_timestamp is None
        # Watermark is MEASURED from the served frames (truncated OHLCV
        # here): max served bar date at the 16:00 ET session close, and
        # never after the decision date.
        wm = getattr(ctx, "_wf_input_watermark")
        assert wm is not None
        parsed = dt.datetime.fromisoformat(wm)
        assert parsed.tzinfo is not None
        assert parsed.date() <= bar.date()
        assert (parsed.hour, parsed.minute) == (16, 0)

    def test_make_context_without_sink_stamps_nothing(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        adapter, ohlcv = _wf_adapter(
            tmp_path, manifest_with_two_retrains, sink=None)
        bar = _first_bar_on_or_after(ohlcv, "2024-04-15")
        ctx = adapter.make_context(bar)
        assert not hasattr(ctx, "_wf_provenance_sink")
        assert not hasattr(ctx, "_wf_active_fold")
        assert not hasattr(ctx, "_wf_input_watermark")


# ─── 3. score_committed: persisted leg (post-INSERT) ────────────────────────


def _synthetic_candidate(ticker: str = "AAPL") -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker, panel_score=0.42, rank_score=1.5,
        expected_return_horizon_days=20, mu=0.03, mu_horizon_days=20,
        sigma=0.2, kelly_target_pct=None,
    )


class TestScoreCommittedPersisted:
    def test_two_bar_pairs_and_digest_roundtrip(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        provenance = pytest.importorskip(_PROV_MOD)
        from kernel.persistence import ensure_schema
        from kernel.pipeline.task_score_distribution import (
            RecordScoreDistributionTask,
        )
        sink = _sink_for(tmp_path)
        adapter, ohlcv = _wf_adapter(
            tmp_path, manifest_with_two_retrains, sink=sink,
            extra_cfg={"score_db": {"enabled": True}},
        )
        conn = sqlite3.connect(":memory:")
        ensure_schema(conn)

        task = RecordScoreDistributionTask()
        bars = [
            _first_bar_on_or_after(ohlcv, "2024-04-15"),
            _first_bar_on_or_after(ohlcv, "2024-07-15"),
        ]
        run_ids = []
        for bar in bars:
            ctx = adapter.make_context(bar)
            ctx.candidates = [_synthetic_candidate()]
            ctx._db = conn  # noqa: SLF001 - what the adapter does when persistence is on
            assert task.run(ctx) is None  # ran (no explicit False skip)
            assert getattr(ctx, "_wf_score_committed") is True
            run_ids.append(ctx.run_id)

        records = _read_jsonl(sink)
        folds = [r for r in records if r["record_kind"] == "fold_resolved"]
        commits = [r for r in records if r["record_kind"] == "score_committed"]
        assert len(folds) == 2 and len(commits) == 2

        for bar, run_id, fold, commit in zip(bars, run_ids, folds, commits):
            date_iso = bar.date().isoformat()
            assert fold["prediction_date"] == date_iso
            assert commit["prediction_date"] == date_iso
            assert commit["score_observation_key"] == [run_id, date_iso, "sim"]
            # Pair-integrity echo (design §2.1).
            assert commit["artifact_digest"] == fold["artifact_digest"]
            assert commit["persisted"] is True
            assert commit["pit_violation"] is False
            assert commit["n_rows"] == 1
            # score_timestamp = simulated session close, ET, on the bar.
            ts = dt.datetime.fromisoformat(commit["score_timestamp"])
            assert (ts.date().isoformat(), ts.hour) == (date_iso, 16)
            # Recompute the payload digest over what the DB reads back —
            # the exact verification Phase-A extraction performs.
            rows = conn.execute(
                """SELECT ticker, raw_panel, mu, rank_score, sigma
                     FROM score_distribution WHERE run_id = ?""",
                (run_id,),
            ).fetchall()
            payload = [
                {"ticker": t, "raw_panel": rp, "mu": mu,
                 "rank_score": rs, "sigma": sg}
                for (t, rp, mu, rs, sg) in rows
            ]
            assert len(payload) == commit["n_rows"]
            assert provenance.score_payload_digest(payload) == (
                commit["score_payload_digest"])


# ─── 4. score_committed: persistence-off leg (adapter, persisted:false) ─────


class TestScoreCommittedPersistenceOff:
    def test_commit_emits_persisted_false_when_no_db(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        sink = _sink_for(tmp_path)
        adapter, ohlcv = _wf_adapter(
            tmp_path, manifest_with_two_retrains, sink=sink)
        assert adapter._db is None  # noqa: SLF001 - persistence disabled
        bar = _first_bar_on_or_after(ohlcv, "2024-04-15")
        ctx = adapter.make_context(bar)
        ctx.candidates = [_synthetic_candidate()]
        adapter.commit(ctx)

        commits = [r for r in _read_jsonl(sink)
                   if r["record_kind"] == "score_committed"]
        assert len(commits) == 1
        assert commits[0]["persisted"] is False
        assert commits[0]["n_rows"] == 1
        # Echo still binds to the bar's fold.
        assert commits[0]["artifact_digest"] == (
            getattr(ctx, "_wf_active_fold")["artifact_digest"])

    def test_no_double_emit_after_persisted_leg(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        from kernel.pipeline.task_score_distribution import (
            emit_unpersisted_wf_score_committed,
        )
        sink = _sink_for(tmp_path)
        adapter, ohlcv = _wf_adapter(
            tmp_path, manifest_with_two_retrains, sink=sink)
        bar = _first_bar_on_or_after(ohlcv, "2024-04-15")
        ctx = adapter.make_context(bar)
        ctx.candidates = [_synthetic_candidate()]
        ctx._wf_score_committed = True  # noqa: SLF001 - persisted leg ran
        assert emit_unpersisted_wf_score_committed(ctx) is False

    def test_empty_payload_emits_nothing(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        sink = _sink_for(tmp_path)
        adapter, ohlcv = _wf_adapter(
            tmp_path, manifest_with_two_retrains, sink=sink)
        bar = _first_bar_on_or_after(ohlcv, "2024-04-15")
        ctx = adapter.make_context(bar)  # no candidates, no holdings
        adapter.commit(ctx)
        commits = [r for r in _read_jsonl(sink)
                   if r["record_kind"] == "score_committed"]
        assert commits == []


# ─── 5. pipeline_runs SECONDARY mirror ──────────────────────────────────────


class TestPipelineRunsMirror:
    def test_record_pipeline_run_persists_fold_columns(self):
        from kernel.persistence import ensure_schema, record_pipeline_run
        conn = sqlite3.connect(":memory:")
        ensure_schema(conn)
        digest = "sha256:" + "ab" * 32
        run_id = record_pipeline_run(
            conn, run_type="sim", run_date=dt.date(2024, 4, 15),
            training_cutoff="2024-03-01", model_content_sha256=digest,
        )
        row = conn.execute(
            """SELECT training_cutoff, model_content_sha256
                 FROM pipeline_runs WHERE run_id = ?""", (run_id,),
        ).fetchone()
        assert row == ("2024-03-01", digest)

    def test_migration_adds_columns_to_old_db(self):
        from kernel.persistence import ensure_schema
        conn = sqlite3.connect(":memory:")
        # Simulate a pre-mirror pipeline_runs table (the real old schema
        # minus the two new columns; the schema's indexes need run_date +
        # strategy to exist).
        conn.execute(
            "CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY, "
            "run_date DATE NOT NULL, run_type TEXT NOT NULL, "
            "strategy TEXT, regime TEXT, confidence REAL, "
            "portfolio_value REAL, cash REAL, n_candidates INTEGER, "
            "n_exits INTEGER, n_rotations INTEGER, n_buys INTEGER, "
            "commit_sha TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        ensure_schema(conn)
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(pipeline_runs)").fetchall()}
        assert {"training_cutoff", "model_content_sha256"} <= cols


# ─── 6. Daily/live path never constructs a sink ─────────────────────────────


class TestDailyPathNeverConstructsSink:
    def test_sim_adapter_default_is_none(self, tmp_path: Path):
        from adapters.sim import SimAdapter
        ohlcv = {"SPY": _tiny_ohlcv(days=50)}
        adapter = SimAdapter(
            config={"watchlist": [], "sector_etf_map": {}, "tax": {},
                    "regime": {}},
            strategy_dir=_STRATEGY_DIR,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
            initial_cash=100_000,
        )
        assert adapter._provenance_sink is None  # noqa: SLF001

    def test_daily_entry_sources_never_reference_the_sink(self):
        """The sink is a SIM-ONLY construction (design §4 non-goal:
        zero live-surface delta). The daily/live/LEAN entries must not
        reference it; only sim.runner constructs it, gated on
        walkforward.enabled."""
        repo = Path(__file__).resolve().parent.parent
        daily_sources = [
            repo / "live" / "runner.py",
            repo / "backtesting" / "renquant_104" / "adapters" / "lean.py",
            repo / "backtesting" / "renquant_104" / "adapters" / "runner.py",
            repo / "backtesting" / "renquant_104" / "adapters"
                 / "runner_artifacts.py",
            repo / "backtesting" / "renquant_104" / "main.py",
        ]
        for src in daily_sources:
            text = src.read_text()
            assert "provenance_sink" not in text, (
                f"{src} references provenance_sink — the daily path must "
                "never construct or pass the WF provenance sink")
            assert "build_wf_provenance_sink" not in text
        # And the ONE construction site is the sim runner, behind the
        # walkforward.enabled gate.
        sim_runner = (repo / "backtesting" / "renquant_104" / "sim"
                      / "runner.py").read_text()
        assert "build_wf_provenance_sink" in sim_runner
        gate = sim_runner.index('(config.get("walkforward") or {})'
                                '.get("enabled", False)')
        call = sim_runner.index("build_wf_provenance_sink(seed=seed)")
        assert gate < call


# ─── 7. Pre-#216 pin: sink construction degrades loudly to None ─────────────


class TestPre216PinDegradesToNone:
    def test_build_sink_returns_none_when_module_missing(
        self, tmp_path: Path, monkeypatch, caplog,
    ):
        import kernel.walk_forward.provenance_adapter as pa
        # Simulate the pinned (pre-#216) pipeline: the provenance module
        # cannot be imported.
        monkeypatch.setitem(sys.modules, _PROV_MOD, None)
        with caplog.at_level("WARNING"):
            sink = pa.build_wf_provenance_sink(seed=3, data_root=tmp_path)
        assert sink is None
        assert any("predates pipeline#216" in r.message
                   for r in caplog.records)
        assert not (tmp_path / "data").exists()

    def test_build_sink_writes_under_data_root(self, tmp_path: Path):
        pytest.importorskip(_PROV_MOD)
        from kernel.walk_forward.provenance_adapter import (
            build_wf_provenance_sink,
        )
        sink = build_wf_provenance_sink(
            seed=11, sim_run_id="wfsim-loc", data_root=tmp_path)
        assert sink is not None
        assert Path(sink.path) == (
            tmp_path / "data" / "wf_provenance" / "wfsim-loc.jsonl")
