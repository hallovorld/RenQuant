"""Tests for kernel.registry — MLflow artifact registry foundation.

Uses MLflow's local file backend (file:./<tmp>/mlruns) — no server.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

mlflow = pytest.importorskip("mlflow")  # noqa: F401


@pytest.fixture
def tracking_dir(tmp_path, monkeypatch):
    """Point MLflow at a per-test file backend, no cross-test bleed."""
    from kernel.registry import init_tracking
    uri = f"file:{tmp_path / 'mlruns'}"
    init_tracking(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    yield tmp_path
    # `with start_run` always closes; tmp_path teardown nukes the backend.


def _seed_artifact(tmp_path: Path, name: str = "calib.json",
                    payload: dict | None = None) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload or {"hello": "world", "n": 42}))
    return p


# ── URI helpers ───────────────────────────────────────────────────────────────

class TestUriHelpers:
    def test_is_mlflow_uri_accepts_well_formed(self):
        from kernel.registry import is_mlflow_uri
        # 32 hex chars is mlflow's run-id format.
        ok = "mlflow://" + ("a" * 32) + "/path/to/file.json"
        assert is_mlflow_uri(ok) is True

    def test_is_mlflow_uri_rejects_local_path(self):
        from kernel.registry import is_mlflow_uri
        assert is_mlflow_uri("/tmp/foo.json") is False
        assert is_mlflow_uri("artifacts/panel-ltr.json") is False
        assert is_mlflow_uri("") is False
        assert is_mlflow_uri(None) is False  # type: ignore[arg-type]

    def test_parse_mlflow_uri_splits_correctly(self):
        from kernel.registry import parse_mlflow_uri
        rid = "0123456789abcdef0123456789abcdef"
        uri = f"mlflow://{rid}/sub/dir/calib.json"
        got_run, got_path = parse_mlflow_uri(uri)
        assert got_run == rid
        assert got_path == "sub/dir/calib.json"

    def test_parse_mlflow_uri_raises_on_malformed(self):
        from kernel.registry import parse_mlflow_uri
        with pytest.raises(ValueError):
            parse_mlflow_uri("mlflow://too-short/path.json")
        with pytest.raises(ValueError):
            parse_mlflow_uri("/local/path.json")


# ── init_tracking ─────────────────────────────────────────────────────────────

class TestInitTracking:
    def test_sets_environment_variable(self, tmp_path, monkeypatch):
        from kernel.registry import init_tracking
        uri = f"file:{tmp_path / 'mlruns'}"
        out = init_tracking(uri)
        import os
        assert os.environ["MLFLOW_TRACKING_URI"] == uri
        assert out == uri


# ── start_run ────────────────────────────────────────────────────────────────

class TestStartRun:
    def test_yields_32char_run_id_and_logs_params(self, tracking_dir):
        from kernel.registry import start_run
        with start_run("registry-test", {"learning_rate": 0.05,
                                          "n_estimators": 200}) as run_id:
            assert isinstance(run_id, str)
            assert len(run_id) == 32
        # After context exit run is closed; verify by querying mlflow.
        import mlflow
        run = mlflow.get_run(run_id)
        assert run.data.params["learning_rate"] == "0.05"
        assert run.data.params["n_estimators"] == "200"

    def test_run_marked_failed_on_exception(self, tracking_dir):
        from kernel.registry import start_run
        captured_id = {}
        with pytest.raises(RuntimeError, match="boom"):
            with start_run("registry-test") as run_id:
                captured_id["id"] = run_id
                raise RuntimeError("boom")
        import mlflow
        run = mlflow.get_run(captured_id["id"])
        # mlflow marks failed runs as FAILED.
        assert run.info.status == "FAILED"


# ── log_artifact_with_meta ───────────────────────────────────────────────────

class TestLogArtifactWithMeta:
    def test_artifact_and_meta_uploaded(self, tracking_dir, tmp_path):
        from kernel.registry import start_run, log_artifact_with_meta
        src = _seed_artifact(tmp_path, "panel-rank-calibration.json",
                              {"prob_x": [0.1, 0.5, 0.9],
                               "prob_y": [0.0, 0.4, 1.0]})
        with start_run("registry-test", {}) as run_id:
            uri = log_artifact_with_meta(
                run_id, src, artifact_path="calibrators",
                meta={"pool_ic": 0.094, "rows": 47000},
            )
        assert uri.startswith(f"mlflow://{run_id}/calibrators/")
        # Re-download and compare bytes.
        import mlflow
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="calibrators",
        )
        files = sorted(p.name for p in Path(local_dir).iterdir())
        assert "panel-rank-calibration.json" in files
        assert "panel-rank-calibration.json.meta.json" in files
        meta_loaded = json.loads(
            (Path(local_dir) / "panel-rank-calibration.json.meta.json").read_text(),
        )
        assert meta_loaded["pool_ic"] == 0.094
        assert meta_loaded["rows"] == 47000

    def test_missing_local_file_raises(self, tracking_dir, tmp_path):
        from kernel.registry import start_run, log_artifact_with_meta
        with start_run("registry-test", {}) as run_id:
            with pytest.raises(FileNotFoundError):
                log_artifact_with_meta(run_id, tmp_path / "nope.json")


# ── resolve_uri ──────────────────────────────────────────────────────────────

class TestResolveUri:
    def test_local_path_passthrough(self, tmp_path):
        from kernel.registry import resolve_uri
        f = _seed_artifact(tmp_path, "x.json")
        out = resolve_uri(str(f))
        assert out == f
        assert out.exists()

    def test_local_path_missing_raises(self, tmp_path):
        from kernel.registry import resolve_uri
        with pytest.raises(FileNotFoundError):
            resolve_uri(str(tmp_path / "ghost.json"))

    def test_mlflow_uri_round_trip(self, tracking_dir, tmp_path):
        from kernel.registry import (
            start_run, log_artifact_with_meta, resolve_uri,
        )
        payload = {"prob_x": [0.0, 1.0], "prob_y": [0.0, 1.0]}
        src = _seed_artifact(tmp_path, "calib.json", payload)
        with start_run("registry-test", {}) as run_id:
            uri = log_artifact_with_meta(run_id, src, artifact_path="cals",
                                          meta={"k": "v"})
        local = resolve_uri(uri)
        assert local.exists()
        assert json.loads(local.read_text()) == payload


# ── register_model ───────────────────────────────────────────────────────────

class TestRegisterModel:
    def test_register_after_log(self, tracking_dir, tmp_path):
        from kernel.registry import (
            start_run, log_artifact_with_meta, register_model,
        )
        src = _seed_artifact(tmp_path, "panel-rank-calibration.json",
                              {"prob_x": [0.0, 1.0], "prob_y": [0.0, 1.0]})
        with start_run("registry-test", {"horizon": 60}) as run_id:
            log_artifact_with_meta(run_id, src, artifact_path="model",
                                    meta={"pool_ic": 0.094})
        handle = register_model("renquant-panel-calibration", run_id,
                                 stage=None, artifact_path="model")
        assert handle["name"] == "renquant-panel-calibration"
        assert handle["run_id"] == run_id
        assert int(handle["version"]) >= 1


# ── PoC integration: GlobalPanelCalibration.save → MLflow (CLAUDE.md §5.13.1) ─

class TestGlobalCalibratorMlflowPoc:
    """Walk the REAL prod-path save() — not a hand-built fixture."""

    def _make_calibrator(self):
        import numpy as np  # noqa: PLC0415
        from training_panel.global_calibrator import (  # noqa: PLC0415
            GlobalPanelCalibration,
        )
        return GlobalPanelCalibration(
            prob_x=np.array([0.0, 0.5, 1.0]),
            prob_y=np.array([0.0, 0.4, 1.0]),
            er_x=np.array([0.0, 0.5, 1.0]),
            er_y=np.array([0.0, 0.05, 0.10]),
            metadata={"pool_ic": 0.094},
        )

    def test_save_default_does_not_log_to_mlflow(self, tmp_path, monkeypatch):
        """RENQUANT_MLFLOW_LOG unset → no mlruns dir created (default off)."""
        monkeypatch.delenv("RENQUANT_MLFLOW_LOG", raising=False)
        cal = self._make_calibrator()
        out_path = tmp_path / "panel-rank-calibration.json"
        cal.save(out_path, metadata={"pool_ic": 0.094})
        assert out_path.exists()
        # No mlruns directory should appear from this save.
        assert not (tmp_path / "mlruns").exists()

    def test_save_with_flag_logs_to_mlflow(self, tmp_path, monkeypatch):
        """RENQUANT_MLFLOW_LOG=1 → JSON file written AND mlflow run exists."""
        tracking_uri = f"file:{tmp_path / 'mlruns'}"
        monkeypatch.setenv("RENQUANT_MLFLOW_LOG", "1")
        monkeypatch.setenv("RENQUANT_MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setenv("RENQUANT_MLFLOW_EXPERIMENT", "poc-test-exp")
        cal = self._make_calibrator()
        out_path = tmp_path / "panel-rank-calibration.json"
        cal.save(out_path, metadata={"pool_ic": 0.094})

        # 1. Local JSON still written, contract preserved for legacy readers.
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["kind"] == "global_panel_calibration"

        # 2. MLflow side: a run exists in the experiment, with our artifact.
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("poc-test-exp")
        assert exp is not None, "experiment not created"
        runs = client.search_runs([exp.experiment_id])
        assert len(runs) == 1
        run = runs[0]
        artifacts = client.list_artifacts(run.info.run_id, "calibrator")
        names = sorted(a.path for a in artifacts)
        assert "calibrator/panel-rank-calibration.json" in names
        assert "calibrator/panel-rank-calibration.json.meta.json" in names

    def test_save_mlflow_failure_does_not_raise(self, tmp_path, monkeypatch):
        """If MLflow fails, calibrator save must still succeed (non-fatal).

        Pin the §5.13.13 invariant: a registry-side fault must never
        regress the production legacy-file write.
        """
        # Force mlflow's `start_run` to raise synchronously by monkey-patching
        # the imported function inside the registry module. Avoids real HTTP
        # / filesystem flakes in tests.
        monkeypatch.setenv("RENQUANT_MLFLOW_LOG", "1")
        monkeypatch.setenv("RENQUANT_MLFLOW_TRACKING_URI",
                           f"file:{tmp_path / 'mlruns'}")
        from kernel.registry import mlflow_registry
        def _boom(*a, **k):
            raise RuntimeError("simulated mlflow outage")
        monkeypatch.setattr(mlflow_registry, "init_tracking", _boom)
        cal = self._make_calibrator()
        out_path = tmp_path / "panel-rank-calibration.json"
        # Should NOT raise even though the mlflow side will fail.
        cal.save(out_path, metadata={})
        assert out_path.exists()
