"""F-7 r7: registered-manifest experiment mode + verified pins + promotion guard.

``scripts/run_sim_104.py`` (PR #471, F-7) resolves ALL strategy configs from
the PINNED ``renquant-strategy-104`` subrepo by default. No filename-based
routing: every config name goes through the pin lookup, HEAD/dirty/remote
verification against subrepos.lock.json, and fails closed if anything is
missing or mismatched.

Experiment mode (--experiment-manifest) resolves the config through a
REGISTERED manifest (location under experiments/manifests/ AND a matching
digest entry in experiments/manifests/INDEX.json — see
renquant_artifacts.verify_manifest_registered) containing experiment_id,
config_digest, 5 required pins, data_manifest_path, model_artifact_path, and
status. All 5 pins are verified against the actual environment
(renquant_artifacts.verify_experiment_pins) before the run proceeds.
Outputs are classified EXPLORATORY_ONLY and the promotion guard
(renquant_artifacts.validation.ValidateArtifactManifestTask, wired via
``provenance_dir``) rejects them at the real promotion boundary.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_sim_104 import (  # noqa: E402
    _normalize_remote,
    _resolve_strategy_config,
    _verify_pin,
    load_experiment_manifest,
    reject_exploratory_promotion,
    verify_and_classify_experiment,
    write_candidate_artifact_manifest,
    write_experiment_classification,
)


_FAKE_COMMIT = "abc123def456" * 3 + "abc123def456"[:4]  # 40 hex chars
_FAKE_REMOTE = "https://github.com/hallovorld/renquant-strategy-104"


def _make_lock(
    tmp_path: Path,
    *,
    commit: str = _FAKE_COMMIT,
    remote: str = _FAKE_REMOTE,
    local_path: str | None = None,
    include_commit: bool = True,
    include_remote: bool = True,
) -> None:
    entry = {
        "name": "renquant-strategy-104",
        "local_path": local_path or str(tmp_path / "pinned_repo"),
    }
    if include_commit:
        entry["commit"] = commit
    if include_remote:
        entry["remote"] = remote
    lock = {"subrepos": [entry]}
    (tmp_path / "subrepos.lock.json").write_text(json.dumps(lock))


_DEFAULT_PINS = {
    "data_snapshot": "d0000000",
    "model_artifact": "m0000000",
    "strategy_config": "s0000000",
    "pipeline_version": "p0000000",
    "calendar_universe": "c0000000",
}


def _register_manifest(tmp_path: Path, manifest_path: Path, experiment_id: str) -> None:
    """Append a registry entry so load_experiment_manifest's registration
    check (renquant_artifacts.verify_manifest_registered) passes."""
    index_path = tmp_path / "experiments" / "manifests" / "INDEX.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    digest = "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    index[experiment_id] = {"digest": digest, "path": str(manifest_path)}
    index_path.write_text(json.dumps(index))


def _make_manifest(
    tmp_path: Path,
    *,
    config_path: str | None = None,
    config_content: bytes = b'{"exp": true}',
    status: str = "ACTIVE",
    experiment_id: str = "sweep-001",
    pins: dict | None = None,
    extra: dict | None = None,
    register: bool = True,
    data_manifest_path: str | None = None,
    model_artifact_path: str | None = None,
) -> tuple[Path, Path]:
    """Create a config file + data manifest + model artifact + matching
    (registered by default) experiment manifest.

    Returns (manifest_path, config_path).
    """
    cfg_path = tmp_path / (config_path or "my_sweep.json")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_bytes(config_content)
    digest = "sha256:" + hashlib.sha256(config_content).hexdigest()

    dm_path = tmp_path / (data_manifest_path or f"{experiment_id}_data_manifest.json")
    dm_path.parent.mkdir(parents=True, exist_ok=True)
    if not dm_path.exists():
        dm_path.write_text(json.dumps({
            "dataset_id": f"{experiment_id}-ds",
            "schema_version": "1.0",
            "fingerprint": "sha256:datapin",
            "uri": "sqlite:///data/runs.alpaca.db",
            "asset_class": "us_equity",
        }))

    ma_path = tmp_path / (model_artifact_path or f"{experiment_id}_model.pt")
    ma_path.parent.mkdir(parents=True, exist_ok=True)
    if not ma_path.exists():
        ma_path.write_bytes(b"weights")

    manifest = {
        "experiment_id": experiment_id,
        "config_path": str(cfg_path),
        "config_digest": digest,
        "status": status,
        "pins": pins if pins is not None else dict(_DEFAULT_PINS),
        "data_manifest_path": str(dm_path),
        "model_artifact_path": str(ma_path),
    }
    if extra:
        manifest.update(extra)
    manifest_path = tmp_path / "experiments" / "manifests" / f"{experiment_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if register:
        _register_manifest(tmp_path, manifest_path, experiment_id)
    return manifest_path, cfg_path


class TestVerifyPinRemote:
    """_verify_pin checks HEAD, dirty state, and remote URL."""

    def test_missing_commit_in_lock_fails(self, tmp_path):
        errors = _verify_pin(tmp_path, "", _FAKE_REMOTE)
        assert any("no commit hash" in e for e in errors)

    def test_missing_remote_in_lock_fails(self, tmp_path):
        errors = _verify_pin(tmp_path, _FAKE_COMMIT, "")
        assert any("no remote URL" in e for e in errors)

    def test_wrong_remote_detected(self, tmp_path):
        with mock.patch("run_sim_104._git") as mock_git:
            def _side(repo, *args):
                if args[0] == "log":
                    return _FAKE_COMMIT
                if args[0] == "status":
                    return ""
                if args[0] == "remote":
                    return "https://github.com/WRONG/wrong-repo"
                return ""
            mock_git.side_effect = _side
            errors = _verify_pin(tmp_path, _FAKE_COMMIT, _FAKE_REMOTE)
        assert any("remote URL mismatch" in e for e in errors)

    def test_matching_remote_passes(self, tmp_path):
        with mock.patch("run_sim_104._git") as mock_git:
            def _side(repo, *args):
                if args[0] == "log":
                    return _FAKE_COMMIT
                if args[0] == "status":
                    return ""
                if args[0] == "remote":
                    return _FAKE_REMOTE
                return ""
            mock_git.side_effect = _side
            errors = _verify_pin(tmp_path, _FAKE_COMMIT, _FAKE_REMOTE)
        assert errors == []

    def test_remote_url_normalized(self):
        assert _normalize_remote("https://github.com/A/B.git") == \
               _normalize_remote("https://github.com/a/b")
        assert _normalize_remote("https://github.com/A/B/") == \
               _normalize_remote("https://github.com/a/b")

    def test_remote_read_failure_is_error(self, tmp_path):
        import subprocess
        with mock.patch("run_sim_104._git") as mock_git:
            def _side(repo, *args):
                if args[0] == "log":
                    return _FAKE_COMMIT
                if args[0] == "status":
                    return ""
                if args[0] == "remote":
                    raise subprocess.CalledProcessError(1, "git")
                return ""
            mock_git.side_effect = _side
            errors = _verify_pin(tmp_path, _FAKE_COMMIT, _FAKE_REMOTE)
        assert any("could not read remote URL" in e for e in errors)


class TestStrictPinnedDefault:
    """Default mode: ALL configs resolve from the pinned subrepo."""

    def test_config_resolves_from_pin(self, tmp_path):
        pinned_dir = tmp_path / "pinned_repo"
        (pinned_dir / "configs").mkdir(parents=True)
        pinned_cfg = pinned_dir / "configs" / "strategy_config.json"
        pinned_cfg.write_text("{}")
        _make_lock(tmp_path, local_path=str(pinned_dir))

        strategy_dir = tmp_path / "umbrella_local"
        strategy_dir.mkdir()
        (strategy_dir / "strategy_config.json").write_text('{"local": true}')

        with mock.patch("run_sim_104._verify_pin", return_value=[]):
            cfg_path, source, manifest = _resolve_strategy_config(
                tmp_path, strategy_dir, "strategy_config.json",
            )
        assert cfg_path == pinned_cfg
        assert source == "PINNED"
        assert manifest is None

    def test_any_config_name_goes_through_pin(self, tmp_path):
        pinned_dir = tmp_path / "pinned_repo"
        (pinned_dir / "configs").mkdir(parents=True)
        cfg = pinned_dir / "configs" / "strategy_config.sim_BB_09.json"
        cfg.write_text("{}")
        _make_lock(tmp_path, local_path=str(pinned_dir))

        strategy_dir = tmp_path / "umbrella_local"
        strategy_dir.mkdir()
        (strategy_dir / "strategy_config.sim_BB_09.json").write_text("{}")

        with mock.patch("run_sim_104._verify_pin", return_value=[]):
            cfg_path, source, manifest = _resolve_strategy_config(
                tmp_path, strategy_dir, "strategy_config.sim_BB_09.json",
            )
        assert cfg_path == cfg
        assert source == "PINNED"

    def test_fails_closed_when_config_missing_from_pin(self, tmp_path):
        pinned_dir = tmp_path / "pinned_repo"
        (pinned_dir / "configs").mkdir(parents=True)
        _make_lock(tmp_path, local_path=str(pinned_dir))

        strategy_dir = tmp_path / "umbrella_local"
        strategy_dir.mkdir()
        (strategy_dir / "strategy_config.json").write_text("{}")

        with mock.patch("run_sim_104._verify_pin", return_value=[]):
            with pytest.raises(SystemExit) as exc:
                _resolve_strategy_config(
                    tmp_path, strategy_dir, "strategy_config.json",
                )
        assert exc.value.code == 1

    def test_fails_closed_when_pin_errors(self, tmp_path):
        pinned_dir = tmp_path / "pinned_repo"
        (pinned_dir / "configs").mkdir(parents=True)
        (pinned_dir / "configs" / "strategy_config.json").write_text("{}")
        _make_lock(tmp_path, local_path=str(pinned_dir))

        strategy_dir = tmp_path / "umbrella_local"
        strategy_dir.mkdir()

        with mock.patch("run_sim_104._verify_pin",
                        return_value=["HEAD mismatch"]):
            with pytest.raises(SystemExit) as exc:
                _resolve_strategy_config(
                    tmp_path, strategy_dir, "strategy_config.json",
                )
        assert exc.value.code == 1

    def test_fails_closed_when_no_lock_file(self, tmp_path):
        strategy_dir = tmp_path / "umbrella_local"
        strategy_dir.mkdir()
        (strategy_dir / "strategy_config.json").write_text("{}")

        with pytest.raises(SystemExit) as exc:
            _resolve_strategy_config(
                tmp_path, strategy_dir, "strategy_config.json",
            )
        assert exc.value.code == 1

    def test_fails_closed_when_lock_missing_metadata(self, tmp_path):
        pinned_dir = tmp_path / "pinned_repo"
        (pinned_dir / "configs").mkdir(parents=True)
        (pinned_dir / "configs" / "strategy_config.json").write_text("{}")
        _make_lock(tmp_path, local_path=str(pinned_dir),
                   include_commit=False, include_remote=True)

        strategy_dir = tmp_path / "umbrella_local"
        strategy_dir.mkdir()

        with pytest.raises(SystemExit) as exc:
            _resolve_strategy_config(
                tmp_path, strategy_dir, "strategy_config.json",
            )
        assert exc.value.code == 1


class TestExperimentManifest:
    """--experiment-manifest: registered manifest with digest verification."""

    def test_manifest_resolves_config(self, tmp_path):
        manifest_path, cfg_path = _make_manifest(tmp_path)

        cfg, source, manifest = _resolve_strategy_config(
            tmp_path, tmp_path, "ignored",
            experiment_manifest=str(manifest_path),
        )
        assert cfg == cfg_path
        assert source == "EXPLORATORY_ONLY"
        assert manifest is not None
        assert manifest["experiment_id"] == "sweep-001"
        # internal resolved fields the caller (main()) relies on
        assert manifest["_manifest_digest"].startswith("sha256:")
        assert Path(manifest["_data_manifest_path"]).exists()
        assert Path(manifest["_model_artifact_path"]).exists()

    def test_manifest_fails_on_missing_keys(self, tmp_path):
        cfg = tmp_path / "my_sweep.json"
        cfg.write_bytes(b'{"exp": true}')
        manifest_path = tmp_path / "bad_manifest.json"
        manifest_path.write_text(json.dumps({"experiment_id": "x"}))

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_manifest_fails_on_invalid_status(self, tmp_path):
        manifest_path, _ = _make_manifest(tmp_path, status="INVALID")
        # Manually fix status to be invalid in the file (re-register since
        # editing the file changes its digest)
        raw = json.loads(manifest_path.read_text())
        raw["status"] = "INVALID"
        manifest_path.write_text(json.dumps(raw))
        _register_manifest(tmp_path, manifest_path, raw["experiment_id"])

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_manifest_fails_on_retired(self, tmp_path):
        manifest_path, _ = _make_manifest(
            tmp_path, experiment_id="old-001", status="RETIRED",
        )

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_manifest_fails_on_digest_mismatch(self, tmp_path):
        manifest_path, cfg_path = _make_manifest(tmp_path)
        cfg_path.write_bytes(b'{"modified": true}')

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_manifest_fails_on_missing_config(self, tmp_path):
        manifest_path, _ = _make_manifest(tmp_path)
        raw = json.loads(manifest_path.read_text())
        raw["config_path"] = str(tmp_path / "nonexistent.json")
        manifest_path.write_text(json.dumps(raw))
        _register_manifest(tmp_path, manifest_path, raw["experiment_id"])

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_manifest_fails_when_data_manifest_path_missing(self, tmp_path):
        manifest_path, _ = _make_manifest(tmp_path)
        raw = json.loads(manifest_path.read_text())
        raw["data_manifest_path"] = str(tmp_path / "nope_data_manifest.json")
        manifest_path.write_text(json.dumps(raw))
        _register_manifest(tmp_path, manifest_path, raw["experiment_id"])

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_manifest_fails_when_model_artifact_path_missing(self, tmp_path):
        manifest_path, _ = _make_manifest(tmp_path)
        raw = json.loads(manifest_path.read_text())
        raw["model_artifact_path"] = str(tmp_path / "nope_model.pt")
        manifest_path.write_text(json.dumps(raw))
        _register_manifest(tmp_path, manifest_path, raw["experiment_id"])

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_manifest_skips_pin_verification_in_load(self, tmp_path):
        """load_experiment_manifest itself never touches git -- pin
        verification against the code checkout happens later, in
        verify_and_classify_experiment / verify_experiment_pins."""
        manifest_path, _ = _make_manifest(tmp_path)

        with mock.patch("run_sim_104._verify_pin") as mock_verify:
            _resolve_strategy_config(
                tmp_path, tmp_path, "ignored",
                experiment_manifest=str(manifest_path),
            )
        mock_verify.assert_not_called()

    def test_manifest_relative_config_path(self, tmp_path):
        sub = tmp_path / "backtesting" / "renquant_104"
        sub.mkdir(parents=True)
        cfg = sub / "my_sweep.json"
        content = b'{"exp": true}'
        cfg.write_bytes(content)
        manifest_path, _ = _make_manifest(
            tmp_path, experiment_id="rel-001",
            config_path="backtesting/renquant_104/my_sweep.json",
            config_content=content,
        )
        # _make_manifest wrote config_path as an absolute path; rewrite it
        # relative to repo_root to exercise the relative-path branch.
        raw = json.loads(manifest_path.read_text())
        raw["config_path"] = "backtesting/renquant_104/my_sweep.json"
        manifest_path.write_text(json.dumps(raw))
        _register_manifest(tmp_path, manifest_path, raw["experiment_id"])

        cfg_path, source, manifest = _resolve_strategy_config(
            tmp_path, tmp_path, "ignored",
            experiment_manifest=str(manifest_path),
        )
        assert cfg_path == cfg
        assert source == "EXPLORATORY_ONLY"

    def test_manifest_with_custom_pins(self, tmp_path):
        custom_pins = {
            **_DEFAULT_PINS,
            "renquant-model": "abc123",
            "renquant-pipeline": "def456",
        }
        manifest_path, _ = _make_manifest(tmp_path, pins=custom_pins)
        manifest = load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert manifest["pins"]["renquant-model"] == "abc123"
        assert manifest["pins"]["renquant-pipeline"] == "def456"

    def test_manifest_fails_on_missing_pins(self, tmp_path):
        manifest_path, _ = _make_manifest(tmp_path, extra={"pins": None})
        raw = json.loads(manifest_path.read_text())
        del raw["pins"]
        manifest_path.write_text(json.dumps(raw))
        _register_manifest(tmp_path, manifest_path, raw["experiment_id"])

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_manifest_fails_on_incomplete_pins(self, tmp_path):
        manifest_path, _ = _make_manifest(
            tmp_path,
            pins={"data_snapshot": "d000"},
        )
        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_manifest_fails_on_non_dict_pins(self, tmp_path):
        manifest_path, _ = _make_manifest(tmp_path, extra={"pins": "not-a-dict"})

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_manifest_path_must_be_under_experiments_manifests(self, tmp_path):
        manifest_path, _ = _make_manifest(tmp_path)
        rogue = tmp_path / "rogue" / "manifest.json"
        rogue.parent.mkdir(parents=True)
        import shutil
        shutil.copy2(manifest_path, rogue)
        with pytest.raises(SystemExit) as exc:
            _resolve_strategy_config(
                tmp_path, tmp_path, "ignored",
                experiment_manifest=str(rogue),
            )
        assert exc.value.code == 1


class TestManifestRegistry:
    """A manifest must be a REGISTERED record, not merely path-restricted."""

    def test_unregistered_manifest_fails_closed(self, tmp_path):
        manifest_path, _ = _make_manifest(tmp_path, register=False)

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_registered_manifest_with_tampered_content_fails_closed(self, tmp_path):
        """Editing a manifest AFTER registration (without re-registering)
        must be rejected -- registration binds the file's digest, not its
        location or experiment_id alone."""
        manifest_path, _ = _make_manifest(tmp_path)
        raw = json.loads(manifest_path.read_text())
        raw["status"] = "COMPLETED"  # innocuous-looking edit, still tampering
        manifest_path.write_text(json.dumps(raw))
        # deliberately do NOT re-register

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_registered_manifest_under_different_experiment_id_fails(self, tmp_path):
        manifest_path, _ = _make_manifest(tmp_path, register=False)
        # Register under the WRONG experiment_id / digest
        index_path = tmp_path / "experiments" / "manifests" / "INDEX.json"
        index_path.write_text(json.dumps({
            "some-other-experiment": {"digest": "sha256:unrelated", "path": "x"},
        }))

        with pytest.raises(SystemExit) as exc:
            load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert exc.value.code == 1

    def test_correctly_registered_manifest_passes(self, tmp_path):
        manifest_path, _ = _make_manifest(tmp_path)
        manifest = load_experiment_manifest(manifest_path, repo_root=tmp_path)
        assert manifest["experiment_id"] == "sweep-001"


class TestVerifyAndClassifyExperiment:
    """F-7 r7: all 5 pins verified against the actual environment BEFORE
    the EXPLORATORY_ONLY classification is written, and the marker landing
    directory is what a real artifact manifest must set as provenance_dir."""

    def _fixture(self, tmp_path):
        strategy_remote = "https://github.com/hallovorld/renquant-strategy-104"
        pipeline_remote = "https://github.com/hallovorld/renquant-pipeline"

        import subprocess

        def _init_repo(path, remote):
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-q"], cwd=path, check=True)
            subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
            subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
            (path / "f.txt").write_text("x")
            subprocess.run(["git", "add", "."], cwd=path, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
            subprocess.run(["git", "remote", "add", "origin", remote], cwd=path, check=True)
            return subprocess.check_output(
                ["git", "-C", str(path), "log", "-1", "--format=%H"], text=True,
            ).strip()

        strategy_commit = _init_repo(tmp_path / "renquant-strategy-104", strategy_remote)
        pipeline_commit = _init_repo(tmp_path / "renquant-pipeline", pipeline_remote)
        (tmp_path / "subrepos.lock.json").write_text(json.dumps({
            "subrepos": [
                {"name": "renquant-strategy-104", "local_path": str(tmp_path / "renquant-strategy-104"),
                 "commit": strategy_commit, "remote": strategy_remote},
                {"name": "renquant-pipeline", "local_path": str(tmp_path / "renquant-pipeline"),
                 "commit": pipeline_commit, "remote": pipeline_remote},
            ]
        }))

        from renquant_artifacts import hash_jsonable
        from renquant_common.model_fingerprint import artifact_sha256
        universe = ["AAPL", "MSFT"]
        pins = {
            "strategy_config": strategy_commit,
            "pipeline_version": pipeline_commit,
            "data_snapshot": "sha256:datapin",
            "model_artifact": artifact_sha256(tmp_path / "model.pt")
            if (tmp_path / "model.pt").exists() else None,
            "calendar_universe": hash_jsonable(sorted(set(universe))),
        }
        (tmp_path / "model.pt").write_bytes(b"weights")
        pins["model_artifact"] = artifact_sha256(tmp_path / "model.pt")

        manifest_path, _ = _make_manifest(
            tmp_path, pins=pins, model_artifact_path="model.pt",
        )
        manifest_data = load_experiment_manifest(manifest_path, repo_root=tmp_path)
        config = {"watchlist": universe}
        strategy_dir = tmp_path / "strategy_dir"
        return manifest_data, config, strategy_dir

    def test_clean_pins_write_classification(self, tmp_path):
        manifest_data, config, strategy_dir = self._fixture(tmp_path)
        output_dir = verify_and_classify_experiment(
            manifest_data, config,
            repo_root=tmp_path, strategy_dir=strategy_dir,
            experiment_manifest_arg="experiments/manifests/sweep-001.json",
            config_digest="sha256:cfg",
        )
        marker = output_dir / "_experiment_classification.json"
        assert marker.exists()
        raw = json.loads(marker.read_text())
        assert raw["classification"] == "EXPLORATORY_ONLY"
        assert raw["experiment_id"] == "sweep-001"

    def test_pin_mismatch_exits_before_writing_classification(self, tmp_path):
        manifest_data, config, strategy_dir = self._fixture(tmp_path)
        # Corrupt the universe so calendar_universe no longer verifies.
        config["watchlist"] = ["GOOG", "TSLA"]

        with pytest.raises(SystemExit) as exc:
            verify_and_classify_experiment(
                manifest_data, config,
                repo_root=tmp_path, strategy_dir=strategy_dir,
                experiment_manifest_arg="experiments/manifests/sweep-001.json",
                config_digest="sha256:cfg",
            )
        assert exc.value.code == 1
        output_dir = strategy_dir / "artifacts" / "experiments" / "sweep-001"
        assert not (output_dir / "_experiment_classification.json").exists()

    def test_dirty_pipeline_checkout_blocks_run(self, tmp_path):
        manifest_data, config, strategy_dir = self._fixture(tmp_path)
        (tmp_path / "renquant-pipeline" / "f.txt").write_text("dirty")

        with pytest.raises(SystemExit) as exc:
            verify_and_classify_experiment(
                manifest_data, config,
                repo_root=tmp_path, strategy_dir=strategy_dir,
                experiment_manifest_arg="experiments/manifests/sweep-001.json",
                config_digest="sha256:cfg",
            )
        assert exc.value.code == 1


class TestCandidateArtifactManifestEmission:
    """F-7 follow-up (Codex round-3 review): run_sim_104.py must actually
    EMIT an artifact manifest for a registered experiment's output, with
    ``provenance`` baked in by this producer -- not just log the reference
    and leave manifest construction to a disconnected, later caller.

    "run_sim_104.py only logs the reference returned by
    build_experiment_provenance_reference(); it does not emit an artifact
    manifest or bind that reference into the registry publication path."
    """

    def _sim_metrics(self) -> dict:
        return {"apy": 0.081, "sharpe": 0.7, "max_dd": -0.12, "n_trades": 42}

    def test_writes_manifest_with_baked_in_provenance(self, tmp_path):
        import renquant_artifacts

        manifest_data, config, strategy_dir = TestVerifyAndClassifyExperiment()._fixture(tmp_path)
        output_dir = verify_and_classify_experiment(
            manifest_data, config,
            repo_root=tmp_path, strategy_dir=strategy_dir,
            experiment_manifest_arg="experiments/manifests/sweep-001.json",
            config_digest="sha256:cfg",
        )

        out_path = write_candidate_artifact_manifest(
            output_dir, manifest_data=manifest_data, sim_metrics=self._sim_metrics(),
        )

        assert out_path == output_dir / "candidate_artifact_manifest.json"
        written = json.loads(out_path.read_text())
        assert written["provenance"] == renquant_artifacts.build_experiment_provenance_reference(
            output_dir, manifest_data["_registry_index_path"],
        )
        assert written["provenance"]["kind"] == "experiment"
        assert written["metrics"]["apy"] == pytest.approx(0.081)
        assert written["metrics"]["n_trades"] == 42
        assert written["metrics"]["accepted"] is False

    def test_written_manifest_is_rejected_for_promotion(self, tmp_path):
        """The manifest this function writes must be rejected by the REAL
        validate_artifact_manifest -- proving the baked-in provenance is
        the honest ``kind="experiment"`` reference, not a
        ``kind="none"``-style bypass. This is what makes a later dishonest
        substitution (hand-building a different manifest that declares
        kind="none" for the SAME output) a detectable divergence from the
        real record this function produced."""
        import renquant_artifacts

        manifest_data, config, strategy_dir = TestVerifyAndClassifyExperiment()._fixture(tmp_path)
        output_dir = verify_and_classify_experiment(
            manifest_data, config,
            repo_root=tmp_path, strategy_dir=strategy_dir,
            experiment_manifest_arg="experiments/manifests/sweep-001.json",
            config_digest="sha256:cfg",
        )
        out_path = write_candidate_artifact_manifest(
            output_dir, manifest_data=manifest_data, sim_metrics=self._sim_metrics(),
        )
        written = json.loads(out_path.read_text())

        with pytest.raises(ValueError, match="registered experiment"):
            renquant_artifacts.validate_artifact_manifest(written)

    def test_dishonest_kind_none_substitution_over_this_output_is_rejected(self, tmp_path):
        """THE exact end-to-end proof that wiring run_sim_104.py's real
        output into publication closes the round-3 bypass: take the SAME
        real experiment output this function just wrote a manifest for,
        but construct a DIFFERENT candidate manifest by hand that
        dishonestly declares provenance={"kind": "none"} instead of using
        write_candidate_artifact_manifest's honest reference. Because the
        dishonest manifest's own local_artifact_path still resolves under
        the real output directory (it has to, to be useful), renquant-
        artifacts' round-3 fix (_verify_none_provenance) rejects it too."""
        import renquant_artifacts

        manifest_data, config, strategy_dir = TestVerifyAndClassifyExperiment()._fixture(tmp_path)
        output_dir = verify_and_classify_experiment(
            manifest_data, config,
            repo_root=tmp_path, strategy_dir=strategy_dir,
            experiment_manifest_arg="experiments/manifests/sweep-001.json",
            config_digest="sha256:cfg",
        )
        write_candidate_artifact_manifest(
            output_dir, manifest_data=manifest_data, sim_metrics=self._sim_metrics(),
        )

        dishonest_manifest = {
            "artifact_id": f"{manifest_data['experiment_id']}-candidate",
            "model_family": "gbdt-panel-ltr",
            "strategy": "renquant_104",
            "fingerprint": "sha256:dishonest",
            "uri": "object://renquant-artifacts/dishonest.json",
            "local_artifact_path": str(output_dir / "candidate_artifact_manifest.json"),
            "promotion_status": "prod",
            "metrics": {"accepted": True},
            "provenance": {"kind": "none"},
        }

        with pytest.raises(ValueError, match="EXPLORATORY_ONLY classification record"):
            renquant_artifacts.validate_artifact_manifest(dishonest_manifest)


class TestExploratoryClassification:
    """Durable EXPLORATORY_ONLY classification file + promotion guard."""

    def test_write_classification_file(self, tmp_path):
        output_dir = tmp_path / "output"
        cls_path = write_experiment_classification(
            output_dir,
            experiment_id="sweep-001",
            manifest_path="manifests/sweep-001.json",
            manifest_digest="sha256:mfdigest",
            config_digest="sha256:abc123",
        )
        assert cls_path.exists()
        raw = json.loads(cls_path.read_text())
        assert raw["classification"] == "EXPLORATORY_ONLY"
        assert raw["experiment_id"] == "sweep-001"
        assert raw["manifest_digest"] == "sha256:mfdigest"

    def test_reject_exploratory_promotion(self, tmp_path):
        output_dir = tmp_path / "output"
        write_experiment_classification(
            output_dir,
            experiment_id="sweep-001",
            manifest_path="manifests/sweep-001.json",
            manifest_digest="sha256:mfdigest",
            config_digest="sha256:abc123",
        )
        with pytest.raises(ValueError, match="EXPLORATORY_ONLY"):
            reject_exploratory_promotion(output_dir)

    def test_marker_required_directory_exists_but_no_marker_now_rejected(self, tmp_path):
        """Behavior CHANGED (renquant-artifacts#24, Codex 2026-07-14
        follow-up): a directory with no classification marker used to be
        silently accepted ("not exploratory, proceed") -- that was flagged
        as "a second layer of the same bypass": a caller presenting
        provenance could point at an empty/fake directory and sail through.
        A missing marker is now unverifiable provenance, not proof of a
        clean run, so this raises."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        with pytest.raises(ValueError, match="no experiment classification"):
            reject_exploratory_promotion(output_dir)

    def test_marker_required_nonexistent_directory_now_rejected(self, tmp_path):
        """Same behavior change as above, for a directory that does not
        exist at all."""
        with pytest.raises(ValueError, match="no experiment classification"):
            reject_exploratory_promotion(tmp_path / "nonexistent")

    def test_reexported_from_renquant_artifacts(self):
        """There is exactly ONE implementation of the marker contract --
        run_sim_104 re-exports it rather than redefining it, so the
        promotion-boundary enforcement in renquant_artifacts.validation
        checks the SAME function this script's output satisfies."""
        import renquant_artifacts
        assert reject_exploratory_promotion is renquant_artifacts.reject_exploratory_promotion
        assert write_experiment_classification is renquant_artifacts.write_experiment_classification


class TestPromotionBoundaryIntegration:
    """Proves run_sim_104's EXPLORATORY_ONLY marker is REAL enforcement at
    the actual promotion boundary -- not a mock, and not an unreferenced
    helper (Codex review 2026-07-14, finding 1).

    Updated for the renquant-artifacts#24 follow-up fix (Codex 2026-07-14):
    "the promotion guard is bypassable because provenance is optional and
    self-declared" -- a flat ``provenance_dir`` string is no longer accepted
    on its own; the candidate manifest must carry a required, typed
    ``provenance`` record bound to the immutable manifest-registry index
    (built via ``renquant_artifacts.build_experiment_provenance_reference``,
    the same helper ``verify_and_classify_experiment`` uses so there is
    exactly one implementation of the reference's shape).
    """

    def test_run_sim_104_marker_blocks_real_validate_artifact_manifest(self, tmp_path):
        import renquant_artifacts

        output_dir = tmp_path / "strategy_dir" / "artifacts" / "experiments" / "sweep-001"
        write_experiment_classification(
            output_dir,
            experiment_id="sweep-001",
            manifest_path="experiments/manifests/sweep-001.json",
            manifest_digest="sha256:mfdigest",
            config_digest="sha256:abc123",
        )
        index_path = tmp_path / "experiments" / "manifests" / "INDEX.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(
            {"sweep-001": {"digest": "sha256:mfdigest", "path": "experiments/manifests/sweep-001.json"}}
        ))

        candidate_manifest = {
            "artifact_id": "candidate-from-sweep-001",
            "model_family": "gbdt-panel-ltr",
            "strategy": "renquant_104",
            "fingerprint": "sha256:candidate",
            "uri": "object://renquant-artifacts/candidate.json",
            "promotion_status": "prod",
            "metrics": {"accepted": True},
            "provenance": renquant_artifacts.build_experiment_provenance_reference(
                output_dir, index_path,
            ),
        }

        with pytest.raises(ValueError, match="registered experiment"):
            renquant_artifacts.validate_artifact_manifest(candidate_manifest)

    def test_omitted_provenance_bypass_closed(self, tmp_path):
        """The exact bypass Codex's follow-up review reported: a candidate
        manifest built from a genuinely EXPLORATORY_ONLY run's output that
        simply omits 'provenance' entirely must be rejected, not silently
        accepted."""
        import renquant_artifacts

        output_dir = tmp_path / "strategy_dir" / "artifacts" / "experiments" / "sweep-002"
        write_experiment_classification(
            output_dir,
            experiment_id="sweep-002",
            manifest_path="experiments/manifests/sweep-002.json",
            manifest_digest="sha256:mfdigest2",
            config_digest="sha256:abc123",
        )
        index_path = tmp_path / "experiments" / "manifests" / "INDEX.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(
            {"sweep-002": {"digest": "sha256:mfdigest2", "path": "experiments/manifests/sweep-002.json"}}
        ))

        candidate_manifest = {
            "artifact_id": "candidate-from-sweep-002",
            "model_family": "gbdt-panel-ltr",
            "strategy": "renquant_104",
            "fingerprint": "sha256:candidate2",
            "uri": "object://renquant-artifacts/candidate2.json",
            "promotion_status": "prod",
            "metrics": {"accepted": True},
            # provenance deliberately omitted
        }

        with pytest.raises(ValueError, match="missing a required 'provenance' record"):
            renquant_artifacts.validate_artifact_manifest(candidate_manifest)

    def test_falsified_provenance_pointing_at_empty_directory_rejected(self, tmp_path):
        """A candidate manifest claims experiment provenance pointing at a
        directory that does not carry the real run's classification record
        (e.g. an empty decoy) -- must be rejected, not treated as clean."""
        import renquant_artifacts

        decoy_dir = tmp_path / "strategy_dir" / "artifacts" / "experiments" / "decoy"
        decoy_dir.mkdir(parents=True)
        index_path = tmp_path / "experiments" / "manifests" / "INDEX.json"

        candidate_manifest = {
            "artifact_id": "candidate-decoy",
            "model_family": "gbdt-panel-ltr",
            "strategy": "renquant_104",
            "fingerprint": "sha256:candidate3",
            "uri": "object://renquant-artifacts/candidate3.json",
            "promotion_status": "prod",
            "metrics": {"accepted": True},
            "provenance": renquant_artifacts.build_experiment_provenance_reference(
                decoy_dir, index_path,
            ),
        }

        with pytest.raises(ValueError, match="no experiment classification"):
            renquant_artifacts.validate_artifact_manifest(candidate_manifest)


class TestLegacyScriptRetirement:
    """Legacy scripts are retired and fail immediately."""

    def test_doe_orchestrate_bb_retired(self):
        script = _REPO / "scripts" / "_doe_orchestrate_bb.sh"
        content = script.read_text()
        assert "RETIRED" in content
        assert "exit 1" in content

    def test_run_parallel_after_trail015_retired(self):
        script = _REPO / "scripts" / "run_parallel_after_trail015.sh"
        content = script.read_text()
        assert "RETIRED" in content
        assert "exit 1" in content


class TestConfigFingerprintFormat:
    def test_run_sim_104_source_uses_full_prefixed_digest(self):
        src = (_REPO / "scripts" / "run_sim_104.py").read_text()
        assert '"sha256:" + hashlib.sha256(cfg_bytes).hexdigest()' in src
        assert ".hexdigest()[:16]" not in src

    def test_fingerprint_format_shape(self):
        cfg_bytes = b'{"foo": "bar"}'
        digest = "sha256:" + hashlib.sha256(cfg_bytes).hexdigest()
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 64
