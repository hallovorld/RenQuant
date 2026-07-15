"""Tests for scripts/run_sim_104.py pin-verification resolver."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

import sys

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from run_sim_104 import _resolve_strategy_config, _verify_pin  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


FAKE_COMMIT = "abc123def456" * 3 + "abc123def456"[:4]  # 40 hex chars
FAKE_REMOTE = "https://github.com/hallovorld/renquant-strategy-104"


def _make_lock(tmp_path: Path, *, commit: str, local_path: str) -> None:
    lock = {
        "subrepos": [
            {
                "name": "renquant-strategy-104",
                "commit": commit,
                "remote": FAKE_REMOTE,
                "local_path": local_path,
            }
        ]
    }
    (tmp_path / "subrepos.lock.json").write_text(json.dumps(lock))


def _make_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "strategy_config.json").write_text('{"foo": 1}')


def _mock_git_clean(commit=FAKE_COMMIT, remote=FAKE_REMOTE):
    def _side(_repo, *args):
        if args[0] == "log":
            return commit
        if args[0] == "status":
            return ""
        if args[0] == "remote":
            return remote
        return ""
    return _side


# ---------------------------------------------------------------------------
# _verify_pin
# ---------------------------------------------------------------------------


class TestVerifyPin:
    def test_clean_match(self, tmp_path: Path) -> None:
        with mock.patch("run_sim_104._git", side_effect=_mock_git_clean()):
            errors = _verify_pin(tmp_path, FAKE_COMMIT, FAKE_REMOTE)
        assert errors == []

    def test_head_mismatch(self, tmp_path: Path) -> None:
        wrong = "0" * 40
        with mock.patch("run_sim_104._git",
                        side_effect=_mock_git_clean(commit=wrong)):
            errors = _verify_pin(tmp_path, FAKE_COMMIT, FAKE_REMOTE)
        assert len(errors) == 1
        assert "does not match lock commit" in errors[0]

    def test_dirty_checkout(self, tmp_path: Path) -> None:
        def _side(_repo, *args):
            if args[0] == "log":
                return FAKE_COMMIT
            if args[0] == "status":
                return " M some_file.py"
            if args[0] == "remote":
                return FAKE_REMOTE
            return ""
        with mock.patch("run_sim_104._git", side_effect=_side):
            errors = _verify_pin(tmp_path, FAKE_COMMIT, FAKE_REMOTE)
        assert len(errors) == 1
        assert "dirty" in errors[0]

    def test_git_failure(self, tmp_path: Path) -> None:
        with mock.patch(
            "run_sim_104._git",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            errors = _verify_pin(tmp_path, FAKE_COMMIT, FAKE_REMOTE)
        assert len(errors) == 1
        assert "git metadata failed" in errors[0]

    def test_wrong_remote(self, tmp_path: Path) -> None:
        with mock.patch("run_sim_104._git",
                        side_effect=_mock_git_clean(
                            remote="https://github.com/WRONG/wrong-repo")):
            errors = _verify_pin(tmp_path, FAKE_COMMIT, FAKE_REMOTE)
        assert any("remote URL mismatch" in e for e in errors)

    def test_missing_commit_in_lock(self, tmp_path: Path) -> None:
        errors = _verify_pin(tmp_path, "", FAKE_REMOTE)
        assert any("no commit hash" in e for e in errors)

    def test_missing_remote_in_lock(self, tmp_path: Path) -> None:
        errors = _verify_pin(tmp_path, FAKE_COMMIT, "")
        assert any("no remote URL" in e for e in errors)


# ---------------------------------------------------------------------------
# _resolve_strategy_config
# ---------------------------------------------------------------------------


class TestResolveStrategyConfig:
    def test_pinned_clean_match(self, tmp_path: Path) -> None:
        strat_dir = tmp_path / "subrepo"
        _make_config(strat_dir / "configs")
        _make_lock(tmp_path, commit=FAKE_COMMIT, local_path=str(strat_dir))

        with mock.patch("run_sim_104._verify_pin", return_value=[]):
            cfg, source, manifest = _resolve_strategy_config(
                tmp_path, tmp_path / "backtesting" / "renquant_104",
                "strategy_config.json",
            )
        assert source == "PINNED"
        assert cfg == strat_dir / "configs" / "strategy_config.json"
        assert manifest is None

    def test_head_mismatch_exits(self, tmp_path: Path) -> None:
        strat_dir = tmp_path / "subrepo"
        _make_config(strat_dir / "configs")
        _make_lock(tmp_path, commit=FAKE_COMMIT, local_path=str(strat_dir))

        with (
            mock.patch(
                "run_sim_104._verify_pin",
                return_value=["HEAD 000000000000 does not match lock commit"],
            ),
            pytest.raises(SystemExit, match="1"),
        ):
            _resolve_strategy_config(
                tmp_path, tmp_path / "backtesting" / "renquant_104",
                "strategy_config.json",
            )

    def test_dirty_checkout_exits(self, tmp_path: Path) -> None:
        strat_dir = tmp_path / "subrepo"
        _make_config(strat_dir / "configs")
        _make_lock(tmp_path, commit=FAKE_COMMIT, local_path=str(strat_dir))

        with (
            mock.patch(
                "run_sim_104._verify_pin",
                return_value=["working tree is dirty"],
            ),
            pytest.raises(SystemExit, match="1"),
        ):
            _resolve_strategy_config(
                tmp_path, tmp_path / "backtesting" / "renquant_104",
                "strategy_config.json",
            )

    def test_relative_local_path(self, tmp_path: Path) -> None:
        strat_dir = tmp_path / "subrepos" / "renquant-strategy-104"
        _make_config(strat_dir / "configs")
        _make_lock(
            tmp_path,
            commit=FAKE_COMMIT,
            local_path="subrepos/renquant-strategy-104",
        )

        with mock.patch("run_sim_104._verify_pin", return_value=[]):
            cfg, source, _ = _resolve_strategy_config(
                tmp_path, tmp_path / "backtesting" / "renquant_104",
                "strategy_config.json",
            )
        assert source == "PINNED"
        assert cfg.exists()

    def test_experiment_manifest_marked(self, tmp_path: Path) -> None:
        # F-7 r7: an experiment manifest must be a REGISTERED record --
        # located under experiments/manifests/ AND digest-matched in
        # experiments/manifests/INDEX.json (renquant_artifacts.
        # verify_manifest_registered) -- and must declare data_manifest_path
        # / model_artifact_path so the 5 pins are verifiable. See
        # tests/test_run_sim_104_config_resolution.py for the full
        # registry/pin-verification coverage; this test only re-confirms
        # that _resolve_strategy_config still routes a valid registered
        # manifest to EXPLORATORY_ONLY without touching _verify_pin.
        strategy_dir = tmp_path / "backtesting" / "renquant_104"
        strategy_dir.mkdir(parents=True)
        cfg_file = strategy_dir / "strategy_config.json"
        content = b'{"exp": true}'
        cfg_file.write_bytes(content)
        digest = "sha256:" + hashlib.sha256(content).hexdigest()

        data_manifest_path = tmp_path / "exp-001_data_manifest.json"
        data_manifest_path.write_text(json.dumps({
            "dataset_id": "exp-001-ds",
            "schema_version": "1.0",
            "fingerprint": "sha256:datapin",
            "uri": "sqlite:///data/runs.alpaca.db",
            "asset_class": "us_equity",
        }))
        model_artifact_path = tmp_path / "exp-001_model.pt"
        model_artifact_path.write_bytes(b"weights")

        manifest_path = tmp_path / "experiments" / "manifests" / "exp-001.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({
            "experiment_id": "exp-001",
            "config_path": str(cfg_file),
            "config_digest": digest,
            "status": "ACTIVE",
            "pins": {
                "data_snapshot": "d0000000",
                "model_artifact": "m0000000",
                "strategy_config": "s0000000",
                "pipeline_version": "p0000000",
                "calendar_universe": "c0000000",
            },
            "data_manifest_path": str(data_manifest_path),
            "model_artifact_path": str(model_artifact_path),
        }))
        manifest_digest = "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        index_path = tmp_path / "experiments" / "manifests" / "INDEX.json"
        index_path.write_text(json.dumps({
            "exp-001": {"digest": manifest_digest, "path": str(manifest_path)},
        }))

        with mock.patch("run_sim_104._verify_pin") as mock_verify:
            cfg, source, manifest = _resolve_strategy_config(
                tmp_path, strategy_dir, "strategy_config.json",
                experiment_manifest=str(manifest_path),
            )
        assert source == "EXPLORATORY_ONLY"
        assert manifest is not None
        assert manifest["experiment_id"] == "exp-001"
        mock_verify.assert_not_called()

    def test_missing_lock_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="1"):
            _resolve_strategy_config(
                tmp_path, tmp_path, "strategy_config.json",
            )

    def test_missing_config_after_verify_exits(self, tmp_path: Path) -> None:
        strat_dir = tmp_path / "subrepo"
        strat_dir.mkdir()
        _make_lock(tmp_path, commit=FAKE_COMMIT, local_path=str(strat_dir))

        with (
            mock.patch("run_sim_104._verify_pin", return_value=[]),
            pytest.raises(SystemExit, match="1"),
        ):
            _resolve_strategy_config(
                tmp_path, tmp_path, "strategy_config.json",
            )
