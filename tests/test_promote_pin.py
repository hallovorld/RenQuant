"""Tests for the atomic/reversible pin-promote tool (no network, --no-sync)."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "promote_pin", Path(__file__).resolve().parent.parent / "scripts" / "promote_pin.py")
pp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pp)

_SNAPSHOT_TEST_MODULE_PATH = (
    Path(__file__).resolve().parent / "test_render_strategy_104_snapshot.py"
)
_SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "test_render_strategy_104_snapshot_for_promote", _SNAPSHOT_TEST_MODULE_PATH)
_snapshot_test_mod = importlib.util.module_from_spec(_SNAPSHOT_SPEC)
_SNAPSHOT_SPEC.loader.exec_module(_snapshot_test_mod)
_REAL_RENDERER = (
    Path(__file__).resolve().parent.parent / "scripts" / "render_strategy_104_snapshot.py"
)


def _lock():
    return {"schema_version": 1, "source_repo": {"never_delete": True},
            "subrepos": [{"name": "renquant-pipeline", "commit": "aaaaaaa1"},
                         {"name": "renquant-common", "commit": "bbbbbbb2"}]}


def test_is_sha():
    assert pp._is_sha("fedd07e5f22fb88bb2f857386")
    assert pp._is_sha("aaaaaaa")
    assert not pp._is_sha("main")
    assert not pp._is_sha("abc")  # too short


def test_bump_pin_updates_only_target():
    old, new = pp.bump_pin(_lock(), "renquant-pipeline", "ccccccc3")
    assert old == "aaaaaaa1"
    assert pp.find_entry(new, "renquant-pipeline")["commit"] == "ccccccc3"
    assert pp.find_entry(new, "renquant-common")["commit"] == "bbbbbbb2"  # untouched


def test_bump_pin_rejects_noop_and_bad_inputs():
    with pytest.raises(ValueError):
        pp.bump_pin(_lock(), "renquant-pipeline", "aaaaaaa1")  # same → no-op
    with pytest.raises(ValueError):
        pp.bump_pin(_lock(), "renquant-pipeline", "not-a-sha")
    with pytest.raises(KeyError):
        pp.bump_pin(_lock(), "renquant-nope", "ccccccc3")


def test_atomic_write_roundtrip(tmp_path):
    p = tmp_path / "lock.json"
    pp.atomic_write_json(p, _lock())
    assert pp.load_lock(p)["subrepos"][0]["commit"] == "aaaaaaa1"
    assert not (tmp_path / "lock.json.tmp").exists()  # temp cleaned up


def test_cli_dry_run_does_not_write(tmp_path):
    p = tmp_path / "lock.json"; pp.atomic_write_json(p, _lock())
    rc = pp.main(["bump", "--subrepo", "renquant-pipeline", "--commit", "ccccccc3",
                  "--lock", str(p), "--no-sync"])  # no --apply
    assert rc == 0
    assert pp.load_lock(p)["subrepos"][0]["commit"] == "aaaaaaa1"  # unchanged


def test_cli_apply_writes_pin_and_backup_then_revert(tmp_path):
    # --skip-snapshot-check: this test exercises lock-file bump/backup/revert
    # mechanics only, against a synthetic lock file with no matching
    # .subrepo_runtime tree — the snapshot backstop is covered separately in
    # test_snapshot_freshness_check_* below.
    # --verify-cmd true: hermetic — without it, the DEFAULT verify
    # (check_conviction_admits.py, present in a full checkout such as the
    # hosted CI runner but absent from some dev worktrees) would run against
    # this synthetic lock, fail for lack of live data, and auto-revert the
    # bump before the behavior under test is reached.
    p = tmp_path / "lock.json"; pp.atomic_write_json(p, _lock())
    rc = pp.main(["bump", "--subrepo", "renquant-pipeline", "--commit", "ccccccc3",
                  "--lock", str(p), "--no-sync", "--apply", "--skip-snapshot-check",
                  "--verify-cmd", "true"])
    assert rc == 0
    assert pp.load_lock(p)["subrepos"][0]["commit"] == "ccccccc3"   # applied
    assert pp.latest_backup(p) is not None                          # backup made
    # revert restores the original pin
    rc = pp.main(["revert", "--lock", str(p), "--no-sync", "--apply", "--skip-snapshot-check"])
    assert rc == 0
    assert pp.load_lock(p)["subrepos"][0]["commit"] == "aaaaaaa1"


def test_bump_apply_defaults_to_checking_snapshot_freshness(tmp_path, monkeypatch):
    """Codex PR #432 round-3 review: the snapshot backstop must run by
    default on --apply (not require an opt-in flag), and must NOT revert the
    pin for a stale-snapshot finding alone — only the doc needs a follow-up."""
    p = tmp_path / "lock.json"; pp.atomic_write_json(p, _lock())
    calls = []

    def fake_check(python, repo=pp.REPO):
        calls.append((python, repo))
        return False, "ACTION REQUIRED: fake staleness for the test"

    monkeypatch.setattr(pp, "check_snapshot_freshness", fake_check)
    rc = pp.main(["bump", "--subrepo", "renquant-pipeline", "--commit", "ccccccc3",
                  "--lock", str(p), "--no-sync", "--apply", "--verify-cmd", "true"])
    assert rc == 1, "a stale snapshot must make the command exit non-zero"
    assert len(calls) == 1
    # The pin itself is NOT reverted — only the snapshot needs a follow-up.
    assert pp.load_lock(p)["subrepos"][0]["commit"] == "ccccccc3"


def test_bump_apply_skip_snapshot_check_flag_bypasses_the_backstop(tmp_path, monkeypatch):
    p = tmp_path / "lock.json"; pp.atomic_write_json(p, _lock())
    called = []
    monkeypatch.setattr(pp, "check_snapshot_freshness",
                         lambda *a, **k: called.append(1) or (False, "should not run"))
    rc = pp.main(["bump", "--subrepo", "renquant-pipeline", "--commit", "ccccccc3",
                  "--lock", str(p), "--no-sync", "--apply", "--skip-snapshot-check",
                  "--verify-cmd", "true"])
    assert rc == 0
    assert not called


def test_revert_apply_also_checks_snapshot_freshness(tmp_path, monkeypatch):
    p = tmp_path / "lock.json"; pp.atomic_write_json(p, _lock())
    pp.main(["bump", "--subrepo", "renquant-pipeline", "--commit", "ccccccc3",
             "--lock", str(p), "--no-sync", "--apply", "--skip-snapshot-check",
             "--verify-cmd", "true"])
    calls = []
    monkeypatch.setattr(pp, "check_snapshot_freshness",
                         lambda *a, **k: calls.append(1) or (True, "fresh"))
    rc = pp.main(["revert", "--lock", str(p), "--no-sync", "--apply"])
    assert rc == 0
    assert len(calls) == 1


def _real_fixture_repo(tmp_path):
    """A fixture repo carrying a genuine copy of the renderer script, so
    check_snapshot_freshness's real (non-mocked) diff/regenerate logic can
    be exercised end to end."""
    renderer_mod = _snapshot_test_mod._load_module()
    root = _snapshot_test_mod._fixture_root(renderer_mod, tmp_path)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(_REAL_RENDERER, root / "scripts" / "render_strategy_104_snapshot.py")
    out = root / "doc" / "arch" / "strategy-104-snapshot.md"
    rc = renderer_mod.main(["--repo-root", str(root), "--output", str(out)])
    assert rc == 0
    return root


def test_snapshot_freshness_check_real_fresh(tmp_path):
    root = _real_fixture_repo(tmp_path)
    fresh, msg = pp.check_snapshot_freshness(sys.executable, repo=root)
    assert fresh, msg


def test_snapshot_freshness_check_real_stale_after_artifact_edit(tmp_path):
    root = _real_fixture_repo(tmp_path)
    renderer_mod = _snapshot_test_mod._load_module()
    artifact_path = root / renderer_mod.STRATEGY_DIR_REL / "artifacts" / "prod" / "primary.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["trained_date"] = "2099-01-01"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    fresh, msg = pp.check_snapshot_freshness(sys.executable, repo=root)
    assert not fresh
    assert "ACTION REQUIRED" in msg
    assert "STALE" in msg
    # Never auto-commits: the committed doc is untouched.
    committed = (root / "doc" / "arch" / "strategy-104-snapshot.md").read_text(encoding="utf-8")
    assert "2099-01-01" not in committed
