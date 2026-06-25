"""Tests for the atomic/reversible pin-promote tool (no network, --no-sync)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "promote_pin", Path(__file__).resolve().parent.parent / "scripts" / "promote_pin.py")
pp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pp)


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
    p = tmp_path / "lock.json"; pp.atomic_write_json(p, _lock())
    rc = pp.main(["bump", "--subrepo", "renquant-pipeline", "--commit", "ccccccc3",
                  "--lock", str(p), "--no-sync", "--apply"])
    assert rc == 0
    assert pp.load_lock(p)["subrepos"][0]["commit"] == "ccccccc3"   # applied
    assert pp.latest_backup(p) is not None                          # backup made
    # revert restores the original pin
    rc = pp.main(["revert", "--lock", str(p), "--no-sync", "--apply"])
    assert rc == 0
    assert pp.load_lock(p)["subrepos"][0]["commit"] == "aaaaaaa1"
