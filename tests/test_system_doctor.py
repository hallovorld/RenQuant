"""Tests for system_doctor (pin/runtime drift + lock integrity)."""
from __future__ import annotations

import importlib.util
import os
import json
import subprocess
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "system_doctor", Path(__file__).resolve().parent.parent / "scripts" / "system_doctor.py")
sd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sd)

_SNAPSHOT_TEST_MODULE_PATH = (
    Path(__file__).resolve().parent / "test_render_strategy_104_snapshot.py"
)
_SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "test_render_strategy_104_snapshot_for_doctor", _SNAPSHOT_TEST_MODULE_PATH)
_snapshot_test_mod = importlib.util.module_from_spec(_SNAPSHOT_SPEC)
_SNAPSHOT_SPEC.loader.exec_module(_snapshot_test_mod)

_RENDERER_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "render_strategy_104_snapshot.py"
)

GOOD = "a" * 40


def _lock(commit=GOOD, never_delete=True, name="renquant-pipeline"):
    return {"source_repo": {"never_delete": never_delete},
            "subrepos": [{"name": name, "commit": commit}]}


def test_lock_integrity_flags_bad_sha_and_never_delete():
    ok = sd.check_lock_integrity(_lock())
    assert all(c["ok"] for c in ok)
    bad = sd.check_lock_integrity(_lock(commit="main"))      # not a sha
    assert any(not c["ok"] for c in bad)
    nd = sd.check_lock_integrity(_lock(never_delete=False))  # never_delete must be true
    assert any(c["check"] == "source_repo.never_delete" and not c["ok"] for c in nd)


def _init_repo_on_branch(tmp_path, branch):
    r = tmp_path / "repo"; r.mkdir(parents=True)
    g = ("git", "-C", str(r))
    subprocess.run((*g, "init", "-q"), check=True)
    subprocess.run((*g, "config", "user.email", "t@t"), check=True)
    subprocess.run((*g, "config", "user.name", "t"), check=True)
    (r / "f").write_text("x")
    subprocess.run((*g, "add", "."), check=True)
    subprocess.run((*g, "commit", "-qm", "init"), check=True)
    subprocess.run((*g, "branch", "-M", branch), check=True)
    return r


def test_live_checkout_branch_opt_in_skips_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("RENQUANT_DOCTOR_EXPECT_BRANCH", raising=False)
    res = sd.check_live_checkout_branch(_init_repo_on_branch(tmp_path / "x", "feat/whatever"))
    assert res.get("skip") and res["ok"]                            # opt-in: not RED on a PR worktree


def test_live_checkout_branch_flags_non_main_when_active(tmp_path):
    on_main = sd.check_live_checkout_branch(_init_repo_on_branch(tmp_path / "a", "main"), expected="main")
    assert on_main["ok"] and "on main" in on_main["detail"]
    stray = sd.check_live_checkout_branch(_init_repo_on_branch(tmp_path / "b", "feat/finnhub-analyst-cron"), expected="main")
    assert not stray["ok"] and "EXPECTED main" in stray["detail"]   # the 2026-06-25 incident class


def test_promote_backups_warns_when_piled_up(tmp_path):
    lock = tmp_path / "subrepos.lock.json"; lock.write_text("{}")
    assert sd.check_promote_backups(lock)[0]["ok"]                 # none
    for i in range(5):
        (tmp_path / f"subrepos.lock.json.promote-bak.2026010{i}").write_text("{}")
    assert not sd.check_promote_backups(lock)[0]["ok"]             # >3 → RED


def _git(repo, *a):
    subprocess.run(("git", "-C", str(repo), *a), check=True,
                   capture_output=True, text=True)


def test_pin_runtime_drift_detects_mismatch_and_dirt(tmp_path):
    rt_root = tmp_path / "repos"; repo = rt_root / "renquant-pipeline"; repo.mkdir(parents=True)
    _git(repo, "init", "-q"); _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("v1"); _git(repo, "add", "."); _git(repo, "commit", "-qm", "c1")
    head = subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD"), text=True).strip()
    # pin == head, clean → all ok
    res = sd.check_pin_runtime_drift(_lock(commit=head), rt_root)
    assert all(c["ok"] for c in res)
    # pin != head → DRIFT red
    res = sd.check_pin_runtime_drift(_lock(commit="b" * 40), rt_root)
    assert any(c["check"].startswith("runtime_at_pin") and not c["ok"] for c in res)
    # dirty working tree → runtime_clean red
    (repo / "f.txt").write_text("v2")
    res = sd.check_pin_runtime_drift(_lock(commit=head), rt_root)
    assert any(c["check"].startswith("runtime_clean") and not c["ok"] for c in res)


def test_unmaterialized_runtime_is_skip_not_red(tmp_path):
    res = sd.check_pin_runtime_drift(_lock(), tmp_path / "nope")
    assert all(c["ok"] for c in res) and res[0].get("skip")


def _committed_snapshot_repo(tmp_path):
    """A fixture repo with a genuinely fresh, committed
    doc/arch/strategy-104-snapshot.md — the state check_strategy_snapshot
    should report green against, before any out-of-band mutation."""
    renderer = _snapshot_test_mod._load_module()
    root = _snapshot_test_mod._fixture_root(renderer, tmp_path)
    out = root / "doc" / "arch" / "strategy-104-snapshot.md"
    rc = renderer.main(["--repo-root", str(root), "--output", str(out)])
    assert rc == 0
    return root


def test_strategy_snapshot_check_green_when_fresh(tmp_path):
    root = _committed_snapshot_repo(tmp_path)
    res = sd.check_strategy_snapshot(repo=root, python=sys.executable, renderer_path=_RENDERER_PATH)
    assert res["ok"], res["detail"]


def test_strategy_snapshot_check_fails_on_artifact_metadata_change(tmp_path):
    """Codex PR #432 round-3 review: an out-of-band artifact metadata edit
    (never going through promote_pin.py) must surface as a doctor RED, not
    persist silently until a human remembers to run `make snapshot-check`."""
    root = _committed_snapshot_repo(tmp_path)
    renderer = _snapshot_test_mod._load_module()
    artifact_path = root / renderer.STRATEGY_DIR_REL / "artifacts" / "prod" / "primary.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["trained_date"] = "2099-01-01"  # out-of-band edit, no promote
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    res = sd.check_strategy_snapshot(repo=root, python=sys.executable, renderer_path=_RENDERER_PATH)
    assert not res["ok"]
    assert "STALE" in res["detail"]


def test_strategy_snapshot_check_fails_on_active_calibrator_change(tmp_path):
    root = _committed_snapshot_repo(tmp_path)
    renderer = _snapshot_test_mod._load_module()
    calib_path = root / renderer.STRATEGY_DIR_REL / "artifacts" / "prod" / "calib.json"
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    calib["metadata"]["pool_ic"] = 0.5  # out-of-band edit, no promote
    calib_path.write_text(json.dumps(calib), encoding="utf-8")

    res = sd.check_strategy_snapshot(repo=root, python=sys.executable, renderer_path=_RENDERER_PATH)
    assert not res["ok"]
    assert "STALE" in res["detail"]


def test_strategy_snapshot_check_skips_when_renderer_absent(tmp_path):
    root = tmp_path / "no-renderer-repo"
    root.mkdir()
    res = sd.check_strategy_snapshot(repo=root, python=sys.executable)
    assert res["ok"] and res.get("skip")


# ─────────────────────────────────────────────────────────────────────────
# check_bundle: PYTHONPATH + --repo wiring
# ─────────────────────────────────────────────────────────────────────────


def test_check_bundle_passes_repo_and_pythonpath_with_existing_pythonpath(tmp_path, monkeypatch):
    """check_bundle must pass --repo <REPO> and prepend orchestrator src
    to PYTHONPATH. When PYTHONPATH already has a value, the result must
    be ``orch_src:<existing>`` with no empty segment."""
    orch_rt = tmp_path / ".subrepo_runtime" / "repos" / "renquant-orchestrator"
    checker = orch_rt / "scripts" / "check_model_bundle_consistency.py"
    checker.parent.mkdir(parents=True)
    checker.write_text("")
    orch_src = orch_rt / "src"

    monkeypatch.setattr(sd, "REPO", tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/existing/path")

    captured = {}
    def mock_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(subprocess, "run", mock_run)
    result = sd.check_bundle()

    assert result["ok"]
    assert "--repo" in captured["cmd"]
    repo_idx = captured["cmd"].index("--repo")
    assert captured["cmd"][repo_idx + 1] == str(tmp_path)

    pp = captured["env"]["PYTHONPATH"]
    assert str(orch_src) in pp
    assert "/existing/path" in pp
    segments = pp.split(os.pathsep)
    assert "" not in segments, f"empty PYTHONPATH segment found: {pp!r}"


def test_check_bundle_pythonpath_no_empty_segment_when_unset(tmp_path, monkeypatch):
    """When PYTHONPATH is absent from env, the constructed value must be
    just the orchestrator src — no trailing pathsep, no empty segment."""
    orch_rt = tmp_path / ".subrepo_runtime" / "repos" / "renquant-orchestrator"
    checker = orch_rt / "scripts" / "check_model_bundle_consistency.py"
    checker.parent.mkdir(parents=True)
    checker.write_text("")
    orch_src = orch_rt / "src"

    monkeypatch.setattr(sd, "REPO", tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    captured = {}
    def mock_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(subprocess, "run", mock_run)
    sd.check_bundle()

    pp = captured["env"]["PYTHONPATH"]
    assert pp == str(orch_src)
    segments = pp.split(os.pathsep)
    assert "" not in segments


def test_check_bundle_subprocess_failure_is_red_not_skip(tmp_path, monkeypatch):
    """A checker that exists but fails (import error, assertion, etc.)
    must be RED, never silently changed to SKIP."""
    orch_rt = tmp_path / ".subrepo_runtime" / "repos" / "renquant-orchestrator"
    checker = orch_rt / "scripts" / "check_model_bundle_consistency.py"
    checker.parent.mkdir(parents=True)
    checker.write_text("")

    monkeypatch.setattr(sd, "REPO", tmp_path)

    def mock_run(cmd, **kwargs):
        class R:
            returncode = 1
            stdout = ""
            stderr = "ModuleNotFoundError: No module named 'renquant_orchestrator'"
        return R()

    monkeypatch.setattr(subprocess, "run", mock_run)
    result = sd.check_bundle()
    assert not result["ok"]
    assert not result.get("skip")


def test_promote_backups_alarms_only_above_the_retention_policy_keep(tmp_path):
    """The reviewed retention policy keeps the 5 newest lock backups; the doctor
    must not stay RED on exactly what `prune-artifacts --execute` leaves behind
    (2026-09-03: 5 left, old threshold 3 => permanent RED)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "system_doctor", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "system_doctor.py"))
    sd = importlib.util.module_from_spec(spec); spec.loader.exec_module(sd)
    lock = tmp_path / "subrepos.lock.json"; lock.write_text("{}")
    assert sd.PROMOTE_BACKUPS_KEEP == 5
    for i in range(5):
        (tmp_path / f"subrepos.lock.json.promote-bak.2026090{i}T000000").write_text("{}")
    assert sd.check_promote_backups(lock)[0]["ok"] is True
    (tmp_path / "subrepos.lock.json.promote-bak.20260908T000000").write_text("{}")
    res = sd.check_promote_backups(lock)[0]
    assert res["ok"] is False and "(>5, prune)" in res["detail"]
