"""Tests for system_doctor (pin/runtime drift + lock integrity)."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "system_doctor", Path(__file__).resolve().parent.parent / "scripts" / "system_doctor.py")
sd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sd)

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


def test_live_checkout_branch_flags_non_main(tmp_path):
    on_main = sd.check_live_checkout_branch(_init_repo_on_branch(tmp_path / "a", "main"))
    assert on_main["ok"] and "on main" in on_main["detail"]
    stray = sd.check_live_checkout_branch(_init_repo_on_branch(tmp_path / "b", "feat/finnhub-analyst-cron"))
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
