from __future__ import annotations

import importlib.util
import plistlib
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_launchagents.py"
    spec = importlib.util.spec_from_file_location("check_launchagents", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plist(label: str) -> bytes:
    return plistlib.dumps(
        {
            "Label": label,
            "ProgramArguments": ["/Users/renhao/git/github/RenQuant/scripts/noop.sh"],
            "EnvironmentVariables": {
                "PATH": "/Users/renhao/git/github/RenQuant/.venv/bin:/usr/bin:/bin",
            },
        }
    )


def _write_active_plists(repo_root: Path) -> None:
    launchd = repo_root / "scripts" / "launchd"
    launchd.mkdir(parents=True)
    (launchd / "com.renquant.daily104.plist").write_bytes(_plist("com.renquant.daily104"))
    (repo_root / "scripts" / "com.renquant.backup.plist").write_bytes(
        _plist("com.renquant.backup")
    )


def test_launchagent_check_passes_for_exact_installed_plists(tmp_path):
    module = _load_module()
    repo_root = tmp_path / "repo"
    agents = tmp_path / "agents"
    _write_active_plists(repo_root)
    agents.mkdir()
    for source in [
        repo_root / "scripts" / "launchd" / "com.renquant.daily104.plist",
        repo_root / "scripts" / "com.renquant.backup.plist",
    ]:
        (agents / source.name).write_bytes(source.read_bytes())

    result = module.inspect_launchagents(repo_root=repo_root, launchagents_dir=agents)

    assert result["ok"] is True
    assert result["issues"] == []


def test_launchagent_check_fails_on_installed_drift(tmp_path):
    module = _load_module()
    repo_root = tmp_path / "repo"
    agents = tmp_path / "agents"
    _write_active_plists(repo_root)
    agents.mkdir()
    for source in [
        repo_root / "scripts" / "launchd" / "com.renquant.daily104.plist",
        repo_root / "scripts" / "com.renquant.backup.plist",
    ]:
        (agents / source.name).write_bytes(source.read_bytes())
    (agents / "com.renquant.daily104.plist").write_bytes(b"not xml")

    result = module.inspect_launchagents(repo_root=repo_root, launchagents_dir=agents)

    assert result["ok"] is False
    reasons = {issue["reason"] for issue in result["issues"]}
    assert any("drifted from repo source" in reason for reason in reasons)
    assert any("installed plist is invalid" in reason for reason in reasons)


def test_launchagent_check_warns_on_extra_by_default(tmp_path):
    module = _load_module()
    repo_root = tmp_path / "repo"
    agents = tmp_path / "agents"
    _write_active_plists(repo_root)
    agents.mkdir()
    for source in [
        repo_root / "scripts" / "launchd" / "com.renquant.daily104.plist",
        repo_root / "scripts" / "com.renquant.backup.plist",
    ]:
        (agents / source.name).write_bytes(source.read_bytes())
    (agents / "com.renquant.daily103.plist").write_bytes(_plist("com.renquant.daily103"))

    result = module.inspect_launchagents(repo_root=repo_root, launchagents_dir=agents)

    assert result["ok"] is True
    assert result["issues"] == [
        {
            "severity": "warning",
            "path": str(agents / "com.renquant.daily103.plist"),
            "reason": "extra RenQuant LaunchAgent is not tracked as active in this repo",
        }
    ]
