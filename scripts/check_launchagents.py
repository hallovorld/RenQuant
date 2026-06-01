#!/usr/bin/env python3
"""Check installed RenQuant LaunchAgents against repo-tracked active plists."""
from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCHAGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
# Personal-workstation contract: active launchd plists are installed for this
# operator account and must resolve the project venv at this absolute path.
VENV_BIN = "/Users/renhao/git/github/RenQuant/.venv/bin"


@dataclass(frozen=True)
class Issue:
    severity: str
    path: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "path": self.path,
            "reason": self.reason,
        }


def _active_sources(repo_root: Path) -> list[Path]:
    launchd_dir = repo_root / "scripts" / "launchd"
    sources = sorted(launchd_dir.glob("*.plist"))
    backup = repo_root / "scripts" / "com.renquant.backup.plist"
    if backup.exists():
        sources.append(backup)
    return sources


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plist(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = plistlib.loads(path.read_bytes())
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "plist root is not a dict"
    return data, None


def _check_path_env(path: Path, data: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    env = data.get("EnvironmentVariables")
    if not isinstance(env, dict):
        return [Issue("error", str(path), "missing EnvironmentVariables dict")]
    launch_path = env.get("PATH")
    if not isinstance(launch_path, str) or not launch_path:
        issues.append(Issue("error", str(path), "missing EnvironmentVariables.PATH"))
    elif VENV_BIN not in launch_path:
        issues.append(Issue("error", str(path), f"PATH must include {VENV_BIN}"))
    return issues


def inspect_launchagents(
    *,
    repo_root: Path = ROOT,
    launchagents_dir: Path = DEFAULT_LAUNCHAGENTS_DIR,
    strict_extra: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    sources = _active_sources(repo_root)
    expected_names = {source.name for source in sources}
    issues: list[Issue] = []
    entries: list[dict[str, Any]] = []

    for source in sources:
        source_data, source_error = _load_plist(source)
        if source_error:
            issues.append(Issue("error", str(source), f"repo plist is invalid: {source_error}"))
            source_hash = None
        else:
            source_hash = _sha256(source)
            issues.extend(_check_path_env(source, source_data or {}))

        installed = launchagents_dir / source.name
        row: dict[str, Any] = {
            "name": source.name,
            "source": str(source),
            "installed_path": str(installed),
            "source_sha256": source_hash,
            "installed_sha256": None,
            "is_installed": installed.exists(),
            "matches_source": False,
        }
        if not installed.exists():
            issues.append(Issue("error", str(installed), "missing installed LaunchAgent"))
            entries.append(row)
            continue

        row["installed_sha256"] = _sha256(installed)
        row["matches_source"] = row["installed_sha256"] == source_hash
        if not row["matches_source"]:
            issues.append(
                Issue(
                    "error",
                    str(installed),
                    "installed plist drifted from repo source; run scripts/install_launchagents.sh",
                )
            )

        installed_data, installed_error = _load_plist(installed)
        if installed_error:
            issues.append(Issue("error", str(installed), f"installed plist is invalid: {installed_error}"))
        else:
            issues.extend(_check_path_env(installed, installed_data or {}))
        entries.append(row)

    extra_severity = "error" if strict_extra else "warning"
    for extra in sorted(launchagents_dir.glob("com.renquant*.plist")):
        if extra.name not in expected_names:
            issues.append(
                Issue(
                    extra_severity,
                    str(extra),
                    "extra RenQuant LaunchAgent is not tracked as active in this repo",
                )
            )

    errors = [issue for issue in issues if issue.severity == "error"]
    return {
        "ok": not errors,
        "repo_root": str(repo_root),
        "launchagents_dir": str(launchagents_dir),
        "entries": entries,
        "issues": [issue.as_dict() for issue in issues],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--launchagents-dir", default=str(DEFAULT_LAUNCHAGENTS_DIR))
    parser.add_argument("--strict-extra", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = inspect_launchagents(
        repo_root=Path(args.repo_root),
        launchagents_dir=Path(args.launchagents_dir),
        strict_extra=args.strict_extra,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = "PASS" if result["ok"] else "FAIL"
        print(f"RenQuant LaunchAgent check: {status}")
        for issue in result["issues"]:
            print(f"- {issue['severity']}: {issue['path']}: {issue['reason']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
