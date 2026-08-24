"""Hermetic harness: a fake repo whose only real files are the scripts under test."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

#: The repo these tests live in — derived, never hardcoded. A literal
#: /Users/... default would make this suite pass or fail on whose machine it is,
#: which is the defect it exists to prevent one level down.
REAL_REPO = Path(__file__).resolve().parent.parent

SUBREPO_ENV_STUB = """\
renquant_load_subrepo_env() { :; }
renquant_subrepo_root() { echo "$1/.subrepo_runtime/repos"; }
renquant_subrepo_src() { echo "$1/$2/src"; }
renquant_subrepo_pythonpath() { echo ""; }
"""

# Stands in for .venv/bin/python. The wrapper calls it twice before the chain:
# the NYSE calendar guard (`-c ...mcal...`) and the trigger check
# (`-m renquant_orchestrator.anomaly_triggers`). Nothing else is exercised.
PYTHON_STUB = """\
#!/bin/bash
for a in "$@"; do
  case "$a" in
    -m) shift; case "${1:-}" in
          renquant_orchestrator.anomaly_triggers) echo "anomaly_vix_5pct"; exit 1 ;;
        esac ;;
  esac
done
exit 0
"""


def build_repo(tmp: Path, child_body: str, *, scripts: list[str]) -> Path:
    repo = tmp / "repo"
    (repo / "scripts" / "lib").mkdir(parents=True)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / "logs").mkdir(parents=True)

    py = repo / ".venv" / "bin" / "python"
    py.write_text(PYTHON_STUB)
    py.chmod(0o755)

    (repo / "scripts" / "subrepo_env.sh").write_text(SUBREPO_ENV_STUB)

    # THE REAL FILES — the classification helper and the wrappers under test.
    (repo / "scripts" / "lib" / "wf_promote_outcome.sh").write_text(
        (REAL_REPO / "scripts" / "lib" / "wf_promote_outcome.sh").read_text())
    for name in scripts:
        (repo / "scripts" / name).write_text((REAL_REPO / "scripts" / name).read_text())

    child = repo / "scripts" / "weekly_wf_promote.sh"
    child.write_text(child_body)
    child.chmod(0o755)
    return repo


def run(repo: Path, script: str, env_extra: dict[str, str]) -> tuple[int, str, str]:
    env = {**os.environ, **env_extra}
    r = subprocess.run(["bash", f"scripts/{script}"], cwd=repo, env=env,
                       capture_output=True, text=True, timeout=120)
    logs = sorted((repo / "logs").rglob("*.log"))
    log = "\n".join(p.read_text(errors="replace") for p in logs)
    notify_path = Path(env_extra.get("RQ_CONDITIONAL_NOTIFY_LOG", "")) if env_extra.get(
        "RQ_CONDITIONAL_NOTIFY_LOG") else None
    notes = notify_path.read_text() if notify_path and notify_path.exists() else ""
    return r.returncode, log, notes
