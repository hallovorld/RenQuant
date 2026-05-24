#!/usr/bin/env python
"""Non-destructive repository hygiene audit.

This script classifies dirty worktree entries so cleanup can be discussed and
reviewed without deleting or moving evidence artifacts. It is intentionally
read-only: no file mutation, no git checkout/reset, no archive moves.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _git_status() -> list[tuple[str, str]]:
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        status = line[:2].strip() or "?"
        path = line[3:].strip()
        rows.append((status, path))
    return rows


def classify_path(path: str, status: str) -> str:
    p = path.replace("\\", "/")
    name = Path(p).name
    suffix = Path(p).suffix.lower()

    if p.startswith((".tmp_", "mlruns/", "catboost_info/")):
        return "local_runtime_scratch"
    if p.startswith(".claude/"):
        return "local_agent_settings"
    if suffix == ".db" or ".db-" in name:
        return "local_runtime_state"
    if ".bak" in name or ".disabled-" in name:
        return "backup_or_disabled_copy"
    if p.startswith(("logs/", "live/logs/", "data/", "backtesting/data/")):
        return "generated_data_or_logs"
    if "live_state." in name:
        return "broker_state"
    if p.startswith("scripts/_train_") and suffix == ".py":
        return "scratch_code_artifact"
    if p.startswith("scripts/") and suffix in {".json", ".csv", ".parquet"}:
        return "experiment_or_diagnostic_artifact"
    if p.startswith("backtesting/renquant_104/models/"):
        return "per_ticker_model_artifact"
    if p.startswith("backtesting/renquant_104/artifacts/prod/"):
        return "production_model_artifact"
    if p.startswith("backtesting/renquant_104/artifacts/shadow/"):
        return "shadow_model_artifact"
    if p.startswith("backtesting/renquant_104/artifacts/sim/"):
        return "sim_model_artifact"
    if p.startswith(("artifacts/", "backtesting/renquant_104/artifacts/")):
        return "experiment_or_diagnostic_artifact"
    if p.startswith(("scripts/", "tests/", "backtesting/renquant_104/kernel/",
                     "backtesting/renquant_104/adapters/", "live/",
                     "dagster_renquant/", "rust/")):
        return "code"
    if p.startswith(("doc/", "CLAUDE.md", "AGENTS.md", "README")):
        return "documentation"
    if p.endswith("strategy_config.json") or "strategy_config." in name:
        return "strategy_config"
    if status == "??":
        return "untracked_uncategorized"
    return "tracked_uncategorized"


def build_report() -> dict:
    rows = [
        {"status": status, "path": path, "class": classify_path(path, status)}
        for status, path in _git_status()
    ]
    counts = Counter(row["class"] for row in rows)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)
    return {
        "total_dirty_entries": len(rows),
        "counts": dict(sorted(counts.items())),
        "classes": {
            key: sorted(value, key=lambda r: r["path"])
            for key, value in sorted(by_class.items())
        },
        "policy": {
            "delete_files": False,
            "default_action": "inventory_only",
            "archive_requires_review": True,
        },
    }


def _emit_markdown(report: dict) -> str:
    lines = [
        "# Repo Hygiene Audit",
        "",
        f"Dirty entries: `{report['total_dirty_entries']}`",
        "",
        "## Counts",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for key, count in report["counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend([
        "",
        "## Review Queues",
        "",
        "No file is deleted or moved by this audit. Archive/delete decisions require review.",
    ])
    for key, rows in report["classes"].items():
        lines.extend(["", f"### {key}", ""])
        for row in rows[:80]:
            lines.append(f"- `{row['status']}` `{row['path']}`")
        if len(rows) > 80:
            lines.append(f"- ... {len(rows) - 80} more")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("json", "md"), default="md")
    args = parser.parse_args()
    report = build_report()
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_emit_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
