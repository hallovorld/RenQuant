"""Regression: train_104.py must accept --trigger argument.

Pre-fix incident: 2026-05-15 13:10 — VIX +5.68% triggered
conditional_retrain_104.sh which called `train_104.py --trigger
anomaly_vix_5pct`. argparse rejected the unknown arg → retrain
failed → ntfy alert "training failed". Model stayed stale through
the volatility spike.

Post-fix: --trigger is a documented, logging-only argument. Default
"cadence". Must be a string and not alter training flow.

This test pins the argparse contract WITHOUT running the full
training pipeline (which would take hours and need real data).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_train_104_accepts_trigger_arg():
    """`train_104.py --help` lists --trigger; argparse doesn't reject it."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train_104.py"), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"--help should exit cleanly; got rc={result.returncode}\n"
        f"stderr: {result.stderr[:200]}"
    )
    assert "--trigger" in result.stdout, (
        "train_104.py --help must list --trigger so conditional_retrain_104.sh "
        "can pass anomaly tags. Pre-fix incident 2026-05-15 13:10."
    )


def test_train_104_rejects_unknown_arg():
    """Sanity: argparse still rejects truly-unknown args (not a permissive
    pass-through)."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train_104.py"),
         "--definitely-not-a-real-arg"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr
