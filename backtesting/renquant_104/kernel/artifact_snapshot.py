"""Artifact snapshot — isolate A/B sims from concurrent retraining.

Problem (2026-04-24 incident): two A/B sim scripts ran at different
times during the day. Between them, the user's notebook retrained the
panel/NGBoost/GMM artifacts. Each A/B run loaded a DIFFERENT set of
model artifacts, so "A_GOLDEN_v4.1" produced different APYs across
runs (+39.82% → +34.56% → +29.96%). Not a code regression — model
drift.

Fix: `snapshot_artifacts(strategy_dir)` returns a tmpdir with a
frozen copy of the artifacts/ + models/ + strategy_config.json at
call time. Subsequent retraining doesn't affect the snapshot.

Usage (A/B script pattern)::

    from kernel.artifact_snapshot import snapshot_artifacts

    STRATEGY_DIR = Path("backtesting/renquant_104")
    snapshot_dir = snapshot_artifacts(STRATEGY_DIR)
    try:
        # Use snapshot_dir for all SimAdapter / run_backtest calls
        cfg = load_strategy_config(snapshot_dir / "strategy_config.json")
        cfg["_strategy_dir"] = str(snapshot_dir)
        run_backtest(config=cfg, strategy_dir=snapshot_dir, ...)
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)

Or as a context manager::

    with snapshot_artifacts_ctx(STRATEGY_DIR) as snapshot_dir:
        cfg = load_strategy_config(snapshot_dir / "strategy_config.json")
        ...
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("kernel.artifact_snapshot")


# The subdirectories + top-level files that the sim/LEAN/live paths
# read during a backtest. If training writes any of these out-of-band,
# we'd see model drift. Snapshot all of them.
_SNAPSHOT_DIRS = ["artifacts", "models"]
# 2026-05-04 fix: include ALL strategy_config*.json side configs.
# Pre-fix, only the production strategy_config.json + .golden.json were
# copied; B2/A-B sims that used --strategy-config-name <side>.json
# silently ran against the production config inside the snapshot,
# making side configs INVISIBLE to the sim path. Result: every
# "side config" test produced identical trades to baseline.
# `_SNAPSHOT_FILE_PATTERNS` is a list of glob patterns; full files
# matching any pattern are copied.
_SNAPSHOT_FILES = ["strategy_config.json", "strategy_config.golden.json"]
_SNAPSHOT_FILE_PATTERNS = ["strategy_config.*.json"]


def snapshot_artifacts(
    strategy_dir: "Path | str",
    target_root: "Path | str | None" = None,
) -> Path:
    """Copy artifacts/ + models/ + strategy_config*.json to a tmp dir.

    Returns the path of the tmp dir. Caller is responsible for cleanup
    (via `shutil.rmtree`) or using `snapshot_artifacts_ctx`.

    Raises ValueError if strategy_dir doesn't have the expected layout.
    """
    strategy_dir = Path(strategy_dir).resolve()
    if not strategy_dir.exists():
        raise ValueError(f"strategy_dir not found: {strategy_dir}")

    if target_root is None:
        target_root = Path(tempfile.mkdtemp(prefix="renquant_ab_snapshot_"))
    else:
        target_root = Path(target_root)
        target_root.mkdir(parents=True, exist_ok=True)

    # Copy subdirs (artifacts/, models/)
    for sub in _SNAPSHOT_DIRS:
        src = strategy_dir / sub
        if not src.exists():
            log.warning("snapshot: missing %s — skipping", src)
            continue
        dst = target_root / sub
        shutil.copytree(src, dst)

    # Copy top-level config files (literal names + glob patterns).
    # 2026-05-04 fix: side configs `strategy_config.<variant>.json` were
    # invisible to the snapshot, silently routing every B2/A-B side-
    # config sim to the production strategy_config.json.
    seen_files: set[str] = set()
    for fname in _SNAPSHOT_FILES:
        src = strategy_dir / fname
        if src.exists():
            shutil.copy2(src, target_root / fname)
            seen_files.add(fname)
    for pattern in _SNAPSHOT_FILE_PATTERNS:
        for src in strategy_dir.glob(pattern):
            if src.name in seen_files:
                continue
            shutil.copy2(src, target_root / src.name)
            seen_files.add(src.name)

    # Record the commit SHA at snapshot time — essential for reproducing
    # this A/B later if needed
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=strategy_dir, text=True, timeout=2,
        ).strip()
        (target_root / ".snapshot_sha").write_text(sha)
    except Exception as exc:
        log.warning("snapshot: git sha capture failed: %s", exc)

    log.info("snapshot: artifacts frozen at %s", target_root)
    return target_root


@contextmanager
def snapshot_artifacts_ctx(strategy_dir: "Path | str"):
    """Context-manager form: snapshots on enter, rm -rf on exit.

    Use for A/B scripts that run N variants back-to-back — all N see
    identical artifacts regardless of concurrent retraining::

        with snapshot_artifacts_ctx(STRATEGY_DIR) as snap:
            for variant in variants:
                cfg = load_strategy_config(snap / "strategy_config.json")
                cfg["_strategy_dir"] = str(snap)
                run_backtest(config=cfg, strategy_dir=snap, ...)
    """
    snap_dir = snapshot_artifacts(strategy_dir)
    try:
        yield snap_dir
    finally:
        shutil.rmtree(snap_dir, ignore_errors=True)
        log.info("snapshot: cleaned up %s", snap_dir)


__all__ = ["snapshot_artifacts", "snapshot_artifacts_ctx"]
