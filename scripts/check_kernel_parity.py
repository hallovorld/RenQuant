#!/usr/bin/env python3
"""Kernel parity check: detect NEW drift between umbrella and pinned pipeline kernels.

Finding F-6 from the 2026-07-04 architecture compliance audit: the §3.5
byte-equivalence invariant between umbrella kernel/ and pinned
renquant_pipeline.kernel/ had no enforcement. This script compares the two
and fails on any file that was previously in parity but has started drifting.

Files already known to have drifted (as of 2026-07-13) are allowlisted so
existing drift does not block PRs. As files are ported/unified, they should
be removed from the allowlist so re-drift is caught.

Exit codes:
  0 — no new drift detected
  1 — new drift detected (a previously-identical file has diverged)
  2 — setup error (missing paths, bad lock file, etc.)
  3 — skipped: pipeline kernel not found. Distinct from 0 so callers (see
      tests/test_kernel_parity.py) can tell "ran and passed" apart from
      "never actually compared anything" instead of a skip silently
      reading as a pass.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

UMBRELLA_ROOT = Path(__file__).resolve().parent.parent
UMBRELLA_KERNEL = UMBRELLA_ROOT / "backtesting" / "renquant_104" / "kernel"
LOCK_FILE = UMBRELLA_ROOT / "subrepos.lock.json"

# Files known to be drifted as of 2026-07-13. This is the baseline from the
# F-6 audit. Remove entries as files are ported/unified; the test will then
# catch any re-drift.
KNOWN_DRIFT_ALLOWLIST: frozenset[str] = frozenset({
    "__init__.py",
    "alert_lifecycle.py",
    "artifact_resolver.py",
    "config.py",
    "data.py",
    "data_cache.py",
    "decision_trace.py",
    "execution/__init__.py",
    "execution/backend.py",
    "execution/backend_lean.py",
    "execution/backend_sim.py",
    "execution/t2_settlement.py",
    "execution/types.py",
    "exit_types.py",
    "exits.py",
    "indicators.py",
    "live_state_v2.py",
    "meta_label/__init__.py",
    "net_safety.py",
    # 2026-08-02 pin bump (Grant C, RenQuant#551): pipeline-side wash-sale
    # materiality floor (pipeline#251, commit ca06733) + per-rate-bucket
    # netting mirror (pipeline#217, commit 86dcdcb); umbrella copy lags
    # until ported.
    "portfolio.py",
    "panel_pipeline/__init__.py",
    "panel_pipeline/alpha158_features.py",
    "panel_pipeline/hf_patchtst_scorer.py",
    "panel_pipeline/job_panel_scoring.py",
    "panel_pipeline/model_registry.py",
    "panel_pipeline/panel_scorer.py",
    "panel_pipeline/patchtst_scorer.py",
    "panel_pipeline/shadow_scoring.py",
    "panel_pipeline/tasks_feature_matrix.py",
    "persistence.py",
    "pipeline/__init__.py",
    "pipeline/context.py",
    "pipeline/exit_params.py",
    "pipeline/job_gates.py",
    "pipeline/job_selection.py",
    "pipeline/job_sell.py",
    "pipeline/job_short_candidates.py",
    "pipeline/job_universe.py",
    "pipeline/order_attribution.py",
    "pipeline/pipeline.py",
    "pipeline/pp_inference.py",
    "pipeline/pp_research_acceptance.py",
    "pipeline/pp_training.py",
    "pipeline/pp_training_full.py",
    "pipeline/soft_exit_guards.py",
    "pipeline/task_candidates.py",
    "pipeline/task_data_freshness.py",
    "pipeline/task_execution.py",
    "pipeline/task_gates.py",
    "pipeline/task_joint_actions.py",
    "pipeline/task_limit_sells.py",
    "pipeline/task_panel_conviction_xs.py",
    "pipeline/task_regime.py",
    "pipeline/task_risk_gates.py",
    "pipeline/task_rotation.py",
    "pipeline/task_score_distribution.py",
    "pipeline/task_selection.py",
    "pipeline/task_sell.py",
    "pipeline/task_short_candidates.py",
    "pipeline/task_software_stops.py",
    "pipeline/task_topup.py",
    "pipeline/task_trim.py",
    "portfolio_qp/allocator_replay.py",
    "portfolio_qp/baseline_allocators.py",
    "portfolio_qp/constraint_snapshot.py",
    "portfolio_qp/job_qp.py",
    "portfolio_qp/qp_solver.py",
    "portfolio_qp/replay_significance.py",
    "portfolio_qp/run_ab_replay.py",
    "portfolio_qp/tasks.py",
    "portfolio_qp/wf_replay_loader.py",
    "preflight.py",
    "preflight_pipeline/__init__.py",
    "preflight_pipeline/pipeline.py",
    "preflight_pipeline/tasks/__init__.py",
    "preflight_pipeline/tasks/artifact.py",
    "preflight_pipeline/tasks/broker_fill_freshness.py",
    "preflight_pipeline/tasks/calibrator.py",
    "preflight_pipeline/tasks/config_fingerprint.py",
    "preflight_pipeline/tasks/gate.py",
    "preflight_pipeline/tasks/state.py",
    "score_audit.py",
    "score_drift.py",
    "selection.py",
    "sizing.py",
    "state_paths.py",
    "trade_events.py",
    "typed_past/typed_data_freshness.py",
    "vol_target.py",
    "walk_forward/__init__.py",
    "walk_forward/loader.py",
    "walk_forward/manifest.py",
    # 2026-08-02 pin bump (Grant C, RenQuant#551): pipeline-side holiday-aware
    # trading-day bound, landed unwired then demoted to a private measurement
    # (pipeline#229, commits 11e49ad + 67b1edd); umbrella copy lags until ported.
    "walk_forward/leakage_guard.py",
})


def _normalize_imports(content: str) -> str:
    """Normalize kernel import paths so umbrella and pipeline forms compare equal."""
    content = re.sub(
        r"\bfrom\s+renquant_pipeline\.kernel\b", "from kernel", content
    )
    content = re.sub(
        r"\bimport\s+renquant_pipeline\.kernel\b", "import kernel", content
    )
    return content


def _list_py_files(root: Path) -> set[str]:
    result: set[str] = set()
    for dirpath, _, filenames in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for f in filenames:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, f), root)
                result.add(rel)
    return result


def _git_head(repo: Path) -> str | None:
    """HEAD commit of ``repo``, or None if it is not a readable git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _resolve_pipeline_kernel() -> Path | None:
    """Resolve the pipeline kernel path — refusing any tree not at the pin.

    2026-08-02 hardening: this guard's estimand is "umbrella kernel vs the
    PINNED pipeline kernel". The old order trusted the lock's ``local_path``
    (a mutable developer checkout) without verifying its HEAD, so on a
    machine whose sibling checkout lagged the pin the check silently measured
    a stale tree — the measured instance: sibling at a14dad11 vs pin
    60871e24 read two genuinely-drifted files as converged. A candidate now
    counts only if its checkout HEAD equals the locked commit; with no
    matching candidate the resolver returns None (exit 3: an honest skip),
    never a wrong-object measurement.

    Resolution order (EVERY leg, including the override, is verified against
    the locked commit when a lock is present — round-2 review: an unverified
    override is the same wrong-object measurement as an unverified
    local_path):
    1. ``RENQUANT_PIPELINE_KERNEL_PATH`` env var — explicit override.
       CI (``.github/workflows/kernel-parity-ci.yml``) reads the pin out of
       ``subrepos.lock.json`` and checks renquant-pipeline out AT that
       commit, so the same HEAD==pin verification passes there with no CI
       carve-out. ``git -C <kernel> rev-parse HEAD`` walks up to the
       enclosing repo, so no layout is assumed for override checkouts.
    2. ``.subrepo_runtime/repos/renquant-pipeline`` — the serving machine's
       pin-materialised clone (``subrepo_assemble.py --sync``).
    3. ``subrepos.lock.json``'s ``local_path``.
    4. The ``../renquant-pipeline`` filesystem sibling.
    Candidates that exist at the wrong commit are reported on stderr.
    """
    pinned_commit: str | None = None
    lock_local: Path | None = None
    if LOCK_FILE.exists():
        with open(LOCK_FILE) as fh:
            lock = json.load(fh)
        for sub in lock.get("subrepos", []):
            if sub.get("name") == "renquant-pipeline":
                pinned_commit = sub.get("commit")
                if sub.get("local_path"):
                    lock_local = Path(sub["local_path"])

    override = os.environ.get("RENQUANT_PIPELINE_KERNEL_PATH")
    if override:
        kernel = Path(override)
        if not kernel.is_dir():
            return None
        if pinned_commit is None:
            # No lock to verify against (exotic checkout): legacy trust.
            return kernel
        head = _git_head(kernel)
        if head == pinned_commit:
            return kernel
        print(
            f"check_kernel_parity: refusing override "
            f"RENQUANT_PIPELINE_KERNEL_PATH={override} — enclosing checkout "
            f"HEAD {(head or 'unreadable')[:12]} != locked pipeline commit "
            f"{pinned_commit[:12]} (an unverified override is a wrong-object "
            f"measurement)", file=sys.stderr)
        return None

    candidates = [
        UMBRELLA_ROOT / ".subrepo_runtime" / "repos" / "renquant-pipeline",
        lock_local,
        UMBRELLA_ROOT.parent / "renquant-pipeline",
    ]
    for repo in candidates:
        if repo is None:
            continue
        kernel = repo / "src" / "renquant_pipeline" / "kernel"
        if not kernel.is_dir():
            continue
        if pinned_commit is None:
            # No lock to verify against (exotic checkout): legacy best-effort.
            return kernel
        head = _git_head(repo)
        if head == pinned_commit:
            return kernel
        print(
            f"check_kernel_parity: refusing {repo} — HEAD "
            f"{(head or 'unreadable')[:12]} != locked pipeline commit "
            f"{pinned_commit[:12]} (a stale tree would be a wrong-object "
            f"measurement)", file=sys.stderr)
    return None


def check_parity(*, verbose: bool = False) -> tuple[list[str], dict[str, object]]:
    """Return (new_drift_files, summary_dict).

    new_drift_files: files that are in BOTH kernels, NOT in the allowlist,
    and have diverged (normalized). An empty list means no new drift.
    """
    pipeline_kernel = _resolve_pipeline_kernel()
    if pipeline_kernel is None:
        return [], {"skipped": True, "reason": "pipeline kernel not found"}

    if not UMBRELLA_KERNEL.is_dir():
        return [], {"skipped": True, "reason": "umbrella kernel not found"}

    u_files = _list_py_files(UMBRELLA_KERNEL)
    p_files = _list_py_files(pipeline_kernel)
    common = sorted(u_files & p_files)

    identical = []
    drifted_allowed = []
    drifted_new = []

    for f in common:
        u_content = (UMBRELLA_KERNEL / f).read_text()
        p_content = (pipeline_kernel / f).read_text()
        if _normalize_imports(u_content) == _normalize_imports(p_content):
            identical.append(f)
        elif f in KNOWN_DRIFT_ALLOWLIST:
            drifted_allowed.append(f)
        else:
            drifted_new.append(f)

    u_only = sorted(u_files - p_files)
    p_only = sorted(p_files - u_files)

    summary = {
        "skipped": False,
        "common_files": len(common),
        "identical": len(identical),
        "drifted_allowed": len(drifted_allowed),
        "drifted_new": len(drifted_new),
        "umbrella_only": len(u_only),
        "pipeline_only": len(p_only),
        "new_drift_files": drifted_new,
    }

    if verbose:
        print(f"Common: {len(common)}, Identical: {len(identical)}, "
              f"Drifted (allowed): {len(drifted_allowed)}, "
              f"Drifted (NEW): {len(drifted_new)}")
        print(f"Umbrella-only: {len(u_only)}, Pipeline-only: {len(p_only)}")
        if drifted_new:
            print("\n!! NEW DRIFT detected in:")
            for f in drifted_new:
                print(f"  {f}")
        if drifted_allowed:
            print(f"\nAllowlisted drift ({len(drifted_allowed)} files, expected)")

    return drifted_new, summary


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    new_drift, summary = check_parity(verbose=verbose)

    if summary.get("skipped"):
        print(f"SKIP: {summary['reason']}")
        return 3

    if not verbose:
        print(f"Kernel parity: {summary['identical']} identical, "
              f"{summary['drifted_allowed']} allowed-drift, "
              f"{summary['drifted_new']} NEW drift")

    if new_drift:
        print(f"\nFAIL: {len(new_drift)} file(s) have NEW drift "
              "(not in KNOWN_DRIFT_ALLOWLIST):")
        for f in new_drift:
            print(f"  {f}")
        print("\nIf this drift is intentional, add the file to "
              "KNOWN_DRIFT_ALLOWLIST in scripts/check_kernel_parity.py")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
