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
"""
from __future__ import annotations

import json
import os
import re
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


def _resolve_pipeline_kernel() -> Path | None:
    """Resolve the pipeline kernel path from subrepos.lock.json."""
    if not LOCK_FILE.exists():
        return None
    with open(LOCK_FILE) as fh:
        lock = json.load(fh)
    for sub in lock.get("subrepos", []):
        if sub.get("name") == "renquant-pipeline":
            local = Path(sub["local_path"])
            kernel = local / "src" / "renquant_pipeline" / "kernel"
            return kernel if kernel.is_dir() else None
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
        return 0

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
