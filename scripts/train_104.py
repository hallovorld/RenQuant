#!/usr/bin/env python
"""End-to-end training driver for renquant_104.

Thin entrypoint — all logic lives in
`backtesting/renquant_104/kernel/pipeline/pp_training_full.py` as a
FullTrainingPipeline (BaselineTournamentJob → PanelTrainingJob →
RecalibrationJob), matching the Job/Task conventions used by the inference
and panel-training pipelines.

Usage::

    python scripts/train_104.py
    python scripts/train_104.py --skip-baseline     # only retrain panel + recalibrate
    python scripts/train_104.py --skip-panel        # only retrain per-ticker tournament
    python scripts/train_104.py --skip-recalibrate  # skip the blend-weight refresh
"""
from __future__ import annotations

# CLAUDE.md §5.10: saturate the local hardware. M2 Pro has 10 cores;
# default BLAS / OpenMP env vars often leave ~60% idle. Set BEFORE any
# numpy / xgboost / ngboost import so libraries pick up the setting.
import os as _os
for _k, _v in (("OMP_NUM_THREADS", "10"),
               ("MKL_NUM_THREADS", "10"),
               ("OPENBLAS_NUM_THREADS", "10"),
               ("VECLIB_MAXIMUM_THREADS", "10"),
               ("NUMEXPR_NUM_THREADS", "10")):
    _os.environ.setdefault(_k, _v)

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-104")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",          default="renquant_104")
    p.add_argument("--skip-baseline",     action="store_true")
    p.add_argument("--skip-panel",        action="store_true")
    p.add_argument("--skip-recalibrate",  action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore the training.cadence gate (run even on non-cadence days).",
    )
    p.add_argument(
        "--strategy-config-name",
        default="strategy_config.json",
        help="Filename of the strategy config (default: strategy_config.json). "
             "Use a side config like strategy_config.hourly_transformer.json "
             "for ablations / Stage C-3 experiments without touching production.",
    )
    p.add_argument(
        "--skip-acceptance",
        action="store_true",
        help="Bypass acceptance gates for this run (operator override). "
             "DANGEROUS — only use for known-broken-but-recoverable cases.",
    )
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    config_path  = strategy_dir / args.strategy_config_name
    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    from kernel.pipeline.pp_training_full import (  # noqa: PLC0415
        FullTrainingContext,
        FullTrainingPipeline,
    )

    config = json.loads(config_path.read_text())
    # 2026-04-28 evening: stamp the active strategy_config filename.
    config["_strategy_config_name"] = args.strategy_config_name
    # External audit fix #2 (2026-04-29): stamp a run_id so all three
    # artifacts from one train run (panel-ltr, ngboost-head, calibrator)
    # share the same ID. Preflight can then verify they are aligned — a
    # mismatch means one artifact came from a different run (stale model).
    import uuid as _uuid  # noqa: PLC0415
    config["_train_run_id"] = str(_uuid.uuid4())[:8]  # 8-char prefix is enough
    # Audit fix #152 (2026-04-26 round-7): acceptance gates wrap the
    # FullTrainingPipeline output. If the new artifact fails any hard
    # gate, the prior production artifact is preserved at panel-ltr.json
    # — live runner sees no change → ZERO downtime on bad retrain.
    #
    # User spec: "我们有没有机制进行模型accpetance verification，如果不
    # 通过的话，继续用原来的模型跑E2E？这关系到工程稳定性和可用性！"
    #
    # Disable via `acceptance.enabled = false` in strategy_config.json
    # (default ON). Operator can also pass --skip-acceptance to bypass
    # for one run.
    acceptance_cfg = config.get("acceptance", {})
    acceptance_enabled = bool(acceptance_cfg.get("enabled", True)) and not args.skip_acceptance

    # Snapshot the active panel-ltr artifact BEFORE training runs.
    # BUG-G7 fix (2026-04-28): respect `panel_ltr.artifact_path` from
    # config instead of hardcoding `panel-ltr.json`. Pre-fix, side
    # configs (like strategy_config.wl178.json) wrote new artifacts to
    # configured side paths but the acceptance gate evaluated PROD
    # (panel-ltr.json) → reported "PASS metric=0.0400" when the new
    # model actually had mean_ic=+0.0067. The "PASS" was the prior
    # production model passing against itself, not the new model.
    panel_cfg = config.get("panel_ltr", {})
    artifact_rel = panel_cfg.get("artifact_path", "artifacts/panel-ltr.json")
    active_path = strategy_dir / artifact_rel
    pre_train_snapshot = None
    if acceptance_enabled and active_path.exists():
        import shutil
        pre_train_snapshot = active_path.with_suffix(".pre-train.json")
        shutil.copy2(str(active_path), str(pre_train_snapshot))
        log.info("Acceptance: snapshotted active artifact to %s", pre_train_snapshot.name)

    ctx = FullTrainingContext(
        config=config,
        strategy=args.strategy,
        strategy_dir=strategy_dir,
        skip_baseline=args.skip_baseline,
        skip_panel=args.skip_panel,
        skip_recalibrate=args.skip_recalibrate,
        force_retrain=args.force,
    )
    FullTrainingPipeline().run(ctx)

    if acceptance_enabled:
        from kernel.model_acceptance import (  # noqa: PLC0415
            ModelAcceptanceGate, promote, reject,
        )
        # The pipeline writes new content to the SAME path (panel-ltr.json)
        # via SaveArtifactTask + shim. So at this point, panel-ltr.json
        # = NEW (staging), pre-train snapshot = PRIOR.
        # Move new content to .staging.json so the gate APIs match
        # (separate staging vs active paths).
        staging_path = active_path.with_suffix(".staging.json")
        if active_path.exists():
            import shutil
            shutil.move(str(active_path), str(staging_path))
        # Restore prior at active for gate evaluation context.
        if pre_train_snapshot is not None and pre_train_snapshot.exists():
            shutil.copy2(str(pre_train_snapshot), str(active_path))

        # Phase 1 (2026-04-26): pass acceptance config so operator can
        # tune G4 max_degradation, G7 floor, severities without forking gate code.
        verdict = ModelAcceptanceGate(config=acceptance_cfg).evaluate(staging_path, active_path)
        log.info("\n%s", verdict.summary())

        archive_dir = strategy_dir / "artifacts" / "_acceptance_log"
        try:
            if verdict.all_hard_passed:
                log.info("Acceptance: ALL HARD GATES PASSED → promoting new model")
                promote(staging_path, active_path)
            else:
                log.error("Acceptance: HARD GATE FAILED → keeping prior model")
                reject(staging_path, archive_dir, verdict)
                # Try ntfy alert (best-effort, do not block on failure)
                try:
                    import subprocess
                    msg = f"RENQUANT-104 RETRAIN REJECTED: {len(verdict.hard_failures())} hard gate(s) failed. Prior model preserved. See {archive_dir}/"
                    subprocess.run(
                        ["curl", "-sf", "-H", f"Title: RenQuant 104 RETRAIN REJECTED",
                         "-d", msg, "https://ntfy.sh/renquant"],
                        timeout=10, check=False,
                    )
                except Exception:
                    pass
                # Exit non-zero so the operator script sees the failure.
                sys.exit(2)
        finally:
            # Audit fix #9 (2026-04-26): always clean up the pre-train
            # snapshot, success or rejection. Pre-fix, rejection path
            # left .pre-train.json files lingering in artifacts/, which
            # confused operators investigating failures.
            if pre_train_snapshot and pre_train_snapshot.exists():
                try:
                    pre_train_snapshot.unlink()
                except OSError as exc:
                    log.warning("could not remove pre-train snapshot %s: %s",
                                pre_train_snapshot, exc)


if __name__ == "__main__":
    main()
