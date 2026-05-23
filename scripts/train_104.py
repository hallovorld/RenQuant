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
import copy
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


def _resolve_strategy_path(strategy_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else strategy_dir / path


def _strategy_relative(strategy_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(strategy_dir))
    except ValueError:
        return str(path)


def _staging_path_for(path: Path) -> Path:
    return path.with_suffix(".staging.json")


def _stage_training_artifact_paths(config: dict, strategy_dir: Path) -> tuple[dict, Path, Path, Path | None]:
    """Return a training config that writes all mutable artifacts to staging.

    The production safety boundary is: training may create candidate artifacts,
    but active production files are touched only by ``promote()`` after
    acceptance/WF gates.  This avoids the transient prod-clobber window where a
    long calibrator refresh can leave live readers pointed at an unaccepted
    model.
    """
    staged = copy.deepcopy(config)
    ranking = staged.setdefault("ranking", {}).setdefault("panel_scoring", {})
    panel_cfg = staged.setdefault("panel_ltr", {})

    panel_rel = ranking.get("artifact_path") or panel_cfg.get(
        "artifact_path", "artifacts/panel-ltr.json"
    )
    active_panel = _resolve_strategy_path(strategy_dir, panel_rel)
    candidate_panel = _staging_path_for(active_panel)
    candidate_panel_rel = _strategy_relative(strategy_dir, candidate_panel)
    panel_cfg["artifact_path"] = candidate_panel_rel
    ranking["artifact_path"] = candidate_panel_rel

    candidate_calibrator: Path | None = None
    gc_cfg = ranking.setdefault("global_calibration", {})
    cal_rel = gc_cfg.get("artifact_path")
    if cal_rel:
        candidate_calibrator = _staging_path_for(_resolve_strategy_path(strategy_dir, cal_rel))
        gc_cfg["artifact_path"] = _strategy_relative(strategy_dir, candidate_calibrator)

    ngb_train = panel_cfg.setdefault("ngboost", {})
    ngb_infer = ranking.setdefault("ngboost", {})
    ngb_rel = ngb_infer.get("artifact_path") or ngb_train.get("artifact_path")
    if ngb_rel:
        candidate_ngb = _staging_path_for(_resolve_strategy_path(strategy_dir, ngb_rel))
        candidate_ngb_rel = _strategy_relative(strategy_dir, candidate_ngb)
        ngb_train["artifact_path"] = candidate_ngb_rel
        ngb_infer["artifact_path"] = candidate_ngb_rel

    staged["_acceptance_staging"] = {
        "active_panel_artifact_path": str(active_panel),
        "candidate_panel_artifact_path": str(candidate_panel),
        "candidate_calibrator_artifact_path": (
            str(candidate_calibrator) if candidate_calibrator is not None else None
        ),
    }
    return staged, active_panel, candidate_panel, candidate_calibrator


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
    # 2026-05-15 fix: conditional_retrain_104.sh passes --trigger=<tag> to
    # tag what fired the off-cadence retrain (anomaly_spy_2pct / anomaly_vix_5pct /
    # anomaly_unknown). Used in audit logging; doesn't change training flow.
    # Pre-fix: argparse rejected unknown arg → "ntfy: training failed" alert
    # at 13:10 on 2026-05-15 (VIX +5.68% triggered retrain).
    p.add_argument(
        "--trigger",
        default="cadence",
        help="Tag identifying what fired this retrain (e.g. anomaly_vix_5pct). "
             "Logged but does not alter training flow. Default: 'cadence'.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight/artifact contract checks only. Does not train, "
             "write artifacts, or touch acceptance staging.",
    )
    p.add_argument(
        "--strict-contract",
        action="store_true",
        help="With --dry-run, hard-fail legacy panel artifacts that lack "
             "OOS evidence fields such as oos_mean_ic/oos_per_fold_ic.",
    )
    args = p.parse_args()
    log.info("train_104: trigger=%s strategy=%s force=%s",
             args.trigger, args.strategy, args.force)

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    config_path  = strategy_dir / args.strategy_config_name
    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    config = json.loads(config_path.read_text())
    # 2026-04-28 evening: stamp the active strategy_config filename.
    config["_strategy_config_name"] = args.strategy_config_name
    # External audit fix #2 (2026-04-29): stamp a run_id so all three
    # artifacts from one train run (panel-ltr, ngboost-head, calibrator)
    # share the same ID. Preflight can then verify they are aligned — a
    # mismatch means one artifact came from a different run (stale model).
    import uuid as _uuid  # noqa: PLC0415
    config["_train_run_id"] = str(_uuid.uuid4())[:8]  # 8-char prefix is enough

    if args.dry_run:
        from kernel.artifact_contract import build_run_bundle  # noqa: PLC0415
        from kernel.preflight import PreflightFailed, run_preflight  # noqa: PLC0415

        if args.strict_contract:
            config.setdefault("preflight", {}).setdefault(
                "artifact_contract", {}
            )["strict"] = True
        try:
            results = run_preflight(
                config,
                broker=None,
                strategy_dir=strategy_dir,
                broker_name=None,
                # Hard preflight failures must always make dry-run fail
                # closed. --strict-contract only controls whether legacy
                # missing OOS fields in P-PANEL-CONTRACT are promoted from
                # soft warnings to hard failures.
                strict=True,
            )
        except PreflightFailed as exc:
            log.error("dry-run preflight failed:\n%s", exc)
            sys.exit(2)
        failures = [r for r in results if not r.ok]
        bundle = build_run_bundle(
            config,
            strategy_dir,
            run_id=f"dryrun-{config['_train_run_id']}",
            run_type="train-dry-run",
        )
        log.info(
            "dry-run complete: checks=%d failures=%d panel_contract=%s "
            "watchlist=%s",
            len(results),
            len(failures),
            bundle.get("panel_contract", {}).get("ok"),
            bundle.get("watchlist_size"),
        )
        if failures:
            for res in failures:
                log.warning("dry-run check not ok: %s [%s] %s",
                            res.name, res.severity, res.message)
        return

    from kernel.pipeline.pp_training_full import (  # noqa: PLC0415
        FullTrainingContext,
        FullTrainingPipeline,
    )
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
    artifact_rel = (
        config.get("ranking", {}).get("panel_scoring", {}).get("artifact_path")
        or panel_cfg.get("artifact_path", "artifacts/panel-ltr.json")
    )
    active_path = _resolve_strategy_path(strategy_dir, artifact_rel)
    candidate_panel_path: Path | None = None
    candidate_calibrator_path: Path | None = None
    if acceptance_enabled:
        config, active_path, candidate_panel_path, candidate_calibrator_path = (
            _stage_training_artifact_paths(config, strategy_dir)
        )
        log.info(
            "Acceptance: training writes to staging panel=%s calibrator=%s; "
            "active remains %s",
            candidate_panel_path,
            candidate_calibrator_path,
            active_path,
        )
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
        # The pipeline now writes directly to the candidate staging path.
        # Legacy fallback is retained for side/diagnostic configurations that
        # intentionally disable the staging rewrite.
        staging_path = candidate_panel_path or active_path.with_suffix(".staging.json")
        if candidate_panel_path is None and active_path.exists():
            import shutil
            shutil.move(str(active_path), str(staging_path))
            if pre_train_snapshot is not None and pre_train_snapshot.exists():
                shutil.copy2(str(pre_train_snapshot), str(active_path))
        if not staging_path.exists():
            raise FileNotFoundError(
                f"Acceptance expected staging artifact but it is missing: {staging_path}"
            )

        # Phase 1 (2026-04-26): pass acceptance config so operator can
        # tune G4 max_degradation, G7 floor, severities without forking gate code.
        verdict = ModelAcceptanceGate(config=acceptance_cfg).evaluate(staging_path, active_path)
        log.info("\n%s", verdict.summary())

        archive_dir = strategy_dir / "artifacts" / "_acceptance_log"
        try:
            if verdict.all_hard_passed:
                # 2026-05-17 §5.13.15 fix — REMOVE daily RQ_ALLOW_NO_WF bypass.
                # Pre-fix: every daily retrain set RQ_ALLOW_NO_WF=1 + called
                # promote() → §5.13.15 "every daily promote set
                # RQ_ALLOW_NO_WF=1 — theatrical gate". Today's Sunday-sweep
                # incident (NGB val_IC=-0.0165 to prod) showed the
                # consequence: light G1-G11 gates aren't enough to catch
                # silent quality regressions on their own.
                #
                # New policy: daily retrain STAGES the new artifact at
                # *.staging.json. Promotion happens via weekly_wf_promote.sh
                # (Saturday 04:00 PT) which runs the full WF 3-cut +
                # §5.2 sanity battery + run_wf_gate.py before flipping
                # the production symlink. Daily retrain keeps the panel
                # warm but does NOT touch production.
                #
                # Emergency override: set RQ_ALLOW_NO_WF=1 in the calling
                # shell environment (NOT here — must be explicit per cron
                # invocation, not script-default).
                import os as _os                                   # noqa: PLC0415
                if _os.environ.get("RQ_ALLOW_NO_WF") == "1":
                    log.warning(
                        "DAILY RETRAIN: RQ_ALLOW_NO_WF=1 set externally — "
                        "promoting without WF gate. This is emergency-only "
                        "(CLAUDE.md §5.5 rollback rehearsal applies)."
                    )
                    promote(staging_path, active_path)
                else:
                    log.info(
                        "Acceptance: ALL HARD GATES PASSED → STAGED at %s. "
                        "Production NOT updated. weekly_wf_promote.sh "
                        "(Saturday 04:00 PT) runs WF 3-cut + sanity + "
                        "promotes if gate passes. Set RQ_ALLOW_NO_WF=1 "
                        "to override (emergency only).",
                        staging_path.name,
                    )
                    # Restore the active path file from the pre-train
                    # snapshot since we already moved active→staging at line
                    # 162. Without this, the active path is "the pre-train
                    # copy we restored at line 165" which is correct already.
                    # Just leave it.
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
