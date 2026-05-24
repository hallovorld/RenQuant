"""DailyRetrainAlpha158FundPipeline — production retrain orchestration.

Per CLAUDE.md §1b ("every logical unit is a Task/Job/Pipeline") + §1c
("split every complex structure"). Replaces the hand-rolled
scripts/daily_retrain_alpha158_fund.sh with a proper Pipeline composed
of 4 atomic Tasks, each with `should_skip(ctx)` for input-mtime
caching so repeated invocations don't redo unchanged work.

Pipeline phases (strictly sequential — each output is next input):

    1. BuildAlpha158PanelTask   — calls scripts.build_alpha158_qlib
    2. MergeFundFeaturesTask    — calls scripts.build_alpha158_fund_panel
    3. TrainPanelLTRTask        — calls scripts.train_production_model +
                                   copies artifact to live config path
    4. RefitCalibratorTask      — calls scripts.fit_calibrator_alpha158_fund

Concurrency: there is no inter-Task parallelism opportunity (strict
data dependency). Internal parallelism INSIDE BuildAlpha158PanelTask
(per-ticker feature loop is sequential in build_alpha158_qlib.py) is
a separate optimization tracked outside this pipeline.

Caching: each Task's `should_skip` reads the mtime of its inputs and
outputs. If output is newer than every input, the Task is skipped
with reason logged.

Usage (from cron via daily_retrain_alpha158_fund.sh):
    python -m training_panel.daily_retrain_alpha158_fund

Usage (interactive):
    from training_panel.daily_retrain_alpha158_fund import (
        DailyRetrainAlpha158FundPipeline, DailyRetrainContext)
    ctx = DailyRetrainContext()
    DailyRetrainAlpha158FundPipeline().run(ctx)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("daily-retrain-alpha158-fund")

REPO = Path(__file__).resolve().parents[3]


@dataclass
class DailyRetrainContext:
    """State container for the daily retrain pipeline."""
    repo_dir: Path = REPO
    strategy_dir: Path = REPO / "backtesting" / "renquant_104"
    artifacts_dir: Path = REPO / "backtesting" / "renquant_104" / "artifacts"

    # Artifact paths (inputs and outputs along the chain)
    ohlcv_dir:           Path = REPO / "data" / "ohlcv"
    alpha158_panel:      Path = REPO / "data" / "alpha158_qlib_dataset.parquet"
    sec_fund_panel:      Path = REPO / "data" / "sec_fundamentals_daily.parquet"
    fund_merged_panel:   Path = REPO / "data" / "alpha158_291_fundamental_dataset.parquet"
    xgb_artifact_src:    Path = REPO / "data" / "panel-ltr-prod-alpha158-fund-fwd60d.json"
    # 2026-05-11 sim/prod isolation: prod artifacts live under artifacts/prod/.
    # The flat artifacts/<file>.json paths these used to write to are no longer
    # served by the prod runner; without this fix daily retrain succeeded but
    # live trading would read the stale frozen model forever.
    xgb_artifact_dst:    Path = REPO / "backtesting" / "renquant_104" / "artifacts" / "prod" / "panel-ltr.alpha158_fund.json"
    calibrator_artifact: Path = REPO / "backtesting" / "renquant_104" / "artifacts" / "prod" / "panel-rank-calibration.json"

    # Run telemetry (populated as Tasks execute)
    skipped: list[str] = field(default_factory=list)
    elapsed: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    truncate_to_sec_max: bool = True


class RetrainTask(ABC):
    """Atomic step in the daily retrain pipeline."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def should_skip(self, ctx: DailyRetrainContext) -> str | None:
        """Return reason if Task can be skipped (inputs unchanged), else None."""
        return None

    @abstractmethod
    def run(self, ctx: DailyRetrainContext) -> None: ...


def _newest_mtime(*paths: Path) -> float:
    """Newest mtime of given paths (recursively for directories)."""
    times: list[float] = []
    for p in paths:
        if not p.exists():
            continue
        if p.is_dir():
            times.extend(q.stat().st_mtime for q in p.rglob("*") if q.is_file())
        else:
            times.append(p.stat().st_mtime)
    return max(times) if times else 0.0


class BuildAlpha158PanelTask(RetrainTask):
    """Phase 1 — recompute the 148-feature alpha158 panel from latest OHLCV.

    Output: data/alpha158_qlib_dataset.parquet (+ .stats.json sidecar).
    """

    def should_skip(self, ctx: DailyRetrainContext) -> str | None:
        if not ctx.alpha158_panel.exists():
            return None
        if ctx.alpha158_panel.stat().st_mtime > _newest_mtime(ctx.ohlcv_dir):
            return "alpha158 panel newer than every OHLCV file"
        return None

    def run(self, ctx: DailyRetrainContext) -> None:
        _run_script(ctx.repo_dir / "scripts" / "build_alpha158_qlib.py")


class MergeFundFeaturesTask(RetrainTask):
    """Phase 2 — left-join alpha158 panel with 5 SEC fund features.

    Output: data/alpha158_291_fundamental_dataset.parquet (163 features).
    """

    def should_skip(self, ctx: DailyRetrainContext) -> str | None:
        if not ctx.fund_merged_panel.exists():
            return None
        if ctx.fund_merged_panel.stat().st_mtime > _newest_mtime(
            ctx.alpha158_panel, ctx.sec_fund_panel
        ):
            return "merged panel newer than alpha158 + fund inputs"
        return None

    def run(self, ctx: DailyRetrainContext) -> None:
        args = ["--truncate-to-sec-max"] if ctx.truncate_to_sec_max else []
        _run_script(ctx.repo_dir / "scripts" / "build_alpha158_fund_panel.py", args)


class TrainPanelLTRTask(RetrainTask):
    """Phase 3 — fit XGBoost rank:pairwise on 163-feature panel.

    Trains via scripts/train_production_model.py and writes directly to
    ``ctx.xgb_artifact_dst``. Production runs may point that at the active
    artifact; acceptance/WF runs point it at a staged candidate path.
    """

    def should_skip(self, ctx: DailyRetrainContext) -> str | None:
        if not ctx.xgb_artifact_dst.exists():
            return None
        if ctx.xgb_artifact_dst.stat().st_mtime > ctx.fund_merged_panel.stat().st_mtime:
            return "live XGB artifact newer than merged panel"
        return None

    def run(self, ctx: DailyRetrainContext) -> None:
        _run_script(
            ctx.repo_dir / "scripts" / "train_production_model.py",
            ["--output-path", str(ctx.xgb_artifact_dst)],
        )
        if not ctx.xgb_artifact_dst.exists():
            raise FileNotFoundError(
                f"train_production_model.py did not produce {ctx.xgb_artifact_dst}"
            )
        log.info("  artifact → %s", ctx.xgb_artifact_dst.relative_to(ctx.repo_dir))


class RefitCalibratorTask(RetrainTask):
    """Phase 4 — refit the global panel calibrator on new XGB predictions.

    Output: artifacts/panel-rank-calibration.json (84+ unique probability
    bins; preflight P-CALIBRATOR-HEALTH requires ≥10).
    """

    def should_skip(self, ctx: DailyRetrainContext) -> str | None:
        if not ctx.calibrator_artifact.exists():
            return None
        if ctx.calibrator_artifact.stat().st_mtime > ctx.xgb_artifact_dst.stat().st_mtime:
            return "calibrator newer than XGB artifact"
        return None

    def run(self, ctx: DailyRetrainContext) -> None:
        _run_script(
            ctx.repo_dir / "scripts" / "fit_calibrator_alpha158_fund.py",
            [
                "--scorer-artifact", str(ctx.xgb_artifact_dst),
                "--out", str(ctx.calibrator_artifact),
            ],
        )


def _run_script(script: Path, args: list[str] | None = None,
                cwd: Path | None = None) -> None:
    """Run a Python script in a subprocess; raise on non-zero exit.

    cwd defaults to the repo root (script.parents[1]) so scripts that
    use repo-relative paths like 'data/foo.parquet' work regardless
    of the parent process's cwd. The bash launcher cd's to the
    strategy dir for module-import purposes, but child Python scripts
    expect repo root.
    """
    if cwd is None:
        cwd = script.resolve().parents[1]   # scripts/foo.py → repo root
    cmd = [sys.executable, str(script), *(args or [])]
    log.info("  $ %s  (cwd=%s)", " ".join(cmd), cwd)
    rc = subprocess.call(cmd, cwd=str(cwd))
    if rc != 0:
        raise RuntimeError(f"{script.name} exited {rc}")


class DailyRetrainAlpha158FundPipeline:
    """Four-Task linear pipeline. Each Task implements should_skip → run."""

    @property
    def tasks(self) -> list[RetrainTask]:
        return [
            BuildAlpha158PanelTask(),
            MergeFundFeaturesTask(),
            TrainPanelLTRTask(),
            RefitCalibratorTask(),
        ]

    def run(self, ctx: DailyRetrainContext) -> DailyRetrainContext:
        log.info("DailyRetrainAlpha158FundPipeline START")
        t0 = time.monotonic()
        for task in self.tasks:
            reason = task.should_skip(ctx)
            if reason is not None:
                log.info("── %s SKIP — %s", task.name, reason)
                ctx.skipped.append(task.name)
                continue
            t1 = time.monotonic()
            log.info("── %s START", task.name)
            try:
                task.run(ctx)
            except Exception as exc:
                ctx.errors.append(f"{task.name}: {exc}")
                log.error("── %s FAILED — %s", task.name, exc)
                raise
            dt = time.monotonic() - t1
            ctx.elapsed[task.name] = dt
            log.info("── %s DONE %.1fs", task.name, dt)
        total = time.monotonic() - t0
        log.info("DailyRetrainAlpha158FundPipeline DONE  total=%.1fs  skipped=%s",
                 total, ctx.skipped or "none")
        return ctx


def _staging_path(path: Path) -> Path:
    return path.with_suffix(".staging.json")


def _resolve_output_override(repo_dir: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_dir / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Write alpha158 panel scorer + calibrator to *.staging.json "
             "candidate paths instead of production paths.",
    )
    parser.add_argument(
        "--xgb-artifact-out",
        default=None,
        help="Override panel scorer output path.",
    )
    parser.add_argument(
        "--calibrator-out",
        default=None,
        help="Override calibrator output path.",
    )
    parser.add_argument(
        "--no-truncate-to-sec-max",
        action="store_true",
        help="Do not truncate alpha158 rows beyond SEC fundamentals coverage. "
             "Mostly useful for validating the guard itself.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ctx = DailyRetrainContext()
    if args.xgb_artifact_out:
        ctx.xgb_artifact_dst = _resolve_output_override(ctx.repo_dir, args.xgb_artifact_out)
    if args.calibrator_out:
        ctx.calibrator_artifact = _resolve_output_override(ctx.repo_dir, args.calibrator_out)
    if args.staged:
        if not args.xgb_artifact_out:
            ctx.xgb_artifact_dst = _staging_path(ctx.xgb_artifact_dst)
        if not args.calibrator_out:
            ctx.calibrator_artifact = _staging_path(ctx.calibrator_artifact)
        log.info(
            "staged mode: scorer=%s calibrator=%s",
            ctx.xgb_artifact_dst,
            ctx.calibrator_artifact,
        )
    ctx.truncate_to_sec_max = not args.no_truncate_to_sec_max
    DailyRetrainAlpha158FundPipeline().run(ctx)


if __name__ == "__main__":
    main()
