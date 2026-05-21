"""Research acceptance pipeline for renquant_104.

This module is intentionally orchestration-only.  The model training,
walk-forward gate, true-OOS evaluation, and DSR/PBO stamping remain in their
single-purpose scripts; this pipeline owns the dependency graph, command
contracts, and safe parallel execution.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .pipeline import Job, Task, resolve_workers

log = logging.getLogger("kernel.pipeline.research_acceptance")

TARGET_CONTRACTS = "contracts"
TARGET_TRUE_OOS = "true-oos"
TARGET_WF_GATE = "wf-gate"
ALL_TARGETS = (TARGET_CONTRACTS, TARGET_TRUE_OOS, TARGET_WF_GATE)


@dataclass(frozen=True)
class CommandSpec:
    """A shell-free command contract."""

    name: str
    argv: tuple[str, ...]
    cwd: Path


@dataclass(frozen=True)
class CommandResult:
    """Normalized command result for subprocess and fake test runners."""

    spec: CommandSpec
    returncode: int = 0


Runner = Callable[[CommandSpec], CommandResult | subprocess.CompletedProcess | int]


def _default_runner(spec: CommandSpec) -> CommandResult:
    proc = subprocess.run(list(spec.argv), cwd=spec.cwd)
    return CommandResult(spec=spec, returncode=int(proc.returncode))


@dataclass
class ResearchAcceptanceContext:
    """Runtime config and command ledger for research acceptance."""

    repo: Path
    python: str = sys.executable
    targets: tuple[str, ...] = (TARGET_CONTRACTS,)
    workers: int | None = None
    dry_run: bool = False
    train_cutoff: str = "2024-07-01"
    artifact_dir: Path | None = None
    eval_json_path: Path | None = None
    artifact: Path | None = None
    strategy_config: str = "strategy_config.sim_wl200.json"
    wf_jobs: int = 3
    strict: bool = True
    skip_retrain: bool = False
    runner: Runner = _default_runner
    executed: list[CommandSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.repo = Path(self.repo)
        if self.artifact_dir is None:
            self.artifact_dir = Path(
                "backtesting/renquant_104/artifacts/walkforward_truly_oos_2024-07-01"
            )
        self.artifact_dir = self._resolve(self.artifact_dir)
        if self.eval_json_path is None:
            self.eval_json_path = Path(
                "backtesting/renquant_104/artifacts/prod/truly_oos_eval/eval_truly_oos.json"
            )
        self.eval_json_path = self._resolve(self.eval_json_path)
        if self.artifact is not None:
            self.artifact = self._resolve(self.artifact)

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.repo / p

    @property
    def eval_json(self) -> Path:
        return self.eval_json_path

    def command(self, name: str, *argv: str | Path) -> CommandSpec:
        return CommandSpec(
            name=name,
            argv=tuple(str(a) for a in argv),
            cwd=self.repo,
        )

    def run_command(self, spec: CommandSpec) -> CommandResult:
        self.executed.append(spec)
        log.info("research-acceptance: %s", " ".join(spec.argv))
        if self.dry_run:
            return CommandResult(spec=spec, returncode=0)
        raw = self.runner(spec)
        if isinstance(raw, CommandResult):
            result = raw
        elif isinstance(raw, subprocess.CompletedProcess):
            result = CommandResult(spec=spec, returncode=int(raw.returncode))
        else:
            result = CommandResult(spec=spec, returncode=int(raw))
        if result.returncode != 0:
            raise RuntimeError(
                f"{spec.name} failed with rc={result.returncode}: "
                f"{' '.join(spec.argv)}"
            )
        return result


class CommandTask(Task):
    """Task wrapper around a command factory."""

    def __init__(self, name: str, factory: Callable[[ResearchAcceptanceContext], CommandSpec]):
        self._name = name
        self._factory = factory

    @property
    def name(self) -> str:
        return self._name

    def run(self, ctx: ResearchAcceptanceContext) -> bool:
        ctx.run_command(self._factory(ctx))
        return True


class ContractsJob(Job):
    """Fast code/test contracts that do not train models."""

    @property
    def tasks(self) -> list[Task]:
        return [
            CommandTask("py_compile_research_tools", _compile_cmd),
            CommandTask("pytest_research_contracts", _pytest_contracts_cmd),
        ]


class TrueOOSJob(Job):
    """Cutoff train -> strict post-cutoff eval -> DSR/PBO stamp."""

    @property
    def tasks(self) -> list[Task]:
        tasks: list[Task] = []
        if not getattr(self, "_skip_retrain", False):
            tasks.append(CommandTask("retrain_true_oos", _retrain_true_oos_cmd))
        tasks.extend([
            CommandTask("eval_true_oos", _eval_true_oos_cmd),
            CommandTask("stamp_dsr_pbo", _dsr_pbo_cmd),
        ])
        return tasks

    def run(self, ctx: ResearchAcceptanceContext) -> None:
        self._skip_retrain = ctx.skip_retrain
        super().run(ctx)


class WFGateJob(Job):
    """Run the existing strict WF gate for a staging artifact."""

    def should_skip(self, ctx: ResearchAcceptanceContext) -> bool:
        return ctx.artifact is None

    @property
    def tasks(self) -> list[Task]:
        return [CommandTask("wf_gate", _wf_gate_cmd)]


class ResearchAcceptancePipeline:
    """Parallel outer pipeline for independent acceptance jobs."""

    def __init__(self, targets: Sequence[str] | None = None):
        self.targets = normalize_targets(targets or (TARGET_CONTRACTS,))

    def jobs_for(self, ctx: ResearchAcceptanceContext) -> list[Job]:
        jobs: list[Job] = []
        if TARGET_CONTRACTS in self.targets:
            jobs.append(ContractsJob())
        if TARGET_TRUE_OOS in self.targets:
            jobs.append(TrueOOSJob())
        if TARGET_WF_GATE in self.targets:
            jobs.append(WFGateJob())
        return [job for job in jobs if not job.should_skip(ctx)]

    def run(self, ctx: ResearchAcceptanceContext) -> None:
        jobs = self.jobs_for(ctx)
        if not jobs:
            log.warning("research-acceptance: no runnable jobs for targets=%s", self.targets)
            return
        n_workers = resolve_workers(ctx.workers, len(jobs))
        log.info("research-acceptance: %d jobs, %d workers", len(jobs), n_workers)
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="research") as ex:
            futures = {ex.submit(job.run, ctx): type(job).__name__ for job in jobs}
            failures: list[str] = []
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 - report all job failures
                    failures.append(f"{name}: {type(exc).__name__}: {exc}")
                    log.error("research-acceptance: %s failed: %s", name, exc)
            if failures:
                raise RuntimeError("; ".join(failures))
        log.info("research-acceptance: DONE %.2fs", time.monotonic() - t0)


def normalize_targets(targets: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in targets:
        target = raw.strip().lower()
        if target == "all":
            for item in ALL_TARGETS:
                if item not in out:
                    out.append(item)
            continue
        if target not in ALL_TARGETS:
            raise ValueError(f"unknown research acceptance target: {raw!r}")
        if target not in out:
            out.append(target)
    return tuple(out)


def _compile_cmd(ctx: ResearchAcceptanceContext) -> CommandSpec:
    return ctx.command(
        "py_compile_research_tools",
        ctx.python, "-m", "py_compile",
        "kernel/regime_labels.py",
        "scripts/patchtst_doe_sweep.py",
        "scripts/retrain_prod_truly_oos.py",
        "scripts/eval_truly_oos.py",
        "scripts/dsr_pbo_truly_oos.py",
        "scripts/eval_prod_vs_shadow.py",
        "scripts/diagnose_calibrator_saturation.py",
        "scripts/transformer_v4.py",
    )


def _pytest_contracts_cmd(ctx: ResearchAcceptanceContext) -> CommandSpec:
    return ctx.command(
        "pytest_research_contracts",
        ctx.python, "-m", "pytest",
        "tests/test_patchtst_doe_sweep.py",
        "tests/test_prod_signal_truly_oos.py",
        "-q",
    )


def _retrain_true_oos_cmd(ctx: ResearchAcceptanceContext) -> CommandSpec:
    return ctx.command(
        "retrain_true_oos",
        ctx.python, "scripts/retrain_prod_truly_oos.py",
        "--train-cutoff", ctx.train_cutoff,
        "--output-dir", ctx.artifact_dir,
    )


def _eval_true_oos_cmd(ctx: ResearchAcceptanceContext) -> CommandSpec:
    return ctx.command(
        "eval_true_oos",
        ctx.python, "scripts/eval_truly_oos.py",
        "--artifact-dir", ctx.artifact_dir,
        "--out", ctx.eval_json,
    )


def _dsr_pbo_cmd(ctx: ResearchAcceptanceContext) -> CommandSpec:
    return ctx.command(
        "stamp_dsr_pbo",
        ctx.python, "scripts/dsr_pbo_truly_oos.py",
        "--eval-json", ctx.eval_json,
    )


def _wf_gate_cmd(ctx: ResearchAcceptanceContext) -> CommandSpec:
    if ctx.artifact is None:
        raise ValueError("wf-gate target requires ctx.artifact")
    argv: list[str | Path] = [
        ctx.python, "scripts/run_wf_gate.py",
        "--artifact", ctx.artifact,
        "--strategy-config", ctx.strategy_config,
        "--jobs", str(ctx.wf_jobs),
    ]
    if ctx.strict:
        argv.append("--strict")
    return ctx.command("wf_gate", *argv)


__all__ = [
    "ALL_TARGETS",
    "CommandResult",
    "CommandSpec",
    "ResearchAcceptanceContext",
    "ResearchAcceptancePipeline",
    "normalize_targets",
]
