#!/usr/bin/env python3
"""Fail-loud sanity check for the vendored portfolio-QP runtime.

Daily operations import the pipeline from ``RENQUANT_SUBREPO_ROOT`` or
``.subrepo_assembly/current.env``. A stale vendored snapshot can keep
running old QP code without crashing. This script imports the post-2026-05-30
QP symbols from the resolved runtime root and verifies the modules are loaded
from that root, not from the umbrella fallback tree.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from subrepo_paths import resolve_subrepo_root  # noqa: E402


@dataclass(frozen=True)
class RuntimeSymbol:
    module: str
    attr: str
    repo: str
    why: str


@dataclass(frozen=True)
class RuntimeCommand:
    """A CLI subcommand that must be registered on a runtime package.

    The check shells out ``python -m <module> --help`` and greps the
    output for the subcommand name. Subprocess (not argparse
    introspection) because we want the post-PYTHONPATH-mutation
    behaviour: if the vendored package is stale, ``python -m`` resolves
    to the stale entry point and the subparser is absent, exactly
    mirroring what daily_104.sh / intraday_sell_104.sh see at runtime.
    """

    module: str
    subcommand: str
    repo: str
    why: str


REQUIRED_SYMBOLS: tuple[RuntimeSymbol, ...] = (
    RuntimeSymbol(
        "renquant_pipeline.kernel.portfolio_qp.davis_norman",
        "davis_norman_band_clamped",
        "renquant-pipeline",
        "Davis-Norman no-trade band path",
    ),
    RuntimeSymbol(
        "renquant_pipeline.kernel.portfolio_qp.proportional_trade",
        "proportional_trade_target",
        "renquant-pipeline",
        "partial-horizon proportional trade path",
    ),
    RuntimeSymbol(
        "renquant_pipeline.kernel.portfolio_qp.constraint_snapshot",
        "ConstraintSnapshot",
        "renquant-pipeline",
        "hard-constraint snapshot contract",
    ),
    RuntimeSymbol(
        "renquant_pipeline.kernel.portfolio_qp.qp_solver",
        "solve_portfolio_qp_from_snapshot",
        "renquant-pipeline",
        "snapshot-based solver entry point",
    ),
    RuntimeSymbol(
        "renquant_pipeline.kernel.portfolio_qp.baseline_allocators",
        "hybrid_option_f_allocator",
        "renquant-pipeline",
        "Hybrid Option F offline A/B candidate",
    ),
    RuntimeSymbol(
        "renquant_pipeline.kernel.portfolio_qp.baseline_allocators",
        "hard_only_qp_allocator",
        "renquant-pipeline",
        "hard-only QP offline A/B baseline",
    ),
    RuntimeSymbol(
        "renquant_pipeline.kernel.portfolio_qp.allocator_replay",
        "replay_all",
        "renquant-pipeline",
        "paired offline replay harness",
    ),
    RuntimeSymbol(
        "renquant_pipeline.kernel.portfolio_qp.replay_significance",
        "compute_significance_verdicts",
        "renquant-pipeline",
        "DSR/PBO significance pass",
    ),
    RuntimeSymbol(
        "renquant_pipeline.kernel.portfolio_qp.wf_replay_loader",
        "load_replay_bars_from_sim_db",
        "renquant-pipeline",
        "walk-forward cut loader",
    ),
    RuntimeSymbol(
        "renquant_pipeline.kernel.portfolio_qp.run_ab_replay",
        "run_replay",
        "renquant-pipeline",
        "Step 4g evidence driver",
    ),
    RuntimeSymbol(
        "renquant_common.metrics.deflated_sharpe",
        "deflated_sharpe_ratio",
        "renquant-common",
        "shared DSR metric dependency",
    ),
    RuntimeSymbol(
        "renquant_common.metrics.pbo",
        "probability_of_backtest_overfitting",
        "renquant-common",
        "shared PBO metric dependency",
    ),
    RuntimeSymbol(
        "renquant_common.metrics.hac_se",
        "hac_t_stat",
        "renquant-common",
        "shared HAC t-stat metric dependency",
    ),
)


def _repo_src(root: Path, repo: str) -> Path:
    return (root / repo / "src").resolve()


def _add_runtime_srcs(root: Path) -> None:
    """Prepend runtime-root source dirs in dependency order."""
    ordered = (
        "renquant-common",
        "renquant-base-data",
        "renquant-artifacts",
        "renquant-model",
        "renquant-pipeline",
        "renquant-execution",
        "renquant-strategy-104",
        "renquant-backtesting",
        "renquant-orchestrator",
    )
    for repo in reversed(ordered):
        src = _repo_src(root, repo)
        if src.is_dir():
            sys.path.insert(0, str(src))


def _origin_under_runtime(module: str, expected_src: Path) -> tuple[bool, str]:
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError as exc:
        return False, f"module parent missing: {exc.name}"
    if spec is None or spec.origin is None:
        return False, "module spec not found"
    origin = Path(spec.origin).resolve()
    try:
        origin.relative_to(expected_src)
    except ValueError:
        return False, f"loaded from {origin}, expected under {expected_src}"
    return True, str(origin)


REQUIRED_COMMANDS: tuple[RuntimeCommand, ...] = (
    RuntimeCommand(
        "renquant_orchestrator",
        "live-bridge",
        "renquant-orchestrator",
        "intraday + daily live-broker entry (scripts/intraday_sell_104.sh:88, scripts/daily_104.sh:496)",
    ),
    RuntimeCommand(
        "renquant_orchestrator",
        "daily-bridge",
        "renquant-orchestrator",
        "daily multirepo runner entry (scripts/daily_104.sh:295)",
    ),
)


def _assembled_pythonpath(root: Path) -> str:
    """Mirror `_add_runtime_srcs` but for a subprocess PYTHONPATH.

    The orchestrator's CLI module imports `renquant_execution`,
    `renquant_pipeline`, etc. at module load (transitively via
    `contract_fixture`). A bare ``-m renquant_orchestrator --help``
    with only the orchestrator src on PYTHONPATH dies during import
    before argparse runs — which would give us false-positive "missing
    subcommand" failures. Reuse the same ordered repo list so the
    subprocess sees what the cron sees.
    """
    ordered = (
        "renquant-common",
        "renquant-base-data",
        "renquant-artifacts",
        "renquant-model",
        "renquant-pipeline",
        "renquant-execution",
        "renquant-strategy-104",
        "renquant-backtesting",
        "renquant-orchestrator",
    )
    parts: list[str] = []
    for repo in ordered:
        src = _repo_src(root, repo)
        if src.is_dir():
            parts.append(str(src))
    return os.pathsep.join(parts)


def check_command(root: Path, command: RuntimeCommand) -> tuple[bool, str]:
    """Verify ``python -m <module> <subcommand>`` is registered.

    Implementation detail: spawns a real subprocess with the full
    multirepo PYTHONPATH (the orchestrator CLI imports
    `renquant_execution` at module load, so a bare orchestrator-only
    path dies during import). Treats import error and missing
    subcommand the same — both block the cron from invoking the
    bridge.
    """
    expected_src = _repo_src(root, command.repo)
    if not expected_src.is_dir():
        return False, (
            f"FAIL {command.module} {command.subcommand}: runtime source "
            f"missing at {expected_src} ({command.why})"
        )
    env = os.environ.copy()
    assembled = _assembled_pythonpath(root)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{assembled}{os.pathsep}{existing}" if existing else assembled
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-m", command.module, "--help"],
            capture_output=True,
            text=True,
            env=env,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"FAIL {command.module} --help timed out ({command.why})"
    except FileNotFoundError as exc:
        return False, f"FAIL {command.module} --help failed: {exc} ({command.why})"
    help_text = (proc.stdout or "") + (proc.stderr or "")
    # Import error on the package's module load is the same failure
    # mode the cron hit: cron sees argparse never gets a chance to
    # register the subparser, so absence in --help is the right signal.
    if command.subcommand not in help_text:
        snippet = help_text.strip().splitlines()[:6]
        snippet_text = " | ".join(snippet) if snippet else "(no output)"
        return False, (
            f"FAIL {command.module} subcommand {command.subcommand!r} "
            f"not in --help output: {snippet_text} ({command.why})"
        )
    return True, (
        f"ok   {command.module} {command.subcommand} "
        f"[registered under {expected_src}]"
    )


def check_symbol(root: Path, symbol: RuntimeSymbol) -> tuple[bool, str]:
    expected_src = _repo_src(root, symbol.repo)
    ok, origin_msg = _origin_under_runtime(symbol.module, expected_src)
    if not ok:
        return False, f"FAIL {symbol.module}: {origin_msg} ({symbol.why})"
    try:
        module = importlib.import_module(symbol.module)
    except Exception as exc:  # noqa: BLE001 - report exact import failure.
        return False, f"FAIL import {symbol.module}: {exc!r} ({symbol.why})"
    if not hasattr(module, symbol.attr):
        return False, f"FAIL {symbol.module}.{symbol.attr} missing ({symbol.why})"
    return True, f"ok   {symbol.module}.{symbol.attr} [{origin_msg}]"


def check_runtime(root: Path) -> list[str]:
    _add_runtime_srcs(root)
    failures: list[str] = []
    required_repos = sorted(
        {symbol.repo for symbol in REQUIRED_SYMBOLS}
        | {command.repo for command in REQUIRED_COMMANDS}
    )
    for repo in required_repos:
        src = _repo_src(root, repo)
        if not src.is_dir():
            message = f"FAIL runtime repo source missing: {src}"
            print(message)
            failures.append(message)
    if failures:
        return failures

    for symbol in REQUIRED_SYMBOLS:
        ok, message = check_symbol(root, symbol)
        print(message)
        if not ok:
            failures.append(message)
    for command in REQUIRED_COMMANDS:
        ok, message = check_command(root, command)
        print(message)
        if not ok:
            failures.append(message)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
        help="Override RENQUANT_SUBREPO_ROOT for tests or manual checks.",
    )
    args = parser.parse_args(argv)

    root = args.runtime_root.resolve() if args.runtime_root else resolve_subrepo_root(ROOT).resolve()
    print(f"runtime_qp_sanity_check: root={root}")
    failures = check_runtime(root)
    if failures:
        print()
        print("FAIL: stale or incomplete QP multirepo runtime.")
        print("Fix: merge required subrepo PRs, run `make subrepo-runtime-root`, then paper-smoke daily_104.")
        print("Runbook: doc/ops/subrepo-runtime-refresh-runbook.md")
        return 1
    print(
        f"OK: {len(REQUIRED_SYMBOLS)} runtime symbols "
        f"+ {len(REQUIRED_COMMANDS)} CLI subcommands present."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
