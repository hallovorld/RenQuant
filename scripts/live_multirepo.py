#!/usr/bin/env python
"""Multi-repo live runner bridge.

Runs the real ``live.runner`` while routing lifted RenQuant modules through
``subrepos.lock.json`` local paths. This is the shared entry point for
production daily, intraday sell-only, and readonly shadow runs; set
``RQ_DAILY_RUNNER=umbrella`` in the shell wrapper to bypass this bridge and use
the umbrella baseline. Set ``RENQUANT_STRICT_SUBREPO_PATHS=1`` to fail closed
when local sibling checkouts do not match ``subrepos.lock.json``.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from subrepo_pin_guard import enforce_or_warn, resolve_subrepo_src_roots
from subrepo_pin_guard import strict_clean_enabled
from subrepo_paths import resolve_subrepo_root

REPO = Path(__file__).resolve().parent.parent
SIBLINGS = REPO.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
LOCK_FILE = REPO / "subrepos.lock.json"

# Pinned subrepo source roots. Names resolve through subrepos.lock.json
# local_path entries, RENQUANT_SUBREPO_ROOT, then SIBLINGS/name fallback.
_PIN_SRCS = [
    "renquant-common",
    "renquant-base-data",
    "renquant-artifacts",
    "renquant-strategy-104",
    "renquant-model",
    "renquant-pipeline",
    "renquant-execution",
    "renquant-backtesting",
]


def _arg_value(argv: list[str], flag: str, default: str | None = None) -> str | None:
    prefix = flag + "="
    for idx, arg in enumerate(argv):
        if arg == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def _without_arg(argv: list[str], flag: str) -> list[str]:
    out: list[str] = []
    skip = False
    prefix = flag + "="
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == flag:
            skip = True
            continue
        if arg.startswith(prefix):
            continue
        out.append(arg)
    return out


def _strategy_config_name(argv: list[str]) -> str:
    explicit = _arg_value(argv, "--strategy-config-name")
    if explicit:
        return explicit
    strategy = _arg_value(argv, "--strategy", "renquant_104")
    broker = _arg_value(argv, "--broker", "paper")
    if strategy == "renquant_104" and broker == "readonly-alpaca":
        return "strategy_config.shadow.json"
    return "strategy_config.json"


def _with_pinned_strategy_config(argv: list[str]) -> list[str]:
    """Route renquant_104 config reads to the pinned strategy subrepo.

    The runtime strategy_dir remains the umbrella checkout so live_state,
    artifacts, and data stay in the existing production location.
    """
    if _arg_value(argv, "--strategy-config-path"):
        return argv
    if _arg_value(argv, "--strategy", "renquant_104") != "renquant_104":
        return argv
    config_name = _strategy_config_name(argv)
    cfg_path = (
        resolve_subrepo_root(REPO)
        / "renquant-strategy-104"
        / "configs"
        / config_name
    )
    return _without_arg(argv, "--strategy-config-name") + [
        "--strategy-config-path",
        str(cfg_path),
    ]


def _subrepo_src_roots() -> tuple[list[Path], list[str]]:
    roots, issues = resolve_subrepo_src_roots(
        lock_file=LOCK_FILE,
        names=_PIN_SRCS,
        siblings=SIBLINGS,
        root_override=str(resolve_subrepo_root(REPO)),
        check_dirty=strict_clean_enabled(),
    )
    missing = [issue.repo for issue in issues if issue.reason == "missing local src root"]
    return roots, missing


def _force_alias(alias: str, target: str, aliased: list[str]) -> None:
    try:
        mod = importlib.import_module(target)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"critical multirepo module unavailable: {target}") from exc
    sys.modules[alias] = mod
    aliased.append(f"{alias}<-{target}")


def _bootstrap_multirepo() -> list[str]:
    """Put sibling subrepos on sys.path and alias lifted kernel modules."""
    for path in (str(REPO), str(STRATEGY_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    src_roots, pin_issues = resolve_subrepo_src_roots(
        lock_file=LOCK_FILE,
        names=_PIN_SRCS,
        siblings=SIBLINGS,
        root_override=str(resolve_subrepo_root(REPO)),
        check_dirty=strict_clean_enabled(),
    )
    enforce_or_warn(pin_issues)
    for src in src_roots:
        if str(src) not in sys.path:
            sys.path.append(str(src))

    pk = importlib.import_module("renquant_pipeline.kernel")
    pk_dir = Path(pk.__file__).resolve().parent

    aliased: list[str] = []
    for entry in sorted(pk_dir.iterdir()):
        stem = entry.stem if entry.suffix == ".py" else entry.name
        if stem in {"__init__", "__pycache__"} or stem.startswith("."):
            continue
        if entry.suffix not in {".py", ""}:
            continue
        modname = f"kernel.{stem}"
        try:
            mod = importlib.import_module(f"renquant_pipeline.kernel.{stem}")
        except Exception:
            continue
        sys.modules[modname] = mod
        aliased.append(modname)

    # Critical production modules must not silently fall back to umbrella. If
    # one of these imports fails, the multirepo runner is not actually running
    # the pinned production path and should fail closed.
    _force_alias("kernel.preflight", "renquant_pipeline.kernel.preflight", aliased)
    _force_alias("kernel.panel_pipeline", "renquant_pipeline.kernel.panel_pipeline", aliased)
    _force_alias(
        "renquant_pipeline.kernel.meta_label",
        "renquant_backtesting.meta_label",
        aliased,
    )
    # Keep the proven fail-closed panel scoring path from the lifted
    # renquant-pipeline kernel package until the load_scorer rewrite passes
    # production parity.
    _force_alias(
        "renquant_pipeline.panel_scoring",
        "renquant_pipeline.kernel.panel_pipeline.job_panel_scoring",
        aliased,
    )
    return aliased


def main() -> int:
    aliased = _bootstrap_multirepo()
    sys.stderr.write(
        f"[multirepo] routed {len(aliased)} lifted modules through sibling subrepos; "
        "live.runner remains the execution handoff.\n"
    )
    if _arg_value(sys.argv[1:], "--strategy") is None:
        sys.argv = [sys.argv[0], "--strategy", "renquant_104"] + sys.argv[1:]
    sys.argv = [sys.argv[0]] + _with_pinned_strategy_config(sys.argv[1:])
    runner = importlib.import_module("live.runner")
    return int(runner.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
