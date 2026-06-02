"""Small helper for umbrella scripts that delegate to pinned subrepo modules."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from subrepo_paths import resolve_subrepo_root


def _has_repo_root_arg(argv: list[str]) -> bool:
    return "--repo-root" in argv or any(arg.startswith("--repo-root=") for arg in argv)


def delegate_to_subrepo_module(
    module_name: str,
    argv: list[str],
    *,
    repo_root: Path,
    packages: tuple[str, ...],
    runner_env: str,
    strict_env: str,
) -> int | None:
    """Run a pinned subrepo module, or return None to use umbrella fallback."""
    if os.environ.get(runner_env, "multirepo") != "multirepo":
        return None

    try:
        subrepo_root = resolve_subrepo_root(repo_root)
        for package in reversed(packages):
            src = subrepo_root / package / "src"
            if not src.exists():
                raise FileNotFoundError(f"subrepo source missing: {src}")
            sys.path.insert(0, str(src))

        os.environ.setdefault("RENQUANT_REPO_ROOT", str(repo_root))
        forwarded = list(argv)
        if not _has_repo_root_arg(forwarded):
            forwarded.extend(["--repo-root", str(repo_root)])

        module = importlib.import_module(module_name)
        old_argv = sys.argv[:]
        sys.argv = [old_argv[0], *forwarded]
        try:
            return int(module.main() or 0)
        finally:
            sys.argv = old_argv
    except Exception as exc:  # noqa: BLE001
        if os.environ.get(strict_env) == "1":
            raise
        print(
            f"[multirepo fallback] {module_name} unavailable ({type(exc).__name__}: {exc}); "
            "using umbrella implementation. Set "
            f"{strict_env}=1 to fail closed.",
            file=sys.stderr,
        )
        return None

