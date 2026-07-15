#!/usr/bin/env python3
"""Pipeline/execution parity check: detect drift in duplicated cross-repo
constants and contracts (G3 Phase A, architecture-audit registry items A2
and A3 — see ``doc/arch/2026-07-13-g3-refactoring-plan.md`` Phase A).

Originally implemented as orchestrator-local pytest tests
(renquant-orchestrator PR #515, ``tests/test_cross_repo_parity.py``). Codex
review flagged the same no-op-green pattern already fixed here for F-6
(umbrella PR #468): renquant-orchestrator can see renquant-pipeline and
renquant-execution as local sibling directories on a developer machine, but
orchestrator's own CI has no job that checks those siblings out, so a green
orchestrator build proved none of the actual pipeline/execution invariants.
This script — and the ONE CI job that runs it strictly,
``.github/workflows/pipeline-execution-parity-ci.yml`` — is the relocation:
this repo's ``subrepos.lock.json`` supplies the exact pipeline/execution
(and their own renquant-common/renquant-base-data/renquant-artifacts
dependency) pins to compare, and that workflow checks all five out as real
siblings.

Checks:
  A2 — ``MIN_FRACTIONAL_NOTIONAL_USD`` value equality between
       ``renquant_pipeline.kernel.sizing`` and ``renquant_execution.broker``,
       plus ``compute_parent_intent_id`` golden-vector parity (3 vectors +
       signature match) between ``renquant_pipeline.intraday_decisioning``
       and ``renquant_execution.order_state_machine``.
  A3 — Calendar-import inventory: count of non-canonical
       ``pandas_market_calendars`` imports in the pipeline kernel (should use
       ``renquant_common.market_calendar``), baselined at 7 (as of
       2026-07-14, matching the count found in renquant-orchestrator#515).

       This is a TEMPORARY DEBT INVENTORY, not a compliance proof: it only
       blocks the count from GROWING past 7; it does not establish that the
       7 existing non-canonical imports are compliant, and it does not
       verify pipeline/execution behavior is integrated end-to-end. The 7
       sites are open debt tracked by G3 Phase B task B2 ("Calendar ->
       adopt renquant_common.market_calendar at all 7 sites",
       doc/arch/2026-07-13-g3-refactoring-plan.md) and by owning issue
       https://github.com/hallovorld/RenQuant/issues/475.

       Retirement condition: once all 7 sites are migrated to the canonical
       import and the count reaches 0, lower CALENDAR_BASELINE to 0 (making
       this a hard zero-tolerance gate) and close issue #475.

Exit codes:
  0 — all checks passed (or the calendar inventory is within baseline)
  1 — drift or a mismatch was detected
  2 — setup error (bad lock file, unexpected import error, missing kernel
      dir, etc.)
  3 — skipped: one or more sibling repos not found/importable. Distinct
      from 0 so callers (``tests/test_pipeline_execution_parity.py``) can
      tell "ran and passed" apart from "never actually compared anything"
      instead of a skip silently reading as a pass.
"""
from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType

UMBRELLA_ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = UMBRELLA_ROOT / "subrepos.lock.json"

# key -> (env var override, subrepos.lock.json name). Order matters only for
# readability; renquant-pipeline transitively needs base-data + artifacts,
# renquant-execution needs only common (see each repo's pyproject.toml).
_SIBLINGS: dict[str, tuple[str, str]] = {
    "common": ("RENQUANT_COMMON_SRC_PATH", "renquant-common"),
    "base_data": ("RENQUANT_BASE_DATA_SRC_PATH", "renquant-base-data"),
    "artifacts": ("RENQUANT_ARTIFACTS_SRC_PATH", "renquant-artifacts"),
    "pipeline": ("RENQUANT_PIPELINE_SRC_PATH", "renquant-pipeline"),
    "execution": ("RENQUANT_EXECUTION_SRC_PATH", "renquant-execution"),
}

CANONICAL_CALENDAR_IMPORT = "renquant_common.market_calendar"
# TEMPORARY debt-inventory ceiling, not a compliance proof (see module
# docstring's A3 section). Owning issue:
# https://github.com/hallovorld/RenQuant/issues/475 . Retirement condition:
# once G3 Phase B task B2 migrates all 7 non-canonical call sites to
# CANONICAL_CALENDAR_IMPORT, lower this to 0 and close #475.
CALENDAR_BASELINE = 7


def _resolve_sibling_src(env_var: str, lock_name: str) -> Path | None:
    """Resolve one sibling repo's ``src`` dir.

    Resolution order mirrors ``check_kernel_parity.py``'s
    ``_resolve_pipeline_kernel``:

    1. explicit env var override — CI
       (``.github/workflows/pipeline-execution-parity-ci.yml``) sets this to
       wherever it checked the sibling out, since a CI runner's checkout
       layout has nothing to do with any developer machine's
       ``subrepos.lock.json`` ``local_path``.
    2. ``subrepos.lock.json``'s ``local_path`` — the developer-machine pin.
    3. a checkout that is a filesystem sibling of this umbrella checkout
       (``../<repo-name>``) — the layout produced by a from-scratch
       multirepo clone.
    """
    import os

    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        return p if p.is_dir() else None

    if LOCK_FILE.exists():
        with open(LOCK_FILE) as fh:
            lock = json.load(fh)
        for sub in lock.get("subrepos", []):
            if sub.get("name") == lock_name:
                src = Path(sub["local_path"]) / "src"
                if src.is_dir():
                    return src

    sibling = UMBRELLA_ROOT.parent / lock_name / "src"
    return sibling if sibling.is_dir() else None


def _ensure_siblings_on_path() -> list[str]:
    """Add every sibling src dir to ``sys.path``. Return missing lock names."""
    missing: list[str] = []
    for env_var, lock_name in _SIBLINGS.values():
        src = _resolve_sibling_src(env_var, lock_name)
        if src is None:
            missing.append(lock_name)
            continue
        s = str(src)
        if s not in sys.path:
            sys.path.insert(0, s)
    return missing


def _import(module_path: str) -> ModuleType:
    return importlib.import_module(module_path)


GOLDEN_VECTORS: list[dict[str, str]] = [
    dict(
        account="DU12345",
        symbol="AAPL",
        trading_day="2026-01-15",
        side="buy",
        signal_version="v1",
    ),
    dict(
        account="DU12345",
        symbol="MSFT",
        trading_day="2026-03-01",
        side="sell",
        signal_version="v2",
    ),
    dict(
        account="LIVE99",
        symbol="NVDA",
        trading_day="2026-07-01",
        side="buy",
        signal_version="v3.1",
    ),
]


def check_min_fractional_notional_parity() -> tuple[bool, str]:
    p_sizing = _import("renquant_pipeline.kernel.sizing")
    e_broker = _import("renquant_execution.broker")

    p_val = p_sizing.MIN_FRACTIONAL_NOTIONAL_USD
    e_val = e_broker.MIN_FRACTIONAL_NOTIONAL_USD

    ok = (
        p_val == e_val
        and isinstance(p_val, (int, float))
        and isinstance(e_val, (int, float))
    )
    detail = f"pipeline={p_val!r} execution={e_val!r}"
    return ok, detail


def check_compute_parent_intent_id_parity() -> tuple[bool, list[str]]:
    p_mod = _import("renquant_pipeline.intraday_decisioning")
    e_mod = _import("renquant_execution.order_state_machine")

    problems: list[str] = []

    p_params = list(inspect.signature(p_mod.compute_parent_intent_id).parameters)
    e_params = list(inspect.signature(e_mod.compute_parent_intent_id).parameters)
    if p_params != e_params:
        problems.append(
            f"parameter names differ: pipeline={p_params} execution={e_params}"
        )

    for kw in GOLDEN_VECTORS:
        p_out = p_mod.compute_parent_intent_id(**kw)
        e_out = e_mod.compute_parent_intent_id(**kw)
        if p_out != e_out:
            problems.append(
                f"compute_parent_intent_id({kw}) drifted: "
                f"pipeline={p_out} execution={e_out}"
            )

    return (len(problems) == 0), problems


def _count_raw_calendar_imports(pipeline_kernel_dir: Path) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for py in sorted(pipeline_kernel_dir.rglob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "pandas_market_calendars" in stripped and "import" in stripped:
                rel = py.relative_to(pipeline_kernel_dir)
                hits.append((str(rel), lineno))
    return hits


def check_calendar_import_inventory() -> tuple[bool, list[tuple[str, int]], str]:
    env_var, lock_name = _SIBLINGS["pipeline"]
    pipeline_src = _resolve_sibling_src(env_var, lock_name)
    if pipeline_src is None:
        raise RuntimeError("pipeline sibling not resolved (should be caught earlier)")

    kernel_dir = pipeline_src / "renquant_pipeline" / "kernel"
    if not kernel_dir.is_dir():
        raise RuntimeError(f"pipeline kernel dir not found at {kernel_dir}")

    hits = _count_raw_calendar_imports(kernel_dir)
    ok = len(hits) <= CALENDAR_BASELINE
    detail = (
        f"{len(hits)} non-canonical pandas_market_calendars import(s) "
        f"(TEMPORARY debt-inventory ceiling {CALENDAR_BASELINE}, not a "
        f"compliance proof, tracked by issue #475; should use "
        f"{CANONICAL_CALENDAR_IMPORT})"
    )
    return ok, hits, detail


def check_parity(*, verbose: bool = False) -> tuple[int, dict]:
    """Return (exit_code, payload). See module docstring for exit codes."""
    missing = _ensure_siblings_on_path()
    if missing:
        return 3, {
            "skipped": True,
            "reason": f"sibling repo(s) not found: {', '.join(sorted(missing))}",
        }

    results: dict[str, dict] = {}
    try:
        ok, detail = check_min_fractional_notional_parity()
        results["MIN_FRACTIONAL_NOTIONAL_USD"] = {"ok": ok, "detail": detail}

        ok, problems = check_compute_parent_intent_id_parity()
        results["compute_parent_intent_id"] = {"ok": ok, "problems": problems}

        ok, hits, detail = check_calendar_import_inventory()
        results["calendar_import_inventory"] = {
            "ok": ok,
            "detail": detail,
            "hits": [f"{p}:{n}" for p, n in hits],
        }
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        return 2, {"skipped": False, "error": str(exc)}

    all_ok = all(r["ok"] for r in results.values())

    if verbose:
        for name, r in results.items():
            status = "OK" if r["ok"] else "FAIL"
            extra = r.get("detail") or r.get("problems")
            print(f"{name}: {status} — {extra}")

    return (0 if all_ok else 1), {"skipped": False, "results": results}


def _write_json_out(path: str, code: int, payload: dict) -> None:
    """Dump ``{"exit_code": ..., **payload}`` to ``path`` for CI run records.

    Used by ``.github/workflows/pipeline-execution-parity-ci.yml`` to build a
    machine-readable, per-pin integration run record (umbrella SHA + all 5
    pin SHAs + dependency-environment digest + these check results) rather
    than only a human-readable pytest log.
    """
    record = {"exit_code": code, **payload}
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    json_out = None
    if "--json-out" in sys.argv:
        json_out = sys.argv[sys.argv.index("--json-out") + 1]
    code, payload = check_parity(verbose=verbose)

    if json_out:
        _write_json_out(json_out, code, payload)

    if payload.get("skipped"):
        print(f"SKIP: {payload['reason']}")
        return code

    if code == 2:
        print(f"SETUP ERROR: {payload.get('error')}")
        return code

    if code == 1:
        print("FAIL: pipeline/execution parity drift detected:")
        for name, r in payload["results"].items():
            if not r["ok"]:
                print(f"  {name}: {r.get('detail') or r.get('problems')}")
        return code

    print("PASS: pipeline/execution parity checks all OK "
          f"({len(payload['results'])} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
