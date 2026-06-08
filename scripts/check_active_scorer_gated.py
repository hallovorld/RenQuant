#!/usr/bin/env python
"""P0b — assert the ACTIVE production scorer(s) carry passing wf_gate_metadata.

RFC #259 P0 ("unified gate→stamp→promote; no primary scorer reaches production
without wf_gate_metadata"). `promote()` enforces this at swap time and
`assert_artifact_gated()` (renquant-backtesting) is the reusable contract, but
the 2026-06-05 PatchTST promotion *bypassed* `promote()` with a direct
`strategy_config.json` edit. This umbrella ops check runs the same contract over
the ACTIVE configs so the bypass fails in CI / preflight — not silently in
production (where live `P-WF-GATE` then blocks all buys → the 2-week no-buy).

For each config it resolves `ranking.panel_scoring.artifact_path` (relative to
the strategy dir, per §7.6 canonical-key rule) and delegates to
`renquant_backtesting.forensics.model_acceptance.assert_artifact_gated`, which
handles JSON artifacts directly and `.pt` checkpoints via their
`<name>.metadata.json` sidecar.

Exit 0 = every checked scorer is gated.
Exit 1 = at least one ACTIVE production scorer is NOT gated (governance violation).
Exit 2 = config / artifact-resolution / import error (cannot determine state).

Multi-repo: must run with the subrepo PYTHONPATH (scripts/subrepo_env.sh) so the
`renquant_backtesting` guard is importable — the daily/weekly wrappers already
export it; CI sets it via the same helper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_DIR / "backtesting" / "renquant_104"

# The configs the daily-full path actually loads (§4.2): the primary live config
# (real paper orders) and the shadow config (readonly leg). Both feed the daily
# signal, so both must carry a gated scorer.
DEFAULT_CONFIGS = ("strategy_config.json", "strategy_config.shadow.json")


def _resolve_artifact(config_path: Path, strategy_dir: Path) -> tuple[Path, str]:
    """Return (resolved_artifact_path, kind) from a strategy config, or raise."""
    cfg = json.loads(config_path.read_text())
    ps = cfg.get("ranking", {}).get("panel_scoring", {})
    if not isinstance(ps, dict):
        raise ValueError(f"{config_path.name}: ranking.panel_scoring is not an object")
    rel = ps.get("artifact_path")
    if not rel:
        raise ValueError(f"{config_path.name}: no ranking.panel_scoring.artifact_path")
    artifact = (strategy_dir / rel).resolve()
    return artifact, str(ps.get("kind", "?"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="strategy config filename (under --strategy-dir) or path; "
        "repeatable. Default: the active primary + shadow configs.",
    )
    ap.add_argument("--strategy-dir", default=str(STRATEGY_DIR))
    args = ap.parse_args(argv)

    strategy_dir = Path(args.strategy_dir)
    config_names = args.configs or list(DEFAULT_CONFIGS)

    try:
        from renquant_backtesting.forensics.model_acceptance import assert_artifact_gated
    except ImportError as exc:  # subrepo PYTHONPATH not set up
        print(
            f"ERROR: cannot import assert_artifact_gated — run with the subrepo "
            f"PYTHONPATH (scripts/subrepo_env.sh). {exc}",
            file=sys.stderr,
        )
        return 2

    violations = 0
    errors = 0
    for name in config_names:
        cand = Path(name)
        config_path = cand if cand.is_absolute() or cand.exists() else strategy_dir / name
        if not config_path.exists():
            print(f"ERROR  {name}: config not found at {config_path}", file=sys.stderr)
            errors += 1
            continue
        try:
            artifact, kind = _resolve_artifact(config_path, strategy_dir)
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            print(f"ERROR  {name}: {exc}", file=sys.stderr)
            errors += 1
            continue
        try:
            wf = assert_artifact_gated(artifact)
        except ValueError as exc:
            print(
                f"VIOLATION  {config_path.name}: production scorer NOT gated "
                f"(kind={kind}) — {exc}",
                file=sys.stderr,
            )
            violations += 1
            continue
        run_at = (wf or {}).get("run_at", "?")
        print(
            f"OK  {config_path.name}: scorer gated "
            f"(kind={kind} artifact={artifact.name} passed={wf.get('passed')} run_at={run_at})"
        )

    if errors:
        return 2
    if violations:
        print(
            f"\n{violations} active production config(s) carry an UNGATED scorer — "
            f"this is the promotion-bypass governance violation (RFC #259 P0).",
            file=sys.stderr,
        )
        return 1
    print(f"\nAll {len(config_names)} active production scorer(s) gated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
