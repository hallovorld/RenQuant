#!/usr/bin/env python3
"""Runtime-authoritative scorer/calibrator BINDING check (defense-in-depth).

2026-07-01 incident: `monthly_calibrator_refresh.sh` fits a new calibrator via
`fit_calibrator_alpha158_fund.py`, then only validated it with the pool_ic /
n_unique_prob_y quality regression gate (`smoke_test_model.py` +
Step 3's inline gate). Neither checks whether the newly-fit calibrator's
stamped `scorer_model_content_fingerprint` actually matches what the LIVE
RUNTIME will compute for the active scorer — the exact contract
`_assert_calibrator_matches_scorer` enforces in
`renquant_pipeline/kernel/panel_pipeline/job_panel_scoring.py`. Root cause:
`fit_calibrator_alpha158_fund.py`'s `model_content_sha256` (renquant-model,
an explicit 11-field allowlist) and `panel_scorer.py`'s `model_content_sha256`
(renquant-pipeline, the runtime-authoritative denylist) were independently
hand-copied and hashed DIFFERENT field sets — a calibrator fit by one could
never match the runtime check by construction. So a calibrator that will
fail-closed the live daily-full at runtime could still pass the monthly gate
and get silently published.

That specific divergence is being fixed at the source via
renquant-common#18 (canonical `model_content_sha256`) +
renquant-pipeline#155 + renquant-model#40 (both consumers import the shared
function instead of hand-copying it). This module is the additional,
defense-in-depth check: it exercises the SAME runtime-authoritative loader
(`PanelScorer.load`) and match logic (`_any_fingerprints_match` /
`_fingerprint_values`, imported — not reimplemented — from
`renquant_pipeline.kernel.panel_pipeline.job_panel_scoring`) so ANY future
re-divergence, or any other cause of a scorer/calibrator mismatch, is caught
here before publish, not after it blocks live trading. It keeps working
correctly regardless of which of the three PRs above lands first, or in what
order — `_any_fingerprints_match` / `_fingerprint_values` are untouched by
that unification (it only changes HOW the fingerprint is computed, not how
it's matched).

FAILS CLOSED: if the runtime-authoritative loader is not importable in this
script's Python environment, that is treated as a gate FAILURE (exit 2), not
a silent skip — a check that exists-but-skips-silently is exactly the
failure mode that let the 2026-07-01 incident through.

Usage::

    scripts/verify_calibrator_scorer_binding.py \\
        --scorer "$PROD_SCORER" --calibrator "$PROD_CAL" [--json]

Exit codes:
    0 = binding OK (calibrator fingerprint matches the runtime-computed
        active scorer fingerprint)
    1 = binding MISMATCH (gate failure — caller should roll back)
    2 = could not evaluate (missing artifact, load error, or the
        runtime-authoritative loader was not importable — fail CLOSED,
        caller should roll back)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


def load_runtime_authorities() -> tuple[Any, Callable, Callable]:
    """Import the runtime-authoritative scorer loader + fingerprint helpers.

    Always imports `PanelScorer`, `_fingerprint_values`, and
    `_any_fingerprints_match` from `renquant_pipeline.kernel.panel_pipeline`
    — that IS the runtime-authoritative path regardless of whether
    renquant-common#18 / renquant-pipeline#155 have landed yet (those PRs
    change how `model_content_sha256` is computed inside `PanelScorer.load`,
    not the public loader entry point or the match/prefix-match semantics
    used here).

    Raises ImportError (or any other exception) if unavailable — callers
    MUST treat that as a gate failure, never a silent skip.
    """
    from renquant_pipeline.kernel.panel_pipeline.panel_scorer import (  # noqa: PLC0415
        PanelScorer,
    )
    from renquant_pipeline.kernel.panel_pipeline.job_panel_scoring import (  # noqa: PLC0415
        _any_fingerprints_match,
        _fingerprint_values,
    )
    return PanelScorer, _fingerprint_values, _any_fingerprints_match


def check_binding(
    scorer_path: Path,
    calibrator_path: Path,
    *,
    panel_scorer_cls: Any | None = None,
    fingerprint_values: Callable[[dict | None], list[str]] | None = None,
    any_fingerprints_match: Callable[[list[str], list[str]], bool] | None = None,
) -> dict[str, Any]:
    """Check whether `calibrator_path`'s stamped fingerprint(s) match the
    fingerprint(s) the live runtime will compute for `scorer_path`.

    Mirrors `_assert_calibrator_matches_scorer` in
    `job_panel_scoring.py` exactly — same helper functions when importable
    (not a reimplementation), so this gate and the runtime check can never
    independently drift the way the fit-time/runtime `model_content_sha256`
    copies did.

    The three `panel_scorer_cls` / `fingerprint_values` /
    `any_fingerprints_match` keyword args exist ONLY for unit tests (fixture
    injection without the strategy venv). Real callers must leave them None
    so the actual runtime-authoritative implementations run.

    Returns a dict with at least ``status`` (``"pass"``, ``"fail"``, or
    ``"error"``), ``match`` (bool), and ``reason`` (str).
    """
    result: dict[str, Any] = {
        "scorer_path": str(scorer_path),
        "calibrator_path": str(calibrator_path),
    }

    need_import = (
        panel_scorer_cls is None
        or fingerprint_values is None
        or any_fingerprints_match is None
    )
    if need_import:
        try:
            _cls, _fv, _afm = load_runtime_authorities()
        except Exception as exc:  # noqa: BLE001 — any import failure fails closed
            result.update(
                status="error",
                match=False,
                reason=(
                    "runtime-authoritative loader "
                    "(renquant_pipeline.kernel.panel_pipeline) not importable "
                    f"— failing CLOSED, not skipping the check "
                    f"({type(exc).__name__}: {exc})"
                ),
            )
            return result
        panel_scorer_cls = panel_scorer_cls if panel_scorer_cls is not None else _cls
        fingerprint_values = (
            fingerprint_values if fingerprint_values is not None else _fv
        )
        any_fingerprints_match = (
            any_fingerprints_match if any_fingerprints_match is not None else _afm
        )

    if not scorer_path.exists():
        result.update(
            status="error", match=False,
            reason=f"active scorer artifact not found: {scorer_path}",
        )
        return result
    if not calibrator_path.exists():
        result.update(
            status="error", match=False,
            reason=f"calibrator artifact not found: {calibrator_path}",
        )
        return result

    try:
        scorer = panel_scorer_cls.load(scorer_path)
    except Exception as exc:  # noqa: BLE001
        result.update(
            status="error", match=False,
            reason=f"active scorer failed to load ({type(exc).__name__}): {exc}",
        )
        return result

    active_fps = fingerprint_values(getattr(scorer, "metadata", {}) or {})

    try:
        cal_payload = json.loads(calibrator_path.read_text())
    except Exception as exc:  # noqa: BLE001
        result.update(
            status="error", match=False,
            reason=f"calibrator artifact unreadable ({type(exc).__name__}): {exc}",
        )
        return result
    cal_meta = cal_payload.get("metadata", {}) if isinstance(cal_payload, dict) else {}
    cal_fps = fingerprint_values(cal_meta or {})

    result["active_fingerprints"] = active_fps
    result["calibrator_fingerprints"] = cal_fps

    if not active_fps or not cal_fps:
        result.update(
            status="fail",
            match=False,
            reason=(
                "missing scorer/calibrator fingerprint — "
                f"active={active_fps!r} calibrator={cal_fps!r}. Refit the "
                "calibrator with scorer_model_content_fingerprint stamped."
            ),
        )
        return result

    matched = bool(any_fingerprints_match(cal_fps, active_fps))
    result.update(
        status="pass" if matched else "fail",
        match=matched,
        reason=(
            "calibrator fingerprint matches the runtime-computed active "
            "scorer fingerprint"
            if matched else
            "calibrator/scorer BINDING MISMATCH — this calibrator was fit "
            "against a different scorer than the live runtime will load "
            f"for this artifact; calibrator={cal_fps} active_scorer={active_fps}"
        ),
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scorer", required=True,
                    help="path to the active production scorer artifact "
                         "(the same PROD_SCORER path monthly_calibrator_refresh.sh resolves)")
    ap.add_argument("--calibrator", required=True,
                    help="path to the newly-fit calibrator artifact (PROD_CAL)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    res = check_binding(Path(args.scorer), Path(args.calibrator))
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"CALIBRATOR-SCORER-BINDING status={res['status']} match={res['match']}")
        print(f"  {res.get('reason', '')}")
        if "active_fingerprints" in res:
            print(f"  active_fingerprints={res['active_fingerprints']}")
            print(f"  calibrator_fingerprints={res['calibrator_fingerprints']}")

    if res["status"] == "pass":
        return 0
    if res["status"] == "error":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
