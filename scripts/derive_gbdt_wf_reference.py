#!/usr/bin/env python
"""Derive the GBDT (xgb) production reference from a pinned blend's component[0].

orch#799 option A. After the full-book z-blend switch, the pinned primary
``strategy_config.json`` declares ``ranking.panel_scoring.kind = "blend"`` and the
xgb GBDT leg lives at ``ranking.panel_scoring.components[0]``. The weekly WF
promote gate needs a ``kind=xgb`` PRODUCTION REFERENCE config so it can

  1. stamp the WF manifest config fingerprint (Step 3.5, ``--fingerprint-config``);
  2. derive the WF eval config + run config parity (Step 4,
     ``--derive-config-from-prod``) — where ``run_wf_gate._prod_config_path``
     FAILS CLOSED on any reference whose ``panel_scoring.kind`` does not match the
     xgb candidate kind (a ``kind=blend`` reference is rejected).

This tool materializes a ``kind=xgb`` VIEW of the SAME pinned recipe: it flips
``panel_scoring.kind`` blend -> xgb, points ``artifact_path`` at the component[0]
GBDT leg, and drops the blend-only ``components``. The model-relevant config
fingerprint is INVARIANT to ``panel_scoring.kind`` / ``components``
(``kernel.config_consistency._model_relevant_fields`` reads only
``panel_ltr`` / ``watchlist`` / ``sector`` fields), so the freshly-retrained xgb
candidate is compared on the SAME walk-forward manifest recipe discipline the
gate already enforces — no gate is weakened, and a recipe/fingerprint mismatch
still fails closed downstream.

HARD constraint (the orch#799 incident): the ONLY reference source is the PINNED
runtime config passed as ``--pinned-config`` (``.subrepo_runtime/...``). This tool
never reads the umbrella working copy (``backtesting/renquant_104/*``) or a
sibling developer checkout — those banned sources are what let the gate simulate
a strategy nobody runs (Sharpe 0.6018 -> 0.0524). If component[0] is not the xgb
GBDT leg, or its referenced leg artifact is absent, this tool FAILS CLOSED
(non-zero exit, no derived config written) rather than paper over a phantom.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

_GBDT_KIND = "xgb"
# Mirror run_wf_gate._normalize_scorer_kind's GBDT vocabulary so the derived
# reference's kind matches whatever the candidate artifact declares.
_GBDT_KIND_ALIASES = {"xgb", "panel_ltr_xgboost", "xgboost"}


class DeriveError(RuntimeError):
    """Raised when a safe xgb reference cannot be derived — always fail closed."""


def _normalize_kind(kind: Any) -> str:
    value = str(kind or "").strip().lower()
    return _GBDT_KIND if value in _GBDT_KIND_ALIASES else value


def _is_panel_ltr_gbdt_leg(artifact_path: Any) -> bool:
    """The blend contract's GBDT leg artifact is ``panel-ltr*.json``."""
    if not (isinstance(artifact_path, str) and artifact_path):
        return False
    base = os.path.basename(artifact_path)
    return base.startswith("panel-ltr") and base.endswith(".json")


def _load_fingerprint_config(strategy_dir: Path):
    """Load ``fingerprint_config`` from the strategy kernel BY FILE PATH.

    Loading the file directly (not importing the ``kernel`` package) keeps this a
    dependency-free, standalone check: ``config_consistency`` imports only stdlib.
    Returns ``None`` if the file is unavailable (best-effort invariance guard).
    """
    cc = Path(strategy_dir) / "kernel" / "config_consistency.py"
    if not cc.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_orch799_config_consistency", cc)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, "fingerprint_config", None)


def derive_reference(
    pinned_config_path: str | Path,
    strategy_dir: str | Path,
    out_path: str | Path,
) -> dict[str, Any]:
    """Derive a ``kind=xgb`` reference from the pinned blend's component[0].

    Returns a summary dict on success; raises ``DeriveError`` (fail closed) when
    component[0] is not the xgb GBDT leg or its artifact is absent.
    """
    pinned_config_path = Path(pinned_config_path)
    strategy_dir = Path(strategy_dir)
    out_path = Path(out_path)

    if not pinned_config_path.is_file():
        raise DeriveError(f"pinned config not found: {pinned_config_path}")

    config = json.loads(pinned_config_path.read_text())
    panel = ((config.get("ranking") or {}).get("panel_scoring") or {})
    top_kind = _normalize_kind(panel.get("kind"))

    if top_kind == _GBDT_KIND:
        raise DeriveError(
            "pinned primary panel_scoring.kind already normalizes to xgb; the "
            "top-level kind scan should have resolved it — no component[0] "
            "derivation is needed."
        )
    if top_kind != "blend":
        raise DeriveError(
            f"pinned primary panel_scoring.kind={panel.get('kind')!r} is neither "
            "xgb (handled by the top-level scan) nor blend; there is no blend "
            "component[0] xgb leg to derive."
        )

    components = panel.get("components")
    if not isinstance(components, list) or not components:
        raise DeriveError(
            "pinned blend panel_scoring has no components[]; cannot derive an "
            "xgb reference from component[0]."
        )
    comp0 = components[0]
    if not isinstance(comp0, dict):
        raise DeriveError("blend components[0] is not a JSON object.")

    comp0_kind = _normalize_kind(comp0.get("kind"))
    comp0_artifact = comp0.get("artifact_path")

    # component[0] must be the xgb GBDT leg: an explicit xgb kind, OR (kind
    # absent AND artifact_path is the panel-ltr GBDT leg per the blend contract,
    # where component[0]._role == "PRODUCTION panel scorer (rank:pairwise xgb)").
    if comp0_kind == _GBDT_KIND:
        reason = "component[0].kind normalizes to xgb"
    elif comp0_kind == "" and _is_panel_ltr_gbdt_leg(comp0_artifact):
        reason = (
            "component[0] artifact_path is the panel-ltr GBDT leg "
            "(kind absent, per the blend contract)"
        )
    else:
        raise DeriveError(
            "blend component[0] is not the xgb GBDT leg (kind="
            f"{comp0.get('kind')!r}, artifact_path={comp0_artifact!r}); refusing "
            "to derive a phantom xgb reference. orch#799 decision: gate a blend "
            "prod on a blend-kind candidate instead."
        )

    if not (isinstance(comp0_artifact, str) and comp0_artifact):
        raise DeriveError("blend component[0] has no artifact_path; fail closed.")

    # The referenced GBDT leg artifact must exist on disk — a WF-comparable
    # reference must point at a real leg artifact, never a phantom. Resolve the
    # same way the gate resolves panel_scoring.artifact_path: relative to the
    # strategy dir.
    leg_artifact = Path(comp0_artifact)
    if not leg_artifact.is_absolute():
        leg_artifact = strategy_dir / comp0_artifact
    if not leg_artifact.is_file():
        raise DeriveError(
            f"blend component[0] GBDT leg artifact absent at {leg_artifact}; "
            "cannot form a WF-comparable reference. Fail closed."
        )

    # Synthesize the kind=xgb view of the SAME recipe.
    derived = copy.deepcopy(config)
    derived_panel = derived["ranking"]["panel_scoring"]
    derived_panel["kind"] = _GBDT_KIND
    derived_panel["artifact_path"] = comp0_artifact
    derived_panel.pop("components", None)

    # Defense-in-depth: prove the derivation did not shift the model-relevant
    # config fingerprint (it only flipped kind + dropped components, both
    # fingerprint-invariant). A shift would mean we compare the candidate
    # against a recipe-changed reference — fail closed.
    fingerprint_config = _load_fingerprint_config(strategy_dir)
    invariance_verified = False
    if fingerprint_config is not None:
        fp_before = fingerprint_config(config)
        fp_after = fingerprint_config(derived)
        if fp_before != fp_after:
            raise DeriveError(
                "derivation changed the model-relevant config fingerprint "
                f"({fp_before} -> {fp_after}); refusing to compare the candidate "
                "against a recipe-shifted reference. Fail closed."
            )
        invariance_verified = True

    derived_panel["_derived_gbdt_reference"] = {
        "source": "orch#799 option A — derived from blend component[0]",
        "pinned_config": str(pinned_config_path),
        "component_index": 0,
        "component_artifact_path": comp0_artifact,
        "leg_artifact": str(leg_artifact),
        "reason": reason,
        "config_fingerprint_invariance_verified": invariance_verified,
        "note": (
            "kind flipped blend->xgb and components[] dropped; the config "
            "fingerprint is invariant to these (model-relevant fields are "
            "panel_ltr/watchlist/sector only), so the freshly-retrained xgb "
            "candidate is compared on the same WF manifest recipe. The umbrella "
            "working copy and sibling checkout were NOT consulted (orch#799)."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(derived, indent=2, sort_keys=False) + "\n")
    return {
        "derived_path": str(out_path),
        "pinned_config": str(pinned_config_path),
        "component_artifact_path": comp0_artifact,
        "leg_artifact": str(leg_artifact),
        "reason": reason,
        "config_fingerprint_invariance_verified": invariance_verified,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pinned-config",
        required=True,
        help=(
            "PINNED runtime strategy_config.json (.subrepo_runtime/...). The ONLY "
            "reference source; the umbrella working copy and sibling checkout are "
            "never consulted (orch#799)."
        ),
    )
    ap.add_argument(
        "--strategy-dir",
        required=True,
        help=(
            "backtesting/renquant_104 — used only for leg-artifact existence and "
            "the config-fingerprint invariance check, NOT as a config reference "
            "source."
        ),
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Derived reference output path (a scratch/logs path, never a prod config).",
    )
    args = ap.parse_args(argv)

    try:
        result = derive_reference(args.pinned_config, args.strategy_dir, args.out)
    except DeriveError as exc:
        print(f"derive_gbdt_wf_reference: FAIL CLOSED — {exc}", file=sys.stderr)
        return 2
    print(
        "derive_gbdt_wf_reference: derived kind=xgb reference from blend "
        f"component[0] ({result['reason']}; "
        f"fingerprint_invariance={result['config_fingerprint_invariance_verified']})",
        file=sys.stderr,
    )
    # stdout carries ONLY the derived path so the shell can capture it.
    print(result["derived_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
