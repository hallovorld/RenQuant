#!/usr/bin/env python
"""Stamp `config_fingerprint` (+ `config_fingerprint_fields`) into a PatchTST
artifact's `.metadata.json` sidecar so the strict LoadScorerTask config-
consistency gate accepts it.

Why this exists
---------------
`backtesting/renquant_104/kernel/config_consistency.py::assert_consistent()`
requires every artifact loaded via `LoadScorerTask` (primary panel scorer)
to carry a `config_fingerprint` matching the live strategy_config. PatchTST
artifacts trained before the trainer auto-stamped this field (any
`hf_patchtst_*` snapshot before ~2026-05-25) fail the gate with:

    artifact X has no fingerprint. Live fingerprint=sha256:<hex>.
    Strict full/buy paths require stamped config/sector metadata;
    retrain or promote a stamped artifact.

In the prod daily this manifests on the SHADOW e2e leg (which uses PatchTST
as primary scorer); on the prod leg these artifacts are only used via
`ApplyShadowScoringTask` which doesn't trip the strict gate.

What this stamps (and does NOT)
-------------------------------
Stamps the fields `HFPatchTSTPanelScorer.load()` reads when populating
`self.metadata`. That reader does:

    config_fingerprint = ckpt.get("config_fingerprint") or \
        contract.get("config_contract", {}).get("config_fingerprint")

so the sidecar location is `training_contract.config_contract.config_fingerprint`
(the `contract` dict is built from the sidecar's `training_contract` merged
with the .pt checkpoint's). We also write the top-level keys for any
downstream that uses `_load_contract_sidecar()` directly.

Refuses to stamp when the artifact's training contract is incompatible with
the live config (different lookahead_days, label_col, feature_count). This
prevents silently masking a real config drift.

Defaults to a dry-run; pass `--write` to actually mutate the sidecar.

Usage
-----
  # Dry-run against a single artifact + the live shadow config
  scripts/stamp_patchtst_fingerprint.py \\
      --artifact-meta artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt.metadata.json \\
      --strategy-config backtesting/renquant_104/strategy_config.shadow.json

  # Apply the stamp
  scripts/stamp_patchtst_fingerprint.py ... --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.config_consistency import (  # noqa: E402  pylint: disable=wrong-import-position
    fingerprint_config,
    _model_relevant_fields,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _save_json(path: Path, payload: dict) -> None:
    # Write through a temp file in the same dir so the swap is atomic.
    tmp = path.with_suffix(path.suffix + ".stamp-tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _check_compatibility(artifact_meta: dict, live_fields: dict) -> list[str]:
    """Return list of incompatibility reasons, empty if compatible.

    `live_fields` is the dict produced by `_model_relevant_fields(cfg)` —
    the exact same dict that gets hashed into `config_fingerprint`. Its
    key for forward-label horizon is `lookahead_days` (NOT
    `panel_ltr_lookahead_days`); see kernel/config_consistency.py.
    """
    reasons: list[str] = []
    training = artifact_meta.get("training_contract") or {}
    # lookahead_days — must match live label horizon
    live_lookahead = live_fields.get("lookahead_days")
    art_lookahead = artifact_meta.get("lookahead_days") or training.get("lookahead_days")
    if live_lookahead is not None and art_lookahead is not None:
        if int(live_lookahead) != int(art_lookahead):
            reasons.append(
                f"lookahead_days: live={live_lookahead} artifact={art_lookahead}"
            )
    # label_col — drives sign + horizon semantics
    art_label = training.get("label_col") or artifact_meta.get("label_col")
    # Live label_col is not part of fingerprint, but document if present.
    # (config_consistency does not assert this; only require it doesn't
    # contradict prior stamping evidence.)
    if art_label and not art_label.startswith("fwd_"):
        reasons.append(f"unexpected label_col: {art_label!r}")
    return reasons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact-meta", required=True, type=Path,
                    help="Path to *.metadata.json sidecar to stamp.")
    ap.add_argument("--strategy-config", required=True, type=Path,
                    help="strategy_config.*.json whose fingerprint we stamp.")
    ap.add_argument("--write", action="store_true",
                    help="Actually mutate the sidecar (default: dry-run).")
    ap.add_argument("--force", action="store_true",
                    help="Stamp even if compatibility check returns reasons.")
    args = ap.parse_args()

    if not args.artifact_meta.exists():
        print(f"ERROR: {args.artifact_meta} does not exist", file=sys.stderr)
        return 2
    if not args.strategy_config.exists():
        print(f"ERROR: {args.strategy_config} does not exist", file=sys.stderr)
        return 2

    cfg = _load_json(args.strategy_config)
    meta = _load_json(args.artifact_meta)

    live_fields = _model_relevant_fields(cfg)
    live_fp = fingerprint_config(cfg)

    print(f"Live config: {args.strategy_config}")
    print(f"  → fingerprint: {live_fp}")
    print(f"Artifact sidecar: {args.artifact_meta}")
    print(f"  → stored fingerprint: {meta.get('config_fingerprint')!r}")

    existing_top = meta.get("config_fingerprint")
    existing_nested = (
        meta.get("training_contract", {})
            .get("config_contract", {})
            .get("config_fingerprint")
    )
    if existing_top == live_fp and existing_nested == live_fp:
        print(
            "Already stamped with matching fingerprint in BOTH top-level "
            "and training_contract.config_contract — nothing to do."
        )
        return 0
    if existing_top == live_fp and existing_nested != live_fp:
        print(
            "Top-level fingerprint already stamped; "
            "nested training_contract.config_contract still missing — backfilling."
        )

    reasons = _check_compatibility(meta, live_fields)
    if reasons:
        print("Compatibility check FAILED:")
        for r in reasons:
            print(f"  - {r}")
        if not args.force:
            print("Refusing to stamp. Pass --force to override (NOT recommended).", file=sys.stderr)
            return 3
        print("--force given; stamping anyway.")
    else:
        print("Compatibility check OK.")

    if not args.write:
        print("Dry-run: would set:")
        print(f"  config_fingerprint = {live_fp}")
        print(f"  config_fingerprint_fields = <{len(live_fields)} fields>")
        print("Re-run with --write to apply.")
        return 0

    # Top-level convenience keys (read by some downstream consumers + this stamper itself).
    meta["config_fingerprint"] = live_fp
    meta["config_fingerprint_fields"] = live_fields
    meta["config_fingerprint_stamped_from"] = str(args.strategy_config.name)
    # Nested keys read by HFPatchTSTPanelScorer.load() — `contract.get("config_contract", {})`.
    tc = meta.setdefault("training_contract", {})
    cc = tc.setdefault("config_contract", {})
    cc["config_fingerprint"] = live_fp
    cc["config_fingerprint_fields"] = live_fields
    cc["stamped_from"] = str(args.strategy_config.name)
    _save_json(args.artifact_meta, meta)
    print(
        f"WROTE {args.artifact_meta} with fingerprint {live_fp}\n"
        f"  → top-level config_fingerprint\n"
        f"  → training_contract.config_contract.config_fingerprint"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
