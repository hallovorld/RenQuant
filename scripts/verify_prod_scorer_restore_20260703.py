#!/usr/bin/env python3
"""Byte-verify the 2026-07-03 prod panel-scorer restore (the 06-21 XGB).

Context: orchestrator PR #274 diagnosed that the 2026-06-25 live-tree recovery
checkout silently reverted the operator's 06-22 promotion of the 2026-06-21
XGB panel scorer (booster a9b1a075..., oos_mean_ic 0.0533) back to the
2026-05-18 booster (a6f5a22f..., oos_mean_ic 0.0447), and the revert was then
misread as a stale fingerprint stamp and re-stamped over the wrong booster
(#413, closed unmerged). This script asserts the restored prod artifact is
bit-for-bit the model the 06-22..06-25 daily runs actually used.

Verification chain (hard assertions, exit non-zero on any failure):

  C1  file sha256 of artifacts/prod/panel-ltr.alpha158_fund.json equals the
      panel sha stamped in the 2026-06-25 run bundle
      (pipeline_runs['2026-06-25-live-6c3aa3fa'].artifact_hashes.panel) and
      equals the weekly_rollback_2026-06-23.json snapshot hash pinned by #274.
  C2  booster content sha256 (the #274 diagnosis identity: sha256 over
      json.dumps(payload['booster_raw_json'], sort_keys=True)) equals the
      family-A booster a9b1a075... .
  C3  identity fields: trained_date 2026-06-21, oos_mean_ic 0.0533...,
      label_col fwd_60d_excess, 172 feature_cols.
  C4  stamped config_fingerprint equals sha256:f8fb2259b2bf1537 and — when a
      pinned strategy-104 config is reachable — equals a fresh
      kernel.config_consistency.fingerprint_config over it with zero
      _model_relevant_fields diff (proves the 06-22 promotion's stamp is
      already valid against the pinned config: restore needed NO re-stamp,
      preserving byte identity with the run-bundle sha).

Operator-machine extras (skipped with a notice when sources are absent):

  C5  byte-equality with artifacts/prod/panel-ltr.alpha158_fund
      .weekly_rollback_2026-06-23.json (the recovery source).
  C6  calibrator pairing status: compares the live calibrator's
      scorer_model_content_fingerprint against model_content_sha256(prod
      artifact) IMPORTED from the pinned renquant-pipeline — never
      re-implemented here (the calibrator fingerprint triple-impl lesson:
      hand-copied hash impls diverge by construction). Default: report only
      (expected MISMATCH between restore and the post-restore calibrator
      refit; the runtime's strict_scorer_match fail-closes buys in that
      window). With --require-calibrator-parity: hard-fail on mismatch (use
      to verify the landing AFTER the refit).

Usage:
  python3 scripts/verify_prod_scorer_restore_20260703.py
  python3 scripts/verify_prod_scorer_restore_20260703.py \
      --repo-root /Users/renhao/git/github/RenQuant --require-calibrator-parity
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PROD_ARTIFACT_REL = Path("backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json")
ROLLBACK_SNAPSHOT_REL = Path(
    "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.weekly_rollback_2026-06-23.json"
)
CALIBRATOR_REL = Path("backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json")
PINNED_CONFIG_REL = Path(".subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json")
PINNED_PIPELINE_SRC_REL = Path(".subrepo_runtime/repos/renquant-pipeline/src")
PINNED_COMMON_SRC_REL = Path(".subrepo_runtime/repos/renquant-common/src")
STRATEGY_DIR_REL = Path("backtesting/renquant_104")

# The 2026-06-25 run bundle (pipeline_runs run_id 2026-06-25-live-6c3aa3fa,
# data/runs.alpaca.db) stamps artifact_hashes.panel with exactly this value;
# it is also the sha256 of the weekly_rollback_2026-06-23.json snapshot as
# pinned in #274's evidence JSON (artifact_A_trained_0621_live_0622_0625).
EXPECTED_FILE_SHA256 = "04d7a381cd6df84721dd938ce74a297cbf3eda9d5bc3385515bc155014dd5b08"
# #274 diagnosis §2 family-A booster content hash (full form of a9b1a075...).
EXPECTED_BOOSTER_SHA256 = "a9b1a07533a028588f7fe12b9917108bf9b31af35a388fc8249fcca8ea970bfe"
EXPECTED_TRAINED_DATE = "2026-06-21"
EXPECTED_OOS_MEAN_IC = 0.05332462944060632
EXPECTED_LABEL_COL = "fwd_60d_excess"
EXPECTED_N_FEATURES = 172
# fingerprint_config over the pinned strategy-104 config the live run uses
# (runtime checkout c019b256: XGB override #32 + watchlist 145), i.e. the
# stamp the 06-22 promotion wrote and #413 later re-stamped onto the wrong
# booster.
EXPECTED_CONFIG_FINGERPRINT = "sha256:f8fb2259b2bf1537"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def booster_content_sha256(payload: dict) -> str:
    """#274 diagnosis booster identity (scripts/diagnose_raw_jump_0626.py)."""
    return hashlib.sha256(
        json.dumps(payload["booster_raw_json"], sort_keys=True).encode()
    ).hexdigest()


class Verifier:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, ok: bool, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            self.failures.append(name)

    def skip(self, name: str, why: str) -> None:
        print(f"  [SKIP] {name}: {why}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                    help="umbrella tree holding the prod artifact (default: this repo)")
    ap.add_argument("--pinned-config", type=Path, default=None,
                    help="pinned strategy-104 strategy_config.json "
                         "(default: <repo-root>/.subrepo_runtime/... when present)")
    ap.add_argument("--require-calibrator-parity", action="store_true",
                    help="hard-fail unless the live calibrator is bound to the restored "
                         "scorer's content fingerprint (run AFTER the post-restore refit)")
    args = ap.parse_args()

    root: Path = args.repo_root.resolve()
    v = Verifier()
    art_path = root / PROD_ARTIFACT_REL
    print(f"prod artifact: {art_path}")

    # C1 — file byte identity with the 06-25 run bundle's stamped panel sha.
    actual_file_sha = sha256_file(art_path)
    v.check("C1 file-sha256 == 06-25 run-bundle panel sha",
            actual_file_sha == EXPECTED_FILE_SHA256,
            f"{actual_file_sha} (expected {EXPECTED_FILE_SHA256})")

    payload = json.loads(art_path.read_text())

    # C2 — booster content identity (family A, trained 2026-06-21).
    actual_booster_sha = booster_content_sha256(payload)
    v.check("C2 booster-content-sha256 == #274 family-A (a9b1a075...)",
            actual_booster_sha == EXPECTED_BOOSTER_SHA256,
            f"{actual_booster_sha}")

    # C3 — identity fields.
    v.check("C3 trained_date", payload.get("trained_date") == EXPECTED_TRAINED_DATE,
            f"{payload.get('trained_date')} (expected {EXPECTED_TRAINED_DATE})")
    oos = payload.get("oos_mean_ic")
    v.check("C3 oos_mean_ic",
            isinstance(oos, float) and abs(oos - EXPECTED_OOS_MEAN_IC) < 1e-12,
            f"{oos}")
    v.check("C3 label_col", payload.get("label_col") == EXPECTED_LABEL_COL,
            f"{payload.get('label_col')}")
    v.check("C3 n feature_cols",
            len(payload.get("feature_cols") or []) == EXPECTED_N_FEATURES,
            f"{len(payload.get('feature_cols') or [])}")

    # C4 — config fingerprint vs the pinned strategy config.
    stamped_fp = payload.get("config_fingerprint")
    v.check("C4 stamped config_fingerprint",
            stamped_fp == EXPECTED_CONFIG_FINGERPRINT,
            f"{stamped_fp} (expected {EXPECTED_CONFIG_FINGERPRINT})")
    pinned_cfg = args.pinned_config or (root / PINNED_CONFIG_REL)
    if pinned_cfg.exists():
        sys.path.insert(0, str(root / STRATEGY_DIR_REL))
        try:
            from kernel.config_consistency import (  # type: ignore
                _model_relevant_fields,
                fingerprint_config,
            )
            cfg = json.loads(Path(pinned_cfg).read_text())
            fresh_fp = fingerprint_config(cfg)
            v.check("C4 recomputed fingerprint over pinned config",
                    fresh_fp == stamped_fp,
                    f"fingerprint_config({pinned_cfg}) = {fresh_fp}")
            live_fields = _model_relevant_fields(cfg)
            stored = payload.get("config_fingerprint_fields") or {}
            diff = [k for k in sorted(set(live_fields) | set(stored))
                    if live_fields.get(k) != stored.get(k)]
            v.check("C4 zero model-relevant field diff (no re-stamp needed)",
                    not diff, f"diff={diff}")
        finally:
            sys.path.pop(0)
    else:
        v.skip("C4 recompute vs pinned config",
               f"pinned config not present at {pinned_cfg} (hosted checkout); "
               "stamped-constant assertion above still binds")

    # C5 — byte equality with the recovery source snapshot (operator machine).
    snap = root / ROLLBACK_SNAPSHOT_REL
    if snap.exists():
        snap_sha = sha256_file(snap)
        v.check("C5 byte-equality with weekly_rollback_2026-06-23 snapshot",
                snap_sha == actual_file_sha, f"snapshot sha256 {snap_sha}")
    else:
        v.skip("C5 rollback-snapshot byte-equality",
               f"snapshot not present at {snap} (uncommitted live-tree file)")

    # C6 — calibrator pairing status. model_content_sha256 is IMPORTED from
    # the pinned pipeline; never re-implement it here (triple-impl lesson).
    cal_path = root / CALIBRATOR_REL
    pipeline_src = root / PINNED_PIPELINE_SRC_REL
    common_src = root / PINNED_COMMON_SRC_REL
    if cal_path.exists() and pipeline_src.exists():
        sys.path.insert(0, str(pipeline_src))
        sys.path.insert(0, str(common_src))
        try:
            from renquant_pipeline.kernel.panel_pipeline.panel_scorer import (  # type: ignore
                model_content_sha256_from_path,
            )
            scorer_fp = model_content_sha256_from_path(art_path)
            cal_meta = (json.loads(cal_path.read_text()).get("metadata") or {})
            bound_fp = cal_meta.get("scorer_model_content_fingerprint")
            paired = bound_fp == scorer_fp
            detail = (f"calibrator bound to {bound_fp}; restored scorer is {scorer_fp}; "
                      f"{'PAIRED' if paired else 'REFIT PENDING (strict_scorer_match will fail-close buys)'}")
            if args.require_calibrator_parity:
                v.check("C6 calibrator parity (required)", paired, detail)
            else:
                print(f"  [INFO] C6 calibrator pairing: {detail}")
        except Exception as exc:  # noqa: BLE001 — report-only path
            if args.require_calibrator_parity:
                v.check("C6 calibrator parity (required)", False, f"cannot evaluate: {exc}")
            else:
                v.skip("C6 calibrator pairing", f"cannot evaluate: {exc}")
        finally:
            sys.path.pop(0)
            sys.path.pop(0)
    else:
        why = "calibrator or pinned pipeline source not present (hosted checkout)"
        if args.require_calibrator_parity:
            v.check("C6 calibrator parity (required)", False, why)
        else:
            v.skip("C6 calibrator pairing", why)

    if v.failures:
        print(f"RESULT: FAIL ({len(v.failures)} failed: {', '.join(v.failures)})")
        return 1
    print("RESULT: PASS — restored prod scorer is byte-identical to the 06-21 "
          "promoted model the 06-22..06-25 runs used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
