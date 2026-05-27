#!/usr/bin/env python
"""Re-stamp a production panel-LTR artifact's config fingerprint in place.

Repair tool for legacy artifacts (trained before 2026-05-18) whose
``config_fingerprint_fields`` predate ``sector_map`` / ``sector_etf_map`` being
added to the model-relevant fingerprint (kernel/config_consistency.py). Such an
artifact stores those two fields as ``None``, so P-CONFIG-FP in full/buy mode
sees a fingerprint mismatch on exactly those keys and routes the daily run to
the sell-only fallback (no new buys) until the artifact is re-stamped.

It never touches model weights. It only records the current sector maps so the
fingerprint is complete and future genuine sector drift is detectable again.

Preconditions enforced before writing:
  * every non-sector model-relevant field already matches the live config
    (watchlist, lookahead, objective, resolution, embeddings, intraday flags),
  * the only differing fields are sector_map / sector_etf_map.

If any other field differs the artifact is genuinely stale (real model drift)
and must be retrained, not re-stamped — the script refuses and exits non-zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.config_consistency import (  # noqa: E402
    _model_relevant_fields,
    fingerprint_config,
)

SECTOR_FIELDS = {"sector_map", "sector_etf_map"}


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="strategy_config.json",
        help="strategy config name under the strategy dir (runtime default).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report the diff and the new fingerprint without writing.",
    )
    args = ap.parse_args()

    cfg_path = STRATEGY_DIR / args.config
    cfg = json.loads(cfg_path.read_text())

    art_rel = cfg["ranking"]["panel_scoring"]["artifact_path"]
    art_path = STRATEGY_DIR / art_rel
    art = json.loads(art_path.read_text())

    live = _model_relevant_fields(cfg)
    stored = art.get("config_fingerprint_fields") or {}
    if not isinstance(stored, dict):
        print(f"REFUSE: stored fingerprint_fields is {type(stored).__name__}, "
              f"not a dict — retrain instead.")
        return 2

    diff = [k for k in sorted(set(live) | set(stored))
            if live.get(k) != stored.get(k)]
    print(f"artifact:   {art_path.relative_to(REPO)}")
    print(f"config:     {cfg_path.relative_to(REPO)}")
    print(f"diff fields: {diff}")

    if not diff:
        print("Already consistent — nothing to do.")
        return 0

    non_sector_diff = [k for k in diff if k not in SECTOR_FIELDS]
    if non_sector_diff:
        print(f"REFUSE: non-sector model-relevant fields differ {non_sector_diff} "
              f"— the artifact is genuinely stale; retrain, do not re-stamp.")
        return 2

    new_fp = fingerprint_config(cfg)
    print(f"old fingerprint: {art.get('config_fingerprint')}")
    print(f"new fingerprint: {new_fp}")

    if args.dry_run:
        print("(dry-run) no write performed.")
        return 0

    backup = art_path.with_suffix(art_path.suffix + ".bak_restamp")
    backup.write_text(art_path.read_text())
    print(f"backup written: {backup.relative_to(REPO)}")

    art["config_fingerprint_fields"] = live
    art["config_fingerprint"] = new_fp
    art.setdefault("metadata", {})["config_fingerprint_source"] = {
        "fingerprint_config_path": args.config,
        "restamped_by": "scripts/restamp_prod_fingerprint.py",
        "reason": "legacy artifact missing sector_map/sector_etf_map fields",
    }
    _atomic_write_json(art_path, art)

    # Verify post-write consistency.
    check = json.loads(art_path.read_text())
    if check.get("config_fingerprint") != fingerprint_config(cfg):
        print("VERIFY FAILED: post-write fingerprint does not match live config.")
        return 1
    residual = [k for k in sorted(set(live) | set(check["config_fingerprint_fields"]))
                if live.get(k) != check["config_fingerprint_fields"].get(k)]
    if residual:
        print(f"VERIFY FAILED: residual diff after stamp {residual}")
        return 1
    print("VERIFY OK: artifact fingerprint now matches live config (0 diff).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
