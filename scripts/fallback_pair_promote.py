#!/usr/bin/env python3
"""RFC#210 fallback pair-promote — the EXACT swap dance extracted VERBATIM
from weekly_wf_promote.sh Step 4b (2026-08-04, --promote-staged mode) so the
scheduled path and the operator promote-staged path run ONE implementation.
License = the promotion_basis stamp (passed=False by design); this script
refuses anything unstamped. argv: STAGING_ART ACTIVE_ART STAGING_CAL ACTIVE_CAL."""
import sys
from pathlib import Path
import json
import os
import shutil

model_src = Path(sys.argv[1])
model_dst = Path(sys.argv[2])
cal_src = Path(sys.argv[3])
cal_dst = Path(sys.argv[4])

model = json.loads(model_src.read_text())
meta = model.get("metadata") or {}
if meta.get("promotion_basis") != "freshness_fallback_rfc210":
    raise SystemExit(
        "staged artifact lacks the freshness_fallback_rfc210 stamp — the "
        "fallback CLI must have stamped it before this promote may run")
gate = meta.get("wf_gate_metadata") or {}
if gate.get("passed") is not False:
    raise SystemExit(
        f"fallback promote requires an explicitly REJECTED candidate "
        f"(stamped passed=False); got {gate.get('passed')!r}")
if "kind" not in model and "feature_cols" not in model:
    raise SystemExit(
        f"staged artifact missing both 'kind' and 'feature_cols' "
        f"({model_src}); refusing to swap into active")

if not cal_src.exists():
    raise SystemExit(f"missing staging calibrator: {cal_src}")
cal_payload = json.loads(cal_src.read_text())
if not isinstance(cal_payload, dict):
    raise SystemExit(f"staging calibrator is not a JSON object: {cal_src}")


def _swap_into_active(staging_path: Path, active_path: Path) -> None:
    # Same atomic-swap file dance as model_acceptance.promote() (same-
    # filesystem os.replace via a .previous rollback target) WITHOUT its
    # _check_wf_gate() call — the license for THIS path is the
    # promotion_basis stamp verified above, not a passing WF gate.
    previous_path = active_path.with_suffix(".previous.json")
    temp_active = active_path.with_suffix(".incoming.json")
    shutil.copy2(str(staging_path), str(temp_active))
    if active_path.exists():
        os.replace(str(active_path), str(previous_path))
    os.replace(str(temp_active), str(active_path))
    staging_path.unlink(missing_ok=True)


cal_incoming = cal_dst.with_suffix(".incoming.json")
shutil.copy2(cal_src, cal_incoming)
try:
    _swap_into_active(model_src, model_dst)
    os.replace(cal_incoming, cal_dst)
except Exception:
    try:
        cal_incoming.unlink()
    except FileNotFoundError:
        pass
    raise
print(f"FALLBACK-promoted {model_src.name} -> {model_dst.name} (rfc210 stamp verified)")
print(f"FALLBACK-promoted {cal_src.name} -> {cal_dst.name}")
