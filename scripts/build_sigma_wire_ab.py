#!/usr/bin/env python3
"""Build sigma_on / sigma_off sim configs for σ-wire A/B test.

2026-05-17: NGB head retrained today (val_IC=+0.0352, σ-calib=+0.274,
fingerprint sha256:3f380b4164d699b0). 5/9 A/B (E55) showed σ-wire ON
loses -3.78 APY pts. But environment changed since 5/9:
  - Phase 3 μ/σ wiring + Upgrades A+B gates activated in golden
  - Calibrator P0 fixed (5/15 EVENING)
  - Calibrator IC improved pool_ic=+0.094
  - This NGB artifact is fresh (5/17) vs the 5/9 one

Re-run the A/B on 8-window dense panel before flipping σ wire on
production. Reuse the existing 5/16-baseline equity as σ-off control
(σ-off + new artifact ≡ σ-off + old artifact since artifact μ/σ
aren't consumed when enabled=False).

Configs built:
  strategy_config.sim_sigma_on_2026-05-17.json
    - ranking.panel_scoring.ngboost.enabled = True
    - ranking.panel_scoring.ngboost.score_mode = "mu_minus_lambda_sigma"
    - ranking.panel_scoring.ngboost.lambda_sigma = 1.0
    - (matches the 5/9 E55 test configuration)
"""
from __future__ import annotations
import copy, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CFG_DIR = REPO / "backtesting" / "renquant_104"


def load(name): return json.loads((CFG_DIR / name).read_text())
def dump(cfg, name):
    cfg["_side_config_label"] = name.replace("strategy_config.", "").replace(".json", "")
    (CFG_DIR / name).write_text(json.dumps(cfg, indent=2))
    print(f"  wrote {name}")


def static_validate(name, baseline_name):
    cmd = [sys.executable, str(REPO / "scripts" / "validate_sim_config_active.py"),
           "--baseline", baseline_name, "--candidate", name]
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        print(f"❌ STATIC validator FAILED for {name}", file=sys.stderr); sys.exit(2)
    print(f"  ✓ {name} static-validated")


def make_pre2024_variant(cfg, base_pre):
    """Copy correlation_artifact + earnings_artifact paths from pre-2024 baseline."""
    out = copy.deepcopy(cfg)
    for k in ("correlation_artifact", "earnings_artifact"):
        def _patch(d, key=k):
            for kk, vv in d.items():
                if kk == key and isinstance(vv, str):
                    base_val = _get(base_pre, key)
                    if base_val: d[kk] = base_val
                if isinstance(vv, dict): _patch(vv, key)
        _patch(out)
    return out


def _get(d, k):
    if k in d: return d[k]
    for v in d.values():
        if isinstance(v, dict):
            r = _get(v, k)
            if r is not None: return r
    return None


def main():
    base = load("strategy_config.sim_baseline_2026-05-16.json")
    base_pre = load("strategy_config.sim_baseline_2026-05-16_pre2024.json")

    print("== σ-wire A/B build ==")
    print(f"  baseline: sim_baseline_2026-05-16 (reuse existing equity for σ-off control)")
    print(f"  treatment: sim_sigma_on_2026-05-17 (NEW)")

    on = copy.deepcopy(base)
    ngb_block = on.setdefault("ranking", {}).setdefault("panel_scoring", {}).setdefault("ngboost", {})
    print(f"\n  current ngboost block keys: {list(ngb_block.keys())}")
    print(f"  current enabled: {ngb_block.get('enabled')}")
    print(f"  current score_mode: {ngb_block.get('score_mode')}")
    print(f"  current lambda_sigma: {ngb_block.get('lambda_sigma')}")

    # Activate σ wire — match 5/9 E55 test config
    ngb_block["enabled"] = True
    ngb_block["score_mode"] = "mu_minus_lambda_sigma"
    ngb_block["lambda_sigma"] = 1.0
    # Remove the disable_reason since we're explicitly testing it again
    ngb_block.pop("_disable_reason", None)

    on["_2026-05-17_sigma_wire_test"] = {
        "purpose": "re-test σ-wire activation after env changes since 5/9 E55",
        "delta_vs_baseline": (
            "ranking.panel_scoring.ngboost: enabled=False→True, "
            "score_mode=additive→mu_minus_lambda_sigma, lambda_sigma=0.0→1.0"
        ),
        "ngb_artifact_fingerprint": "sha256:3f380b4164d699b0",
        "ngb_val_ic": 0.0352,
        "ngb_sigma_calib": 0.274,
        "reference_5_9_E55": "NGB-on lost -3.78 APY pts on 27-mo A/B",
        "env_changes_since_5_9": [
            "Phase 3 μ/σ wiring + Upgrades A+B gates ON",
            "Calibrator P0 fixed (5/15 EVENING)",
            "Calibrator pool_ic improved to +0.094",
            "Fresh NGB head (5/17) with σ-calib +0.274"
        ],
    }
    dump(on, "strategy_config.sim_sigma_on_2026-05-17.json")
    dump(make_pre2024_variant(on, base_pre),
         "strategy_config.sim_sigma_on_2026-05-17_pre2024.json")
    static_validate("strategy_config.sim_sigma_on_2026-05-17.json",
                    "strategy_config.sim_baseline_2026-05-16.json")
    print()
    print("Done. Launch with scripts/run_sigma_wire_ab.sh")


if __name__ == "__main__":
    main()
