#!/usr/bin/env python3
"""B-track config: per-regime CVaR overlay (kernel-patched 2026-05-16).

Pre-condition: kernel patch at portfolio_qp/tasks.py::_qp_cfg merged.
Test: tests/test_qp_cfg_per_regime_override.py.

Builds:
  strategy_config.sim_btrack_cvar025_BC.json
    - regime_params.BEAR.qp_cvar_lambda   = 0.25
    - regime_params.CHOPPY.qp_cvar_lambda = 0.25
    - rotation.joint_actions.qp_cvar_lambda = 0.0  (unchanged from baseline)
    - other regimes: no override → fallback to 0.0

  strategy_config.sim_btrack_cvar025_BC_pre2024.json — same + pre-2024 aux paths

Validated via scripts/validate_sim_config_active.py (which now knows about
the per-regime QP override paths after the same-day map extension).

Compared against fresh same-day baseline sim_baseline_2026-05-16.json.
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
    out = copy.deepcopy(cfg)
    for k in ("correlation_artifact", "earnings_artifact"):
        def _patch(d):
            for kk, vv in d.items():
                if kk == k and isinstance(vv, str):
                    base_val = _get(base_pre, k)
                    if base_val: d[kk] = base_val
                if isinstance(vv, dict): _patch(vv)
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
    base = load("strategy_config.sim_baseline_hmm.json")
    base_pre = load("strategy_config.sim_baseline_hmm_pre2024.json")

    print("== B-track: per-regime CVaR overlay (kernel-patched) ==")
    c = copy.deepcopy(base)
    c.setdefault("regime_params", {}).setdefault("BEAR",   {})["qp_cvar_lambda"] = 0.25
    c.setdefault("regime_params", {}).setdefault("CHOPPY", {})["qp_cvar_lambda"] = 0.25
    # Leave joint_actions.qp_cvar_lambda at baseline default (typically 0.0)
    c["_2026-05-16_btrack_hypothesis"] = {
        "knob": "regime_params.{BEAR,CHOPPY}.qp_cvar_lambda = 0.25",
        "kernel_reader": "portfolio_qp/tasks.py::_qp_cfg (post 2026-05-16 patch)",
        "test_pin": "tests/test_qp_cfg_per_regime_override.py",
        "global_default": (base.get("rotation", {}).get("joint_actions", {})
                              .get("qp_cvar_lambda", 0.0)),
        "expected": (
            "True regime-conditional version of 5/16 re_cvar025: tail penalty "
            "only fires when ctx.regime ∈ {BEAR, CHOPPY}; other regimes get "
            "baseline behavior. If BEAR/CHOPPY same-side directional gain "
            "observed in 5/16 was real, this should isolate it without BULL "
            "drag pulling pooled mean negative."
        ),
    }
    dump(c, "strategy_config.sim_btrack_cvar025_BC.json")
    dump(make_pre2024_variant(c, base_pre),
         "strategy_config.sim_btrack_cvar025_BC_pre2024.json")
    static_validate("strategy_config.sim_btrack_cvar025_BC.json",
                    "strategy_config.sim_baseline_hmm.json")
    print()
    print("Done. Launch with scripts/run_dense_panel.sh")


if __name__ == "__main__":
    main()
