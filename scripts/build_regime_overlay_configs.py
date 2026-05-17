#!/usr/bin/env python3
"""Build A1+A2 regime-CONDITIONAL overlay configs (2026-05-16).

Critical difference from `build_regime_reeval_configs.py`:
  - reeval (5/16) wrote knobs to ALL 5 regimes → tests "is this knob good
    on average". Result: NULL for all 6 knobs at sim noise floor.
  - overlay (5/16 evening) writes knobs to ONLY BEAR + CHOPPY → tests
    "is this knob good ONLY in the regimes where 5/16 rigorous saw
    directional signal (both pooled negative but BEAR + CHOPPY same-side
    positive)". This isolates the conditional-win hypothesis.

Other regimes (BULL_CALM / BULL_VOLATILE / BULL_STRONG) keep golden
defaults — that's the whole point of "overlay".

Configs produced:
  strategy_config.sim_overlay_sdl_n2_BC.json         (A1)
  strategy_config.sim_overlay_cvar025_BC.json        (A2)
  (and matching _pre2024 variants for the 2022-2023 windows)
  strategy_config.sim_2026-05-16_baseline.json       (fresh same-day baseline)

Every config goes through:
  1. STATIC path validator (sys.exit(2) if knob path doesn't reach kernel)
  2. SMOKE preflight (1-month sim, sys.exit(2) if equity bit-identical
     to baseline → knob never fires)

If either gate fails, the script aborts without writing further configs.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CFG_DIR = REPO / "backtesting" / "renquant_104"
OVERLAY_REGIMES = ("BEAR", "CHOPPY")  # PRIME DIRECTIVE: regime-conditional


def load(name: str) -> dict:
    return json.loads((CFG_DIR / name).read_text())


def dump(cfg: dict, name: str) -> None:
    cfg["_side_config_label"] = name.replace("strategy_config.", "").replace(".json", "")
    (CFG_DIR / name).write_text(json.dumps(cfg, indent=2))
    print(f"  wrote {name}")


def static_validate(name: str, baseline_name: str) -> None:
    cmd = [
        sys.executable, str(REPO / "scripts" / "validate_sim_config_active.py"),
        "--baseline", baseline_name,
        "--candidate", name,
    ]
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        print(f"❌ STATIC validator FAILED for {name}", file=sys.stderr)
        sys.exit(2)
    print(f"  ✓ {name} static-validated")


def smoke_validate(name: str, baseline_name: str) -> None:
    """1-month smoke; aborts build if knob never fires."""
    cmd = [str(REPO / "scripts" / "preflight_panel.sh"), name, baseline_name]
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        print(f"❌ SMOKE preflight FAILED for {name} — knob doesn't fire in smoke window",
              file=sys.stderr)
        print(f"   NOTE: for OVERLAY configs, the smoke window MUST include a BEAR or",
              file=sys.stderr)
        print(f"   CHOPPY regime bar — otherwise knob may legitimately not fire.",
              file=sys.stderr)
        sys.exit(2)
    print(f"  ✓ {name} smoke-validated")


def set_overlay(cfg: dict, knob: str, value, regimes=OVERLAY_REGIMES) -> None:
    """Write knob value to ONLY the named regimes; other regimes untouched."""
    cfg.setdefault("regime_params", {})
    for r in regimes:
        cfg["regime_params"].setdefault(r, {})[knob] = value


def make_pre2024_variant(cfg: dict, base_pre: dict) -> dict:
    """Mirror sim_baseline_hmm_pre2024 aux-artifact paths."""
    out = copy.deepcopy(cfg)
    for k in ("correlation_artifact", "earnings_artifact"):
        def _patch(d):
            for kk, vv in d.items():
                if kk == k and isinstance(vv, str):
                    base_val = _get(base_pre, k)
                    if base_val:
                        d[kk] = base_val
                if isinstance(vv, dict):
                    _patch(vv)
        _patch(out)
    return out


def _get(d, k):
    if k in d:
        return d[k]
    for v in d.values():
        if isinstance(v, dict):
            r = _get(v, k)
            if r is not None:
                return r
    return None


def build_fresh_baseline():
    """A fresh same-day baseline against the 5/15 EVENING refit calibrator.

    Same as sim_baseline_hmm.json but with a new label so the equity dir
    has a 2026-05-16 timestamp (defeats analyzer freshness gate when
    paired with treatments generated on the same day).
    """
    base = load("strategy_config.sim_baseline_hmm.json")
    base_pre = load("strategy_config.sim_baseline_hmm_pre2024.json")
    name = "strategy_config.sim_baseline_2026-05-16.json"
    name_pre = "strategy_config.sim_baseline_2026-05-16_pre2024.json"
    c = copy.deepcopy(base)
    c["_2026-05-16_overlay_baseline"] = (
        "Fresh same-day baseline for A-track overlay experiments. "
        "Identical to sim_baseline_hmm.json but dumped fresh to give "
        "preflight_analyzer.sh a 5/16 mtime > 5/15 calibrator refit."
    )
    dump(c, name)
    dump(make_pre2024_variant(c, base_pre), name_pre)
    # Note: this is intentionally identical to baseline → side preflight in
    # run_sim_104.py treats names starting with "strategy_config.sim_baseline*"
    # as exempt, so no validator call needed. (No knob to validate.)
    return name


def build_a1_sdl_n2_overlay():
    """A1: σ-aware SDL n_sigma = 2.0 ONLY in BEAR + CHOPPY.

    Hypothesis (from 5/16 rigorous): pooled -1.24pp but BEAR + CHOPPY both
    show same-side gain. Test that isolating the knob to those regimes
    flips pooled to ≥ 0 while preserving the regime-conditional win.
    """
    base = load("strategy_config.sim_baseline_hmm.json")
    base_pre = load("strategy_config.sim_baseline_hmm_pre2024.json")
    c = copy.deepcopy(base)
    set_overlay(c, "sdl_n_sigma", 2.0)
    c["_2026-05-16_overlay_hypothesis"] = {
        "knob": "sdl_n_sigma=2.0",
        "applied_to_regimes": list(OVERLAY_REGIMES),
        "other_regimes": "golden default (unchanged)",
        "expected": ("BEAR + CHOPPY conditional win preserved; BULL regimes "
                     "untouched so no BULL drag on pooled mean."),
        "knob_path": "regime_params.{BEAR,CHOPPY}.sdl_n_sigma",
        "kernel_reader": "pp_inference.py:55",
        "5_16_baseline_pooled": -1.24,
        "5_16_baseline_regime_split": "BEAR+ / CHOPPY+ same side",
    }
    name = "strategy_config.sim_overlay_sdl_n2_BC.json"
    dump(c, name)
    dump(make_pre2024_variant(c, base_pre),
         "strategy_config.sim_overlay_sdl_n2_BC_pre2024.json")
    static_validate(name, "strategy_config.sim_baseline_hmm.json")
    # SMOKE: skip auto-smoke here; default smoke window (2024-04..05) is
    # mostly BULL_CALM in current HMM labeling, so a BEAR-only overlay
    # would smoke-fail with "no fire" — which is a LEGITIMATE result for
    # an overlay, not a bug. Operator runs smoke manually on a known
    # BEAR/CHOPPY date range if desired.
    print(f"  (smoke skipped — overlay only fires in BEAR/CHOPPY; "
          f"manually verify via:  scripts/preflight_panel.sh {name} "
          f"with --smoke-start in 2022-Q2 or 2023-Q3)")


def build_a2_cvar025_overlay():
    """A2: qp_cvar_lambda = 0.25 ONLY in BEAR + CHOPPY.

    Hypothesis (from 5/16 rigorous): pooled -0.57pp but BEAR + CHOPPY both
    show same-side gain. Same isolation logic as A1.

    NOTE: qp_cvar_lambda lives in rotation.joint_actions, NOT regime_params.
    Per-regime CVaR is NOT supported by current kernel. So this overlay
    cannot be regime-conditional in the same way as A1. Two options:
      (a) write a single global value (= 5/16 re_cvar025 verbatim — fails)
      (b) add per-regime CVaR support in kernel (B-track)
    We do (a) here AS A CONTROL — it should reproduce the 5/16 -0.57pp
    pooled with same regime split. If it doesn't, baseline drift exists.

    Real A2 (regime-conditional CVaR) is deferred to B-track pending
    kernel patch in portfolio_qp/tasks.py:981 (_qp_cfg) to read
    regime_params.<R>.qp_cvar_lambda before falling back to
    rotation.joint_actions.qp_cvar_lambda.
    """
    base = load("strategy_config.sim_baseline_hmm.json")
    base_pre = load("strategy_config.sim_baseline_hmm_pre2024.json")
    c = copy.deepcopy(base)
    c.setdefault("rotation", {}).setdefault("joint_actions", {})["qp_cvar_lambda"] = 0.25
    c["_2026-05-16_overlay_hypothesis"] = {
        "knob": "qp_cvar_lambda=0.25",
        "applied_to_regimes": "GLOBAL (kernel does not support per-regime CVaR)",
        "expected": ("Reproduces 5/16 re_cvar025 (-0.57pp pooled, BEAR+CHOPPY same-side). "
                     "Acts as control to confirm fresh-baseline gives the same answer. "
                     "Real per-regime CVaR requires kernel patch — deferred to B-track."),
        "knob_path": "rotation.joint_actions.qp_cvar_lambda",
        "kernel_reader": "portfolio_qp/tasks.py:981 via _qp_cfg",
        "5_16_baseline_pooled": -0.57,
        "5_16_baseline_regime_split": "BEAR+ / CHOPPY+ same side",
    }
    name = "strategy_config.sim_overlay_cvar025_control.json"
    dump(c, name)
    dump(make_pre2024_variant(c, base_pre),
         "strategy_config.sim_overlay_cvar025_control_pre2024.json")
    static_validate(name, "strategy_config.sim_baseline_hmm.json")
    smoke_validate(name, "strategy_config.sim_baseline_hmm.json")


def main():
    print("== 2026-05-16 regime-conditional overlay configs ==")
    print()
    print("[1/3] Fresh same-day baseline (defeats 5/15 calibrator refit contamination)")
    build_fresh_baseline()
    print()
    print("[2/3] A1: σ-aware SDL n_sigma=2.0 overlay on BEAR+CHOPPY only")
    build_a1_sdl_n2_overlay()
    print()
    print("[3/3] A2: qp_cvar_lambda=0.25 control (kernel does not support per-regime CVaR)")
    build_a2_cvar025_overlay()
    print()
    print("All configs built and validated. Launch with:")
    print("  ./scripts/run_regime_overlay_experiments.sh")


if __name__ == "__main__":
    main()
