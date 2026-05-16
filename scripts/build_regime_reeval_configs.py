#!/usr/bin/env python3
"""Build sim configs for the 2026-05-15 regime-conditional re-evaluation.

⚠️ POST-MORTEM (2026-05-16): the FIRST version of this script wrote
every knob to a config path that no kernel code reads. Result: 5 panels
× 16 windows = ~5h of compute wasted, every output bit-identical to
baseline (no-ops). See `doc/research/failed-experiments-log.md` entry
"2026-05-15 regime-reeval panels — no-op build script".

KERNEL READER PATHS (verified by grep on backtesting/renquant_104/kernel):

  stop_loss_pct            ← regime_params.<REGIME>.stop_loss_pct
                             (pp_inference.py:43 — `regime_p.get(...)`)
  trailing_stop_trigger_pct ← regime_params.<REGIME>.trailing_stop_trigger_pct
                             (pp_inference.py:41)
  sdl_n_sigma              ← regime_params.<REGIME>.sdl_n_sigma
                             (pp_inference.py:55)
  qp_cvar_lambda           ← rotation.joint_actions.qp_cvar_lambda
                             (portfolio_qp/tasks.py:981 via _qp_cfg)
  min_model_score (tier1)  ← tiered_thresholds[0].min_model_score
                             (selection.py:400, task_joint_actions.py:284)

PRIME DIRECTIVE (CLAUDE.md): every numeric knob lives in
`regime_params.<REGIME>` for per-regime knobs. The build script writes
the SAME value into all five regimes so the panel sweeps the knob
globally; downstream `scripts/analyze_regime_stratified.py` does the
per-regime split on the OUTPUT.

The script calls `scripts/validate_sim_config_active.py` after writing
each config; if static validation reports NO-OP, the build aborts with
non-zero so the bug is caught at config-write time, not 5h later.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CFG_DIR = REPO / "backtesting" / "renquant_104"
REGIMES = ("BULL_CALM", "BEAR", "CHOPPY", "BULL_VOLATILE", "BULL_STRONG")


def load(name: str) -> dict:
    return json.loads((CFG_DIR / name).read_text())


def dump(cfg: dict, name: str) -> None:
    cfg["_side_config_label"] = name.replace("strategy_config.", "").replace(".json", "")
    (CFG_DIR / name).write_text(json.dumps(cfg, indent=2))
    print(f"  wrote {name}")


def validate(name: str, baseline_name: str = "strategy_config.sim_baseline_hmm.json") -> None:
    """Run static validator; abort if NO-OP."""
    cmd = [
        sys.executable, str(REPO / "scripts" / "validate_sim_config_active.py"),
        "--baseline",  baseline_name,
        "--candidate", name,
    ]
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        print(f"❌ VALIDATOR FAILED for {name} — config is a no-op. Aborting build.",
              file=sys.stderr)
        sys.exit(2)
    print(f"  ✓ {name} validator passed (ACTIVE)")


def set_per_regime(cfg: dict, knob: str, value) -> None:
    """Write the same knob value into all 5 regime_params blocks."""
    cfg.setdefault("regime_params", {})
    for r in REGIMES:
        cfg["regime_params"].setdefault(r, {})[knob] = value


def make_pre2024_variant(cfg: dict, base_pre: dict) -> dict:
    """Mirror sim_baseline_hmm_pre2024 aux-artifact paths into a config."""
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


def build():
    base     = load("strategy_config.sim_baseline_hmm.json")
    base_pre = load("strategy_config.sim_baseline_hmm_pre2024.json")

    # ── 1. stop-loss = 0.07 in ALL regimes (tighter than baseline mix) ──
    c = copy.deepcopy(base)
    set_per_regime(c, "stop_loss_pct", 0.07)
    c["_2026-05-15_re_eval_hypothesis"] = {
        "expected": "Tighter stops (-7%) in all regimes. Pooled rejection -7.5pt hides regime heterogeneity.",
        "knob_path": "regime_params.<REGIME>.stop_loss_pct",
        "kernel_reader": "pp_inference.py:43",
    }
    dump(c, "strategy_config.sim_re_stop007.json")
    dump(make_pre2024_variant(c, base_pre), "strategy_config.sim_re_stop007_pre2024.json")
    validate("strategy_config.sim_re_stop007.json")

    # ── 2. σ-aware SDL n_sigma = 2.0 in all regimes ─────────────────────
    c = copy.deepcopy(base)
    set_per_regime(c, "sdl_n_sigma", 2.0)
    c["_2026-05-15_re_eval_hypothesis"] = {
        "expected": "2σ SDL fires often in BEAR/VOL (real catastrophe protection), cuts winners in BULL_CALM.",
        "knob_path": "regime_params.<REGIME>.sdl_n_sigma",
        "kernel_reader": "pp_inference.py:55",
    }
    dump(c, "strategy_config.sim_re_sdl_n2.json")
    dump(make_pre2024_variant(c, base_pre), "strategy_config.sim_re_sdl_n2_pre2024.json")
    validate("strategy_config.sim_re_sdl_n2.json")

    # ── 3. trailing-stop trigger 15% in all regimes ─────────────────────
    c = copy.deepcopy(base)
    set_per_regime(c, "trailing_stop_trigger_pct", 0.15)
    c["_2026-05-15_re_eval_hypothesis"] = {
        "expected": "Tighter trailing protects CHOPPY gains; cuts BULL_STRONG winners.",
        "knob_path": "regime_params.<REGIME>.trailing_stop_trigger_pct",
        "kernel_reader": "pp_inference.py:41",
    }
    dump(c, "strategy_config.sim_re_trail015.json")
    dump(make_pre2024_variant(c, base_pre), "strategy_config.sim_re_trail015_pre2024.json")
    validate("strategy_config.sim_re_trail015.json")

    # ── 4. CVaR λ = 0.25 (rotation.joint_actions.qp_cvar_lambda) ────────
    c = copy.deepcopy(base)
    c.setdefault("rotation", {}).setdefault("joint_actions", {})["qp_cvar_lambda"] = 0.25
    c["_2026-05-15_re_eval_hypothesis"] = {
        "expected": "Rockafellar-Uryasev 2002 tail-risk penalty. BEAR/VOL gain, BULL_CALM cosmetic cost.",
        "knob_path": "rotation.joint_actions.qp_cvar_lambda",
        "kernel_reader": "portfolio_qp/tasks.py:981 via _qp_cfg",
    }
    dump(c, "strategy_config.sim_re_cvar025.json")
    dump(make_pre2024_variant(c, base_pre), "strategy_config.sim_re_cvar025_pre2024.json")
    validate("strategy_config.sim_re_cvar025.json")

    # ── 5. CVaR λ = 0.50 ────────────────────────────────────────────────
    c = copy.deepcopy(base)
    c.setdefault("rotation", {}).setdefault("joint_actions", {})["qp_cvar_lambda"] = 0.50
    c["_2026-05-15_re_eval_hypothesis"] = {
        "expected": "Stronger tail-risk penalty than λ=0.25. Same regime split, larger magnitude.",
        "knob_path": "rotation.joint_actions.qp_cvar_lambda",
        "kernel_reader": "portfolio_qp/tasks.py:981 via _qp_cfg",
    }
    dump(c, "strategy_config.sim_re_cvar050.json")
    dump(make_pre2024_variant(c, base_pre), "strategy_config.sim_re_cvar050_pre2024.json")
    validate("strategy_config.sim_re_cvar050.json")

    # ── 6. Kelly tier-1 raise: tiered_thresholds[0].min_model_score 0.27 → 0.35 ──
    c = copy.deepcopy(base)
    tiered = c.setdefault("tiered_thresholds", [])
    if not tiered:
        tiered.append({"min_model_score": 0.35})
    else:
        # Defensive copy so we don't mutate base's list
        tiered = copy.deepcopy(tiered)
        c["tiered_thresholds"] = tiered
        tiered[0]["min_model_score"] = 0.35
    c["_2026-05-15_re_eval_hypothesis"] = {
        "expected": "Higher quality gate → fewer/better entries. BEAR/CHOPPY +, BULL_CALM -.",
        "knob_path": "tiered_thresholds[0].min_model_score",
        "kernel_reader": "selection.py:400, task_joint_actions.py:284",
    }
    dump(c, "strategy_config.sim_re_kelly_t1_035.json")
    dump(make_pre2024_variant(c, base_pre), "strategy_config.sim_re_kelly_t1_035_pre2024.json")
    validate("strategy_config.sim_re_kelly_t1_035.json")


if __name__ == "__main__":
    print("Building regime-conditional re-eval configs (with active-path validation)...")
    build()
    print("All configs validated as ACTIVE.")
