#!/usr/bin/env python3
"""Build sim configs for the 2026-05-15 regime-conditional re-evaluation.

The PRIME DIRECTIVE (CLAUDE.md): RenQuant is REGIME-CONDITIONAL. Earlier
pool-mean rejections (E55 NGBoost, stop-loss family, CVaR sweep, multi-
horizon, Kelly tier-1 raise) were evaluated WITHOUT regime stratification.
The long-short clean test showed pooled NEITHER (+6.23pt p=0.23) hid a
3-regime WIN (+13~+22pt) — the same bias likely applies to these others.

This script generates 6 sim configs (+ pre2024 variants) from the
sim_baseline_hmm template, each toggling exactly ONE knob. All other
settings inherit from baseline_hmm — guarantees apples-to-apples vs
the existing baseline panel.

Output files (all under backtesting/renquant_104/):
  strategy_config.sim_re_stop007.json + _pre2024
  strategy_config.sim_re_sdl_n2.json + _pre2024
  strategy_config.sim_re_trail015.json + _pre2024
  strategy_config.sim_re_cvar025.json + _pre2024
  strategy_config.sim_re_cvar050.json + _pre2024
  strategy_config.sim_re_kelly_t1_035.json + _pre2024
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CFG_DIR = REPO / "backtesting" / "renquant_104"


def load(name: str) -> dict:
    return json.loads((CFG_DIR / name).read_text())


def dump(cfg: dict, name: str) -> None:
    cfg["_side_config_label"] = name.replace("strategy_config.", "").replace(".json", "")
    (CFG_DIR / name).write_text(json.dumps(cfg, indent=2))
    print(f"  wrote {name}")


def make_pre2024_variant(cfg: dict, base_pre: dict) -> dict:
    """Mirror sim_baseline_hmm_pre2024 aux-artifact paths into a config."""
    out = copy.deepcopy(cfg)
    # Copy the point-in-time aux artifact paths from baseline_hmm_pre2024
    for k in ("correlation_artifact", "earnings_artifact"):
        # Search both top-level and any sub-dicts
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

    # ── 1. stop-loss = 0.07 (tighter than golden 0.15) ─────────────────
    c = copy.deepcopy(base)
    c.setdefault("risk", {})["stop_loss_pct"] = 0.07
    c.setdefault("_2026-05-15_re_eval_hypothesis", {})
    c["_2026-05-15_re_eval_hypothesis"]["expected"] = (
        "Tighter stops (-7% vs -15%) protect capital in BEAR/CHOPPY but "
        "clip winners in BULL_CALM. Pooled rejection -7.5pt hides regime "
        "heterogeneity. Expected: BEAR +2~5pt, BULL_CALM -12~-15pt."
    )
    dump(c, "strategy_config.sim_re_stop007.json")
    dump(make_pre2024_variant(c, base_pre),
         "strategy_config.sim_re_stop007_pre2024.json")

    # ── 2. σ-aware single-day-loss n_sigma = 2.0 (tightest) ─────────────
    c = copy.deepcopy(base)
    sdl = c.setdefault("risk", {}).setdefault("sigma_aware_sdl", {})
    sdl["enabled"] = True
    sdl["n_sigma"] = 2.0
    c["_2026-05-15_re_eval_hypothesis"] = {
        "expected": (
            "2σ SDL fires often in BEAR/VOL regimes (real catastrophe "
            "protection) but cuts winners in BULL_CALM (false positives). "
            "Pooled -10.4pt. Expected: BEAR/VOL +3~5pt, BULL_CALM -12~-15pt."
        ),
    }
    dump(c, "strategy_config.sim_re_sdl_n2.json")
    dump(make_pre2024_variant(c, base_pre),
         "strategy_config.sim_re_sdl_n2_pre2024.json")

    # ── 3. trailing-stop trigger 15% (was 25%) ──────────────────────────
    c = copy.deepcopy(base)
    c.setdefault("risk", {})["trailing_stop_trigger_pct"] = 0.15
    c["_2026-05-15_re_eval_hypothesis"] = {
        "expected": (
            "Tighter trailing protects gains in CHOPPY/REVERT; cuts "
            "winners in TRENDING BULL_STRONG. Pooled negative; expect "
            "regime split."
        ),
    }
    dump(c, "strategy_config.sim_re_trail015.json")
    dump(make_pre2024_variant(c, base_pre),
         "strategy_config.sim_re_trail015_pre2024.json")

    # ── 4. CVaR λ = 0.25 ────────────────────────────────────────────────
    c = copy.deepcopy(base)
    qp = c.setdefault("rotation", {}).setdefault("joint_actions", {})
    qp["cvar_lambda"] = 0.25
    c["_2026-05-15_re_eval_hypothesis"] = {
        "expected": (
            "Rockafellar-Uryasev 2002 tail-risk penalty. CVaR cuts "
            "positions when tail VaR is high (BEAR/BULL_VOL). In "
            "BULL_CALM tail is small → penalty cosmetic, costs upside. "
            "Pooled -3.3pt. Expected: BEAR +1~3pt, BULL_CALM -4~-6pt."
        ),
    }
    dump(c, "strategy_config.sim_re_cvar025.json")
    dump(make_pre2024_variant(c, base_pre),
         "strategy_config.sim_re_cvar025_pre2024.json")

    # ── 5. CVaR λ = 0.50 (more aggressive) ──────────────────────────────
    c = copy.deepcopy(base)
    c.setdefault("rotation", {}).setdefault("joint_actions", {})["cvar_lambda"] = 0.50
    c["_2026-05-15_re_eval_hypothesis"] = {
        "expected": (
            "Aggressive tail-risk penalty. Same regime split as λ=0.25 "
            "but stronger effect both directions. Pooled -1.6pt."
        ),
    }
    dump(c, "strategy_config.sim_re_cvar050.json")
    dump(make_pre2024_variant(c, base_pre),
         "strategy_config.sim_re_cvar050_pre2024.json")

    # ── 6. Kelly tier-1 raise rank_score 0.27 → 0.35 ────────────────────
    c = copy.deepcopy(base)
    ks = c.setdefault("ranking", {}).setdefault("kelly_sizing", {})
    ks["tier1_rank_score_threshold"] = 0.35
    c["_2026-05-15_re_eval_hypothesis"] = {
        "expected": (
            "Higher quality gate (rank 0.35) → 82%→91% hit-rate. Should "
            "help BEAR/CHOPPY (fewer bad trades), hurt BULL_CALM (fewer "
            "shots at mean-revert winners). Pooled -8.88pt APY / +0.74 "
            "Sharpe. Expected: BEAR +2~4pt, BULL_CALM -10~-12pt."
        ),
    }
    dump(c, "strategy_config.sim_re_kelly_t1_035.json")
    dump(make_pre2024_variant(c, base_pre),
         "strategy_config.sim_re_kelly_t1_035_pre2024.json")


if __name__ == "__main__":
    print("Building regime-conditional re-eval configs...")
    build()
    print("Done.")
