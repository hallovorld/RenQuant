#!/usr/bin/env python3
"""Build sigma_on_BEAR_CHOPPY sim config for per-regime σ-wire A/B.

2026-05-17: per-regime σ-wire kernel patch (job_panel_scoring.py::
_ngb_cfg). regime_params.{BEAR,CHOPPY}.ngboost.{enabled,score_mode,
lambda_sigma} overrides the global ranking.panel_scoring.ngboost.

5/17 dense A/B showed:
  BEAR/crisis windows (n=4): σ-on mean +14.4pp
  BULL windows (n=2):       σ-on mean -14.4pp
  Pooled (n=8):              σ-on mean +3.0pp (NULL, CI crosses 0)

This config activates σ-wire ONLY in BEAR/CHOPPY (also tries
BULL_VOLATILE since W7 Aug-2024 vol was a similar win).

Other regimes (BULL_CALM, BULL_STRONG) keep global default (off).
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


def make_pre2024(cfg, base_pre):
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

    print("== σ-wire per-regime A/B (BEAR/CHOPPY/BULL_VOLATILE only) ==")
    on = copy.deepcopy(base)

    # IMPORTANT: keep global ngboost.enabled = false. Only the per-regime
    # overlay flips it on in target regimes.
    ngb_block = on.setdefault("ranking", {}).setdefault("panel_scoring", {}).setdefault("ngboost", {})
    assert ngb_block.get("enabled") is False, "expected baseline ngboost.enabled=false"
    ngb_block.pop("_disable_reason", None)  # remove stale comment

    # Per-regime activation
    for regime in ("BEAR", "CHOPPY", "BULL_VOLATILE"):
        on.setdefault("regime_params", {}).setdefault(regime, {})["ngboost"] = {
            "enabled": True,
            "score_mode": "mu_minus_lambda_sigma",
            "lambda_sigma": 1.0,
        }

    on["_2026-05-17_sigma_wire_per_regime"] = {
        "purpose": "regime-conditional σ-wire activation per 5/17 dense A/B finding",
        "global_default": "ngboost.enabled=false (BULL_CALM/BULL_STRONG keep this)",
        "per_regime_overlay": {
            "BEAR":          {"enabled": True, "score_mode": "mu_minus_lambda_sigma", "lambda_sigma": 1.0},
            "CHOPPY":        {"enabled": True, "score_mode": "mu_minus_lambda_sigma", "lambda_sigma": 1.0},
            "BULL_VOLATILE": {"enabled": True, "score_mode": "mu_minus_lambda_sigma", "lambda_sigma": 1.0},
        },
        "kernel_reader": "job_panel_scoring.py::_ngb_cfg (post 2026-05-17 patch)",
        "test_pin": "tests/test_per_regime_sigma_wire.py",
        "expected": (
            "Captures σ-on +14pp BEAR/crisis gains (W1,W4,W5,W7) without "
            "paying -14pp BULL drag (W3,W6). Pooled mean should swing from "
            "+3pp NULL to materially positive."
        ),
        "ngb_artifact_fingerprint": "sha256:3f380b4164d699b0",
    }
    dump(on, "strategy_config.sim_sigma_on_BEAR_CHOPPY_2026-05-17.json")
    dump(make_pre2024(on, base_pre),
         "strategy_config.sim_sigma_on_BEAR_CHOPPY_2026-05-17_pre2024.json")
    static_validate("strategy_config.sim_sigma_on_BEAR_CHOPPY_2026-05-17.json",
                    "strategy_config.sim_baseline_2026-05-16.json")
    print()
    print("Done. Launch with: nohup ./scripts/run_sigma_wire_BC_ab.sh > "
          "logs/reeval_queue/sigma_wire_BC_ab.log 2>&1 &")


if __name__ == "__main__":
    main()
