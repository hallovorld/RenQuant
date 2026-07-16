"""Pre-activation power / type-I simulation for the equal-weight deployment prereg.

Implements §4.6 of doc/experiments/2026-07-13-equal-weight-deployment-prereg.md
(merged RenQuant#480). Validates that the FROZEN inference procedure (§4.1
moving-block bootstrap, one-sided 90% lower bound compared against the
MDE of 3 bps/session) controls the false-positive rate (type-I <= 0.10)
and achieves power >= 0.80 at the MDE, BEFORE any prospective observation.

The GO criterion under test is exactly §4.1 step 6: reject (declare GO)
iff the 10th percentile of 10,000 MBB bootstrap means EXCEEDS the MDE.
This script does not soften or reinterpret that criterion; if the frozen
design fails its own gate, the failure is the result (§4.6: "If this
simulation fails, the protocol's horizon or MDE is revised — the
experiment does not start with unvalidated operating characteristics.").

DGP calibration status: PROVISIONAL PARAMETRIC. §4.6 asks for calibration
from an A1-vs-B historical replay or D6 sim replay. The two-arm harness
has produced only ~4 paired sessions since 2026-07-11 — far too few to
fit an AR(p) or empirical ACF — and no per-session paired active-return
series for THIS experiment's two arms exists in the artifact store. The
simulation therefore uses an AR(1) DGP with stated parameters and a
sensitivity grid over (sigma_d, phi, b); re-running with a data-fitted
DGP before activation is mandatory and cheap (one flag).

All quantities are in bps/session. Deterministic under --seed.

Usage:
    python scripts/prereg_ew_power_simulation.py \
        --out doc/experiments/2026-07-16-ew-prereg-power-simulation-results.json

Research-only: no production paths read or written.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MDE_BPS = 3.0                # §4.1 / §4.3 G1
BOOT_DRAWS = 10_000          # §4.1 step 3 (frozen)
ALPHA = 0.10                 # one-sided 90% lower bound (§4.1 step 5)
TRIALS = 5_000               # §4.6 steps 2-3 (frozen)
T_PRIMARY = 240              # §4.4 minimum observation period
T_MAX = 480                  # §4.6 step 4 horizon cap
T_STEP = 20                  # §4.6 step 4 increment
TYPE_I_REQUIREMENT = 0.10    # §4.6 step 2
POWER_REQUIREMENT = 0.80     # §4.6 step 3

# Provisional DGP parameters (see module docstring). sigma_d is the
# marginal std of the daily paired active return d_t in bps/session;
# phi the AR(1) coefficient standing in for holding-overlap dependence;
# max_holding_days drives the MBB block length b = ceil(1.75 * mhd).
PRIMARY_SIGMA_BPS = 25.0
PRIMARY_PHI = 0.35
PRIMARY_MAX_HOLDING_DAYS = 20
SENSITIVITY_SIGMA = (10.0, 25.0, 40.0)
SENSITIVITY_PHI = (0.0, 0.35, 0.6)
SENSITIVITY_MHD = (10, 20, 40)   # 40 = the §4.1 cap
SENSITIVITY_TRIALS = 1_000       # grid cells only; primary runs use TRIALS
SENSITIVITY_BOOT = 2_000         # grid cells only; primary runs use BOOT_DRAWS


def mbb_block_length(max_holding_days: int) -> int:
    """§4.1 step 2: b = ceil(1.75 * max_holding_days), mhd capped at 40."""
    return math.ceil(1.75 * min(max_holding_days, 40))


def simulate_ar1(n_trials: int, t_sessions: int, mean_bps: float,
                 sigma_bps: float, phi: float, rng: np.random.Generator,
                 burn_in: int = 200) -> np.ndarray:
    """(n_trials, t_sessions) AR(1) series with marginal std sigma_bps."""
    if not 0.0 <= phi < 1.0:
        raise ValueError(f"phi must be in [0, 1): {phi}")
    eps_sigma = sigma_bps * math.sqrt(1.0 - phi * phi)
    total = burn_in + t_sessions
    eps = rng.normal(0.0, eps_sigma, size=(n_trials, total))
    out = np.empty((n_trials, total))
    out[:, 0] = rng.normal(0.0, sigma_bps, size=n_trials)
    for t in range(1, total):
        out[:, t] = phi * out[:, t - 1] + eps[:, t]
    return out[:, burn_in:] + mean_bps


def mbb_lower_bound(d: np.ndarray, b: int, rng: np.random.Generator,
                    n_boot: int = BOOT_DRAWS, alpha: float = ALPHA) -> float:
    """§4.1 steps 3-5 for one series: 10th pctile of n_boot MBB means."""
    t_sessions = d.shape[0]
    if b > t_sessions:
        raise ValueError(f"block length {b} exceeds series length {t_sessions}")
    n_blocks = math.ceil(t_sessions / b)
    starts = rng.integers(0, t_sessions - b + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)
    means = d[idx[:, :t_sessions]].mean(axis=1)
    return float(np.quantile(means, alpha))


def rejection_rate(n_trials: int, t_sessions: int, mean_bps: float,
                   sigma_bps: float, phi: float, b: int, seed: int,
                   n_boot: int = BOOT_DRAWS) -> dict:
    """§4.6 steps 2-3: fraction of trials whose MBB lower bound > MDE."""
    rng = np.random.default_rng(seed)
    series = simulate_ar1(n_trials, t_sessions, mean_bps, sigma_bps, phi, rng)
    rejections = 0
    for i in range(n_trials):
        if mbb_lower_bound(series[i], b, rng, n_boot=n_boot) > MDE_BPS:
            rejections += 1
    rate = rejections / n_trials
    # 95% simulation CI (normal approximation on a binomial proportion)
    half = 1.96 * math.sqrt(max(rate * (1.0 - rate), 1e-12) / n_trials)
    return {
        "rate": rate,
        "ci95": [max(0.0, rate - half), min(1.0, rate + half)],
        "trials": n_trials,
        "rejections": rejections,
    }


def run_protocol(seed: int, quick: bool = False) -> dict:
    """Full §4.6 protocol: null at T=240, power sweep T=240..480, grid."""
    trials = 50 if quick else TRIALS
    boots = 200 if quick else BOOT_DRAWS
    b_primary = mbb_block_length(PRIMARY_MAX_HOLDING_DAYS)

    null_res = rejection_rate(trials, T_PRIMARY, 0.0, PRIMARY_SIGMA_BPS,
                              PRIMARY_PHI, b_primary, seed, n_boot=boots)

    # §4.6 step 4 horizon adjustment: sweep T upward until power passes
    # or the 480-session cap is reached.
    power_by_t: dict[str, dict] = {}
    final_t = None
    for t_sessions in range(T_PRIMARY, T_MAX + 1, T_STEP):
        res = rejection_rate(trials, t_sessions, MDE_BPS, PRIMARY_SIGMA_BPS,
                             PRIMARY_PHI, b_primary, seed + t_sessions,
                             n_boot=boots)
        power_by_t[str(t_sessions)] = res
        if res["rate"] >= POWER_REQUIREMENT:
            final_t = t_sessions
            break
        if quick:
            break

    grid = []
    if not quick:
        for sigma in SENSITIVITY_SIGMA:
            for phi in SENSITIVITY_PHI:
                for mhd in SENSITIVITY_MHD:
                    b = mbb_block_length(mhd)
                    cell_seed = seed + int(sigma * 1000) + int(phi * 100) + mhd
                    grid.append({
                        "sigma_bps": sigma, "phi": phi,
                        "max_holding_days": mhd, "b": b,
                        "type_i": rejection_rate(
                            SENSITIVITY_TRIALS, T_PRIMARY, 0.0, sigma, phi,
                            b, cell_seed, n_boot=SENSITIVITY_BOOT),
                        "power_at_mde": rejection_rate(
                            SENSITIVITY_TRIALS, T_PRIMARY, MDE_BPS, sigma,
                            phi, b, cell_seed + 1, n_boot=SENSITIVITY_BOOT),
                    })

    type_i_pass = null_res["rate"] <= TYPE_I_REQUIREMENT
    power_pass = final_t is not None
    return {
        "prereg": "doc/experiments/2026-07-13-equal-weight-deployment-prereg.md",
        "section": "4.6",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "frozen_parameters": {
            "mde_bps_per_session": MDE_BPS,
            "bootstrap_draws": boots,
            "alpha_one_sided": ALPHA,
            "monte_carlo_trials": trials,
            "t_primary": T_PRIMARY, "t_max": T_MAX, "t_step": T_STEP,
        },
        "dgp": {
            "kind": "AR(1) — PROVISIONAL PARAMETRIC (see caveat)",
            "sigma_bps": PRIMARY_SIGMA_BPS,
            "phi": PRIMARY_PHI,
            "max_holding_days": PRIMARY_MAX_HOLDING_DAYS,
            "b": b_primary,
            "caveat": ("not fitted to an A1-vs-B replay: only ~4 paired "
                       "sessions exist (two-arm harness live 2026-07-11); "
                       "refit before activation is mandatory"),
        },
        "type_i": {**null_res, "requirement": TYPE_I_REQUIREMENT,
                   "pass": type_i_pass},
        "power_at_mde_by_t": power_by_t,
        "power": {"requirement": POWER_REQUIREMENT,
                  "final_t": final_t, "pass": power_pass},
        "sensitivity_grid": grid,
        "verdict": {
            "type_i_pass": type_i_pass,
            "power_pass": power_pass,
            "gate_pass": type_i_pass and power_pass,
        },
        "code_commit": _git_head(),
    }


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, check=False).stdout.strip() or "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true",
                        help="50-trial smoke run (NOT valid for activation)")
    args = parser.parse_args()

    results = run_protocol(args.seed, quick=args.quick)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")

    v = results["verdict"]
    print(f"type-I: {results['type_i']['rate']:.4f} "
          f"(req <= {TYPE_I_REQUIREMENT}) pass={v['type_i_pass']}")
    for t_sessions, res in results["power_at_mde_by_t"].items():
        print(f"power@MDE T={t_sessions}: {res['rate']:.4f}")
    print(f"power gate (>= {POWER_REQUIREMENT} by T={T_MAX}): "
          f"pass={v['power_pass']}")
    print(f"SECTION 4.6 GATE: {'PASS' if v['gate_pass'] else 'FAIL'}")
    return 0 if v["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
