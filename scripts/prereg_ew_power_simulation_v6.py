"""Pre-activation power / type-I simulation — v6 (BCa coverage-corrected bound).

Implements the v6 amendment
(``doc/experiments/2026-07-19-equal-weight-deployment-prereg-v6.md``) to the merged
v5 §4.6 pre-activation simulation. v6 REPLACES only the §4.1 inference method — the
one-sided 90% lower-bound computation — and adds the HARD §4.6-step-2 method
CALIBRATION gate that v5 was frozen without.

WHY v6 EXISTS. The v5 §4.6 sim (merged RenQuant#511) proved that the frozen §4.1
inference — the raw **percentile of the MBB bootstrap means** (10th percentile) — is
intrinsically ANTI-CONSERVATIVE at the deployment boundary: boundary type-I at
``mu = MEE`` runs ~0.13-0.15 (T=240) / ~0.12-0.13 (T=480), with every 95% MC CI
EXCLUDING the nominal alpha = 0.10, and it persists even under iid (phi=0). So no
sealed pilot fit could ever yield a valid GO under v5. This module is the empirical
acceptance evidence for the v6 fix.

THE REPAIRED BOUND (§4.1 v6). The one-sided 90% lower confidence bound for
``mean(d_t)`` is computed by a **BCa (bias-corrected + accelerated) block bootstrap**:

  * resampling stays the moving-block bootstrap with the v5 block-length rule
    ``b = ceil(1.75 * max_holding_days)`` capped at 40 sessions, B = 10,000 draws
    (the resampling primitive is REUSED, byte-identical, from the merged
    ``prereg_ew_power_simulation`` module — a pinned test asserts the equivalence);
  * bias-correction ``z0`` from the proportion of bootstrap means below the observed
    mean;
  * acceleration ``a`` from a DELETE-ONE-BLOCK jackknife over non-overlapping
    length-``b`` blocks (so the acceleration respects the same dependence the MBB
    models), via the standard
    ``a = sum (thbar - th_(i))^3 / (6 [sum (thbar - th_(i))^2]^{3/2})``;
  * the 10th percentile is mapped through the BCa-adjusted percentile
    ``alpha1 = Phi(z0 + (z0 + z_alpha) / (1 - a (z0 + z_alpha)))`` and the bound is the
    ``alpha1``-quantile of the bootstrap means.

Both bounds are kept side by side (``mbb_lower_bound`` = the merged v5 percentile bound;
``mbb_bca_lower_bound`` = the v6 bound) so the calibration can be COMPARED, never
silently swapped.

HARD §4.6 CALIBRATION GATE (v6 §C). Activation now requires PROVING, before any pilot,
that the BCa boundary type-I at ``mu = MEE`` is ``<= 0.10`` with 95% CI upper
``<= 0.12`` under BOTH an iid control AND the pilot null. If BCa still under-controls,
the honest outcome is ``INFEASIBLE — INFERENCE METHOD UNCALIBRATED`` (→ a v7
studentized-t amendment), NEVER a relaxation of alpha / MEE / block length.

WHAT IS UNCHANGED. MEE = 3 / PE = 6, the deployment rule ``LB90 > MEE`` (decides), the
pilot-calibrated-null input contract (:class:`PilotNull`, reused from v5), the two-stage
start, and the power target (``>= 0.80`` at ``mu = PE`` within ``T <= 480``). Only the
bound COMPUTATION and the calibration gate change.

All quantities are bps/session. Deterministic under ``--seed``. Research-only: no
production paths are read or written.

Usage (VALIDATION — the load-bearing calibration evidence, percentile vs BCa):
    python scripts/prereg_ew_power_simulation_v6.py --validate \
        --trials 8000 --out doc/experiments/<date>-ew-prereg-v6-validation.json

Usage (ACTIVATION — real sealed pilot fit, full v6 protocol):
    python scripts/prereg_ew_power_simulation_v6.py \
        --pilot-fit <sealed_pilot_fit.json> --seed 20260719 \
        --out doc/experiments/<date>-ew-prereg-v6-activation-sim.json

Usage (PREVIEW / smoke — synthetic default, NOT for activation):
    python scripts/prereg_ew_power_simulation_v6.py --preview --out /tmp/preview.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np

# Reuse the merged, version-agnostic numeric primitives (the DGP + the percentile
# MBB lower bound). The frozen simulation_code_commit pins the whole tree, so these
# imports are audit-safe.
from prereg_ew_power_simulation import mbb_lower_bound, simulate_ar1
# Reuse the v5 pilot-null contract, the block-length rule, the synthetic default,
# and the frozen decision constants unchanged by v6.
from prereg_ew_power_simulation_v5 import (  # noqa: F401  (mbb_block_length re-exported)
    ALPHA,
    BOOT_DRAWS,
    MEE_BPS,
    PE_BPS,
    POWER_REQUIREMENT,
    T_MAX,
    T_PRIMARY,
    T_STEP,
    TRIALS,
    TYPE_I_REQUIREMENT,
    PilotNull,
    SYNTHETIC_DEFAULT,
    mbb_block_length,
)

# --- v6-only frozen constants ------------------------------------------------
# §4.6 step 2 (v6): the 95% simulation-CI UPPER limit for the boundary type-I must
# not exceed this — a margin so a borderline method can't slip through on MC noise.
TYPE_I_CI_UPPER_MAX = 0.12

_NORM = NormalDist()

# Outcomes (§4.6 verdict).
GO_VALID = "OPERATING_CHARACTERISTICS_VALID"
# Power unreachable at the conservative pilot noise (unchanged from v5).
INFEASIBLE_NOISE = "INFEASIBLE AT CONSERVATIVE PILOT NOISE"
# BCa still under-controls boundary type-I -> the method is uncalibrated -> v7.
INFEASIBLE_INFERENCE = "INFEASIBLE — INFERENCE METHOD UNCALIBRATED"


def _rate_ci(rejections: int, n_trials: int) -> dict:
    """Point rejection rate + 95% simulation CI (normal binomial approximation)."""
    rate = rejections / n_trials
    half = 1.96 * math.sqrt(max(rate * (1.0 - rate), 1e-12) / n_trials)
    return {
        "rate": rate,
        "ci95": [max(0.0, rate - half), min(1.0, rate + half)],
        "rejections": rejections,
        "trials": n_trials,
    }


def _mbb_bootstrap_means(d: np.ndarray, b: int, rng: np.random.Generator,
                         n_boot: int = BOOT_DRAWS) -> np.ndarray:
    """MBB resample means — MIRRORS the merged ``mbb_lower_bound`` resampling EXACTLY
    (same rng draw order), returning the full bootstrap-mean vector the BCa
    correction needs (the merged primitive only exposes a single quantile).

    A pinned test (``test_bootstrap_means_mirror_merged_primitive``) asserts that the
    ``alpha``-quantile of this vector equals ``mbb_lower_bound(...)`` under an identical
    seed, so the v6 bound provably reuses the v5 resampling rather than re-deriving it.
    """
    t_sessions = d.shape[0]
    if b > t_sessions:
        raise ValueError(f"block length {b} exceeds series length {t_sessions}")
    n_blocks = math.ceil(t_sessions / b)
    starts = rng.integers(0, t_sessions - b + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)
    return d[idx[:, :t_sessions]].mean(axis=1)


def block_jackknife_acceleration(d: np.ndarray, b: int) -> float:
    """BCa acceleration ``a`` from a DELETE-ONE-BLOCK jackknife over non-overlapping
    length-``b`` blocks tiling the series (respecting the MBB block dependence).

    ``a = sum_i (thbar - th_(i))^3 / (6 [sum_i (thbar - th_(i))^2]^{3/2})`` where
    ``th_(i)`` is the sample MEAN with block ``i`` deleted and ``thbar`` their average
    (Efron & Tibshirani 1993, eq. 14.15, with observations grouped into blocks so the
    influence estimate respects the dependence the MBB models). Returns 0.0 in the
    degenerate zero-dispersion case.
    """
    t_sessions = d.shape[0]
    n_blocks = math.ceil(t_sessions / b)
    total = float(d.sum())
    n = t_sessions
    theta_jack = np.empty(n_blocks)
    for i in range(n_blocks):
        lo = i * b
        hi = min((i + 1) * b, t_sessions)
        block_sum = float(d[lo:hi].sum())
        block_len = hi - lo
        theta_jack[i] = (total - block_sum) / (n - block_len)
    theta_bar = float(theta_jack.mean())
    diff = theta_bar - theta_jack
    denom = 6.0 * float(np.sum(diff ** 2)) ** 1.5
    if denom == 0.0:
        return 0.0
    return float(np.sum(diff ** 3) / denom)


def mbb_bca_lower_bound(d: np.ndarray, b: int, rng: np.random.Generator,
                        n_boot: int = BOOT_DRAWS, alpha: float = ALPHA, *,
                        means: np.ndarray | None = None) -> float:
    """One-sided lower 90% confidence bound for ``mean(d)`` via the BCa block
    bootstrap (§4.1 v6). See module docstring for the exact construction.

    ``means`` may be a precomputed bootstrap-mean vector (from
    :func:`_mbb_bootstrap_means` on the SAME series) so a paired percentile-vs-BCa
    comparison can share one resample; otherwise it is drawn here.
    """
    if means is None:
        means = _mbb_bootstrap_means(d, b, rng, n_boot)
    n_eff = means.shape[0]
    theta_hat = float(np.mean(d))
    prop_below = float(np.mean(means < theta_hat))
    # Clamp so z0 stays finite in the degenerate all-above / all-below cases.
    lo = 0.5 / n_eff
    prop_below = min(max(prop_below, lo), 1.0 - lo)
    z0 = _NORM.inv_cdf(prop_below)
    a = block_jackknife_acceleration(d, b)
    z_alpha = _NORM.inv_cdf(alpha)
    denom = 1.0 - a * (z0 + z_alpha)
    if denom == 0.0:
        alpha1 = alpha
    else:
        alpha1 = _NORM.cdf(z0 + (z0 + z_alpha) / denom)
    alpha1 = min(max(alpha1, lo), 1.0 - lo)
    return float(np.quantile(means, alpha1))


def type_i_calibrated(rate_result: dict) -> bool:
    """v6 §4.6-step-2 acceptance for ONE boundary cell: MC point estimate
    ``<= 0.10`` AND 95% CI upper ``<= 0.12``. Both bars are required so a
    borderline method cannot pass on Monte-Carlo noise."""
    return (rate_result["rate"] <= TYPE_I_REQUIREMENT
            and rate_result["ci95"][1] <= TYPE_I_CI_UPPER_MAX)


def v6_outcome(inference_calibrated: bool, power_feasible: bool) -> str:
    """v6 §4.6 verdict mapping. The method-calibration gate is evaluated FIRST: an
    uncalibrated bound invalidates the entire inference regardless of power."""
    if not inference_calibrated:
        return INFEASIBLE_INFERENCE
    if not power_feasible:
        return INFEASIBLE_NOISE
    return GO_VALID


def rejection_rate(n_trials: int, t_sessions: int, mean_bps: float, sigma_bps: float,
                   phi: float, b: int, threshold_bps: float, seed: int,
                   n_boot: int = BOOT_DRAWS, method: str = "bca") -> dict:
    """Fraction of trials whose one-sided 90% lower bound EXCEEDS ``threshold_bps``,
    under an AR(1) DGP with the given mean. ``method`` in {'percentile', 'bca'}.

    * boundary type-I: ``mean_bps = MEE``, ``threshold_bps = MEE``.
    * power:           ``mean_bps = PE``,  ``threshold_bps = MEE``.
    """
    rng = np.random.default_rng(seed)
    series = simulate_ar1(n_trials, t_sessions, mean_bps, sigma_bps, phi, rng)
    rejections = 0
    for i in range(n_trials):
        if method == "percentile":
            lb = mbb_lower_bound(series[i], b, rng, n_boot=n_boot, alpha=ALPHA)
        elif method == "bca":
            lb = mbb_bca_lower_bound(series[i], b, rng, n_boot, ALPHA)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown method: {method!r}")
        if lb > threshold_bps:
            rejections += 1
    return _rate_ci(rejections, n_trials)


def boundary_rejection_rates(n_trials: int, t_sessions: int, mean_bps: float,
                             sigma_bps: float, phi: float, b: int,
                             threshold_bps: float, seed: int,
                             n_boot: int = BOOT_DRAWS, alpha: float = ALPHA) -> dict:
    """PAIRED percentile-vs-BCa boundary rejection rates under one AR(1) DGP.

    Each trial draws ONE series and ONE MBB resample; BOTH bounds are read off the
    SAME bootstrap-mean vector (shared MC noise → a clean paired comparison of the
    two inference methods). Returns 'percentile' and 'bca' rate + 95% CI.
    """
    rng = np.random.default_rng(seed)
    series = simulate_ar1(n_trials, t_sessions, mean_bps, sigma_bps, phi, rng)
    rej_pct = 0
    rej_bca = 0
    for i in range(n_trials):
        means = _mbb_bootstrap_means(series[i], b, rng, n_boot)
        lb_pct = float(np.quantile(means, alpha))
        lb_bca = mbb_bca_lower_bound(series[i], b, rng, n_boot, alpha, means=means)
        if lb_pct > threshold_bps:
            rej_pct += 1
        if lb_bca > threshold_bps:
            rej_bca += 1
    return {
        "percentile": _rate_ci(rej_pct, n_trials),
        "bca": _rate_ci(rej_bca, n_trials),
        "t_sessions": t_sessions,
        "trials": n_trials,
    }


def validate_boundary_type_i(pilot: PilotNull = SYNTHETIC_DEFAULT, *,
                             trials: int = 8_000, boots: int = BOOT_DRAWS,
                             ts: tuple[int, ...] = (T_PRIMARY, T_MAX),
                             seed: int = 20260719) -> dict:
    """§4.6-step-2 (v6) VALIDATION — the load-bearing calibration evidence.

    Measures the boundary type-I at ``mu = MEE`` (rule ``LB > MEE``) for BOTH bounds,
    under an iid control (``phi = 0``) AND the pilot AR(1) null, at each T in ``ts``.
    The DGP marginal std is the pilot's marginal std (synthetic default = 25 bps,
    matching the #485/v5 convention); ``b`` is the frozen pilot block length. Fresh,
    per-cell deterministic seeds.

    The v6 gate: BCa CONTROLS boundary type-I iff EVERY cell has BCa point estimate
    ``<= 0.10`` AND CI upper ``<= 0.12``. Reports the percentile bound alongside so the
    reduction is explicit, and whether BCa alone suffices or a v7 studentized-t is
    required.
    """
    b = pilot.block_length
    sigma = pilot.marginal_sigma_bps
    dgps = {"iid": 0.0, "ar1": pilot.ar1_phi}

    grid: dict[str, dict] = {}
    for dgp_name, phi in dgps.items():
        dgp_offset = 0 if dgp_name == "iid" else 500_000
        for t_sessions in ts:
            cell_seed = seed + dgp_offset + t_sessions
            res = boundary_rejection_rates(trials, t_sessions, MEE_BPS, sigma, phi, b,
                                           MEE_BPS, cell_seed, n_boot=boots)
            reduction = res["percentile"]["rate"] - res["bca"]["rate"]
            grid[f"{dgp_name}_T{t_sessions}"] = {
                "dgp": dgp_name,
                "phi": phi,
                "t_sessions": t_sessions,
                "percentile": res["percentile"],
                "bca": res["bca"],
                "bca_reduces_type_i_by": reduction,
                "bca_cell_calibrated": type_i_calibrated(res["bca"]),
            }

    bca_controls = all(c["bca_cell_calibrated"] for c in grid.values())
    percentile_anticonservative = any(
        c["percentile"]["rate"] > TYPE_I_REQUIREMENT for c in grid.values())

    return {
        "prereg": "doc/experiments/2026-07-19-equal-weight-deployment-prereg-v6.md",
        "section": "4.6 step 2 (method calibration)",
        "protocol_version": "v6",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "frozen_parameters": {
            "bootstrap_draws": boots,
            "monte_carlo_trials": trials,
            "alpha_one_sided": ALPHA,
            "type_i_requirement": TYPE_I_REQUIREMENT,
            "type_i_ci_upper_max": TYPE_I_CI_UPPER_MAX,
            "mee_bps_per_session": MEE_BPS,
        },
        "dgp": {
            "marginal_sigma_bps": sigma,
            "phi_ar1": pilot.ar1_phi,
            "mbb_block_length": b,
            "source": pilot.source,
        },
        "cells": grid,
        "bca_controls_boundary_type_i": bca_controls,
        "percentile_anticonservative": percentile_anticonservative,
        "verdict": {
            "bca_controls": bca_controls,
            "note": (
                "BCa boundary type-I <= 0.10 (CI upper <= 0.12) in EVERY cell — the "
                "method is calibrated; proceed to the §4.6-step-3 power feasibility "
                "check." if bca_controls else
                "BCa still under-controls boundary type-I in at least one cell -> "
                "INFEASIBLE — INFERENCE METHOD UNCALIBRATED -> a v7 studentized-t "
                "block bootstrap is required (do NOT relax alpha / MEE)."),
        },
        "code_commit": _git_head(),
    }


def feasibility_preview(pilot: PilotNull = SYNTHETIC_DEFAULT, *, method: str = "bca",
                        trials: int = TRIALS, boots: int = BOOT_DRAWS,
                        t_primary: int = T_PRIMARY, t_max: int = T_MAX,
                        t_step: int = T_STEP, seed: int = 20260719) -> dict:
    """§4.6-step-3 power feasibility sweep with the chosen bound: is power ``>= 0.80``
    at ``mu = PE`` reachable within ``T <= t_max``? Returns the power curve + final_t.
    """
    b = pilot.block_length
    sigma = pilot.marginal_sigma_bps
    phi = pilot.ar1_phi
    power_by_t: dict[str, dict] = {}
    final_t: int | None = None
    for t_sessions in range(t_primary, t_max + 1, t_step):
        res = rejection_rate(trials, t_sessions, PE_BPS, sigma, phi, b, MEE_BPS,
                             seed + t_sessions, n_boot=boots, method=method)
        power_by_t[str(t_sessions)] = res
        if res["rate"] >= POWER_REQUIREMENT:
            final_t = t_sessions
            break
    return {
        "method": method,
        "power_requirement": POWER_REQUIREMENT,
        "power_at_pe_by_t": power_by_t,
        "final_t": final_t,
        "power_feasible": final_t is not None,
    }


def run_protocol_v6(pilot: PilotNull, seed: int, *, trials: int = TRIALS,
                    boots: int = BOOT_DRAWS, t_primary: int = T_PRIMARY,
                    t_max: int = T_MAX, t_step: int = T_STEP,
                    method: str = "bca") -> dict:
    """Full §4.6 v6 protocol on one pilot null.

    1. §4.6 step 2 (v6) — method calibration gate: boundary type-I at ``mu = MEE``
       (rule ``LB > MEE``) under the pilot null AND an iid control, at the binding
       few-blocks horizon ``t_primary``. BOTH cells must be ``<= 0.10`` with CI upper
       ``<= 0.12`` (:func:`type_i_calibrated`) or the method is uncalibrated.
    2. §4.6 step 3 — power at ``mu = PE`` (same bound), sweep T; stop at the first
       ``>= 0.80``.
    3. Emit the frozen outputs + the v6 verdict via :func:`v6_outcome`.
    """
    b = pilot.block_length
    sigma = pilot.marginal_sigma_bps
    phi = pilot.ar1_phi

    cal_pilot = rejection_rate(trials, t_primary, MEE_BPS, sigma, phi, b, MEE_BPS,
                               seed + 101, n_boot=boots, method=method)
    cal_iid = rejection_rate(trials, t_primary, MEE_BPS, sigma, 0.0, b, MEE_BPS,
                             seed + 202, n_boot=boots, method=method)
    inference_calibrated = type_i_calibrated(cal_pilot) and type_i_calibrated(cal_iid)

    power_by_t: dict[str, dict] = {}
    final_t: int | None = None
    for t_sessions in range(t_primary, t_max + 1, t_step):
        res = rejection_rate(trials, t_sessions, PE_BPS, sigma, phi, b, MEE_BPS,
                             seed + t_sessions, n_boot=boots, method=method)
        power_by_t[str(t_sessions)] = res
        if res["rate"] >= POWER_REQUIREMENT:
            final_t = t_sessions
            break
    power_feasible = final_t is not None
    reported_t = final_t if power_feasible else t_max
    power_at_reported = power_by_t[str(reported_t)]

    type_i = rejection_rate(trials, reported_t, MEE_BPS, sigma, phi, b, MEE_BPS,
                            seed - 1, n_boot=boots, method=method)

    outcome = v6_outcome(inference_calibrated, power_feasible)
    gate_pass = inference_calibrated and power_feasible
    activation_ready = bool(gate_pass and not pilot.is_synthetic)

    return {
        "prereg": "doc/experiments/2026-07-19-equal-weight-deployment-prereg-v6.md",
        "base_prereg": "doc/experiments/2026-07-17-equal-weight-deployment-prereg-v5.md",
        "materiality_rationale": "doc/experiments/2026-07-18-g1-mee-pe-rationale.md",
        "section": "4.6",
        "protocol_version": "v6",
        "inference_method": method,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "decision_rule": {
            "deployment_rule": f"one-sided 90% {method} MBB lower bound of mean(d_t) > MEE",
            "mee_bps_per_session": MEE_BPS,
            "pe_bps_per_session": PE_BPS,
            "type_i_null_mean_bps": MEE_BPS,
            "power_alt_mean_bps": PE_BPS,
        },
        "frozen_parameters": {
            "bootstrap_draws": boots,
            "alpha_one_sided": ALPHA,
            "monte_carlo_trials": trials,
            "t_primary": t_primary, "t_max": t_max, "t_step": t_step,
            "type_i_requirement": TYPE_I_REQUIREMENT,
            "type_i_ci_upper_max": TYPE_I_CI_UPPER_MAX,
            "power_requirement": POWER_REQUIREMENT,
        },
        "pilot_fit": pilot.to_dict(),
        "calibration_gate": {
            "pilot_null": {**cal_pilot, "calibrated": type_i_calibrated(cal_pilot)},
            "iid_control": {**cal_iid, "calibrated": type_i_calibrated(cal_iid)},
            "t_sessions": t_primary,
            "inference_calibrated": inference_calibrated,
        },
        "type_i": {**type_i, "requirement": TYPE_I_REQUIREMENT,
                   "ci_upper_max": TYPE_I_CI_UPPER_MAX, "null_mean_bps": MEE_BPS,
                   "calibrated": type_i_calibrated(type_i)},
        "power_at_pe_by_t": power_by_t,
        "power": {"requirement": POWER_REQUIREMENT, "final_t": final_t,
                  "at_pe": power_at_reported["rate"],
                  "ci95": power_at_reported["ci95"], "pass": power_feasible},
        "activation_simulation_block": {
            "inference_method": method,
            "type_i_rate": type_i["rate"],
            "type_i_ci_95": type_i["ci95"],
            "inference_calibrated": inference_calibrated,
            "power_at_pe": power_at_reported["rate"],
            "power_ci_95": power_at_reported["ci95"],
            "final_n_sessions": reported_t if power_feasible else None,
            "final_n_blocks": (reported_t // 20) if power_feasible else None,
            "mbb_block_length": b,
            "dgp_parameters": pilot.to_dict(),
            "pilot_manifest_digest": pilot.pilot_manifest_digest,
            "simulation_code_commit": _git_head(),
            "simulation_seed": seed,
        },
        "verdict": {
            "outcome": outcome,
            "inference_calibrated": inference_calibrated,
            "power_feasible": power_feasible,
            "gate_pass": gate_pass,
            "activation_ready": activation_ready,
            "note": (
                "SYNTHETIC DEFAULT — NOT VALID FOR ACTIVATION; a real fitted blinded "
                "pilot (§4.7) is required." if pilot.is_synthetic else
                ("Operating characteristics valid; activation may proceed once the "
                 "pilot manifest + this run are frozen in the activation commit."
                 if gate_pass else
                 ("INFEASIBLE — INFERENCE METHOD UNCALIBRATED: the BCa bound's "
                  "boundary type-I still exceeds the gate; escalate to a v7 "
                  "studentized-t block bootstrap. Do NOT relax alpha/MEE/block."
                  if not inference_calibrated else
                  "INFEASIBLE at the conservative pilot noise: power >= 0.80 at PE "
                  "unreachable within T <= 480. Do NOT walk MEE/PE/block; a "
                  "variance-reduction amendment needs a NEW pilot."))),
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
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--validate", action="store_true",
        help="Run the §4.6-step-2 (v6) calibration VALIDATION: percentile vs BCa "
             "boundary type-I under iid + AR(1) at T=240/480. If BCa controls it, "
             "also run the BCa power feasibility preview.")
    parser.add_argument(
        "--pilot-fit", type=Path, default=None,
        help="JSON with the sealed blinded-pilot fitted null (PilotNull). REQUIRED "
             "for an activation-valid run; omitted => synthetic default.")
    parser.add_argument(
        "--preview", action="store_true",
        help="1000-trial / 2000-boot preview (indicative, NOT the frozen config).")
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke run (NOT valid for activation).")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--boots", type=int, default=None)
    args = parser.parse_args()

    pilot = PilotNull.from_json(args.pilot_fit) if args.pilot_fit else SYNTHETIC_DEFAULT

    if args.quick:
        trials, boots = 60, 300
    elif args.preview:
        trials, boots = 1_000, 2_000
    else:
        trials, boots = TRIALS, BOOT_DRAWS
    if args.trials is not None:
        trials = args.trials
    if args.boots is not None:
        boots = args.boots

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.validate:
        val = validate_boundary_type_i(pilot, trials=trials, boots=boots,
                                       seed=args.seed)
        results = {"validation": val}
        print("=== §4.6 step 2 (v6) — boundary type-I: percentile vs BCa ===")
        for name, c in val["cells"].items():
            print(f"  {name:9s} pct={c['percentile']['rate']:.4f} "
                  f"CI{tuple(round(x, 4) for x in c['percentile']['ci95'])}  "
                  f"BCa={c['bca']['rate']:.4f} "
                  f"CI{tuple(round(x, 4) for x in c['bca']['ci95'])}  "
                  f"calibrated={c['bca_cell_calibrated']}")
        print(f"BCa controls boundary type-I (<=0.10, CI upper <=0.12): "
              f"{val['bca_controls_boundary_type_i']}")
        if val["bca_controls_boundary_type_i"]:
            feas_bca = feasibility_preview(pilot, method="bca", trials=trials,
                                           boots=boots, seed=args.seed)
            feas_pct = feasibility_preview(pilot, method="percentile", trials=trials,
                                           boots=boots, seed=args.seed)
            results["feasibility_bca"] = feas_bca
            results["feasibility_percentile"] = feas_pct
            print(f"BCa power feasible (>=0.80 @ PE, T<=480): "
                  f"{feas_bca['power_feasible']} final_t={feas_bca['final_t']}")
            results["resolved_outcome"] = (
                GO_VALID if feas_bca["power_feasible"] else INFEASIBLE_NOISE)
        else:
            results["resolved_outcome"] = INFEASIBLE_INFERENCE
            print("RESOLVED: INFEASIBLE — INFERENCE METHOD UNCALIBRATED (needs v7)")
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        return 0

    results = run_protocol_v6(pilot, args.seed, trials=trials, boots=boots)
    args.out.write_text(json.dumps(results, indent=2) + "\n")

    v = results["verdict"]
    cg = results["calibration_gate"]
    print(f"pilot: {pilot.source}")
    print(f"  phi={pilot.ar1_phi} marginal_sigma={pilot.marginal_sigma_bps:.2f}bps "
          f"b={pilot.block_length}")
    print(f"calibration (BCa boundary type-I @ mu=MEE, T={cg['t_sessions']}): "
          f"pilot={cg['pilot_null']['rate']:.4f} iid={cg['iid_control']['rate']:.4f} "
          f"calibrated={cg['inference_calibrated']}")
    print(f"final_T={results['power']['final_t']} "
          f"power_feasible={v['power_feasible']}")
    print(f"OUTCOME: {v['outcome']} (gate_pass={v['gate_pass']}, "
          f"activation_ready={v['activation_ready']})")
    return 0 if v["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
