"""Pre-activation type-I / power simulation — v7 (studentized-t block bootstrap).

Implements + EMPIRICALLY VALIDATES the v7 amendment
(``doc/experiments/2026-07-19-equal-weight-deployment-prereg-v7.md``) to the merged
v5 §4.6 pre-activation simulation. v7 REPLACES only the §4.1 inference method — the
one-sided 90% lower-bound computation — and STRENGTHENS the §4.6 method-calibration
gate per the codex review of the v6 amendment.

WHY v7 EXISTS. Two merged pre-activation sims proved the earlier bounds are
anti-conservative at the deployment boundary (rule ``LB90 > MEE``, boundary null
``mu = MEE``):

  * v5 percentile MBB lower bound (10th pct of bootstrap means): boundary type-I
    ~0.13-0.15 (T=240), ~0.12-0.13 (T=480) — RenQuant#511.
  * v6 BCa (bias-corrected + accelerated) block bootstrap: 0.119-0.151, essentially
    unchanged, because the flaw is VARIANCE-SCALE (a large block b over a short
    series under-estimates the sample mean's sampling variance with only ~T/b
    blocks), NOT the median-bias / skewness BCa corrects — RenQuant#513.

THE v7 BOUND (§4.1, replaced). A **studentized-t (bootstrap-t) block bootstrap**
lower bound. It rescales the statistic by a per-resample block-based standard error
so the pivot is (approximately) scale-free — the standard textbook fix for a
variance-scale coverage error:

  * resampling stays the moving-block bootstrap with the frozen v5 block length
    ``b = min(ceil(1.75*max_holding_days), 40)`` and B = 10,000 draws (the block
    resample draw structure is byte-identical to the merged v5/v6 primitive);
  * per-resample block-based standard error ``se*`` = the NON-OVERLAPPING batch-means
    (a.k.a. delete-one-block jackknife — algebraically identical for the mean)
    estimator: split the series into ``L = floor(T/b)`` consecutive length-``b``
    batches, ``se = sd(batch_means, ddof=1)/sqrt(L)``;
  * bootstrap the pivot ``t* = (mean* - mean)/se*`` over the B resamples, and take
    the one-sided lower bound ``LB = mean - quantile(t*, 1-alpha) * se`` where ``se``
    is the same batch-means SE on the ORIGINAL series.

STRENGTHENED §4.6 GATE (v7 §C, per the codex review of v6). Activation requires
PROVING, before any pilot, that the bound CONTROLS the boundary type-I across a
FROZEN DGP suite (iid, AR(1) persistence, volatility clustering, heavy tails, skew,
missing sessions), under a decision rule that actually controls the target rather
than a point estimate that can accept an over-alpha method:

  * retained bars (v6): MC point estimate ``<= 0.10`` AND 95% MC CI upper ``<= 0.12``;
  * ADDED (codex): a one-sided Clopper-Pearson 95% UPPER confidence bound on the
    true boundary type-I ``<= alpha = 0.10`` in EVERY frozen cell (conjunction =
    multiplicity control), at a FROZEN replication count with NO optional stopping.

THE v7 FINDING (asserted, not papered over). The studentized-t bound roughly HALVES
the excess — boundary type-I falls from ~0.15 (percentile) to ~0.11 — but STILL does
not clear ``<= 0.10`` at the binding few-blocks operating point (b=35 over T=240 is
only ~7 blocks; the SE uses L=6 batches). Every core cell fails both the retained and
the strengthened bar. The residual is invariant to the block-based SE estimator
(batch-means == delete-block jackknife; the overlapping-block LRV variant is worse),
so it is a genuine few-blocks finite-sample floor, not an SE-estimator artifact.
Verdict: ``INFEASIBLE — INFERENCE METHOD UNCALIBRATED`` → v8 needs a DIFFERENT pivot
or a small-sample calibration layer (e.g. a calibrated / double bootstrap), NEVER an
alpha / MEE / block relaxation.

Both bounds are kept side by side (``mbb_lower_bound`` = the merged v5 percentile
bound, reused as a positive control; ``studentized_block_lower_bound`` = the v7 bound)
so the calibration is COMPARED, never silently swapped.

All quantities are bps/session. Deterministic under ``--seed``. Research-only: no
production paths are read or written.

Usage (VALIDATION — the load-bearing calibration evidence, percentile vs studentized):
    python scripts/prereg_ew_power_simulation_v7.py --validate \
        --trials 5000 --out doc/experiments/<date>-ew-prereg-v7-validation.json

Usage (ACTIVATION — real sealed pilot fit, full v7 protocol):
    python scripts/prereg_ew_power_simulation_v7.py \
        --pilot-fit <sealed_pilot_fit.json> --seed 20260719 \
        --out doc/experiments/<date>-ew-prereg-v7-activation-sim.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Reuse the merged, version-agnostic numeric primitives: the AR(1) DGP and the
# percentile MBB lower bound (retained here as a positive control). The frozen
# simulation_code_commit pins the whole tree, so these imports are audit-safe.
from prereg_ew_power_simulation import mbb_lower_bound, simulate_ar1
# Reuse the v5 pilot-null contract, block-length rule, synthetic default and the
# frozen decision constants unchanged by v7.
from prereg_ew_power_simulation_v5 import (  # noqa: F401
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

# --- v7-only frozen constants ------------------------------------------------
# Retained v6 bar: the 95% MC-CI upper limit for the boundary type-I.
TYPE_I_CI_UPPER_MAX = 0.12
# Strengthened bar (codex review of v6): a one-sided Clopper-Pearson upper
# confidence bound on the TRUE boundary type-I must not exceed alpha. Confidence
# level of that one-sided bound:
CP_ACCEPT_CONF = 0.95
# Standard-error floor (bps/session) so a degenerate zero-dispersion resample
# keeps the studentized pivot finite. Continuous data essentially never hits it.
SE_FLOOR = 1e-9

# Outcomes (§4.6 verdict).
GO_VALID = "OPERATING_CHARACTERISTICS_VALID"
INFEASIBLE_NOISE = "INFEASIBLE AT CONSERVATIVE PILOT NOISE"
# The bound still under-controls boundary type-I -> uncalibrated inference -> v8.
INFEASIBLE_INFERENCE = "INFEASIBLE — INFERENCE METHOD UNCALIBRATED"

# Frozen DGP calibration suite (v7 §C). Each entry: (name, kind, phi). `kind`
# drives the innovation / volatility structure; `phi` the AR(1) persistence. The
# marginal std and block length come from the pilot fit. `iid` and `ar1` are the
# TASK-required core cells; the rest broaden the domain per the codex review
# (persistence, volatility clustering, heavy tails, skew, missing sessions).
DGP_SUITE: tuple[tuple[str, str, float], ...] = (
    ("iid", "gaussian", 0.0),
    ("ar1", "gaussian", 0.35),
    ("ar1_persist", "gaussian", 0.60),
    ("garch", "garch", 0.35),
    ("heavy_t5", "t5", 0.35),
    ("skew", "skew", 0.35),
    ("missing", "missing", 0.35),
)
CORE_DGP_NAMES = ("iid", "ar1")  # the task-required decisive cells


# --- DGP generators ----------------------------------------------------------
def _standardized_innovations(kind: str, shape: tuple[int, ...],
                              rng: np.random.Generator) -> np.ndarray:
    """Mean-0, unit-variance innovations of the requested marginal shape."""
    if kind in ("gaussian", "garch", "missing"):
        return rng.standard_normal(shape)
    if kind == "t5":
        # Student-t(5): var = nu/(nu-2) = 5/3; standardize to unit variance.
        return rng.standard_t(5, size=shape) / math.sqrt(5.0 / 3.0)
    if kind == "skew":
        # Centred unit exponential: mean 1, var 1, skew 2 -> shift to mean 0.
        return rng.standard_exponential(size=shape) - 1.0
    raise ValueError(f"unknown innovation kind: {kind!r}")


def _simulate_ar1_kind(n_trials: int, t_sessions: int, mean_bps: float,
                       sigma_bps: float, phi: float, kind: str,
                       rng: np.random.Generator, burn_in: int = 200) -> np.ndarray:
    """AR(1) with marginal std ``sigma_bps`` and innovation family ``kind``.

    Mirrors ``simulate_ar1`` (identical recursion + burn-in) but lets the
    innovation distribution vary so the calibration suite can probe heavy tails
    and skew. For ``kind='gaussian'`` this is numerically the merged
    ``simulate_ar1`` up to the innovation RNG call.
    """
    if not 0.0 <= phi < 1.0:
        raise ValueError(f"phi must be in [0, 1): {phi}")
    eps_sigma = sigma_bps * math.sqrt(1.0 - phi * phi)
    total = burn_in + t_sessions
    eps = _standardized_innovations(kind, (n_trials, total), rng) * eps_sigma
    out = np.empty((n_trials, total))
    out[:, 0] = rng.standard_normal(n_trials) * sigma_bps
    for t in range(1, total):
        out[:, t] = phi * out[:, t - 1] + eps[:, t]
    return out[:, burn_in:] + mean_bps


def _simulate_garch(n_trials: int, t_sessions: int, mean_bps: float,
                    sigma_bps: float, phi: float, rng: np.random.Generator,
                    burn_in: int = 300, omega_frac: float = 0.1,
                    alpha_arch: float = 0.1, beta_garch: float = 0.8) -> np.ndarray:
    """AR(1)-in-mean with GARCH(1,1) innovation variance (volatility clustering).

    The unconditional innovation variance is ``omega/(1-alpha-beta)``; ``omega`` is
    set so the AR(1) marginal std equals ``sigma_bps``. Volatility clusters via the
    GARCH recursion while the marginal scale matches the other cells (a like-for-like
    comparison that isolates clustering).
    """
    if alpha_arch + beta_garch >= 1.0:
        raise ValueError("GARCH not stationary")
    eps_var_uncond = (sigma_bps ** 2) * (1.0 - phi * phi)
    omega = eps_var_uncond * (1.0 - alpha_arch - beta_garch)
    total = burn_in + t_sessions
    z = rng.standard_normal((n_trials, total))
    eps = np.empty((n_trials, total))
    h = np.full(n_trials, eps_var_uncond)
    prev_eps2 = np.full(n_trials, eps_var_uncond)
    for t in range(total):
        h = omega + alpha_arch * prev_eps2 + beta_garch * h
        eps[:, t] = np.sqrt(h) * z[:, t]
        prev_eps2 = eps[:, t] ** 2
    out = np.empty((n_trials, total))
    out[:, 0] = rng.standard_normal(n_trials) * sigma_bps
    for t in range(1, total):
        out[:, t] = phi * out[:, t - 1] + eps[:, t]
    return out[:, burn_in:] + mean_bps


def _simulate_missing(n_trials: int, t_sessions: int, mean_bps: float,
                      sigma_bps: float, phi: float, rng: np.random.Generator,
                      miss_rate: float = 0.08) -> np.ndarray:
    """AR(1) Gaussian with ~``miss_rate`` of sessions missing at random (gaps).

    Models an incomplete paired series: dependence is broken across gaps. Generates
    a longer latent AR(1) path, drops sessions at random, and keeps the first
    ``t_sessions`` observed values so the analysis length is held fixed.
    """
    over = int(math.ceil(t_sessions / (1.0 - miss_rate))) + 64
    full = _simulate_ar1_kind(n_trials, over, 0.0, sigma_bps, phi, "gaussian", rng)
    keep = rng.random((n_trials, over)) >= miss_rate
    out = np.empty((n_trials, t_sessions))
    for i in range(n_trials):
        kept = full[i][keep[i]]
        if kept.shape[0] < t_sessions:  # pragma: no cover - rare short draw
            kept = full[i]
        out[i] = kept[:t_sessions]
    return out + mean_bps


def simulate_dgp(kind: str, n_trials: int, t_sessions: int, mean_bps: float,
                 sigma_bps: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Dispatch to the requested calibration-suite DGP (all matched to marginal
    ``sigma_bps`` so the cells differ only in dependence / tails / skew)."""
    if kind in ("gaussian",):
        return simulate_ar1(n_trials, t_sessions, mean_bps, sigma_bps, phi, rng)
    if kind in ("t5", "skew"):
        return _simulate_ar1_kind(n_trials, t_sessions, mean_bps, sigma_bps, phi,
                                  kind, rng)
    if kind == "garch":
        return _simulate_garch(n_trials, t_sessions, mean_bps, sigma_bps, phi, rng)
    if kind == "missing":
        return _simulate_missing(n_trials, t_sessions, mean_bps, sigma_bps, phi, rng)
    raise ValueError(f"unknown DGP kind: {kind!r}")


# --- the v7 studentized-t block bootstrap bound ------------------------------
def _batch_means_se(mat: np.ndarray, b: int) -> np.ndarray:
    """Non-overlapping batch-means SE of the mean along the LAST axis.

    Split each row into ``L = floor(T/b)`` consecutive length-``b`` batches (the
    final ``T mod b`` sessions are dropped from the SE only), take the L batch
    means, and return ``sd(batch_means, ddof=1)/sqrt(L)`` — the standard batch-means
    long-run-variance SE of a serially-dependent mean. For the mean this equals the
    delete-one-block jackknife SE over the same blocks. Requires ``L >= 2``.
    """
    t_sessions = mat.shape[-1]
    n_batches = t_sessions // b
    if n_batches < 2:
        raise ValueError(
            f"studentized bound needs >= 2 non-overlapping blocks: T={t_sessions}, "
            f"b={b} gives {n_batches}")
    trimmed = mat[..., :n_batches * b]
    batches = trimmed.reshape(*mat.shape[:-1], n_batches, b).mean(axis=-1)
    return batches.std(axis=-1, ddof=1) / math.sqrt(n_batches)


def _mbb_resample_matrix(d: np.ndarray, b: int, rng: np.random.Generator,
                         n_boot: int) -> np.ndarray:
    """(n_boot, T) moving-block bootstrap resamples — SAME draw structure as the
    merged v5/v6 MBB primitive (``rng.integers(0, T-b+1, (n_boot, n_blocks))``)."""
    t_sessions = d.shape[0]
    if b > t_sessions:
        raise ValueError(f"block length {b} exceeds series length {t_sessions}")
    n_blocks = math.ceil(t_sessions / b)
    starts = rng.integers(0, t_sessions - b + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)
    return d[idx[:, :t_sessions]]


def studentized_block_lower_bound(d: np.ndarray, b: int, rng: np.random.Generator,
                                  n_boot: int = BOOT_DRAWS,
                                  alpha: float = ALPHA) -> float:
    """One-sided lower ``1-alpha`` studentized-t (bootstrap-t) block-bootstrap bound
    for ``mean(d)`` (§4.1 v7). See the module docstring for the exact construction.

    ``LB = mean - quantile(t*, 1-alpha) * se_hat`` with ``t* = (mean* - mean)/se*``,
    ``se_hat`` / ``se*`` the non-overlapping batch-means SE on the original series /
    each resample. Quantile uses numpy's default linear interpolation (frozen).
    """
    theta_hat = float(np.mean(d))
    se_hat = max(float(_batch_means_se(d[None, :], b)[0]), SE_FLOOR)
    samples = _mbb_resample_matrix(d, b, rng, n_boot)
    theta_star = samples.mean(axis=1)
    se_star = np.maximum(_batch_means_se(samples, b), SE_FLOOR)
    t_star = (theta_star - theta_hat) / se_star
    q = float(np.quantile(t_star, 1.0 - alpha))
    return theta_hat - q * se_hat


# --- rates + acceptance ------------------------------------------------------
def _cp_upper(rejections: int, n_trials: int, conf: float = CP_ACCEPT_CONF) -> float:
    """One-sided Clopper-Pearson (exact) upper confidence bound for a binomial
    proportion. ``k`` successes in ``n`` trials -> upper ``conf`` bound via the
    Beta inverse CDF: ``BetaInv(conf, k+1, n-k)``; 1.0 when ``k == n``."""
    if rejections >= n_trials:
        return 1.0
    from scipy.stats import beta  # local import: only the gate needs scipy
    return float(beta.ppf(conf, rejections + 1, n_trials - rejections))


def _rate_ci(rejections: int, n_trials: int) -> dict:
    rate = rejections / n_trials
    half = 1.96 * math.sqrt(max(rate * (1.0 - rate), 1e-12) / n_trials)
    return {
        "rate": rate,
        "ci95": [max(0.0, rate - half), min(1.0, rate + half)],
        "cp95_upper": _cp_upper(rejections, n_trials),
        "rejections": rejections,
        "trials": n_trials,
    }


def type_i_accepted(rate_result: dict) -> tuple[bool, dict]:
    """v7 §4.6-step-2 acceptance for ONE boundary cell. Requires BOTH the retained
    v6 bars (point ``<= 0.10`` AND 95% MC CI upper ``<= 0.12``) AND the strengthened
    codex rule (one-sided CP-95% upper ``<= alpha``). Returns (accepted, detail)."""
    retained = (rate_result["rate"] <= TYPE_I_REQUIREMENT
                and rate_result["ci95"][1] <= TYPE_I_CI_UPPER_MAX)
    binom = rate_result["cp95_upper"] <= ALPHA
    detail = {"retained_bars": retained, "cp95_upper": rate_result["cp95_upper"],
              "binomial_accept": binom}
    return (retained and binom), detail


def v7_outcome(inference_calibrated: bool, power_feasible: bool) -> str:
    """§4.6 verdict mapping. The method-calibration gate is evaluated FIRST: an
    uncalibrated bound invalidates the entire inference regardless of power."""
    if not inference_calibrated:
        return INFEASIBLE_INFERENCE
    if not power_feasible:
        return INFEASIBLE_NOISE
    return GO_VALID


def rejection_rate(n_trials: int, t_sessions: int, mean_bps: float, sigma_bps: float,
                   phi: float, b: int, threshold_bps: float, seed: int,
                   n_boot: int = BOOT_DRAWS, method: str = "studentized",
                   dgp_kind: str = "gaussian") -> dict:
    """Fraction of trials whose one-sided 90% lower bound EXCEEDS ``threshold_bps``.

    ``method`` in {'percentile', 'studentized'}; ``dgp_kind`` selects the DGP family.
    boundary type-I: ``mean_bps = MEE, threshold = MEE``; power: ``mean_bps = PE,
    threshold = MEE``.
    """
    rng = np.random.default_rng(seed)
    series = simulate_dgp(dgp_kind, n_trials, t_sessions, mean_bps, sigma_bps, phi, rng)
    rejections = 0
    for i in range(n_trials):
        if method == "percentile":
            lb = mbb_lower_bound(series[i], b, rng, n_boot=n_boot, alpha=ALPHA)
        elif method == "studentized":
            lb = studentized_block_lower_bound(series[i], b, rng, n_boot, ALPHA)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown method: {method!r}")
        if lb > threshold_bps:
            rejections += 1
    return _rate_ci(rejections, n_trials)


def validate_boundary_type_i(pilot: PilotNull = SYNTHETIC_DEFAULT, *,
                             trials: int = TRIALS, boots: int = BOOT_DRAWS,
                             ts: tuple[int, ...] = (T_PRIMARY, T_MAX),
                             seed: int = 20260719,
                             suite: tuple[tuple[str, str, float], ...] = DGP_SUITE
                             ) -> dict:
    """§4.6-step-2 (v7) VALIDATION — the load-bearing calibration evidence.

    Boundary type-I at ``mu = MEE`` (rule ``LB > MEE``) for the studentized-t bound
    AND the percentile control, across the frozen DGP suite at each T in ``ts``.
    Fresh, per-cell deterministic seeds; FROZEN replication count (no optional
    stopping). The method is CALIBRATED iff EVERY cell is accepted (conjunction).
    """
    b = pilot.block_length
    sigma = pilot.marginal_sigma_bps

    grid: dict[str, dict] = {}
    dgp_offset = 0
    for dgp_name, kind, phi in suite:
        dgp_offset += 1_000_000
        for t_sessions in ts:
            cell_seed = seed + dgp_offset + t_sessions
            stud = rejection_rate(trials, t_sessions, MEE_BPS, sigma, phi, b, MEE_BPS,
                                  cell_seed, n_boot=boots, method="studentized",
                                  dgp_kind=kind)
            pct = rejection_rate(trials, t_sessions, MEE_BPS, sigma, phi, b, MEE_BPS,
                                 cell_seed + 7, n_boot=boots, method="percentile",
                                 dgp_kind=kind)
            accepted, detail = type_i_accepted(stud)
            grid[f"{dgp_name}_T{t_sessions}"] = {
                "dgp": dgp_name, "kind": kind, "phi": phi, "t_sessions": t_sessions,
                "core_cell": dgp_name in CORE_DGP_NAMES,
                "percentile": pct, "studentized": stud,
                "studentized_reduces_type_i_by": pct["rate"] - stud["rate"],
                "studentized_cell_accepted": accepted, "acceptance_detail": detail,
            }

    core = {k: v for k, v in grid.items() if v["core_cell"]}
    studentized_controls_core = all(v["studentized_cell_accepted"] for v in core.values())
    studentized_controls_all = all(v["studentized_cell_accepted"] for v in grid.values())
    percentile_anticonservative = any(
        v["percentile"]["rate"] > TYPE_I_REQUIREMENT for v in grid.values())
    worst = max(v["studentized"]["rate"] for v in grid.values())

    return {
        "prereg": "doc/experiments/2026-07-19-equal-weight-deployment-prereg-v7.md",
        "section": "4.6 step 2 (method calibration)",
        "protocol_version": "v7",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "frozen_parameters": {
            "bootstrap_draws": boots, "monte_carlo_trials": trials,
            "alpha_one_sided": ALPHA, "type_i_requirement": TYPE_I_REQUIREMENT,
            "type_i_ci_upper_max": TYPE_I_CI_UPPER_MAX,
            "cp_accept_conf": CP_ACCEPT_CONF, "mee_bps_per_session": MEE_BPS,
            "acceptance_rule": ("point <= 0.10 AND 95% MC-CI upper <= 0.12 AND "
                                "one-sided Clopper-Pearson 95% upper <= alpha, in "
                                "EVERY frozen DGP cell (conjunction); fixed N, no "
                                "optional stopping"),
        },
        "dgp": {"marginal_sigma_bps": sigma, "mbb_block_length": b,
                "suite": [{"name": n, "kind": k, "phi": p} for n, k, p in suite],
                "source": pilot.source},
        "cells": grid,
        "studentized_controls_core_cells": studentized_controls_core,
        "studentized_controls_all_cells": studentized_controls_all,
        "percentile_anticonservative": percentile_anticonservative,
        "worst_cell_studentized_type_i": worst,
        "verdict": {
            "studentized_controls": studentized_controls_all,
            "note": (
                "Studentized-t boundary type-I <= 0.10 (CI upper <= 0.12, CP-95% "
                "upper <= alpha) in EVERY frozen cell — the method is calibrated; "
                "proceed to the §4.6-step-3 power feasibility check."
                if studentized_controls_all else
                "Studentized-t still under-controls boundary type-I in at least one "
                "cell (it roughly halves the percentile excess, ~0.15 -> ~0.11, but "
                "does not clear <= 0.10 at the few-blocks operating point) -> "
                "INFEASIBLE — INFERENCE METHOD UNCALIBRATED -> v8 needs a different "
                "pivot or a small-sample calibration layer (do NOT relax "
                "alpha / MEE / block)."),
        },
        "code_commit": _git_head(),
    }


def feasibility_preview(pilot: PilotNull = SYNTHETIC_DEFAULT, *,
                        method: str = "studentized", trials: int = TRIALS,
                        boots: int = BOOT_DRAWS, t_primary: int = T_PRIMARY,
                        t_max: int = T_MAX, t_step: int = T_STEP,
                        seed: int = 20260719) -> dict:
    """§4.6-step-3 power feasibility sweep: is power ``>= 0.80`` at ``mu = PE``
    reachable within ``T <= t_max``? INFORMATIONAL when the bound is uncalibrated —
    power is not a valid feasibility verdict for an uncalibrated gate."""
    b = pilot.block_length
    sigma = pilot.marginal_sigma_bps
    phi = pilot.ar1_phi
    power_by_t: dict[str, dict] = {}
    final_t: int | None = None
    for t_sessions in range(t_primary, t_max + 1, t_step):
        res = rejection_rate(trials, t_sessions, PE_BPS, sigma, phi, b, MEE_BPS,
                             seed + t_sessions, n_boot=boots, method=method,
                             dgp_kind="gaussian")
        power_by_t[str(t_sessions)] = res
        if res["rate"] >= POWER_REQUIREMENT:
            final_t = t_sessions
            break
    return {"method": method, "power_requirement": POWER_REQUIREMENT,
            "power_at_pe_by_t": power_by_t, "final_t": final_t,
            "power_feasible": final_t is not None}


def run_protocol_v7(pilot: PilotNull, seed: int, *, trials: int = TRIALS,
                    boots: int = BOOT_DRAWS, t_primary: int = T_PRIMARY,
                    t_max: int = T_MAX, t_step: int = T_STEP,
                    method: str = "studentized") -> dict:
    """Full §4.6 v7 protocol on one pilot null: calibration gate FIRST (core iid +
    pilot-AR(1) cells at the binding horizon), then power, then the verdict."""
    b = pilot.block_length
    sigma = pilot.marginal_sigma_bps
    phi = pilot.ar1_phi

    cal_pilot = rejection_rate(trials, t_primary, MEE_BPS, sigma, phi, b, MEE_BPS,
                               seed + 101, n_boot=boots, method=method,
                               dgp_kind="gaussian")
    cal_iid = rejection_rate(trials, t_primary, MEE_BPS, sigma, 0.0, b, MEE_BPS,
                             seed + 202, n_boot=boots, method=method,
                             dgp_kind="gaussian")
    acc_pilot, det_pilot = type_i_accepted(cal_pilot)
    acc_iid, det_iid = type_i_accepted(cal_iid)
    inference_calibrated = acc_pilot and acc_iid

    power_by_t: dict[str, dict] = {}
    final_t: int | None = None
    for t_sessions in range(t_primary, t_max + 1, t_step):
        res = rejection_rate(trials, t_sessions, PE_BPS, sigma, phi, b, MEE_BPS,
                             seed + t_sessions, n_boot=boots, method=method,
                             dgp_kind="gaussian")
        power_by_t[str(t_sessions)] = res
        if res["rate"] >= POWER_REQUIREMENT:
            final_t = t_sessions
            break
    power_feasible = final_t is not None
    reported_t = final_t if power_feasible else t_max
    power_at_reported = power_by_t[str(reported_t)]

    type_i = rejection_rate(trials, reported_t, MEE_BPS, sigma, phi, b, MEE_BPS,
                            seed - 1, n_boot=boots, method=method, dgp_kind="gaussian")
    acc_ti, det_ti = type_i_accepted(type_i)

    outcome = v7_outcome(inference_calibrated, power_feasible)
    gate_pass = inference_calibrated and power_feasible
    activation_ready = bool(gate_pass and not pilot.is_synthetic)

    return {
        "prereg": "doc/experiments/2026-07-19-equal-weight-deployment-prereg-v7.md",
        "base_prereg": "doc/experiments/2026-07-17-equal-weight-deployment-prereg-v5.md",
        "section": "4.6", "protocol_version": "v7", "inference_method": method,
        "generated_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "decision_rule": {
            "deployment_rule": f"one-sided 90% {method} block-bootstrap lower bound of mean(d_t) > MEE",
            "mee_bps_per_session": MEE_BPS, "pe_bps_per_session": PE_BPS,
            "type_i_null_mean_bps": MEE_BPS, "power_alt_mean_bps": PE_BPS,
        },
        "frozen_parameters": {
            "bootstrap_draws": boots, "alpha_one_sided": ALPHA,
            "monte_carlo_trials": trials, "t_primary": t_primary, "t_max": t_max,
            "t_step": t_step, "type_i_requirement": TYPE_I_REQUIREMENT,
            "type_i_ci_upper_max": TYPE_I_CI_UPPER_MAX, "cp_accept_conf": CP_ACCEPT_CONF,
            "power_requirement": POWER_REQUIREMENT,
        },
        "pilot_fit": pilot.to_dict(),
        "calibration_gate": {
            "pilot_null": {**cal_pilot, "accepted": acc_pilot, "detail": det_pilot},
            "iid_control": {**cal_iid, "accepted": acc_iid, "detail": det_iid},
            "t_sessions": t_primary, "inference_calibrated": inference_calibrated,
        },
        "type_i": {**type_i, "requirement": TYPE_I_REQUIREMENT,
                   "ci_upper_max": TYPE_I_CI_UPPER_MAX, "null_mean_bps": MEE_BPS,
                   "accepted": acc_ti, "detail": det_ti},
        "power_at_pe_by_t": power_by_t,
        "power": {"requirement": POWER_REQUIREMENT, "final_t": final_t,
                  "at_pe": power_at_reported["rate"], "ci95": power_at_reported["ci95"],
                  "pass": power_feasible,
                  "note": ("INFORMATIONAL — not a valid feasibility verdict while the "
                           "bound is uncalibrated" if not inference_calibrated else "")},
        "verdict": {
            "outcome": outcome, "inference_calibrated": inference_calibrated,
            "power_feasible": power_feasible, "gate_pass": gate_pass,
            "activation_ready": activation_ready,
            "note": (
                "SYNTHETIC DEFAULT — NOT VALID FOR ACTIVATION; a real fitted blinded "
                "pilot (§4.7) is required." if pilot.is_synthetic else
                ("Operating characteristics valid; activation may proceed once the "
                 "pilot manifest + this run are frozen in the activation commit."
                 if gate_pass else
                 ("INFEASIBLE — INFERENCE METHOD UNCALIBRATED: the studentized-t "
                  "bound's boundary type-I still exceeds the gate; escalate to a v8 "
                  "pivot / calibration. Do NOT relax alpha/MEE/block."
                  if not inference_calibrated else
                  "INFEASIBLE at the conservative pilot noise: power >= 0.80 at PE "
                  "unreachable within T <= 480."))),
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
    parser.add_argument("--validate", action="store_true",
                        help="§4.6-step-2 (v7) calibration VALIDATION: percentile vs "
                             "studentized boundary type-I across the frozen DGP suite "
                             "at T=240/480, plus (if it controls) a power preview.")
    parser.add_argument("--core-only", action="store_true",
                        help="validate only the task-required iid + AR(1) core cells.")
    parser.add_argument("--pilot-fit", type=Path, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--quick", action="store_true")
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
        suite = tuple(s for s in DGP_SUITE if s[0] in CORE_DGP_NAMES) \
            if args.core_only else DGP_SUITE
        val = validate_boundary_type_i(pilot, trials=trials, boots=boots, seed=args.seed,
                                       suite=suite)
        results = {"validation": val}
        print("=== §4.6 step 2 (v7) — boundary type-I: percentile vs studentized-t ===")
        for name, c in val["cells"].items():
            tag = "CORE" if c["core_cell"] else "    "
            print(f"  [{tag}] {name:14s} pct={c['percentile']['rate']:.4f}  "
                  f"STUD={c['studentized']['rate']:.4f} "
                  f"CI{tuple(round(x, 4) for x in c['studentized']['ci95'])} "
                  f"CPup={c['studentized']['cp95_upper']:.4f} "
                  f"accept={c['studentized_cell_accepted']}")
        print(f"worst-cell studentized type-I = {val['worst_cell_studentized_type_i']:.4f}")
        print(f"studentized controls CORE cells: {val['studentized_controls_core_cells']}")
        print(f"studentized controls ALL cells:  {val['studentized_controls_all_cells']}")
        if val["studentized_controls_all_cells"]:
            feas = feasibility_preview(pilot, method="studentized", trials=trials,
                                       boots=boots, seed=args.seed)
            results["feasibility_studentized"] = feas
            print(f"studentized power feasible (>=0.80 @ PE, T<=480): "
                  f"{feas['power_feasible']} final_t={feas['final_t']}")
            results["resolved_outcome"] = (
                GO_VALID if feas["power_feasible"] else INFEASIBLE_NOISE)
        else:
            results["resolved_outcome"] = INFEASIBLE_INFERENCE
            print("RESOLVED: INFEASIBLE — INFERENCE METHOD UNCALIBRATED (needs v8)")
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        return 0

    results = run_protocol_v7(pilot, args.seed, trials=trials, boots=boots)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    v = results["verdict"]
    cg = results["calibration_gate"]
    print(f"pilot: {pilot.source}")
    print(f"calibration (studentized boundary type-I @ mu=MEE, T={cg['t_sessions']}): "
          f"pilot={cg['pilot_null']['rate']:.4f} iid={cg['iid_control']['rate']:.4f} "
          f"calibrated={cg['inference_calibrated']}")
    print(f"OUTCOME: {v['outcome']} (gate_pass={v['gate_pass']}, "
          f"activation_ready={v['activation_ready']})")
    return 0 if v["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
