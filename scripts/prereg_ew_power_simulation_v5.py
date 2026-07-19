"""Pre-activation power / type-I simulation for the equal-weight deployment prereg — v5.

Implements §4.6 of
``doc/experiments/2026-07-17-equal-weight-deployment-prereg-v5.md`` (the v5
decision rule), with the MEE=3 / PE=6 materiality constants and rationale from
``doc/experiments/2026-07-18-g1-mee-pe-rationale.md`` (merged as RenQuant#500).

SUPERSEDES ``scripts/prereg_ew_power_simulation.py`` (the #485 sim), which
validated the v3 rule ("one-sided 90% MBB lower bound > MDE", with power sized
AT the MDE) and proved it STRUCTURALLY UNSATISFIABLE: power evaluated at
true-effect = MDE is ≈ α by one-sided-bound coverage (P(LB > θ | μ = θ) ≈ α),
so the v3 gate could not pass at its own design point. v5 fixes this by
separating the two operating characteristics:

  * DEPLOYMENT rule (decides — same FORM, new threshold constant): declare GO
    iff the one-sided 90% MBB lower bound of ``mean(d_t)`` EXCEEDS
    ``MEE = 3 bps/session``.
  * TYPE-I is evaluated at the LEAST-FAVOURABLE NULL ``μ = MEE`` (the boundary of
    the composite null H0: μ ≤ MEE). The empirical rejection rate must be
    ``<= 0.10``; the Monte-Carlo POINT ESTIMATE governs the gate and its 95%
    simulation CI is reported alongside (§4.6 step 2).
  * POWER is evaluated at the planning effect ``μ = PE = 6 bps/session`` (= 2×MEE),
    strictly ABOVE the rule threshold. Because the true effect exceeds the
    threshold, ``P(LB > MEE | μ = PE)`` is NOT coverage-capped and grows with T —
    this is precisely the v3 structural error removed (§4.6 step 3).

Because the null and the alternative are evaluated at DIFFERENT means (MEE vs
PE) while the rule threshold stays MEE, the same routine
(:func:`rejection_rate`, parameterised by ``threshold_bps``) serves both the v5
characteristics and the v3 differential (v3 = ``mean == threshold`` ⇒ ≈ α).

PILOT-CALIBRATED NULL — INPUT CONTRACT (§4.6 step 1 / §4.7). The DGP dependence
structure and a CONSERVATIVE upper-90%-CL long-run variance are FITTED FROM the
blinded calibration pilot (§4.7); this simulation is the piece the ACTIVATION
commit runs on the SEALED pilot output. The pilot data does not exist yet, so
those parameters are an EXPLICIT INPUT (:class:`PilotNull`, loaded from
``--pilot-fit`` JSON) with a documented SYNTHETIC DEFAULT for testing / preview
ONLY. A synthetic-default run is loudly marked ``NOT VALID FOR ACTIVATION``.

Sizing uses the conservative upper-90%-CL of the pilot LONG-RUN variance, never
a point estimate (§4.6 step 1). The AR(1) DGP's marginal variance is back-solved
so its long-run variance equals that conservative bound (see
:meth:`PilotNull.marginal_sigma_bps`).

INFEASIBILITY (§4.6 step 4). If power ``>= 0.80`` at PE cannot be reached within
``T <= 480`` under the conservative pilot variance, the recorded outcome is
``INFEASIBLE AT CONSERVATIVE PILOT NOISE``. No MEE / PE / block-length parameter
is walked to force feasibility — variance-reduction options are each a NEW
amendment with a NEW pilot.

FROZEN OUTPUTS (§4.6 step 5 / §7 yaml ``simulation:`` block). The activation
commit records: type-I rate + 95% CI, power at PE + 95% CI, final T, final
n_blocks, MBB block length ``b``, the pilot-fitted DGP parameters + pilot
manifest digest, and this script's git commit hash + seed. The frozen commit
hash pins the ENTIRE tree, including the two numeric primitives reused from the
#485 module below.

Sound, version-agnostic numeric primitives (AR(1) DGP + MBB lower bound) are
REUSED from the #485 harness; only the decision rule, the pilot input contract,
the two-stage output emission, and the infeasibility branch are new here.

All quantities are bps/session. Deterministic under ``--seed``. Research-only:
no production paths are read or written.

Usage (ACTIVATION — real sealed pilot fit):
    python scripts/prereg_ew_power_simulation_v5.py \
        --pilot-fit <sealed_pilot_fit.json> \
        --seed 20260718 \
        --out doc/experiments/<date>-ew-prereg-v5-activation-sim.json

Usage (PREVIEW / smoke — synthetic default, NOT for activation):
    python scripts/prereg_ew_power_simulation_v5.py --preview --out /tmp/preview.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Reuse the sound, version-agnostic numeric primitives from the #485 harness.
# (The frozen `simulation_code_commit` pins the whole tree, so this import is
# audit-safe: both files are captured by the recorded commit hash.)
from prereg_ew_power_simulation import mbb_lower_bound, simulate_ar1

# --- Frozen constants (v5) ---------------------------------------------------
MEE_BPS = 3.0                # §4.1 deployment eligibility threshold / rule
PE_BPS = 6.0                 # §4.1 planning effect for power (= 2 * MEE)
BOOT_DRAWS = 10_000          # §4.1 step 3 (frozen)
ALPHA = 0.10                 # one-sided 90% lower bound (§4.1 step 5)
TRIALS = 5_000               # §4.6 steps 2-3 (frozen)
T_PRIMARY = 240              # §4.4 minimum observation period
T_MAX = 480                  # §4.6 step 4 horizon cap
T_STEP = 20                  # §4.6 step 4 increment
TYPE_I_REQUIREMENT = 0.10    # §4.6 step 2
POWER_REQUIREMENT = 0.80     # §4.6 step 3
MBB_BLOCK_CAP = 40           # §4.1 step 2 / §4.7: b capped at 40 SESSIONS

INFEASIBLE = "INFEASIBLE AT CONSERVATIVE PILOT NOISE"  # §4.6 step 4 verbatim
GO_VALID = "OPERATING_CHARACTERISTICS_VALID"
TYPE_I_EXCEEDED = "TYPE_I_EXCEEDS_ALPHA"

_SYNTHETIC_DIGEST = "SYNTHETIC-DEFAULT-NOT-A-REAL-PILOT"


def mbb_block_length(max_holding_days: int) -> int:
    """v5 §4.1 step 2 / §4.7 rule: ``b = ceil(1.75 * max_holding_days)``, with the
    result **capped at 40 sessions**.

    NOTE — this differs from the #485 (v3) implementation, which capped
    ``max_holding_days`` at 40 BEFORE the multiply (giving b up to 70). v5 caps
    ``b`` itself at 40 sessions. For the default pilot holding period of 20
    sessions both agree (b = 35); they diverge once mhd > ~22.
    """
    if max_holding_days < 1:
        raise ValueError(f"max_holding_days must be >= 1: {max_holding_days}")
    return min(math.ceil(1.75 * max_holding_days), MBB_BLOCK_CAP)


@dataclass(frozen=True)
class PilotNull:
    """Fitted-null input contract consumed by the §4.6 simulation.

    These fields are produced by the BLINDED pilot fit (§4.7): the pilot sizing
    analysis sees a demeaned, arm-label-blinded daily series and may estimate
    ONLY dispersion and dependence. The activation commit passes the sealed fit
    here as ``--pilot-fit`` JSON.

    Fields:
      pilot_manifest_digest: SHA-256 of the prospective-only pilot session
        manifest (§4.7). Emitted verbatim into the frozen outputs.
      ar1_phi: fitted lag-1 dependence coefficient (AR(1) stand-in for the
        holding-overlap autocorrelation), in [0, 1).
      long_run_std_upper90_bps: the CONSERVATIVE upper-90%-CL of the pilot
        long-run STD (= sqrt of the long-run variance = sqrt(sum of
        autocovariances)) in bps/session. This — never a point estimate — is
        the sizing input (§4.6 step 1).
      max_holding_days: pilot-observed max holding period; drives the frozen
        block length via :func:`mbb_block_length`.
      source: provenance string; a real fit must NOT use the synthetic default.
    """

    pilot_manifest_digest: str
    ar1_phi: float
    long_run_std_upper90_bps: float
    max_holding_days: int
    source: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.ar1_phi < 1.0:
            raise ValueError(f"ar1_phi must be in [0, 1): {self.ar1_phi}")
        if not self.long_run_std_upper90_bps > 0.0:
            raise ValueError(
                f"long_run_std_upper90_bps must be > 0: "
                f"{self.long_run_std_upper90_bps}")
        if self.max_holding_days < 1:
            raise ValueError(
                f"max_holding_days must be >= 1: {self.max_holding_days}")

    @property
    def is_synthetic(self) -> bool:
        return self.pilot_manifest_digest == _SYNTHETIC_DIGEST

    @property
    def block_length(self) -> int:
        return mbb_block_length(self.max_holding_days)

    @property
    def marginal_sigma_bps(self) -> float:
        """Back-solve the AR(1) marginal std so the process's LONG-RUN variance
        equals the conservative ``long_run_std_upper90_bps ** 2``.

        AR(1) long-run variance ``LRV = sigma_marginal^2 * (1+phi)/(1-phi)``;
        inverting, ``sigma_marginal = long_run_std * sqrt((1-phi)/(1+phi))``.
        This makes the simulated CI width driven by the conservative long-run
        variance the pilot bounds, not by a marginal point estimate.
        """
        phi = self.ar1_phi
        return self.long_run_std_upper90_bps * math.sqrt((1.0 - phi) / (1.0 + phi))

    @classmethod
    def from_marginal(cls, sigma_marginal_bps: float, ar1_phi: float,
                      max_holding_days: int, *,
                      pilot_manifest_digest: str = _SYNTHETIC_DIGEST,
                      source: str = "synthetic-default") -> "PilotNull":
        """Construct from a MARGINAL std (the #485 convention), converting to the
        long-run-std contract. Used by the synthetic default and tests so the DGP
        is directly comparable to the #485 provisional parameterisation."""
        if not 0.0 <= ar1_phi < 1.0:
            raise ValueError(f"ar1_phi must be in [0, 1): {ar1_phi}")
        long_run_std = sigma_marginal_bps * math.sqrt((1.0 + ar1_phi) / (1.0 - ar1_phi))
        return cls(pilot_manifest_digest=pilot_manifest_digest, ar1_phi=ar1_phi,
                   long_run_std_upper90_bps=long_run_std,
                   max_holding_days=max_holding_days, source=source)

    @classmethod
    def from_json(cls, path: Path) -> "PilotNull":
        raw = json.loads(Path(path).read_text())
        try:
            return cls(
                pilot_manifest_digest=str(raw["pilot_manifest_digest"]),
                ar1_phi=float(raw["ar1_phi"]),
                long_run_std_upper90_bps=float(raw["long_run_std_upper90_bps"]),
                max_holding_days=int(raw["max_holding_days"]),
                source=str(raw.get("source", f"pilot-fit:{path}")),
            )
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"pilot-fit JSON missing required key: {exc}") from exc

    def to_dict(self) -> dict:
        return {
            "kind": "AR(1) matched to pilot conservative long-run variance",
            "pilot_manifest_digest": self.pilot_manifest_digest,
            "ar1_phi": self.ar1_phi,
            "long_run_std_upper90_bps": self.long_run_std_upper90_bps,
            "marginal_sigma_bps": self.marginal_sigma_bps,
            "max_holding_days": self.max_holding_days,
            "mbb_block_length": self.block_length,
            "source": self.source,
            "is_synthetic": self.is_synthetic,
        }


# Synthetic default — mirrors the #485 provisional parameterisation
# (marginal sigma_d = 25 bps, phi = 0.35, mhd = 20) so preview numbers are
# directly comparable. NOT a fitted pilot; any run using it is flagged
# NOT VALID FOR ACTIVATION.
SYNTHETIC_DEFAULT = PilotNull.from_marginal(
    sigma_marginal_bps=25.0,
    ar1_phi=0.35,
    max_holding_days=20,
    pilot_manifest_digest=_SYNTHETIC_DIGEST,
    source=("synthetic-default: mirrors #485 provisional (marginal sigma_d=25bps, "
            "phi=0.35, mhd=20) — NOT a fitted blinded pilot"),
)


def rejection_rate(n_trials: int, t_sessions: int, mean_bps: float,
                   sigma_bps: float, phi: float, b: int, threshold_bps: float,
                   seed: int, n_boot: int = BOOT_DRAWS) -> dict:
    """Fraction of trials whose one-sided 90% MBB lower bound EXCEEDS
    ``threshold_bps`` (§4.6 steps 2-3), under an AR(1) DGP with the given mean.

    * v5 type-I: ``mean_bps = MEE``, ``threshold_bps = MEE`` (boundary ⇒ ≈ α).
    * v5 power:  ``mean_bps = PE``,  ``threshold_bps = MEE`` (not capped).
    * v3 differential: ``mean_bps = threshold_bps`` (power sized at the rule ⇒ ≈ α).

    Returns the point-estimate rate plus a 95% simulation CI (normal
    approximation on the binomial proportion).
    """
    rng = np.random.default_rng(seed)
    series = simulate_ar1(n_trials, t_sessions, mean_bps, sigma_bps, phi, rng)
    rejections = 0
    for i in range(n_trials):
        if mbb_lower_bound(series[i], b, rng, n_boot=n_boot, alpha=ALPHA) > threshold_bps:
            rejections += 1
    rate = rejections / n_trials
    half = 1.96 * math.sqrt(max(rate * (1.0 - rate), 1e-12) / n_trials)
    return {
        "rate": rate,
        "ci95": [max(0.0, rate - half), min(1.0, rate + half)],
        "trials": n_trials,
        "rejections": rejections,
    }


def run_protocol(pilot: PilotNull, seed: int, *, trials: int = TRIALS,
                 boots: int = BOOT_DRAWS, t_primary: int = T_PRIMARY,
                 t_max: int = T_MAX, t_step: int = T_STEP) -> dict:
    """Full §4.6 v5 protocol.

    1. Sweep T = t_primary..t_max (step t_step); at each T estimate POWER at
       ``μ = PE`` with the deployment rule ``LB > MEE``. Stop at the first T with
       power ``>= 0.80`` (final T). If none reaches it, the horizon is infeasible.
    2. Estimate TYPE-I at the least-favourable null ``μ = MEE`` (rule ``LB > MEE``)
       at the reported horizon.
    3. Emit the frozen §4.6-step-5 outputs and the verdict. On infeasibility the
       outcome is INFEASIBLE AT CONSERVATIVE PILOT NOISE — nothing is walked.
    """
    b = pilot.block_length
    sigma = pilot.marginal_sigma_bps
    phi = pilot.ar1_phi

    power_by_t: dict[str, dict] = {}
    final_t: int | None = None
    for t_sessions in range(t_primary, t_max + 1, t_step):
        res = rejection_rate(trials, t_sessions, PE_BPS, sigma, phi, b,
                             MEE_BPS, seed + t_sessions, n_boot=boots)
        power_by_t[str(t_sessions)] = res
        if res["rate"] >= POWER_REQUIREMENT:
            final_t = t_sessions
            break

    feasible = final_t is not None
    reported_t = final_t if feasible else t_max
    power_at_reported = power_by_t[str(reported_t)]

    # Type-I at the least-favourable null, evaluated at the reported horizon.
    type_i = rejection_rate(trials, reported_t, MEE_BPS, sigma, phi, b,
                            MEE_BPS, seed - 1, n_boot=boots)
    type_i_pass = type_i["rate"] <= TYPE_I_REQUIREMENT

    if not feasible:
        outcome = INFEASIBLE
    elif not type_i_pass:
        outcome = TYPE_I_EXCEEDED
    else:
        outcome = GO_VALID
    gate_pass = feasible and type_i_pass
    activation_ready = bool(gate_pass and not pilot.is_synthetic)

    return {
        "prereg": "doc/experiments/2026-07-17-equal-weight-deployment-prereg-v5.md",
        "materiality_rationale": "doc/experiments/2026-07-18-g1-mee-pe-rationale.md",
        "section": "4.6",
        "protocol_version": "v5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "decision_rule": {
            "deployment_rule": "one-sided 90% MBB lower bound of mean(d_t) > MEE",
            "mee_bps_per_session": MEE_BPS,
            "pe_bps_per_session": PE_BPS,
            "type_i_null_mean_bps": MEE_BPS,   # least-favourable null (boundary)
            "power_alt_mean_bps": PE_BPS,      # planning effect (above threshold)
        },
        "frozen_parameters": {
            "bootstrap_draws": boots,
            "alpha_one_sided": ALPHA,
            "monte_carlo_trials": trials,
            "t_primary": t_primary, "t_max": t_max, "t_step": t_step,
        },
        "pilot_fit": pilot.to_dict(),
        "type_i": {**type_i, "requirement": TYPE_I_REQUIREMENT,
                   "null_mean_bps": MEE_BPS, "pass": type_i_pass},
        "power_at_pe_by_t": power_by_t,
        "power": {"requirement": POWER_REQUIREMENT, "final_t": final_t,
                  "at_pe": power_at_reported["rate"],
                  "ci95": power_at_reported["ci95"], "pass": feasible},
        # §4.6 step 5 / §7 yaml `simulation:` block — paste-ready for activation.
        "activation_simulation_block": {
            "type_i_rate": type_i["rate"],
            "type_i_ci_95": type_i["ci95"],
            "power_at_pe": power_at_reported["rate"],
            "power_ci_95": power_at_reported["ci95"],
            "final_n_sessions": reported_t if feasible else None,
            "final_n_blocks": (reported_t // 20) if feasible else None,
            "mbb_block_length": b,
            "dgp_parameters": pilot.to_dict(),
            "pilot_manifest_digest": pilot.pilot_manifest_digest,
            "simulation_code_commit": _git_head(),
            "simulation_seed": seed,
        },
        "verdict": {
            "outcome": outcome,
            "type_i_pass": type_i_pass,
            "power_feasible": feasible,
            "gate_pass": gate_pass,
            "activation_ready": activation_ready,
            "note": (
                "SYNTHETIC DEFAULT — NOT VALID FOR ACTIVATION; a real fitted "
                "blinded pilot (§4.7) is required." if pilot.is_synthetic else
                ("Operating characteristics valid; activation may proceed once "
                 "the pilot manifest + this run are frozen in the activation "
                 "commit." if gate_pass else
                 ("INFEASIBLE at the conservative pilot noise: power >= 0.80 at "
                  "PE unreachable within T <= 480. Do NOT walk MEE/PE/block; a "
                  "variance-reduction amendment needs a NEW pilot (§4.6 step 4)."
                  if not feasible else
                  "Type-I of the deployment rule exceeds alpha at the boundary; "
                  "recalibrate b / coverage before activation — do NOT walk "
                  "MEE/PE."))),
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
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pilot-fit", type=Path, default=None,
        help="JSON with the sealed blinded-pilot fitted null (PilotNull). "
             "REQUIRED for an activation-valid run; omitted => synthetic default.")
    parser.add_argument(
        "--preview", action="store_true",
        help="1000-trial / 2000-boot preview (statistically indicative, NOT "
             "the frozen 5000/10000 activation config).")
    parser.add_argument(
        "--quick", action="store_true",
        help="tiny smoke run (NOT valid for activation).")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--boots", type=int, default=None)
    args = parser.parse_args()

    if args.pilot_fit is not None:
        pilot = PilotNull.from_json(args.pilot_fit)
    else:
        pilot = SYNTHETIC_DEFAULT

    if args.quick:
        trials, boots = 40, 200
    elif args.preview:
        trials, boots = 1_000, 2_000
    else:
        trials, boots = TRIALS, BOOT_DRAWS
    if args.trials is not None:
        trials = args.trials
    if args.boots is not None:
        boots = args.boots

    results = run_protocol(pilot, args.seed, trials=trials, boots=boots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")

    v = results["verdict"]
    ti = results["type_i"]
    print(f"pilot: {pilot.source}")
    print(f"  phi={pilot.ar1_phi} long_run_std={pilot.long_run_std_upper90_bps:.2f}bps "
          f"marginal_sigma={pilot.marginal_sigma_bps:.2f}bps b={pilot.block_length}")
    print(f"type-I @ mu=MEE={MEE_BPS} (rule LB>MEE): {ti['rate']:.4f} "
          f"CI{tuple(round(x, 4) for x in ti['ci95'])} (req <= {TYPE_I_REQUIREMENT}) "
          f"pass={ti['pass']}")
    for t_sessions, res in results["power_at_pe_by_t"].items():
        print(f"power @ mu=PE={PE_BPS} T={t_sessions}: {res['rate']:.4f}")
    print(f"final_T={results['power']['final_t']} "
          f"power_feasible={v['power_feasible']}")
    print(f"OUTCOME: {v['outcome']} (gate_pass={v['gate_pass']}, "
          f"activation_ready={v['activation_ready']})")
    return 0 if v["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
