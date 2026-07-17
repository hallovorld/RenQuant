# Amendment v5: deployment materiality and calibration integrity

Date: 2026-07-17
Status: RFC amendment to v3/v4. v4 is retained as the record of a rejected
pre-activation rule; this document REPLACES its G1 decision rule and its
calibration language. No observation or capital deployment has been activated.

## 1. Disposition of v4

v4 correctly identified that a confidence-bound threshold equal to the planning
effect has no useful power at that effect. Its repair, `LB90 > 0`, made a
different error: it turned statistical positivity into the deployment rule.
That can authorize a positive but economically immaterial edge. It is rejected.

This is a deterministic paired prospective comparison, not a randomized causal
experiment. The inference below is evidence about a frozen time-series process;
it does not establish a causal claim beyond the specified counterfactual pair.

## 2. Frozen quantities and decisions

For every admitted session, `d_t` is the fully net, same-snapshot,
self-financing return difference (equal weight minus conviction), in bps per
session. Both arms use the same universe, timestamp, execution convention,
fees, slippage, financing treatment, and risk constraints. Missing, failed, or
asymmetrically generated arm outputs make the session inadmissible and are
reported; they are never imputed.

Before calibration, the base preregistration must give a written economic
rationale for:

- `MEE = 3 bps/session`: the minimum economically material net benefit.
- `PE = 6 bps/session`: the planning alternative, deliberately above MEE.

The numbers are not fitted from the paired data. If the economic rationale
cannot support both values, activation is refused and a new amendment is
required.

There are two distinct terminal statements at the fixed sample size `T`:

1. **Statistical efficacy:** report whether the one-sided 90% MBB lower bound
   for mean(`d_t`) exceeds zero.
2. **Deployment eligibility:** `GO` only if that same lower bound exceeds MEE.
   Otherwise the result is `NO-GO`; a statistically positive but sub-MEE result
   is explicitly `NO-GO (economically insufficient)`.

No verdict itself arms capital. A GO is only evidence eligible for the existing
operator-visible, separately governed deployment decision.

## 3. Calibration is a blinded pilot, not observation-free plumbing

At least 40 paired sessions may be collected solely for sizing the future
experiment. They are a named **calibration pilot**, are stored under a distinct
pilot manifest, and are permanently excluded from the terminal test and every
performance claim.

Before the pilot starts, universe, arm definitions, timing, costs, admissibility
rules, block-length rule, MEE, PE, and the terminal decision rule are frozen.
The sizing report receives an arm-label-blinded series with one global random
orientation; it may estimate dispersion and dependence but must not reveal the
mean, sign, cumulative PnL, arm-level returns, or a provisional verdict. The
unblinding key is sealed until the pilot report and activation commit are both
immutable. Any change to frozen inputs, access to unblinded pilot outcomes, or
pilot-data performance analysis invalidates the pilot for sizing and requires a
new registration.

The MBB block length is fixed before pilot collection as
`ceil(1.75 * max_holding_days)` for every later terminal analysis. It is not
tuned from 40 observations. Sizing uses a conservative upper 90% confidence
limit for the pilot long-run variance; the point estimate alone is forbidden.

## 4. Activation and terminal inference

The activation commit must contain the pilot manifest digest, the blinded
sizing program/version, variance upper bound, fixed block length, simulated
type-I and power tables, chosen `T`, and all input/configuration fingerprints.
It is valid only when a reproducible simulation demonstrates:

- type-I error no greater than 0.10 under `mu = MEE` for the deployment rule;
- power at least 0.80 under `mu = PE`; and
- `T <= 480` admitted sessions.

If those conditions fail, record `INFEASIBLE AT CONSERVATIVE PILOT NOISE`.
Do not relax MEE, change PE, retune blocks, or reuse the pilot silently. Any
variance-reduction or design change is a fresh amendment and pilot.

After activation, `T` is fixed. There are no efficacy looks, optional stopping,
or adaptive arm/universe/cost changes. One terminal MBB analysis is run; its
valid-session count, dropped-session reasons, pair-integrity ledger, full bound
curve, point estimate, turnover, costs, and all frozen fingerprints are
persisted in the verdict bundle. The terminal series excludes every pilot
session.

## 5. Acceptance before implementation

1. Update the frozen simulation to test the deployment rule `LB90 > MEE`, with
   size at MEE and power at PE; publish code, seed, and result digests.
2. Implement a blinded pilot collector, sealed-orientation handling, and
   immutable pilot/activation manifests; add tests proving pilot rows cannot
   enter the terminal series.
3. Implement terminal-session admissibility and pair-integrity accounting,
   including failure and missing-data tests.
4. Obtain independent review. This RFC neither starts measurement nor permits
   a production configuration change.
