# 2026-07-13 — G1 equal-weight deployment pre-registration (umbrella)

STATUS: RFC for review
WHAT: Pre-registration for equal-weight top-k deployment experiment,
      relocated to umbrella per codex review of orchestrator PR #509.
WHY: Cross-repo strategy specification (touches pipeline sizing,
     orchestrator scheduling, strategy config) belongs in the umbrella,
     not in a single subrepo.

## Changes from orchestrator #509

The original move addressed placement and artifact-definition feedback. The
Codex revision below corrects the remaining causal and inferential blockers;
this remains a paper-only pre-registration until the activation record is
frozen.

1. **DeMiguel citation** — reframed as motivation, not applicability
   evidence. Explicitly notes our setting differs (concentrated,
   conditional, dynamic k, exits/taxes/shared holdings).

2. **Exposure is identified separately from weighting** — section 3 defines
   A0 (observational status quo), A1 (conviction weights rescaled to the same
   exposure), and B (equal weights at that exposure). The primary comparison is
   B-A1; A1-A0 is exposure attribution, not a weighting result.

3. **Pre-registration fields** — section 7 adds typed YAML block with
   all required fields (timestamps, universe/data/feature/strategy
   digests, session calendar, regime source, initial cash/positions,
   shared marked-to-market state, block inference settings, and config digests).
   Marked TBD-AT-ACTIVATION.

4. **Risk gate algebra** — section 4.3 defines drawdown as positive loss
   (DD = abs(peak - trough) / peak), fixes the inequality to
   DD_B <= 1.2 * DD_A with worked example. Adds k < 4 floor:
   k_effective = max(k, 4), excess weight held as cash.

5. **Independent counterfactual ledgers** — section 6 defines the full
   artifact contract: immutable decision-time snapshot (shared) flowing
   into three independent ledger paths with separately cloned cash, positions,
   fills, exits, NAV/turnover/drawdown series. Typed artifacts with
   repo ownership specified. All arms begin from the same cash, lots,
   positions, pending orders, reservations, marks, and as-of time.

6. **Dependence-aware prospective inference** — section 4 makes a 20-session
   non-overlapping block the unit of inference, requires 12 prospective blocks,
   uses a fixed whole-block bootstrap, and treats all historical replay as
   diagnostic only. Regime claims require their own prospective block evidence.

Also addresses architectural placement: ownership map (section 5)
aligns with subrepo operating model (base-data = bars, strategy = policy,
pipeline = decision/sizing, execution = fills, orchestrator = scheduling,
umbrella = experiment spec).

## Operator review fixes (2026-07-13)

7. **Drift rebalance trigger** — section 3.2 now triggers a full rebalance
   when any name's weight breaches `1/k ± 0.5/k`. Without this, "equal weight"
   degrades into equal-initial-allocation with uncontrolled momentum exposure
   as winners grow and losers shrink.

8. **Matched cash rules (k < 4)** — both A1 and B now apply
   `k_effective = max(k, 4)` and hold identical residual cash. Previously
   arm B held cash while A1 could deploy full conviction weights, confounding
   the weighting comparison with a beta/exposure difference.

9. **Bayesian sequential interim analysis (§4.6)** — at 6 blocks (~6 months),
   an optional early GO/NO-GO via posterior `P(δ > MDE | data)` with frozen
   prior and stopping boundaries. Reduces expected wall-clock by ~50% in
   clear-signal cases without inflating the primary frequentist test.

10. **S-FRAC default + G7 tracking-error gate** — fractional shares now the
    default sizing mode. If S-FRAC is unavailable, whole-share floor with
    explicit per-block tracking error < 2% of target weight (gate G7).
    Prevents rounding artifacts from dominating the weighting signal when
    stock prices differ significantly.

## Codex round 2 fixes (2026-07-13)

11. **Calendar rebalance replaces drift band (§3.2)** — both arms rebalance
    on the SAME calendar sessions (every `rebalance_period=20` sessions +
    membership change + corporate action). Between rebalances, both arms
    drift freely. Eliminates the asymmetry where conviction concentrations
    in A1 would continuously breach a 1/k drift threshold, causing A1 to
    rebalance far more often than B and confounding the weighting comparison.

12. **Futility-only interim (§4.6)** — removed Early GO. The 12-block
    minimum (§4.4/G2) is the binding constraint for any GO decision. The
    interim can only stop the experiment early for futility (P < 0.05),
    not authorize deployment.

13. **Generic prior, D6 excluded (§4.6)** — prior calibrated from
    published cross-sectional sizing dispersion literature, NOT from the
    D6 replay (which is a discovery result). D6 evidence excluded from
    confirmation prior calibration, interim computation, and all
    confirmation statistics.

## Codex round 3 fixes (2026-07-13)

14. **Block statistic explicitly defined (§4.1)** — the block statistic is
    now formally defined as `S_j = (1/20) * sum(r_B_t - r_A1_t)` within
    each 20-session block, expressed in basis points per session. The MDE
    (3 bps/session) applies directly to this statistic — same units, same
    scale. G1 gate language updated to reference `mean(S_j)`. Drawdown and
    turnover summaries use the same daily returns aggregated at block level.

15. **Interim analysis removed entirely (§4.4, §4.6, §7)** — deleted the
    Bayesian futility stop. The prior selection (`σ_generic` from "generic
    literature") introduced a researcher degree of freedom at activation
    time — codex correctly identified that "generic literature" is not a
    frozen selection rule. The experiment now runs to 12 complete blocks
    with no interim analysis or early stopping. This is cleaner and
    eliminates all prior-sensitivity concerns. Pre-registration YAML
    updated: `early_stopping: none`.

## Codex round 4 fixes (2026-07-14)

16. **Unit chain made explicit (§2.1, §4.1, G1)** — the block statistic
    `S_j`, the grand mean `mean(S_j)`, the MDE, and the bootstrap bound
    are all now explicitly annotated as bps/session in every occurrence.
    §4.1 adds a "Units" callout listing all four quantities and their
    common scale. §2.1 clarifies that the block is the sampling unit but
    the statistic is per-session. G1 gate text includes the full
    definition chain.

17. **Prior sigma / Bayesian — already addressed in r3** — codex r4
    flagged `prior_sigma` as TBD-AT-ACTIVATION, but this was removed in
    the r3 fix (commit 3c5086d). The current doc has no Bayesian futility
    stop, no prior, and `early_stopping: none` in the pre-registration
    YAML (§7). §4.4 explicitly states "No interim analysis or early
    stopping is permitted."

## Files

- `doc/experiments/2026-07-13-equal-weight-deployment-prereg.md` (new)
- `doc/progress/2026-07-13-g1-prereg-umbrella.md` (this file)
