# Pre-registration v5: equal-weight top-k deployment experiment (G1)

STATUS: RFC — start is TWO-STAGE (r2): the PILOT REGISTRATION commit
(§4.7) authorizes ONLY blinded pilot collection; no TERMINAL observation
begins until the ACTIVATION commit fills all TBD fields, freezes every
executable schema, and is pushed to main.
DATE: 2026-07-17 (v5). Drafted personally per design-review policy.
REPO: RenQuant (umbrella) — cross-repo strategy specification
PRIOR: D6 confirmatory replay REJECTED deployment governor (PBO 0.874)
SUPERSEDES: v2 (PR #465, CHANGES_REQUESTED), v3 (PR #480, merged then
REVERTED by #491 after the §4.6 simulation, PR #485, proved its G1 gate
structurally unsatisfiable), and the v4/v5 amendment drafts from earlier
rounds of this PR. This document is SELF-CONTAINED: it restates the full
protocol and references no document outside the tree. No observation or
capital deployment has been activated under any prior version.

**Version history (decision-rule lineage, kept for audit):**
- v3 gate: one-sided 90% MBB lower bound > MDE(3 bps). The #485
  simulation showed power at true-effect = MDE is structurally ≈ α
  (0.125–0.147 across T = 240…480) — coverage makes P(LB > θ_true) ≈ α.
  Unsatisfiable at its own design point; caught pre-activation.
- v4 repair: LB > 0 at α = 0.10, MDE demoted to power alternative.
  Rejected in review: statistical positivity is not a deployment
  criterion — it can authorize a positive but economically immaterial
  edge.
- v5 (this document): separates statistical efficacy (LB > 0, reported)
  from deployment eligibility (LB > MEE, decides), sizes power at a
  planning effect PE > MEE, and makes null calibration a blinded pilot
  with hard data-hygiene rules (§4.7).

This is a deterministic paired prospective comparison, not a randomized
causal experiment. The inference below is evidence about a frozen
time-series process; it does not establish a causal claim beyond the
specified counterfactual pair.

---

## 1. Motivation

The D6 confirmatory replay (orchestrator PR #466) rejected the deployment
governor family but produced one actionable finding: `equal_weight_top_k`
at its natural deployment level (~0.47) beat every governor arm by +9.3%
(annualized), Sharpe 0.53.

**Motivating literature.** DeMiguel, Garlappi & Uppal (2009), "Optimal
Versus Naive Diversification: How Inefficient is the 1/N Portfolio
Strategy?", *Review of Financial Studies* 22(5), showed 1/N outperformed
14 Markowitz-family optimizers out-of-sample when N was small and
estimation windows limited. Their result *motivates* equal-weighting but
does **not** directly apply here: our portfolio is concentrated,
conditional, dynamic top-k with Kelly sizing, exit rules, tax
constraints, and shared holdings. The 1/N result is an existence proof
that naive weights can dominate optimized weights under estimation
error — not that they do so in a portfolio with active admission/exit
and wash-sale constraints. This experiment tests the hypothesis
prospectively in our operating context.

The D6 result is a hypothesis, not a verdict: zero CHOPPY/BEAR sessions
in the eval window, +9.3% may be period-specific, and lower turnover
(not equal-weighting per se) may be the real driver.

## 2. Hypothesis

**H0 (null):** Conditional on the same selected names and
exposure-ratio-matched portfolio budgets, equal-weight top-k produces the
same or worse risk-adjusted net return as conviction-weighted sizing.

**H1 (alternative):** Equal-weight top-k produces higher risk-adjusted net
return, with non-degradation on drawdown.

### 2.1 Power considerations

Daily portfolio returns from overlapping holdings are not independent
observations. The INFERENTIAL series is the daily paired active return
`d_t` under the moving-block bootstrap (§4.1); predeclared
non-overlapping 20-session calendar blocks are DESCRIPTIVE reporting
units only, never the inferential basis. Every quantity in the decision
chain — the statistic, MEE, and PE — lives on one common scale: bps per
trading session.

Historical results, including D6, are hypothesis-generating diagnostics
only and never enter confirmation.

**Block dependence.** Holdings, state, and regime labels carry across
block boundaries; 20-session blocks are therefore NOT automatically
independent. Section 4.6 specifies a mandatory pre-activation simulation
that demonstrates the chosen inference method controls type-I error and
achieves adequate power under realistic dependence structure. If
12 blocks cannot achieve power >= 0.80 at the planning effect PE, the
horizon is raised before activation — not after — up to the T = 480 cap,
beyond which the honest outcome is infeasibility (§4.6 step 4).

## 3. Experimental arms

### 3.1 Arm definitions (fixed, no adaptive components)

| Arm | Label | Sizing rule |
|-----|-------|-------------|
| A0 | `status_quo_observational` | Current production sizing, retained only to measure the separate exposure intervention |
| A1 | `conviction_exposure_matched` | Current conviction/signal raw weights rescaled to the common target exposure ratio |
| B | `equal_weight_exposure_matched` | Equal weight across the same admitted names at the same common target exposure ratio |

All arms share the SAME admission chain, exit logic, k selection, and
candidate ranking. The primary estimand is `B - A1`: equal weights versus
conviction weights holding selected names and portfolio-level exposure
ratio fixed.

`A1 - A0` is reported separately as the exposure-normalisation effect;
it is not evidence that equal weights helped.

### 3.2 Self-financing rule and exposure matching

**Problem.** After the first session, A1 and B generate different fills,
costs, tax, and P&L. Their NAVs diverge. A common dollar exposure target
`E_t` is infeasible without capital transfers between arms, which would
violate self-financing. Therefore each arm targets a common exposure
**ratio**, not a common dollar amount.

**Common risk scalar `g_t`.** At each decision time `t`, a single
exogenous risk scalar `g_t` is computed from the frozen reference
snapshot:

    g_t = risk_engine(snapshot_t) / reference_NAV_t

where `risk_engine(snapshot_t)` returns the gross-exposure budget
permitted by the frozen risk engine after reserves, pending exits,
wash-sale holds, and margin requirements, and `reference_NAV_t` is the
reference-arm (A0) marked-to-market NAV at decision time. `g_t` is
computed ONCE, recorded in the shared snapshot, and consumed by both
arms. Neither arm's private state enters the `g_t` computation.

**Arm-specific dollar targets.** Each arm converts the common ratio to
its own dollar budget using its own pre-trade NAV:

    E_A1_t = g_t * NAV_A1_t
    E_B_t  = g_t * NAV_B_t

This is self-financing: each arm's dollar exposure is funded entirely
from its own NAV. No capital transfers, borrowing, or hidden reference
ledger is required. The ratio `g_t` is the controlled variable; the
dollar amounts differ by construction and are reported separately (see
gate G6).

**Arm-specific feasibility rules.** When `E_arm_t` exceeds the arm's
available capital (after reserves, unsettled cash, pending exits), the
arm scales to its own feasibility ceiling. A feasibility ceiling hit is
logged with the shortfall amount and the arm label. Both arms apply the
same feasibility logic; only the inputs (arm-specific NAV, cash, lots)
differ.

**Allocation within the budget:**

When `k >= 4`:
- **Arm B** allocates `E_B_t / k` per admitted name (equal weight).
- **Arm A1** normalizes its non-negative raw conviction weights to sum
  to `E_A1_t`.

When `k < 4`: both arms apply `k_effective = max(k, 4)`. B allocates
`E_B_t / k_effective` per admitted name; A1 normalizes conviction weights
to sum to `E_A1_t * k / k_effective`. The remaining exposure fraction
`(1 - k / k_effective)` is held as cash in BOTH arms. This prevents a
beta/exposure confound from polluting the B − A1 comparison. A0 retains
its production target only as an observational arm.

**Rebalance schedule (identical for A1 and B).** Both arms rebalance on
the SAME sessions, ensuring any return difference reflects only the
weight vector, never rebalance timing or frequency. Rebalance occurs
when ANY of the following triggers fires:

1. **Calendar rebalance:** every `rebalance_period` sessions (default 20,
   matching the block length). The rebalance calendar is frozen at
   activation; session indices are drawn from the pre-registered session
   calendar (§7).
2. **Membership change:** a name enters or exits the top-k set.
3. **Corporate action:** a split changes share count.

Between rebalance sessions, BOTH arms drift freely — no intra-period
correction for either arm. This eliminates the asymmetry a per-name
drift band would create: conviction concentrations in A1 would
continuously breach a 1/k-based drift threshold, causing A1 to
rebalance far more often than B and confounding the comparison with
turnover/timing effects.

**Rounding:** fractional shares (S-FRAC) by default. If S-FRAC is not
available at activation, whole-share floor `floor(target_dollar / price)`
is used with per-block tracking error reported (see gate G7).

**Residual cash:** held uninvested (NOT redistributed to largest-
remainder name). Available at next rebalance trigger.

**Corporate actions:** splits trigger rebalance; dividends add to cash.

**Costs and tax.** The cost and tax accounting MODEL is identical across
arms (same fee schedule, same FIFO lot selection, same wash-sale window).
However, because each arm executes different trades at different prices
and quantities, the REALIZED costs and tax amounts are arm-specific and
will generally differ. Per-arm cost and tax totals are reported separately
and enter the net return calculation for that arm only.

## 4. Evaluation protocol

### 4.1 Statistical test

**Block statistic (primary).** For each 20-session block `j`, compute
the arithmetic mean daily active return:

    S_j = (1/20) * sum_{t in block j}( r_B_t - r_A1_t )

where `r_B_t` and `r_A1_t` are the daily net-of-cost, net-of-tax
portfolio returns for arms B and A1 on session `t`. `S_j` is a mean
daily return, expressed in basis points per session (bps/session).

**Units:** every quantity in the decision chain is in bps/session:
- `S_j` (block statistic, DESCRIPTIVE) = arithmetic mean daily active
  return within block j (bps/session).
- `mean(S_j)` = grand mean of block statistics (bps/session),
  reported for transparency only.
- MEE = 3 bps/session (deployment eligibility threshold, §4.1 step 6).
- PE = 6 bps/session (planning effect for power, §4.6).
- The bootstrap operates on the DAILY series `{d_1, …, d_T}` and
  produces a confidence bound in the same bps/session unit.

**Inference method: moving-block bootstrap.**

The primary inference uses a moving-block bootstrap (MBB) on the daily
paired active returns `d_t = r_B_t - r_A1_t`, NOT a naive percentile
bootstrap on the 12 pre-cut block summaries. The naive approach treats
blocks as independent draws; because holdings and state carry across
block boundaries, this assumption is violated and the resulting
confidence interval has unknown coverage.

MBB procedure (frozen):
1. Compute the daily paired active return series
   `d_1, d_2, …, d_T` (T = total sessions).
2. Use the FROZEN numeric MBB block length `b` from the activation
   commit — computed at activation via the registered rule
   `b = ceil(1.75 * max_holding_days)` (capped at 40 sessions) applied
   to PILOT-observed holding periods (§4.7), validated by the §4.6
   simulation, and used here VERBATIM. No re-derivation from
   terminal-window data: `max_holding_days` observed during the terminal
   evaluation never enters this analysis (an analysis-time b would be a
   researcher degree of freedom at verdict time). The rule ensures
   blocks span the dominant autocorrelation induced by overlapping
   holdings.
3. Draw 10,000 bootstrap samples: each sample draws `ceil(T/b)`
   overlapping blocks of length `b` with replacement from positions
   `{1, …, T-b+1}`, concatenates, and trims to length `T`.
4. For each bootstrap sample, compute the mean daily active return.
5. The one-sided 90% lower confidence bound is the 10th percentile of
   the 10,000 bootstrap means.
6. **Terminal statements (two, distinct — v5):**
   - **Statistical efficacy (reported):** whether the lower bound
     exceeds **0** (test of H0: μ ≤ 0 at α = 0.10).
   - **Deployment eligibility (decides):** **GO** only if that same
     lower bound exceeds **MEE = 3 bps/session**. A statistically
     positive but sub-MEE result is explicitly
     `NO-GO (economically insufficient)`.

The 12-block summaries `{S_1, …, S_12}` are reported as descriptive
diagnostics (per-block effect size, within-block variance) but are NOT
the inferential basis.

**Materiality constants (frozen, with required economic rationale):**

- **MEE = 3 bps/session** (~7.5% annualized): the minimum economically
  material net benefit. Below this, the status quo's track record and
  operational familiarity dominate the operational cost of change.
- **PE = 6 bps/session**: the planning alternative at which the design
  must demonstrate power ≥ 0.80 — deliberately ABOVE MEE so the test is
  sized to detect an effect worth acting on, not the bare materiality
  threshold (sizing power AT the pass line is the v3 structural error;
  see version history).

Neither number is fitted from paired data. Before the calibration pilot
(§4.7) starts, the activation owner must record a written economic
rationale for both values; if the rationale cannot support them,
activation is refused and a new amendment is required. No verdict itself
arms capital: a GO is only evidence eligible for the existing
operator-visible, separately governed deployment decision (§9).

### 4.2 Regime results (DESCRIPTIVE ONLY)

All regime-stratified results in this experiment are **descriptive** and
**hypothesis-generating**. They are NOT confirmatory evidence and do NOT
support regime-specific deployment claims.

Specifically:
- Each block receives the regime label prevailing at its first decision
  session and its within-block regime composition.
- Per-regime summaries (mean `S_j`, count of blocks, exposure ratio) are
  reported for transparency.
- A regime with fewer than 4 complete blocks OR less than 75% session
  purity within its blocks is labeled `NOT_ESTABLISHED_IN_REGIME`.
- Even regimes meeting the 4-block/75% threshold are reported as
  DESCRIPTIVE observations, not confirmatory results. Four blocks cannot
  provide adequate power for a regime-specific claim.
- Any future regime-specific deployment claim requires a SEPARATE
  pre-registration with its own multiplicity-controlled hypothesis family,
  independently powered sample size, and prospective data collection.

The go/no-go verdict (§4.3) is based SOLELY on the pooled (all-regime)
primary test. No regime breakdown can upgrade a pooled NO-GO to GO or
downgrade a pooled GO to NO-GO.

### 4.3 Go/no-go decision rule (frozen before evaluation)

**GO** requires ALL gates to pass:

| Gate | Criterion | Rationale |
|------|-----------|-----------|
| G1 | Deployment eligibility: the one-sided 90% MBB lower bound for `mean(d_t)` exceeds MEE = 3 bps/session; statistical efficacy (LB > 0) is reported alongside but does not by itself pass the gate (§4.1 step 6) | Economic materiality after dependence-aware inference; positivity alone cannot authorize deployment |
| G2 | At least T completed admitted sessions (T fixed at activation from §4.6, 240 ≤ T ≤ 480); the pre-activation simulation demonstrated type-I ≤ 0.10 under μ = MEE for the deployment rule AND power ≥ 0.80 under μ = PE, using the pilot-calibrated null (§4.7) | Adequate statistical properties confirmed before observation, sized at the correct alternative |
| G3 | DD_B <= 1.2 * DD_A1 (drawdown as positive loss, see note) | Non-degradation on tail risk at matched exposure |
| G4 | Mean block turnover of B <= A1 and costs are fully reconciled per arm | Prevent a gross-return-only win |
| G5 | No single-name concentration > 1/k at any rebalance point; hard floor k >= 4 (cap <= 25%); BOTH arms apply same k_effective and cash rule (§3.2) | Safety cap; identical exposure ratio prevents beta confound |
| G6 | A1 - A0 reported separately with per-arm exposure, turnover, cost, tax, and NAV attribution; both gross ratio g_t and dollar exposure per arm logged | Do not mislabel an exposure effect as a weighting effect |
| G7 | If S-FRAC disabled: mean per-name tracking error from rounding < 2% of target weight across the evaluation period | Prevent rounding artifacts from dominating the weighting signal |

**Drawdown sign convention (G3):** drawdown is defined as a POSITIVE loss:
DD = abs(peak_value - trough_value) / peak_value. Both DD_A1 and DD_B are
non-negative. The gate DD_B <= 1.2 * DD_A1 correctly requires that arm B's
maximum loss does not exceed 120% of arm A's maximum loss.

**Concentration cap and matched cash (G5):** when k < 4, the equal-weight
allocation 1/k > 25%. Both arms apply `k_effective = max(k, 4)` and hold
identical residual cash fractions (section 3.2). This prevents a single-name
concentration breach AND prevents the B − A1 comparison from measuring
a beta/exposure difference instead of a weighting effect.

**NO-GO** if any gate fails. No re-tuning, no "close enough" exceptions.

### 4.4 Minimum observation period

- Shadow: exactly T admitted sessions (T fixed at activation from §4.6,
  240 ≤ T ≤ 480) for the primary decision. Missing, failed, or
  asymmetrically generated arm outputs make a session inadmissible; it
  is reported with its reason and never imputed.
- Historical replay: any duration, explicitly diagnostic-only and excluded
  from all go/no-go statistics.
- No interim analysis, efficacy looks, optional stopping, or adaptive
  arm/universe/cost changes after activation. One terminal MBB analysis
  is run at T. This eliminates researcher degrees of freedom in prior
  selection, stopping-boundary calibration, and interim timing.

### 4.5 Cost assumptions (frozen)

| Parameter | Value | Source |
|-----------|-------|--------|
| Base transaction cost | 5 bps round-trip | Existing sim |
| Adverse selection | 2x base for names with daily volume < $50M | Existing sim |
| Tax rate (short-term) | 50% | Existing convention |
| Tax rate (long-term) | 32% | Existing convention |
| Slippage model | Existing sim infrastructure | No change |

These parameters are identical across arms. Realized per-arm costs and
tax differ because trades differ (§3.2).

### 4.6 Pre-activation simulation (power and type-I validation)

**Purpose.** After the pilot is sealed and before the first TERMINAL
observation, demonstrate that the chosen inference method (MBB, §4.1)
controls the false-positive rate of the DEPLOYMENT rule and has adequate
power at the planning effect under realistic dependence — the simulation
is fitted FROM the sealed pilot (§4.7), so it necessarily runs between
the two stages. If this simulation fails, the recorded
outcome is `INFEASIBLE AT CONSERVATIVE PILOT NOISE` — the experiment
does not start with unvalidated operating characteristics, and no
parameter is walked to force feasibility.

**Simulation design (v5):**

1. **Data-generating process — pilot-calibrated only.** Generate
   synthetic paired daily active returns `d_t` with autocorrelation and
   dispersion fitted from the BLINDED calibration pilot (§4.7). Sizing
   uses the conservative upper 90% confidence limit of the pilot
   long-run variance, never the point estimate. Historical replays (D6
   or other) may inform qualitative structure checks but are FORBIDDEN
   as the quantitative DGP source — the #485 run showed provisional
   parameterization (σ_d = 25 bps from ~4 unblinded pairs) swings the
   required T from ~234 to ~649 sessions; feasibility must rest on
   measured, blinded noise.

2. **Null simulation (type-I of the deployment rule).** Set the true
   mean of `d_t` to **MEE** (the deployment rule's null boundary,
   H0: μ ≤ MEE). Run 5,000 Monte Carlo trials of length T. For each,
   apply the full MBB inference (§4.1) and record whether LB > MEE. The
   empirical rejection rate must be ≤ 0.10 — the Monte Carlo POINT
   ESTIMATE governs the gate; its 95% simulation confidence interval is
   reported alongside so borderline calibration is visible.

3. **Alternative simulation (power).** Set the true mean of `d_t` to
   **PE = 6 bps/session**. Run 5,000 Monte Carlo trials under the same
   dependence structure. The empirical rejection rate (power) must be
   ≥ 0.80. Report the exact power with a 95% simulation confidence
   interval.

4. **Horizon adjustment.** If power < 0.80 at T = 240 sessions:
   increase T in 20-session increments until power ≥ 0.80 or T reaches
   480 sessions (24 blocks). If T = 480 cannot achieve power ≥ 0.80 at
   PE under the conservative pilot variance, the experiment is NOT
   activated; the recorded outcome is `INFEASIBLE AT CONSERVATIVE PILOT
   NOISE`. Variance-reduction options (longer rebalance interval,
   paired-difference denoising, larger MEE/PE with fresh economic
   rationale) are each a NEW amendment with a NEW pilot — never a
   silent parameter walk, and never a reuse of the invalidated pilot.

5. **Frozen outputs.** The simulation produces and freezes:
   - Empirical type-I rate (deployment rule, μ = MEE) and 95% CI.
   - Empirical power at PE and 95% CI.
   - Final T (number of sessions).
   - Final n (number of blocks = T / 20).
   - MBB block length `b` used in the simulation.
   - The pilot-fitted DGP parameters (dependence structure, conservative
     variance upper bound) + the pilot manifest digest.
   - The simulation code commit hash and seed.

These outputs are recorded in the activation commit (§7) and are
prerequisites for activation.

### 4.7 Calibration pilot (blinded) and data hygiene — HARD activation prerequisites

**Two-stage start (r2 — resolves the pilot/activation deadlock).** The
protocol starts in two explicitly ordered commits:

1. **PILOT REGISTRATION commit**: freezes this document, the frozen
   inputs listed below, the burned-sessions manifest (hygiene rule 1),
   and the pilot scope (epoch, target n ≥ 40). It authorizes ONLY
   blinded pilot collection — no terminal observation, no capital. The
   burn boundary "pre-freeze" is DEFINED as this commit's timestamp;
   pilot sessions must strictly POSTDATE it (the pilot manifest is
   prospective-only, it can never admit an already-collected session).
2. **ACTIVATION commit** (§7): fills every TBD — including the §4.6
   outputs, which are computed FROM the sealed pilot — and starts the
   terminal series. The header's "no observation before activation"
   applies to the TERMINAL series; the pilot is authorized by commit 1
   alone. There is no state in which either series runs unregistered.

**Pilot definition.** At least **40 paired sessions** are collected from
the two-arm harness solely to size the future experiment. They are a
named **calibration pilot**: stored under a distinct pilot manifest and
**permanently excluded from the terminal test and every performance
claim**. The pilot is observation FOR SIZING, not "observation-free
plumbing" (the v4 language claiming otherwise is repudiated) — which is
exactly why it is blinded:

- At pilot registration, universe, arm definitions, timing, costs,
  admissibility rules, the MBB block-length RULE, MEE, PE, and the
  terminal decision rule are frozen (this document + the registration
  commit).
- The sizing analysis receives a **DEMEANED, arm-label-blinded** series:
  the per-series sample mean is removed before delivery and one global
  random orientation is applied, so neither the sign NOR the magnitude
  of the mean can leak through the sizing path; it may estimate
  dispersion and dependence only, and must not reveal cumulative PnL,
  arm-level returns, or any provisional verdict. The unblinding key is
  sealed until the pilot report and activation commit are both
  immutable.
- The MBB block length RULE `b = ceil(1.75 × max_holding_days)` (cap 40)
  is fixed at pilot registration; the NUMERIC b is computed from
  pilot-observed holding periods, frozen in the activation commit, used
  by the §4.6 simulation, and the terminal analysis uses that frozen b
  VERBATIM — no re-derivation from terminal-window data (this binds the
  simulation-validated b to the realized analysis).
- Any change to frozen inputs, any access to unblinded pilot outcomes,
  or any performance analysis of pilot data invalidates the pilot for
  sizing and requires a new registration and a new pilot.

**Data-hygiene exclusions (frozen; violations invalidate the affected
series):**

1. **Pre-freeze sessions are burned.** Every paired session observed
   BEFORE the pilot-registration commit — including the ~4 unblinded
   pairs whose σ_d informed the #485 provisional parameterization, and
   every session already analyzed in any memo — is permanently excluded
   from BOTH the calibration pilot AND the terminal series. The burned
   set is ENUMERATED in a committed **burned-sessions manifest** at
   pilot registration; the manifest, not prose, is the auditable
   object ("analyzed in any memo" is resolved once, at registration,
   by listing the sessions). Analyzed data cannot re-enter as if
   unseen.
2. **No cross-epoch pooling, and pilot epoch = terminal epoch.** The
   two-arm harness freezes a pin manifest per epoch (e.g. epoch-3 was
   refrozen as epoch-4 on 2026-07-17 after the GOAL-5 deployments
   changed 9 pins). Sessions from different epochs are draws from
   DIFFERENT processes and may never be pooled — not in the pilot fit,
   not in the terminal series. Additionally the TERMINAL series must
   run in the SAME epoch the pilot was collected in: a pin refreeze
   between pilot completion and terminal start invalidates the sizing
   (T, power, type-I were fitted to a dead process) and requires a new
   pilot in the new epoch. The activation commit records the epoch_id
   and is valid only while that epoch is current.
3. **Epoch break mid-series.** A pin/deployment change that re-freezes
   the harness during pilot collection restarts the pilot (counter to
   zero, prior pilot sessions retained only as a labeled archive). An
   epoch break during the terminal window makes all subsequent sessions
   inadmissible under the frozen registration: the terminal analysis
   either completes on the pre-break admitted sessions IF T was already
   reached, or the experiment is recorded as
   `INVALIDATED (epoch break at session <t>)` and requires
   re-registration. No exception path.
4. **Mechanical checkability.** The harness telemetry reports
   `n_paired_sessions` per epoch and per role (pilot / terminal /
   burned), so the ≥40 prerequisite and every exclusion above are
   auditable at activation time from the recorded manifests alone.

## 5. Ownership map

Cross-repo responsibilities per the subrepo operating model:

| Component | Owner repo | Responsibility |
|-----------|-----------|----------------|
| Immutable bars + watermarks | base-data | Supplies OHLCV + fundamentals; no universe/strategy logic |
| Universe policy + trend rule | strategy-104 | Defines watchlist, sector map, regime thresholds |
| Candidate ranking + sizing + decision | pipeline | Admission chain, panel score, QP, sizing (arm A/B diverge here) |
| Broker positions + fills + costs | execution | Reports actual fills; provides cost model inputs |
| Shadow scheduling + snapshot capture | orchestrator | Triggers daily shadow, captures decision-time snapshot, logs telemetry |
| Experiment specification + verdict | RenQuant (this doc) | Pre-registration, go/no-go decision, result memo |

## 6. Artifact contract: paired shadow with independent counterfactual ledgers

### 6.1 Decision-time snapshot (immutable input)

At each trading session open, the orchestrator captures a single immutable
snapshot consumed by ALL arms:

| Artifact | Type | Owner | Content |
|----------|------|-------|---------|
| `snapshot.market_data` | DataFrame | base-data | OHLCV + fundamentals at decision time |
| `snapshot.regime_label` | enum | pipeline | Current regime (HMM detector) |
| `snapshot.panel_scores` | DataFrame | pipeline | Panel scores, all universe names |
| `snapshot.admission_mask` | Series[bool] | pipeline | Names passing admission gates |
| `snapshot.config_digest` | str (SHA-256) | orchestrator | Hash of strategy+pipeline+execution configs |
| `snapshot.timestamp` | datetime (UTC) | orchestrator | Decision-time wall clock |
| `snapshot.g_t` | float | orchestrator | Common risk scalar (§3.2) |
| `snapshot.reference_nav` | float | orchestrator | A0 marked-to-market NAV used for g_t |

Written once, read by all arms. No arm may modify it.

### 6.2 Independent counterfactual paths

Each arm maintains SEPARATE, independent state that is never shared:

| Artifact | Type | Per-arm | Content |
|----------|------|---------|---------|
| `ledger.cash` | float | yes | Arm-specific available cash |
| `ledger.positions` | Dict[str, Position] | yes | Arm-specific holdings and lots |
| `ledger.fills` | List[Fill] | yes | Arm-specific executed fills with price, quantity, cost, tax |
| `ledger.exits` | List[ExitEvent] | yes | Arm-specific stop/trailing/panel-exit events |
| `ledger.nav_series` | Series[float] | yes | Arm-specific daily NAV for return calculation |
| `ledger.turnover_series` | Series[float] | yes | Arm-specific daily turnover (dollar volume / NAV) |
| `ledger.drawdown_series` | Series[float] | yes | Arm-specific running max drawdown (positive loss) |
| `ledger.cost_series` | Series[float] | yes | Arm-specific cumulative transaction costs |
| `ledger.tax_series` | Series[float] | yes | Arm-specific cumulative realized tax |

### 6.3 Pipeline flow

```
snapshot (immutable, shared — includes g_t)
  +---> Arm A0 (production sizing) ---> ledger_A0 update
  +---> Arm A1 (conviction, g_t * NAV_A1) ---> ledger_A1 update
  +---> Arm B  (equal, g_t * NAV_B)       ---> ledger_B update
  +---> telemetry row (all NAVs, g_t, per-arm E, per-arm cost/tax logged)
```

Neither arm's subsequent state is visible to the other. At activation all three
ledgers are cloned from one marked-to-market state: cash, positions, tax lots,
pending orders, unsettled cash, reservations, prices, and as-of timestamp. The
first operation is a same-session transition rebalance under each arm's frozen
rule. Starting A0/A1 from live positions and B from cash is prohibited.

### 6.4 Output artifacts

| Artifact | Type | Owner | When |
|----------|------|-------|------|
| `shadow_telemetry.parquet` | DataFrame | orchestrator | Daily append |
| `replay_result.json` | Dict | orchestrator | After historical replay |
| `prereg_verdict.json` | Dict | orchestrator | At go/no-go decision |

### 6.5 Executable artifact schemas (frozen at activation)

The following schemas are frozen at activation. Any schema change after
the first observation invalidates all collected data.

**`shadow_telemetry.parquet` columns (frozen):**

```
session_date     : date     # trading session
block_index      : int      # 0-indexed block number (session // block_length)
regime_label     : str      # regime at block start
g_t              : float64  # common risk scalar
reference_nav    : float64  # A0 NAV used for g_t
arm              : str      # "A0" | "A1" | "B"
nav              : float64  # arm-specific marked-to-market NAV
daily_return     : float64  # arm-specific daily net return (bps)
exposure_dollar  : float64  # arm-specific dollar gross exposure
exposure_ratio   : float64  # arm-specific exposure / NAV
turnover         : float64  # arm-specific dollar turnover / NAV
cost_dollar      : float64  # arm-specific transaction cost this session
tax_dollar       : float64  # arm-specific realized tax this session
n_names          : int      # names held at session close
max_weight       : float64  # largest single-name weight at session close
cash_pct         : float64  # cash / NAV
feasibility_hit  : bool     # true if arm hit its feasibility ceiling
```

**`prereg_verdict.json` schema (frozen):**

```json
{
  "experiment_id": "g1-ew-prereg",
  "protocol_version": "v5",
  "activation_commit": "<sha>",
  "completed_sessions": "<int>",
  "completed_blocks": "<int>",
  "mbb_block_length": "<int>",
  "mbb_lower_bound_bps": "<float>",
  "mee_bps": 3.0,
  "pe_bps": 6.0,
  "statistical_efficacy": "POSITIVE | NOT_POSITIVE",
  "verdict": "GO | NO_GO | NO_GO_ECONOMICALLY_INSUFFICIENT",
  "pilot_manifest_digest": "<sha256>",
  "epoch_id": "<str>",
  "gates": {
    "G1": {"passed": "<bool>", "value": "<float>", "threshold": 3.0, "efficacy_lb_positive": "<bool>"},
    "G2": {"passed": "<bool>", "sessions": "<int>", "blocks": "<int>"},
    "G3": {"passed": "<bool>", "dd_b": "<float>", "dd_a1": "<float>"},
    "G4": {"passed": "<bool>", "turnover_b": "<float>", "turnover_a1": "<float>"},
    "G5": {"passed": "<bool>", "max_concentration": "<float>"},
    "G6": {"passed": "<bool>"},
    "G7": {"passed": "<bool>", "tracking_error": "<float | null>"}
  },
  "regime_descriptive": {
    "<regime>": {
      "blocks": "<int>",
      "mean_s_j_bps": "<float>",
      "status": "DESCRIPTIVE | NOT_ESTABLISHED_IN_REGIME"
    }
  },
  "per_arm_summary": {
    "<arm>": {
      "final_nav": "<float>",
      "total_return_pct": "<float>",
      "total_cost": "<float>",
      "total_tax": "<float>",
      "mean_exposure_ratio": "<float>"
    }
  }
}
```

## 7. Pre-registration fields (filled at activation)

This document is an RFC until the activation commit fills ALL fields
below and pushes to main. No TERMINAL observation may begin before the
activation commit; the calibration pilot is authorized solely and
sufficiently by the pilot-registration commit (§4.7 two-stage start).
Any field left as TBD at activation is a protocol violation that
invalidates all subsequent data.

```yaml
activation:
  start_date:        TBD-AT-ACTIVATION  # first shadow trading day
  end_date_earliest: TBD-AT-ACTIVATION  # start + n complete blocks (n from §4.6)
  shared_state_digest: TBD-AT-ACTIVATION # SHA-256 of serialized initial state
  initial_cash:      TBD-AT-ACTIVATION  # copied identically into A0/A1/B
  initial_positions: TBD-AT-ACTIVATION  # copied identically into A0/A1/B
  initial_lots:      TBD-AT-ACTIVATION  # tax basis and holding period, same for all arms
  pending_orders:    TBD-AT-ACTIVATION  # copied identically into A0/A1/B
  reserved_capital:  TBD-AT-ACTIVATION  # copied identically into A0/A1/B

universe:
  watchlist_digest:  TBD-AT-ACTIVATION  # SHA-256 of strategy watchlist
  sector_map_digest: TBD-AT-ACTIVATION  # SHA-256 of sector mapping
  regime_source:     TBD-AT-ACTIVATION  # e.g. "hmm_regime_labels v3, common@<commit>"

data:
  base_data_pin:     TBD-AT-ACTIVATION  # base-data commit hash
  feature_digest:    TBD-AT-ACTIVATION  # SHA-256 of alpha158 feature config

strategy:
  strategy_pin:      TBD-AT-ACTIVATION  # strategy-104 commit hash
  pipeline_pin:      TBD-AT-ACTIVATION  # pipeline commit hash
  execution_pin:     TBD-AT-ACTIVATION  # execution commit hash
  orchestrator_pin:  TBD-AT-ACTIVATION  # orchestrator commit hash

calendar:
  session_calendar:  TBD-AT-ACTIVATION  # e.g. "NYSE, pandas_market_calendars"
  excluded_dates:    TBD-AT-ACTIVATION  # any manually excluded dates

config:
  strategy_config_digest:  TBD-AT-ACTIVATION  # SHA-256 of full strategy config
  pipeline_config_digest:  TBD-AT-ACTIVATION  # SHA-256 of full pipeline config
  k_range:                 TBD-AT-ACTIVATION  # regime-aware k values per regime
  g_t_formula_digest:      TBD-AT-ACTIVATION  # SHA-256 of exact g_t computation code
  feasibility_rule_digest: TBD-AT-ACTIVATION  # SHA-256 of arm feasibility logic

rebalance:
  rebalance_period:  20                 # sessions between calendar rebalances
  rebalance_calendar_digest: TBD-AT-ACTIVATION  # SHA-256 of frozen rebalance dates
  sfrac_enabled:     TBD-AT-ACTIVATION  # true if fractional shares available

inference:
  method: moving_block_bootstrap
  historical_use: diagnostic_only
  block_length_sessions: 20            # evaluation block length
  mbb_block_length_formula: "ceil(1.75 * max_holding_days), capped at 40"
  bootstrap_resamples: 10000
  one_sided_confidence: 0.90
  mee_bps_per_session: 3               # deployment eligibility threshold (§4.1)
  pe_bps_per_session: 6                # planning effect for power (§4.1)
  materiality_rationale_doc: TBD-AT-ACTIVATION  # written economic rationale for MEE and PE

pilot:                                  # §4.7 calibration pilot
  pilot_registration_commit: TBD-AT-ACTIVATION # commit that authorized pilot collection (§4.7 stage 1)
  epoch_id:              TBD-AT-ACTIVATION  # harness pin-manifest epoch; terminal series MUST run in this same epoch (§4.7 rule 2)
  pilot_manifest_digest: TBD-AT-ACTIVATION  # SHA-256 of the pilot session manifest (prospective-only)
  n_pilot_sessions:      TBD-AT-ACTIVATION  # must be >= 40, single epoch
  burned_sessions_digest: TBD-AT-ACTIVATION # manifest committed at pilot registration (§4.7 rule 1)
  blinding_program_commit: TBD-AT-ACTIVATION # demeaned arm-label-blinded sizing program version
  orientation_seal_digest: TBD-AT-ACTIVATION # sealed unblinding key commitment
  variance_upper_bound:  TBD-AT-ACTIVATION  # conservative 90% upper CL used for sizing

simulation:                             # outputs from §4.6 pre-activation simulation
  type_i_rate:       TBD-AT-ACTIVATION  # deployment rule at mu=MEE; must be <= 0.10
  type_i_ci_95:      TBD-AT-ACTIVATION  # 95% simulation CI
  power_at_pe:       TBD-AT-ACTIVATION  # at mu=PE; must be >= 0.80
  power_ci_95:       TBD-AT-ACTIVATION  # 95% simulation CI
  final_n_sessions:  TBD-AT-ACTIVATION  # 240..480 per §4.6 step 4
  final_n_blocks:    TBD-AT-ACTIVATION  # final_n_sessions / 20
  mbb_block_length:  TBD-AT-ACTIVATION  # b used in simulation
  dgp_parameters:    TBD-AT-ACTIVATION  # pilot-fitted dependence + conservative variance
  simulation_code_commit: TBD-AT-ACTIVATION  # commit hash of simulation script
  simulation_seed:   TBD-AT-ACTIVATION  # reproducibility

schemas:
  telemetry_schema_digest: TBD-AT-ACTIVATION  # SHA-256 of frozen telemetry schema (§6.5)
  verdict_schema_digest:   TBD-AT-ACTIVATION  # SHA-256 of frozen verdict schema (§6.5)
  verdict_generator_commit: TBD-AT-ACTIVATION # commit hash of the verdict computation code

early_stopping: none  # no interim analysis or early stopping permitted
```

## 8. Non-goals

- NOT a deployment governor (no regime-linked exposure/hysteresis).
- NOT a change to admission or exit logic.
- NOT alpha research — tests capital efficiency, not stock selection.
- NOT a source of regime-specific deployment claims (§4.2).

## 9. Rollout (if GO)

1. **S0:** Shadow (D2) — no live footprint; complete the prospective blocks.
2. **S1:** Independent evidence reproduction and explicit operator decision.
3. **S2:** A separate canary proposal, with its own capital, duration, and
   rollback pre-registration — not authorized by this document.

Kill switch at every stage: single config flag reverts to status quo.

## 10. Deliverables

| # | Deliverable | Repo | Nature |
|---|-------------|------|--------|
| D1 | This pre-registration | RenQuant | experiment spec |
| D1.5 | Pre-activation simulation (§4.6) | RenQuant | power/type-I validation |
| D2 | Shadow telemetry implementation | orchestrator + pipeline | code (read-only shadow) |
| D3 | Historical replay on held-out window | orchestrator | evaluation (diagnostic only) |
| D4 | Result memo with per-regime descriptive breakdowns | RenQuant | research output |
| D5 | Go/no-go verdict | RenQuant | decision artifact |

D1.5 and D2 must be completed before activation.

## 11. Relationship to prior work

- **Deployment governor (rejected):** tests D6's strongest control arm.
- **S-FRAC / G4 ensemble:** orthogonal (affect both arms equally).
