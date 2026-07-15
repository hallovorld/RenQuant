# Pre-registration: equal-weight top-k deployment experiment (G1)

STATUS: pre-registration RFC (no behavior change until activation)
DATE: 2026-07-13
REPO: RenQuant (umbrella) — cross-repo strategy specification
PRIOR: D6 confirmatory replay REJECTED deployment governor (PBO 0.874)
SUPERSEDES: orchestrator PR #509 (relocated per codex review)

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

**H0 (null):** Conditional on the same selected names and portfolio-level
exposure budget, equal-weight top-k produces the same or worse risk-adjusted
net return as conviction-weighted sizing.

**H1 (alternative):** Equal-weight top-k produces higher risk-adjusted net
return, with non-degradation on drawdown, across regime types including
CHOPPY and BEAR.

### 2.1 Power considerations

Daily portfolio returns from overlapping holdings are not independent
observations. The sampling unit is a predeclared non-overlapping
20-session calendar block — but the test statistic within each block is
the arithmetic mean daily active return (bps/session), not a cumulated
block return. This keeps the statistic, the MDE, and the gate threshold
on one common scale: bps per trading session.

Historical results, including D6, are hypothesis-generating diagnostics
only and never enter confirmation.

## 3. Experimental arms

### 3.1 Arm definitions (fixed, no adaptive components)

| Arm | Label | Sizing rule |
|-----|-------|-------------|
| A0 | `status_quo_observational` | Current production sizing, retained only to measure the separate exposure intervention |
| A1 | `conviction_exposure_matched` | Current conviction/signal raw weights rescaled to the common target exposure |
| B | `equal_weight_exposure_matched` | Equal weight across the same admitted names at the same common target exposure |

All arms share the SAME admission chain, exit logic, k selection, candidate
ranking, prices, costs, taxes, and state snapshot. The primary estimand is
`B - A1`: equal weights versus conviction weights holding selected names and
portfolio-level exposure fixed. `A1 - A0` is reported separately as the
exposure-normalisation effect; it is not evidence that equal weights helped.

### 3.2 Operational definition of rebalancing (shared schedule)

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

On each rebalance session:
- **Arm B** allocates `E_t / k` per admitted name (equal weight).
- **Arm A1** allocates conviction-proportional weights normalized to
  sum to `E_t`.

Between rebalance sessions, BOTH arms drift freely — no intra-period
correction for either arm. This eliminates the asymmetry a per-name
drift band would create: conviction concentrations in A1 would
continuously breach a 1/k-based drift threshold, causing A1 to
rebalance far more often than B and confounding the comparison with
turnover/timing effects. The calendar trigger ensures both arms
experience identical drift windows and transact on the same dates.

**Common exposure budget:** at each decision time `E_t` is the dollar gross
exposure permitted by the frozen risk engine after reserves, pending exits,
wash-sale holds, and margin requirements. `E_t` is recorded once in the shared
snapshot. Both A1 and B deploy to the SAME `E_t`.

When `k >= 4`: A1 normalizes its non-negative raw conviction weights to
sum to `E_t`; B targets `E_t / k` per admitted name.

When `k < 4`: both arms apply `k_effective = max(k, 4)`. B allocates
`E_t / k_effective` per admitted name; A1 normalizes conviction weights
to sum to `E_t * k / k_effective`. The remaining exposure
`E_t * (1 - k / k_effective)` is held as cash in BOTH arms identically.
This prevents a beta/exposure confound from polluting the B − A1
comparison. A0 retains its production target only as an observational arm.

**Rounding:** fractional shares (S-FRAC) by default. If S-FRAC is not
available at activation, whole-share floor `floor(target_dollar / price)`
is used with per-block tracking error reported (see gate G7).

**Residual cash:** held uninvested (NOT redistributed to largest-
remainder name). Available at next rebalance trigger.

**Corporate actions:** splits trigger rebalance; dividends add to cash.

**Cost and tax:** identical across arms. Existing sim cost model (section
4.5), lot-level `tax_drag()` with FIFO, wash-sale 30-day window.

## 4. Evaluation protocol

### 4.1 Statistical test

**Block statistic (primary).** For each 20-session block `j`, compute
the arithmetic mean daily active return:

    S_j = (1/20) * sum_{t in block j}( r_B_t - r_A1_t )

where `r_B_t` and `r_A1_t` are the daily net-of-cost, net-of-tax
portfolio returns for arms B and A1 on session `t`. `S_j` is a mean
daily return, expressed in basis points per session (bps/session).

**Units:** every quantity in the decision chain is in bps/session:
- `S_j` (block statistic) = arithmetic mean daily active return
  within block j (bps/session).
- `mean(S_j)` (point estimate) = grand mean of block statistics
  across all blocks (bps/session).
- MDE = 3 bps/session.
- The bootstrap operates on the vector `{S_1, …, S_n}` and produces
  a confidence bound in the same bps/session unit.

**Primary test:** the study collects at least 12 complete, non-overlapping
blocks (240 completed sessions). A one-sided 90% percentile confidence
interval is computed from 10,000 block-bootstrap resamples of the
block statistics `{S_1, …, S_n}`. The lower bound of this interval must
exceed the MDE of 3 bps/session. No daily Newey-West result is a
decision criterion.

**Effect size reporting:** the point estimate is `mean(S_j)` across all
blocks (bps/session). Drawdown (G3) and turnover (G4) summaries use the
same daily returns aggregated at the block level for consistency.

**Regime result:** each block receives the regime prevailing at its first
decision session and its within-block regime composition. A claim for a regime
requires at least 4 complete blocks with at least 75% of sessions in that
regime. Otherwise the result is `NOT_ESTABLISHED_IN_REGIME`, not a pooled
substitute.

**MDE justification:** 3 bps/session (~7.5% annualized) is the minimum
improvement worth deploying given operational complexity of the change.
Below this, the status quo's track record and operational familiarity
dominate.

### 4.2 Data sources

- **Shadow (prospective):** daily counterfactual from activation date
  forward. Both arms computed from the same immutable decision-time
  snapshot (section 6); no live execution footprint.
- **Historical replay:** D6 and every existing or newly-run historical replay
  are diagnostic only. They may validate accounting and identify failure
  modes, but cannot be combined with prospective blocks or satisfy a
  confirmation gate.
- **Regime coverage:** see the complete-block rule above. No number of selected
  historical days can make a missing prospective regime observation valid.

### 4.3 Go/no-go decision rule (frozen before evaluation)

**GO** requires ALL gates to pass:

| Gate | Criterion | Rationale |
|------|-----------|-----------|
| G1 | The one-sided 90% block-bootstrap lower bound for `mean(S_j)` exceeds 3 bps/session. `S_j` = arithmetic mean daily active return within block j; `mean(S_j)` and MDE are both in bps/session (§4.1) | Economic effect after dependence-aware inference |
| G2 | At least 12 prospective complete blocks; every claimed regime satisfies the 4-block/75% rule | No daily-row pseudo-replication or pooled regime claim |
| G3 | DD_B <= 1.2 * DD_A1 (drawdown as positive loss, see note) | Non-degradation on tail risk at matched exposure |
| G4 | Mean block turnover of B <= A1 and costs are fully reconciled | Prevent a gross-return-only win |
| G5 | No single-name concentration > 1/k at any rebalance point; hard floor k >= 4 (cap <= 25%); BOTH arms apply same k_effective and cash rule (§3.2) | Safety cap; identical exposure prevents beta confound |
| G6 | A1 - A0 is reported separately with exposure, turnover, and NAV attribution | Do not mislabel an exposure effect as a weighting effect |
| G7 | If S-FRAC disabled: mean per-name tracking error from rounding < 2% of target weight across the evaluation period | Prevent rounding artifacts from dominating the weighting signal |

**Drawdown sign convention (G3):** drawdown is defined as a POSITIVE loss:
DD = abs(peak_value - trough_value) / peak_value. Both DD_A1 and DD_B are
non-negative. The gate DD_B <= 1.2 * DD_A1 correctly requires that arm B's
maximum loss does not exceed 120% of arm A's maximum loss. Example: if
arm A draws down 10% (DD_A = 0.10), arm B must not exceed 12%
(DD_B <= 0.12).

**Concentration cap and matched cash (G5):** when k < 4, the equal-weight
allocation 1/k > 25%. Both arms apply `k_effective = max(k, 4)` and hold
identical residual cash (section 3.2). This prevents a single-name
concentration breach AND prevents the B − A1 comparison from measuring
a beta/exposure difference instead of a weighting effect.

**NO-GO** if any gate fails. No re-tuning, no "close enough" exceptions.

### 4.4 Minimum observation period

- Shadow: at least 12 complete 20-session prospective blocks for the primary
  decision. Earlier observations are operational diagnostics only.
- Historical replay: any duration, explicitly diagnostic-only and excluded
  from all go/no-go statistics.
- No interim analysis or early stopping is permitted. The experiment runs
  to 12 complete blocks. This eliminates researcher degrees of freedom in
  prior selection, stopping-boundary calibration, and interim timing.

### 4.5 Cost assumptions (frozen)

| Parameter | Value | Source |
|-----------|-------|--------|
| Base transaction cost | 5 bps round-trip | Existing sim |
| Adverse selection | 2x base for names with daily volume < $50M | Existing sim |
| Tax rate (short-term) | 50% | Existing convention |
| Tax rate (long-term) | 32% | Existing convention |
| Slippage model | Existing sim infrastructure | No change |

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
snapshot consumed by BOTH arms:

| Artifact | Type | Owner | Content |
|----------|------|-------|---------|
| `snapshot.market_data` | DataFrame | base-data | OHLCV + fundamentals at decision time |
| `snapshot.regime_label` | enum | pipeline | Current regime (HMM detector) |
| `snapshot.panel_scores` | DataFrame | pipeline | Panel scores, all universe names |
| `snapshot.admission_mask` | Series[bool] | pipeline | Names passing admission gates |
| `snapshot.config_digest` | str (SHA-256) | orchestrator | Hash of strategy+pipeline+execution configs |
| `snapshot.timestamp` | datetime (UTC) | orchestrator | Decision-time wall clock |

Written once, read by both arms. Neither arm may modify it.

### 6.2 Independent counterfactual paths

Each arm maintains SEPARATE, independent state that is never shared:

| Artifact | Type | Per-arm | Content |
|----------|------|---------|---------|
| `ledger.cash` | float | yes | Available cash copied from the shared marked-to-market snapshot |
| `ledger.positions` | Dict[str, Position] | yes | Holdings and lots copied from the shared marked-to-market snapshot |
| `ledger.fills` | List[Fill] | yes | Executed fills with price, quantity, cost, tax |
| `ledger.exits` | List[ExitEvent] | yes | Stop/trailing/panel-exit events |
| `ledger.nav_series` | Series[float] | yes | Daily NAV for return calculation |
| `ledger.turnover_series` | Series[float] | yes | Daily turnover (dollar volume / NAV) |
| `ledger.drawdown_series` | Series[float] | yes | Running max drawdown (positive loss) |

### 6.3 Pipeline flow

```
snapshot (immutable, shared)
  +---> Arm A0 (production sizing) ---> ledger_A0 update
  +---> Arm A1 (matched conviction) ---> ledger_A1 update
  +---> Arm B  (matched 1/k)        ---> ledger_B update
  +---> telemetry row (all NAVs and E_t logged)
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

## 7. Pre-registration fields (filled at activation)

The following fields MUST be recorded at experiment activation time.
They are listed here with placeholder values; the activation commit
fills them with actuals.

```yaml
activation:
  start_date:        TBD-AT-ACTIVATION  # first shadow trading day
  end_date_earliest: TBD-AT-ACTIVATION  # start + 12 complete 20-session blocks
  shared_state_digest: TBD-AT-ACTIVATION # cash, lots, positions, orders, reservations, marks
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
  common_exposure_rule:    TBD-AT-ACTIVATION  # exact E_t risk-engine formula and digest

rebalance:
  rebalance_period:  20                 # sessions between calendar rebalances (matches block length)
  rebalance_calendar_digest: TBD-AT-ACTIVATION  # SHA-256 of frozen rebalance dates
  sfrac_enabled:     TBD-AT-ACTIVATION  # true if fractional shares available

inference:
  historical_use: diagnostic_only
  block_length_sessions: 20
  minimum_complete_blocks: 12
  bootstrap_resamples: 10000
  one_sided_confidence: 0.90
  mde_bps_per_session: 3

early_stopping: none  # no interim analysis or early stopping permitted
```

## 8. Non-goals

- NOT a deployment governor (no regime-linked exposure/hysteresis).
- NOT a change to admission or exit logic.
- NOT alpha research — tests capital efficiency, not stock selection.

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
| D2 | Shadow telemetry implementation | orchestrator + pipeline | code (read-only shadow) |
| D3 | Historical replay on held-out window | orchestrator | evaluation |
| D4 | Result memo with per-regime breakdowns | RenQuant | research output |
| D5 | Go/no-go verdict | RenQuant | decision artifact |

D2 must be implemented before activation (artifact contract: section 6).

## 11. Relationship to prior work

- **Deployment governor (rejected):** tests D6's strongest control arm.
- **S-FRAC / G4 ensemble:** orthogonal (affect both arms equally).
