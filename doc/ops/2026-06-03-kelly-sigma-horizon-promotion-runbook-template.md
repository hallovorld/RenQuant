# Kelly sigma-horizon promotion report/runbook template

**Status**: Template for the decision report after the Kelly sigma-horizon
A/B completes. Fill this from committed A/B plan/result artifacts and runner
logs. Do not mutate golden or experiment configs while preparing this report.

**Scope**: Kelly sizing sigma horizon only. This template does not cover
changes to `fractional`, `max_concentration`, regime `max_position_pct`,
regime `cash_reserve_pct`, `qp_turnover_max`, `top_up_threshold`, buy gates,
or NGBoost enablement.

**Hard rule**: No production flip without Tier 3. A sigma-horizon change can
be merged into `strategy_config.golden.json` only after this report shows
Tier 2 plus DSR/PBO evidence satisfying `doc/research/promotion-methodology.md`
Tier 3.

## Report metadata

| Field | Value |
|---|---|
| Report date | `<YYYY-MM-DD>` |
| Prepared by | `<name/agent>` |
| Source audit | `doc/research/2026-06-03-kelly-sizing-audit.md` |
| A/B plan artifact | `<repo-relative path>` |
| A/B result artifact | `<repo-relative path>` |
| Runner command log | `<repo-relative path>` |
| Golden commit | `<sha>` |
| Candidate commit | `<sha>` |
| OOS window | `<start>` to `<end>` |

## Required comparison

The report must compare exactly these primary variants before any promotion
decision:

| Variant | Required setting | Role |
|---|---|---|
| A | golden, legacy sigma horizon (`sigma_horizon_days` omitted or 252) | Production baseline |
| B | `sigma_horizon_days = 60` | Same-period Kelly candidate |

The headline comparison must say **golden vs sigma=60** and must not include
any additional config changes. If any extra knobs changed, mark the report
invalid and rerun the A/B.

## Required validation runs

All rows must be complete and linked to artifacts before the verdict can be
read as promotion evidence.

| Check | Required evidence | Artifact |
|---|---|---|
| A/B | golden vs sigma=60 on the 27-month OOS window | `<path>` |
| A/A resplit | Baseline-vs-baseline split sanity; expected near-zero deltas | `<path>` |
| shuffle-placebo | Label/signal shuffle placebo; candidate must not win for the wrong reason | `<path>` |
| time-shift-placebo | Time-shift placebo; candidate must not survive broken temporal alignment | `<path>` |
| Seeds | >=5 seeds with mean±std for every headline metric | `<path>` |
| DSR/PBO | DSR and PBO computed with the correct trial count and candidate matrix | `<path>` |

## Seed summary

Report >=5 seeds mean±std across all headline metrics. Do not quote the best
seed as the verdict.

| Metric | Golden mean±std | sigma=60 mean±std | Delta mean±std | Notes |
|---|---:|---:|---:|---|
| APY | `<value>` | `<value>` | `<value>` |  |
| Sharpe | `<value>` | `<value>` | `<value>` |  |
| MaxDD | `<value>` | `<value>` | `<value>` |  |
| Cash % | `<value>` | `<value>` | `<value>` |  |
| Turnover | `<value>` | `<value>` | `<value>` |  |
| Kelly target median | `<value>` | `<value>` | `<value>` |  |
| Kelly target p90 | `<value>` | `<value>` | `<value>` |  |

## Per-regime results

Per-regime evidence is mandatory. BULL_CALM/BULL_VOLATILE/CHOPPY Sharpe/APY/MaxDD/cash%/Kelly distribution is required before any pooled result is used. BEAR can be listed as a sanity row, but BEAR is not promotion evidence for this Kelly sizing change because 100% cash is by design.

### BULL_CALM

| Metric | Golden mean±std | sigma=60 mean±std | Delta mean±std | Gate note |
|---|---:|---:|---:|---|
| Sharpe | `<value>` | `<value>` | `<value>` |  |
| APY | `<value>` | `<value>` | `<value>` |  |
| MaxDD | `<value>` | `<value>` | `<value>` |  |
| Cash % | `<value>` | `<value>` | `<value>` |  |
| Kelly distribution median / p75 / p90 | `<value>` | `<value>` | `<value>` |  |

### BULL_VOLATILE

| Metric | Golden mean±std | sigma=60 mean±std | Delta mean±std | Gate note |
|---|---:|---:|---:|---|
| Sharpe | `<value>` | `<value>` | `<value>` |  |
| APY | `<value>` | `<value>` | `<value>` |  |
| MaxDD | `<value>` | `<value>` | `<value>` |  |
| Cash % | `<value>` | `<value>` | `<value>` |  |
| Kelly distribution median / p75 / p90 | `<value>` | `<value>` | `<value>` |  |

### CHOPPY

| Metric | Golden mean±std | sigma=60 mean±std | Delta mean±std | Gate note |
|---|---:|---:|---:|---|
| Sharpe | `<value>` | `<value>` | `<value>` |  |
| APY | `<value>` | `<value>` | `<value>` |  |
| MaxDD | `<value>` | `<value>` | `<value>` |  |
| Cash % | `<value>` | `<value>` | `<value>` |  |
| Kelly distribution median / p75 / p90 | `<value>` | `<value>` | `<value>` |  |

### Pooled summary

Use this only after the per-regime tables are complete.

| Metric | Golden mean±std | sigma=60 mean±std | Delta mean±std | Notes |
|---|---:|---:|---:|---|
| Sharpe | `<value>` | `<value>` | `<value>` |  |
| APY | `<value>` | `<value>` | `<value>` |  |
| MaxDD | `<value>` | `<value>` | `<value>` |  |
| Cash % | `<value>` | `<value>` | `<value>` |  |

## Placebo and A/A interpretation

| Run | Pass condition | Observed result | Pass? |
|---|---|---|---|
| A/A resplit | Deltas are near zero and do not imply a false candidate edge | `<summary>` | `<yes/no>` |
| shuffle-placebo | sigma=60 does not retain a Sharpe/APY edge after signal shuffle | `<summary>` | `<yes/no>` |
| time-shift-placebo | sigma=60 does not retain a Sharpe/APY edge after temporal break | `<summary>` | `<yes/no>` |

If a placebo run improves with sigma=60, stop. Treat the result as invalid
promotion evidence and audit implementation, leakage, artifact alignment, and
runner parity before rerunning.

## DSR/PBO and Tier 3

| Field | Value |
|---|---|
| Number of configs/trials used for DSR | `<K>` |
| Raw Sharpe, golden | `<value>` |
| Raw Sharpe, sigma=60 | `<value>` |
| DSR, sigma=60 | `<value>` |
| PBO, candidate matrix | `<value>` |
| Tier 2 screen passed? | `<yes/no>` |
| Tier 3 passed? | `<yes/no>` |

Tier 3 requires Tier 2 plus one of the Tier 3 disjuncts in
`doc/research/promotion-methodology.md`. For this report, explicitly state
which condition passed, such as `DSR > 0.5` or `PBO < 0.5`. If neither passes,
the only valid verdict is "do not flip production".

## Verdict

Choose exactly one:

- `PROMOTE`: All required validation runs passed, per-regime evidence is
  complete, Tier 3 passed, and no other config knobs changed.
- `SCREEN_ONLY`: Tier 2 passed but Tier 3 did not pass. Keep sigma=60 out of
  production and schedule additional OOS/significance work.
- `REJECT`: A/B did not improve the required metrics or worsened risk beyond
  the documented tolerance.
- `INVALID`: Missing artifacts, failed A/A/placebo, fewer than 5 seeds,
  missing DSR/PBO, missing per-regime table, or any non-Kelly-sigma config
  mutation.

**Selected verdict**: `<PROMOTE | SCREEN_ONLY | REJECT | INVALID>`

**Production action**: `<none | follow-up PR to flip sigma_horizon_days to 60>`

Re-state the hard rule in the final paragraph: No production flip without
Tier 3.
